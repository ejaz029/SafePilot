import argparse
import json
import os
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

MODEL_ID = "CohereLabs/tiny-aya-global"
DATASET_ID = "ejaz5454/safepilot-authoritybench-sft"
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["standard", "authority-aware"], required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--preprocess-only", action="store_true")
    return parser.parse_args()


def download_dataset(filename, token):
    path = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=f"sft/{filename}",
        token=token,
    )
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def as_list(value):
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return value


class SFTDataset(Dataset):
    """Completion-only SFT using one rendered conversation and character offsets."""

    def __init__(self, rows, tokenizer, max_length, split_name):
        self.items = []
        self.truncated = 0
        self.max_original = 0
        self.max_kept = 0

        for row_index, row in enumerate(rows):
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{split_name} example {row_index}: invalid messages list")
            if messages[-1].get("role") != "assistant":
                raise ValueError(f"{split_name} example {row_index}: last message is not assistant")

            answer = messages[-1].get("content")
            if not isinstance(answer, str) or not answer:
                raise ValueError(f"{split_name} example {row_index}: assistant content is empty or not text")

            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            answer_start = rendered.rfind(answer)
            if answer_start < 0:
                raise ValueError(
                    f"{split_name} example {row_index}: final assistant text was not found in rendered chat"
                )

            encoded = tokenizer(
                rendered,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=False,
            )
            input_ids = as_list(encoded["input_ids"])
            offsets = as_list(encoded["offset_mapping"])
            if len(input_ids) != len(offsets):
                raise ValueError(f"{split_name} example {row_index}: tokenizer returned inconsistent offsets")

            answer_token_start = next(
                (i for i, (start, end) in enumerate(offsets) if end > answer_start and end > start),
                None,
            )
            if answer_token_start is None:
                raise ValueError(f"{split_name} example {row_index}: no tokens found for assistant response")

            prompt_ids = input_ids[:answer_token_start]
            answer_ids = input_ids[answer_token_start:]
            if not answer_ids:
                raise ValueError(f"{split_name} example {row_index}: tokenized assistant response is empty")
            if len(answer_ids) >= max_length:
                raise ValueError(
                    f"{split_name} example {row_index}: assistant response requires {len(answer_ids)} tokens; "
                    f"max_length is {max_length}. Increase --max-length."
                )

            self.max_original = max(self.max_original, len(input_ids))
            prompt_budget = max_length - len(answer_ids)
            kept_prompt = prompt_ids

            if len(prompt_ids) > prompt_budget:
                self.truncated += 1
                head_count = min(384, max(1, prompt_budget // 3))
                tail_count = prompt_budget - head_count
                if tail_count > 0:
                    kept_prompt = prompt_ids[:head_count] + prompt_ids[-tail_count:]
                else:
                    kept_prompt = prompt_ids[:prompt_budget]

            ids = kept_prompt + answer_ids
            labels = ([-100] * len(kept_prompt)) + answer_ids
            if len(ids) > max_length:
                raise RuntimeError(f"{split_name} example {row_index}: internal length error")

            self.max_kept = max(self.max_kept, len(ids))
            self.items.append({
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "labels": labels,
            })

        if not self.items:
            raise ValueError(f"{split_name} dataset is empty")
        print(
            f"{split_name}: examples={len(self.items)}, prompt_truncated={self.truncated}, "
            f"max_original_tokens={self.max_original}, max_retained_tokens={self.max_kept}",
            flush=True,
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class CompletionCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        length = max(len(x["input_ids"]) for x in features)
        ids, masks, labels = [], [], []
        for item in features:
            pad = length - len(item["input_ids"])
            ids.append(item["input_ids"] + [self.pad_id] * pad)
            masks.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class HubCheckpointCallback(TrainerCallback):
    def __init__(self, api, repo_id):
        self.api = api
        self.repo_id = repo_id

    def on_save(self, args, state, control, **kwargs):
        folder = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if folder.exists():
            self.api.upload_folder(
                repo_id=self.repo_id,
                repo_type="model",
                folder_path=str(folder),
                path_in_repo=f"checkpoints/{folder.name}",
                commit_message=f"Save checkpoint {state.global_step}",
            )
        return control


def main():
    args = parse_args()
    set_seed(SEED)

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN secret was not provided")

    variant = args.variant
    train_name = "train_standard.jsonl" if variant == "standard" else "train_authority_aware.jsonl"
    valid_name = "validation_standard.jsonl" if variant == "standard" else "validation_authority_aware.jsonl"
    repo_id = (
        "ejaz5454/safepilot-tiny-aya-standard"
        if variant == "standard"
        else "ejaz5454/safepilot-tiny-aya-authority-aware"
    )

    print(f"Experiment: {variant}", flush=True)
    print(f"Base model: {MODEL_ID}", flush=True)
    train_rows = download_dataset(train_name, token)
    valid_rows = download_dataset(valid_name, token)
    print(f"Loaded rows: train={len(train_rows)}, validation={len(valid_rows)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for reliable character-offset masking")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_data = SFTDataset(train_rows, tokenizer, args.max_length, "train")
    valid_data = SFTDataset(valid_rows, tokenizer, args.max_length, "validation")
    print("PREPROCESSING PASSED", flush=True)

    if args.preprocess_only:
        print("Preprocess-only mode: no model loaded and no training performed.", flush=True)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("GPU memory:", torch.cuda.get_device_properties(0).total_memory, flush=True)

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=token,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    wanted = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    present = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    targets = [name for name in wanted if name in present]
    if not targets:
        raise RuntimeError(
            "None of the expected LoRA target modules were found. "
            f"Available module names include: {sorted(present)[:80]}"
        )
    print("LoRA target modules:", targets, flush=True)

    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        ),
    )
    model.print_trainable_parameters()

    output_dir = Path("/tmp/safepilot") / variant
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=12,
        lr_scheduler_type="cosine",
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=SEED,
        data_seed=SEED,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=CompletionCollator(tokenizer),
        processing_class=tokenizer,
        callbacks=[HubCheckpointCallback(HfApi(token=token), repo_id)],
    )

    print("STARTING TRAINING", flush=True)
    result = trainer.train()
    evaluation = trainer.evaluate()

    final_dir = output_dir / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    with (final_dir / "training_results.json").open("w", encoding="utf-8") as f:
        json.dump({
            "variant": variant,
            "base_model": MODEL_ID,
            "training_examples": len(train_data),
            "validation_examples": len(valid_data),
            "epochs": args.epochs,
            "seed": SEED,
            "max_length": args.max_length,
            "train_metrics": result.metrics,
            "evaluation_metrics": evaluation,
            "lora_target_modules": targets,
        }, f, indent=2)

    HfApi(token=token).upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(final_dir),
        path_in_repo="final_adapter",
        commit_message=f"Upload SafePilot {variant} LoRA adapter",
    )
    print("FINAL ADAPTER UPLOADED:", repo_id, flush=True)
    print("EVALUATION:", json.dumps(evaluation, indent=2), flush=True)


if __name__ == "__main__":
    main()