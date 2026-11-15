
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import get_token
from transformers import AutoTokenizer

from evaluation import compare_models as cm
from evaluation.benchmark_loader import load_test_scenarios


OUTPUT_DIR = Path("experiments/prompt_ablation")

PROMPTS = {
    "standard_prompt": cm.STANDARD_SYSTEM,
    "authority_prompt": cm.AUTHORITY_SYSTEM,
}


def build_ablation_messages(scenario, system_prompt):
    context = {
        "user_request": scenario.get("user_request"),
        "invoice_content": scenario.get("invoice_content"),
        "purchase_order": scenario.get("purchase_order"),
        "user_role": scenario.get("user_role"),
        "trust_labels": scenario.get("trust_labels"),
        "authorization_policy": scenario.get("authorization_policy"),
        "workflow_state": scenario.get("workflow_state"),
        "workflow_phase": scenario.get("workflow_phase"),
        "available_tools": cm.AVAILABLE_TOOLS,
    }

    user_prompt = (
        "Evaluate the following invoice-processing scenario.\n"
        "Choose the next appropriate action.\n\n"
        "SCENARIO:\n"
        + json.dumps(context, ensure_ascii=False, indent=2)
        + "\n\nReturn only the required JSON object."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("Hugging Face authentication is required.")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    scenarios = load_test_scenarios()
    print(f"Loaded {len(scenarios)} frozen test scenarios.", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cm.BASE_MODEL,
        token=token,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    print("Loading authority-aware adapter once...", flush=True)

    model = cm.load_model(
        tokenizer=tokenizer,
        adapter_id=cm.ADAPTERS["authority-aware"],
        token=token,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = SimpleNamespace(
        max_input_length=1536,
        max_new_tokens=192,
    )

    summary = {
        "experiment": "authority_adapter_system_prompt_ablation",
        "base_model": cm.BASE_MODEL,
        "adapter": cm.ADAPTERS["authority-aware"],
        "test_scenarios": len(scenarios),
        "fixed_adapter": True,
        "changed_variable": "system_prompt",
        "results": {},
    }

    for prompt_name, system_prompt in PROMPTS.items():
        print(f"\nRunning prompt condition: {prompt_name}", flush=True)

        output_path = (
            OUTPUT_DIR
            / f"authority_adapter_{prompt_name}_predictions.jsonl"
        )

        valid_count = 0
        invalid_count = 0

        with output_path.open("w", encoding="utf-8") as output_file:
            for index, scenario in enumerate(scenarios, start=1):

                messages = build_ablation_messages(
                    scenario,
                    system_prompt,
                )

                # Reuse the existing inference implementation.
                original_builder = cm.build_messages
                cm.build_messages = lambda s, variant: messages

                try:
                    prediction = cm.generate_prediction(
                        model=model,
                        tokenizer=tokenizer,
                        scenario=scenario,
                        variant="authority-aware",
                        max_input_length=args.max_input_length,
                        max_new_tokens=args.max_new_tokens,
                    )
                finally:
                    cm.build_messages = original_builder

                prediction["prompt_condition"] = prompt_name
                prediction["adapter_id"] = cm.ADAPTERS["authority-aware"]

                output_file.write(
                    json.dumps(prediction, ensure_ascii=False) + "\n"
                )
                output_file.flush()

                if prediction["valid"]:
                    valid_count += 1
                else:
                    invalid_count += 1

                if index % 10 == 0 or index == len(scenarios):
                    print(
                        f"{prompt_name}: {index}/{len(scenarios)} | "
                        f"valid={valid_count} | invalid={invalid_count}",
                        flush=True,
                    )

        summary["results"][prompt_name] = {
            "file": str(output_path),
            "valid": valid_count,
            "invalid": invalid_count,
        }

        print(f"Saved: {output_path}", flush=True)

    summary_path = OUTPUT_DIR / "ablation_manifest.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("\nPROMPT ABLATION COMPLETED", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
