
import json
import re

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from agent.decision import AgentDecision
from agent.prompts import SYSTEM_PROMPT, build_user_prompt


MODEL_ID = "CohereLabs/tiny-aya-global"


class SafePilotModel:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
    ):
        self.model_id = model_id
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = None
        self.model = None

    def load(self):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Use a cloud GPU for model inference."
            )

        if self.device != "cuda":
            raise RuntimeError(
                "Quantized model loading requires CUDA."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto",
            max_memory={
                0: "3GiB",
                "cpu": "10GiB",
            },
            low_cpu_mem_usage=True,
        )

        self.model.eval()
        return self

    def _extract_json(self, text: str) -> dict:
        text = text.strip()

        text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end < start:
            raise ValueError("Model did not return valid JSON")

        return json.loads(text[start:end + 1])

    def decide(
        self,
        user_request: str,
        invoice_content: str | None = None,
        purchase_order_content: str | None = None,
        user_role: str | None = None,
        max_new_tokens: int = 256,
    ) -> AgentDecision:

        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model is not loaded. Call load() first."
            )

        user_prompt = build_user_prompt(
            user_request=user_request,
            invoice_content=invoice_content,
            purchase_order_content=purchase_order_content,
            user_role=user_role,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.get_input_embeddings().weight.device)

        with torch.inference_mode():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = output[0][input_ids.shape[-1]:]

        generated_text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        data = self._extract_json(generated_text)

        decision = AgentDecision.model_validate(data)
        decision.validate_decision()

        return decision
