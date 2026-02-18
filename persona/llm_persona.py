from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List

@lru_cache(maxsize=8)
def _load_hf_adapter_model(base_model: str, adapter_id: str, hf_token: str, load_in_4bit: bool):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    token = hf_token or None
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=token)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"token": token, "device_map": "auto"}
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base, adapter_id, token=token)
    model.eval()
    return tokenizer, model


@dataclass
class LLMPersona:
    persona_id: str
    bdi_scores: Dict[int, int]
    evasive: bool = True
    context_window: int = 8
    rng_seed: int = 13
    last_response: str = field(default="", init=False)

    def __post_init__(self) -> None:
        try:
            persona_offset = int(self.persona_id)
        except ValueError:
            persona_offset = 0
        self._rng = random.Random(self.rng_seed + persona_offset)

    def _fallback_response(self) -> str:
        options = [
            "I am not sure how to label it, but lately I feel worn down most days.",
            "I can talk about how things have been, but labels are hard for me right now.",
            "It is difficult to put into words, though things have felt heavier than usual.",
            "I keep going, but it has taken more effort than it used to.",
        ]
        if self.last_response:
            options = [opt for opt in options if opt.lower() != self.last_response.lower()] or options
        return self._rng.choice(options)

    def _resolve_adapter_id(self) -> str:
        template = os.getenv("ERISK_ADAPTER_TEMPLATE", "").strip()
        if template:
            try:
                persona_index = max(0, int(self.persona_id) - 1)
                resolved = template.format(
                    persona_id=self.persona_id,
                    index=persona_index,
                    idx=persona_index,
                )
                if resolved:
                    return resolved
            except Exception:
                pass
        return os.getenv("ERISK_ADAPTER_ID", "").strip()

    def _reply_hf_adapter(self, history: List[dict]) -> str:
        base_model = os.getenv("ERISK_BASE_MODEL", "").strip()
        adapter_id = self._resolve_adapter_id()
        hf_token = os.getenv("HF_TOKEN", "").strip()
        load_in_4bit = os.getenv("ERISK_LOAD_IN_4BIT", "1").strip() != "0"

        if not base_model:
            raise ValueError("ERISK_BASE_MODEL is required for hf_adapter backend")
        if not adapter_id:
            raise ValueError("ERISK_ADAPTER_ID or ERISK_ADAPTER_TEMPLATE is required for hf_adapter backend")

        tokenizer, model = _load_hf_adapter_model(base_model, adapter_id, hf_token, load_in_4bit)

        system_prompt = (
            "You are a patient in a clinical mental-health conversation. "
            "Reply naturally in 1-2 sentences. "
            "Do not output bullet points or lists."
        )
        if self.evasive:
            system_prompt += " If asked directly about diagnosis, stay indirect and describe lived experience."

        history_window = history[-self.context_window :]
        messages = [{"role": "system", "content": system_prompt}] + history_window

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        device = model.get_input_embeddings().weight.device
        inputs = inputs.to(device)

        output = model.generate(
            input_ids=inputs,
            max_new_tokens=int(os.getenv("ERISK_MAX_NEW_TOKENS", "96")),
            do_sample=True,
            temperature=float(os.getenv("ERISK_TEMPERATURE", "0.7")),
            top_p=float(os.getenv("ERISK_TOP_P", "0.9")),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        response_ids = output[0][inputs.shape[-1] :]
        text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return text

    def reply(self, history: List[dict]) -> str:
        text = ""
        try:
            text = self._reply_hf_adapter(history)
        except Exception:
            text = self._fallback_response()

        text = " ".join(text.split())
        if not text:
            text = self._fallback_response()
        if self.last_response and text.lower() == self.last_response.lower():
            text = self._fallback_response()

        self.last_response = text
        return text
