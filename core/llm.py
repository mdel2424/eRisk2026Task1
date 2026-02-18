from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMResponse:
    content: str


@lru_cache(maxsize=4)
def _load_base_chat_model(model_id: str, hf_token: str, load_in_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    token = hf_token or None
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"token": token, "device_map": "auto"}
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    return tokenizer, model


class LocalHFChatLLM:
    def __init__(
        self,
        model_id: str,
        hf_token: str,
        load_in_4bit: bool,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        self.tokenizer, self.model = _load_base_chat_model(model_id, hf_token, load_in_4bit)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    @staticmethod
    def _normalize_messages(messages: Sequence[dict | tuple[str, str]]) -> List[dict]:
        normalized: List[dict] = []
        for msg in messages:
            if isinstance(msg, tuple):
                role, content = msg
                normalized.append({"role": role, "content": content})
            else:
                normalized.append(
                    {"role": str(msg.get("role", "user")), "content": str(msg.get("content", ""))}
                )
        return normalized

    def invoke(self, messages: Sequence[dict | tuple[str, str]]) -> LLMResponse:
        normalized = self._normalize_messages(messages)
        if not normalized:
            return LLMResponse(content="")

        inputs = self.tokenizer.apply_chat_template(
            normalized,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        device = self.model.get_input_embeddings().weight.device
        inputs = inputs.to(device)

        output = self.model.generate(
            input_ids=inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0.0,
            temperature=max(0.0, self.temperature),
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        response_ids = output[0][inputs.shape[-1] :]
        text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return LLMResponse(content=text)


@lru_cache(maxsize=1)
def get_llm() -> LocalHFChatLLM:
    model_id = os.getenv("DETECTOR_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct").strip()
    if not model_id:
        raise ValueError("DETECTOR_MODEL is required")

    return LocalHFChatLLM(
        model_id=model_id,
        hf_token=os.getenv("HF_TOKEN", "").strip(),
        load_in_4bit=os.getenv("DETECTOR_LOAD_IN_4BIT", "1").strip() != "0",
        max_new_tokens=int(os.getenv("DETECTOR_MAX_NEW_TOKENS", "96")),
        temperature=float(os.getenv("DETECTOR_TEMPERATURE", "0.2")),
        top_p=float(os.getenv("DETECTOR_TOP_P", "0.9")),
    )
