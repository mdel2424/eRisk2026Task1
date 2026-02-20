from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

from core.llm_backends import LocalHFChatLLM, OpenRouterChatLLM
from core.llm_types import LLMBudgetExceeded, LLMResponse
from core.llm_usage import get_llm_usage, reset_llm_usage, set_llm_call_budget
from core.runtime_policy import resolve_detector_backend

load_dotenv()


@lru_cache(maxsize=1)
def get_llm() -> LocalHFChatLLM | OpenRouterChatLLM:
    max_new_tokens = int(os.getenv("DETECTOR_MAX_NEW_TOKENS", "96"))
    temperature = float(os.getenv("DETECTOR_TEMPERATURE", "0.2"))
    top_p = float(os.getenv("DETECTOR_TOP_P", "0.9"))

    if resolve_detector_backend() == "openrouter":
        model_id = os.getenv("OPENROUTER_DETECTOR_MODEL", "openrouter/auto").strip()
        return OpenRouterChatLLM(
            model_id=model_id,
            api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip(),
            timeout_sec=int(os.getenv("OPENROUTER_TIMEOUT_SEC", "120")),
        )

    model_id = os.getenv("DETECTOR_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct").strip()
    if not model_id:
        raise ValueError("DETECTOR_MODEL is required")

    return LocalHFChatLLM(
        model_id=model_id,
        hf_token=os.getenv("HF_TOKEN", "").strip(),
        load_in_4bit=os.getenv("DETECTOR_LOAD_IN_4BIT", "1").strip() != "0",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


@lru_cache(maxsize=1)
def get_persona_openrouter_llm() -> OpenRouterChatLLM:
    return OpenRouterChatLLM(
        model_id=os.getenv("OPENROUTER_PERSONA_MODEL", "openrouter/auto").strip(),
        api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
        max_new_tokens=int(os.getenv("ERISK_MAX_NEW_TOKENS", "96")),
        temperature=float(os.getenv("ERISK_TEMPERATURE", "0.7")),
        top_p=float(os.getenv("ERISK_TOP_P", "0.9")),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip(),
        timeout_sec=int(os.getenv("OPENROUTER_TIMEOUT_SEC", "120")),
    )


__all__ = [
    "LLMResponse",
    "LLMBudgetExceeded",
    "reset_llm_usage",
    "get_llm_usage",
    "set_llm_call_budget",
    "get_llm",
    "get_persona_openrouter_llm",
]
