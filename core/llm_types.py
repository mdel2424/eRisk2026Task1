from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    backend: str = ""
    model_id: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMBudgetExceeded(RuntimeError):
    pass
