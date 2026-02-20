from __future__ import annotations

import threading

from core.llm_types import LLMBudgetExceeded

_USAGE_LOCK = threading.Lock()
_USAGE = {
    "calls_total": 0,
    "prompt_tokens_total": 0,
    "completion_tokens_total": 0,
    "total_tokens_total": 0,
    "errors_total": 0,
}
_CALL_BUDGET: int | None = None


def reset_llm_usage() -> None:
    with _USAGE_LOCK:
        for key in _USAGE:
            _USAGE[key] = 0


def get_llm_usage() -> dict:
    with _USAGE_LOCK:
        payload = dict(_USAGE)
        payload["max_calls"] = _CALL_BUDGET
        payload["calls_remaining"] = (
            None if _CALL_BUDGET is None else max(0, int(_CALL_BUDGET) - int(_USAGE["calls_total"]))
        )
    return payload


def set_llm_call_budget(max_calls: int | None) -> None:
    global _CALL_BUDGET
    if max_calls is None:
        _CALL_BUDGET = None
        return
    try:
        value = int(max_calls)
    except (TypeError, ValueError):
        value = 0
    _CALL_BUDGET = value if value > 0 else None


def _reserve_llm_call() -> None:
    with _USAGE_LOCK:
        calls_total = int(_USAGE["calls_total"])
        if _CALL_BUDGET is not None and calls_total >= int(_CALL_BUDGET):
            raise LLMBudgetExceeded(
                f"LLM API call budget exceeded: used={calls_total}, max_calls={int(_CALL_BUDGET)}"
            )
        _USAGE["calls_total"] = calls_total + 1


def _record_llm_error() -> None:
    with _USAGE_LOCK:
        _USAGE["errors_total"] = int(_USAGE["errors_total"]) + 1


def _record_token_usage(prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
    with _USAGE_LOCK:
        _USAGE["prompt_tokens_total"] = int(_USAGE["prompt_tokens_total"]) + max(0, int(prompt_tokens))
        _USAGE["completion_tokens_total"] = int(_USAGE["completion_tokens_total"]) + max(
            0, int(completion_tokens)
        )
        _USAGE["total_tokens_total"] = int(_USAGE["total_tokens_total"]) + max(0, int(total_tokens))
