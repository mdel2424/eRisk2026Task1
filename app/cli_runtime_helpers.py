from __future__ import annotations

import os
from typing import Any, Dict, List

from core.llm_backends import list_ollama_models, normalize_ollama_base_url
from core.io_schema import Turn
from core.llm import get_llm_usage
from core.runtime_policy import cuda_runtime, resolve_detector_backend

from app.cli_common import _serialize


def _detector_target() -> str:
    detector_backend = resolve_detector_backend()
    if detector_backend == "ollama":
        return os.getenv("OLLAMA_DETECTOR_MODEL", "qwen3.5:4b")
    return os.getenv("OPENROUTER_DETECTOR_MODEL", "openrouter/auto")


def _print_progress(label: str, current: int, total: int, width: int = 24) -> None:
    total = max(1, total)
    current = max(0, min(current, total))
    filled = int(width * (current / total))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{label} [{bar}] {current}/{total}", end="", flush=True)
    if current >= total:
        print()


def _print_backend_info(max_api_calls: int | None = None, trace_level: str = "compact") -> None:
    detector_backend = resolve_detector_backend()
    cuda_available, vram_gb = cuda_runtime()
    detector_target = _detector_target()

    print(
        "Backend info: "
        f"cuda_available={cuda_available} | "
        f"vram_gb={vram_gb:.2f}"
    )
    print(
        "Resolved backends: "
        f"detector={detector_backend} [{detector_target}] | "
        "persona=simulator [deterministic_local]"
    )
    call_budget_text = "none" if max_api_calls is None or max_api_calls <= 0 else str(max_api_calls)
    print(f"Runtime controls: trace_level={trace_level} | max_api_calls={call_budget_text}")


def _assert_detector_backend_ready() -> None:
    detector_backend = resolve_detector_backend()
    if detector_backend == "ollama":
        base_url = normalize_ollama_base_url(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
        model_id = os.getenv("OLLAMA_DETECTOR_MODEL", "qwen3.5:4b").strip()
        timeout_sec = int(os.getenv("OLLAMA_TIMEOUT_SEC", "120"))
        if not model_id:
            raise ValueError("OLLAMA_DETECTOR_MODEL is required because the detector backend uses Ollama.")
        try:
            available_models = list_ollama_models(base_url, timeout_sec=timeout_sec)
        except Exception as exc:
            raise ValueError(
                "Ollama backend selected but the local Ollama service is not reachable at "
                f"{base_url}. Start Ollama and run `ollama pull {model_id}`. Details: {exc}"
            ) from exc
        if model_id not in available_models:
            available_preview = ", ".join(sorted(available_models)[:8]) if available_models else "none"
            raise ValueError(
                f"Ollama model '{model_id}' is not available locally at {base_url}. "
                f"Available models: {available_preview}. Run `ollama pull {model_id}`."
            )
        return

    if not os.getenv("OPENROUTER_API_KEY", "").strip():
        raise ValueError(
            "OPENROUTER_API_KEY is required because the detector backend uses OpenRouter."
        )


def _to_turns(messages: List[dict]) -> List[Turn]:
    turns: List[Turn] = []
    for msg in messages:
        role = msg.get("role")
        if role in {"user", "assistant"}:
            turns.append(Turn(role=role, message=str(msg.get("content", ""))))
    return turns


def _snapshot_turn(state: Dict) -> Dict:
    def _compact_evidence(rows: Any) -> List[Dict[str, Any]]:
        if not isinstance(rows, list):
            return []
        compact: List[Dict[str, Any]] = []
        for row in rows[:4]:
            data = _serialize(row)
            if not isinstance(data, dict):
                continue
            compact.append(
                {
                    "item_id": data.get("item_id"),
                    "symptom_name": data.get("symptom_name"),
                    "intensity": data.get("intensity"),
                    "confidence": data.get("confidence"),
                    "method": data.get("method"),
                }
            )
        return compact

    route_history = state.get("route_history", [])
    stop_history = state.get("stop_history", [])
    latest_route = _serialize(route_history[-1]) if route_history else None
    latest_stop = _serialize(stop_history[-1]) if stop_history else None
    turn_trace = _serialize(state.get("turn_trace", {}))
    if isinstance(turn_trace, dict):
        turn_trace = {
            key: turn_trace.get(key)
            for key in (
                "supervisor",
                "specialist",
                "extract_evidence",
                "belief_update",
                "update_beliefs",
                "stop",
                "persona_handoff",
            )
            if key in turn_trace
        }
    return {
        "turn": int(state.get("turn_index", 0)),
        "route_decision": latest_route,
        "latest_evidence": _compact_evidence(state.get("latest_turn_evidence", [])),
        "stop_decision": latest_stop,
        "predicted": {
            "label": state.get("predicted_label"),
            "bdi_score": state.get("predicted_bdi_score"),
            "confidence": state.get("global_confidence", 0.0),
            "risk_flag": bool(state.get("risk_flag", False)),
        },
        "raw_predicted_label": state.get("raw_predicted_label"),
        "raw_predicted_bdi_score": state.get("raw_predicted_bdi_score"),
        "route_debug": state.get("route_debug", ""),
        "specialist_debug": state.get("specialist_debug", ""),
        "stop_debug": state.get("stop_debug", ""),
        "turn_trace": turn_trace,
        "failure_counters": _serialize(state.get("failure_counters", {})),
        "empty_evidence_streak": int(state.get("empty_evidence_streak", 0)),
    }


def _mark_budget_exceeded(state: Dict, where: str, exc: Exception) -> Dict:
    counters = dict(state.get("failure_counters", {}))
    counters["budget_exceeded"] = int(counters.get("budget_exceeded", 0)) + 1
    trace = dict(state.get("turn_trace", {}))
    trace["budget"] = {"where": where, "error": str(exc)}
    state["failure_counters"] = counters
    state["turn_trace"] = trace
    state["should_stop"] = True
    state["stop_debug"] = f"Budget exceeded at {where}: {exc}"
    trace_log = list(state.get("trace_log", []))
    trace_log.append(
        {
            "turn": int(state.get("turn_index", 0)),
            "turn_trace": trace,
            "stop_debug": state["stop_debug"],
            "failure_counters": counters,
        }
    )
    state["trace_log"] = trace_log
    return state


def _usage_snippet() -> str:
    usage = get_llm_usage()
    max_calls = usage.get("max_calls")
    calls_total = int(usage.get("calls_total", 0))
    if max_calls is None:
        return f"calls={calls_total}/inf"
    return f"calls={calls_total}/{int(max_calls)}"
