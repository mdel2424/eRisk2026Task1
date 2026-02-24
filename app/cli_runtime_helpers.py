from __future__ import annotations

import os
from typing import Any, Dict, List

from core.io_schema import Turn
from core.llm import get_llm_usage
from core.runtime_policy import (
    auto_backend_switch_enabled,
    cuda_runtime,
    min_cuda_vram_gb,
    resolve_detector_backend,
    resolve_persona_backend,
)

from app.cli_common import _serialize


def _print_progress(label: str, current: int, total: int, width: int = 24) -> None:
    total = max(1, total)
    current = max(0, min(current, total))
    filled = int(width * (current / total))
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{label} [{bar}] {current}/{total}", end="", flush=True)
    if current >= total:
        print()


def _print_backend_info(max_api_calls: int | None = None, trace_level: str = "compact") -> None:
    auto_on = auto_backend_switch_enabled()
    detector_backend = resolve_detector_backend()
    persona_backend = resolve_persona_backend()
    cuda_available, vram_gb = cuda_runtime()
    min_vram = min_cuda_vram_gb()
    cuda_gate = "pass" if (cuda_available and vram_gb >= min_vram) else "fail"

    if detector_backend == "openrouter":
        detector_target = os.getenv("OPENROUTER_DETECTOR_MODEL", "openrouter/auto")
    else:
        detector_target = os.getenv("DETECTOR_MODEL", "")

    persona_target = "deterministic_sim"

    print(
        "Backend info: "
        f"auto_switch={'on' if auto_on else 'off'} | "
        f"cuda_available={cuda_available} | vram_gb={vram_gb:.2f} | "
        f"min_vram_gb={min_vram:.2f} | cuda_gate={cuda_gate}"
    )
    print(
        "Resolved backends: "
        f"detector={detector_backend} [{detector_target}] | "
        f"persona={persona_backend} [{persona_target}]"
    )
    call_budget_text = "none" if max_api_calls is None or max_api_calls <= 0 else str(max_api_calls)
    print(f"Runtime controls: trace_level={trace_level} | max_api_calls={call_budget_text}")


def _assert_openrouter_ready() -> None:
    detector_backend = resolve_detector_backend()
    resolve_persona_backend()
    if detector_backend == "openrouter":
        if not os.getenv("OPENROUTER_API_KEY", "").strip():
            raise ValueError(
                "OPENROUTER_API_KEY is required because the resolved detector backend uses OpenRouter."
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
            for key in ("supervisor", "specialist", "extract_evidence", "update_beliefs", "stop", "persona_handoff")
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
