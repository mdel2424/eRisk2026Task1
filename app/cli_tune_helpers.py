from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, List


@contextmanager
def _temporary_env(overrides: Dict[str, Any]):
    previous: Dict[str, str | None] = {}
    for key, value in overrides.items():
        previous[key] = os.getenv(key)
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _route_collapse_flag(route_distribution: Dict[str, int]) -> tuple[bool, float]:
    total = sum(int(v) for v in route_distribution.values())
    if total <= 0:
        return False, 0.0
    max_share = max(float(v) for v in route_distribution.values()) / float(total)
    return max_share > 0.85, round(max_share, 6)


def _candidate_sort_key(record: Dict[str, Any]) -> tuple[float, float, float]:
    metrics = record.get("metrics", {}) or {}
    objective = float(metrics.get("objective", 0.0))
    binary_f1 = float(metrics.get("binary_f1", 0.0))
    avg_turns = float(metrics.get("avg_turns_to_decision", 0.0))
    penalty = 0.03 if bool(record.get("route_collapse_flag", False)) else 0.0
    return (objective - penalty, binary_f1, -avg_turns)


def _apply_guardrails(
    *,
    record: Dict[str, Any],
    baseline_binary_f1: float,
    baseline_risk_recall: float,
) -> Dict[str, Any]:
    reasons: List[str] = []
    profiles_evaluated = int(record.get("profiles_evaluated", 0))
    expected_eval_profiles = int(record.get("expected_eval_profiles", 0))
    if expected_eval_profiles > 0 and profiles_evaluated < expected_eval_profiles:
        reasons.append("truncated_profiles_evaluated")

    failure_counters = record.get("failure_counters", {}) or {}
    if int(failure_counters.get("budget_exceeded", 0)) > 0:
        reasons.append("budget_exceeded")

    usage = record.get("llm_usage", {}) or {}
    max_calls = usage.get("max_calls")
    calls_total = int(usage.get("calls_total", 0))
    if max_calls is not None and calls_total >= int(max_calls) and profiles_evaluated < expected_eval_profiles:
        reasons.append("api_budget_hit_before_completion")

    metrics = record.get("metrics", {}) or {}
    binary_f1 = float(metrics.get("binary_f1", 0.0))
    risk_recall = float(metrics.get("risk_recall", 0.0))
    if binary_f1 < (baseline_binary_f1 - 0.02):
        reasons.append("binary_f1_guardrail")
    if risk_recall < (baseline_risk_recall - 0.10):
        reasons.append("risk_recall_guardrail")

    route_distribution = record.get("route_distribution", {}) or {}
    collapse_flag, collapse_share = _route_collapse_flag(route_distribution)
    record["route_collapse_flag"] = collapse_flag
    record["route_collapse_max_share"] = collapse_share
    record["valid"] = len(reasons) == 0
    record["invalid_reasons"] = reasons
    return record


def _pick_top_candidates(records: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    if not records:
        return []
    valid = [row for row in records if bool(row.get("valid", False))]
    pool = valid if valid else records
    ranked = sorted(pool, key=_candidate_sort_key, reverse=True)
    return ranked[: max(1, int(top_k))]


def _build_best_thresholds_env(overrides: Dict[str, Any]) -> str:
    keys = [
        "MIN_TURNS",
        "MAX_TURNS",
        "STOP_CONFIDENCE",
        "MIN_EVIDENCE_FOR_CONF_STOP",
        "SUPERVISOR_EVIDENCE_MIN_SCORE",
        "SUPERVISOR_EVIDENCE_RISK_THRESHOLD",
        "SUPERVISOR_ESCAPE_EMPTY_STREAK",
    ]
    lines = [f"{key}={overrides[key]}" for key in keys if key in overrides]
    return "\n".join(lines) + ("\n" if lines else "")
