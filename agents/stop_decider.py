from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from core.bdi_modules import MODULE_TO_ITEMS
from core.state import AgentState, ControlState, StopDecision


RISK_REENTRY_LOOKBACK = 6
RISK_REENTRY_TOTAL_BDI_THRESHOLD = 18.0
RISK_REENTRY_CORE_SUPPORT_THRESHOLD = 3
EVIDENCE_SATURATION_MIN_CONFIDENCE = 0.72
EVIDENCE_SATURATION_MIN_ATTEMPTED_ITEMS = 12
EVIDENCE_SATURATION_EMPTY_STREAK = 3
EVIDENCE_SATURATION_UPDATE_WINDOW = 3
EVIDENCE_SATURATION_HIGH_ENTROPY_THRESHOLD = 1.25
EVIDENCE_SATURATION_MAX_HIGH_ENTROPY_ITEMS = 5


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _belief_support(state: AgentState, item_id: int) -> int:
    belief = dict(state.get("item_beliefs", {})).get(int(item_id))
    try:
        return int(getattr(belief, "support_count", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _route_history_target_items(row: Any) -> list[int]:
    if isinstance(row, dict):
        raw_items = row.get("target_items", []) or []
    else:
        raw_items = getattr(row, "target_items", []) or []
    target_items: list[int] = []
    for raw_item in raw_items:
        try:
            item_id = int(raw_item)
        except (TypeError, ValueError):
            continue
        if 1 <= item_id <= 21 and item_id not in target_items:
            target_items.append(item_id)
    return target_items


def _route_history_string(row: Any, field: str, default: str = "") -> str:
    if isinstance(row, dict):
        value = row.get(field, default)
    else:
        value = getattr(row, field, default)
    text = str(value or "").strip().lower()
    return text or str(default or "").strip().lower()


def _items_attempted(state: AgentState) -> set[int]:
    """Return the set of item_ids that have been targeted at least once."""
    attempted: set[int] = set()
    for row in state.get("route_history", []):
        for item_id in _route_history_target_items(row):
            attempted.add(item_id)
    return attempted


def _structural_stop_eligible(state: AgentState) -> Tuple[bool, float, float]:
    """Check whether coverage and evidence depth are sufficient to stop.

    Coverage counts items that have been *attempted* (asked about at least
    once), not just items with positive evidence.  This ensures low-BDI
    personas — where most items legitimately score 0 — can still reach
    the coverage threshold.
    """
    min_coverage = float(os.getenv("STOP_MIN_COVERAGE", "0.714"))
    min_avg_support = float(os.getenv("STOP_MIN_AVG_SUPPORT", "1.0"))

    attempted = _items_attempted(state)
    coverage = len(attempted) / 21.0

    # avg_support is computed over items that DO have evidence.
    beliefs = state.get("item_beliefs", {})
    observed_supports = []
    for item_id in range(1, 22):
        belief = beliefs.get(item_id)
        support = 0
        if belief is not None:
            try:
                support = int(getattr(belief, "support_count", 0))
            except (TypeError, ValueError):
                support = 0
        if support > 0:
            observed_supports.append(support)

    avg_support = (sum(observed_supports) / len(observed_supports)) if observed_supports else 0.0

    # For low-BDI personas most items have no evidence.  Require avg_support
    # only when there IS evidence; otherwise coverage alone is sufficient.
    if observed_supports:
        eligible = coverage >= min_coverage and avg_support >= min_avg_support
    else:
        eligible = coverage >= min_coverage
    return eligible, coverage, avg_support


def _recent_updated_item_count(state: AgentState, *, window: int) -> int:
    turn_index = int(state.get("turn_index", 0))
    if turn_index <= 0:
        return 0
    lower_bound = max(1, turn_index - int(window) + 1)
    count = 0
    for belief in dict(state.get("item_beliefs", {})).values():
        try:
            last_update_turn = int(getattr(belief, "last_update_turn", 0) or 0)
        except (TypeError, ValueError):
            last_update_turn = 0
        if last_update_turn >= lower_bound:
            count += 1
    return count


def _high_entropy_unresolved_count(state: AgentState, *, threshold: float) -> int:
    attempted = _items_attempted(state)
    if not attempted:
        return 0
    count = 0
    beliefs = dict(state.get("item_beliefs", {}))
    for item_id in attempted:
        belief = beliefs.get(int(item_id))
        if belief is None:
            continue
        try:
            support_count = int(getattr(belief, "support_count", 0) or 0)
        except (TypeError, ValueError):
            support_count = 0
        try:
            entropy = float(getattr(belief, "entropy", 2.0) or 2.0)
        except (TypeError, ValueError):
            entropy = 2.0
        if support_count == 0 and entropy >= float(threshold):
            count += 1
    return count


def _recent_risk_attempted(state: AgentState, *, window: int) -> bool:
    for row in list(state.get("route_history", []))[-window:]:
        if 9 in _route_history_target_items(row):
            return True
        if _route_history_string(row, "chosen_node", "") == "risk":
            return True
    return False


def _core_module_support_count(state: AgentState) -> int:
    core_items = set(MODULE_TO_ITEMS[1]) | set(MODULE_TO_ITEMS[3])
    return sum(1 for item_id in core_items if _belief_support(state, item_id) > 0)


def _risk_reentry_reason(state: AgentState) -> str:
    if _belief_support(state, 9) > 0:
        return ""
    if _recent_risk_attempted(state, window=_env_int("RISK_REENTRY_LOOKBACK", RISK_REENTRY_LOOKBACK)):
        return ""
    if bool(state.get("risk_flag", False)):
        return "risk_flag"

    metrics = state.get("metrics")
    total_expected_bdi = 0.0
    try:
        total_expected_bdi = float(getattr(metrics, "total_expected_bdi", 0.0) or 0.0)
    except (TypeError, ValueError):
        total_expected_bdi = 0.0
    if total_expected_bdi >= _env_float("RISK_REENTRY_TOTAL_BDI_THRESHOLD", RISK_REENTRY_TOTAL_BDI_THRESHOLD):
        return "high_expected_bdi"

    if _core_module_support_count(state) >= _env_int(
        "RISK_REENTRY_CORE_SUPPORT_THRESHOLD",
        RISK_REENTRY_CORE_SUPPORT_THRESHOLD,
    ):
        return "supported_core_modules"
    return ""


def _risk_probe_pending(state: AgentState) -> tuple[bool, str, bool]:
    lookback = _env_int("RISK_REENTRY_LOOKBACK", RISK_REENTRY_LOOKBACK)
    recent_risk_attempted = _recent_risk_attempted(state, window=lookback)
    reason = _risk_reentry_reason(state)
    pending = bool(reason) and not recent_risk_attempted
    return pending, reason, recent_risk_attempted


def _evidence_saturation_eligible(state: AgentState) -> tuple[bool, Dict[str, float | int | bool | str]]:
    min_confidence = _env_float("STOP_MIN_GLOBAL_CONFIDENCE", EVIDENCE_SATURATION_MIN_CONFIDENCE)
    min_attempted_items = _env_int("STOP_MIN_ATTEMPTED_ITEMS", EVIDENCE_SATURATION_MIN_ATTEMPTED_ITEMS)
    empty_streak_threshold = _env_int("STOP_EMPTY_STREAK", EVIDENCE_SATURATION_EMPTY_STREAK)
    update_window = _env_int("STOP_RECENT_UPDATE_WINDOW", EVIDENCE_SATURATION_UPDATE_WINDOW)
    high_entropy_threshold = _env_float(
        "STOP_HIGH_ENTROPY_THRESHOLD",
        EVIDENCE_SATURATION_HIGH_ENTROPY_THRESHOLD,
    )
    max_high_entropy_items = _env_int(
        "STOP_MAX_HIGH_ENTROPY_ITEMS",
        EVIDENCE_SATURATION_MAX_HIGH_ENTROPY_ITEMS,
    )

    confidence = float(state.get("global_confidence", 0.0))
    items_attempted_count = len(_items_attempted(state))
    empty_evidence_streak = max(0, int(state.get("empty_evidence_streak", 0)))
    recent_updated_item_count = _recent_updated_item_count(state, window=update_window)
    high_entropy_unresolved_count = _high_entropy_unresolved_count(state, threshold=high_entropy_threshold)
    risk_probe_pending, risk_reentry_reason, recent_risk_attempted = _risk_probe_pending(state)

    productivity_stalled = (
        empty_evidence_streak >= empty_streak_threshold or recent_updated_item_count == 0
    )
    eligible = (
        confidence >= min_confidence
        and items_attempted_count >= min_attempted_items
        and productivity_stalled
        and high_entropy_unresolved_count <= max_high_entropy_items
        and not risk_probe_pending
    )
    return eligible, {
        "items_attempted_count": int(items_attempted_count),
        "empty_evidence_streak": int(empty_evidence_streak),
        "recent_updated_item_count": int(recent_updated_item_count),
        "high_entropy_unresolved_count": int(high_entropy_unresolved_count),
        "risk_probe_pending": bool(risk_probe_pending),
        "risk_reentry_reason": str(risk_reentry_reason),
        "risk_recent_attempted": bool(recent_risk_attempted),
        "productivity_stalled": bool(productivity_stalled),
        "min_confidence": float(min_confidence),
        "min_attempted_items": int(min_attempted_items),
        "empty_streak_threshold": int(empty_streak_threshold),
        "update_window": int(update_window),
        "high_entropy_threshold": float(high_entropy_threshold),
        "max_high_entropy_items": int(max_high_entropy_items),
    }


def compute_stop_decision(state: AgentState) -> Tuple[bool, str, float]:
    min_turns = int(os.getenv("MIN_TURNS", "20"))
    max_turns = int(os.getenv("MAX_TURNS", "40"))

    turn_index = int(state.get("turn_index", 0))
    has_new_persona_input = bool(state.get("has_new_persona_input", False))
    confidence = float(state.get("global_confidence", 0.0))

    should_stop = False
    reason = "continue"

    if not has_new_persona_input:
        reason = "opening_turn" if turn_index == 0 else "awaiting_persona_input"
    else:
        if turn_index >= max_turns:
            should_stop = True
            reason = "max_turns_reached"
        elif turn_index >= min_turns:
            saturation_eligible, saturation_stats = _evidence_saturation_eligible(state)
            eligible, _, _ = _structural_stop_eligible(state)
            risk_probe_pending = bool(saturation_stats["risk_probe_pending"])
            risk_flag = bool(state.get("risk_flag", False))
            if saturation_eligible:
                should_stop = True
                reason = "evidence_saturation_met"
            elif eligible and (not risk_probe_pending or not risk_flag):
                should_stop = True
                reason = "structural_coverage_met"

    return should_stop, reason, confidence



def stop_decider(state: AgentState) -> Dict:
    should_stop, stop_reason, confidence = compute_stop_decision(state)

    predicted_label = str(state.get("raw_predicted_label") or state.get("predicted_label") or "control")
    if predicted_label not in {"control", "depressed"}:
        predicted_label = "control"

    predicted_bdi_score = int(state.get("raw_predicted_bdi_score") or state.get("predicted_bdi_score") or 0)

    stop_history_payload = []
    if bool(state.get("has_new_persona_input", False)):
        stop_record = StopDecision(
            turn=max(1, int(state.get("turn_index", 0))),
            should_stop=should_stop,
            reason=stop_reason,
            predicted_label=predicted_label,
            predicted_bdi_score=max(0, min(63, predicted_bdi_score)),
            confidence=max(0.0, min(1.0, confidence)),
        )
        stop_history_payload = [stop_record]

    _, coverage, avg_support = _structural_stop_eligible(state)
    saturation_eligible, saturation_stats = _evidence_saturation_eligible(state)
    structural_eligible = should_stop and stop_reason == "structural_coverage_met"

    debug_line = (
        f"Stop decider: turn={int(state.get('turn_index', 0))}, "
        f"conf={confidence:.2f} (logging only), "
        f"coverage={coverage:.2f}, avg_support={avg_support:.2f}, "
        f"attempted={int(saturation_stats['items_attempted_count'])}, "
        f"recent_updates={int(saturation_stats['recent_updated_item_count'])}, "
        f"high_entropy_unresolved={int(saturation_stats['high_entropy_unresolved_count'])}, "
        f"risk_probe_pending={bool(saturation_stats['risk_probe_pending'])}, "
        f"structural_eligible={structural_eligible}, "
        f"risk={bool(state.get('risk_flag', False))}, "
        f"stop={should_stop} ({stop_reason})"
    )

    turn_trace = dict(state.get("turn_trace", {}))
    stop_trace = {
        "turn": int(state.get("turn_index", 0)),
        "confidence": round(confidence, 4),
        "confidence_logging_only": True,
        "stop_method": "evidence_saturation_support",
        "coverage": round(coverage, 4),
        "avg_support": round(avg_support, 4),
        "items_attempted_count": int(saturation_stats["items_attempted_count"]),
        "recent_updated_item_count": int(saturation_stats["recent_updated_item_count"]),
        "high_entropy_unresolved_count": int(saturation_stats["high_entropy_unresolved_count"]),
        "evidence_saturation_eligible": bool(saturation_eligible),
        "risk_probe_pending": bool(saturation_stats["risk_probe_pending"]),
        "risk_reentry_reason": str(saturation_stats["risk_reentry_reason"]),
        "risk_recent_attempted": bool(saturation_stats["risk_recent_attempted"]),
        "structural_eligible": structural_eligible,
        "should_stop": should_stop,
        "reason": stop_reason,
        "label": predicted_label,
        "risk_flag": bool(state.get("risk_flag", False)),
        "min_turns": int(os.getenv("MIN_TURNS", "20")),
        "max_turns": int(os.getenv("MAX_TURNS", "40")),
        "stop_min_coverage": float(os.getenv("STOP_MIN_COVERAGE", "0.714")),
        "stop_min_avg_support": float(os.getenv("STOP_MIN_AVG_SUPPORT", "1.0")),
        "stop_min_global_confidence": float(saturation_stats["min_confidence"]),
        "stop_min_attempted_items": int(saturation_stats["min_attempted_items"]),
        "stop_empty_streak_threshold": int(saturation_stats["empty_streak_threshold"]),
        "stop_recent_update_window": int(saturation_stats["update_window"]),
        "stop_high_entropy_threshold": float(saturation_stats["high_entropy_threshold"]),
        "stop_max_high_entropy_items": int(saturation_stats["max_high_entropy_items"]),
    }
    turn_trace["stop_decider"] = stop_trace
    turn_trace["stop"] = stop_trace

    return {
        "control": ControlState(stop=should_stop, stop_reason=stop_reason),
        "should_stop": should_stop,
        "stop_debug": debug_line,
        "stop_history": stop_history_payload,
        "turn_trace": turn_trace,
    }
