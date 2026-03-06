from __future__ import annotations

import os
from typing import Dict, Tuple

from core.state import AgentState, ControlState, StopDecision



def _items_attempted(state: AgentState) -> set[int]:
    """Return the set of item_ids that have been targeted at least once."""
    attempted: set[int] = set()
    for row in state.get("route_history", []):
        if isinstance(row, dict):
            targets = row.get("target_items", []) or []
        else:
            targets = getattr(row, "target_items", []) or []
        for item in targets:
            try:
                item_id = int(item)
                if 1 <= item_id <= 21:
                    attempted.add(item_id)
            except (TypeError, ValueError):
                continue
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
            eligible, _, _ = _structural_stop_eligible(state)
            if eligible:
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
    structural_eligible = should_stop and stop_reason == "structural_coverage_met"

    debug_line = (
        f"Stop decider: turn={int(state.get('turn_index', 0))}, "
        f"conf={confidence:.2f} (logging only), "
        f"coverage={coverage:.2f}, avg_support={avg_support:.2f}, "
        f"structural_eligible={structural_eligible}, "
        f"risk={bool(state.get('risk_flag', False))}, "
        f"stop={should_stop} ({stop_reason})"
    )

    turn_trace = dict(state.get("turn_trace", {}))
    stop_trace = {
        "turn": int(state.get("turn_index", 0)),
        "confidence": round(confidence, 4),
        "confidence_logging_only": True,
        "stop_method": "structural_coverage_support",
        "coverage": round(coverage, 4),
        "avg_support": round(avg_support, 4),
        "structural_eligible": structural_eligible,
        "should_stop": should_stop,
        "reason": stop_reason,
        "label": predicted_label,
        "risk_flag": bool(state.get("risk_flag", False)),
        "min_turns": int(os.getenv("MIN_TURNS", "20")),
        "max_turns": int(os.getenv("MAX_TURNS", "40")),
        "stop_min_coverage": float(os.getenv("STOP_MIN_COVERAGE", "0.714")),
        "stop_min_avg_support": float(os.getenv("STOP_MIN_AVG_SUPPORT", "1.0")),
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
