from __future__ import annotations

from typing import Dict, List

from core.state import AgentState, NextAction, RouteDecision, SPECIALIST_ITEM_MAP


STYLE_CYCLE = ("gentle_probe", "clarify_frequency", "functional_impact")


def _priority_from_gain(expected_gain: float) -> float:
    gain = max(0.0, float(expected_gain))
    # Smooth normalization so larger expected-gain targets get higher handoff priority.
    return max(0.0, min(1.0, gain / (gain + 2.0)))



def _recent_target_counts(route_history: List, window: int = 4) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for row in route_history[-window:]:
        if isinstance(row, dict):
            target_items = row.get("target_items", []) or []
        else:
            target_items = getattr(row, "target_items", []) or []
        for item in target_items:
            try:
                item_id = int(item)
            except (TypeError, ValueError):
                continue
            counts[item_id] = int(counts.get(item_id, 0)) + 1
    return counts



def _route_for_item(item_id: int) -> str:
    if item_id in SPECIALIST_ITEM_MAP["risk"]:
        return "risk"
    if item_id in SPECIALIST_ITEM_MAP["somatic"]:
        return "somatic"
    return "cognitive"



def _select_target_item(state: AgentState) -> tuple[int, float, str]:
    beliefs = state.get("item_beliefs", {})
    metrics = state.get("metrics")
    top_uncertain_items = list(getattr(metrics, "top_uncertain_items", []) or [])
    ig_estimates = dict(getattr(metrics, "last_ig_estimates", {}) or {})

    if not top_uncertain_items:
        top_uncertain_items = list(range(1, 22))

    recent_counts = _recent_target_counts(list(state.get("route_history", [])))

    best_item_id = 2
    best_score = float("-inf")
    for item_id in top_uncertain_items:
        try:
            parsed = int(item_id)
        except (TypeError, ValueError):
            continue
        if parsed < 1 or parsed > 21:
            continue

        entropy = 1.0
        belief = beliefs.get(parsed)
        if belief is not None:
            try:
                entropy = float(getattr(belief, "entropy", 1.0))
            except (TypeError, ValueError):
                entropy = 1.0

        ig = float(ig_estimates.get(parsed, entropy))
        risk_weight = 0.20 if parsed == 9 else 0.0
        repetition_penalty = 0.35 * float(recent_counts.get(parsed, 0))
        score = ig + entropy + risk_weight - repetition_penalty

        if score > best_score:
            best_score = score
            best_item_id = parsed

    rationale = (
        "entropy + IG objective"
        if best_score > float("-inf")
        else "fallback default"
    )
    return best_item_id, max(0.0, best_score if best_score > float("-inf") else 0.0), rationale



def target_selector(state: AgentState) -> Dict:
    turn_index = int(state.get("turn_index", 0))
    has_new_persona_input = bool(state.get("has_new_persona_input", False))

    if turn_index == 0 and not has_new_persona_input:
        target_item_id, expected_gain, rationale = 2, 0.0, "opening bootstrap"
        route = "cognitive"
        style = "opening"
    else:
        target_item_id, expected_gain, rationale = _select_target_item(state)
        route = _route_for_item(target_item_id)
        style = STYLE_CYCLE[turn_index % len(STYLE_CYCLE)]

    next_action = NextAction(
        target_item_id=target_item_id,
        route=route,
        style=style,
        mode="normal",
        directness="indirect",
        priority=_priority_from_gain(expected_gain),
        rationale=rationale,
    )

    decision = RouteDecision(
        turn=max(1, turn_index),
        chosen_node=route,
        policy="entropy_penalized",
        reason=rationale,
        target_items=[target_item_id],
        expected_gain=float(expected_gain),
    )

    turn_trace = dict(state.get("turn_trace", {}))
    supervisor_trace = {
        "turn": max(1, turn_index),
        "policy": "entropy_penalized",
        "chosen_node": route,
        "reason": rationale,
        "target_items": [target_item_id],
        "expected_gain": round(float(expected_gain), 4),
    }
    turn_trace["target_selector"] = supervisor_trace
    turn_trace["supervisor"] = supervisor_trace

    debug = (
        f"Target selector -> {route}; target_item={target_item_id}; "
        f"style={style}; gain={expected_gain:.2f}"
    )

    return {
        "next_action": next_action,
        "next_node": route,
        "active_node": route,
        "route_history": [decision],
        "route_debug": debug,
        "turn_trace": turn_trace,
    }
