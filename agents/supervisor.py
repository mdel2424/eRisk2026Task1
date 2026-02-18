from __future__ import annotations

from typing import Dict, List, Tuple

from core.state import AgentState, ItemBelief, RouteDecision, SPECIALIST_ITEM_MAP

ROUTE_CUES: Dict[str, List[str]] = {
    "risk": [
        "suicide",
        "kill myself",
        "end it",
        "better off dead",
        "don't want to live",
        "hurt myself",
        "self harm",
    ],
    "somatic": [
        "sleep",
        "rest",
        "insomnia",
        "tired",
        "fatigue",
        "energy",
        "appetite",
        "eat",
        "weight",
        "agitated",
    ],
    "cognitive": [
        "worthless",
        "guilty",
        "failure",
        "hopeless",
        "pessimistic",
        "hate myself",
        "no future",
        "stuck",
    ],
}


def _latest_persona_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""


def _belief_uncertainty(value) -> float:
    if isinstance(value, ItemBelief):
        return float(value.uncertainty)
    if isinstance(value, dict):
        return float(value.get("uncertainty", 1.0))
    return 1.0


def _route_from_info_gain(state: AgentState) -> Tuple[str, float, List[int]]:
    item_beliefs = state.get("item_beliefs", {})
    node_gain: Dict[str, float] = {}
    node_item_utility: Dict[str, List[Tuple[int, float]]] = {}

    for node, item_ids in SPECIALIST_ITEM_MAP.items():
        weighted = []
        for item_id in item_ids:
            belief = item_beliefs.get(item_id)
            uncertainty = _belief_uncertainty(belief)
            clinical_weight = 2.0 if item_id == 9 else 1.0
            weighted.append((item_id, uncertainty * clinical_weight))
        node_item_utility[node] = weighted
        node_gain[node] = sum(value for _, value in weighted)

    chosen_node = max(node_gain.items(), key=lambda pair: pair[1])[0]
    ranked_items = sorted(node_item_utility[chosen_node], key=lambda pair: pair[1], reverse=True)
    target_items = [item_id for item_id, _ in ranked_items[:3]]
    return chosen_node, node_gain[chosen_node], target_items


def _route_decision(
    state: AgentState,
    chosen_node: str,
    policy: str,
    reason: str,
    target_items: List[int],
    expected_gain: float,
) -> Dict:
    turn = int(state.get("turn_index", 0)) + 1
    decision = RouteDecision(
        turn=turn,
        chosen_node=chosen_node,
        policy=policy,
        reason=reason,
        target_items=target_items,
        expected_gain=expected_gain,
    )
    return {
        "next_node": chosen_node,
        "active_node": chosen_node,
        "route_debug": (
            f"Supervisor -> {chosen_node} ({policy}: {reason}; "
            f"targets={target_items}; gain={expected_gain:.2f})"
        ),
        "route_history": [decision],
    }


def supervisor_router(state: AgentState):
    latest_message = _latest_persona_message(state)
    text = latest_message.lower()

    risk_hits = [cue for cue in ROUTE_CUES["risk"] if cue in text]
    if risk_hits:
        return _route_decision(
            state=state,
            chosen_node="risk",
            policy="lexical_override",
            reason=", ".join(risk_hits[:3]),
            target_items=[9],
            expected_gain=2.0,
        )

    somatic_hits = [cue for cue in ROUTE_CUES["somatic"] if cue in text]
    if somatic_hits:
        return _route_decision(
            state=state,
            chosen_node="somatic",
            policy="lexical",
            reason=", ".join(somatic_hits[:3]),
            target_items=SPECIALIST_ITEM_MAP["somatic"][:3],
            expected_gain=1.0,
        )

    cognitive_hits = [cue for cue in ROUTE_CUES["cognitive"] if cue in text]
    if cognitive_hits:
        return _route_decision(
            state=state,
            chosen_node="cognitive",
            policy="lexical",
            reason=", ".join(cognitive_hits[:3]),
            target_items=SPECIALIST_ITEM_MAP["cognitive"][:3],
            expected_gain=1.0,
        )

    chosen_node, expected_gain, target_items = _route_from_info_gain(state)
    reason = "highest uncertainty utility"
    if not text:
        reason = "opening turn uncertainty bootstrap"
    return _route_decision(
        state=state,
        chosen_node=chosen_node,
        policy="info_gain",
        reason=reason,
        target_items=target_items,
        expected_gain=expected_gain,
    )
