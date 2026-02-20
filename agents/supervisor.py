from __future__ import annotations

import os
from typing import Dict, List, Tuple

from core.state import AgentState, BDI_ITEM_NAMES, ItemBelief, RouteDecision, SPECIALIST_ITEM_MAP


def _latest_persona_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""


def _record_item_id(record) -> int | None:
    if isinstance(record, dict):
        raw = record.get("item_id")
    else:
        raw = getattr(record, "item_id", None)
    try:
        item_id = int(raw)
    except (TypeError, ValueError):
        return None
    if 1 <= item_id <= 21:
        return item_id
    return None


def _record_confidence(record) -> float:
    if isinstance(record, dict):
        raw = record.get("confidence", 0.0)
    else:
        raw = getattr(record, "confidence", 0.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.0
    return max(0.0, min(1.0, value))


def _record_intensity(record) -> float:
    if isinstance(record, dict):
        raw = record.get("intensity", 0.0)
    else:
        raw = getattr(record, "intensity", 0.0)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.0
    return max(0.0, min(3.0, value))


def _route_from_evidence(state: AgentState) -> Tuple[str, float, List[int], str] | None:
    latest_turn_evidence = list(state.get("latest_turn_evidence", []))
    if not latest_turn_evidence:
        return None

    node_scores: Dict[str, float] = {node: 0.0 for node in SPECIALIST_ITEM_MAP}
    node_item_scores: Dict[str, Dict[int, float]] = {node: {} for node in SPECIALIST_ITEM_MAP}

    for record in latest_turn_evidence:
        item_id = _record_item_id(record)
        if item_id is None:
            continue
        confidence = _record_confidence(record)
        intensity = _record_intensity(record)
        signal = confidence * max(0.15, intensity / 3.0)
        if signal <= 0.0:
            continue
        for node, item_ids in SPECIALIST_ITEM_MAP.items():
            if item_id in item_ids:
                node_scores[node] += signal
                node_item_scores[node][item_id] = node_item_scores[node].get(item_id, 0.0) + signal

    risk_override_threshold = float(os.getenv("SUPERVISOR_EVIDENCE_RISK_THRESHOLD", "0.22"))
    risk_signal = float(node_item_scores.get("risk", {}).get(9, 0.0))
    if risk_signal >= risk_override_threshold:
        return "risk", risk_signal, [9], f"item_9_signal={risk_signal:.2f}"

    chosen_node, score = max(node_scores.items(), key=lambda pair: pair[1])
    min_score = float(os.getenv("SUPERVISOR_EVIDENCE_MIN_SCORE", "0.30"))
    if score < min_score:
        return None

    item_score_pairs = sorted(
        node_item_scores.get(chosen_node, {}).items(),
        key=lambda pair: pair[1],
        reverse=True,
    )
    target_items = [item_id for item_id, _ in item_score_pairs[:3]]
    if not target_items:
        target_items = SPECIALIST_ITEM_MAP.get(chosen_node, [])[:3]

    top_items = ", ".join(
        f"{item_id}:{BDI_ITEM_NAMES.get(item_id, f'Item {item_id}')}" for item_id in target_items[:3]
    )
    reason = f"score={score:.2f}; items={top_items or 'n/a'}"
    return chosen_node, score, target_items, reason


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


def _node_expected_gain(state: AgentState, node_name: str) -> float:
    item_beliefs = state.get("item_beliefs", {})
    total = 0.0
    for item_id in SPECIALIST_ITEM_MAP.get(node_name, []):
        belief = item_beliefs.get(item_id)
        uncertainty = _belief_uncertainty(belief)
        clinical_weight = 2.0 if item_id == 9 else 1.0
        total += uncertainty * clinical_weight
    return float(total)


def _escape_node(active_node: str) -> str:
    if active_node == "somatic":
        return "cognitive"
    if active_node == "cognitive":
        return "somatic"
    return "cognitive"


def _route_decision(
    state: AgentState,
    chosen_node: str,
    policy: str,
    reason: str,
    target_items: List[int],
    expected_gain: float,
    matched_cues: List[str],
) -> Dict:
    current_turn = int(state.get("turn_index", 0))
    has_new_persona_input = bool(state.get("has_new_persona_input", False))
    turn = max(1, current_turn if has_new_persona_input else (current_turn + 1))
    decision = RouteDecision(
        turn=turn,
        chosen_node=chosen_node,
        policy=policy,
        reason=reason,
        target_items=target_items,
        expected_gain=expected_gain,
    )
    trace = dict(state.get("turn_trace", {}))
    trace["supervisor"] = {
        "turn": turn,
        "policy": policy,
        "chosen_node": chosen_node,
        "reason": reason,
        "matched_cues": matched_cues[:6],
        "target_items": target_items,
        "expected_gain": round(float(expected_gain), 4),
    }
    return {
        "next_node": chosen_node,
        "active_node": chosen_node,
        "route_debug": (
            f"Supervisor -> {chosen_node} ({policy}: {reason}; "
            f"targets={target_items}; gain={expected_gain:.2f})"
        ),
        "route_history": [decision],
        "turn_trace": trace,
    }


def supervisor_router(state: AgentState):
    latest_message = _latest_persona_message(state)
    text = latest_message.lower()

    evidence_route = _route_from_evidence(state)
    if evidence_route is not None:
        chosen_node, evidence_score, target_items, evidence_reason = evidence_route
        policy = "evidence_vote"
        if chosen_node == "risk":
            policy = "evidence_risk_override"
        return _route_decision(
            state=state,
            chosen_node=chosen_node,
            policy=policy,
            reason=evidence_reason,
            target_items=target_items,
            expected_gain=evidence_score,
            matched_cues=[],
        )

    empty_streak = int(state.get("empty_evidence_streak", 0))
    if empty_streak >= 2:
        active_node = str(state.get("active_node", "cognitive"))
        chosen_node = _escape_node(active_node)
        target_items = SPECIALIST_ITEM_MAP.get(chosen_node, [])[:3]
        expected_gain = _node_expected_gain(state, chosen_node)
        return _route_decision(
            state=state,
            chosen_node=chosen_node,
            policy="escape_streak",
            reason=f"empty_evidence_streak={empty_streak}",
            target_items=target_items,
            expected_gain=expected_gain,
            matched_cues=[],
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
        matched_cues=[],
    )
