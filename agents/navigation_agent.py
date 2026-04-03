from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple

from core.bdi_modules import MODULE_TO_ITEMS, choose_target_module
from core.probabilistic_runtime import CLUSTER_TO_ITEMS, cluster_for_route, route_for_cluster
from core.state import AgentState, ConversationThreadState, NextAction, QuestionPlan, RouteDecision


THREAD_MAX_QUESTIONS = 2


def _state_value(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _thread_dict(state: AgentState) -> Dict[str, Any]:
    thread = state.get("conversation_thread")
    return {
        "active": bool(_state_value(thread, "active", False)),
        "route": str(_state_value(thread, "route", "cognitive") or "cognitive"),
        "module_id": int(_state_value(thread, "module_id", 0) or 0),
        "source_item_id": int(_state_value(thread, "source_item_id", 0) or 0),
        "question_count": int(_state_value(thread, "question_count", 0) or 0),
        "denial_streak": int(_state_value(thread, "denial_streak", 0) or 0),
        "timeframe_introduced": bool(_state_value(thread, "timeframe_introduced", False)),
        "anchor_text": str(_state_value(thread, "anchor_text", "") or ""),
    }


def _item_state(state: AgentState, item_id: int) -> Any:
    return dict(state.get("bayes_items", {})).get(int(item_id))


def _item_uncertainty(state: AgentState, item_id: int) -> float:
    return max(0.0, min(1.0, float(_state_value(_item_state(state, item_id), "uncertainty", 1.0) or 1.0)))


def _item_presence(state: AgentState, item_id: int) -> float:
    return max(0.0, min(1.0, float(_state_value(_item_state(state, item_id), "presence_prob", 0.0) or 0.0)))


def _item_expected(state: AgentState, item_id: int) -> float:
    return max(0.0, min(3.0, float(_state_value(_item_state(state, item_id), "expected_score", 0.0) or 0.0)))


def _latest_anchor_for_item(state: AgentState, item_id: int) -> str:
    for row in reversed(list(state.get("assertion_log", []))):
        try:
            row_item_id = int(_state_value(row, "item_id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if row_item_id != item_id:
            continue
        quote = str(_state_value(row, "anchor_quote", "") or "").strip()
        if quote:
            return quote
    for row in reversed(list(state.get("evidence_log", []))):
        try:
            row_item_id = int(_state_value(row, "item_id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if row_item_id != item_id:
            continue
        text = str(_state_value(row, "evidence_text", "") or "").strip()
        if text:
            return text
    return ""


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(value or "").lower())).strip()


def _token_overlap(left: str, right: str) -> float:
    left_tokens = {token for token in _normalize_text(left).split() if token}
    right_tokens = {token for token in _normalize_text(right).split() if token}
    if not left_tokens or not right_tokens:
        return 0.0
    return float(len(left_tokens & right_tokens)) / float(len(left_tokens | right_tokens))


def _recent_assistant_messages(state: AgentState, limit: int = 2) -> List[str]:
    messages = [str(msg.get("content", "") or "") for msg in list(state.get("messages", [])) if msg.get("role") == "assistant"]
    return messages[-limit:]


def _stale_thread_reply(state: AgentState) -> bool:
    messages = _recent_assistant_messages(state, limit=2)
    if len(messages) < 2:
        return False
    left, right = messages[-2], messages[-1]
    if _normalize_text(left) == _normalize_text(right):
        return True
    return _token_overlap(left, right) >= 0.82


def _judgment_value(state: AgentState, key: str, default: Any) -> Any:
    judgment = state.get("judgment")
    if isinstance(judgment, dict):
        return judgment.get(key, default)
    return getattr(judgment, key, default)


def _latest_bound_positive_count(state: AgentState) -> int:
    return int(_judgment_value(state, "bound_positive_assertion_count", 0) or 0)


def _latest_emitted_evidence_count(state: AgentState) -> int:
    return int(_judgment_value(state, "emitted_evidence_count", 0) or 0)


def _opening_signal_cluster(state: AgentState) -> str:
    return str(state.get("opening_signal_cluster", "") or "").strip()


def _opening_signal_item_ids(state: AgentState) -> List[int]:
    item_ids: List[int] = []
    for raw_item_id in list(state.get("opening_signal_item_ids", []) or []):
        try:
            item_id = int(raw_item_id)
        except (TypeError, ValueError):
            continue
        if 1 <= item_id <= 21 and item_id not in item_ids:
            item_ids.append(item_id)
    return item_ids


def _thread_helpful(state: AgentState, item_id: int) -> bool:
    if _latest_bound_positive_count(state) > 0 or _latest_emitted_evidence_count(state) > 0:
        return True
    item_state = _item_state(state, item_id)
    return float(_state_value(item_state, "uncertainty", 1.0) or 1.0) <= 0.38 and float(_state_value(item_state, "presence_prob", 0.0) or 0.0) >= 0.40


def _strongest_recent_assertion(state: AgentState) -> Tuple[str, int, float]:
    best_cluster = ""
    best_item_id = 0
    best_score = -1.0
    for row in reversed(list(state.get("latest_turn_assertions", []))):
        label = str(_state_value(row, "assertion_label", "") or "")
        binding_status = str(_state_value(row, "binding_status", "") or "")
        if label not in {"present", "conditional", "contrastive"} or binding_status not in {"exact", "normalized_exact"}:
            continue
        try:
            item_id = int(_state_value(row, "item_id", 0) or 0)
        except (TypeError, ValueError):
            continue
        confidence = max(0.0, min(1.0, float(_state_value(row, "confidence", 0.0) or 0.0)))
        intensity = max(0.0, min(3.0, float(_state_value(row, "intensity", 0.0) or 0.0)))
        score = confidence + (0.12 * intensity)
        cluster_name = "somatic_vegetative" if item_id in CLUSTER_TO_ITEMS.get("somatic_vegetative", []) else (
            "risk" if item_id == 9 else "cognitive_affective"
        )
        if score > best_score:
            best_cluster = cluster_name
            best_item_id = item_id
            best_score = score
    return best_cluster, best_item_id, best_score


def _cluster_reselection_choice(state: AgentState, *, opening_transition: bool = False) -> Tuple[str, str]:
    cognitive_score = _cluster_score(state, "cognitive_affective")
    somatic_score = _cluster_score(state, "somatic_vegetative")
    opening_signal_cluster = _opening_signal_cluster(state)
    opening_signal_item_ids = set(_opening_signal_item_ids(state))
    if (
        opening_signal_cluster == "cognitive_affective"
        and opening_signal_item_ids.intersection({1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 17, 19})
        and (opening_transition or int(state.get("turn_index", 0) or 0) <= 3)
        and somatic_score <= cognitive_score + (0.12 if opening_transition else 0.08)
    ):
        return "cognitive_affective", "opening_cognitive_anchor"
    strongest_recent_cluster, strongest_recent_item_id, _strongest_recent_score = _strongest_recent_assertion(state)
    if (
        opening_transition
        and strongest_recent_cluster == "cognitive_affective"
        and strongest_recent_item_id in {2, 3, 5, 6, 7, 8, 14, 19}
        and somatic_score <= cognitive_score + 0.10
    ):
        return "cognitive_affective", "opening_cognitive_anchor"
    if strongest_recent_cluster in {"cognitive_affective", "somatic_vegetative"}:
        primary_score = cognitive_score if strongest_recent_cluster == "cognitive_affective" else somatic_score
        secondary_score = somatic_score if strongest_recent_cluster == "cognitive_affective" else cognitive_score
        if abs(primary_score - secondary_score) <= (0.10 if opening_transition else 0.08):
            return strongest_recent_cluster, "recent_bound_assertion_preference"
    if cognitive_score >= somatic_score - 0.03:
        return "cognitive_affective", "cognitive_tie_break"
    return "somatic_vegetative", "higher_cluster_score"


def _same_module_followup_target(state: AgentState, module_id: int, source_item_id: int, active_cluster: str) -> int:
    candidate_ids = [
        int(item_id)
        for item_id in MODULE_TO_ITEMS.get(int(module_id), [])
        if int(item_id) != int(source_item_id) and int(item_id) in CLUSTER_TO_ITEMS.get(active_cluster, [])
    ]
    if not candidate_ids:
        return _target_item_for_cluster(state, active_cluster)
    ranked = sorted(
        candidate_ids,
        key=lambda item_id: (
            -(_item_uncertainty(state, item_id) * 0.65 + _item_presence(state, item_id) * 0.35),
            -_item_expected(state, item_id),
            item_id,
        ),
    )
    return int(ranked[0])


def _increment_runtime_counter(counters: Dict[str, int], key: str, amount: int = 1) -> Dict[str, int]:
    updated = dict(counters)
    updated[str(key)] = int(updated.get(str(key), 0)) + max(0, int(amount))
    return updated


def _cluster_score(state: AgentState, cluster_name: str) -> float:
    bayes_nodes = dict(state.get("bayes_nodes", {}))
    node_state = bayes_nodes.get(cluster_name)
    node_prob = float(_state_value(node_state, "probability", 0.0) or 0.0)
    cluster_items = CLUSTER_TO_ITEMS.get(cluster_name, [])
    if not cluster_items:
        return 0.0
    unresolved = sum(_item_uncertainty(state, item_id) for item_id in cluster_items) / float(len(cluster_items))
    active_presence = max((_item_presence(state, item_id) for item_id in cluster_items), default=0.0)
    return (0.55 * unresolved) + (0.30 * node_prob) + (0.15 * active_presence)


def _target_item_for_cluster(state: AgentState, cluster_name: str, preferred_item_id: int | None = None) -> int:
    items = list(CLUSTER_TO_ITEMS.get(cluster_name, []))
    if preferred_item_id and preferred_item_id in items and _item_uncertainty(state, preferred_item_id) >= 0.18:
        return int(preferred_item_id)
    ranked = sorted(
        items,
        key=lambda item_id: (
            -(_item_uncertainty(state, item_id) * 0.70 + _item_presence(state, item_id) * 0.30),
            -_item_expected(state, item_id),
            item_id,
        ),
    )
    return int(ranked[0]) if ranked else 2


def _urgency_mode(turn_index: int) -> str:
    max_turns = max(8, int(os.getenv("MAX_TURNS", "40")))
    return "direct_structured" if turn_index >= max_turns - 4 else "adaptive"


def navigation_agent(state: AgentState) -> Dict[str, Any]:
    turn_index = int(state.get("turn_index", 0) or 0)
    thread = _thread_dict(state)
    risk_prob = float(_state_value(dict(state.get("bayes_nodes", {})).get("risk"), "probability", state.get("risk_prob", 0.0)) or 0.0)
    urgency_mode = _urgency_mode(turn_index)
    runtime_counters = dict(state.get("runtime_counters", {}))
    stale_reply_detected = _stale_thread_reply(state)
    cluster_reselection_reason = ""

    if bool(state.get("risk_flag", False)) or risk_prob >= 0.42:
        active_cluster = "risk"
        route = "risk"
        target_item_id = 9
        question_kind = "risk_check"
        policy = "risk_interrupt"
        transition_reason = "risk_posterior_or_flag"
        thread_turn_index = 1
        module_id = 9
    else:
        active_cluster = ""
        policy = "cluster_topic_open"
        transition_reason = "cluster_reselection"
        question_kind = "topic_open"
        module_id = 0
        thread_turn_index = 1

        if thread["active"] and thread["question_count"] < THREAD_MAX_QUESTIONS:
            candidate_cluster = cluster_for_route(thread["route"])
            if (
                candidate_cluster == "somatic_vegetative"
                and _opening_signal_cluster(state) == "cognitive_affective"
                and str(state.get("opening_followup_cluster", "") or "") == "somatic_vegetative"
                and int(state.get("turn_index", 0) or 0) <= 3
                and _latest_bound_positive_count(state) <= 1
                and _latest_emitted_evidence_count(state) <= 1
            ):
                runtime_counters = _increment_runtime_counter(
                    runtime_counters,
                    "opening_somatic_pivot_after_cognitive_signal_count",
                )
                active_cluster = ""
                transition_reason = "weak_somatic_lockin_rebalance"
                runtime_counters = _increment_runtime_counter(runtime_counters, "same_item_loop_exit_count")
            elif _cluster_score(state, candidate_cluster) >= 0.18:
                active_cluster = candidate_cluster
                route = route_for_cluster(candidate_cluster)
                same_item_allowed = (
                    int(thread["source_item_id"]) > 0
                    and _thread_helpful(state, int(thread["source_item_id"]))
                    and not stale_reply_detected
                    and int(thread["question_count"]) < THREAD_MAX_QUESTIONS
                    and _item_uncertainty(state, int(thread["source_item_id"])) >= 0.22
                )
                if same_item_allowed:
                    target_item_id = int(thread["source_item_id"])
                    question_kind = "same_item_followup"
                    policy = "cluster_same_item_followup"
                    transition_reason = "bounded_thread_continuation"
                else:
                    if stale_reply_detected:
                        runtime_counters = _increment_runtime_counter(runtime_counters, "stale_thread_count")
                        runtime_counters = _increment_runtime_counter(runtime_counters, "same_item_loop_exit_count")
                        transition_reason = "stale_thread_same_module_pivot"
                    elif int(thread["question_count"]) >= THREAD_MAX_QUESTIONS - 1:
                        runtime_counters = _increment_runtime_counter(runtime_counters, "same_item_loop_exit_count")
                        transition_reason = "same_item_loop_cap"
                    else:
                        transition_reason = "same_module_followup_rebalance"
                    target_item_id = _same_module_followup_target(
                        state,
                        module_id=int(thread["module_id"]),
                        source_item_id=int(thread["source_item_id"]),
                        active_cluster=candidate_cluster,
                    )
                    question_kind = "same_module_followup"
                    policy = "cluster_same_module_followup"
                thread_turn_index = int(thread["question_count"]) + 1
                module_id = int(thread["module_id"] or choose_target_module(route, [target_item_id], state.get("item_beliefs", {})))
            else:
                runtime_counters = _increment_runtime_counter(runtime_counters, "cluster_saturation_count")
                active_cluster = ""
                transition_reason = "cluster_saturated_reselect"

        if not active_cluster:
            active_cluster, cluster_reselection_reason = _cluster_reselection_choice(
                state,
                opening_transition=(turn_index <= 1),
            )
            transition_reason = cluster_reselection_reason or transition_reason
            runtime_counters = _increment_runtime_counter(runtime_counters, "cluster_reselection_count")
            runtime_counters = _increment_runtime_counter(
                runtime_counters,
                f"cluster_reselection_reason::{cluster_reselection_reason or 'default'}",
            )
            route = route_for_cluster(active_cluster)
            target_item_id = _target_item_for_cluster(state, active_cluster)
            module_id = int(choose_target_module(route, [target_item_id], state.get("item_beliefs", {})))
            question_kind = "opening" if turn_index <= 0 else "topic_open"

    anchor_text = _latest_anchor_for_item(state, target_item_id)
    timeframe_mode = "introduce"
    if question_kind in {"same_item_followup", "same_module_followup"}:
        timeframe_mode = "clarify"
    elif thread["active"]:
        timeframe_mode = "carry"

    probe_goal = "comparison" if question_kind == "contrastive_pivot" else (
        "impact" if urgency_mode == "direct_structured" and _item_presence(state, target_item_id) >= 0.35
        else "frequency" if _item_presence(state, target_item_id) >= 0.45
        else "exemplar"
    )
    style = {
        "impact": "functional_impact",
        "frequency": "clarify_frequency",
        "comparison": "gentle_probe",
        "exemplar": "gentle_probe",
    }[probe_goal]
    directness = "direct" if urgency_mode == "direct_structured" or route == "risk" else "indirect"
    priority = max(0.15, min(0.95, _cluster_score(state, active_cluster)))

    question_plan = QuestionPlan(
        active_cluster=active_cluster,
        route=route,  # type: ignore[arg-type]
        target_item_id=int(target_item_id),
        target_module_id=int(module_id),
        probe_goal=probe_goal,  # type: ignore[arg-type]
        question_mode=question_kind,
        urgency_mode=urgency_mode,  # type: ignore[arg-type]
        transition_reason=transition_reason,
        anchor_text=anchor_text,
        question_kind=question_kind,
        timeframe_mode=timeframe_mode,
        thread_turn_index=int(thread_turn_index),
        thread_module_id=int(module_id),
        thread_source_item_id=int(target_item_id if question_kind == "topic_open" else (thread["source_item_id"] or target_item_id)),
    )

    next_action = NextAction(
        target_item_id=int(target_item_id),
        route=route,  # type: ignore[arg-type]
        style=style,
        mode="wrapup" if urgency_mode == "direct_structured" else "normal",
        directness=directness,  # type: ignore[arg-type]
        priority=priority,
        rationale=transition_reason,
        question_kind=question_kind,
        thread_turn_index=int(question_plan.thread_turn_index),
        thread_module_id=int(question_plan.thread_module_id),
        thread_source_item_id=int(question_plan.thread_source_item_id),
        timeframe_mode=timeframe_mode,
        anchor_text=anchor_text,
    )

    route_decision = RouteDecision(
        turn=max(1, turn_index),
        chosen_node=route,  # type: ignore[arg-type]
        policy=policy,
        reason=transition_reason,
        target_items=[int(target_item_id)],
        expected_gain=max(0.0, min(3.0, _item_uncertainty(state, target_item_id) * 2.0 + _item_presence(state, target_item_id))),
    )

    next_thread = ConversationThreadState(
        active=(route != "risk"),
        route=route if route in {"cognitive", "somatic", "risk"} else "cognitive",  # type: ignore[arg-type]
        module_id=int(module_id),
        source_item_id=int(question_plan.thread_source_item_id),
        question_count=int(question_plan.thread_turn_index),
        denial_streak=0,
        last_question_kind=question_kind,
        timeframe_introduced=(timeframe_mode == "introduce"),
        anchor_text=anchor_text,
        exit_reason="",
    )

    turn_trace = dict(state.get("turn_trace", {}))
    turn_trace["navigation_agent"] = {
        "active_cluster": active_cluster,
        "route": route,
        "target_item_id": int(target_item_id),
        "target_module_id": int(module_id),
        "policy": policy,
        "question_kind": question_kind,
        "urgency_mode": urgency_mode,
        "transition_reason": transition_reason,
        "cluster_reselection_reason": cluster_reselection_reason,
        "stale_reply_detected": bool(stale_reply_detected),
        "anchor_text": anchor_text,
        "opening_signal_cluster": _opening_signal_cluster(state),
        "opening_signal_item_ids": _opening_signal_item_ids(state),
    }

    opening_followup_cluster = str(state.get("opening_followup_cluster", "") or "")
    opening_cognitive_anchor_preserved = bool(state.get("opening_cognitive_anchor_preserved", False))
    if int(turn_index) == 1 and _opening_signal_cluster(state):
        opening_followup_cluster = active_cluster
        if _opening_signal_cluster(state) == "cognitive_affective" and active_cluster == "cognitive_affective":
            opening_cognitive_anchor_preserved = True
            runtime_counters = _increment_runtime_counter(runtime_counters, "opening_cognitive_anchor_preserved_count")
        elif _opening_signal_cluster(state) == "cognitive_affective" and active_cluster == "somatic_vegetative":
            opening_cognitive_anchor_preserved = False
            runtime_counters = _increment_runtime_counter(
                runtime_counters,
                "opening_somatic_pivot_after_cognitive_signal_count",
            )

    return {
        "question_plan": question_plan,
        "next_action": next_action,
        "conversation_thread": next_thread,
        "route_history": [route_decision],
        "active_node": route,
        "next_node": route,
        "runtime_counters": runtime_counters,
        "opening_followup_cluster": opening_followup_cluster,
        "opening_cognitive_anchor_preserved": opening_cognitive_anchor_preserved,
        "route_debug": (
            f"Navigation agent: cluster={active_cluster}; route={route}; target_item={target_item_id}; "
            f"question_kind={question_kind}; urgency={urgency_mode}"
        ),
        "turn_trace": turn_trace,
    }
