from __future__ import annotations

import random
from typing import Dict, List

from core.bdi_modules import (
    MODULE_GOALS,
    MODULE_NAMES,
    MODULE_TO_ITEMS,
    choose_target_module,
)
from core.llm import get_llm
from core.prompts import OPENING_MESSAGE_FIXED, get_fallback_questions, get_prompt
from core.state import AgentState, BDI_ITEM_NAMES, SPECIALIST_ITEM_MAP

rng = random.Random(17)
PROBE_GOALS = ("frequency", "duration", "impact", "exemplar")


def _latest_persona_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""


def _previous_detector_question(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _recent_context(state: AgentState, limit: int = 4) -> str:
    turns = state.get("messages", [])[-limit:]
    lines = []
    for msg in turns:
        role = "Detector" if msg.get("role") == "user" else "Persona"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


def _pick_fallback_question(node_name: str, previous_question: str) -> str:
    options = get_fallback_questions(node_name)
    if not options:
        return "Could you tell me a little more about that?"
    if previous_question:
        candidates = [q for q in options if q.strip().lower() != previous_question.strip().lower()]
        if candidates:
            return rng.choice(candidates)
    return rng.choice(options)


def _support_count_from_belief(value) -> int:
    if isinstance(value, dict):
        try:
            return int(value.get("support_count", 0))
        except (TypeError, ValueError):
            return 0
    try:
        return int(getattr(value, "support_count", 0))
    except (TypeError, ValueError):
        return 0


def _latest_route_target_items(state: AgentState, node_name: str) -> List[int]:
    route_history = list(state.get("route_history", []))
    if route_history:
        latest = route_history[-1]
        target_items = []
        if isinstance(latest, dict):
            target_items = latest.get("target_items", []) or []
        else:
            target_items = getattr(latest, "target_items", []) or []
        parsed: List[int] = []
        for item in target_items:
            try:
                parsed.append(int(item))
            except (TypeError, ValueError):
                continue
        if parsed:
            return parsed
    return list(SPECIALIST_ITEM_MAP.get(node_name, []))


def _pick_target_item(state: AgentState, target_items: List[int]) -> int:
    beliefs = state.get("item_beliefs", {})
    for item_id in target_items:
        support = _support_count_from_belief(beliefs.get(item_id))
        if support <= 0:
            return int(item_id)
    if target_items:
        return int(target_items[0])
    return 2


def _build_question(
    node_name: str,
    state: AgentState,
    latest_message: str,
    target_item_id: int,
    target_module_id: int,
) -> tuple[str, bool, str]:
    previous_question = _previous_detector_question(state)
    if not latest_message.strip() and not previous_question.strip():
        return OPENING_MESSAGE_FIXED, False, "opening"

    turn_index = int(state.get("turn_index", 0))
    probe_goal = PROBE_GOALS[turn_index % len(PROBE_GOALS)]
    module_name = MODULE_NAMES.get(target_module_id, "General Screening")
    module_goal = MODULE_GOALS.get(target_module_id, "Assess current depressive symptom expression.")
    module_items = MODULE_TO_ITEMS.get(target_module_id, [])
    target_item_name = BDI_ITEM_NAMES.get(target_item_id, f"Item {target_item_id}")
    prompt_template = get_prompt("specialist_question")
    prompt = prompt_template.format(
        node_name=node_name,
        latest_message=latest_message or "none",
        previous_question=previous_question or "none",
        recent_context=_recent_context(state) or "none",
        probe_goal=probe_goal,
        target_module_id=target_module_id,
        target_module_name=module_name,
        target_module_goal=module_goal,
        target_module_items=module_items,
        target_item_id=target_item_id,
        target_item_name=target_item_name,
    )
    fallback = _pick_fallback_question(node_name, previous_question)
    llm = get_llm()
    text = llm.invoke([("system", prompt)]).content.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    if not text:
        return fallback, True, probe_goal
    if previous_question and text.strip().lower() == previous_question.strip().lower():
        return fallback, True, probe_goal
    cleaned = " ".join(text.split())
    if not cleaned.endswith("?"):
        cleaned = cleaned.rstrip(".!") + "?"
    return cleaned, False, probe_goal


def _specialist_node(state: AgentState, node_name: str) -> Dict:
    latest = _latest_persona_message(state)
    target_items = _latest_route_target_items(state, node_name)
    try:
        target_item_id = _pick_target_item(state, target_items)
        target_module_id = choose_target_module(
            node_name=node_name,
            target_items=target_items,
            item_beliefs=state.get("item_beliefs", {}),
        )
        question, used_fallback, probe_goal = _build_question(
            node_name=node_name,
            state=state,
            latest_message=latest,
            target_item_id=target_item_id,
            target_module_id=target_module_id,
        )
    except Exception as exc:
        raise RuntimeError(f"Detector specialist question generation failed for node '{node_name}'.") from exc
    module_items = MODULE_TO_ITEMS.get(target_module_id, [])
    module_name = MODULE_NAMES.get(target_module_id, "General Screening")
    target_item_name = BDI_ITEM_NAMES.get(target_item_id, f"Item {target_item_id}")
    debug = (
        f"{node_name.title()} specialist: question generated; target_items={target_items}; "
        f"target_module={target_module_id}:{module_name}; target_item={target_item_id}:{target_item_name}"
    )
    turn = int(state.get("turn_index", 0)) + 1
    trace = dict(state.get("turn_trace", {}))
    trace["specialist"] = {
        "turn": turn,
        "node": node_name,
        "target_items": target_items,
        "target_item_id": target_item_id,
        "target_item_name": target_item_name,
        "target_module_id": target_module_id,
        "target_module_name": module_name,
        "target_module_items": module_items,
        "probe_goal": probe_goal,
        "used_fallback": used_fallback,
        "question_preview": question[:120],
    }
    return {
        "messages": [{"role": "user", "content": question}],
        "specialist_debug": debug,
        "active_node": node_name,
        "turn_trace": trace,
    }


def somatic_specialist(state: AgentState):
    return _specialist_node(state, "somatic")


def cognitive_specialist(state: AgentState):
    return _specialist_node(state, "cognitive")


def risk_specialist(state: AgentState):
    return _specialist_node(state, "risk")
