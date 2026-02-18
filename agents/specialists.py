from __future__ import annotations

import random
from typing import Dict, List

from core.llm import get_llm
from core.prompts import get_fallback_questions, get_prompt
from core.state import AgentState, SPECIALIST_ITEM_MAP

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


def _build_question(node_name: str, state: AgentState, latest_message: str) -> str:
    previous_question = _previous_detector_question(state)
    if not latest_message.strip() and not previous_question.strip():
        opening = get_prompt("opening_question").strip()
        if opening:
            return " ".join(opening.split())

    turn_index = int(state.get("turn_index", 0))
    probe_goal = PROBE_GOALS[turn_index % len(PROBE_GOALS)]
    prompt_template = get_prompt("specialist_question")
    prompt = prompt_template.format(
        node_name=node_name,
        latest_message=latest_message or "none",
        previous_question=previous_question or "none",
        recent_context=_recent_context(state) or "none",
        probe_goal=probe_goal,
    )
    fallback = _pick_fallback_question(node_name, previous_question)
    try:
        llm = get_llm()
        text = llm.invoke([("system", prompt)]).content.strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()
        if not text:
            return fallback
        if previous_question and text.strip().lower() == previous_question.strip().lower():
            return fallback
        cleaned = " ".join(text.split())
        if "past two weeks" not in cleaned.lower():
            cleaned = f"In the past two weeks, {cleaned[0].lower() + cleaned[1:]}" if cleaned else cleaned
        if not cleaned.endswith("?"):
            cleaned = cleaned.rstrip(".!") + "?"
        return cleaned
    except Exception:
        return fallback


def _specialist_node(state: AgentState, node_name: str) -> Dict:
    latest = _latest_persona_message(state)
    question = _build_question(node_name, state, latest)
    target_items = SPECIALIST_ITEM_MAP.get(node_name, [])
    debug = f"{node_name.title()} specialist: question generated; target_items={target_items}"
    return {
        "messages": [{"role": "user", "content": question}],
        "specialist_debug": debug,
        "active_node": node_name,
    }


def somatic_specialist(state: AgentState):
    return _specialist_node(state, "somatic")


def cognitive_specialist(state: AgentState):
    return _specialist_node(state, "cognitive")


def risk_specialist(state: AgentState):
    return _specialist_node(state, "risk")
