from __future__ import annotations

import os
from typing import Dict, Optional, Protocol

from core.prompts import get_prompt
from core.state import AgentState, BDI_ITEM_NAMES, OutgoingState


class QuestionPolisher(Protocol):
    def invoke(self, messages): ...



def _has_detector_message(messages: list[dict]) -> bool:
    for msg in messages:
        if msg.get("role") == "user":
            return True
    return False



def _template_question(route: str, style: str, target_item_id: int) -> str:
    item_name = BDI_ITEM_NAMES.get(target_item_id, f"item {target_item_id}").lower()

    if style == "clarify_frequency":
        return f"How often has {item_name} felt noticeable for you this past week?"
    if style == "functional_impact":
        return f"How has {item_name} affected your routine or responsibilities lately?"

    if route == "risk":
        return "When things felt most heavy recently, what helped you stay safe in that moment?"
    return f"Could you share one recent example of how {item_name} has shown up for you?"



def _polish_question(
    question: str,
    state: AgentState,
    node_name: str,
    polisher: Optional[QuestionPolisher],
) -> str:
    if polisher is None:
        return question

    latest_text = ""
    turn_state = state.get("turn")
    if turn_state is not None:
        latest_text = str(getattr(turn_state, "latest_text_raw", "") or "")

    prompt = (
        "Rewrite this as one short, empathetic follow-up question under 20 words. "
        "Keep semantics and do not add diagnosis labels. Return only the question.\n\n"
        f"Node: {node_name}\n"
        f"Persona message: {latest_text or 'none'}\n"
        f"Draft question: {question}"
    )
    try:
        polished = str(polisher.invoke([("system", prompt)]).content or "").strip()
    except Exception:
        return question

    cleaned = " ".join(polished.split())
    if not cleaned:
        return question
    if not cleaned.endswith("?"):
        cleaned = cleaned.rstrip(".!") + "?"
    return cleaned



def question_generator(state: AgentState, polisher: Optional[QuestionPolisher] = None) -> Dict:
    messages = list(state.get("messages", []))
    turn_index = int(state.get("turn_index", 0))

    opening_question = get_prompt("opening_question").strip()
    if turn_index == 0 and not _has_detector_message(messages) and opening_question:
        question = " ".join(opening_question.split())
        route = "cognitive"
        style = "opening"
        target_item_id = 2
        rationale = "opening bootstrap"
    else:
        next_action = state.get("next_action")
        target_item_id = int(getattr(next_action, "target_item_id", 2) or 2)
        route = str(getattr(next_action, "route", "cognitive") or "cognitive")
        style = str(getattr(next_action, "style", "gentle_probe") or "gentle_probe")
        rationale = str(getattr(next_action, "rationale", "targeted follow-up") or "targeted follow-up")
        question = _template_question(route=route, style=style, target_item_id=target_item_id)

    use_polisher = os.getenv("QUESTION_LLM_POLISH", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    if use_polisher:
        if polisher is None:
            from core.llm import get_llm

            polisher = get_llm()
        question = _polish_question(question=question, state=state, node_name=route, polisher=polisher)

    turn_trace = dict(state.get("turn_trace", {}))
    specialist_trace = {
        "turn": max(1, turn_index),
        "node": route,
        "target_items": [target_item_id],
        "target_item_id": target_item_id,
        "target_item_name": BDI_ITEM_NAMES.get(target_item_id, f"Item {target_item_id}"),
        "probe_goal": style,
        "used_fallback": False,
        "question_preview": question[:120],
    }
    turn_trace["question_generator"] = specialist_trace
    turn_trace["specialist"] = specialist_trace

    debug = (
        f"Question generator: route={route}; target_item={target_item_id}; "
        f"style={style}; rationale={rationale}"
    )

    return {
        "outgoing": OutgoingState(detector_message=question),
        "messages": [{"role": "user", "content": question}],
        "specialist_debug": debug,
        "turn_trace": turn_trace,
    }
