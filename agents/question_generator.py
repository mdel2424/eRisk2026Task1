from __future__ import annotations

from typing import Any, Dict

from core.bdi_modules import MODULE_GOALS, MODULE_NAMES, MODULE_TO_ITEMS, choose_target_module
from core.llm import get_llm
from core.prompts import OPENING_MESSAGE_FIXED, get_prompt
from core.state import AgentState, BDI_ITEM_NAMES, OutgoingState


def _has_detector_message(messages: list[dict]) -> bool:
    for msg in messages:
        if msg.get("role") == "user":
            return True
    return False


def _next_action_value(state: AgentState, key: str, default: Any) -> Any:
    action = state.get("next_action")
    if isinstance(action, dict):
        return action.get(key, default)
    return getattr(action, key, default)


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


def _recent_context(state: AgentState, limit: int = 4, include_latest_persona: bool = True) -> str:
    turns = list(state.get("messages", [])[-limit:])
    if not include_latest_persona and turns and turns[-1].get("role") == "assistant":
        turns = turns[:-1]
    lines = []
    for msg in turns:
        role = "Detector" if msg.get("role") == "user" else "Persona"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)


def _probe_goal_from_style(style: str) -> str:
    mapping = {
        "clarify_frequency": "frequency",
        "functional_impact": "impact",
        "gentle_probe": "exemplar",
        "opening": "exemplar",
    }
    return mapping.get(style, "exemplar")


def _build_llm_question(
    state: AgentState,
    *,
    route: str,
    style: str,
    target_item_id: int,
) -> tuple[str, int, str]:
    prompt_template = get_prompt("specialist_question")
    if not prompt_template.strip():
        raise RuntimeError("Detector question generation failed: missing 'specialist_question' prompt template.")

    previous_question = _previous_detector_question(state)
    latest_message = _latest_persona_message(state)
    probe_goal = _probe_goal_from_style(style)
    module_id = choose_target_module(
        node_name=route,
        target_items=[target_item_id],
        item_beliefs=state.get("item_beliefs", {}),
    )
    module_name = MODULE_NAMES.get(module_id, "General Screening")
    module_goal = MODULE_GOALS.get(module_id, "Assess current depressive symptom expression.")
    module_items = MODULE_TO_ITEMS.get(module_id, [])
    target_item_name = BDI_ITEM_NAMES.get(target_item_id, f"Item {target_item_id}")

    prompt = prompt_template.format(
        node_name=route,
        latest_message=latest_message or "none",
        previous_question=previous_question or "none",
        recent_context=_recent_context(state, include_latest_persona=False) or "none",
        probe_goal=probe_goal,
        target_module_id=module_id,
        target_module_name=module_name,
        target_module_goal=module_goal,
        target_module_items=module_items,
        target_item_id=target_item_id,
        target_item_name=target_item_name,
    )

    llm = get_llm()
    raw = str(llm.invoke([("system", prompt)]).content or "").strip()
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1].strip()
    cleaned = " ".join(raw.split())
    if not cleaned:
        raise RuntimeError(
            f"Detector question generation failed for node '{route}' and item '{target_item_id}': empty model output."
        )
    if not cleaned.endswith("?"):
        cleaned = cleaned.rstrip(".!") + "?"
    return cleaned, module_id, probe_goal


def question_generator(state: AgentState) -> Dict:
    messages = list(state.get("messages", []))
    turn_index = int(state.get("turn_index", 0))

    if turn_index == 0 and not _has_detector_message(messages):
        question = OPENING_MESSAGE_FIXED
        route = "cognitive"
        style = "opening"
        target_item_id = 2
        rationale = "opening bootstrap"
        target_module_id = 2
        probe_goal = "exemplar"
    else:
        target_item_id = int(_next_action_value(state, "target_item_id", 2) or 2)
        route = str(_next_action_value(state, "route", "cognitive") or "cognitive")
        style = str(_next_action_value(state, "style", "gentle_probe") or "gentle_probe")
        rationale = str(_next_action_value(state, "rationale", "targeted follow-up") or "targeted follow-up")
        question, target_module_id, probe_goal = _build_llm_question(
            state,
            route=route,
            style=style,
            target_item_id=target_item_id,
        )

    turn_trace = dict(state.get("turn_trace", {}))
    specialist_trace = {
        "turn": max(1, turn_index),
        "node": route,
        "target_items": [target_item_id],
        "target_item_id": target_item_id,
        "target_item_name": BDI_ITEM_NAMES.get(target_item_id, f"Item {target_item_id}"),
        "target_module_id": target_module_id,
        "target_module_name": MODULE_NAMES.get(target_module_id, "General Screening"),
        "probe_goal": style,
        "probe_goal_kind": probe_goal,
        "used_fallback": False,
        "llm_generated": not (turn_index == 0 and not _has_detector_message(messages)),
        "question_preview": question[:120],
    }
    turn_trace["question_generator"] = specialist_trace
    turn_trace["specialist"] = specialist_trace

    debug = (
        f"Question generator: route={route}; target_item={target_item_id}; "
        f"style={style}; rationale={rationale}; llm_generated={specialist_trace['llm_generated']}"
    )

    return {
        "outgoing": OutgoingState(detector_message=question),
        "messages": [{"role": "user", "content": question}],
        "specialist_debug": debug,
        "turn_trace": turn_trace,
    }
