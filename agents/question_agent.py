from __future__ import annotations

import re
from typing import Any, Dict

from core.bdi_modules import MODULE_GOALS, MODULE_NAMES, MODULE_TO_ITEMS, choose_target_module
from core.llm import get_llm
from core.prompts import OPENING_MESSAGE_FIXED, get_fallback_questions, get_prompt
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


def _sanitize_anchor_text(value: str, limit_words: int = 12) -> str:
    cleaned = " ".join(str(value or "").strip().split())
    if not cleaned:
        return ""
    words = cleaned.split()
    if len(words) > limit_words:
        cleaned = " ".join(words[:limit_words]).rstrip(" ,;:.")
    return cleaned


def _strip_timeframe_lead(text: str) -> str:
    cleaned = str(text or "").strip()
    patterns = (
        r"^(in|over|during)\s+the\s+(last|past)\s+two\s+weeks[,:\-\s]+",
        r"^(in|over|during)\s+the\s+last\s+couple\s+of\s+weeks[,:\-\s]+",
    )
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def _strip_stock_followup_lead(text: str) -> str:
    cleaned = str(text or "").strip()
    patterns = (
        r"^you\s+mentioned\s+.+?[—\-:]\s*",
        r"^you\s+mentioned\s+.+?,\s*(?=(how|what|when|can|could|does|do|is|are|has|have)\b)",
    )
    for pattern in patterns:
        stripped = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        if stripped != cleaned:
            cleaned = stripped.strip()
            break
    return cleaned


def _sanitize_question_text(
    text: str,
    *,
    question_kind: str,
) -> tuple[str, bool]:
    cleaned = " ".join(str(text or "").split()).strip()
    blocked_repeated_timeframe = False
    if question_kind in {"same_item_followup", "same_module_followup", "contrastive_pivot"}:
        stripped = _strip_timeframe_lead(cleaned)
        blocked_repeated_timeframe = stripped != cleaned
        cleaned = stripped
        cleaned = _strip_stock_followup_lead(cleaned)
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1].strip()
    if cleaned and cleaned[0].isalpha():
        cleaned = cleaned[0].upper() + cleaned[1:]
    if cleaned and not cleaned.endswith("?"):
        cleaned = cleaned.rstrip(".!") + "?"
    return cleaned, blocked_repeated_timeframe


def _fallback_question(
    *,
    route: str,
    target_item_id: int,
    turn_index: int,
    question_kind: str,
) -> str:
    options = get_fallback_questions(route, question_kind=question_kind)
    if not options:
        return "Could you tell me a little more about how that has been affecting you lately?"
    index = (max(0, int(turn_index)) + max(1, int(target_item_id))) % len(options)
    return str(options[index]).strip()


def _build_llm_question(
    state: AgentState,
    *,
    route: str,
    style: str,
    target_item_id: int,
    question_kind: str,
    timeframe_mode: str,
    anchor_text: str,
    thread_turn_index: int,
) -> tuple[str, int, str, bool, str, bool]:
    prompt_template = get_prompt("question_agent_prompt")
    if not prompt_template.strip():
        raise RuntimeError("Detector question generation failed: missing 'question_agent_prompt' prompt template.")

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
        question_kind=question_kind,
        timeframe_mode=timeframe_mode,
        anchor_text=anchor_text or "none",
        thread_turn_index=int(thread_turn_index),
    )

    llm = get_llm()
    raw = str(llm.invoke([("system", prompt)]).content or "").strip()
    cleaned, blocked_repeated_timeframe = _sanitize_question_text(raw, question_kind=question_kind)
    if not cleaned:
        fallback = _fallback_question(
            route=route,
            target_item_id=target_item_id,
            turn_index=int(state.get("turn_index", 0)),
            question_kind=question_kind,
        )
        fallback, fallback_blocked_timeframe = _sanitize_question_text(fallback, question_kind=question_kind)
        return fallback, module_id, probe_goal, True, "empty_model_output", fallback_blocked_timeframe
    return cleaned, module_id, probe_goal, False, "", blocked_repeated_timeframe


def question_agent(state: AgentState) -> Dict:
    messages = list(state.get("messages", []))
    turn_index = int(state.get("turn_index", 0))
    runtime_counters = dict(state.get("runtime_counters", {}))

    if turn_index == 0 and not _has_detector_message(messages):
        question = OPENING_MESSAGE_FIXED
        route = "cognitive"
        style = "opening"
        question_kind = "opening"
        timeframe_mode = "introduce"
        thread_turn_index = 0
        anchor_text = ""
        target_item_id = 2
        rationale = "opening bootstrap"
        target_module_id = 2
        probe_goal = "exemplar"
        used_fallback = False
        fallback_reason = ""
        repeated_timeframe_lead_blocked = False
    else:
        target_item_id = int(_next_action_value(state, "target_item_id", 2) or 2)
        route = str(_next_action_value(state, "route", "cognitive") or "cognitive")
        style = str(_next_action_value(state, "style", "gentle_probe") or "gentle_probe")
        question_kind = str(_next_action_value(state, "question_kind", "topic_open") or "topic_open")
        timeframe_mode = str(_next_action_value(state, "timeframe_mode", "introduce") or "introduce")
        thread_turn_index = int(_next_action_value(state, "thread_turn_index", 0) or 0)
        anchor_text = _sanitize_anchor_text(str(_next_action_value(state, "anchor_text", "") or ""))
        rationale = str(_next_action_value(state, "rationale", "targeted follow-up") or "targeted follow-up")
        question, target_module_id, probe_goal, used_fallback, fallback_reason, repeated_timeframe_lead_blocked = _build_llm_question(
            state,
            route=route,
            style=style,
            target_item_id=target_item_id,
            question_kind=question_kind,
            timeframe_mode=timeframe_mode,
            anchor_text=anchor_text,
            thread_turn_index=thread_turn_index,
        )

    question_starts_with_timeframe = bool(
        re.match(
            r"^(in|over|during)\s+the\s+(last|past)\s+two\s+weeks",
            str(question or "").strip(),
            flags=re.IGNORECASE,
        )
    )

    turn_trace = dict(state.get("turn_trace", {}))
    question_trace = {
        "turn": max(1, turn_index),
        "route": route,
        "target_items": [target_item_id],
        "target_item_id": target_item_id,
        "target_item_name": BDI_ITEM_NAMES.get(target_item_id, f"Item {target_item_id}"),
        "target_module_id": target_module_id,
        "target_module_name": MODULE_NAMES.get(target_module_id, "General Screening"),
        "probe_goal": style,
        "probe_goal_kind": probe_goal,
        "conversation_thread_active": bool(question_kind in {"topic_open", "same_item_followup", "same_module_followup", "contrastive_pivot"}),
        "question_kind": question_kind,
        "thread_turn_index": int(thread_turn_index),
        "timeframe_mode": timeframe_mode,
        "anchor_text": anchor_text,
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        "repeated_timeframe_lead_blocked": bool(repeated_timeframe_lead_blocked),
        "question_starts_with_timeframe": bool(question_starts_with_timeframe),
        "llm_generated": not (turn_index == 0 and not _has_detector_message(messages)),
        "question_preview": question[:120],
    }
    if question_kind == "topic_open":
        runtime_counters["thread_start_count"] = int(runtime_counters.get("thread_start_count", 0)) + 1
        runtime_counters["thread_question_total"] = int(runtime_counters.get("thread_question_total", 0)) + 1
    elif question_kind in {"same_item_followup", "same_module_followup", "contrastive_pivot"}:
        runtime_counters["threaded_followup_count"] = int(runtime_counters.get("threaded_followup_count", 0)) + 1
        runtime_counters["thread_question_total"] = int(runtime_counters.get("thread_question_total", 0)) + 1
    if timeframe_mode == "introduce":
        runtime_counters["timeframe_intro_count"] = int(runtime_counters.get("timeframe_intro_count", 0)) + 1
    if question_starts_with_timeframe and question_kind in {"same_item_followup", "same_module_followup", "contrastive_pivot"}:
        runtime_counters["repeated_timeframe_lead_count"] = int(runtime_counters.get("repeated_timeframe_lead_count", 0)) + 1
    if question_kind == "contrastive_pivot":
        runtime_counters["contrastive_pivot_count"] = int(runtime_counters.get("contrastive_pivot_count", 0)) + 1
    plan = state.get("question_plan")
    if plan is not None:
        question_trace["active_cluster"] = getattr(plan, "active_cluster", "")
        question_trace["transition_reason"] = getattr(plan, "transition_reason", "")
        question_trace["urgency_mode"] = getattr(plan, "urgency_mode", "")
    turn_trace["question_agent"] = question_trace

    debug = (
        f"Question agent: route={route}; target_item={target_item_id}; "
        f"style={style}; question_kind={question_kind}; rationale={rationale}; "
        f"llm_generated={question_trace['llm_generated']}"
    )

    return {
        "outgoing": OutgoingState(detector_message=question),
        "messages": [{"role": "user", "content": question}],
        "runtime_counters": runtime_counters,
        "question_debug": debug,
        "turn_trace": turn_trace,
    }
