from __future__ import annotations

import json
from typing import Dict, List

from core.llm import get_llm
from core.prompts import SUPERVISOR_ROUTE_FALLBACK_PROMPT
from core.state import AgentState

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


def _parse_json_object(raw_text: str) -> Dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def supervisor_router(state: AgentState):
    latest_message = _latest_persona_message(state)
    text = latest_message.lower()

    risk_hits = [cue for cue in ROUTE_CUES["risk"] if cue in text]
    if risk_hits:
        return {
            "next_node": "risk",
            "route_debug": f"Supervisor -> risk (lexical: {', '.join(risk_hits[:3])})",
        }

    somatic_hits = [cue for cue in ROUTE_CUES["somatic"] if cue in text]
    if somatic_hits:
        return {
            "next_node": "somatic",
            "route_debug": f"Supervisor -> somatic (lexical: {', '.join(somatic_hits[:3])})",
        }

    cognitive_hits = [cue for cue in ROUTE_CUES["cognitive"] if cue in text]
    if cognitive_hits:
        return {
            "next_node": "cognitive",
            "route_debug": f"Supervisor -> cognitive (lexical: {', '.join(cognitive_hits[:3])})",
        }

    if text:
        fallback_route = "cognitive"
        fallback_reason = "unclear signal"
        try:
            llm = get_llm()
            raw = llm.invoke(
                [("system", SUPERVISOR_ROUTE_FALLBACK_PROMPT.format(latest_message=latest_message))]
            ).content
            parsed = _parse_json_object(str(raw))
            route = str(parsed.get("route", "")).strip().lower()
            if route in {"risk", "somatic", "cognitive"}:
                fallback_route = route
            reason = str(parsed.get("reason", "")).strip()
            if reason:
                fallback_reason = reason
        except Exception:
            fallback_reason = "fallback unavailable"
        return {
            "next_node": fallback_route,
            "route_debug": f"Supervisor -> {fallback_route} (llm-fallback: {fallback_reason})",
        }
    return {
        "next_node": "cognitive",
        "route_debug": "Supervisor -> cognitive (opening turn)",
    }
