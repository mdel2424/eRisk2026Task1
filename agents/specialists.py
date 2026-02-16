from __future__ import annotations

import json
import random
from typing import Dict, List, Tuple

from core.llm import get_llm
from core.prompts import (
    FALLBACK_QUESTIONS,
    SPECIALIST_QUESTION_PROMPT,
    SPECIALIST_SIGNAL_FALLBACK_PROMPT,
)
from core.state import AgentState

# keyword, symptom label, score delta
SOMATIC_CUES: List[Tuple[str, str, float]] = [
    ("sleep", "Changes in Sleeping Pattern", 0.08),
    ("rest", "Changes in Sleeping Pattern", 0.06),
    ("tired", "Tiredness or Fatigue", 0.08),
    ("fatigue", "Tiredness or Fatigue", 0.08),
    ("energy", "Loss of Energy", 0.07),
    ("appetite", "Changes in Appetite", 0.07),
]

COGNITIVE_CUES: List[Tuple[str, str, float]] = [
    ("worthless", "Worthlessness", 0.1),
    ("guilty", "Guilty Feelings", 0.08),
    ("hopeless", "Pessimism", 0.1),
    ("failure", "Past Failure", 0.08),
    ("hate myself", "Self-Dislike", 0.1),
    ("can't focus", "Concentration Difficulty", 0.07),
]

RISK_CUES: List[Tuple[str, str, float]] = [
    ("suicide", "Suicidal Thoughts or Wishes", 0.2),
    ("kill myself", "Suicidal Thoughts or Wishes", 0.25),
    ("end it", "Suicidal Thoughts or Wishes", 0.2),
    ("better off dead", "Suicidal Thoughts or Wishes", 0.25),
    ("don't want to live", "Suicidal Thoughts or Wishes", 0.25),
    ("hurt myself", "Suicidal Thoughts or Wishes", 0.25),
]

ALL_CUE_GROUPS: Dict[str, List[Tuple[str, str, float]]] = {
    "somatic": SOMATIC_CUES,
    "cognitive": COGNITIVE_CUES,
    "risk": RISK_CUES,
}

FALLBACK_QUESTION_OPTIONS: Dict[str, List[str]] = {
    "somatic": [
        "How has your sleep been this week?",
        "How has your energy changed across the day?",
        "Have meals or appetite felt different lately?",
    ],
    "cognitive": [
        "What thought has been the loudest in your head lately?",
        "When things feel heavy, what do you tell yourself?",
        "What feels hardest to believe about tomorrow right now?",
    ],
    "risk": [
        "When things feel very heavy, what helps you stay safe in that moment?",
        "Who or what helps you get through your hardest moments?",
        "What do you usually do first when thoughts become overwhelming?",
    ],
}

rng = random.Random(17)


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


def _pick_fallback_question(node_name: str, previous_question: str) -> str:
    options = FALLBACK_QUESTION_OPTIONS[node_name]
    if previous_question:
        candidates = [q for q in options if q.strip().lower() != previous_question.strip().lower()]
        if candidates:
            return rng.choice(candidates)
    return rng.choice(options) if options else FALLBACK_QUESTIONS[node_name]


def _build_question(node_name: str, state: AgentState, latest_message: str) -> str:
    previous_question = _previous_detector_question(state)
    prompt = SPECIALIST_QUESTION_PROMPT.format(
        node_name=node_name,
        latest_message=latest_message,
        previous_question=previous_question or "none",
        recent_context=_recent_context(state) or "none",
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
        if not cleaned.endswith("?"):
            cleaned = cleaned.rstrip(".!") + "?"
        return cleaned
    except Exception:
        return fallback


def _lexical_signal(text: str) -> Tuple[float, List[str], bool, List[str]]:
    lowered = text.lower()

    delta = 0.0
    hits: List[str] = []
    matched_keywords: List[str] = []
    risk_flag = False
    for group_name, cues in ALL_CUE_GROUPS.items():
        for keyword, symptom_name, weight in cues:
            if keyword in lowered:
                delta += weight
                hits.append(symptom_name)
                matched_keywords.append(keyword)
                if group_name == "risk":
                    risk_flag = True

    unique_hits: List[str] = []
    for hit in hits:
        if hit not in unique_hits:
            unique_hits.append(hit)

    return min(0.18, delta), unique_hits, risk_flag, matched_keywords


def _llm_fallback_signal(state: AgentState, node_name: str, latest_message: str) -> Tuple[float, List[str], bool, str]:
    default = (0.0, [], False, "no strong evidence")
    prompt = SPECIALIST_SIGNAL_FALLBACK_PROMPT.format(
        node_name=node_name,
        recent_context=_recent_context(state) or "none",
        latest_message=latest_message or "none",
    )

    try:
        llm = get_llm()
        raw = llm.invoke([("system", prompt)]).content
        parsed = _parse_json_object(str(raw))

        hits_raw = parsed.get("symptom_hits", [])
        hits: List[str] = []
        if isinstance(hits_raw, list):
            for value in hits_raw:
                label = str(value).strip()
                if label and label not in hits:
                    hits.append(label)
                if len(hits) >= 3:
                    break

        try:
            score_delta = float(parsed.get("score_delta", 0.0))
        except (TypeError, ValueError):
            score_delta = 0.0
        score_delta = max(0.0, min(0.18, score_delta))

        risk_flag = bool(parsed.get("risk_flag", False))
        reason = str(parsed.get("reason", "")).strip() or "weak implicit signal"
        return score_delta, hits, risk_flag, reason
    except Exception:
        return default


def _score_from_text(state: AgentState, node_name: str, text: str) -> Tuple[float, List[str], bool, str, str]:
    lexical_delta, lexical_hits, lexical_risk, matched_keywords = _lexical_signal(text)
    if lexical_hits:
        reason = ", ".join(matched_keywords[:3]) if matched_keywords else "matched lexical cues"
        return lexical_delta, lexical_hits, lexical_risk, "lexical", reason

    llm_delta, llm_hits, llm_risk, llm_reason = _llm_fallback_signal(state, node_name, text)
    return llm_delta, llm_hits, llm_risk, "llm-fallback", llm_reason


def _specialist_node(state: AgentState, node_name: str) -> Dict:
    latest = _latest_persona_message(state)
    question = _build_question(node_name, state, latest)
    delta, hits, risk_found, source, reason = _score_from_text(state, node_name, latest)

    updated_score = min(1.0, float(state.get("depression_score", 0.0)) + delta)
    updated_risk = bool(state.get("risk_flag", False)) or risk_found
    debug = (
        f"{node_name.title()} specialist: source={source}; delta={delta:.2f}; "
        f"hits={hits if hits else ['none']}; risk={updated_risk}; reason={reason}"
    )

    return {
        "messages": [{"role": "user", "content": question}],
        "depression_score": updated_score,
        "risk_flag": updated_risk,
        "symptom_hits": hits,
        "specialist_debug": debug,
    }


def somatic_specialist(state: AgentState):
    return _specialist_node(state, "somatic")


def cognitive_specialist(state: AgentState):
    return _specialist_node(state, "cognitive")


def risk_specialist(state: AgentState):
    return _specialist_node(state, "risk")
