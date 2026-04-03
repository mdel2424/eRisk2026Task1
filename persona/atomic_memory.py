from __future__ import annotations

from typing import Dict, Tuple

from core.state import PersonaAtomicFact, PersonaMemoryState, symptom_name_from_item
from persona.sim_templates import CONTEXT_TAG_ANCHORS, NEUTRAL_CONTEXT_TAG_ANCHORS


POSITIVE_ITEM_LEXEMES: Dict[int, Tuple[str, ...]] = {
    1: ("sad", "down", "low"),
    2: ("worst", "hopeless", "future"),
    3: ("failure", "let people down"),
    4: ("less pleasure", "less enjoyable", "interest"),
    5: ("guilty", "fault"),
    6: ("punished", "deserve"),
    7: ("don't like myself", "lost confidence", "hate who i've been"),
    8: ("hard on myself", "should be doing better", "second-guess"),
    9: ("rather not wake", "not be here", "disappear"),
    14: ("burden", "don't matter", "don't measure up", "failure"),
    15: ("no energy", "wiped out", "hard to get going"),
    16: ("can't sleep", "sleep is a mess", "wake up"),
    18: ("appetite", "eating", "up and down"),
    19: ("can't focus", "rereading", "hard to concentrate"),
    20: ("tired", "fatigue", "exhausted"),
    21: ("interest in sex", "that side of things"),
}


def build_persona_memory(
    *,
    bdi_scores: Dict[int, int],
    context_tag: str,
    style_tag: str,
) -> PersonaMemoryState:
    context_options = list(CONTEXT_TAG_ANCHORS.get(str(context_tag or ""), CONTEXT_TAG_ANCHORS.get("routine_stable", [])))
    neutral_context_options = list(
        NEUTRAL_CONTEXT_TAG_ANCHORS.get(str(context_tag or ""), NEUTRAL_CONTEXT_TAG_ANCHORS.get("routine_stable", []))
    )
    facts = []
    for item_id in range(1, 22):
        severity = max(0, min(3, int(bdi_scores.get(item_id, 0) or 0)))
        if severity <= 0:
            continue
        context_anchor_options = context_options if severity >= 1 else neutral_context_options
        context_anchor = context_anchor_options[(int(item_id) - 1) % len(context_anchor_options)] if context_anchor_options else "day to day feels different lately"
        facts.append(
            PersonaAtomicFact(
                item_id=item_id,
                symptom_name=symptom_name_from_item(item_id),
                severity=severity,
                duration_phrase="lately",
                context_anchor=str(context_anchor or ""),
                disclosure_style=str(style_tag or ""),
                polarity="positive",
            )
        )
    return PersonaMemoryState(facts=facts, disclosed_item_ids=[], context_ledger=[], verification_failures=0)


def select_atomic_fact(memory_state: PersonaMemoryState, target_item_id: int) -> PersonaAtomicFact | None:
    for fact in memory_state.facts:
        if int(fact.item_id) == int(target_item_id):
            return fact
    ranked = sorted(memory_state.facts, key=lambda fact: (-int(fact.severity), int(fact.item_id)))
    return ranked[0] if ranked else None


def record_disclosure(memory_state: PersonaMemoryState, item_id: int, reply_text: str) -> PersonaMemoryState:
    if int(item_id) not in memory_state.disclosed_item_ids:
        memory_state.disclosed_item_ids.append(int(item_id))
    cleaned = " ".join(str(reply_text or "").split())
    if cleaned:
        memory_state.context_ledger.append(cleaned[:120])
        if len(memory_state.context_ledger) > 8:
            memory_state.context_ledger = memory_state.context_ledger[-8:]
    return memory_state


def verify_reply_against_memory(
    memory_state: PersonaMemoryState,
    *,
    target_item_id: int,
    target_score: int,
    reply_text: str,
) -> tuple[bool, str]:
    lowered = str(reply_text or "").lower()
    if target_score <= 0:
        for lexeme in POSITIVE_ITEM_LEXEMES.get(int(target_item_id), ()):
            if lexeme in lowered:
                return False, f"unexpected_positive_lexeme:{lexeme}"
    return True, ""
