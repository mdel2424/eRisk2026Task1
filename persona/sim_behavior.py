from __future__ import annotations

import re
from typing import Dict, List

from core.state import BDI_ITEM_NAMES
from persona.sim_templates import (
    CONTEXT_ANCHORS,
    DIRECT_DIAGNOSIS_TOKENS,
    HEDGE_PHRASES,
    ITEM_CONTEXT_HINTS,
    ITEM_SENTENCE_BANK,
    NORMALIZATION_PHRASES,
    QUESTION_KEYWORDS_TO_ITEMS,
    RISK_PROTECTIVE_FACTORS,
    RISK_RESPONSE_BANK,
    SIM_TEMPLATE_BANKS,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def validate_template_disjointness() -> Dict[str, object]:
    details: Dict[str, Dict[str, int]] = {}
    total_overlap = 0
    split_names = list(SIM_TEMPLATE_BANKS.keys())

    for category in ("openers", "bridges", "deflectors"):
        cat_details: Dict[str, int] = {}
        for idx in range(len(split_names)):
            left = split_names[idx]
            left_set = {_normalize(v) for v in SIM_TEMPLATE_BANKS[left][category]}
            for jdx in range(idx + 1, len(split_names)):
                right = split_names[jdx]
                right_set = {_normalize(v) for v in SIM_TEMPLATE_BANKS[right][category]}
                overlap = left_set.intersection(right_set)
                key = f"{left}__{right}"
                cat_details[key] = len(overlap)
                total_overlap += len(overlap)
        details[category] = cat_details

    return {
        "total_overlap": total_overlap,
        "details": details,
        "strict_pass": total_overlap == 0,
    }


def _latest_question(history: List[dict]) -> str:
    for msg in reversed(history):
        if msg.get("role") == "user":
            return str(msg.get("content", "")).strip().lower()
    return ""


def pick_target_item(question: str, bdi_scores: Dict[int, int], family: str, rng) -> int:
    for keywords, item_id in QUESTION_KEYWORDS_TO_ITEMS:
        if any(keyword in question for keyword in keywords):
            return item_id

    non_zero = [item_id for item_id, score in bdi_scores.items() if int(score) > 0]
    if non_zero:
        return int(rng.choice(non_zero))

    family_defaults = {
        "control_stressed": [16, 19, 15],
        "control_neutral": [4, 15],
        "somatic_evasive": [16, 20, 15],
        "cognitive_ruminative": [2, 3, 8],
        "mixed_moderate": [4, 15, 19],
        "functional_masked": [4, 15, 19],
        "risk_leaning": [9, 2, 14],
    }
    choices = family_defaults.get(family, [15, 4, 19])
    return int(rng.choice(choices))


def _item_sentence(item_id: int, score: int) -> str:
    clipped = max(0, min(3, int(score)))
    if clipped == 0:
        name = BDI_ITEM_NAMES.get(item_id, "general stress").lower()
        return f"that area feels mostly manageable, with occasional stress around {name}"

    bank = ITEM_SENTENCE_BANK.get(item_id)
    if bank:
        return bank.get(clipped, bank[max(bank.keys())])

    symptom = BDI_ITEM_NAMES.get(item_id, f"item {item_id}").lower()
    return f"{symptom} has been more noticeable lately"


def response_style_flags(text: str) -> Dict[str, bool]:
    lowered = _normalize(text)
    hedged = any(phrase.lower() in lowered for phrase in HEDGE_PHRASES)
    deflect = any(token in lowered for token in ("labels", "diagnosis", "diagnostic", "not naming", "not really labels"))
    context = any(anchor.lower() in lowered for anchor in CONTEXT_ANCHORS)
    context = context or any(
        token in lowered
        for token in ("work", "kids", "family", "messages", "meeting", "phone", "evening", "morning", "routine")
    )
    return {"hedged": hedged, "deflect": deflect, "context_anchor": context}


def _intent(
    question: str,
    evasiveness: float,
    contradiction_rate: float,
    direct_answer_rate: float,
    rng,
) -> str:
    direct_label_prompt = any(token in question for token in DIRECT_DIAGNOSIS_TOKENS)
    if direct_label_prompt and rng.random() < max(0.18, min(0.55, evasiveness * 0.55)):
        return "deflect"
    if rng.random() > max(0.45, min(0.95, direct_answer_rate)) and rng.random() < max(0.22, min(0.5, evasiveness + 0.1)):
        return "partial"
    if contradiction_rate > 0.0 and rng.random() < min(0.2, contradiction_rate * 0.8):
        return "partial"
    return "disclose"


def _safe_join(chunks: List[str], limit_words: int = 38) -> str:
    normalized_chunks: List[str] = []
    for chunk in chunks:
        if not chunk or not chunk.strip():
            continue
        piece = re.sub(r"\s+", " ", chunk.strip())
        if piece and piece[0].isalpha():
            piece = piece[0].upper() + piece[1:]
        if piece and piece[-1] not in {".", "?", "!"}:
            piece += "."
        normalized_chunks.append(piece)
    text = " ".join(normalized_chunks)
    words = text.split()
    if len(words) > limit_words:
        text = " ".join(words[:limit_words]).rstrip(" ,;")
        if text and text[-1] not in {".", "?", "!"}:
            text += "."
    if text and text[-1] not in {".", "?", "!"}:
        text += "."
    return re.sub(r"\s+", " ", text).strip()


def _maybe_context_anchor(item_id: int, context_anchor_rate: float, rng) -> str:
    if rng.random() >= context_anchor_rate:
        return ""
    hints = ITEM_CONTEXT_HINTS.get(item_id, CONTEXT_ANCHORS)
    return rng.choice(hints)


def _maybe_tail(hedge_rate: float, normalization_rate: float, rng) -> str:
    parts: List[str] = []
    if rng.random() < hedge_rate:
        hedge = rng.choice(HEDGE_PHRASES)
        if hedge.lower() == "i don't know":
            parts.append("I don't know, that is how it has felt")
        elif hedge.lower() == "maybe":
            parts.append("maybe that is why everything feels heavier")
        else:
            parts.append(f"{hedge.lower()}, that seems to be the pattern")
    if rng.random() < normalization_rate:
        parts.append(rng.choice(NORMALIZATION_PHRASES))
    return _safe_join(parts, limit_words=24) if parts else ""


def _risk_tier_reply(risk_score: int, rng) -> str:
    clipped = max(0, min(3, int(risk_score)))
    line = rng.choice(RISK_RESPONSE_BANK.get(clipped, RISK_RESPONSE_BANK[0]))
    if clipped >= 1 and rng.random() < 0.55:
        return _safe_join([line, rng.choice(RISK_PROTECTIVE_FACTORS)], limit_words=34)
    return _safe_join([line], limit_words=26)


def build_deterministic_reply(
    *,
    family: str,
    split: str,
    bdi_scores: Dict[int, int],
    behavior_params: Dict[str, float | str],
    history: List[dict],
    evasive: bool,
    rng,
) -> str:
    split_key = split if split in SIM_TEMPLATE_BANKS else "test"
    bank = SIM_TEMPLATE_BANKS[split_key]

    question = _latest_question(history)
    target_item = pick_target_item(question, bdi_scores, family, rng)
    target_score = int(bdi_scores.get(target_item, 0))

    evasiveness = float(behavior_params.get("evasiveness", 0.45))
    contradiction_rate = float(behavior_params.get("contradiction", 0.08))
    hedge_rate = float(behavior_params.get("hedge_rate", 0.65))
    normalization_rate = float(behavior_params.get("normalization_rate", 0.45))
    context_anchor_rate = float(behavior_params.get("context_anchor_rate", 0.55))
    direct_answer_rate = float(behavior_params.get("direct_answer_rate", 0.78))
    intent = _intent(question, evasiveness if evasive else 0.1, contradiction_rate, direct_answer_rate, rng)

    question_turns = sum(1 for msg in history if msg.get("role") == "user")
    opener_pool = bank["openers"][::2] if question_turns % 2 == 0 else bank["openers"][1::2] or bank["openers"]
    opener = rng.choice(opener_pool)
    bridge = rng.choice(bank["bridges"])

    if target_item == 9:
        return _risk_tier_reply(target_score, rng)

    if intent == "deflect":
        tail = _maybe_tail(hedge_rate, normalization_rate, rng)
        return _safe_join([rng.choice(bank["deflectors"]), tail], limit_words=32)

    if intent == "partial":
        softened = max(0, target_score - 1)
        direct = _item_sentence(target_item, softened)
        context = _maybe_context_anchor(target_item, context_anchor_rate, rng)
        tail = _maybe_tail(hedge_rate, normalization_rate, rng)
        lead = f"{opener} it is a bit hard to pin down, but {direct}"
        return _safe_join([lead, context, tail], limit_words=36)

    direct = _item_sentence(target_item, target_score)
    context = _maybe_context_anchor(target_item, context_anchor_rate, rng)
    tail = _maybe_tail(hedge_rate, normalization_rate, rng)
    answer = _safe_join([f"{opener} {direct}", f"{bridge} {context}" if context else "", tail], limit_words=38)
    if rng.random() < min(0.2, hedge_rate * 0.35):
        return _safe_join(["I don't know, but", answer], limit_words=38)
    return answer


def normalize_response(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text).strip())
    if cleaned.count(".") > 2:
        parts = [part.strip() for part in cleaned.split(".") if part.strip()]
        cleaned = ". ".join(parts[:2]).strip()
        if cleaned and cleaned[-1] not in {".", "?", "!"}:
            cleaned += "."
    return cleaned
