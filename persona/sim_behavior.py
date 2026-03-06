from __future__ import annotations

import re
from typing import Dict, List

from core.state import BDI_ITEM_NAMES
from persona.sim_templates import (
    CONTEXT_ANCHORS,
    HEDGE_PHRASES,
    ITEM_CONTEXT_HINTS,
    ITEM_SENTENCE_BANK,
    NEUTRAL_CONTEXT_ANCHORS,
    NEUTRAL_ITEM_CONTEXT_HINTS,
    NORMALIZATION_PHRASES,
    RISK_PROTECTIVE_FACTORS,
    RISK_RESPONSE_BANK,
    SIM_TEMPLATE_BANK,
)


def _coerce_probe_intent(probe_intent: Dict[str, object] | None) -> Dict[str, object]:
    if not isinstance(probe_intent, dict):
        raise RuntimeError("Missing probe_intent for deterministic persona generation.")

    target_item_id = probe_intent.get("target_item_id")
    route = str(probe_intent.get("route", "")).strip().lower()
    style = str(probe_intent.get("style", "")).strip()
    mode = str(probe_intent.get("mode", "normal")).strip().lower()
    directness = str(probe_intent.get("directness", "indirect")).strip().lower()
    priority_raw = probe_intent.get("priority", 0.5)

    try:
        target_item = int(target_item_id)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Invalid probe_intent.target_item_id for deterministic persona generation.") from exc
    if target_item < 1 or target_item > 21:
        raise RuntimeError("Invalid probe_intent.target_item_id for deterministic persona generation.")
    if route not in {"somatic", "cognitive", "risk"}:
        raise RuntimeError("Invalid probe_intent.route for deterministic persona generation.")
    if not style:
        raise RuntimeError("Invalid probe_intent.style for deterministic persona generation.")
    if mode not in {"normal", "wrapup"}:
        raise RuntimeError("Invalid probe_intent.mode for deterministic persona generation.")
    if directness not in {"indirect", "direct"}:
        raise RuntimeError("Invalid probe_intent.directness for deterministic persona generation.")
    try:
        priority = float(priority_raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Invalid probe_intent.priority for deterministic persona generation.") from exc
    priority = max(0.0, min(1.0, priority))

    return {
        "target_item_id": target_item,
        "route": route,
        "style": style,
        "mode": mode,
        "directness": directness,
        "priority": priority,
    }


def _item_sentence(item_id: int, score: int) -> str:
    clipped = max(0, min(3, int(score)))
    if clipped == 0:
        neutral_zero_bank = [
            "that's been okay honestly, not really a problem",
            "I haven't noticed anything different there",
            "no, that side of things has been fine",
            "that's one area that hasn't really been an issue",
        ]
        return neutral_zero_bank[(item_id - 1) % len(neutral_zero_bank)]

    bank = ITEM_SENTENCE_BANK.get(item_id)
    if bank:
        return bank.get(clipped, bank[max(bank.keys())])

    symptom = BDI_ITEM_NAMES.get(item_id, f"item {item_id}").lower()
    return f"{symptom} has been more noticeable lately"


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


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
    route: str,
    mode: str,
    directness: str,
    style: str,
    priority: float,
    evasiveness: float,
    contradiction_rate: float,
    direct_answer_rate: float,
    rng,
) -> str:
    if route == "risk":
        return "disclose"
    direct_probe = directness == "direct"
    if direct_probe and rng.random() < max(0.12, min(0.45, evasiveness * 0.55)):
        return "deflect"
    adjusted_direct_rate = max(0.45, min(0.98, direct_answer_rate + 0.15 * priority))
    partial_floor = 0.18 if mode == "normal" else 0.24
    if rng.random() > adjusted_direct_rate and rng.random() < max(partial_floor, min(0.5, evasiveness + 0.1)):
        return "partial"
    if contradiction_rate > 0.0 and rng.random() < min(0.2, contradiction_rate * 0.8):
        return "partial"
    if "impact" in style and rng.random() < 0.12:
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


def _maybe_context_anchor(item_id: int, score: int, context_anchor_rate: float, rng) -> str:
    if rng.random() >= context_anchor_rate:
        return ""
    if score <= 1:
        hints = NEUTRAL_ITEM_CONTEXT_HINTS.get(item_id, NEUTRAL_CONTEXT_ANCHORS)
    else:
        hints = ITEM_CONTEXT_HINTS.get(item_id, CONTEXT_ANCHORS)
    return rng.choice(hints)


def _maybe_tail(hedge_rate: float, normalization_rate: float, rng) -> str:
    parts: List[str] = []
    if rng.random() < hedge_rate:
        hedge = rng.choice(HEDGE_PHRASES)
        hedge_completions = {
            "i guess": "I guess that's just how it is right now",
            "maybe": "maybe that's why everything feels heavier",
            "i don't know": "I don't know, that's just how it's been",
            "i think": "I think that's what it comes down to",
            "kind of": "kind of hard to explain but that's the feeling",
            "to be fair": "to be fair I'm probably not the best judge right now",
            "honestly": "honestly I don't really know what to do about it",
            "if i am being real": "if I'm being real it's been getting to me",
            "it is hard to say exactly": "it's hard to say exactly but that's how it feels",
            "in a way": "in a way I've just gotten used to it",
        }
        completed = hedge_completions.get(hedge.lower(), f"{hedge.lower()}, that's just how it's been")
        parts.append(completed)
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
    probe_intent: Dict[str, object],
    evasive: bool,
    rng,
) -> str:
    bank = SIM_TEMPLATE_BANK

    _ = family  # retained for API compatibility; behavior is now intent-driven.
    _ = split
    intent_payload = _coerce_probe_intent(probe_intent)
    target_item = int(intent_payload["target_item_id"])
    target_score = int(bdi_scores.get(target_item, 0))
    route = str(intent_payload["route"])
    style = str(intent_payload["style"])
    mode = str(intent_payload["mode"])
    directness = str(intent_payload["directness"])
    priority = float(intent_payload["priority"])

    evasiveness = float(behavior_params.get("evasiveness", 0.45))
    contradiction_rate = float(behavior_params.get("contradiction", 0.08))
    hedge_rate = float(behavior_params.get("hedge_rate", 0.65))
    normalization_rate = float(behavior_params.get("normalization_rate", 0.45))
    context_anchor_rate = float(behavior_params.get("context_anchor_rate", 0.55))
    direct_answer_rate = float(behavior_params.get("direct_answer_rate", 0.78))
    intent = _intent(
        route,
        mode,
        directness,
        style,
        priority,
        evasiveness if evasive else 0.1,
        contradiction_rate,
        direct_answer_rate,
        rng,
    )

    question_turns = sum(1 for msg in history if msg.get("role") == "user")
    opener_pool = bank["openers"][::2] if question_turns % 2 == 0 else bank["openers"][1::2] or bank["openers"]
    opener = rng.choice(opener_pool)
    bridge = rng.choice(bank["bridges"])

    if target_item == 9 or route == "risk":
        return _risk_tier_reply(target_score, rng)

    if intent == "deflect":
        tail = _maybe_tail(hedge_rate, normalization_rate, rng)
        return _safe_join([rng.choice(bank["deflectors"]), tail], limit_words=32)

    if intent == "partial":
        softened = max(0, target_score - 1)
        direct = _item_sentence(target_item, softened)
        context = _maybe_context_anchor(target_item, softened, context_anchor_rate, rng)
        tail = _maybe_tail(hedge_rate, normalization_rate, rng) if softened > 0 else ""
        hedged_lead = rng.choice([
            "it's a bit hard to pin down, but",
            "I'm not sure how to put it, but",
            "it's hard to say exactly, but",
        ])
        lead = f"{hedged_lead} {direct}"
        return _safe_join([lead, context, tail], limit_words=36)

    direct = _item_sentence(target_item, target_score)
    context = _maybe_context_anchor(target_item, target_score, context_anchor_rate, rng)
    tail = _maybe_tail(hedge_rate, normalization_rate, rng) if target_score > 0 else ""
    # Vary structure: sometimes use opener, sometimes go direct; drop rigid bridge
    parts = []
    if rng.random() < 0.45:
        parts.append(f"{opener} {direct}")
    else:
        parts.append(direct)
    if context:
        parts.append(context)
    if tail:
        parts.append(tail)
    answer = _safe_join(parts, limit_words=38)
    if rng.random() < min(0.2, hedge_rate * 0.35) and answer:
        # Blend prefix naturally into the sentence
        low = answer[0].lower() + answer[1:] if answer[0].isupper() else answer
        return _safe_join([f"I don't know, but {low}"], limit_words=38)
    return answer


def normalize_response(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text).strip())
    if cleaned.count(".") > 2:
        parts = [part.strip() for part in cleaned.split(".") if part.strip()]
        cleaned = ". ".join(parts[:2]).strip()
        if cleaned and cleaned[-1] not in {".", "?", "!"}:
            cleaned += "."
    return cleaned
