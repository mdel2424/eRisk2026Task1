from __future__ import annotations

import re
from typing import Dict, List

from core.state import BDI_ITEM_NAMES
from persona.sim_templates import (
    ALL_CONTEXT_ANCHORS,
    BASELINE_COMPARISON_PHRASES,
    CONTRASTIVE_NEGATIVE_BANK,
    CONTEXT_ANCHORS,
    CONTEXT_TAG_ANCHORS,
    CONTROL_OPENING_SUMMARY_BANK,
    HEDGE_PHRASES,
    ITEM_CONTEXT_HINTS,
    ITEM_CONCRETE_EXAMPLES,
    ITEM_SENTENCE_BANK,
    MINIMIZATION_FRAGMENTS,
    NEUTRAL_CONTEXT_ANCHORS,
    NEUTRAL_CONTEXT_TAG_ANCHORS,
    NEUTRAL_ITEM_CONTEXT_HINTS,
    NORMALIZATION_PHRASES,
    OPENING_SUMMARY_BANK,
    PARTIAL_ANSWER_BANK,
    QUALIFIED_UNSURE_PHRASES,
    RESPONSE_QUALIFIERS,
    RISK_PROTECTIVE_FACTORS,
    RISK_RESPONSE_BANK,
    SIM_TEMPLATE_BANK,
    SOFT_DENIAL_PHRASES,
    CONTEXT_RESPONSE_EXAMPLES,
    STYLE_DEFLECTORS,
    STYLE_OPENERS,
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
        return SOFT_DENIAL_PHRASES[(item_id - 1) % len(SOFT_DENIAL_PHRASES)]

    bank = ITEM_SENTENCE_BANK.get(item_id)
    if bank:
        return bank.get(clipped, bank[max(bank.keys())])

    symptom = BDI_ITEM_NAMES.get(item_id, f"item {item_id}").lower()
    return f"{symptom} has been more noticeable lately"


def _is_depressed_family(family: str) -> bool:
    return str(family or "").strip().lower() not in {"control_stressed", "control_neutral"}


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def response_style_flags(text: str) -> Dict[str, bool]:
    lowered = _normalize(text)
    qualifier = any(_normalize(phrase) in lowered for phrase in list(HEDGE_PHRASES) + list(RESPONSE_QUALIFIERS))
    qualifier = qualifier or any(token in lowered for token in ("a bit", "a little", "mostly", "not really"))
    deflect = any(token in lowered for token in ("labels", "diagnosis", "diagnostic", "not naming", "not really labels"))
    context = any(anchor.lower() in lowered for anchor in ALL_CONTEXT_ANCHORS)
    context = context or any(
        token in lowered
        for token in (
            "work",
            "kids",
            "family",
            "messages",
            "meeting",
            "phone",
            "evening",
            "morning",
            "routine",
            "traffic",
            "partner",
            "bills",
            "guitar",
            "painting",
            "couch",
            "bed",
            "texts",
            "deadline",
        )
    )
    mixed = any(
        token in lowered
        for token in (
            "a bit of both",
            "both show up",
            "both are in there",
            "a mix of both",
            "mostly heavy but",
            "more toward irritability",
        )
    )
    soft_denial = any(_normalize(phrase) in lowered for phrase in SOFT_DENIAL_PHRASES) or any(
        token in lowered for token in ("not much change there", "about the same", "pretty close to normal")
    )
    baseline = any(
        token in lowered
        for token in (
            "than usual",
            "compared with usual",
            "compared with my normal",
            "more than it used to be",
            "more noticeable than usual",
            "most days now",
            "last couple of weeks",
        )
    )
    return {
        "hedged": qualifier,
        "qualifier": qualifier,
        "deflect": deflect,
        "context_anchor": context,
        "mixed_answer": mixed,
        "soft_denial": soft_denial,
        "baseline_comparison": baseline,
    }


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
    kept_chunks: List[str] = []
    used_words = 0
    for piece in normalized_chunks:
        piece_words = piece.split()
        if not kept_chunks:
            if len(piece_words) > limit_words:
                text = " ".join(piece_words[:limit_words]).rstrip(" ,;")
                if text and text[-1] not in {".", "?", "!"}:
                    text += "."
                return re.sub(r"\s+", " ", text).strip()
            kept_chunks.append(piece)
            used_words = len(piece_words)
            continue
        if used_words + len(piece_words) > limit_words:
            break
        kept_chunks.append(piece)
        used_words += len(piece_words)
    text = " ".join(kept_chunks)
    if text and text[-1] not in {".", "?", "!"}:
        text += "."
    return re.sub(r"\s+", " ", text).strip()


def _style_bank(values: List[str], defaults: List[str]) -> List[str]:
    return list(values or []) + list(defaults)


def _maybe_context_anchor(item_id: int, score: int, context_anchor_rate: float, context_tag: str, rng) -> str:
    if rng.random() >= context_anchor_rate:
        return ""
    if score <= 1:
        hints = list(NEUTRAL_ITEM_CONTEXT_HINTS.get(item_id, [])) + list(
            NEUTRAL_CONTEXT_TAG_ANCHORS.get(context_tag, NEUTRAL_CONTEXT_ANCHORS)
        )
    else:
        hints = list(ITEM_CONTEXT_HINTS.get(item_id, [])) + list(CONTEXT_TAG_ANCHORS.get(context_tag, CONTEXT_ANCHORS))
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


def _disclosure_progress(question_turns: int) -> float:
    return max(0.0, min(1.0, float(question_turns) / 10.0))


def _profile_total(bdi_scores: Dict[int, int]) -> int:
    return sum(int(score) for score in bdi_scores.values())


def _is_low_severity_precision_profile(family: str, bdi_scores: Dict[int, int]) -> bool:
    normalized_family = str(family or "").strip().lower()
    total = _profile_total(bdi_scores)
    if normalized_family == "control_stressed":
        return True
    return normalized_family in {"risk_leaning", "cognitive_ruminative"} and total <= 13


def _cluster_for_target(item_id: int) -> str:
    if item_id in {1, 10, 17}:
        return "tone_balance"
    if item_id in {4, 12, 15, 20}:
        return "energy_interest"
    if item_id in {11, 15, 20}:
        return "slowed_restless"
    if item_id == 18:
        return "appetite_variability"
    return ""


def _cluster_support(cluster: str, bdi_scores: Dict[int, int]) -> bool:
    if cluster == "tone_balance":
        return sum(1 for item_id in (1, 10, 17, 20) if int(bdi_scores.get(item_id, 0)) >= 1) >= 2
    if cluster == "energy_interest":
        return sum(1 for item_id in (4, 12, 15, 20) if int(bdi_scores.get(item_id, 0)) >= 1) >= 2
    if cluster == "slowed_restless":
        return sum(1 for item_id in (11, 15, 20) if int(bdi_scores.get(item_id, 0)) >= 1) >= 2
    if cluster == "appetite_variability":
        return int(bdi_scores.get(18, 0)) >= 1
    return False


def _cluster_strong_count(cluster: str, bdi_scores: Dict[int, int]) -> int:
    if cluster == "tone_balance":
        item_ids = (1, 10, 17, 20)
    elif cluster == "energy_interest":
        item_ids = (4, 12, 15, 20)
    elif cluster == "slowed_restless":
        item_ids = (11, 15, 20)
    elif cluster == "appetite_variability":
        item_ids = (18,)
    else:
        return 0
    return sum(1 for item_id in item_ids if int(bdi_scores.get(item_id, 0)) >= 2)


def _soft_denial_redirect(target_item: int, bdi_scores: Dict[int, int]) -> str:
    if target_item == 18 and any(int(bdi_scores.get(item_id, 0)) >= 2 for item_id in (15, 20)):
        return "if anything, the bigger shift has been energy more than appetite"
    if target_item == 21 and any(int(bdi_scores.get(item_id, 0)) >= 2 for item_id in (15, 20)):
        return "if anything, it is the fatigue side that stands out more than closeness"
    if target_item == 1 and any(int(bdi_scores.get(item_id, 0)) >= 2 for item_id in (17, 20)):
        return "if anything, it comes out more as irritability and feeling worn down"
    if target_item in {4, 12} and any(int(bdi_scores.get(item_id, 0)) >= 2 for item_id in (15, 20)):
        return "the bigger issue is that everything takes more effort than it used to"
    if target_item in {11, 15, 20} and int(bdi_scores.get(19, 0)) >= 2:
        return "the mental slowing and focus side is more noticeable than that specific piece"
    return ""


def _contrastive_negative_claim(target_item: int, bdi_scores: Dict[int, int], rng) -> str:
    bank_key = ""
    if target_item == 1 and any(int(bdi_scores.get(item_id, 0)) >= 2 for item_id in (17, 20)):
        bank_key = "sadness_vs_irritability"
    elif target_item in {4, 12} and any(int(bdi_scores.get(item_id, 0)) >= 2 for item_id in (15, 20)):
        bank_key = "interest_vs_energy"
    elif target_item == 18 and any(int(bdi_scores.get(item_id, 0)) >= 2 for item_id in (15, 16, 20)):
        bank_key = "appetite_vs_fatigue"
    elif target_item in {11, 15, 20} and int(bdi_scores.get(19, 0)) >= 2:
        bank_key = "sleep_vs_focus"
    if not bank_key:
        return ""
    return rng.choice(CONTRASTIVE_NEGATIVE_BANK[bank_key])


def _select_soft_denial_phrase(style_tag: str, directness: str, rng) -> str:
    if style_tag == "minimizing_practical" or directness == "direct":
        preferred = [
            phrase
            for phrase in SOFT_DENIAL_PHRASES
            if any(token in phrase.lower() for token in ("same", "normal", "not really"))
        ]
        if preferred:
            return rng.choice(preferred)
    return rng.choice(SOFT_DENIAL_PHRASES)


def _cluster_strengths(bdi_scores: Dict[int, int]) -> Dict[str, int]:
    return {
        "somatic_fatigue_sleep": sum(int(bdi_scores.get(item_id, 0)) for item_id in (15, 16, 20)),
        "appetite_variability": sum(int(bdi_scores.get(item_id, 0)) for item_id in (18, 21)),
        "interest_withdrawal": sum(int(bdi_scores.get(item_id, 0)) for item_id in (4, 12)),
        "irritability_tension": sum(int(bdi_scores.get(item_id, 0)) for item_id in (1, 11, 17)),
        "cognitive_self_eval": sum(int(bdi_scores.get(item_id, 0)) for item_id in (5, 7, 8, 14)),
        "focus_decision": sum(int(bdi_scores.get(item_id, 0)) for item_id in (13, 19)),
        "hopeless_risk": sum(int(bdi_scores.get(item_id, 0)) for item_id in (2, 6, 9)),
    }


def _top_opening_clusters(bdi_scores: Dict[int, int]) -> List[str]:
    ranked = sorted(_cluster_strengths(bdi_scores).items(), key=lambda pair: (-int(pair[1]), pair[0]))
    positive = [name for name, score in ranked if int(score) > 0]
    return positive[:2]


def _cluster_example_item(cluster_name: str, bdi_scores: Dict[int, int]) -> int:
    cluster_items = {
        "somatic_fatigue_sleep": (15, 16, 20),
        "appetite_variability": (18, 21),
        "interest_withdrawal": (4, 12),
        "irritability_tension": (1, 11, 17),
        "cognitive_self_eval": (5, 7, 8, 14),
        "focus_decision": (13, 19),
        "hopeless_risk": (2, 6, 9),
    }.get(cluster_name, ())
    ranked = sorted(cluster_items, key=lambda item_id: (-int(bdi_scores.get(item_id, 0)), item_id))
    return int(ranked[0]) if ranked else 15


def _recent_soft_denial(history: List[dict], *, window: int = 1) -> bool:
    assistant_messages = [
        str(msg.get("content") or msg.get("message") or "").strip()
        for msg in history
        if str(msg.get("role", "")).strip().lower() == "assistant"
    ]
    for text in assistant_messages[-window:]:
        if response_style_flags(text).get("soft_denial"):
            return True
    return False


def _qualified_unsure_count(history: List[dict]) -> int:
    normalized_phrases = [_normalize(phrase) for phrase in QUALIFIED_UNSURE_PHRASES]
    count = 0
    for message in history:
        if str(message.get("role", "")).strip().lower() != "assistant":
            continue
        text = _normalize(str(message.get("content") or message.get("message") or ""))
        if not text:
            continue
        if any(phrase in text for phrase in normalized_phrases):
            count += 1
    return count


def _opening_summary_reply(
    *,
    family: str,
    context_tag: str,
    style_tag: str,
    bdi_scores: Dict[int, int],
    rng,
) -> str:
    depressed = _is_depressed_family(family)
    if not depressed:
        lead = rng.choice(CONTROL_OPENING_SUMMARY_BANK)
        context = rng.choice(CONTEXT_RESPONSE_EXAMPLES.get(context_tag, CONTEXT_RESPONSE_EXAMPLES["routine_stable"]))
        return _safe_join([lead, context], limit_words=36)

    top_clusters = _top_opening_clusters(bdi_scores)
    if not top_clusters:
        top_clusters = ["interest_withdrawal"]
    claims = [rng.choice(OPENING_SUMMARY_BANK.get(cluster_name, OPENING_SUMMARY_BANK["interest_withdrawal"])) for cluster_name in top_clusters[:2]]
    lead = claims[0]
    if len(claims) > 1 and style_tag != "terse_guarded":
        lead = f"{claims[0]}, and {claims[1]}"
    example_item = _cluster_example_item(top_clusters[0], bdi_scores)
    example = _concrete_example(example_item, context_tag, int(bdi_scores.get(example_item, 0)), rng)
    context = rng.choice(CONTEXT_RESPONSE_EXAMPLES.get(context_tag, CONTEXT_RESPONSE_EXAMPLES["routine_stable"]))
    parts = [lead]
    if context:
        parts.append(context)
    if example and style_tag in {"contextual_reflective", "hedged_uncertain"} and rng.random() < 0.30:
        parts.append(example)
    return _safe_join(parts, limit_words=34)


def _qualifier_prefix(style_tag: str, score: int, question_turns: int, rng) -> str:
    progress = _disclosure_progress(question_turns)
    base_rate = 0.16 if score >= 2 else 0.26
    if style_tag in {"hedged_uncertain", "minimizing_practical"}:
        base_rate += 0.08
    if style_tag == "open_but_flat":
        base_rate -= 0.08
    base_rate *= max(0.45, 1.0 - (0.35 * progress))
    if rng.random() >= base_rate:
        return ""
    return rng.choice(RESPONSE_QUALIFIERS)


def _baseline_fragment(score: int, rng) -> str:
    bank = BASELINE_COMPARISON_PHRASES.get(max(0, min(3, int(score))), [])
    return rng.choice(bank) if bank else ""


def _concrete_example(item_id: int, context_tag: str, target_score: int, rng) -> str:
    if target_score <= 0:
        return ""
    item_examples = list(ITEM_CONCRETE_EXAMPLES.get(item_id, []))
    context_examples = list(CONTEXT_RESPONSE_EXAMPLES.get(context_tag, []))
    choices = item_examples + context_examples
    if not choices:
        return ""
    return rng.choice(choices)


def _minimization_fragment(style_tag: str, normalization_rate: float, question_turns: int, rng) -> str:
    progress = _disclosure_progress(question_turns)
    base_rate = normalization_rate * max(0.25, 1.0 - (0.55 * progress))
    if style_tag == "minimizing_practical":
        base_rate += 0.06
    if style_tag == "contextual_reflective":
        base_rate -= 0.04
    if rng.random() >= max(0.0, min(0.9, base_rate)):
        return ""
    return rng.choice(MINIMIZATION_FRAGMENTS)


def _prefixed_claim(opener: str, qualifier: str, claim: str, rng) -> str:
    lead = claim
    if qualifier:
        lead = f"{qualifier}, {lead}"
    if opener and rng.random() < 0.35:
        lead = f"{opener} {lead}"
    return lead


def _response_mode(
    *,
    family: str,
    target_item: int,
    target_score: int,
    route: str,
    style_tag: str,
    directness: str,
    priority: float,
    question_turns: int,
    recent_soft_denial: bool,
    history: List[dict],
    bdi_scores: Dict[int, int],
    evasiveness: float,
    direct_answer_rate: float,
    rng,
) -> str:
    if route == "risk":
        return "risk"

    progress = _disclosure_progress(question_turns)
    low_severity_profile = _is_low_severity_precision_profile(family, bdi_scores)
    qualified_unsure_used = _qualified_unsure_count(history)
    direct_probe = directness == "direct"
    deflect_rate = 0.0
    if direct_probe and question_turns <= 3 and style_tag in {"terse_guarded", "hedged_uncertain"}:
        deflect_rate = min(0.08, max(0.0, evasiveness * 0.12))
    if rng.random() < deflect_rate:
        return "deflect"

    if target_score <= 0:
        depressed = _is_depressed_family(family)
        has_contrastive_claim = bool(_contrastive_negative_claim(target_item, bdi_scores, rng))
        if has_contrastive_claim:
            return "contrastive_negative"
        if low_severity_profile:
            return "soft_denial"
        if depressed:
            if qualified_unsure_used >= 1:
                return "soft_denial"
            return "qualified_unsure"
        if style_tag == "hedged_uncertain" and rng.random() < 0.12:
            return "qualified_unsure"
        return "soft_denial"

    cluster = _cluster_for_target(target_item)
    strong_cluster_count = _cluster_strong_count(cluster, bdi_scores)
    if cluster and _cluster_support(cluster, bdi_scores):
        if (
            cluster in {"energy_interest", "slowed_restless"}
            and directness == "indirect"
            and target_score >= 2
            and strong_cluster_count >= 2
            and (
                (priority >= 0.75 and style_tag in {"contextual_reflective", "hedged_uncertain"})
                or (family == "somatic_evasive" and priority >= 0.6 and target_item in {4, 12})
            )
        ):
            return "mixed"
        mixed_rate = 0.14 + (0.10 if style_tag in {"contextual_reflective", "hedged_uncertain"} else 0.0)
        mixed_rate += 0.05 if target_score <= 2 else -0.03
        mixed_rate += 0.05 if priority >= 0.6 else 0.0
        if cluster == "energy_interest" and target_score >= 2:
            mixed_rate += 0.08
        if cluster == "slowed_restless" and any(int(bdi_scores.get(item_id, 0)) >= 2 for item_id in (11, 15, 20)):
            mixed_rate += 0.06
        mixed_rate *= max(0.9, 1.0 + (0.15 * progress))
        if rng.random() < max(0.0, min(0.55, mixed_rate)):
            return "mixed"

    if target_score == 1:
        return "soft_positive"
    if style_tag in {"hedged_uncertain", "minimizing_practical"} and rng.random() < max(0.18, 0.35 - (0.15 * progress)):
        return "soft_positive"
    if (
        direct_answer_rate < 0.65
        and rng.random() > (direct_answer_rate + (0.08 * progress))
        and not (low_severity_profile and qualified_unsure_used >= 1)
    ):
        return "qualified_unsure"
    return "direct_positive"


def _build_deterministic_reply_payload(
    *,
    family: str,
    split: str,
    context_tag: str,
    style_tag: str,
    bdi_scores: Dict[int, int],
    behavior_params: Dict[str, float | str],
    history: List[dict],
    probe_intent: Dict[str, object],
    evasive: bool,
    rng,
) -> tuple[str, str]:
    bank = SIM_TEMPLATE_BANK

    _ = family
    _ = split
    intent_payload = _coerce_probe_intent(probe_intent)
    target_item = int(intent_payload["target_item_id"])
    target_score = int(bdi_scores.get(target_item, 0))
    route = str(intent_payload["route"])
    style = str(intent_payload["style"])
    directness = str(intent_payload["directness"])
    priority = float(intent_payload["priority"])

    evasiveness = float(behavior_params.get("evasiveness", 0.45))
    hedge_rate = float(behavior_params.get("hedge_rate", 0.52))
    normalization_rate = float(behavior_params.get("normalization_rate", 0.18))
    context_anchor_rate = float(behavior_params.get("context_anchor_rate", 0.50))
    direct_answer_rate = float(behavior_params.get("direct_answer_rate", 0.86))
    question_turns = sum(1 for msg in history if msg.get("role") == "user")
    assistant_turns = sum(1 for msg in history if msg.get("role") == "assistant")
    opening_turn = style == "opening" or assistant_turns == 0
    recent_soft_denial = _recent_soft_denial(history)
    style_openers = _style_bank(STYLE_OPENERS.get(style_tag, []), bank["openers"])
    style_deflectors = _style_bank(STYLE_DEFLECTORS.get(style_tag, []), bank["deflectors"])
    opener_pool = style_openers[::2] if question_turns % 2 == 0 else style_openers[1::2] or style_openers
    opener = rng.choice(opener_pool)
    progress = _disclosure_progress(question_turns)
    low_severity_profile = _is_low_severity_precision_profile(family, bdi_scores)

    if opening_turn and route != "risk":
        return _opening_summary_reply(
            family=family,
            context_tag=context_tag,
            style_tag=style_tag,
            bdi_scores=bdi_scores,
            rng=rng,
        ), "opening_summary"

    mode_name = _response_mode(
        family=family,
        target_item=target_item,
        target_score=target_score,
        route=route,
        style_tag=style_tag,
        directness=directness,
        priority=priority,
        question_turns=question_turns,
        recent_soft_denial=recent_soft_denial,
        history=history,
        bdi_scores=bdi_scores,
        evasiveness=evasiveness if evasive else 0.1,
        direct_answer_rate=direct_answer_rate,
        rng=rng,
    )

    if target_item == 9 or route == "risk" or mode_name == "risk":
        return _risk_tier_reply(target_score, rng), "risk"

    qualifier = _qualifier_prefix(style_tag, target_score, question_turns, rng)
    baseline = _baseline_fragment(target_score, rng)
    example = _concrete_example(target_item, context_tag, target_score, rng)
    context = _maybe_context_anchor(
        target_item,
        target_score,
        min(0.90, context_anchor_rate + (0.08 * max(0.0, 0.8 - progress))),
        context_tag,
        rng,
    )
    minimization = _minimization_fragment(style_tag, normalization_rate, question_turns, rng)

    if mode_name == "deflect":
        tail = _maybe_tail(hedge_rate * 0.4, normalization_rate * 0.6, rng)
        return _safe_join([rng.choice(style_deflectors), tail], limit_words=32), "deflect"

    if mode_name == "contrastive_negative":
        claim = _contrastive_negative_claim(target_item, bdi_scores, rng)
        parts = [_prefixed_claim(opener, qualifier, claim, rng)]
        if context and rng.random() < 0.70:
            parts.append(context)
        elif example and rng.random() < 0.55:
            parts.append(example)
        return _safe_join(parts, limit_words=36), "contrastive_negative"

    if mode_name == "soft_denial":
        denial = _select_soft_denial_phrase(style_tag, directness, rng)
        redirect = _soft_denial_redirect(target_item, bdi_scores)
        parts = [_prefixed_claim(opener, qualifier, denial, rng)]
        if redirect:
            parts.append(redirect)
        elif context and rng.random() < 0.35:
            parts.append(context)
        return _safe_join(parts, limit_words=30), "soft_denial"

    if mode_name == "qualified_unsure":
        unsure = rng.choice(QUALIFIED_UNSURE_PHRASES)
        parts = [_prefixed_claim(opener, qualifier, unsure, rng)]
        if baseline and not low_severity_profile and rng.random() < 0.18:
            parts.append(baseline)
        if example and rng.random() < 0.80:
            parts.append(example)
        elif context and rng.random() < (0.45 if low_severity_profile else 0.65):
            parts.append(context)
        return _safe_join(parts, limit_words=36), "qualified_unsure"

    if mode_name == "mixed":
        cluster = _cluster_for_target(target_item)
        claim = rng.choice(PARTIAL_ANSWER_BANK.get(cluster, PARTIAL_ANSWER_BANK["energy_interest"]))
        parts = [_prefixed_claim(opener, qualifier, claim, rng)]
        if baseline and rng.random() < 0.22:
            parts.append(baseline)
        if example:
            parts.append(example)
        elif context:
            parts.append(context)
        if minimization and rng.random() < 0.45:
            parts.append(minimization)
        return _safe_join(parts, limit_words=38), "mixed"

    direct = _item_sentence(target_item, target_score)
    parts = [_prefixed_claim(opener, qualifier, direct, rng)]
    if baseline and rng.random() < 0.22:
        parts.append(baseline)
    if example and rng.random() < 0.90:
        parts.append(example)
    elif context:
        parts.append(context)
    elif rng.random() < 0.55:
        extra_context = rng.choice(CONTEXT_RESPONSE_EXAMPLES.get(context_tag, CONTEXT_RESPONSE_EXAMPLES["routine_stable"]))
        parts.append(extra_context)
    if mode_name == "soft_positive" and minimization and not (low_severity_profile and target_score <= 1 and not example):
        parts.append(minimization)
    elif mode_name == "direct_positive" and progress < 0.35 and rng.random() < max(0.0, hedge_rate * 0.15):
        tail = _maybe_tail(hedge_rate * 0.35, normalization_rate * 0.35, rng)
        if tail:
            parts.append(tail)
    return _safe_join(parts, limit_words=38), mode_name


def build_deterministic_reply(
    *,
    family: str,
    split: str,
    context_tag: str,
    style_tag: str,
    bdi_scores: Dict[int, int],
    behavior_params: Dict[str, float | str],
    history: List[dict],
    probe_intent: Dict[str, object],
    evasive: bool,
    rng,
) -> str:
    text, _mode_name = _build_deterministic_reply_payload(
        family=family,
        split=split,
        context_tag=context_tag,
        style_tag=style_tag,
        bdi_scores=bdi_scores,
        behavior_params=behavior_params,
        history=history,
        probe_intent=probe_intent,
        evasive=evasive,
        rng=rng,
    )
    return text


def normalize_response(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text).strip())
    if cleaned.count(".") > 3:
        parts = [part.strip() for part in cleaned.split(".") if part.strip()]
        cleaned = ". ".join(parts[:3]).strip()
        if cleaned and cleaned[-1] not in {".", "?", "!"}:
            cleaned += "."
    return cleaned
