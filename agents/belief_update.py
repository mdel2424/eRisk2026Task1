from __future__ import annotations

import hashlib
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List

from core.bdi_modules import MODULE_TO_ITEMS
from core.state import AgentState, BeliefState, ItemBelief, coerce_item_belief

GUARDED_ITEM_IDS = {9, *MODULE_TO_ITEMS[1], *MODULE_TO_ITEMS[3], *MODULE_TO_ITEMS[4]}



def _normalize(values: List[float]) -> List[float]:
    clipped = [max(1e-8, float(v)) for v in values]
    total = sum(clipped)
    if total <= 0:
        return [0.25, 0.25, 0.25, 0.25]
    return [value / total for value in clipped]



def _posterior_stats(posterior: List[float]) -> tuple[float, float]:
    import math

    expected = sum(idx * prob for idx, prob in enumerate(posterior))
    entropy = 0.0
    for prob in posterior:
        p = max(1e-12, min(1.0, float(prob)))
        entropy -= p * math.log2(p)
    return max(0.0, min(3.0, expected)), max(0.0, min(2.0, entropy))



def _coerce_belief(item_id: int, value) -> ItemBelief:
    return coerce_item_belief(item_id, value)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _normalize_text_for_evidence_id(text: str) -> str:
    lowered = str(text or "").lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _fallback_evidence_id(item_id: int, direction: str, spans: List[str], symptom_name: str) -> str:
    source_text = ""
    for span in spans:
        if str(span).strip():
            source_text = str(span)
            break
    if not source_text:
        source_text = str(symptom_name or "")
    normalized = _normalize_text_for_evidence_id(source_text)
    payload = f"{int(item_id)}|{direction}|{normalized}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _normalize_direction(raw_direction: str) -> str:
    value = str(raw_direction or "").strip().lower()
    if value in {"increase", "decrease", "neutral"}:
        return value
    return "neutral"


def _dominant_direction(direction_counts: Dict[str, int]) -> str:
    increase = int(direction_counts.get("increase", 0) or 0)
    decrease = int(direction_counts.get("decrease", 0) or 0)
    if increase <= 0 and decrease <= 0:
        return "neutral"
    if increase > decrease:
        return "increase"
    if decrease > increase:
        return "decrease"
    return "neutral"


def _is_contradiction(current_direction: str, dominant_direction: str) -> bool:
    if current_direction not in {"increase", "decrease"}:
        return False
    if dominant_direction not in {"increase", "decrease"}:
        return False
    return current_direction != dominant_direction


def _method_weight(evidence_type: str) -> float:
    method_key = str(evidence_type or "").strip().lower()
    if method_key == "llm_extractor":
        return _env_float("BELIEF_WEIGHT_LLM_EXTRACTOR", 1.00)
    if method_key == "llm_salvage":
        return _env_float("BELIEF_WEIGHT_LLM_SALVAGE", 0.60)
    if method_key == "lexical_fallback":
        return _env_float("BELIEF_WEIGHT_LEXICAL_FALLBACK", 0.45)
    if method_key == "lexical_prefilter":
        return _env_float("BELIEF_WEIGHT_LEXICAL_PREFILTER", 0.40)
    return _env_float("BELIEF_WEIGHT_DEFAULT", 0.50)


def _support_increment_allowed(
    *,
    item_id: int,
    evidence_type: str,
    extract_confidence: float,
    extract_intensity: float,
    support_increment_blocked: bool,
) -> tuple[bool, str]:
    if bool(support_increment_blocked):
        return False, "precision_gate_blocked"

    if int(item_id) not in GUARDED_ITEM_IDS:
        return True, "non_guarded_item"

    method_key = str(evidence_type or "").strip().lower()
    if method_key == "llm_extractor":
        return True, "llm_extractor_allowed"
    if method_key == "llm_salvage":
        if float(extract_confidence) >= 0.55 and float(extract_intensity) >= 1.5:
            return True, "salvage_threshold_met"
        return False, "salvage_threshold_blocked"
    if method_key in {"lexical_fallback", "lexical_prefilter"}:
        if float(extract_confidence) >= 0.65 and float(extract_intensity) >= 1.75:
            return True, "lexical_threshold_met"
        return False, "lexical_threshold_blocked"
    return True, "unrecognized_method_allowed"



def update_beliefs(state: AgentState) -> Dict:
    turn = int(state.get("turn_index", 0))
    latest_likelihoods = list(state.get("latest_turn_likelihoods", []))
    prior_beliefs = state.get("item_beliefs", {})

    beliefs: Dict[int, ItemBelief] = {}
    for item_id in range(1, 22):
        beliefs[item_id] = _coerce_belief(item_id, prior_beliefs.get(item_id))

    combined_likelihood_by_item: Dict[int, List[float]] = {}
    support_increments_by_item: Dict[int, int] = {}
    method_counts: Counter[str] = Counter()
    per_item_stats = defaultdict(lambda: {"unique": 0, "duplicate": 0, "contradiction": 0, "weight_sum": 0.0, "rows": 0, "support_increments": 0})

    duplicate_rows_count = 0
    contradiction_rows_count = 0
    unique_rows_count = 0
    support_increments_count = 0
    support_rejected_by_method_count = 0
    support_rejected_guarded_item_count = 0

    duplicate_weight = _clamp(_env_float("BELIEF_DUPLICATE_WEIGHT", 0.15), 0.01, 1.0)
    decay_start_support = max(0, _env_int("BELIEF_DECAY_START_SUPPORT", 2))
    decay_tau = max(1e-6, _env_float("BELIEF_DECAY_TAU", 2.0))
    contradiction_weight = _clamp(_env_float("BELIEF_CONTRADICTION_WEIGHT", 0.50), 0.01, 1.0)
    contradiction_neutral_blend = _clamp(_env_float("BELIEF_CONTRADICTION_NEUTRAL_BLEND", 0.35), 0.0, 1.0)
    support_min_weight = _clamp(_env_float("BELIEF_SUPPORT_MIN_WEIGHT", 0.45), 0.0, 1.0)
    memory_per_item = max(1, _env_int("BELIEF_MEMORY_PER_ITEM", 24))

    raw_memory = state.get("item_evidence_memory", {})
    item_evidence_memory: Dict[int, List[str]] = {}
    if isinstance(raw_memory, dict):
        for raw_item_id, raw_values in raw_memory.items():
            try:
                item_id = int(raw_item_id)
            except (TypeError, ValueError):
                continue
            if item_id < 1 or item_id > 21:
                continue
            values: List[str] = []
            if isinstance(raw_values, list):
                for value in raw_values:
                    token = str(value or "").strip()
                    if token:
                        values.append(token)
            item_evidence_memory[item_id] = values[-memory_per_item:]
    for item_id in range(1, 22):
        item_evidence_memory.setdefault(item_id, [])

    raw_tally = state.get("item_direction_tally", {})
    item_direction_tally: Dict[int, Dict[str, int]] = {}
    if isinstance(raw_tally, dict):
        for raw_item_id, raw_counts in raw_tally.items():
            try:
                item_id = int(raw_item_id)
            except (TypeError, ValueError):
                continue
            if item_id < 1 or item_id > 21:
                continue
            counts = {"increase": 0, "decrease": 0, "neutral": 0}
            if isinstance(raw_counts, dict):
                counts["increase"] = max(0, int(raw_counts.get("increase", 0) or 0))
                counts["decrease"] = max(0, int(raw_counts.get("decrease", 0) or 0))
                counts["neutral"] = max(0, int(raw_counts.get("neutral", 0) or 0))
            item_direction_tally[item_id] = counts
    for item_id in range(1, 22):
        item_direction_tally.setdefault(item_id, {"increase": 0, "decrease": 0, "neutral": 0})

    for row in latest_likelihoods:
        item_id = int(getattr(row, "item_id", 0) or 0)
        if item_id < 1 or item_id > 21:
            continue
        values = [float(v) for v in list(getattr(row, "likelihood", [1.0, 1.0, 1.0, 1.0]))[:4]]
        if len(values) < 4:
            values.extend([1.0] * (4 - len(values)))

        evidence_type = str(getattr(row, "evidence_type", "llm_extractor") or "llm_extractor")
        method_counts[evidence_type] += 1
        extract_confidence = float(getattr(row, "extract_confidence", 0.0) or 0.0)
        extract_intensity = float(getattr(row, "extract_intensity", 0.0) or 0.0)
        support_increment_blocked = bool(getattr(row, "support_increment_blocked", False))
        row_direction = _normalize_direction(str(getattr(row, "direction", "neutral") or "neutral"))
        evidence_id = str(getattr(row, "evidence_id", "") or "").strip()
        spans = [str(v) for v in list(getattr(row, "spans", [])) if str(v).strip()]
        if not evidence_id:
            evidence_id = _fallback_evidence_id(
                item_id=item_id,
                direction=row_direction,
                spans=spans,
                symptom_name=str(getattr(row, "symptom_name", "") or ""),
            )

        existing_ids = item_evidence_memory.setdefault(item_id, [])
        is_duplicate = evidence_id in existing_ids if evidence_id else False
        novelty_factor = duplicate_weight if is_duplicate else 1.0
        if is_duplicate:
            duplicate_rows_count += 1
            per_item_stats[item_id]["duplicate"] += 1
        else:
            unique_rows_count += 1
            per_item_stats[item_id]["unique"] += 1

        current_support = int(beliefs[item_id].support_count) + int(support_increments_by_item.get(item_id, 0))
        support_decay = 1.0 / (1.0 + (max(0.0, float(current_support - decay_start_support)) / decay_tau))
        support_decay = _clamp(support_decay, 0.05, 1.0)

        direction_counts = item_direction_tally.get(item_id, {"increase": 0, "decrease": 0, "neutral": 0})
        dominant_direction = _dominant_direction(direction_counts)
        contradiction = _is_contradiction(row_direction, dominant_direction)
        contradiction_factor = contradiction_weight if contradiction else 1.0
        if contradiction:
            contradiction_rows_count += 1
            per_item_stats[item_id]["contradiction"] += 1

        method_weight = _method_weight(evidence_type)
        hint = float(getattr(row, "method_weight_hint", 0.0) or 0.0)
        if hint > 0.0:
            method_weight = (method_weight + hint) / 2.0
        method_weight = _clamp(method_weight, 0.05, 1.25)

        effective_weight = _clamp(
            method_weight * novelty_factor * support_decay * contradiction_factor,
            0.05,
            1.0,
        )

        if contradiction and contradiction_neutral_blend > 0.0:
            values = [((1.0 - contradiction_neutral_blend) * value) + contradiction_neutral_blend for value in values]

        weighted_values = [max(1e-8, value) ** effective_weight for value in values]
        if item_id not in combined_likelihood_by_item:
            combined_likelihood_by_item[item_id] = [1.0, 1.0, 1.0, 1.0]
        combined_likelihood_by_item[item_id] = [
            combined_likelihood_by_item[item_id][idx] * max(1e-8, weighted_values[idx])
            for idx in range(4)
        ]

        per_item_stats[item_id]["rows"] += 1
        per_item_stats[item_id]["weight_sum"] += float(effective_weight)

        allow_support_increment, support_reject_reason = _support_increment_allowed(
            item_id=item_id,
            evidence_type=evidence_type,
            extract_confidence=extract_confidence,
            extract_intensity=extract_intensity,
            support_increment_blocked=support_increment_blocked,
        )

        if (not is_duplicate) and effective_weight >= support_min_weight and allow_support_increment:
            support_increments_by_item[item_id] = int(support_increments_by_item.get(item_id, 0)) + 1
            support_increments_count += 1
            per_item_stats[item_id]["support_increments"] += 1
            direction_counts[row_direction] = int(direction_counts.get(row_direction, 0)) + 1
            item_direction_tally[item_id] = direction_counts
        elif (not is_duplicate) and effective_weight >= support_min_weight and not allow_support_increment:
            support_rejected_guarded_item_count += 1
            if support_reject_reason != "non_guarded_item":
                support_rejected_by_method_count += 1

        if (not is_duplicate) and evidence_id:
            existing_ids.append(evidence_id)
            if len(existing_ids) > memory_per_item:
                item_evidence_memory[item_id] = existing_ids[-memory_per_item:]

    updated_item_ids: List[int] = []
    for item_id, combined in combined_likelihood_by_item.items():
        prior = beliefs[item_id]
        prior_posterior = [float(v) for v in list(prior.posterior)[:4]]
        if len(prior_posterior) < 4:
            prior_posterior.extend([0.25] * (4 - len(prior_posterior)))

        posterior = _normalize(
            [prior_posterior[idx] * max(1e-8, combined[idx]) for idx in range(4)]
        )
        expected_score, entropy = _posterior_stats(posterior)
        beliefs[item_id] = ItemBelief(
            item_id=item_id,
            posterior=posterior,
            expected_score=expected_score,
            entropy=entropy,
            support_count=int(prior.support_count) + int(support_increments_by_item.get(item_id, 0)),
            last_update_turn=max(0, turn),
        )
        updated_item_ids.append(item_id)

    turn_trace = dict(state.get("turn_trace", {}))
    window_size = 4
    new_items_this_turn = len(updated_item_ids)
    nonempty_this_turn = 1 if len(latest_likelihoods) > 0 else 0
    recent_new_items = list(state.get("recent_new_items_window", [])) + [new_items_this_turn]
    recent_nonempty = list(state.get("recent_nonempty_window", [])) + [nonempty_this_turn]
    if len(recent_new_items) > window_size:
        recent_new_items = recent_new_items[-window_size:]
    if len(recent_nonempty) > window_size:
        recent_nonempty = recent_nonempty[-window_size:]

    belief_trace_payload = {
        "turn": turn,
        "updated_item_ids": sorted(updated_item_ids),
        "likelihood_rows": len(latest_likelihoods),
        "new_items_this_turn": new_items_this_turn,
        "recent_new_items_window": recent_new_items,
        "recent_nonempty_window": recent_nonempty,
        "unique_rows_count": int(unique_rows_count),
        "duplicate_rows_count": int(duplicate_rows_count),
        "contradiction_rows_count": int(contradiction_rows_count),
        "support_increments_count": int(support_increments_count),
        "support_rejected_by_method_count": int(support_rejected_by_method_count),
        "support_rejected_guarded_item_count": int(support_rejected_guarded_item_count),
        "method_counts": {str(key): int(value) for key, value in method_counts.items()},
        "per_item_stats": [
            {
                "item_id": int(item_id),
                "unique_rows": int(stats["unique"]),
                "duplicate_rows": int(stats["duplicate"]),
                "contradiction_rows": int(stats["contradiction"]),
                "effective_weight_mean": round(
                    float(stats["weight_sum"]) / float(max(1, int(stats["rows"]))),
                    6,
                ),
                "support_increments": int(stats["support_increments"]),
            }
            for item_id, stats in sorted(per_item_stats.items(), key=lambda pair: int(pair[0]))
        ],
    }
    turn_trace["belief_update"] = belief_trace_payload
    turn_trace["update_beliefs"] = belief_trace_payload

    return {
        "beliefs": BeliefState(items=beliefs),
        "item_beliefs": beliefs,
        "item_evidence_memory": item_evidence_memory,
        "item_direction_tally": item_direction_tally,
        "new_items_this_turn": new_items_this_turn,
        "recent_new_items_window": recent_new_items,
        "recent_nonempty_window": recent_nonempty,
        "turn_trace": turn_trace,
    }
