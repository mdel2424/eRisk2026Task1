from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

from core.bdi_modules import ITEM_TO_MODULES, MODULE_NAMES, MODULE_TO_ITEMS, MODULE_WEIGHTS
from core.state import (
    AgentState,
    ControlState,
    FinalState,
    ItemBelief,
    coerce_item_belief,
    symptom_name_from_item,
    top_symptoms_from_scores,
)

CORE_ITEM_IDS = [2, 3, 4, 5, 7, 8, 14, 15, 16, 19, 20]
LOW_SIGNAL_COGNITIVE_AFFECTIVE_MODULES = {1, 2, 3, 4}
LOW_SIGNAL_SOMATIC_INTERPERSONAL_MODULES = {5, 6, 7, 8}
SEVERE_ANCHOR_NEGATION_PHRASES = [
    "not really a problem",
    "nothing different",
    "havent noticed anything different",
    "hasnt really been an issue",
    "that side of things has been fine",
    "about normal",
    "okay honestly",
    "not a big thing",
    "i feel reasonably confident",
]
SEVERE_ANCHOR_RULES = [
    {
        "module_id": 3,
        "item_ids": [5, 7, 8, 14],
        "strong_phrases": [
            "genuinely feel worthless",
            "feel worthless",
            "i am worthless",
            "im worthless",
            "feel like a burden",
            "i am a burden",
            "im a burden",
            "dont matter",
            "do not matter",
            "dont contribute",
            "do not contribute",
            "dont measure up",
            "do not measure up",
            "feel like a failure",
            "i am a failure",
            "im a failure",
        ],
        "mild_phrases": [],
    },
    {
        "module_id": 1,
        "item_ids": [4, 12, 17],
        "strong_phrases": [
            "not worth the effort",
            "way more effort than its worth",
            "way more effort than it is worth",
            "putting off replies",
            "putting off messages",
            "turned down things id normally go to",
            "turned down a few things id normally go to",
            "social stuff feels like way more effort",
            "social things feel like way more effort",
        ],
        "mild_phrases": [],
    },
    {
        "module_id": 5,
        "item_ids": [11, 15, 20],
        "strong_phrases": [
            "getting out of bed is a battle",
            "everything takes so much energy",
            "cant get going",
            "cannot get going",
            "takes so much more effort",
        ],
        "mild_phrases": [
            "takes a little more effort to get going than it used to",
            "takes a little more effort to get going",
            "it takes a little more effort to get going",
        ],
    },
    {
        "module_id": 6,
        "item_ids": [16],
        "strong_phrases": [
            "wake up in the middle of the night",
            "cant get back to sleep",
            "cannot get back to sleep",
            "lie there staring at the ceiling",
        ],
        "mild_phrases": [],
    },
    {
        "module_id": 4,
        "item_ids": [13, 19],
        "strong_phrases": [
            "cant focus",
            "cannot focus",
            "cant concentrate",
            "cannot concentrate",
        ],
        "mild_phrases": [
            "decisions take me longer",
            "decisions take a little longer than they used to",
            "hard to make simple decisions",
        ],
    },
]



def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: str) -> float:
    try:
        value = float(os.getenv(name, default))
    except (TypeError, ValueError):
        value = float(default)
    return float(value)


def _round_item_score(value: float) -> int:
    return max(0, min(3, int(round(_clamp(value, 0.0, 3.0)))))


def _normalize_text(text: object) -> str:
    normalized = str(text or "").strip().lower()
    normalized = normalized.replace("’", "'").replace("`", "'")
    normalized = normalized.replace("'", "")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def _contains_any(text: str, phrases: List[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _module_stats_from_beliefs(
    item_beliefs: Dict[int, ItemBelief],
    *,
    excluded_item_id: int | None = None,
) -> Dict[int, Dict[str, float | int | List[int]]]:
    module_stats: Dict[int, Dict[str, float | int | List[int]]] = {}
    for module_id, module_items in MODULE_TO_ITEMS.items():
        observed_items = [
            item_id
            for item_id in module_items
            if item_id != excluded_item_id and int(item_beliefs[item_id].support_count) > 0
        ]
        if not observed_items:
            continue

        observed_item_count = len(observed_items)
        coverage = float(observed_item_count) / float(max(1, len(module_items)))
        avg_support = sum(float(item_beliefs[item_id].support_count) for item_id in observed_items) / float(
            observed_item_count
        )
        support_strength = _clamp(avg_support / 2.0, 0.0, 1.0)
        module_conf = _clamp((0.20 + (0.50 * coverage) + (0.30 * support_strength)), 0.0, 1.0)

        weighted_sum = 0.0
        weight_total = 0.0
        for item_id in observed_items:
            belief = item_beliefs[item_id]
            local_weight = _clamp(0.5 + (0.25 * float(belief.support_count)), 0.0, 1.0)
            weighted_sum += float(belief.expected_score) * local_weight
            weight_total += local_weight
        module_mean = (weighted_sum / weight_total) if weight_total > 0 else 0.0
        module_signal = module_mean * module_conf

        module_stats[module_id] = {
            "module_id": module_id,
            "module_name": MODULE_NAMES.get(module_id, f"Module {module_id}"),
            "items": list(module_items),
            "observed_items": observed_items,
            "observed_item_count": observed_item_count,
            "coverage": round(coverage, 6),
            "avg_support": round(avg_support, 6),
            "support_strength": round(support_strength, 6),
            "module_conf": round(module_conf, 6),
            "module_mean": round(module_mean, 6),
            "module_signal": round(module_signal, 6),
        }
    return module_stats



def _impute_missing_item_score(
    item_id: int,
    module_stats: Dict[int, Dict[str, float | int | List[int]]],
) -> Tuple[float, List[Dict[str, float | int | str | bool]]]:
    candidates = ITEM_TO_MODULES.get(item_id, [])
    contributions: List[Dict[str, float | int | str | bool]] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for module_id in candidates:
        module_row = module_stats.get(module_id)
        if not module_row:
            continue
        module_conf = float(module_row.get("module_conf", 0.0))
        module_signal = float(module_row.get("module_signal", 0.0))
        module_weight = float(MODULE_WEIGHTS.get(module_id, 1.0)) * module_conf
        if module_weight <= 0:
            continue
        weighted_sum += module_signal * module_weight
        weight_total += module_weight
        contributions.append(
            {
                "module_id": module_id,
                "module_name": MODULE_NAMES.get(module_id, f"Module {module_id}"),
                "base_weight": round(float(MODULE_WEIGHTS.get(module_id, 1.0)), 6),
                "weight": round(module_weight, 6),
                "module_signal": round(module_signal, 6),
                "module_conf": round(module_conf, 6),
                "module_mean": round(float(module_row.get("module_mean", 0.0)), 6),
                "coverage": round(float(module_row.get("coverage", 0.0)), 6),
                "observed_item_count": int(module_row.get("observed_item_count", 0) or 0),
                "eligible": bool(module_weight > 0.0 and module_signal > 0.0),
            }
        )

    if weight_total > 0:
        imputed_float = weighted_sum / weight_total
    else:
        imputed_float = 0.0

    return _clamp(imputed_float, 0.0, 3.0), contributions


def _support_geometry_from_state(
    item_beliefs: Dict[int, ItemBelief],
    *,
    evidence_rows: List[object],
    risk_reason: str,
) -> Dict[str, object]:
    evidence_turns_by_item: Dict[int, set[int]] = {item_id: set() for item_id in range(1, 22)}
    evidence_methods_by_item: Dict[int, set[str]] = {item_id: set() for item_id in range(1, 22)}
    for row in evidence_rows:
        item_id = int(getattr(row, "item_id", 0) or 0)
        if item_id < 1 or item_id > 21 or bool(getattr(row, "support_increment_blocked", False)):
            continue
        turn = int(getattr(row, "turn", 0) or 0)
        if turn > 0:
            evidence_turns_by_item[item_id].add(turn)
        method = str(getattr(row, "method", "") or "")
        if method:
            evidence_methods_by_item[item_id].add(method)

    observed_supported_item_ids = [
        item_id for item_id in range(1, 22) if int(item_beliefs[item_id].support_count) > 0
    ]
    observed_positive_item_ids = [
        item_id
        for item_id in range(1, 22)
        if int(item_beliefs[item_id].support_count) > 0 and float(item_beliefs[item_id].expected_score) >= 1.0
    ]
    observed_core_item_ids = [
        item_id
        for item_id in CORE_ITEM_IDS
        if int(item_beliefs[item_id].support_count) > 0 and float(item_beliefs[item_id].expected_score) >= 1.0
    ]
    observed_positive_item_set = set(observed_positive_item_ids)

    item_support_geometry: Dict[str, Dict[str, object]] = {}
    corroborated_item_ids: List[int] = []
    corroborated_affective_cognitive_module_ids: List[int] = []

    for item_id in range(1, 22):
        same_module_supported_item_ids = sorted(
            {
                other_item_id
                for module_id in ITEM_TO_MODULES.get(item_id, [])
                for other_item_id in MODULE_TO_ITEMS.get(module_id, [])
                if other_item_id != item_id and other_item_id in observed_positive_item_set
            }
        )
        support_count = int(item_beliefs[item_id].support_count)
        expected_score = float(item_beliefs[item_id].expected_score)
        evidence_turn_count = len(evidence_turns_by_item[item_id])
        evidence_method_count = len(evidence_methods_by_item[item_id])
        same_module_corroborated_item_ids = []
        if support_count >= 1:
            for other_item_id in same_module_supported_item_ids:
                other_support_count = int(item_beliefs[other_item_id].support_count)
                other_expected_score = float(item_beliefs[other_item_id].expected_score)
                other_evidence_turn_count = len(evidence_turns_by_item[other_item_id])
                if other_support_count < 1 or other_evidence_turn_count < 1:
                    continue
                if (
                    support_count >= 2
                    or expected_score >= 1.5
                    or other_support_count >= 2
                    or other_expected_score >= 1.5
                ):
                    same_module_corroborated_item_ids.append(other_item_id)
        is_corroborated_item = False
        if item_id == 9:
            if risk_reason in {"active self-harm cue match", "multiple passive death ideation cues"}:
                is_corroborated_item = True
        elif support_count >= 2 and evidence_turn_count >= 2:
            is_corroborated_item = True
        elif support_count >= 1 and same_module_corroborated_item_ids:
            is_corroborated_item = True

        if is_corroborated_item:
            corroborated_item_ids.append(item_id)

        item_support_geometry[str(item_id)] = {
            "evidence_turn_count": evidence_turn_count,
            "evidence_method_count": evidence_method_count,
            "same_module_supported_item_count": len(same_module_supported_item_ids),
            "same_module_supported_item_ids": same_module_supported_item_ids,
            "same_module_corroborated_item_count": len(same_module_corroborated_item_ids),
            "same_module_corroborated_item_ids": same_module_corroborated_item_ids,
            "is_corroborated_item": is_corroborated_item,
        }

    corroborated_item_id_set = set(corroborated_item_ids)
    corroborated_core_hits = sum(1 for item_id in CORE_ITEM_IDS if item_id in corroborated_item_id_set)
    for module_id in sorted(LOW_SIGNAL_COGNITIVE_AFFECTIVE_MODULES):
        module_items = MODULE_TO_ITEMS.get(module_id, [])
        if any(item_id in corroborated_item_id_set for item_id in module_items):
            corroborated_affective_cognitive_module_ids.append(module_id)

    total_observed_support_count = sum(int(item_beliefs[item_id].support_count) for item_id in observed_supported_item_ids)
    dominant_support_item_id = 0
    dominant_support_count = 0
    if observed_supported_item_ids:
        dominant_support_item_id = min(
            observed_supported_item_ids,
            key=lambda item_id: (-int(item_beliefs[item_id].support_count), item_id),
        )
        dominant_support_count = int(item_beliefs[dominant_support_item_id].support_count)
    dominant_support_share = (
        float(dominant_support_count) / float(total_observed_support_count)
        if total_observed_support_count > 0
        else 0.0
    )
    support_concentration_dominant = (
        dominant_support_share >= 0.40
        or sum(1 for item_id in observed_supported_item_ids if int(item_beliefs[item_id].support_count) >= 2) == 1
    )

    return {
        "observed_supported_item_ids": observed_supported_item_ids,
        "observed_positive_item_ids": observed_positive_item_ids,
        "observed_core_item_ids": observed_core_item_ids,
        "item_support_geometry": item_support_geometry,
        "corroborated_item_ids": sorted(corroborated_item_ids),
        "corroborated_core_hits": corroborated_core_hits,
        "corroborated_affective_cognitive_module_ids": corroborated_affective_cognitive_module_ids,
        "corroborated_affective_cognitive_module_breadth": len(corroborated_affective_cognitive_module_ids),
        "dominant_support_item_id": dominant_support_item_id,
        "dominant_support_share": round(dominant_support_share, 6),
        "support_concentration_dominant": bool(support_concentration_dominant),
        "total_observed_support_count": total_observed_support_count,
    }


def _low_signal_guardrail_context(
    item_beliefs: Dict[int, ItemBelief],
    *,
    evidence_rows: List[object],
    risk_reason: str,
) -> Dict[str, object]:
    support_geometry = _support_geometry_from_state(
        item_beliefs,
        evidence_rows=evidence_rows,
        risk_reason=risk_reason,
    )
    observed_positive_item_ids = list(support_geometry["observed_positive_item_ids"])
    observed_core_item_ids = list(support_geometry["observed_core_item_ids"])
    strong_observed_item_ids = [
        item_id
        for item_id in range(1, 22)
        if int(item_beliefs[item_id].support_count) >= 2 and float(item_beliefs[item_id].expected_score) >= 1.75
    ]
    strong_module_ids = sorted(
        module_id
        for module_id, module_items in MODULE_TO_ITEMS.items()
        if any(
            int(item_beliefs[item_id].support_count) > 0 and float(item_beliefs[item_id].expected_score) >= 1.5
            for item_id in module_items
        )
    )
    affective_cognitive_corroboration = any(
        module_id in LOW_SIGNAL_COGNITIVE_AFFECTIVE_MODULES
        for module_id, module_items in MODULE_TO_ITEMS.items()
        if any(
            int(item_beliefs[item_id].support_count) > 0 and float(item_beliefs[item_id].expected_score) >= 1.0
            for item_id in module_items
        )
    )

    observed_positive_breadth = len(observed_positive_item_ids)
    observed_core_hits = len(observed_core_item_ids)
    strong_observed_item_count = len(strong_observed_item_ids)
    strong_module_breadth = len(strong_module_ids)
    corroborated_core_hits = int(support_geometry["corroborated_core_hits"])
    corroborated_affective_cognitive_module_breadth = int(
        support_geometry["corroborated_affective_cognitive_module_breadth"]
    )
    support_concentration_dominant = bool(support_geometry["support_concentration_dominant"])
    bypass_conditions = [
        ("observed_positive_breadth>=4", observed_positive_breadth >= 4, observed_positive_breadth),
        ("corroborated_core_hits>=2", corroborated_core_hits >= 2, corroborated_core_hits),
        (
            "corroborated_affective_cognitive_module_breadth>=2",
            corroborated_affective_cognitive_module_breadth >= 2,
            corroborated_affective_cognitive_module_breadth,
        ),
        ("support_concentration_dominant==False", not support_concentration_dominant, support_concentration_dominant),
    ]
    support_geometry_candidate_bypass = all(passed for _, passed, _ in bypass_conditions)
    support_geometry_bypass_reasons = [f"{label} ({value})" for label, _, value in bypass_conditions]

    return {
        "support_geometry_candidate_bypass": support_geometry_candidate_bypass,
        "support_geometry_bypass_reasons": support_geometry_bypass_reasons,
        "item_support_geometry": support_geometry["item_support_geometry"],
        "observed_positive_item_ids": observed_positive_item_ids,
        "observed_positive_breadth": observed_positive_breadth,
        "observed_supported_item_ids": list(support_geometry["observed_supported_item_ids"]),
        "observed_core_item_ids": observed_core_item_ids,
        "observed_core_hits": observed_core_hits,
        "strong_observed_item_ids": strong_observed_item_ids,
        "strong_observed_item_count": strong_observed_item_count,
        "strong_module_ids": strong_module_ids,
        "strong_module_breadth": strong_module_breadth,
        "affective_cognitive_corroboration": affective_cognitive_corroboration,
        "corroborated_item_ids": list(support_geometry["corroborated_item_ids"]),
        "corroborated_core_hits": corroborated_core_hits,
        "corroborated_affective_cognitive_module_ids": list(
            support_geometry["corroborated_affective_cognitive_module_ids"]
        ),
        "corroborated_affective_cognitive_module_breadth": corroborated_affective_cognitive_module_breadth,
        "dominant_support_item_id": int(support_geometry["dominant_support_item_id"]),
        "dominant_support_share": float(support_geometry["dominant_support_share"]),
        "support_concentration_dominant": support_concentration_dominant,
        "total_observed_support_count": int(support_geometry["total_observed_support_count"]),
    }


def _low_signal_imputed_budget(observed_positive_breadth: int, observed_core_hits: int) -> int:
    if observed_positive_breadth <= 1:
        return 0
    if observed_positive_breadth in {2, 3}:
        return 1
    if observed_positive_breadth == 4 and observed_core_hits == 1:
        return 2
    return 1


def _risk_reason_from_state(state: AgentState) -> str:
    risk_state = state.get("risk")
    if risk_state is not None:
        return str(getattr(risk_state, "reason", "") or "")

    turn_trace = state.get("turn_trace", {})
    if isinstance(turn_trace, dict):
        risk_trace = turn_trace.get("risk_sentinel", {})
        if isinstance(risk_trace, dict):
            return str(risk_trace.get("reason", "") or "")
    return ""


def _persona_utterances(state: AgentState) -> List[str]:
    utterances: List[str] = []
    for message in list(state.get("messages", [])):
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).strip().lower() != "assistant":
            continue
        content = str(message.get("content", "") or "").strip()
        if content:
            utterances.append(content)
    return utterances


def _severe_anchor_context(
    state: AgentState,
    *,
    low_signal_context: Dict[str, object],
) -> Dict[str, object]:
    strong_anchor_module_ids: set[int] = set()
    severe_anchor_module_ids: set[int] = set()
    severe_anchor_item_ids: set[int] = set()

    for utterance in _persona_utterances(state):
        normalized = _normalize_text(utterance)
        if not normalized or _contains_any(normalized, SEVERE_ANCHOR_NEGATION_PHRASES):
            continue
        for rule in SEVERE_ANCHOR_RULES:
            module_id = int(rule["module_id"])
            strong_hit = _contains_any(normalized, list(rule.get("strong_phrases", [])))
            mild_hit = _contains_any(normalized, list(rule.get("mild_phrases", [])))
            if not strong_hit and not mild_hit:
                continue
            severe_anchor_module_ids.add(module_id)
            severe_anchor_item_ids.update(int(item_id) for item_id in list(rule.get("item_ids", [])))
            if strong_hit:
                strong_anchor_module_ids.add(module_id)

    corroborated_nonrisk_item_ids = [
        int(item_id)
        for item_id in list(low_signal_context.get("corroborated_item_ids", []))
        if int(item_id) != 9
    ]
    corroborated_nonrisk_module_ids = sorted(
        {
            module_id
            for item_id in corroborated_nonrisk_item_ids
            for module_id in ITEM_TO_MODULES.get(item_id, [])
            if module_id != 9
        }
    )
    cognitive_affective_anchor = bool(strong_anchor_module_ids.intersection(LOW_SIGNAL_COGNITIVE_AFFECTIVE_MODULES))
    severe_recovery_mode_active = False
    severe_recovery_reason = ""
    if len(strong_anchor_module_ids) >= 2 and cognitive_affective_anchor:
        severe_recovery_mode_active = True
        severe_recovery_reason = "multiple_strong_anchor_modules"
    elif (
        len(corroborated_nonrisk_item_ids) >= 3
        and len(corroborated_nonrisk_module_ids) >= 2
        and len(strong_anchor_module_ids) >= 1
    ):
        severe_recovery_mode_active = True
        severe_recovery_reason = "corroborated_observed_plus_strong_anchor"

    return {
        "severe_recovery_mode_active": severe_recovery_mode_active,
        "severe_anchor_item_ids": sorted(severe_anchor_item_ids),
        "severe_anchor_module_ids": sorted(severe_anchor_module_ids),
        "severe_strong_anchor_module_ids": sorted(strong_anchor_module_ids),
        "severe_recovery_reason": severe_recovery_reason,
        "corroborated_nonrisk_item_ids": corroborated_nonrisk_item_ids,
        "corroborated_nonrisk_module_ids": corroborated_nonrisk_module_ids,
    }


def _item_evidence_summary(
    evidence_rows: List[object],
) -> Dict[int, Dict[str, object]]:
    turns_by_item: Dict[int, set[int]] = {item_id: set() for item_id in range(1, 22)}
    summary_by_item: Dict[int, Dict[str, object]] = {
        item_id: {
            "evidence_row_count": 0,
            "evidence_turn_count": 0,
            "max_confidence": 0.0,
            "max_intensity": 0.0,
            "llm_extractor_row_count": 0,
            "has_strong_row": False,
            "has_very_strong_row": False,
        }
        for item_id in range(1, 22)
    }

    for row in evidence_rows:
        item_id = int(getattr(row, "item_id", 0) or 0)
        if item_id < 1 or item_id > 21 or bool(getattr(row, "support_increment_blocked", False)):
            continue

        summary = summary_by_item[item_id]
        summary["evidence_row_count"] = int(summary["evidence_row_count"]) + 1

        turn = int(getattr(row, "turn", 0) or 0)
        if turn > 0:
            turns_by_item[item_id].add(turn)

        confidence = _clamp(float(getattr(row, "confidence", 0.0) or 0.0), 0.0, 1.0)
        intensity = _clamp(float(getattr(row, "intensity", 0.0) or 0.0), 0.0, 3.0)
        summary["max_confidence"] = max(float(summary["max_confidence"]), confidence)
        summary["max_intensity"] = max(float(summary["max_intensity"]), intensity)

        method = str(getattr(row, "method", "") or "").strip().lower()
        if method == "llm_extractor":
            summary["llm_extractor_row_count"] = int(summary["llm_extractor_row_count"]) + 1

        if confidence >= 0.55 and intensity >= 1.5:
            summary["has_strong_row"] = True
        if confidence >= 0.65 and intensity >= 1.75:
            summary["has_very_strong_row"] = True

    for item_id in range(1, 22):
        summary_by_item[item_id]["evidence_turn_count"] = len(turns_by_item[item_id])

    return summary_by_item



def _evidence_report(state: AgentState) -> Dict[str, object]:
    evidence_rows = list(state.get("evidence_log", []))
    top_rows: List[Dict[str, object]] = []
    for row in evidence_rows[-6:]:
        item_id = int(getattr(row, "item_id", 0) or 0)
        top_rows.append(
            {
                "item_id": item_id,
                "symptom_name": str(getattr(row, "symptom_name", "") or ""),
                "confidence": float(getattr(row, "confidence", 0.0) or 0.0),
                "intensity": float(getattr(row, "intensity", 0.0) or 0.0),
                "method": str(getattr(row, "method", "") or ""),
                "evidence_text": str(getattr(row, "evidence_text", "") or ""),
            }
        )
    return {
        "evidence_count": len(evidence_rows),
        "recent_evidence": top_rows,
    }


def _rank_key_items(final_item_scores: Dict[int, int], item_details: Dict[str, Dict[str, object]], limit: int = 4) -> List[int]:
    def _rank_key(item_id: int) -> tuple[int, int, int, int]:
        score = int(final_item_scores.get(item_id, 0))
        detail = item_details.get(str(item_id), {})
        source = str(detail.get("source", "imputed"))
        observed_rank = 0 if source in {"observed", "observed_blended"} else 1
        support_count = int(detail.get("support_count", 0) or 0)
        return (-score, observed_rank, -support_count, item_id)

    candidates = [int(item_id) for item_id, score in final_item_scores.items() if int(score) > 0]
    candidates.sort(key=_rank_key)
    return candidates[:limit]



def finalize_outputs(state: AgentState) -> Dict:
    control = state.get("control")
    control_stop = bool(getattr(control, "stop", False))
    control_reason = str(getattr(control, "stop_reason", "") or "")
    should_finalize = control_stop

    turn_trace = dict(state.get("turn_trace", {}))
    final_trace = {
        "turn": int(state.get("turn_index", 0)),
        "ran_final_imputation": bool(should_finalize),
        "control_stop": control_stop,
        "reason": control_reason if control_reason else "continue",
    }

    if not should_finalize:
        final_state = FinalState(
            predicted_bdi_score=int(state.get("predicted_bdi_score") or state.get("raw_predicted_bdi_score") or 0),
            predicted_label=str(state.get("predicted_label") or state.get("raw_predicted_label") or "control"),
            top_symptoms=list(state.get("predicted_key_symptoms") or []),
            evidence_report=_evidence_report(state),
            risk_flag=bool(state.get("risk_flag", False)),
            debug_trace=final_trace,
        )
        turn_trace["finalize_outputs"] = final_trace
        return {
            "final": final_state,
            "turn_trace": turn_trace,
        }

    raw_predicted_bdi_score = int(state.get("raw_predicted_bdi_score") or state.get("predicted_bdi_score") or 0)
    raw_predicted_label = str(state.get("raw_predicted_label") or state.get("predicted_label") or "control")
    risk_flag = bool(state.get("risk_flag", False))
    bdi_threshold = int(os.getenv("DETERMINISTIC_BDI_LABEL_THRESHOLD", "14"))
    evidence_rows = list(state.get("evidence_log", []))
    risk_reason = _risk_reason_from_state(state)

    prior_beliefs = state.get("item_beliefs", {})
    beliefs: Dict[int, ItemBelief] = {}
    for item_id in range(1, 22):
        beliefs[item_id] = coerce_item_belief(item_id, prior_beliefs.get(item_id))

    low_signal_context = _low_signal_guardrail_context(
        beliefs,
        evidence_rows=evidence_rows,
        risk_reason=risk_reason,
    )
    item_evidence_summary = _item_evidence_summary(evidence_rows)
    severe_anchor_context = _severe_anchor_context(
        state,
        low_signal_context=low_signal_context,
    )
    severe_recovery_mode_active = bool(severe_anchor_context["severe_recovery_mode_active"])
    support_geometry_candidate_bypass = bool(low_signal_context["support_geometry_candidate_bypass"])
    anchor_gated_guardrail_blocked = support_geometry_candidate_bypass and not severe_recovery_mode_active
    low_signal_guardrail_active = not severe_recovery_mode_active
    guardrail_bypass_source = "severe_recovery" if severe_recovery_mode_active else "none"
    module_stats = _module_stats_from_beliefs(beliefs)
    final_item_scores: Dict[int, int] = {}
    item_details: Dict[str, Dict[str, object]] = {}
    imputed_item_count = 0
    blended_observed_item_count = 0
    blended_item_ids: List[int] = []
    low_signal_observed_cap_item_ids: List[int] = []
    low_signal_singleton_trimmed_item_ids: List[int] = []
    low_signal_item9_cap_reason = ""
    severe_recovered_item_ids: List[int] = []
    severe_amplitude_observed_item_ids: List[int] = []
    severe_amplitude_imputed_item_ids: List[int] = []
    severe_item9_rescued = False
    severe_anchor_item_id_set = {int(item_id) for item_id in severe_anchor_context["severe_anchor_item_ids"]}
    severe_anchor_module_id_set = {int(module_id) for module_id in severe_anchor_context["severe_anchor_module_ids"]}
    severe_strong_anchor_module_id_set = {
        int(module_id) for module_id in severe_anchor_context["severe_strong_anchor_module_ids"]
    }

    obs_blend_enabled = _env_bool("FINAL_OBS_BLEND_ENABLED", "1")
    obs_blend_conf_threshold = _clamp(_env_float("FINAL_OBS_BLEND_CONF_THRESHOLD", "0.60"), 0.0, 1.0)
    obs_blend_support_max = max(1, int(_env_float("FINAL_OBS_BLEND_SUPPORT_MAX", "2")))
    obs_blend_module_conf_min = _clamp(_env_float("FINAL_OBS_BLEND_MODULE_CONF_MIN", "0.50"), 0.0, 1.0)
    obs_blend_max_alpha = _clamp(_env_float("FINAL_OBS_BLEND_MAX_ALPHA", "0.35"), 0.0, 1.0)

    for item_id in range(1, 22):
        belief = beliefs[item_id]
        item_support_geometry = dict(low_signal_context["item_support_geometry"].get(str(item_id), {}))
        evidence_summary = dict(item_evidence_summary.get(item_id, {}))
        is_corroborated_item = bool(item_support_geometry.get("is_corroborated_item", False))
        severe_anchor_hit = int(item_id) in severe_anchor_item_id_set
        severe_anchor_modules = [
            int(module_id)
            for module_id in ITEM_TO_MODULES.get(item_id, [])
            if int(module_id) in severe_anchor_module_id_set
        ]
        if int(belief.support_count) > 0:
            observed_float = _clamp(float(belief.expected_score), 0.0, 3.0)
            observed_int = int(round(observed_float))
            observed_confidence = _clamp(1.0 - float(belief.uncertainty), 0.0, 1.0)
            support_count = int(belief.support_count)
            leave_one_out_stats = _module_stats_from_beliefs(beliefs, excluded_item_id=item_id)
            leave_one_out_estimate_float, leave_one_out_contributions = _impute_missing_item_score(item_id, leave_one_out_stats)
            has_leave_one_out_corroboration = any(
                bool(contribution.get("eligible", False)) or float(contribution.get("weight", 0.0) or 0.0) > 0.0
                for contribution in leave_one_out_contributions
            )
            module_estimate_float, contributions = _impute_missing_item_score(item_id, module_stats)
            best_module_conf = 0.0
            for contribution in contributions:
                try:
                    best_module_conf = max(best_module_conf, float(contribution.get("module_conf", 0.0)))
                except (TypeError, ValueError):
                    continue

            blend_applied = False
            blend_alpha = 0.0
            blend_reason = "high_conf_kept"
            final_float = observed_float

            if not obs_blend_enabled:
                blend_reason = "blend_disabled"
            elif not contributions:
                blend_reason = "no_module_signal"
            elif observed_confidence >= obs_blend_conf_threshold:
                blend_reason = "high_conf_kept"
            elif support_count > obs_blend_support_max:
                blend_reason = "high_support_kept"
            elif best_module_conf < obs_blend_module_conf_min:
                blend_reason = "low_module_conf_kept"
            else:
                confidence_gap = (obs_blend_conf_threshold - observed_confidence) / max(obs_blend_conf_threshold, 1e-6)
                confidence_gap = _clamp(confidence_gap, 0.0, 1.0)
                support_factor = _clamp(
                    float((obs_blend_support_max + 1) - support_count) / float(max(1, obs_blend_support_max)),
                    0.0,
                    1.0,
                )
                module_factor = _clamp(best_module_conf, 0.0, 1.0)
                blend_alpha = _clamp(confidence_gap * support_factor * module_factor, 0.0, obs_blend_max_alpha)
                if blend_alpha > 0.0:
                    final_float = _clamp(
                        ((1.0 - blend_alpha) * observed_float) + (blend_alpha * module_estimate_float),
                        0.0,
                        3.0,
                    )
                    blend_applied = True
                    blend_reason = "low_conf_blended"
                else:
                    blend_reason = "low_conf_blend_zero_alpha"

            pre_trim_score = _clamp(final_float, 0.0, 3.0)
            final_int = _round_item_score(pre_trim_score)
            severe_amplitude_observed_applied = False
            anchored_observed_severe = (
                severe_recovery_mode_active
                and item_id != 9
                and support_count >= 1
                and (
                    item_id in severe_anchor_item_id_set
                    or any(module_id in severe_strong_anchor_module_id_set for module_id in ITEM_TO_MODULES.get(item_id, []))
                )
                and int(evidence_summary.get("evidence_turn_count", 0) or 0) >= 1
                and (
                    observed_float >= 1.6
                    or bool(evidence_summary.get("has_strong_row", False))
                    or support_count >= 2
                )
            )
            if anchored_observed_severe:
                pre_trim_score = max(pre_trim_score, observed_float)
                severe_candidate_int = min(2, _round_item_score(pre_trim_score))
                if severe_candidate_int > final_int:
                    final_int = severe_candidate_int
                    severe_amplitude_observed_applied = True
                    severe_amplitude_observed_item_ids.append(item_id)
            low_signal_observed_cap_applied = False
            item9_guardrail_applied = False
            if item_id == 9 and (low_signal_guardrail_active or severe_recovery_mode_active):
                item9_guardrail_applied = True
                if risk_reason == "active self-harm cue match":
                    final_int = min(final_int, 2)
                    low_signal_item9_cap_reason = "active_self_harm_normal_rounding"
                elif risk_reason == "multiple passive death ideation cues":
                    final_int = min(max(final_int, 1), 1)
                    low_signal_item9_cap_reason = "multiple_passive_capped_at_one"
                elif risk_reason == "passive death ideation cue":
                    if int(low_signal_context["corroborated_core_hits"]) >= 1:
                        final_int = min(max(1, final_int), 1)
                        low_signal_item9_cap_reason = "passive_with_corroborated_core_capped_at_one"
                    else:
                        final_int = 0
                        low_signal_item9_cap_reason = "passive_without_corroborated_core_forced_zero"
                else:
                    final_int = 0
                    low_signal_item9_cap_reason = "non_specific_low_signal_forced_zero"
            elif low_signal_guardrail_active:
                if (not severe_recovery_mode_active) and (
                    support_count == 1
                    and not is_corroborated_item
                ):
                    if pre_trim_score < 1.5:
                        trimmed_int = 0
                    else:
                        trimmed_int = 1
                    if trimmed_int < final_int:
                        low_signal_observed_cap_applied = True
                        low_signal_singleton_trimmed_item_ids.append(item_id)
                        low_signal_observed_cap_item_ids.append(item_id)
                    final_int = trimmed_int
                elif (not severe_recovery_mode_active) and final_int >= 2 and not is_corroborated_item:
                    final_int = 1
                    low_signal_observed_cap_applied = True
                    low_signal_observed_cap_item_ids.append(item_id)

            final_item_scores[item_id] = final_int
            source = "observed_blended" if blend_applied else "observed"
            if blend_applied:
                blended_observed_item_count += 1
                blended_item_ids.append(item_id)
            item_details[str(item_id)] = {
                "source": source,
                "support_count": support_count,
                "expected_score": round(float(belief.expected_score), 6),
                "observed_confidence": round(observed_confidence, 6),
                "observed_int": observed_int,
                "item_evidence_summary": {
                    "evidence_row_count": int(evidence_summary.get("evidence_row_count", 0) or 0),
                    "evidence_turn_count": int(evidence_summary.get("evidence_turn_count", 0) or 0),
                    "max_confidence": round(float(evidence_summary.get("max_confidence", 0.0) or 0.0), 6),
                    "max_intensity": round(float(evidence_summary.get("max_intensity", 0.0) or 0.0), 6),
                    "llm_extractor_row_count": int(evidence_summary.get("llm_extractor_row_count", 0) or 0),
                    "has_strong_row": bool(evidence_summary.get("has_strong_row", False)),
                    "has_very_strong_row": bool(evidence_summary.get("has_very_strong_row", False)),
                },
                "evidence_row_count": int(evidence_summary.get("evidence_row_count", 0) or 0),
                "module_estimate_float": round(float(module_estimate_float), 6),
                "best_module_conf": round(float(best_module_conf), 6),
                "leave_one_out_estimate_float": round(float(leave_one_out_estimate_float), 6),
                "leave_one_out_has_corroboration": bool(has_leave_one_out_corroboration),
                "evidence_turn_count": int(evidence_summary.get("evidence_turn_count", 0) or 0),
                "evidence_method_count": int(item_support_geometry.get("evidence_method_count", 0) or 0),
                "max_confidence": round(float(evidence_summary.get("max_confidence", 0.0) or 0.0), 6),
                "max_intensity": round(float(evidence_summary.get("max_intensity", 0.0) or 0.0), 6),
                "has_strong_row": bool(evidence_summary.get("has_strong_row", False)),
                "has_very_strong_row": bool(evidence_summary.get("has_very_strong_row", False)),
                "same_module_supported_item_count": int(
                    item_support_geometry.get("same_module_supported_item_count", 0) or 0
                ),
                "same_module_supported_item_ids": list(
                    item_support_geometry.get("same_module_supported_item_ids", [])
                ),
                "same_module_corroborated_item_count": int(
                    item_support_geometry.get("same_module_corroborated_item_count", 0) or 0
                ),
                "same_module_corroborated_item_ids": list(
                    item_support_geometry.get("same_module_corroborated_item_ids", [])
                ),
                "is_corroborated_item": bool(is_corroborated_item),
                "severe_anchor_hit": bool(severe_anchor_hit),
                "severe_anchor_modules": severe_anchor_modules,
                "severe_recovery_mode_active": severe_recovery_mode_active,
                "pre_trim_score": round(pre_trim_score, 6),
                "post_trim_score": final_int,
                "weak_positive_trim_applied": bool(low_signal_observed_cap_applied or item9_guardrail_applied),
                "low_signal_singleton_trim_applied": bool(support_count == 1 and low_signal_observed_cap_applied),
                "low_signal_observed_cap_applied": bool(low_signal_observed_cap_applied),
                "low_signal_item9_guardrail_applied": bool(item9_guardrail_applied),
                "low_signal_item9_cap_reason": low_signal_item9_cap_reason if item_id == 9 else "",
                "low_signal_guardrail_active": low_signal_guardrail_active,
                "anchored_observed_severe": bool(anchored_observed_severe),
                "severe_amplitude_observed_applied": bool(severe_amplitude_observed_applied),
                "blend_alpha": round(float(blend_alpha), 6),
                "blend_applied": bool(blend_applied),
                "blend_reason": blend_reason,
                "final_score": final_item_scores[item_id],
                "candidate_modules": ITEM_TO_MODULES.get(item_id, []),
                "contributions": contributions,
                "leave_one_out_contributions": leave_one_out_contributions,
            }
            continue

        imputed_float, contributions = _impute_missing_item_score(item_id, module_stats)
        final_item_scores[item_id] = _round_item_score(imputed_float)
        if final_item_scores[item_id] > 0:
            imputed_item_count += 1
        item_details[str(item_id)] = {
            "source": "imputed",
            "support_count": 0,
            "imputed_float": round(float(imputed_float), 6),
            "item_evidence_summary": {
                "evidence_row_count": int(evidence_summary.get("evidence_row_count", 0) or 0),
                "evidence_turn_count": int(evidence_summary.get("evidence_turn_count", 0) or 0),
                "max_confidence": round(float(evidence_summary.get("max_confidence", 0.0) or 0.0), 6),
                "max_intensity": round(float(evidence_summary.get("max_intensity", 0.0) or 0.0), 6),
                "llm_extractor_row_count": int(evidence_summary.get("llm_extractor_row_count", 0) or 0),
                "has_strong_row": bool(evidence_summary.get("has_strong_row", False)),
                "has_very_strong_row": bool(evidence_summary.get("has_very_strong_row", False)),
            },
            "observed_confidence": None,
            "evidence_row_count": int(evidence_summary.get("evidence_row_count", 0) or 0),
            "evidence_turn_count": int(evidence_summary.get("evidence_turn_count", 0) or 0),
            "evidence_method_count": int(item_support_geometry.get("evidence_method_count", 0) or 0),
            "max_confidence": round(float(evidence_summary.get("max_confidence", 0.0) or 0.0), 6),
            "max_intensity": round(float(evidence_summary.get("max_intensity", 0.0) or 0.0), 6),
            "has_strong_row": bool(evidence_summary.get("has_strong_row", False)),
            "has_very_strong_row": bool(evidence_summary.get("has_very_strong_row", False)),
            "same_module_supported_item_count": int(item_support_geometry.get("same_module_supported_item_count", 0) or 0),
            "same_module_supported_item_ids": list(item_support_geometry.get("same_module_supported_item_ids", [])),
            "same_module_corroborated_item_count": int(
                item_support_geometry.get("same_module_corroborated_item_count", 0) or 0
            ),
            "same_module_corroborated_item_ids": list(
                item_support_geometry.get("same_module_corroborated_item_ids", [])
            ),
            "is_corroborated_item": False,
            "severe_anchor_hit": bool(severe_anchor_hit),
            "severe_anchor_modules": severe_anchor_modules,
            "severe_recovery_mode_active": severe_recovery_mode_active,
            "module_estimate_float": round(float(imputed_float), 6),
            "best_module_conf": round(
                max((float(c.get("module_conf", 0.0)) for c in contributions), default=0.0),
                6,
            ),
            "pre_trim_score": round(float(imputed_float), 6),
            "post_trim_score": final_item_scores[item_id],
            "weak_positive_trim_applied": False,
            "low_signal_observed_cap_applied": False,
            "low_signal_item9_cap_reason": "",
            "low_signal_guardrail_active": low_signal_guardrail_active,
            "severe_amplitude_imputed_applied": False,
            "blend_alpha": 0.0,
            "blend_applied": False,
            "blend_reason": "missing_imputed",
            "final_score": final_item_scores[item_id],
            "candidate_modules": ITEM_TO_MODULES.get(item_id, []),
            "contributions": contributions,
        }

    imputed_points_before_guardrail = sum(
        int(final_item_scores[item_id])
        for item_id in range(1, 22)
        if str(item_details.get(str(item_id), {}).get("source", "")) == "imputed"
    )
    suppressed_imputed_item_ids: List[int] = []
    somatic_corroboration_blocked_item_ids: List[int] = []
    imputed_point_budget: int | None = None

    if low_signal_guardrail_active:
        imputed_point_budget = _low_signal_imputed_budget(
            int(low_signal_context["observed_positive_breadth"]),
            int(low_signal_context["observed_core_hits"]),
        )
        for item_id in range(1, 22):
            detail = item_details.get(str(item_id), {})
            if str(detail.get("source", "")) != "imputed":
                continue
            detail["low_signal_somatic_blocked"] = False
            detail["low_signal_budget_suppressed"] = False
            if final_item_scores[item_id] > 0:
                final_item_scores[item_id] = min(final_item_scores[item_id], 1)

        for item_id in range(1, 22):
            detail = item_details.get(str(item_id), {})
            if str(detail.get("source", "")) != "imputed" or final_item_scores[item_id] <= 0:
                continue
            contributions = list(detail.get("contributions", []))
            eligible_contributions = [
                contribution
                for contribution in contributions
                if bool(contribution.get("eligible", False)) or float(contribution.get("weight", 0.0) or 0.0) > 0.0
            ]
            eligible_module_ids = {int(contribution.get("module_id", 0) or 0) for contribution in eligible_contributions}
            only_somatic_interpersonal = bool(eligible_module_ids) and eligible_module_ids.issubset(
                LOW_SIGNAL_SOMATIC_INTERPERSONAL_MODULES
            )
            same_source_has_depth = any(
                int(module_stats.get(module_id, {}).get("observed_item_count", 0) or 0) >= 2
                for module_id in eligible_module_ids
            )
            if only_somatic_interpersonal and not bool(low_signal_context["affective_cognitive_corroboration"]) and not same_source_has_depth:
                final_item_scores[item_id] = 0
                detail["low_signal_somatic_blocked"] = True
                detail["post_trim_score"] = 0
                detail["final_score"] = 0
                somatic_corroboration_blocked_item_ids.append(item_id)
                suppressed_imputed_item_ids.append(item_id)

        ranked_imputed_candidates: List[Tuple[float, int, float, int]] = []
        for item_id in range(1, 22):
            detail = item_details.get(str(item_id), {})
            if str(detail.get("source", "")) != "imputed" or final_item_scores[item_id] <= 0:
                continue
            contributions = list(detail.get("contributions", []))
            eligible_contributions = [
                contribution
                for contribution in contributions
                if bool(contribution.get("eligible", False)) or float(contribution.get("weight", 0.0) or 0.0) > 0.0
            ]
            aggregate_weight = sum(float(contribution.get("weight", 0.0) or 0.0) for contribution in eligible_contributions)
            ranked_imputed_candidates.append(
                (
                    float(detail.get("imputed_float", 0.0) or 0.0),
                    len(eligible_contributions),
                    aggregate_weight,
                    item_id,
                )
            )

        ranked_imputed_candidates.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
        keep_item_ids = {item_id for _, _, _, item_id in ranked_imputed_candidates[:imputed_point_budget]}
        for _, _, _, item_id in ranked_imputed_candidates:
            detail = item_details[str(item_id)]
            if item_id in keep_item_ids:
                continue
            final_item_scores[item_id] = 0
            detail["low_signal_budget_suppressed"] = True
            detail["post_trim_score"] = 0
            detail["final_score"] = 0
            suppressed_imputed_item_ids.append(item_id)

    if severe_recovery_mode_active:
        strong_anchor_module_ids = set(severe_strong_anchor_module_id_set)
        anchored_module_ids = set(severe_anchor_module_id_set)
        recovery_module_ids = {
            module_id
            for module_id in anchored_module_ids
            if module_id in strong_anchor_module_ids
            or int(module_stats.get(module_id, {}).get("observed_item_count", 0) or 0) >= 1
        }
        for item_id in range(1, 22):
            detail = item_details.get(str(item_id), {})
            if str(detail.get("source", "")) != "imputed" or int(final_item_scores[item_id]) > 0:
                continue
            if float(detail.get("imputed_float", 0.0) or 0.0) <= 0.0:
                continue
            candidate_modules = [
                module_id for module_id in ITEM_TO_MODULES.get(item_id, []) if int(module_id) in recovery_module_ids
            ]
            if not candidate_modules:
                continue
            final_item_scores[item_id] = 1
            detail["severe_recovery_restored"] = True
            detail["severe_recovery_source_modules"] = candidate_modules
            severe_recovered_item_ids.append(item_id)

        eligible_strong_anchor_modules: Dict[int, Dict[str, float | int]] = {}
        for module_id in sorted(strong_anchor_module_ids):
            module_items = list(MODULE_TO_ITEMS.get(module_id, []))
            observed_supported_items = [
                item_id for item_id in module_items if int(beliefs[item_id].support_count) >= 1
            ]
            if not observed_supported_items:
                continue
            if not any(
                float(beliefs[item_id].expected_score) >= 1.75
                or int(beliefs[item_id].support_count) >= 2
                or bool(item_evidence_summary.get(item_id, {}).get("has_strong_row", False))
                for item_id in observed_supported_items
            ):
                continue
            eligible_strong_anchor_modules[module_id] = {
                "module_mean": float(module_stats.get(module_id, {}).get("module_mean", 0.0) or 0.0),
                "observed_item_count": int(module_stats.get(module_id, {}).get("observed_item_count", 0) or 0),
            }

        amplitude_candidates_by_module: Dict[int, List[Tuple[float, float, float, int]]] = {}
        for module_id in eligible_strong_anchor_modules:
            amplitude_candidates_by_module[module_id] = []

        for item_id in range(1, 22):
            detail = item_details.get(str(item_id), {})
            if str(detail.get("source", "")) != "imputed" or item_id == 9 or int(final_item_scores[item_id]) != 1:
                continue
            imputed_float = float(detail.get("imputed_float", 0.0) or 0.0)
            if imputed_float < 1.0:
                continue
            contribution_by_module = {
                int(contribution.get("module_id", 0) or 0): contribution
                for contribution in list(detail.get("contributions", []))
            }
            candidate_modules = [
                module_id
                for module_id in ITEM_TO_MODULES.get(item_id, [])
                if module_id in eligible_strong_anchor_modules
            ]
            for module_id in candidate_modules:
                contribution = contribution_by_module.get(module_id, {})
                module_mean = float(eligible_strong_anchor_modules[module_id]["module_mean"] or 0.0)
                aggregate_weight = float(contribution.get("weight", 0.0) or 0.0)
                amplitude_candidates_by_module[module_id].append(
                    (
                        imputed_float,
                        module_mean,
                        aggregate_weight,
                        item_id,
                    )
                )

        for module_id in sorted(amplitude_candidates_by_module):
            budget = 2 if module_id in {1, 3} else 1
            candidates = sorted(
                amplitude_candidates_by_module[module_id],
                key=lambda row: (-row[0], -row[1], -row[2], row[3]),
            )
            used = 0
            for _, _, _, item_id in candidates:
                if used >= budget:
                    break
                if int(final_item_scores[item_id]) != 1:
                    continue
                final_item_scores[item_id] = 2
                detail = item_details[str(item_id)]
                detail["severe_amplitude_imputed_applied"] = True
                detail["severe_amplitude_source_module"] = module_id
                severe_amplitude_imputed_item_ids.append(item_id)
                used += 1

        item9_detail = item_details.get("9", {})
        item9_summary = dict(item_evidence_summary.get(9, {}))
        item9_support_count = int(item9_detail.get("support_count", 0) or 0)
        item9_expected_score = float(beliefs[9].expected_score)
        if (
            int(final_item_scores.get(9, 0)) == 0
            and item9_support_count >= 1
            and int(item9_summary.get("evidence_turn_count", 0) or 0) >= 1
            and (
                bool(item9_summary.get("has_strong_row", False))
                or item9_expected_score >= 1.5
            )
            and int(low_signal_context["corroborated_core_hits"]) >= 2
        ):
            final_item_scores[9] = 1
            item9_detail["severe_item9_rescued"] = True
            severe_item9_rescued = True

    imputed_points_after_guardrail = sum(
        int(final_item_scores[item_id])
        for item_id in range(1, 22)
        if str(item_details.get(str(item_id), {}).get("source", "")) == "imputed"
    )
    imputed_item_count = sum(
        1
        for item_id in range(1, 22)
        if str(item_details.get(str(item_id), {}).get("source", "")) == "imputed" and int(final_item_scores[item_id]) > 0
    )
    suppressed_imputed_item_ids = sorted(set(suppressed_imputed_item_ids))
    somatic_corroboration_blocked_item_ids = sorted(set(somatic_corroboration_blocked_item_ids))
    low_signal_observed_cap_item_ids = sorted(set(low_signal_observed_cap_item_ids))
    low_signal_singleton_trimmed_item_ids = sorted(set(low_signal_singleton_trimmed_item_ids))
    severe_recovered_item_ids = sorted(set(severe_recovered_item_ids))
    severe_amplitude_observed_item_ids = sorted(set(severe_amplitude_observed_item_ids))
    severe_amplitude_imputed_item_ids = sorted(set(severe_amplitude_imputed_item_ids))

    for item_id in range(1, 22):
        detail = item_details.get(str(item_id), {})
        if not detail:
            continue
        detail["post_trim_score"] = int(final_item_scores[item_id])
        detail["final_score"] = int(final_item_scores[item_id])
        detail["severe_item9_rescued"] = bool(severe_item9_rescued) if item_id == 9 else False

    final_bdi_score = max(0, min(63, sum(int(final_item_scores[item_id]) for item_id in range(1, 22))))

    final_core_item_min_hits = int(os.getenv("FINAL_CORE_ITEM_MIN_HITS", "2"))
    final_core_signal_gate = float(os.getenv("FINAL_CORE_SIGNAL_GATE", "1.0"))
    core_hits = sum(1 for item_id in CORE_ITEM_IDS if int(final_item_scores.get(item_id, 0)) >= 1)
    module_signal_total = sum(float(module_row.get("module_signal", 0.0)) for module_row in module_stats.values())

    depression_from_bdi = final_bdi_score >= bdi_threshold
    depression_from_core_coverage = (
        core_hits >= max(1, final_core_item_min_hits) and module_signal_total >= final_core_signal_gate
    )
    final_label = "depressed" if (depression_from_bdi or risk_flag or depression_from_core_coverage) else "control"
    ranked_key_item_ids = _rank_key_items(final_item_scores, item_details=item_details, limit=4)
    if ranked_key_item_ids:
        final_key_symptoms = [symptom_name_from_item(item_id) for item_id in ranked_key_item_ids]
    else:
        final_key_symptoms = top_symptoms_from_scores(final_item_scores, limit=4)

    final_trace.update(
        {
            "raw_predicted_bdi_score": raw_predicted_bdi_score,
            "raw_predicted_label": raw_predicted_label,
            "final_bdi_score": final_bdi_score,
            "final_label": final_label,
            "imputed_item_count": imputed_item_count,
            "blended_observed_item_count": blended_observed_item_count,
            "blended_item_ids": blended_item_ids,
            "low_signal_guardrail_active": low_signal_guardrail_active,
            "low_signal_guardrail_reasons": list(low_signal_context["support_geometry_bypass_reasons"]),
            "support_geometry_candidate_bypass": support_geometry_candidate_bypass,
            "support_geometry_bypass_reasons": list(low_signal_context["support_geometry_bypass_reasons"]),
            "anchor_gated_guardrail_blocked": anchor_gated_guardrail_blocked,
            "guardrail_bypass_source": guardrail_bypass_source,
            "imputed_point_budget": imputed_point_budget,
            "imputed_points_before_guardrail": imputed_points_before_guardrail,
            "imputed_points_after_guardrail": imputed_points_after_guardrail,
            "predicted_key_item_ids": ranked_key_item_ids,
            "corroborated_item_ids": list(low_signal_context["corroborated_item_ids"]),
            "corroborated_core_hits": int(low_signal_context["corroborated_core_hits"]),
            "corroborated_affective_cognitive_module_breadth": int(
                low_signal_context["corroborated_affective_cognitive_module_breadth"]
            ),
            "dominant_support_item_id": int(low_signal_context["dominant_support_item_id"]),
            "dominant_support_share": round(float(low_signal_context["dominant_support_share"]), 6),
            "support_concentration_dominant": bool(low_signal_context["support_concentration_dominant"]),
            "low_signal_observed_cap_item_ids": low_signal_observed_cap_item_ids,
            "low_signal_item9_cap_reason": low_signal_item9_cap_reason,
            "severe_recovery_mode_active": severe_recovery_mode_active,
            "severe_anchor_item_ids": list(severe_anchor_context["severe_anchor_item_ids"]),
            "severe_anchor_module_ids": list(severe_anchor_context["severe_anchor_module_ids"]),
            "severe_recovery_reason": str(severe_anchor_context["severe_recovery_reason"]),
            "severe_recovered_item_ids": severe_recovered_item_ids,
            "severe_amplitude_observed_item_ids": severe_amplitude_observed_item_ids,
            "severe_amplitude_imputed_item_ids": severe_amplitude_imputed_item_ids,
            "severe_item9_rescued": bool(severe_item9_rescued),
        }
    )

    final_state = FinalState(
        predicted_bdi_score=final_bdi_score,
        predicted_label=final_label,
        top_symptoms=final_key_symptoms,
        evidence_report=_evidence_report(state),
        risk_flag=risk_flag,
        debug_trace=final_trace,
    )

    module_imputation = {
        "module_stats": module_stats,
        "item_details": item_details,
        "imputed_item_count": imputed_item_count,
        "blended_observed_item_count": blended_observed_item_count,
        "blended_item_ids": blended_item_ids,
        "obs_blend_enabled": obs_blend_enabled,
        "obs_blend_conf_threshold": obs_blend_conf_threshold,
        "obs_blend_support_max": obs_blend_support_max,
        "obs_blend_module_conf_min": obs_blend_module_conf_min,
        "obs_blend_max_alpha": obs_blend_max_alpha,
        "threshold": bdi_threshold,
        "risk_flag": risk_flag,
        "risk_reason": risk_reason,
        "core_hits": core_hits,
        "core_item_ids": CORE_ITEM_IDS,
        "final_core_item_min_hits": final_core_item_min_hits,
        "module_signal_total": round(module_signal_total, 6),
        "final_core_signal_gate": final_core_signal_gate,
        "low_signal_guardrail_active": low_signal_guardrail_active,
        "low_signal_guardrail_reasons": list(low_signal_context["support_geometry_bypass_reasons"]),
        "support_geometry_candidate_bypass": support_geometry_candidate_bypass,
        "support_geometry_bypass_reasons": list(low_signal_context["support_geometry_bypass_reasons"]),
        "anchor_gated_guardrail_blocked": anchor_gated_guardrail_blocked,
        "guardrail_bypass_source": guardrail_bypass_source,
        "observed_positive_breadth": int(low_signal_context["observed_positive_breadth"]),
        "observed_positive_item_ids": list(low_signal_context["observed_positive_item_ids"]),
        "observed_core_hits": int(low_signal_context["observed_core_hits"]),
        "observed_core_item_ids": list(low_signal_context["observed_core_item_ids"]),
        "strong_observed_item_count": int(low_signal_context["strong_observed_item_count"]),
        "strong_observed_item_ids": list(low_signal_context["strong_observed_item_ids"]),
        "strong_module_breadth": int(low_signal_context["strong_module_breadth"]),
        "strong_module_ids": list(low_signal_context["strong_module_ids"]),
        "affective_cognitive_corroboration": bool(low_signal_context["affective_cognitive_corroboration"]),
        "corroborated_item_ids": list(low_signal_context["corroborated_item_ids"]),
        "corroborated_core_hits": int(low_signal_context["corroborated_core_hits"]),
        "corroborated_affective_cognitive_module_ids": list(
            low_signal_context["corroborated_affective_cognitive_module_ids"]
        ),
        "corroborated_affective_cognitive_module_breadth": int(
            low_signal_context["corroborated_affective_cognitive_module_breadth"]
        ),
        "dominant_support_item_id": int(low_signal_context["dominant_support_item_id"]),
        "dominant_support_share": round(float(low_signal_context["dominant_support_share"]), 6),
        "support_concentration_dominant": bool(low_signal_context["support_concentration_dominant"]),
        "total_observed_support_count": int(low_signal_context["total_observed_support_count"]),
        "imputed_point_budget": imputed_point_budget,
        "imputed_points_before_guardrail": imputed_points_before_guardrail,
        "imputed_points_after_guardrail": imputed_points_after_guardrail,
        "suppressed_imputed_item_ids": suppressed_imputed_item_ids,
        "somatic_corroboration_blocked_item_ids": somatic_corroboration_blocked_item_ids,
        "low_signal_observed_cap_item_ids": low_signal_observed_cap_item_ids,
        "low_signal_singleton_trimmed_item_ids": low_signal_singleton_trimmed_item_ids,
        "low_signal_item9_cap_reason": low_signal_item9_cap_reason,
        "severe_recovery_mode_active": severe_recovery_mode_active,
        "severe_anchor_item_ids": list(severe_anchor_context["severe_anchor_item_ids"]),
        "severe_anchor_module_ids": list(severe_anchor_context["severe_anchor_module_ids"]),
        "severe_recovery_reason": str(severe_anchor_context["severe_recovery_reason"]),
        "severe_recovered_item_ids": severe_recovered_item_ids,
        "severe_amplitude_observed_item_ids": severe_amplitude_observed_item_ids,
        "severe_amplitude_imputed_item_ids": severe_amplitude_imputed_item_ids,
        "severe_item9_rescued": bool(severe_item9_rescued),
        "depression_from_bdi": depression_from_bdi,
        "depression_from_core_coverage": depression_from_core_coverage,
        "raw_predicted_bdi_score": raw_predicted_bdi_score,
        "raw_predicted_label": raw_predicted_label,
        "final_bdi_score": final_bdi_score,
        "final_label": final_label,
    }

    turn_trace["finalize_outputs"] = final_trace

    return {
        "control": ControlState(
            stop=True,
            stop_reason=control_reason or "finalized",
        ),
        "should_stop": True,
        "final": final_state,
        "raw_predicted_bdi_score": raw_predicted_bdi_score,
        "raw_predicted_label": raw_predicted_label,
        "predicted_bdi_score": final_bdi_score,
        "predicted_label": final_label,
        "predicted_key_symptoms": final_key_symptoms,
        "predicted_key_item_ids": ranked_key_item_ids,
        "final_item_scores": final_item_scores,
        "module_imputation": module_imputation,
        "turn_trace": turn_trace,
    }
