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
SOMATIC_CLUSTER_RECOVERY_OBSERVED_ITEM_IDS = {15, 16, 18, 20, 21}
SOMATIC_CLUSTER_RECOVERY_IMPUTED_ITEM_IDS = {16, 18, 20, 21}
SOLO_MODULE_IMPUTATION_BLOCKED_ITEMS = {18}
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
ITEM14_WORTHLESSNESS_PRIORITY_PHRASES = [
    "genuinely feel worthless",
    "feel worthless",
    "worthless",
    "feel like a burden",
    "i am a burden",
    "im a burden",
    "dont matter",
    "do not matter",
    "dont contribute",
    "do not contribute",
]
ITEM21_MILD_DIRECT_PHRASES = [
    "that side of things is a little lower than usual",
    "side of things is a little lower than usual",
    "less interested than usual",
    "little less interested",
    "reduced interest in sexual activity",
    "reduced interest in sex",
]
ITEM14_LATENT_RESTORE_PHRASES = [
    "feel worthless",
    "worthless",
    "feel like a burden",
    "i am a burden",
    "im a burden",
    "my own fault",
    "my fault",
    "blame myself",
    "feel like a failure",
    "i am a failure",
    "im a failure",
    "dont like who i am",
    "dont like the person ive been",
    "dislike who ive become",
    "dont measure up",
    "do not measure up",
    "hard on myself",
]
ITEM21_QUESTION_HISTORY_PHRASES = [
    "reduced interest in sexual activity",
    "interest in sexual activity",
    "interest in sex",
]
ITEM21_DIRECT_DENIAL_PHRASES = [
    "that side of things has been fine",
    "that side of things is fine",
    "not really a problem",
    "hasnt really been an issue",
    "nothing different there",
    "okay honestly",
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
    has_strong_row_by_item: Dict[int, bool] = {item_id: False for item_id in range(1, 22)}
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
        confidence = _clamp(float(getattr(row, "confidence", 0.0) or 0.0), 0.0, 1.0)
        intensity = _clamp(float(getattr(row, "intensity", 0.0) or 0.0), 0.0, 3.0)
        if confidence >= 0.55 and intensity >= 1.5:
            has_strong_row_by_item[item_id] = True

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
        has_strong_row = bool(has_strong_row_by_item[item_id])
        same_module_corroborated_item_ids = []
        if support_count >= 1:
            for other_item_id in same_module_supported_item_ids:
                other_support_count = int(item_beliefs[other_item_id].support_count)
                other_expected_score = float(item_beliefs[other_item_id].expected_score)
                other_evidence_turn_count = len(evidence_turns_by_item[other_item_id])
                other_has_strong_row = bool(has_strong_row_by_item[other_item_id])
                if other_support_count < 1 or other_evidence_turn_count < 1:
                    continue
                if (
                    other_support_count >= 2
                    or other_evidence_turn_count >= 2
                    or other_has_strong_row
                    or other_expected_score >= 1.75
                ):
                    same_module_corroborated_item_ids.append(other_item_id)
        is_corroborated_item = False
        if item_id == 9:
            if risk_reason in {"active self-harm cue match", "multiple passive death ideation cues"}:
                is_corroborated_item = True
        elif support_count >= 2 and evidence_turn_count >= 2:
            is_corroborated_item = True
        elif support_count >= 1 and has_strong_row:
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
            "has_strong_row": has_strong_row,
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
    observed_mean_severity = 0.0
    if observed_positive_item_ids:
        observed_mean_severity = sum(
            float(item_beliefs[item_id].expected_score) for item_id in observed_positive_item_ids
        ) / len(observed_positive_item_ids)
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
        ("strong_observed_item_count>=2", strong_observed_item_count >= 2, strong_observed_item_count),
        ("observed_mean_severity>=1.25", observed_mean_severity >= 1.25, round(observed_mean_severity, 6)),
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
        "observed_mean_severity": round(observed_mean_severity, 6),
    }


def _low_signal_imputed_budget(
    observed_positive_breadth: int,
    observed_core_hits: int,
    *,
    observed_mean_severity: float = 0.0,
) -> int:
    if observed_positive_breadth <= 1:
        return 0
    if observed_positive_breadth in {2, 3}:
        return 1
    if observed_positive_breadth == 4:
        if observed_mean_severity >= 2.0 and observed_core_hits >= 2:
            return 2
        return 1
    if observed_positive_breadth >= 5:
        base = 2
        if observed_mean_severity >= 2.0 and observed_core_hits >= 3:
            base += 1
        return min(base, 3)
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


def _recent_persona_hit(state: AgentState, phrases: List[str], *, limit: int = 4) -> bool:
    for utterance in _persona_utterances(state)[-limit:]:
        normalized = _normalize_text(utterance)
        if normalized and _contains_any(normalized, phrases):
            return True
    return False


def _detector_persona_pairs(state: AgentState) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    pending_question = ""
    for message in list(state.get("messages", [])):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip().lower()
        content = _normalize_text(message.get("content", ""))
        if not content:
            continue
        if role == "user":
            pending_question = content
            continue
        if role == "assistant" and pending_question:
            pairs.append((pending_question, content))
            pending_question = ""
    return pairs


def _item21_question_history_and_denials(state: AgentState) -> Tuple[bool, int]:
    question_history_hit = False
    denial_count = 0
    for question_text, reply_text in _detector_persona_pairs(state):
        if not _contains_any(question_text, ITEM21_QUESTION_HISTORY_PHRASES):
            continue
        question_history_hit = True
        if reply_text and _contains_any(reply_text, ITEM21_DIRECT_DENIAL_PHRASES):
            denial_count += 1
    return question_history_hit, denial_count


def _clear_imputed_suppression_tracking(
    *,
    item_id: int,
    detail: Dict[str, object],
    suppressed_imputed_item_ids: List[int],
    somatic_corroboration_blocked_item_ids: List[int],
) -> None:
    suppressed_imputed_item_ids[:] = [
        suppressed_item_id
        for suppressed_item_id in suppressed_imputed_item_ids
        if int(suppressed_item_id) != int(item_id)
    ]
    somatic_corroboration_blocked_item_ids[:] = [
        blocked_item_id
        for blocked_item_id in somatic_corroboration_blocked_item_ids
        if int(blocked_item_id) != int(item_id)
    ]
    detail["low_signal_somatic_blocked"] = False
    detail["low_signal_budget_suppressed"] = False


def _severe_anchor_context(
    state: AgentState,
    *,
    low_signal_context: Dict[str, object],
    item_beliefs: Dict[int, ItemBelief],
    item_evidence_summary: Dict[int, Dict[str, object]],
    raw_predicted_bdi_score: int,
    observed_mean_severity: float,
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
    single_anchor_anchor_module_ids = sorted(strong_anchor_module_ids) if len(strong_anchor_module_ids) == 1 else []
    single_anchor_activation_eligible = False
    single_anchor_qualified_supported_item_ids: List[int] = []
    if len(single_anchor_anchor_module_ids) == 1:
        anchor_module_id = int(single_anchor_anchor_module_ids[0])
        single_anchor_qualified_supported_item_ids = [
            item_id
            for item_id in MODULE_TO_ITEMS.get(anchor_module_id, [])
            if int(item_beliefs[item_id].support_count) >= 1
            and (
                float(item_beliefs[item_id].expected_score) >= 1.5
                or int(item_beliefs[item_id].support_count) >= 2
                or bool(item_evidence_summary.get(item_id, {}).get("has_strong_row", False))
            )
        ]
        single_anchor_activation_eligible = (
            (int(raw_predicted_bdi_score) >= 18 or float(observed_mean_severity) >= 1.6)
            and len(corroborated_nonrisk_item_ids) >= 4
            and len(corroborated_nonrisk_module_ids) >= 3
            and len(single_anchor_qualified_supported_item_ids) >= 2
        )

    severe_recovery_mode_active = False
    severe_recovery_reason = ""
    if len(strong_anchor_module_ids) >= 2 and cognitive_affective_anchor:
        severe_recovery_mode_active = True
        severe_recovery_reason = "multiple_strong_anchor_modules"
    elif single_anchor_activation_eligible:
        severe_recovery_mode_active = True
        severe_recovery_reason = "single_strong_anchor_with_severity_support"

    return {
        "severe_recovery_mode_active": severe_recovery_mode_active,
        "severe_anchor_item_ids": sorted(severe_anchor_item_ids),
        "severe_anchor_module_ids": sorted(severe_anchor_module_ids),
        "severe_strong_anchor_module_ids": sorted(strong_anchor_module_ids),
        "severe_recovery_reason": severe_recovery_reason,
        "severe_recovery_activation_path": severe_recovery_reason if severe_recovery_reason else "none",
        "corroborated_nonrisk_item_ids": corroborated_nonrisk_item_ids,
        "corroborated_nonrisk_module_ids": corroborated_nonrisk_module_ids,
        "single_anchor_activation_eligible": bool(single_anchor_activation_eligible),
        "single_anchor_anchor_module_ids": single_anchor_anchor_module_ids,
        "single_anchor_qualified_supported_item_ids": sorted(single_anchor_qualified_supported_item_ids),
    }


def _somatic_cluster_context(
    item_beliefs: Dict[int, ItemBelief],
    *,
    item_evidence_summary: Dict[int, Dict[str, object]],
    low_signal_guardrail_active: bool,
    raw_predicted_bdi_score: int,
    observed_mean_severity: float,
) -> Dict[str, object]:
    somatic_observed_positive_item_ids: List[int] = []
    somatic_observed_module_ids: set[int] = set()
    somatic_strong_item_ids: List[int] = []
    somatic_qualifying_module_ids: set[int] = set()

    for item_id in range(1, 22):
        somatic_modules = [
            module_id
            for module_id in ITEM_TO_MODULES.get(item_id, [])
            if module_id in LOW_SIGNAL_SOMATIC_INTERPERSONAL_MODULES
        ]
        if not somatic_modules:
            continue
        support_count = int(item_beliefs[item_id].support_count)
        expected_score = float(item_beliefs[item_id].expected_score)
        evidence_summary = dict(item_evidence_summary.get(item_id, {}))
        has_strong_row = bool(evidence_summary.get("has_strong_row", False))

        if support_count >= 1 and expected_score >= 1.0:
            somatic_observed_positive_item_ids.append(item_id)
            somatic_observed_module_ids.update(somatic_modules)

        if support_count >= 1 and (support_count >= 2 or expected_score >= 1.5 or has_strong_row):
            somatic_strong_item_ids.append(item_id)
            somatic_qualifying_module_ids.update(somatic_modules)

    somatic_observed_module_ids_list = sorted(somatic_observed_module_ids)
    somatic_cluster_recovery_active = (
        low_signal_guardrail_active
        and (int(raw_predicted_bdi_score) >= 18 or float(observed_mean_severity) >= 1.5)
        and len(somatic_observed_positive_item_ids) >= 3
        and len(somatic_observed_module_ids_list) >= 2
        and len(somatic_strong_item_ids) >= 2
    )

    return {
        "somatic_cluster_recovery_active": bool(somatic_cluster_recovery_active),
        "somatic_observed_positive_item_ids": sorted(somatic_observed_positive_item_ids),
        "somatic_observed_module_ids": somatic_observed_module_ids_list,
        "somatic_observed_module_breadth": len(somatic_observed_module_ids_list),
        "somatic_strong_item_ids": sorted(somatic_strong_item_ids),
        "somatic_qualifying_module_ids": sorted(somatic_qualifying_module_ids),
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
    denied_item_id_set = set(int(x) for x in state.get("denied_item_ids", []) if 1 <= int(x) <= 21)

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
        item_beliefs=beliefs,
        item_evidence_summary=item_evidence_summary,
        raw_predicted_bdi_score=raw_predicted_bdi_score,
        observed_mean_severity=float(low_signal_context.get("observed_mean_severity", 0.0) or 0.0),
    )
    severe_recovery_mode_active = bool(severe_anchor_context["severe_recovery_mode_active"])
    support_geometry_candidate_bypass = bool(low_signal_context["support_geometry_candidate_bypass"])
    anchor_gated_guardrail_blocked = support_geometry_candidate_bypass and not severe_recovery_mode_active
    # Keep the low-signal guardrail active globally and let severe recovery
    # operate through narrower item-level bypasses/restores instead.
    low_signal_guardrail_active = True
    somatic_cluster_context = _somatic_cluster_context(
        beliefs,
        item_evidence_summary=item_evidence_summary,
        low_signal_guardrail_active=low_signal_guardrail_active,
        raw_predicted_bdi_score=raw_predicted_bdi_score,
        observed_mean_severity=float(low_signal_context.get("observed_mean_severity", 0.0) or 0.0),
    )
    somatic_cluster_recovery_active = bool(somatic_cluster_context["somatic_cluster_recovery_active"])
    broad_shallow_profile_active = (
        low_signal_guardrail_active
        and not severe_recovery_mode_active
        and not somatic_cluster_recovery_active
        and int(low_signal_context["observed_positive_breadth"]) >= 4
        and int(low_signal_context["strong_observed_item_count"]) <= 1
        and int(low_signal_context["total_observed_support_count"])
        <= int(low_signal_context["observed_positive_breadth"]) + 2
        and float(low_signal_context["dominant_support_share"]) < 0.40
    )
    guardrail_bypass_source = "item_local_severe_recovery" if severe_recovery_mode_active else "none"
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
    severe_recovered_item_ids_by_module: Dict[str, List[int]] = {}
    severe_module3_restored_item_ids: List[int] = []
    severe_module3_restore_budget = 0
    severe_module3_item14_priority_applied = False
    item14_latent_restored_item_ids: List[int] = []
    severe_amplitude_observed_item_ids: List[int] = []
    severe_amplitude_observed_to_three_item_ids: List[int] = []
    severe_amplitude_imputed_item_ids: List[int] = []
    strong_anchor_local_bypass_item_ids: List[int] = []
    single_anchor_local_bypass_blocked_item_ids: List[int] = []
    broad_shallow_observed_keep_budget = (
        2 if broad_shallow_profile_active and raw_predicted_bdi_score < 14 else 3 if broad_shallow_profile_active else 0
    )
    broad_shallow_observed_trimmed_item_ids: List[int] = []
    somatic_cluster_floor_item_ids: List[int] = []
    somatic_cluster_imputed_restored_item_ids: List[int] = []
    single_anchor_module3_restore_blocked = False
    item21_mild_observed_retained = False
    item21_imputed_restored = False
    severe_item9_rescued = False
    severe_anchor_item_id_set = {int(item_id) for item_id in severe_anchor_context["severe_anchor_item_ids"]}
    severe_anchor_module_id_set = {int(module_id) for module_id in severe_anchor_context["severe_anchor_module_ids"]}
    severe_strong_anchor_module_id_set = {
        int(module_id) for module_id in severe_anchor_context["severe_strong_anchor_module_ids"]
    }
    severe_recovery_reason = str(severe_anchor_context.get("severe_recovery_reason", "") or "")
    severe_recovery_activation_path = str(severe_anchor_context.get("severe_recovery_activation_path", "none") or "none")
    recent_worthlessness_priority_hit = _recent_persona_hit(state, ITEM14_WORTHLESSNESS_PRIORITY_PHRASES)
    recent_item14_latent_restore_hit = _recent_persona_hit(state, ITEM14_LATENT_RESTORE_PHRASES, limit=6)
    recent_item21_mild_direct_hit = _recent_persona_hit(state, ITEM21_MILD_DIRECT_PHRASES, limit=8)
    item21_question_history_hit, item21_direct_denial_count = _item21_question_history_and_denials(state)

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
        strong_anchor_modules = [
            int(module_id)
            for module_id in ITEM_TO_MODULES.get(item_id, [])
            if int(module_id) in severe_strong_anchor_module_id_set
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
            severe_amplitude_observed_to_three = False
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
            strong_anchor_local_bypass_applied = False
            somatic_cluster_floor_applied = False
            strong_anchor_local_eligible = False
            if severe_recovery_mode_active and item_id != 9 and bool(strong_anchor_modules):
                strong_anchor_local_eligible = (
                    bool(severe_anchor_hit)
                    or support_count >= 2
                    or bool(evidence_summary.get("has_strong_row", False))
                    or (
                        severe_recovery_reason != "single_strong_anchor_with_severity_support"
                        and bool(is_corroborated_item)
                    )
                )
                if (
                    severe_recovery_reason == "single_strong_anchor_with_severity_support"
                    and not strong_anchor_local_eligible
                    and bool(is_corroborated_item)
                ):
                    single_anchor_local_bypass_blocked_item_ids.append(item_id)
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
                if support_count == 1 and not is_corroborated_item:
                    if strong_anchor_local_eligible:
                        strong_anchor_local_bypass_applied = True
                        strong_anchor_local_bypass_item_ids.append(item_id)
                    else:
                        if pre_trim_score < 1.5:
                            trimmed_int = 0
                        else:
                            trimmed_int = 1
                        if trimmed_int < final_int:
                            low_signal_observed_cap_applied = True
                            low_signal_singleton_trimmed_item_ids.append(item_id)
                            low_signal_observed_cap_item_ids.append(item_id)
                        final_int = trimmed_int
                elif final_int >= 2 and not is_corroborated_item:
                    if strong_anchor_local_eligible:
                        strong_anchor_local_bypass_applied = True
                        strong_anchor_local_bypass_item_ids.append(item_id)
                    else:
                        final_int = 1
                        low_signal_observed_cap_applied = True
                        low_signal_observed_cap_item_ids.append(item_id)

            if (
                somatic_cluster_recovery_active
                and item_id in SOMATIC_CLUSTER_RECOVERY_OBSERVED_ITEM_IDS
                and support_count >= 1
                and int(evidence_summary.get("evidence_turn_count", 0) or 0) >= 1
            ):
                somatic_adjusted = False
                if final_int == 0:
                    final_int = 1
                    somatic_adjusted = True
                if (
                    observed_float >= 1.75
                    and (
                        int(item_support_geometry.get("same_module_supported_item_count", 0) or 0) >= 1
                        or int(somatic_cluster_context["somatic_observed_module_breadth"]) >= 3
                    )
                    and (
                        float(evidence_summary.get("max_confidence", 0.0) or 0.0) >= 0.50
                        or bool(evidence_summary.get("has_strong_row", False))
                    )
                    and final_int < 2
                ):
                    final_int = 2
                    somatic_adjusted = True
                if somatic_adjusted:
                    somatic_cluster_floor_applied = True
                    somatic_cluster_floor_item_ids.append(item_id)

            if (
                strong_anchor_local_eligible
                and any(module_id in LOW_SIGNAL_SOMATIC_INTERPERSONAL_MODULES for module_id in strong_anchor_modules)
                and int(evidence_summary.get("evidence_turn_count", 0) or 0) >= 2
                and (
                    support_count >= 2
                    or bool(evidence_summary.get("has_very_strong_row", False))
                )
                and observed_float >= 2.25
                and final_int < 3
            ):
                final_int = 3
                severe_amplitude_observed_applied = True
                severe_amplitude_observed_to_three = True
                severe_amplitude_observed_item_ids.append(item_id)
                severe_amplitude_observed_to_three_item_ids.append(item_id)

            if (
                severe_recovery_mode_active
                and item_id == 21
                and final_int == 0
                and support_count >= 1
                and int(evidence_summary.get("evidence_turn_count", 0) or 0) >= 1
                and float(evidence_summary.get("max_confidence", 0.0) or 0.0) >= 0.55
                and recent_item21_mild_direct_hit
            ):
                final_int = 1
                item21_mild_observed_retained = True

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
                "strong_anchor_modules": strong_anchor_modules,
                "severe_recovery_mode_active": severe_recovery_mode_active,
                "severe_module3_restore_applied": False,
                "item14_latent_restore_applied": False,
                "item21_mild_observed_retained": bool(item_id == 21 and item21_mild_observed_retained),
                "item21_imputed_restore_applied": False,
                "broad_shallow_budget_trim_applied": False,
                "somatic_cluster_floor_applied": bool(somatic_cluster_floor_applied),
                "somatic_cluster_imputed_restore_applied": False,
                "pre_trim_score": round(pre_trim_score, 6),
                "post_trim_score": final_int,
                "weak_positive_trim_applied": bool(low_signal_observed_cap_applied or item9_guardrail_applied),
                "low_signal_singleton_trim_applied": bool(support_count == 1 and low_signal_observed_cap_applied),
                "low_signal_observed_cap_applied": bool(low_signal_observed_cap_applied),
                "low_signal_item9_guardrail_applied": bool(item9_guardrail_applied),
                "low_signal_item9_cap_reason": low_signal_item9_cap_reason if item_id == 9 else "",
                "low_signal_guardrail_active": low_signal_guardrail_active,
                "strong_anchor_local_bypass_applied": bool(strong_anchor_local_bypass_applied),
                "anchored_observed_severe": bool(anchored_observed_severe),
                "severe_amplitude_observed_applied": bool(severe_amplitude_observed_applied),
                "severe_amplitude_observed_to_three": bool(severe_amplitude_observed_to_three),
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
        denied_override = item_id in denied_item_id_set
        if denied_override:
            final_item_scores[item_id] = 0
        else:
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
            "denied_override": bool(denied_override),
            "severe_anchor_hit": bool(severe_anchor_hit),
            "severe_anchor_modules": severe_anchor_modules,
            "strong_anchor_modules": strong_anchor_modules,
            "severe_recovery_mode_active": severe_recovery_mode_active,
            "severe_module3_restore_applied": False,
            "item14_latent_restore_applied": False,
            "item21_mild_observed_retained": False,
            "item21_imputed_restore_applied": False,
            "broad_shallow_budget_trim_applied": False,
            "somatic_cluster_floor_applied": False,
            "somatic_cluster_imputed_restore_applied": False,
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
            "strong_anchor_local_bypass_applied": False,
            "severe_amplitude_imputed_applied": False,
            "severe_amplitude_observed_to_three": False,
            "blend_alpha": 0.0,
            "blend_applied": False,
            "blend_reason": "missing_imputed",
            "final_score": final_item_scores[item_id],
            "candidate_modules": ITEM_TO_MODULES.get(item_id, []),
            "contributions": contributions,
        }

    if broad_shallow_profile_active:
        broad_shallow_candidates: List[Tuple[int, float, int, float, int]] = []
        candidate_item_ids: List[int] = []
        for item_id in range(1, 22):
            detail = item_details.get(str(item_id), {})
            if str(detail.get("source", "")) not in {"observed", "observed_blended"}:
                continue
            if int(detail.get("support_count", 0) or 0) != 1 or item_id == 9:
                continue
            if int(detail.get("evidence_turn_count", 0) or 0) > 2:
                continue
            if bool(detail.get("has_strong_row", False)) or bool(detail.get("has_very_strong_row", False)):
                continue
            if float(detail.get("max_confidence", 0.0) or 0.0) > 0.45:
                continue
            broad_shallow_candidates.append(
                (
                    1 if bool(detail.get("is_corroborated_item", False)) else 0,
                    float(detail.get("pre_trim_score", 0.0) or 0.0),
                    int(detail.get("same_module_corroborated_item_count", 0) or 0),
                    float(detail.get("max_confidence", 0.0) or 0.0),
                    item_id,
                )
            )
            candidate_item_ids.append(item_id)

        broad_shallow_candidates.sort(key=lambda row: (-row[0], -row[1], -row[2], -row[3], row[4]))
        keep_item_ids: set[int] = set()
        used_module_ids: set[int] = set()
        for _, _, _, _, item_id in broad_shallow_candidates:
            candidate_module_ids = {int(module_id) for module_id in ITEM_TO_MODULES.get(item_id, [])}
            if used_module_ids.intersection(candidate_module_ids):
                continue
            keep_item_ids.add(item_id)
            used_module_ids.update(candidate_module_ids)
            if len(keep_item_ids) >= broad_shallow_observed_keep_budget:
                break

        for item_id in candidate_item_ids:
            if item_id in keep_item_ids:
                continue
            detail = item_details[str(item_id)]
            current_score = int(final_item_scores[item_id])
            capped_score = 0 if float(detail.get("pre_trim_score", 0.0) or 0.0) < 1.5 else 1
            updated_score = min(current_score, capped_score)
            if updated_score >= current_score:
                continue
            final_item_scores[item_id] = updated_score
            detail["broad_shallow_budget_trim_applied"] = True
            detail["weak_positive_trim_applied"] = True
            detail["post_trim_score"] = updated_score
            detail["final_score"] = updated_score
            broad_shallow_observed_trimmed_item_ids.append(item_id)

    imputed_points_before_guardrail = sum(
        int(final_item_scores[item_id])
        for item_id in range(1, 22)
        if str(item_details.get(str(item_id), {}).get("source", "")) == "imputed"
    )
    suppressed_imputed_item_ids: List[int] = []
    somatic_corroboration_blocked_item_ids: List[int] = []
    solo_module_blocked_item_ids: List[int] = []
    imputed_point_budget: int | None = None

    if low_signal_guardrail_active:
        imputed_point_budget = _low_signal_imputed_budget(
            int(low_signal_context["observed_positive_breadth"]),
            int(low_signal_context["observed_core_hits"]),
            observed_mean_severity=float(low_signal_context.get("observed_mean_severity", 0.0) or 0.0),
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

        for item_id in range(1, 22):
            detail = item_details.get(str(item_id), {})
            if str(detail.get("source", "")) != "imputed" or final_item_scores[item_id] <= 0:
                continue
            if item_id not in SOLO_MODULE_IMPUTATION_BLOCKED_ITEMS:
                continue
            contributions = list(detail.get("contributions", []))
            eligible_contributions = [
                c for c in contributions
                if bool(c.get("eligible", False)) or float(c.get("weight", 0.0) or 0.0) > 0.0
            ]
            all_solo = all(
                int(module_stats.get(int(c.get("module_id", 0)), {}).get("observed_item_count", 0) or 0) < 2
                for c in eligible_contributions
            ) if eligible_contributions else True
            if all_solo:
                final_item_scores[item_id] = 0
                detail["low_signal_solo_module_blocked"] = True
                detail["post_trim_score"] = 0
                detail["final_score"] = 0
                solo_module_blocked_item_ids.append(item_id)
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
        restore_candidates_by_module: Dict[int, List[Tuple[float, float, float, int]]] = {
            module_id: [] for module_id in sorted(strong_anchor_module_ids)
        }
        for item_id in range(1, 22):
            detail = item_details.get(str(item_id), {})
            if str(detail.get("source", "")) != "imputed" or int(final_item_scores[item_id]) > 0:
                continue
            imputed_float = float(detail.get("imputed_float", 0.0) or 0.0)
            if imputed_float <= 0.0:
                continue
            contribution_by_module = {
                int(contribution.get("module_id", 0) or 0): contribution
                for contribution in list(detail.get("contributions", []))
            }
            candidate_modules = []
            for module_id in ITEM_TO_MODULES.get(item_id, []):
                contribution = contribution_by_module.get(int(module_id), {})
                if int(module_id) not in strong_anchor_module_ids:
                    continue
                if not (
                    bool(contribution.get("eligible", False))
                    or float(contribution.get("weight", 0.0) or 0.0) > 0.0
                ):
                    continue
                candidate_modules.append(int(module_id))
                restore_candidates_by_module[int(module_id)].append(
                    (
                        imputed_float,
                        float(module_stats.get(int(module_id), {}).get("module_mean", 0.0) or 0.0),
                        float(contribution.get("weight", 0.0) or 0.0),
                        item_id,
                    )
                )
            detail["severe_recovery_source_modules"] = candidate_modules

        for module_id in sorted(restore_candidates_by_module):
            candidates = sorted(
                restore_candidates_by_module[module_id],
                key=lambda row: (-row[0], -row[1], -row[2], row[3]),
            )
            restored = 0
            for _, _, _, item_id in candidates:
                if restored >= 1:
                    break
                detail = item_details[str(item_id)]
                if int(final_item_scores[item_id]) > 0:
                    continue
                if item_id in denied_item_id_set:
                    continue
                final_item_scores[item_id] = 1
                detail["severe_recovery_restored"] = True
                detail["severe_recovery_source_modules"] = [module_id]
                severe_recovered_item_ids.append(item_id)
                severe_recovered_item_ids_by_module.setdefault(str(module_id), []).append(item_id)
                restored += 1

        module3_item_ids = set(MODULE_TO_ITEMS.get(3, []))
        module3_observed_severe_item_ids = [
            item_id
            for item_id in sorted(module3_item_ids)
            if int(beliefs[item_id].support_count) >= 1
            and (
                float(beliefs[item_id].expected_score) >= 1.5
                or bool(item_evidence_summary.get(item_id, {}).get("has_strong_row", False))
            )
        ]
        corroborated_nonrisk_outside_module3 = any(
            int(item_id) not in module3_item_ids
            for item_id in list(severe_anchor_context.get("corroborated_nonrisk_item_ids", []))
        )
        allow_module3_restore = False
        if module3_observed_severe_item_ids and corroborated_nonrisk_outside_module3:
            if severe_recovery_reason != "single_strong_anchor_with_severity_support":
                allow_module3_restore = True
            elif 3 in severe_strong_anchor_module_id_set:
                allow_module3_restore = True
            elif len(module3_observed_severe_item_ids) >= 2 and int(raw_predicted_bdi_score) >= 18:
                allow_module3_restore = True
            else:
                single_anchor_module3_restore_blocked = True
        if allow_module3_restore:
            severe_module3_restore_budget = 2
            module3_restore_candidates: List[Tuple[int, float, float, int]] = []
            for item_id in sorted(module3_item_ids):
                detail = item_details.get(str(item_id), {})
                if str(detail.get("source", "")) != "imputed" or int(final_item_scores[item_id]) > 0:
                    continue
                imputed_float = float(detail.get("imputed_float", 0.0) or 0.0)
                item14_with_worthlessness = (item_id == 14 and recent_worthlessness_priority_hit)
                restore_threshold = 0.70 if item14_with_worthlessness else 1.0
                if imputed_float < restore_threshold:
                    continue
                contribution_by_module = {
                    int(contribution.get("module_id", 0) or 0): contribution
                    for contribution in list(detail.get("contributions", []))
                }
                module3_contribution = contribution_by_module.get(3, {})
                module3_weight = float(module3_contribution.get("weight", 0.0) or 0.0)
                if not (
                    bool(module3_contribution.get("eligible", False))
                    or module3_weight > 0.0
                ):
                    continue
                worthlessness_priority = 1 if item14_with_worthlessness else 0
                if worthlessness_priority > 0:
                    severe_module3_item14_priority_applied = True
                module3_restore_candidates.append((worthlessness_priority, imputed_float, module3_weight, item_id))

            module3_restore_candidates.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
            restored = 0
            for _, _, _, item_id in module3_restore_candidates:
                if restored >= severe_module3_restore_budget:
                    break
                if int(final_item_scores[item_id]) > 0:
                    continue
                if item_id in denied_item_id_set:
                    continue
                final_item_scores[item_id] = 1
                detail = item_details[str(item_id)]
                detail["severe_recovery_restored"] = True
                detail["severe_recovery_source_modules"] = [3]
                detail["severe_module3_restore_applied"] = True
                severe_recovered_item_ids.append(item_id)
                severe_module3_restored_item_ids.append(item_id)
                restored += 1

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
                if item_id in denied_item_id_set:
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

    if somatic_cluster_recovery_active:
        qualifying_module_ids = {
            int(module_id)
            for module_id in list(somatic_cluster_context.get("somatic_qualifying_module_ids", []))
            if int(module_id) in LOW_SIGNAL_SOMATIC_INTERPERSONAL_MODULES
        }
        somatic_restore_candidates: List[Tuple[float, float, float, int, int]] = []
        for item_id in sorted(SOMATIC_CLUSTER_RECOVERY_IMPUTED_ITEM_IDS):
            detail = item_details.get(str(item_id), {})
            if str(detail.get("source", "")) != "imputed" or int(final_item_scores.get(item_id, 0)) > 0:
                continue
            if item_id in denied_item_id_set:
                continue
            imputed_float = float(detail.get("imputed_float", 0.0) or 0.0)
            if imputed_float < 1.25:
                continue
            contribution_by_module = {
                int(contribution.get("module_id", 0) or 0): contribution
                for contribution in list(detail.get("contributions", []))
            }
            best_candidate: Tuple[float, float, int] | None = None
            for module_id in ITEM_TO_MODULES.get(item_id, []):
                if module_id not in qualifying_module_ids:
                    continue
                contribution = contribution_by_module.get(int(module_id), {})
                contribution_weight = float(contribution.get("weight", 0.0) or 0.0)
                if not (
                    bool(contribution.get("eligible", False))
                    or contribution_weight > 0.0
                ):
                    continue
                module_mean = float(module_stats.get(int(module_id), {}).get("module_mean", 0.0) or 0.0)
                candidate_row = (contribution_weight, module_mean, int(module_id))
                if best_candidate is None or candidate_row > best_candidate:
                    best_candidate = candidate_row
            if best_candidate is None:
                continue
            contribution_weight, module_mean, module_id = best_candidate
            somatic_restore_candidates.append(
                (
                    imputed_float,
                    contribution_weight,
                    module_mean,
                    module_id,
                    item_id,
                )
            )

        somatic_restore_candidates.sort(key=lambda row: (-row[0], -row[1], -row[2], row[4]))
        used_restore_modules: set[int] = set()
        for _, _, _, module_id, item_id in somatic_restore_candidates:
            if module_id in used_restore_modules:
                continue
            detail = item_details[str(item_id)]
            final_item_scores[item_id] = 1
            detail["somatic_cluster_imputed_restore_applied"] = True
            detail["post_trim_score"] = 1
            detail["final_score"] = 1
            _clear_imputed_suppression_tracking(
                item_id=item_id,
                detail=detail,
                suppressed_imputed_item_ids=suppressed_imputed_item_ids,
                somatic_corroboration_blocked_item_ids=somatic_corroboration_blocked_item_ids,
            )
            somatic_cluster_imputed_restored_item_ids.append(item_id)
            used_restore_modules.add(module_id)

    module3_latent_companion_count = sum(
        1 for companion_item_id in (5, 7, 8) if int(final_item_scores.get(companion_item_id, 0)) >= 1
    )
    module3_companion_mean = 0.0
    module3_companion_scores = [
        int(final_item_scores.get(companion_item_id, 0))
        for companion_item_id in (5, 7, 8)
        if int(final_item_scores.get(companion_item_id, 0)) >= 1
    ]
    if module3_companion_scores:
        module3_companion_mean = sum(module3_companion_scores) / len(module3_companion_scores)
    item14_detail = item_details.get("14", {})
    item14_restore_score = 1
    if module3_latent_companion_count >= 2 and module3_companion_mean >= 2.0 and recent_worthlessness_priority_hit:
        item14_restore_score = min(3, _round_item_score(module3_companion_mean))
    elif module3_latent_companion_count >= 2 and module3_companion_mean >= 1.5:
        item14_restore_score = 2
    if (
        str(item14_detail.get("source", "")) == "imputed"
        and int(final_item_scores.get(14, 0)) < item14_restore_score
        and float(item14_detail.get("imputed_float", 0.0) or 0.0) >= 0.75
        and recent_item14_latent_restore_hit
        and (
            module3_latent_companion_count >= 2
            or (module3_latent_companion_count >= 1 and severe_recovery_mode_active)
        )
    ):
        final_item_scores[14] = item14_restore_score
        item14_detail["item14_latent_restore_applied"] = True
        item14_detail["post_trim_score"] = item14_restore_score
        item14_detail["final_score"] = item14_restore_score
        _clear_imputed_suppression_tracking(
            item_id=14,
            detail=item14_detail,
            suppressed_imputed_item_ids=suppressed_imputed_item_ids,
            somatic_corroboration_blocked_item_ids=somatic_corroboration_blocked_item_ids,
        )
        item14_latent_restored_item_ids.append(14)

    item20_summary = dict(item_evidence_summary.get(20, {}))
    item21_detail = item_details.get("21", {})
    if (
        severe_recovery_mode_active
        and str(item21_detail.get("source", "")) == "imputed"
        and int(final_item_scores.get(21, 0)) == 0
        and float(item21_detail.get("imputed_float", 0.0) or 0.0) >= 1.0
        and item21_question_history_hit
        and int(item21_direct_denial_count) <= 1
        and int(beliefs[20].support_count) >= 1
        and (
            float(beliefs[20].expected_score) >= 1.5
            or bool(item20_summary.get("has_strong_row", False))
        )
    ):
        final_item_scores[21] = 1
        item21_detail["item21_imputed_restore_applied"] = True
        item21_detail["post_trim_score"] = 1
        item21_detail["final_score"] = 1
        _clear_imputed_suppression_tracking(
            item_id=21,
            detail=item21_detail,
            suppressed_imputed_item_ids=suppressed_imputed_item_ids,
            somatic_corroboration_blocked_item_ids=somatic_corroboration_blocked_item_ids,
        )
        item21_imputed_restored = True

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
    broad_shallow_observed_trimmed_item_ids = sorted(set(broad_shallow_observed_trimmed_item_ids))
    severe_recovered_item_ids = sorted(set(severe_recovered_item_ids))
    severe_module3_restored_item_ids = sorted(set(severe_module3_restored_item_ids))
    item14_latent_restored_item_ids = sorted(set(item14_latent_restored_item_ids))
    severe_amplitude_observed_item_ids = sorted(set(severe_amplitude_observed_item_ids))
    severe_amplitude_observed_to_three_item_ids = sorted(set(severe_amplitude_observed_to_three_item_ids))
    severe_amplitude_imputed_item_ids = sorted(set(severe_amplitude_imputed_item_ids))
    strong_anchor_local_bypass_item_ids = sorted(set(strong_anchor_local_bypass_item_ids))
    somatic_cluster_floor_item_ids = sorted(set(somatic_cluster_floor_item_ids))
    somatic_cluster_imputed_restored_item_ids = sorted(set(somatic_cluster_imputed_restored_item_ids))

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
            "broad_shallow_profile_active": bool(broad_shallow_profile_active),
            "broad_shallow_observed_keep_budget": int(broad_shallow_observed_keep_budget),
            "broad_shallow_observed_trimmed_item_ids": broad_shallow_observed_trimmed_item_ids,
            "low_signal_observed_cap_item_ids": low_signal_observed_cap_item_ids,
            "low_signal_item9_cap_reason": low_signal_item9_cap_reason,
            "severe_recovery_mode_active": severe_recovery_mode_active,
            "somatic_cluster_recovery_active": bool(somatic_cluster_recovery_active),
            "somatic_observed_module_breadth": int(somatic_cluster_context["somatic_observed_module_breadth"]),
            "somatic_cluster_floor_item_ids": somatic_cluster_floor_item_ids,
            "somatic_cluster_imputed_restored_item_ids": somatic_cluster_imputed_restored_item_ids,
            "severe_anchor_item_ids": list(severe_anchor_context["severe_anchor_item_ids"]),
            "severe_anchor_module_ids": list(severe_anchor_context["severe_anchor_module_ids"]),
            "severe_recovery_reason": severe_recovery_reason,
            "severe_recovery_activation_path": severe_recovery_activation_path,
            "single_anchor_activation_eligible": bool(severe_anchor_context["single_anchor_activation_eligible"]),
            "single_anchor_anchor_module_ids": list(severe_anchor_context["single_anchor_anchor_module_ids"]),
            "single_anchor_module3_restore_blocked": bool(single_anchor_module3_restore_blocked),
            "single_anchor_local_bypass_blocked_item_ids": sorted(set(single_anchor_local_bypass_blocked_item_ids)),
            "severe_recovered_item_ids": severe_recovered_item_ids,
            "severe_recovered_item_ids_by_module": severe_recovered_item_ids_by_module,
            "severe_module3_restored_item_ids": severe_module3_restored_item_ids,
            "severe_module3_restore_budget": severe_module3_restore_budget,
            "severe_module3_item14_priority_applied": bool(severe_module3_item14_priority_applied),
            "item14_latent_restored_item_ids": item14_latent_restored_item_ids,
            "strong_anchor_local_bypass_item_ids": strong_anchor_local_bypass_item_ids,
            "severe_amplitude_observed_item_ids": severe_amplitude_observed_item_ids,
            "severe_amplitude_observed_to_three_item_ids": severe_amplitude_observed_to_three_item_ids,
            "severe_amplitude_imputed_item_ids": severe_amplitude_imputed_item_ids,
            "item21_mild_observed_retained": bool(item21_mild_observed_retained),
            "item21_imputed_restored": bool(item21_imputed_restored),
            "item21_question_history_hit": bool(item21_question_history_hit),
            "item21_direct_denial_count": int(item21_direct_denial_count),
            "severe_item9_rescued": bool(severe_item9_rescued),
            "denied_item_ids": sorted(denied_item_id_set),
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
        "broad_shallow_profile_active": bool(broad_shallow_profile_active),
        "broad_shallow_observed_keep_budget": int(broad_shallow_observed_keep_budget),
        "broad_shallow_observed_trimmed_item_ids": broad_shallow_observed_trimmed_item_ids,
        "total_observed_support_count": int(low_signal_context["total_observed_support_count"]),
        "imputed_point_budget": imputed_point_budget,
        "imputed_points_before_guardrail": imputed_points_before_guardrail,
        "imputed_points_after_guardrail": imputed_points_after_guardrail,
        "suppressed_imputed_item_ids": suppressed_imputed_item_ids,
        "somatic_corroboration_blocked_item_ids": somatic_corroboration_blocked_item_ids,
        "solo_module_blocked_item_ids": sorted(set(solo_module_blocked_item_ids)),
        "low_signal_observed_cap_item_ids": low_signal_observed_cap_item_ids,
        "low_signal_singleton_trimmed_item_ids": low_signal_singleton_trimmed_item_ids,
        "low_signal_item9_cap_reason": low_signal_item9_cap_reason,
        "severe_recovery_mode_active": severe_recovery_mode_active,
        "somatic_cluster_recovery_active": bool(somatic_cluster_recovery_active),
        "somatic_observed_positive_item_ids": list(somatic_cluster_context["somatic_observed_positive_item_ids"]),
        "somatic_observed_module_ids": list(somatic_cluster_context["somatic_observed_module_ids"]),
        "somatic_observed_module_breadth": int(somatic_cluster_context["somatic_observed_module_breadth"]),
        "somatic_strong_item_ids": list(somatic_cluster_context["somatic_strong_item_ids"]),
        "somatic_cluster_floor_item_ids": somatic_cluster_floor_item_ids,
        "somatic_cluster_imputed_restored_item_ids": somatic_cluster_imputed_restored_item_ids,
        "severe_anchor_item_ids": list(severe_anchor_context["severe_anchor_item_ids"]),
        "severe_anchor_module_ids": list(severe_anchor_context["severe_anchor_module_ids"]),
        "severe_recovery_reason": severe_recovery_reason,
        "severe_recovery_activation_path": severe_recovery_activation_path,
        "single_anchor_activation_eligible": bool(severe_anchor_context["single_anchor_activation_eligible"]),
        "single_anchor_anchor_module_ids": list(severe_anchor_context["single_anchor_anchor_module_ids"]),
        "single_anchor_module3_restore_blocked": bool(single_anchor_module3_restore_blocked),
        "single_anchor_local_bypass_blocked_item_ids": sorted(set(single_anchor_local_bypass_blocked_item_ids)),
        "severe_recovered_item_ids": severe_recovered_item_ids,
        "severe_recovered_item_ids_by_module": severe_recovered_item_ids_by_module,
        "severe_module3_restored_item_ids": severe_module3_restored_item_ids,
        "severe_module3_restore_budget": severe_module3_restore_budget,
        "severe_module3_item14_priority_applied": bool(severe_module3_item14_priority_applied),
        "item14_latent_restored_item_ids": item14_latent_restored_item_ids,
        "strong_anchor_local_bypass_item_ids": strong_anchor_local_bypass_item_ids,
        "severe_amplitude_observed_item_ids": severe_amplitude_observed_item_ids,
        "severe_amplitude_observed_to_three_item_ids": severe_amplitude_observed_to_three_item_ids,
        "severe_amplitude_imputed_item_ids": severe_amplitude_imputed_item_ids,
        "item21_mild_observed_retained": bool(item21_mild_observed_retained),
        "item21_imputed_restored": bool(item21_imputed_restored),
        "item21_question_history_hit": bool(item21_question_history_hit),
        "item21_direct_denial_count": int(item21_direct_denial_count),
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
