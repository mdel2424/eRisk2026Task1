from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple

from core.probabilistic_runtime import (
    CLUSTER_TO_ITEMS,
    ITEM_PARENT_WEIGHTS,
    NODE_LEAKS,
    NODE_PRIORS,
    bounded_uncertainty,
    noisy_or_probability,
)
from core.state import (
    AgentState,
    BayesItemState,
    BayesNodeState,
    BeliefState,
    ItemBelief,
    coerce_item_belief,
)


POSITIVE_ASSERTIONS = {"present", "conditional", "contrastive"}
NEGATIVE_ASSERTIONS = {"absent"}


def _state_value(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _bound_positive_assertions(assertions: Iterable[Any]) -> List[Any]:
    rows: List[Any] = []
    for row in assertions:
        label = str(_state_value(row, "assertion_label", "") or "")
        binding_status = str(_state_value(row, "binding_status", "") or "")
        if label in POSITIVE_ASSERTIONS and binding_status in {"exact", "normalized_exact"}:
            rows.append(row)
    return rows


def _latest_assertions_by_item(assertions: Iterable[Any]) -> Dict[int, List[Any]]:
    grouped: Dict[int, List[Any]] = {}
    for row in assertions:
        try:
            item_id = int(_state_value(row, "item_id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if not 1 <= item_id <= 21:
            continue
        grouped.setdefault(item_id, []).append(row)
    return grouped


def _update_probability(prior: float, delta: float) -> float:
    return max(0.0, min(1.0, (0.60 * float(prior)) + (0.40 * float(delta))))


def _cluster_evidence_delta(
    cluster_items: List[int],
    assertions_by_item: Dict[int, List[Any]],
) -> tuple[float, int, int, int]:
    positive = 0.0
    negative = 0.0
    evidence_count = 0
    positive_items: set[int] = set()
    strong_positive_items: set[int] = set()
    for item_id in cluster_items:
        strongest_positive = 0.0
        for row in assertions_by_item.get(item_id, []):
            label = str(_state_value(row, "assertion_label", "") or "")
            confidence = max(0.0, min(1.0, float(_state_value(row, "confidence", 0.0) or 0.0)))
            intensity = max(0.0, min(3.0, float(_state_value(row, "intensity", 0.0) or 0.0)))
            binding_status = str(_state_value(row, "binding_status", "") or "")
            bound_bonus = 1.0 if binding_status in {"exact", "normalized_exact"} else 0.6
            if label in POSITIVE_ASSERTIONS:
                evidence_count += 1
                positive_items.add(int(item_id))
                strongest_positive = max(strongest_positive, confidence * bound_bonus)
                if confidence >= 0.8 or intensity >= 2.0:
                    strong_positive_items.add(int(item_id))
            elif label in NEGATIVE_ASSERTIONS:
                evidence_count += 1
                negative += confidence * 0.5
        positive += strongest_positive
    if positive <= 0.0 and negative <= 0.0:
        return 0.0, evidence_count, 0, 0
    corroboration_factor = 0.42 if len(positive_items) <= 1 else 0.74 if len(positive_items) == 2 else 1.0
    if len(strong_positive_items) >= 2:
        corroboration_factor = max(corroboration_factor, 0.88)
    delta = max(0.0, min(1.0, (positive * 0.18 * corroboration_factor) - (negative * 0.11)))
    return delta, evidence_count, len(positive_items), len(strong_positive_items)


def _expected_from_posterior(posterior: List[float]) -> float:
    return max(0.0, min(3.0, sum(float(idx) * float(prob) for idx, prob in enumerate(posterior))))


def _score_posterior(presence_prob: float, severity_signal: float) -> List[float]:
    presence = max(0.0, min(1.0, float(presence_prob)))
    severity = max(0.0, min(3.0, float(severity_signal)))
    if presence <= 0.08:
        return [0.94, 0.05, 0.009, 0.001]

    mild = max(0.0, min(1.0, 1.0 - abs(severity - 1.0) / 2.0))
    moderate = max(0.0, min(1.0, 1.0 - abs(severity - 2.0) / 2.0))
    severe = max(0.0, min(1.0, 1.0 - abs(severity - 3.0) / 2.0))
    score0 = max(0.02, 1.0 - presence)
    residual = max(0.0, 1.0 - score0)
    raw = [
        score0,
        residual * max(0.12, mild),
        residual * max(0.08, moderate),
        residual * max(0.04, severe * presence),
    ]
    total = sum(raw)
    if total <= 0:
        return [0.25, 0.25, 0.25, 0.25]
    return [value / total for value in raw]


def _compatibility_support_count(evidence_log: Iterable[Any], item_id: int) -> int:
    unique_pairs = set()
    for row in evidence_log:
        try:
            row_item_id = int(_state_value(row, "item_id", 0) or 0)
        except (TypeError, ValueError):
            continue
        if row_item_id != item_id:
            continue
        label = str(_state_value(row, "assertion_label", "present") or "present")
        if label not in POSITIVE_ASSERTIONS and label not in {"", "present"}:
            continue
        evidence_text = str(_state_value(row, "evidence_text", "") or "").strip().lower()
        turn = int(_state_value(row, "turn", 0) or 0)
        unique_pairs.add((turn, evidence_text))
    return max(0, len(unique_pairs))


def _default_leak_for_item(item_id: int) -> float:
    parent_weights = dict(ITEM_PARENT_WEIGHTS.get(int(item_id), {}) or {})
    if not parent_weights:
        return 0.03
    if int(item_id) == 9:
        return NODE_LEAKS.get("risk", 0.03)
    dominant_parent = max(parent_weights.items(), key=lambda pair: (float(pair[1]), pair[0]))[0]
    base = float(NODE_LEAKS.get(dominant_parent, NODE_LEAKS.get("cognitive_affective", 0.05)))
    if dominant_parent in {"somatic_vegetative", "physiological_disruption"}:
        return max(0.015, min(0.05, base * 0.45))
    if dominant_parent == "negative_self_schema":
        return max(0.02, min(0.05, base * 0.60))
    return max(0.02, min(0.05, base * 0.55))


def bayes_state_update(state: AgentState) -> Dict[str, Any]:
    turn = int(state.get("turn_index", 0) or 0)
    prior_nodes = dict(state.get("bayes_nodes", {}))
    prior_items = dict(state.get("bayes_items", {}))
    assertions = list(state.get("latest_turn_assertions", []))
    evidence_log = list(state.get("evidence_log", []))
    assertions_by_item = _latest_assertions_by_item(assertions)

    bayes_nodes: Dict[str, BayesNodeState] = {}
    for node_id, prior_prob in NODE_PRIORS.items():
        previous = prior_nodes.get(node_id)
        previous_prob = float(_state_value(previous, "probability", prior_prob) or prior_prob)
        cluster_items = list(CLUSTER_TO_ITEMS.get(node_id, []))
        delta, evidence_count, distinct_positive_items, strong_positive_items = _cluster_evidence_delta(cluster_items, assertions_by_item)
        probability = _update_probability(previous_prob, max(prior_prob, previous_prob + delta))
        if distinct_positive_items <= 1:
            saturation_cap = 0.68 if strong_positive_items <= 0 else 0.78
            probability = min(probability, max(previous_prob, saturation_cap))
        bayes_nodes[node_id] = BayesNodeState(
            node_id=node_id,
            probability=probability,
            uncertainty=bounded_uncertainty(probability),
            evidence_count=int(_state_value(previous, "evidence_count", 0) or 0) + int(evidence_count),
            last_update_turn=turn,
        )

    parent_probs = {node_id: node.probability for node_id, node in bayes_nodes.items()}
    bayes_items: Dict[int, BayesItemState] = {}
    item_beliefs: Dict[int, ItemBelief] = {}

    for item_id in range(1, 22):
        previous_item = prior_items.get(item_id)
        previous_presence = float(_state_value(previous_item, "presence_prob", 0.0) or 0.0)
        structural_presence = noisy_or_probability(
            parent_probs=parent_probs,
            parent_weights=ITEM_PARENT_WEIGHTS.get(item_id, {}),
            leak=_default_leak_for_item(item_id),
        )

        evidence_presence = structural_presence
        severity_samples: List[float] = []
        audit_trail: List[Dict[str, Any]] = list(_state_value(previous_item, "audit_trail", []) or [])
        positive_assertion_count = 0
        for row in assertions_by_item.get(item_id, []):
            label = str(_state_value(row, "assertion_label", "") or "")
            confidence = max(0.0, min(1.0, float(_state_value(row, "confidence", 0.0) or 0.0)))
            intensity = max(0.0, min(3.0, float(_state_value(row, "intensity", 0.0) or 0.0)))
            binding_status = str(_state_value(row, "binding_status", "") or "")
            bound_factor = 1.0 if binding_status in {"exact", "normalized_exact"} else 0.55
            if label in POSITIVE_ASSERTIONS:
                positive_assertion_count += 1
                evidence_presence = max(
                    evidence_presence,
                    confidence * (0.40 if label == "conditional" else 0.55) * bound_factor,
                )
                severity_samples.append(max(0.8, intensity))
            elif label in NEGATIVE_ASSERTIONS:
                evidence_presence *= max(0.15, 1.0 - (0.45 * confidence))
            audit_trail.append(
                {
                    "turn": int(_state_value(row, "turn", turn) or turn),
                    "item_id": item_id,
                    "assertion_label": label,
                    "binding_status": binding_status,
                    "anchor_quote": str(_state_value(row, "anchor_quote", "") or ""),
                    "reason": str(_state_value(row, "reason", "") or ""),
                }
            )

        presence_delta = max(structural_presence, evidence_presence)
        if positive_assertion_count <= 1:
            presence_delta = max(structural_presence, evidence_presence * 0.88)
        presence_prob = _update_probability(previous_presence or structural_presence, presence_delta)
        severity_signal = sum(severity_samples) / float(len(severity_samples)) if severity_samples else (presence_prob * 2.4)
        score_posterior = _score_posterior(presence_prob, severity_signal)
        expected_score = _expected_from_posterior(score_posterior)
        uncertainty = 1.0 - max(score_posterior)

        bayes_items[item_id] = BayesItemState(
            item_id=item_id,
            presence_prob=presence_prob,
            score_posterior=score_posterior,
            expected_score=expected_score,
            uncertainty=max(0.0, min(1.0, uncertainty)),
            audit_trail=audit_trail[-24:],
        )

        support_count = _compatibility_support_count(evidence_log, item_id)
        item_beliefs[item_id] = ItemBelief(
            item_id=item_id,
            posterior=score_posterior,
            expected_score=expected_score,
            entropy=max(0.0, min(2.0, -sum(max(1e-12, p) * math.log2(max(1e-12, p)) for p in score_posterior))),
            support_count=support_count,
            last_update_turn=turn if support_count > 0 else int(_state_value(coerce_item_belief(item_id, state.get("item_beliefs", {}).get(item_id)), "last_update_turn", 0) or 0),
        )

    total_expected_bdi = sum(float(item.expected_score) for item in bayes_items.values())
    top_uncertain_items = sorted(
        range(1, 22),
        key=lambda item_id: (
            -float(bayes_items[item_id].uncertainty),
            -float(bayes_items[item_id].presence_prob),
            item_id,
        ),
    )[:6]

    turn_trace = dict(state.get("turn_trace", {}))
    turn_trace["bayes_state_update"] = {
        "turn": turn,
        "bayes_node_posteriors": {
            node_id: round(float(node.probability), 4) for node_id, node in bayes_nodes.items()
        },
        "top_uncertain_items": list(top_uncertain_items),
        "total_expected_bdi": round(float(total_expected_bdi), 4),
        "binding_positive_assertion_count": len(_bound_positive_assertions(assertions)),
    }

    return {
        "bayes_nodes": bayes_nodes,
        "bayes_items": bayes_items,
        "beliefs": BeliefState(items=item_beliefs),
        "item_beliefs": item_beliefs,
        "metrics": {
            "total_expected_bdi": max(0.0, min(63.0, total_expected_bdi)),
            "top_uncertain_items": top_uncertain_items,
            "last_ig_estimates": {
                item_id: round(float(bayes_items[item_id].uncertainty) * (0.6 + float(bayes_items[item_id].presence_prob)), 6)
                for item_id in range(1, 22)
            },
            "coverage": round(
                float(sum(1 for item in bayes_items.values() if item.presence_prob >= 0.25 or item.presence_prob <= 0.10))
                / 21.0,
                6,
            ),
            "mean_entropy": round(
                sum(float(item_beliefs[item_id].entropy) for item_id in range(1, 22)) / 21.0,
                6,
            ),
        },
        "risk_prob": float(bayes_nodes["risk"].probability),
        "risk_flag": bool(bayes_nodes["risk"].probability >= 0.38 or state.get("risk_flag", False)),
        "turn_trace": turn_trace,
    }
