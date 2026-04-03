from __future__ import annotations

from typing import Dict, List


CLUSTER_TO_ITEMS: Dict[str, List[int]] = {
    "cognitive_affective": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 17, 19],
    "somatic_vegetative": [11, 15, 16, 18, 20, 21],
    "risk": [9],
    "negative_self_schema": [5, 6, 7, 8, 14],
    "physiological_disruption": [15, 16, 18, 20, 21],
}

ROUTE_TO_CLUSTER: Dict[str, str] = {
    "cognitive": "cognitive_affective",
    "somatic": "somatic_vegetative",
    "risk": "risk",
}

CLUSTER_TO_ROUTE: Dict[str, str] = {
    "cognitive_affective": "cognitive",
    "somatic_vegetative": "somatic",
    "risk": "risk",
    "negative_self_schema": "cognitive",
    "physiological_disruption": "somatic",
}

NODE_PRIORS: Dict[str, float] = {
    "cognitive_affective": 0.18,
    "somatic_vegetative": 0.18,
    "risk": 0.06,
    "negative_self_schema": 0.14,
    "physiological_disruption": 0.16,
}

NODE_LEAKS: Dict[str, float] = {
    "cognitive_affective": 0.08,
    "somatic_vegetative": 0.08,
    "risk": 0.03,
    "negative_self_schema": 0.05,
    "physiological_disruption": 0.05,
}

ITEM_PARENT_WEIGHTS: Dict[int, Dict[str, float]] = {
    1: {"cognitive_affective": 0.62},
    2: {"cognitive_affective": 0.74},
    3: {"cognitive_affective": 0.72},
    4: {"cognitive_affective": 0.64, "somatic_vegetative": 0.18},
    5: {"cognitive_affective": 0.44, "negative_self_schema": 0.48},
    6: {"cognitive_affective": 0.34, "negative_self_schema": 0.54},
    7: {"cognitive_affective": 0.32, "negative_self_schema": 0.60},
    8: {"cognitive_affective": 0.36, "negative_self_schema": 0.58},
    9: {"risk": 0.92, "negative_self_schema": 0.18},
    10: {"cognitive_affective": 0.42},
    11: {"somatic_vegetative": 0.42},
    12: {"cognitive_affective": 0.46, "somatic_vegetative": 0.18},
    13: {"cognitive_affective": 0.50},
    14: {"cognitive_affective": 0.44, "negative_self_schema": 0.66},
    15: {"somatic_vegetative": 0.62, "physiological_disruption": 0.34},
    16: {"somatic_vegetative": 0.66, "physiological_disruption": 0.48},
    17: {"cognitive_affective": 0.46, "somatic_vegetative": 0.12},
    18: {"somatic_vegetative": 0.64, "physiological_disruption": 0.52},
    19: {"cognitive_affective": 0.54, "somatic_vegetative": 0.14},
    20: {"somatic_vegetative": 0.74, "physiological_disruption": 0.56},
    21: {"somatic_vegetative": 0.44, "physiological_disruption": 0.40},
}


def route_for_cluster(cluster_name: str) -> str:
    return CLUSTER_TO_ROUTE.get(str(cluster_name or "").strip().lower(), "cognitive")


def cluster_for_route(route_name: str) -> str:
    return ROUTE_TO_CLUSTER.get(str(route_name or "").strip().lower(), "cognitive_affective")


def cluster_items(cluster_name: str) -> List[int]:
    return list(CLUSTER_TO_ITEMS.get(str(cluster_name or "").strip().lower(), []))


def bounded_uncertainty(probability: float) -> float:
    return max(0.0, min(1.0, 1.0 - abs((2.0 * float(probability)) - 1.0)))


def noisy_or_probability(
    *,
    parent_probs: Dict[str, float],
    parent_weights: Dict[str, float],
    leak: float,
) -> float:
    residual = 1.0 - max(0.0, min(1.0, float(leak)))
    for parent_name, weight in parent_weights.items():
        parent_prob = max(0.0, min(1.0, float(parent_probs.get(parent_name, 0.0))))
        residual *= 1.0 - (parent_prob * max(0.0, min(1.0, float(weight))))
    return max(0.0, min(1.0, 1.0 - residual))

