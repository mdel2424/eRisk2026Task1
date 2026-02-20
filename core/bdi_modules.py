from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from core.state import ItemBelief

MODULE_NAMES: Dict[int, str] = {
    0: "Frame/Baseline",
    1: "Affect and Reward",
    2: "Future Orientation and Meaning",
    3: "Self Evaluation and Moral Emotions",
    4: "Decision Making and Cognition",
    5: "Activation, Energy, Psychomotor",
    6: "Sleep and Appetite",
    7: "Somatic and Health Related Affect",
    8: "Interpersonal Functioning and Interest",
    9: "Safety Screen",
}

MODULE_GOALS: Dict[int, str] = {
    0: "Set baseline and two-week frame.",
    1: "Assess sadness, pleasure, crying, irritability tone, and emotional reactivity.",
    2: "Assess pessimism, hope, future expectancy, and failure framing.",
    3: "Assess guilt, self-criticism, worthlessness, and punishment beliefs.",
    4: "Assess concentration, decisional friction, and cognitive load.",
    5: "Assess activation, fatigue, agitation/slowing, and productivity impact.",
    6: "Assess sleep and appetite changes versus baseline.",
    7: "Assess fatigue-like somatic burden, health-linked distress, and libido change.",
    8: "Assess social withdrawal, engagement loss, and interpersonal payoff.",
    9: "Assess safety-related wish-for-escape or passive death ideation indirectly.",
}

MODULE_TO_ITEMS: Dict[int, List[int]] = {
    1: [1, 4, 10, 12, 17],
    2: [2, 3],
    3: [3, 5, 6, 7, 8, 14],
    4: [13, 19],
    5: [11, 15, 20],
    6: [16, 18],
    7: [20, 21],
    8: [12, 17],
    9: [9],
}

ITEM_TO_MODULES: Dict[int, List[int]] = {}
for _module_id, _item_ids in MODULE_TO_ITEMS.items():
    for _item_id in _item_ids:
        ITEM_TO_MODULES.setdefault(_item_id, []).append(_module_id)

MODULE_WEIGHTS: Dict[int, float] = {
    1: 1.0,
    2: 1.1,
    3: 1.15,
    4: 1.0,
    5: 1.05,
    6: 1.0,
    7: 0.9,
    8: 0.95,
    9: 1.35,
}

NODE_ALLOWED_MODULES: Dict[str, List[int]] = {
    "cognitive": [1, 2, 3, 4],
    "somatic": [5, 6, 7, 8],
    "risk": [9],
}


def allowed_modules_for_node(node_name: str) -> List[int]:
    return list(NODE_ALLOWED_MODULES.get(str(node_name).strip().lower(), [1, 2, 3, 4]))


def modules_for_item(item_id: int) -> List[int]:
    try:
        resolved = int(item_id)
    except (TypeError, ValueError):
        return []
    return list(ITEM_TO_MODULES.get(resolved, []))


def _support_count_from_belief(value) -> int:
    if isinstance(value, ItemBelief):
        return int(value.support_count)
    if isinstance(value, dict):
        try:
            return int(value.get("support_count", 0))
        except (TypeError, ValueError):
            return 0
    return 0


def _unresolved_ratio(module_id: int, item_beliefs: dict) -> float:
    module_items = MODULE_TO_ITEMS.get(module_id, [])
    if not module_items:
        return 1.0
    unresolved = 0
    for item_id in module_items:
        support_count = _support_count_from_belief(item_beliefs.get(item_id))
        if support_count <= 0:
            unresolved += 1
    return float(unresolved) / float(len(module_items))


def choose_target_module(
    node_name: str,
    target_items: Sequence[int],
    item_beliefs: dict,
) -> int:
    allowed = allowed_modules_for_node(node_name)
    if not allowed:
        return 1

    candidates: List[int] = []
    normalized_target_items: List[int] = []
    for item_id in target_items:
        try:
            normalized_target_items.append(int(item_id))
        except (TypeError, ValueError):
            continue
    for item_id in normalized_target_items:
        for module_id in modules_for_item(item_id):
            if module_id in allowed and module_id not in candidates:
                candidates.append(module_id)
    if not candidates:
        candidates = list(allowed)

    best_module = candidates[0]
    best_key: Tuple[int, float, int] = (-1, -1.0, -best_module)
    for module_id in sorted(candidates):
        overlap = 0
        for item_id in normalized_target_items:
            if module_id in modules_for_item(item_id):
                overlap += 1
        unresolved = _unresolved_ratio(module_id, item_beliefs)
        key = (overlap, unresolved, -module_id)
        if key > best_key:
            best_key = key
            best_module = module_id
    return best_module


def validate_module_map() -> None:
    missing = [item_id for item_id in range(1, 22) if item_id not in ITEM_TO_MODULES]
    if missing:
        raise ValueError(f"BDI module mapping missing items: {missing}")


validate_module_map()
