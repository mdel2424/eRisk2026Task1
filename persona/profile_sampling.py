from __future__ import annotations

import os
import random
from typing import Dict, List

from core.bdi_modules import ITEM_TO_MODULES, MODULE_TO_ITEMS
from persona.profiles import PersonaProfile

FAMILY_BLUEPRINTS: Dict[str, Dict[str, object]] = {
    "somatic_evasive": {
        "depressed": True,
        "core_items": [11, 15, 16, 18, 20],
        "secondary_items": [4, 12, 19],
        "risk_prob": 0.08,
        "behavior": {"evasiveness": 0.62, "verbosity": 0.40, "contradiction": 0.12, "affect_volatility": 0.28},
    },
    "cognitive_ruminative": {
        "depressed": True,
        "core_items": [2, 3, 5, 7, 8, 14, 19],
        "secondary_items": [4, 15, 16],
        "risk_prob": 0.12,
        "behavior": {"evasiveness": 0.48, "verbosity": 0.52, "contradiction": 0.10, "affect_volatility": 0.22},
    },
    "mixed_moderate": {
        "depressed": True,
        "core_items": [2, 4, 12, 15, 16, 19, 20],
        "secondary_items": [3, 5, 8, 14, 18],
        "risk_prob": 0.10,
        "behavior": {"evasiveness": 0.45, "verbosity": 0.50, "contradiction": 0.08, "affect_volatility": 0.24},
    },
    "functional_masked": {
        "depressed": True,
        "core_items": [4, 12, 15, 19, 20],
        "secondary_items": [2, 3, 8, 14, 16],
        "risk_prob": 0.06,
        "behavior": {"evasiveness": 0.70, "verbosity": 0.42, "contradiction": 0.14, "affect_volatility": 0.18},
    },
    "risk_leaning": {
        "depressed": True,
        "core_items": [2, 5, 8, 9, 14],
        "secondary_items": [3, 4, 12, 15, 19],
        "risk_prob": 0.70,
        "behavior": {"evasiveness": 0.58, "verbosity": 0.38, "contradiction": 0.10, "affect_volatility": 0.32},
    },
    "control_stressed": {
        "depressed": False,
        "core_items": [15, 16, 19],
        "secondary_items": [1, 2, 20],
        "risk_prob": 0.01,
        "behavior": {"evasiveness": 0.35, "verbosity": 0.50, "contradiction": 0.05, "affect_volatility": 0.14},
    },
    "control_neutral": {
        "depressed": False,
        "core_items": [4, 15],
        "secondary_items": [1, 19],
        "risk_prob": 0.0,
        "behavior": {"evasiveness": 0.30, "verbosity": 0.45, "contradiction": 0.03, "affect_volatility": 0.10},
    },
}

FAMILY_SAMPLING_WEIGHTS: Dict[str, float] = {
    "somatic_evasive": 0.16,
    "cognitive_ruminative": 0.16,
    "mixed_moderate": 0.16,
    "functional_masked": 0.12,
    "risk_leaning": 0.10,
    "control_stressed": 0.18,
    "control_neutral": 0.12,
}

FAMILY_SUBTYPE_CATALOG: Dict[str, Dict[str, Dict[str, Dict[int, float]]]] = {
    "somatic_evasive": {
        "irritable_sleep_fatigue": {
            "module_bias": {5: 0.10, 6: 0.18, 8: 0.10},
            "item_bias": {16: 0.26, 17: 0.22, 18: 0.10, 19: 0.10, 20: 0.22},
        },
        "appetite_variability_low_libido": {
            "module_bias": {6: 0.16, 7: 0.14},
            "item_bias": {16: 0.10, 18: 0.22, 20: 0.10, 21: 0.20},
        },
        "restless_slowed": {
            "module_bias": {5: 0.14, 6: 0.10},
            "item_bias": {11: 0.22, 15: 0.12, 16: 0.10, 20: 0.16},
        },
    },
    "cognitive_ruminative": {
        "burdened_failure_tearful": {
            "module_bias": {1: 0.10, 3: 0.14},
            "item_bias": {3: 0.18, 5: 0.10, 10: 0.18, 14: 0.14},
        },
        "flat_worthlessness": {
            "module_bias": {1: 0.10, 3: 0.14},
            "item_bias": {4: 0.18, 7: 0.14, 8: 0.08, 14: 0.22},
        },
        "selfcritical_indecisive": {
            "module_bias": {3: 0.12, 4: 0.14},
            "item_bias": {7: 0.08, 8: 0.22, 13: 0.12, 19: 0.16},
        },
    },
    "mixed_moderate": {
        "practical_anhedonia_fatigue": {
            "module_bias": {1: 0.10, 5: 0.12},
            "item_bias": {4: 0.14, 15: 0.12, 20: 0.16},
        },
        "social_withdrawal_contextual": {
            "module_bias": {1: 0.06, 8: 0.14},
            "item_bias": {12: 0.16, 17: 0.14, 20: 0.08},
        },
        "cognitive_somatic_blend": {
            "module_bias": {3: 0.08, 4: 0.10, 6: 0.08},
            "item_bias": {14: 0.12, 18: 0.12, 19: 0.16},
        },
    },
    "functional_masked": {
        "practical_masked_strain": {
            "module_bias": {4: 0.10, 5: 0.10},
            "item_bias": {15: 0.14, 19: 0.14, 20: 0.10},
        },
        "caregiving_masked_overload": {
            "module_bias": {5: 0.10, 8: 0.10},
            "item_bias": {12: 0.12, 17: 0.10, 20: 0.14},
        },
        "detached_high_functioning": {
            "module_bias": {1: 0.08, 8: 0.08},
            "item_bias": {4: 0.14, 12: 0.10, 19: 0.10},
        },
    },
    "risk_leaning": {
        "passive_escape_overwhelmed": {
            "module_bias": {2: 0.12, 5: 0.06, 9: 0.18},
            "item_bias": {2: 0.16, 9: 0.24, 20: 0.10},
        },
        "burden_punishment_hopeless": {
            "module_bias": {2: 0.10, 3: 0.12, 9: 0.16},
            "item_bias": {6: 0.16, 9: 0.22, 14: 0.18},
        },
        "wish_not_wake_flattened": {
            "module_bias": {1: 0.08, 2: 0.12, 9: 0.16},
            "item_bias": {2: 0.16, 4: 0.12, 9: 0.24},
        },
    },
    "control_stressed": {
        "stressed_practical_minimizing": {
            "module_bias": {5: 0.06, 6: 0.04},
            "item_bias": {15: 0.08, 17: 0.06, 19: 0.08},
        },
        "caregiving_busy_but_steady": {
            "module_bias": {5: 0.06, 8: 0.04},
            "item_bias": {12: 0.08, 20: 0.08},
        },
        "deadline_pressure_recovering": {
            "module_bias": {4: 0.06, 5: 0.06},
            "item_bias": {15: 0.06, 19: 0.08},
        },
    },
    "control_neutral": {
        "routine_stable": {"module_bias": {}, "item_bias": {}},
        "socially_engaged_steady": {"module_bias": {8: -0.04}, "item_bias": {12: -0.06, 17: -0.06}},
        "lightly_busy_stable": {"module_bias": {5: 0.02, 6: 0.02}, "item_bias": {15: 0.04}},
    },
}

FAMILY_CONTEXT_OPTIONS: Dict[str, List[str]] = {
    "somatic_evasive": ["workload", "caregiving", "health_stress", "routine_stable"],
    "cognitive_ruminative": ["school", "workload", "financial_pressure", "social_isolation", "relationship_strain"],
    "mixed_moderate": ["workload", "caregiving", "financial_pressure", "relationship_strain"],
    "functional_masked": ["workload", "caregiving", "financial_pressure", "routine_stable"],
    "risk_leaning": ["relationship_strain", "financial_pressure", "social_isolation", "health_stress"],
    "control_stressed": ["workload", "school", "caregiving", "routine_stable"],
    "control_neutral": ["routine_stable", "workload", "relationship_strain"],
}

FAMILY_STYLE_OPTIONS: Dict[str, List[str]] = {
    "somatic_evasive": ["terse_guarded", "minimizing_practical", "hedged_uncertain"],
    "cognitive_ruminative": ["contextual_reflective", "hedged_uncertain", "open_but_flat"],
    "mixed_moderate": ["contextual_reflective", "open_but_flat", "minimizing_practical"],
    "functional_masked": ["terse_guarded", "minimizing_practical", "open_but_flat"],
    "risk_leaning": ["terse_guarded", "hedged_uncertain", "contextual_reflective"],
    "control_stressed": ["minimizing_practical", "open_but_flat", "contextual_reflective"],
    "control_neutral": ["open_but_flat", "minimizing_practical", "contextual_reflective"],
}

CONTEXT_TAG_NUDGES: Dict[str, Dict[str, Dict[int, float]]] = {
    "workload": {"module_bias": {4: 0.06, 5: 0.10, 8: 0.06}, "item_bias": {15: 0.08, 17: 0.10, 19: 0.08}},
    "school": {"module_bias": {2: 0.06, 4: 0.12}, "item_bias": {3: 0.08, 13: 0.08, 19: 0.10}},
    "caregiving": {"module_bias": {5: 0.10, 8: 0.08}, "item_bias": {12: 0.08, 20: 0.10}},
    "relationship_strain": {"module_bias": {1: 0.05, 3: 0.05, 8: 0.12}, "item_bias": {12: 0.10, 17: 0.12, 21: 0.08}},
    "health_stress": {"module_bias": {6: 0.08, 7: 0.14}, "item_bias": {18: 0.12, 20: 0.10, 21: 0.10}},
    "financial_pressure": {"module_bias": {2: 0.08, 3: 0.06, 5: 0.04}, "item_bias": {2: 0.08, 3: 0.08, 14: 0.08}},
    "social_isolation": {"module_bias": {1: 0.04, 8: 0.14}, "item_bias": {4: 0.08, 12: 0.12, 17: 0.12}},
    "routine_stable": {"module_bias": {1: -0.04, 5: -0.04, 8: -0.04}, "item_bias": {12: -0.04, 15: -0.04, 20: -0.04}},
}

STYLE_TAG_BEHAVIOR_SHIFTS: Dict[str, Dict[str, float]] = {
    "terse_guarded": {"evasiveness": 0.08, "verbosity": -0.12, "context_anchor_rate": -0.03},
    "contextual_reflective": {"verbosity": 0.10, "context_anchor_rate": 0.12, "direct_answer_rate": 0.05},
    "minimizing_practical": {"normalization_rate": 0.05, "hedge_rate": 0.02, "context_anchor_rate": 0.06, "direct_answer_rate": 0.02},
    "open_but_flat": {"verbosity": 0.05, "affect_volatility": -0.08, "direct_answer_rate": 0.10, "hedge_rate": -0.03},
    "hedged_uncertain": {"hedge_rate": 0.05, "contradiction": 0.02, "direct_answer_rate": -0.02, "verbosity": 0.04},
}

# Clinical severity tiers matching standard settings.
# General population 0-13, primary care 10-25, outpatient 18-35, inpatient 25-50.
SEVERITY_TIERS: Dict[str, Dict[str, int | float]] = {
    "minimal": {"target": 7, "jitter": 5, "floor": 0, "ceiling": 13},
    "mild": {"target": 17, "jitter": 5, "floor": 10, "ceiling": 25},
    "moderate": {"target": 26, "jitter": 5, "floor": 18, "ceiling": 35},
    "severe": {"target": 38, "jitter": 6, "floor": 25, "ceiling": 50},
}

DEPRESSED_SEVERITY_WEIGHTS: Dict[str, float] = {
    "minimal": 0.15,
    "mild": 0.25,
    "moderate": 0.35,
    "severe": 0.25,
}

SEVERITY_MODULE_BASE: Dict[str, float] = {
    "minimal": 0.65,
    "mild": 1.05,
    "moderate": 1.65,
    "severe": 2.25,
}

CONTROL_MODULE_BASE: Dict[str, float] = {
    "minimal": 0.25,
    "mild": 0.35,
    "moderate": 0.45,
    "severe": 0.55,
}

MODULE_BACKGROUND_WEIGHT: Dict[str, float] = {
    "minimal": 0.10,
    "mild": 0.14,
    "moderate": 0.18,
    "severe": 0.22,
}

MODULE_JITTER_BY_SEVERITY: Dict[str, float] = {
    "minimal": 0.18,
    "mild": 0.22,
    "moderate": 0.28,
    "severe": 0.32,
}

ITEM_JITTER_BY_SEVERITY: Dict[str, float] = {
    "minimal": 0.35,
    "mild": 0.40,
    "moderate": 0.45,
    "severe": 0.50,
}

MODULE_CORE_WEIGHT = 1.0
MODULE_SECONDARY_WEIGHT = 0.55


def _empty_scores() -> Dict[int, int]:
    return {item_id: 0 for item_id in range(1, 22)}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _module_ids() -> List[int]:
    return sorted(MODULE_TO_ITEMS.keys())


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(0.0, min(1.0, value))


def _env_int(name: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        value = int(default)
    else:
        try:
            value = int(raw)
        except ValueError:
            value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    if maximum is not None:
        value = min(int(maximum), value)
    return value


def _style_defaults() -> Dict[str, float]:
    return {
        "hedge_rate": _env_float("SIM_HEDGE_RATE", 0.52),
        "normalization_rate": _env_float("SIM_NORMALIZATION_RATE", 0.18),
        "context_anchor_rate": _env_float("SIM_CONTEXT_ANCHOR_RATE", 0.50),
        "direct_answer_rate": _env_float("SIM_DIRECT_ANSWER_RATE", 0.86),
    }


def _stable_token_seed(token: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(str(token)))


def _shuffled_cycle(values: List[str], *, seed: int, namespace: str) -> List[str]:
    cycle = list(values)
    if len(cycle) <= 1:
        return cycle
    rng = random.Random((int(seed) * 9973) + _stable_token_seed(namespace))
    rng.shuffle(cycle)
    return cycle


def _merge_bias_maps(*bias_maps: Dict[int, float]) -> Dict[int, float]:
    merged: Dict[int, float] = {}
    for mapping in bias_maps:
        for key, value in mapping.items():
            merged[int(key)] = float(merged.get(int(key), 0.0)) + float(value)
    return merged


def _subtype_config(family: str, subtype_tag: str) -> Dict[str, Dict[int, float]]:
    family_catalog = FAMILY_SUBTYPE_CATALOG.get(family, {})
    config = dict(family_catalog.get(subtype_tag, {}))
    module_bias = {int(key): float(value) for key, value in dict(config.get("module_bias", {})).items()}
    item_bias = {int(key): float(value) for key, value in dict(config.get("item_bias", {})).items()}
    return {"module_bias": module_bias, "item_bias": item_bias}


def _context_config(context_tag: str) -> Dict[str, Dict[int, float]]:
    config = dict(CONTEXT_TAG_NUDGES.get(context_tag, {}))
    module_bias = {int(key): float(value) for key, value in dict(config.get("module_bias", {})).items()}
    item_bias = {int(key): float(value) for key, value in dict(config.get("item_bias", {})).items()}
    return {"module_bias": module_bias, "item_bias": item_bias}


def _jitter_behavior(base: Dict[str, float], rng: random.Random, *, style_tag: str) -> Dict[str, float]:
    def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, value))

    style = _style_defaults()
    behavior = {
        "evasiveness": round(_clip(float(base["evasiveness"]) + rng.uniform(-0.08, 0.08)), 3),
        "verbosity": round(_clip(float(base["verbosity"]) + rng.uniform(-0.08, 0.08)), 3),
        "contradiction": round(_clip(float(base["contradiction"]) + rng.uniform(-0.04, 0.04)), 3),
        "affect_volatility": round(_clip(float(base["affect_volatility"]) + rng.uniform(-0.06, 0.06)), 3),
        "hedge_rate": round(_clip(float(style["hedge_rate"]) + rng.uniform(-0.05, 0.05)), 3),
        "normalization_rate": round(_clip(float(style["normalization_rate"]) + rng.uniform(-0.05, 0.05)), 3),
        "context_anchor_rate": round(_clip(float(style["context_anchor_rate"]) + rng.uniform(-0.05, 0.05)), 3),
        "direct_answer_rate": round(_clip(float(style["direct_answer_rate"]) + rng.uniform(-0.04, 0.04)), 3),
    }
    for key, delta in STYLE_TAG_BEHAVIOR_SHIFTS.get(style_tag, {}).items():
        behavior[key] = round(_clip(float(behavior.get(key, 0.0)) + float(delta)), 3)
    return behavior


def _ensure_risk_consistency(scores: Dict[int, int], rng: random.Random) -> None:
    risk = int(scores.get(9, 0))
    if risk <= 0:
        return

    if risk >= 2:
        cognitive_support = [2, 3, 5, 8, 14]
        rng.shuffle(cognitive_support)
        for item_id in cognitive_support[:2]:
            scores[item_id] = max(scores[item_id], rng.choice([2, 3]))


def _family_module_emphasis(blueprint: Dict[str, object]) -> Dict[int, float]:
    module_weights = {module_id: 0.0 for module_id in _module_ids()}

    for item_id in blueprint["core_items"]:
        modules = ITEM_TO_MODULES.get(int(item_id), [])
        if not modules:
            continue
        contribution = MODULE_CORE_WEIGHT / float(len(modules))
        for module_id in modules:
            module_weights[module_id] += contribution

    for item_id in blueprint["secondary_items"]:
        modules = ITEM_TO_MODULES.get(int(item_id), [])
        if not modules:
            continue
        contribution = MODULE_SECONDARY_WEIGHT / float(len(modules))
        for module_id in modules:
            module_weights[module_id] += contribution

    max_weight = max(module_weights.values(), default=0.0)
    if max_weight <= 0.0:
        return {module_id: 0.0 for module_id in _module_ids()}
    return {module_id: round(weight / max_weight, 6) for module_id, weight in module_weights.items()}


def _sample_risk_module_latent(
    *,
    depressed: bool,
    severity: str,
    risk_prob: float,
    base_anchor: float,
    rng: random.Random,
) -> float:
    if not depressed:
        return 0.0

    latent = 0.0
    if severity in {"moderate", "severe"}:
        if rng.random() < risk_prob:
            latent = max(base_anchor, rng.choice([1.0, 2.0, 3.0]) + rng.uniform(-0.20, 0.20))
    elif severity == "mild":
        if rng.random() < (risk_prob * 0.30):
            latent = max(base_anchor * 0.60, 1.0 + rng.uniform(-0.15, 0.15))

    return _clamp(latent, 0.0, 3.0)


def _sample_module_latents(
    blueprint: Dict[str, object],
    *,
    severity: str,
    module_biases: Dict[int, float],
    rng: random.Random,
) -> tuple[Dict[int, float], Dict[int, float]]:
    depressed = bool(blueprint["depressed"])
    module_emphasis = _family_module_emphasis(blueprint)
    base_value = (SEVERITY_MODULE_BASE if depressed else CONTROL_MODULE_BASE)[severity]
    background_weight = MODULE_BACKGROUND_WEIGHT[severity]
    noise_scale = MODULE_JITTER_BY_SEVERITY[severity]
    risk_prob = float(blueprint["risk_prob"])

    module_latents: Dict[int, float] = {}
    for module_id in _module_ids():
        emphasis = float(module_emphasis.get(module_id, 0.0))
        anchor = base_value * (background_weight + ((1.0 - background_weight) * emphasis))
        anchor += float(module_biases.get(module_id, 0.0))
        if module_id == 9:
            module_latents[module_id] = _sample_risk_module_latent(
                depressed=depressed,
                severity=severity,
                risk_prob=risk_prob,
                base_anchor=anchor,
                rng=rng,
            )
            continue
        if not depressed and emphasis <= 0.0:
            anchor = 0.0
        noisy_anchor = anchor + rng.uniform(-noise_scale, noise_scale)
        module_latents[module_id] = _clamp(noisy_anchor, 0.0, 3.0)

    return module_latents, module_emphasis


def _soft_bounds_for_item(item_id: int, module_latents: Dict[int, float]) -> tuple[int, int]:
    min_allowed = 0
    max_allowed = 3
    for module_id in ITEM_TO_MODULES.get(int(item_id), []):
        module_items = MODULE_TO_ITEMS.get(module_id, [])
        if len(module_items) < 2:
            continue
        module_center = int(round(float(module_latents.get(module_id, 0.0))))
        min_allowed = max(min_allowed, max(0, module_center - 1))
        max_allowed = min(max_allowed, min(3, module_center + 1))
    return min_allowed, max_allowed


def _item_priority(
    item_id: int,
    *,
    blueprint: Dict[str, object],
    module_emphasis: Dict[int, float],
    increasing: bool,
) -> float:
    core_items = {int(value) for value in blueprint["core_items"]}
    secondary_items = {int(value) for value in blueprint["secondary_items"]}
    item_modules = ITEM_TO_MODULES.get(int(item_id), [])
    avg_emphasis = (
        sum(float(module_emphasis.get(module_id, 0.0)) for module_id in item_modules) / float(len(item_modules))
        if item_modules
        else 0.0
    )

    if increasing:
        base = 1.0 + (1.8 * avg_emphasis)
        if item_id in core_items:
            base += 1.2
        elif item_id in secondary_items:
            base += 0.6
        return base

    base = 1.0 + (1.2 * (1.0 - avg_emphasis))
    if item_id not in core_items and item_id not in secondary_items:
        base += 0.8
    elif item_id in secondary_items:
        base += 0.35
    if item_id == 9 and blueprint.get("family") != "risk_leaning":
        base += 0.25
    return base


def _build_adjustment_candidates(
    blueprint: Dict[str, object],
    module_emphasis: Dict[int, float],
    *,
    increasing: bool,
) -> List[int]:
    candidates: List[int] = []
    for item_id in range(1, 22):
        weight = _item_priority(item_id, blueprint=blueprint, module_emphasis=module_emphasis, increasing=increasing)
        repeats = max(1, int(round(weight * 4.0)))
        candidates.extend([item_id] * repeats)
    return candidates


def _can_shift_item(
    scores: Dict[int, int],
    *,
    item_id: int,
    direction: int,
    family: str,
    module_latents: Dict[int, float],
) -> bool:
    current_value = int(scores.get(item_id, 0))
    next_value = current_value + direction
    if next_value < 0 or next_value > 3:
        return False
    if item_id == 9 and family == "risk_leaning" and direction < 0 and current_value >= 2:
        return False

    for module_id in ITEM_TO_MODULES.get(int(item_id), []):
        module_items = MODULE_TO_ITEMS.get(module_id, [])
        if len(module_items) < 2:
            continue
        candidate_scores = [next_value if member == item_id else int(scores.get(member, 0)) for member in module_items]
        if max(candidate_scores) - min(candidate_scores) > 1:
            return False
        module_center = int(round(float(module_latents.get(module_id, 0.0))))
        if any(abs(score - module_center) > 1 for score in candidate_scores):
            return False

    return True


def _enforce_module_soft_coupling(scores: Dict[int, int], module_latents: Dict[int, float]) -> None:
    for _ in range(3):
        for module_id, module_items in MODULE_TO_ITEMS.items():
            if len(module_items) < 2:
                continue
            module_center = int(round(float(module_latents.get(module_id, 0.0))))
            lower = max(0, module_center - 1)
            upper = min(3, module_center + 1)

            current_values = [max(lower, min(upper, int(scores.get(item_id, 0)))) for item_id in module_items]
            module_low = min(current_values)
            module_high = max(current_values)
            if module_high - module_low > 1:
                ceiling = module_low + 1
                current_values = [min(value, ceiling) for value in current_values]
                module_low = min(current_values)
                module_high = max(current_values)
                if module_high - module_low > 1:
                    floor = module_high - 1
                    current_values = [max(value, floor) for value in current_values]

            for item_id, value in zip(module_items, current_values):
                scores[item_id] = int(value)


def _sample_item_scores_from_modules(
    blueprint: Dict[str, object],
    *,
    severity: str,
    depressed: bool,
    item_biases: Dict[int, float],
    rng: random.Random,
    module_latents: Dict[int, float],
    module_emphasis: Dict[int, float],
) -> Dict[int, int]:
    core_items = {int(value) for value in blueprint["core_items"]}
    secondary_items = {int(value) for value in blueprint["secondary_items"]}
    scores = _empty_scores()
    item_noise = ITEM_JITTER_BY_SEVERITY[severity]

    for item_id in range(1, 22):
        item_modules = ITEM_TO_MODULES.get(item_id, [])
        if item_modules:
            weighted_sum = 0.0
            weight_total = 0.0
            for module_id in item_modules:
                weight = 0.20 + float(module_emphasis.get(module_id, 0.0))
                weighted_sum += float(module_latents.get(module_id, 0.0)) * weight
                weight_total += weight
            item_anchor = weighted_sum / max(weight_total, 1e-6)
        else:
            item_anchor = 0.0

        if item_id in core_items:
            item_anchor += 0.22
        elif item_id in secondary_items:
            item_anchor += 0.10

        item_anchor += float(item_biases.get(item_id, 0.0))

        if not depressed and item_anchor < 0.35:
            item_anchor *= 0.65

        raw_score = int(round(item_anchor + rng.uniform(-item_noise, item_noise)))
        lower, upper = _soft_bounds_for_item(item_id, module_latents)
        scores[item_id] = int(max(lower, min(upper, max(0, min(3, raw_score)))))

    _enforce_module_soft_coupling(scores, module_latents)
    return scores


def _adjust_depressed_total(
    scores: Dict[int, int],
    *,
    family: str,
    blueprint: Dict[str, object],
    module_emphasis: Dict[int, float],
    module_latents: Dict[int, float],
    rng: random.Random,
    target_total: int,
    target_jitter: int,
    target_blend: float,
    floor: int = 0,
    ceiling: int = 63,
) -> None:
    current_total = sum(int(value) for value in scores.values())
    sampled_target = target_total + (rng.randint(-target_jitter, target_jitter) if target_jitter > 0 else 0)
    sampled_target = max(floor, min(ceiling, sampled_target))
    desired_total = int(round((1.0 - target_blend) * current_total + target_blend * sampled_target))
    desired_total = max(floor, min(ceiling, desired_total))

    increase_candidates = _build_adjustment_candidates(blueprint, module_emphasis, increasing=True)
    decrease_candidates = _build_adjustment_candidates(blueprint, module_emphasis, increasing=False)

    if family != "risk_leaning":
        increase_candidates = [item_id for item_id in increase_candidates if item_id != 9] + [9]

    delta = desired_total - current_total
    if delta > 0:
        attempts = 0
        while delta > 0 and attempts < 1200:
            attempts += 1
            item_id = int(rng.choice(increase_candidates))
            if not _can_shift_item(scores, item_id=item_id, direction=1, family=family, module_latents=module_latents):
                continue
            scores[item_id] = int(scores.get(item_id, 0)) + 1
            delta -= 1
    elif delta < 0:
        attempts = 0
        while delta < 0 and attempts < 1200:
            attempts += 1
            item_id = int(rng.choice(decrease_candidates))
            if not _can_shift_item(scores, item_id=item_id, direction=-1, family=family, module_latents=module_latents):
                continue
            scores[item_id] = int(scores.get(item_id, 0)) - 1
            delta += 1

    _enforce_module_soft_coupling(scores, module_latents)


def _apply_minimal_precision_caps(scores: Dict[int, int], *, family: str, severity: str) -> None:
    if severity != "minimal":
        return

    original_scores = {item_id: int(scores.get(item_id, 0)) for item_id in range(1, 22)}

    def _rank_items(item_ids: set[int]) -> List[int]:
        return sorted(
            [item_id for item_id in item_ids if int(original_scores.get(item_id, 0)) > 0],
            key=lambda item_id: (-int(original_scores.get(item_id, 0)), item_id),
        )

    if family == "risk_leaning":
        nonrisk_core_ids = {2, 5, 8, 14}
        secondary_ids = {3, 4, 12, 15, 19}
        for item_id in range(1, 22):
            if item_id != 9:
                scores[item_id] = min(int(scores.get(item_id, 0)), 1)

        keep_nonrisk_ids: List[int] = []
        for item_id in _rank_items(nonrisk_core_ids)[:2]:
            keep_nonrisk_ids.append(item_id)
        for item_id in _rank_items(secondary_ids)[:1]:
            if item_id not in keep_nonrisk_ids:
                keep_nonrisk_ids.append(item_id)

        other_nonrisk_ids = {
            item_id for item_id in range(1, 22) if item_id != 9 and item_id not in nonrisk_core_ids and item_id not in secondary_ids
        }
        remaining_slots = max(0, 4 - len(keep_nonrisk_ids))
        for item_id in _rank_items(other_nonrisk_ids)[:remaining_slots]:
            if item_id not in keep_nonrisk_ids:
                keep_nonrisk_ids.append(item_id)

        keep_nonrisk_set = set(keep_nonrisk_ids)
        for item_id in range(1, 22):
            if item_id == 9:
                continue
            if item_id not in keep_nonrisk_set:
                scores[item_id] = 0
        return

    if family == "cognitive_ruminative":
        module34_item_ids = set(MODULE_TO_ITEMS.get(3, [])) | set(MODULE_TO_ITEMS.get(4, []))
        for item_id in range(1, 22):
            scores[item_id] = min(int(scores.get(item_id, 0)), 1)

        preferred_ids = _rank_items(module34_item_ids)
        spillover_ids = _rank_items(set(range(1, 22)) - module34_item_ids)

        keep_item_ids: List[int] = []
        if spillover_ids:
            keep_item_ids.append(spillover_ids[0])
        remaining_slots = max(0, 4 - len(keep_item_ids))
        for item_id in preferred_ids[:remaining_slots]:
            if item_id not in keep_item_ids:
                keep_item_ids.append(item_id)
        keep_item_ids = keep_item_ids[:4]
        keep_item_set = set(keep_item_ids)

        for item_id in range(1, 22):
            if item_id not in keep_item_set:
                scores[item_id] = 0


def _sample_bdi_scores_for_family(
    family: str,
    rng: random.Random,
    *,
    severity: str = "moderate",
    subtype_tag: str | None = None,
    context_tag: str | None = None,
) -> Dict[int, int]:
    blueprint = {**FAMILY_BLUEPRINTS[family], "family": family}
    depressed = bool(blueprint["depressed"])
    tier = SEVERITY_TIERS[severity]
    resolved_subtype = subtype_tag or next(iter(FAMILY_SUBTYPE_CATALOG[family].keys()))
    resolved_context = context_tag or FAMILY_CONTEXT_OPTIONS[family][0]
    subtype_config = _subtype_config(family, resolved_subtype)
    context_config = _context_config(resolved_context)
    module_biases = _merge_bias_maps(subtype_config["module_bias"], context_config["module_bias"])
    item_biases = _merge_bias_maps(subtype_config["item_bias"], context_config["item_bias"])

    module_latents, module_emphasis = _sample_module_latents(
        blueprint,
        severity=severity,
        module_biases=module_biases,
        rng=rng,
    )
    scores = _sample_item_scores_from_modules(
        blueprint,
        severity=severity,
        depressed=depressed,
        item_biases=item_biases,
        rng=rng,
        module_latents=module_latents,
        module_emphasis=module_emphasis,
    )

    if depressed:
        _adjust_depressed_total(
            scores,
            family=family,
            blueprint=blueprint,
            module_emphasis=module_emphasis,
            module_latents=module_latents,
            rng=rng,
            target_total=int(tier["target"]),
            target_jitter=int(tier["jitter"]),
            target_blend=0.85,
            floor=int(tier["floor"]),
            ceiling=int(tier["ceiling"]),
        )
    else:
        scores[9] = 0

    if family == "functional_masked":
        scores[1] = min(scores.get(1, 0), 1)
        scores[10] = min(scores.get(10, 0), 1)

    _ensure_risk_consistency(scores, rng)
    _enforce_module_soft_coupling(scores, module_latents)
    _apply_minimal_precision_caps(scores, family=family, severity=severity)

    if not depressed:
        capped_total = min(sum(scores.values()), 11)
        if sum(scores.values()) > capped_total:
            for item_id in sorted(scores.keys(), key=lambda x: scores[x], reverse=True):
                while sum(scores.values()) > capped_total and scores[item_id] > 0:
                    scores[item_id] -= 1

    return {item_id: int(max(0, min(3, score))) for item_id, score in scores.items()}


def generate_persona_pool(
    count: int,
    seed: int,
) -> List[PersonaProfile]:
    rng = random.Random(seed)
    count = max(1, int(count))
    family_names = list(FAMILY_SAMPLING_WEIGHTS.keys())
    family_weights = [float(FAMILY_SAMPLING_WEIGHTS[name]) for name in family_names]

    severity_names = list(DEPRESSED_SEVERITY_WEIGHTS.keys())
    severity_weights = [float(DEPRESSED_SEVERITY_WEIGHTS[n]) for n in severity_names]
    subtype_cycles = {
        family: _shuffled_cycle(list(FAMILY_SUBTYPE_CATALOG[family].keys()), seed=seed, namespace=f"{family}:subtype")
        for family in family_names
    }
    context_cycles = {
        family: _shuffled_cycle(list(FAMILY_CONTEXT_OPTIONS[family]), seed=seed, namespace=f"{family}:context")
        for family in family_names
    }
    style_cycles = {
        family: _shuffled_cycle(list(FAMILY_STYLE_OPTIONS[family]), seed=seed, namespace=f"{family}:style")
        for family in family_names
    }
    family_seen_counts = {family: 0 for family in family_names}

    profiles: List[PersonaProfile] = []
    for idx in range(1, count + 1):
        family = rng.choices(family_names, weights=family_weights, k=1)[0]
        blueprint = FAMILY_BLUEPRINTS[family]
        generation_seed = (seed * 1000) + idx
        local_rng = random.Random(generation_seed)
        family_index = int(family_seen_counts.get(family, 0))
        family_seen_counts[family] = family_index + 1
        subtype_cycle = subtype_cycles[family]
        context_cycle = context_cycles[family]
        style_cycle = style_cycles[family]
        subtype_tag = subtype_cycle[family_index % len(subtype_cycle)]
        context_tag = context_cycle[(family_index + (generation_seed % len(context_cycle))) % len(context_cycle)]
        style_tag = style_cycle[(family_index + (generation_seed % len(style_cycle))) % len(style_cycle)]

        if bool(blueprint["depressed"]):
            severity = local_rng.choices(severity_names, weights=severity_weights, k=1)[0]
        else:
            severity = "minimal"

        scores = _sample_bdi_scores_for_family(
            family,
            local_rng,
            severity=severity,
            subtype_tag=subtype_tag,
            context_tag=context_tag,
        )
        behavior = _jitter_behavior(dict(blueprint["behavior"]), local_rng, style_tag=style_tag)

        profiles.append(
            PersonaProfile(
                persona_id=str(idx),
                split="eval",
                family=family,
                severity_tier=severity,
                subtype_tag=subtype_tag,
                context_tag=context_tag,
                style_tag=style_tag,
                bdi_scores=scores,
                depressed=bool(blueprint["depressed"]),
                source="synthetic",
                has_ground_truth=True,
                behavior_params=behavior,
                template_bank="default",
                generation_seed=generation_seed,
            )
        )

    return profiles
