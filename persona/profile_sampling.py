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
        "hedge_rate": _env_float("SIM_HEDGE_RATE", 0.65),
        "normalization_rate": _env_float("SIM_NORMALIZATION_RATE", 0.45),
        "context_anchor_rate": _env_float("SIM_CONTEXT_ANCHOR_RATE", 0.55),
        "direct_answer_rate": _env_float("SIM_DIRECT_ANSWER_RATE", 0.78),
    }


def _jitter_behavior(base: Dict[str, float], rng: random.Random) -> Dict[str, float]:
    def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, value))

    style = _style_defaults()
    return {
        "evasiveness": round(_clip(float(base["evasiveness"]) + rng.uniform(-0.08, 0.08)), 3),
        "verbosity": round(_clip(float(base["verbosity"]) + rng.uniform(-0.08, 0.08)), 3),
        "contradiction": round(_clip(float(base["contradiction"]) + rng.uniform(-0.04, 0.04)), 3),
        "affect_volatility": round(_clip(float(base["affect_volatility"]) + rng.uniform(-0.06, 0.06)), 3),
        "hedge_rate": round(_clip(float(style["hedge_rate"]) + rng.uniform(-0.05, 0.05)), 3),
        "normalization_rate": round(_clip(float(style["normalization_rate"]) + rng.uniform(-0.05, 0.05)), 3),
        "context_anchor_rate": round(_clip(float(style["context_anchor_rate"]) + rng.uniform(-0.05, 0.05)), 3),
        "direct_answer_rate": round(_clip(float(style["direct_answer_rate"]) + rng.uniform(-0.04, 0.04)), 3),
    }


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


def _sample_bdi_scores_for_family(family: str, rng: random.Random, severity: str = "moderate") -> Dict[int, int]:
    blueprint = {**FAMILY_BLUEPRINTS[family], "family": family}
    depressed = bool(blueprint["depressed"])
    tier = SEVERITY_TIERS[severity]
    module_latents, module_emphasis = _sample_module_latents(blueprint, severity=severity, rng=rng)
    scores = _sample_item_scores_from_modules(
        blueprint,
        severity=severity,
        depressed=depressed,
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

    profiles: List[PersonaProfile] = []
    for idx in range(1, count + 1):
        family = rng.choices(family_names, weights=family_weights, k=1)[0]
        blueprint = FAMILY_BLUEPRINTS[family]
        generation_seed = (seed * 1000) + idx
        local_rng = random.Random(generation_seed)

        if bool(blueprint["depressed"]):
            severity = local_rng.choices(severity_names, weights=severity_weights, k=1)[0]
        else:
            severity = "minimal"

        scores = _sample_bdi_scores_for_family(family, local_rng, severity=severity)
        behavior = _jitter_behavior(dict(blueprint["behavior"]), local_rng)

        profiles.append(
            PersonaProfile(
                persona_id=str(idx),
                split="eval",
                family=family,
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
