from __future__ import annotations

import os
import random
from typing import Dict, List

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


def _empty_scores() -> Dict[int, int]:
    return {item_id: 0 for item_id in range(1, 22)}


def _sample_item_score(rng: random.Random, severity: str) -> int:
    """Sample a single BDI item score appropriate for the severity tier."""
    if severity == "minimal":
        return rng.choice([0, 0, 1, 1])
    if severity == "mild":
        return rng.choice([1, 1, 1, 2])
    if severity == "moderate":
        return rng.choices([1, 2, 3], weights=[0.35, 0.45, 0.20], k=1)[0]
    # severe
    return rng.choices([2, 3], weights=[0.45, 0.55], k=1)[0]


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


def _adjust_depressed_total(
    scores: Dict[int, int],
    *,
    family: str,
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

    blueprint = FAMILY_BLUEPRINTS[family]
    core_items = [int(item_id) for item_id in blueprint["core_items"]]
    secondary_items = [int(item_id) for item_id in blueprint["secondary_items"]]
    tertiary_items = [item_id for item_id in range(1, 22) if item_id not in core_items and item_id not in secondary_items]

    # Preserve risk signal behavior: non-risk families should not be pushed toward severe item 9.
    increase_candidates = [
        *(core_items * 3),
        *(secondary_items * 2),
        *tertiary_items,
    ]
    decrease_candidates = [
        *tertiary_items,
        *(secondary_items * 2),
        *(core_items * 3),
    ]

    if family != "risk_leaning":
        increase_candidates = [item_id for item_id in increase_candidates if item_id != 9] + [9]

    delta = desired_total - current_total
    if delta > 0:
        attempts = 0
        while delta > 0 and attempts < 500:
            attempts += 1
            item_id = int(rng.choice(increase_candidates))
            if int(scores.get(item_id, 0)) >= 3:
                continue
            scores[item_id] = int(scores.get(item_id, 0)) + 1
            delta -= 1
    elif delta < 0:
        attempts = 0
        while delta < 0 and attempts < 500:
            attempts += 1
            item_id = int(rng.choice(decrease_candidates))
            if int(scores.get(item_id, 0)) <= 0:
                continue
            if item_id == 9 and family == "risk_leaning" and int(scores.get(9, 0)) >= 2:
                continue
            scores[item_id] = int(scores.get(item_id, 0)) - 1
            delta += 1


def _sample_bdi_scores_for_family(family: str, rng: random.Random, severity: str = "moderate") -> Dict[int, int]:
    blueprint = FAMILY_BLUEPRINTS[family]
    depressed = bool(blueprint["depressed"])
    core_items = list(blueprint["core_items"])
    secondary_items = list(blueprint["secondary_items"])
    risk_prob = float(blueprint["risk_prob"])
    tier = SEVERITY_TIERS[severity]

    scores = _empty_scores()

    if depressed:
        # Core item activation varies by severity tier.
        core_activation = {"minimal": 0.45, "mild": 0.70, "moderate": 1.0, "severe": 1.0}[severity]
        for item_id in core_items:
            if rng.random() < core_activation:
                scores[item_id] = _sample_item_score(rng, severity)

        # Secondary items: fewer activated at lower severity.
        sec_activation = {"minimal": 0.25, "mild": 0.40, "moderate": 0.65, "severe": 0.80}[severity]
        for item_id in secondary_items:
            if rng.random() < sec_activation:
                scores[item_id] = max(scores[item_id], _sample_item_score(rng, severity))

        # Risk signal: scaled by severity.
        if severity in ("moderate", "severe"):
            if rng.random() < risk_prob:
                scores[9] = max(scores[9], rng.choice([1, 2, 3]))
        elif severity == "mild":
            if rng.random() < risk_prob * 0.3:
                scores[9] = max(scores[9], 1)

        _adjust_depressed_total(
            scores,
            family=family,
            rng=rng,
            target_total=int(tier["target"]),
            target_jitter=int(tier["jitter"]),
            target_blend=0.85,
            floor=int(tier["floor"]),
            ceiling=int(tier["ceiling"]),
        )
    else:
        for item_id in core_items:
            if rng.random() < 0.6:
                scores[item_id] = rng.choice([0, 1])
        for item_id in secondary_items:
            if rng.random() < 0.35:
                scores[item_id] = max(scores[item_id], 1)
        scores[9] = 0

    if family == "functional_masked":
        scores[1] = min(scores.get(1, 0), 1)
        scores[10] = min(scores.get(10, 0), 1)

    _ensure_risk_consistency(scores, rng)

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
    generator_version: str = "sim_v3",
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
                generator_version=generator_version,
            )
        )

    return profiles
