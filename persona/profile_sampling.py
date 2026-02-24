from __future__ import annotations

import os
import random
from dataclasses import replace
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


def _empty_scores() -> Dict[int, int]:
    return {item_id: 0 for item_id in range(1, 22)}


def _sample_depressed_score(rng: random.Random) -> int:
    severity = rng.choices(
        population=["mild", "moderate", "severe"],
        weights=[0.42, 0.42, 0.16],
        k=1,
    )[0]
    if severity == "mild":
        return rng.choice([1, 1, 2])
    if severity == "moderate":
        return rng.choice([1, 2, 2, 3])
    return rng.choice([2, 2, 3, 3])


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(0.0, min(1.0, value))


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


def _sample_bdi_scores_for_family(family: str, rng: random.Random) -> Dict[int, int]:
    blueprint = FAMILY_BLUEPRINTS[family]
    depressed = bool(blueprint["depressed"])
    core_items = list(blueprint["core_items"])
    secondary_items = list(blueprint["secondary_items"])
    risk_prob = float(blueprint["risk_prob"])

    scores = _empty_scores()

    if depressed:
        for item_id in core_items:
            scores[item_id] = _sample_depressed_score(rng)
        secondary_k = rng.randint(2, min(5, len(secondary_items)))
        for item_id in rng.sample(secondary_items, k=secondary_k):
            scores[item_id] = max(scores[item_id], rng.choice([1, 2, 2, 3]))
        if rng.random() < risk_prob:
            scores[9] = max(scores[9], rng.choice([1, 2, 3]))
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

    profiles: List[PersonaProfile] = []
    for idx in range(1, count + 1):
        family = rng.choices(family_names, weights=family_weights, k=1)[0]
        blueprint = FAMILY_BLUEPRINTS[family]
        generation_seed = (seed * 1000) + idx
        local_rng = random.Random(generation_seed)

        scores = _sample_bdi_scores_for_family(family, local_rng)
        behavior = _jitter_behavior(dict(blueprint["behavior"]), local_rng)

        profiles.append(
            PersonaProfile(
                persona_id=str(idx),
                split="train",
                family=family,
                bdi_scores=scores,
                depressed=bool(blueprint["depressed"]),
                source="synthetic",
                has_ground_truth=True,
                behavior_params=behavior,
                template_bank="train_bank_v1",
                generation_seed=generation_seed,
                generator_version=generator_version,
            )
        )

    return profiles


def _split_counts(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    train_ratio, val_ratio, _ = ratios
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    test_n = n - train_n - val_n
    if n >= 3:
        train_n = max(1, train_n)
        val_n = max(1, val_n)
        test_n = max(1, n - train_n - val_n)
        while train_n + val_n + test_n > n:
            if train_n >= val_n and train_n >= test_n and train_n > 1:
                train_n -= 1
            elif val_n >= test_n and val_n > 1:
                val_n -= 1
            elif test_n > 1:
                test_n -= 1
            else:
                break
        while train_n + val_n + test_n < n:
            train_n += 1
    return train_n, val_n, test_n


def assign_splits(
    pool: List[PersonaProfile],
    seed: int,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> Dict[str, List[PersonaProfile]]:
    rng = random.Random(seed + 17)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    train_n, val_n, _ = _split_counts(len(shuffled), ratios)

    split_map: Dict[str, List[PersonaProfile]] = {
        "train": [replace(item, split="train", template_bank="train_bank_v1") for item in shuffled[:train_n]],
        "val": [
            replace(item, split="val", template_bank="val_bank_v1")
            for item in shuffled[train_n : train_n + val_n]
        ],
        "test": [replace(item, split="test", template_bank="test_bank_v1") for item in shuffled[train_n + val_n :]],
    }

    return split_map


def build_split_profiles(
    count: int,
    seed: int,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> Dict[str, List[PersonaProfile]]:
    generator_version = os.getenv("SIM_GENERATOR_VERSION", "sim_v3").strip() or "sim_v3"
    pool = generate_persona_pool(count=count, seed=seed, generator_version=generator_version)
    split_profiles = assign_splits(pool=pool, seed=seed, ratios=ratios)

    enforce_disjoint = os.getenv("SIM_TEMPLATE_DISJOINT_ENFORCE", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    if enforce_disjoint:
        from persona.sim_behavior import validate_template_disjointness

        report = validate_template_disjointness()
        if not bool(report.get("strict_pass", False)):
            raise ValueError(f"Split template banks are not disjoint: {report}")

    return {
        "synthetic_train": split_profiles["train"],
        "synthetic_val": split_profiles["val"],
        "synthetic_test": split_profiles["test"],
    }


def generate_persona_profiles(count: int, seed: int = 42) -> List[PersonaProfile]:
    split_profiles = build_split_profiles(count=count, seed=seed)
    return split_profiles["synthetic_train"] + split_profiles["synthetic_val"] + split_profiles["synthetic_test"]


def split_synthetic_profiles(
    profiles: List[PersonaProfile],
    seed: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Dict[str, List[PersonaProfile]]:
    if profiles and all(getattr(profile, "split", None) in {"train", "val", "test"} for profile in profiles):
        return {
            "synthetic_train": [profile for profile in profiles if profile.split == "train"],
            "synthetic_val": [profile for profile in profiles if profile.split == "val"],
            "synthetic_test": [profile for profile in profiles if profile.split == "test"],
        }

    ratios = (train_ratio, val_ratio, max(0.0, 1.0 - train_ratio - val_ratio))
    assigned = assign_splits(pool=profiles, seed=seed, ratios=ratios)
    return {
        "synthetic_train": assigned["train"],
        "synthetic_val": assigned["val"],
        "synthetic_test": assigned["test"],
    }
