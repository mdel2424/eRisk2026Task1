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


def _depressed_target_config() -> tuple[int, int, float]:
    target = _env_int("SIM_DEPRESSED_TARGET_BDI", 30, minimum=0, maximum=63)
    jitter = _env_int("SIM_DEPRESSED_TARGET_JITTER", 4, minimum=0, maximum=16)
    blend = _env_float("SIM_DEPRESSED_TARGET_BLEND", 0.85)
    return target, jitter, blend


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
) -> None:
    current_total = sum(int(value) for value in scores.values())
    sampled_target = target_total + (rng.randint(-target_jitter, target_jitter) if target_jitter > 0 else 0)
    sampled_target = max(14, min(45, sampled_target))
    desired_total = int(round((1.0 - target_blend) * current_total + target_blend * sampled_target))
    desired_total = max(14, min(45, desired_total))

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
        target_total, target_jitter, target_blend = _depressed_target_config()
        _adjust_depressed_total(
            scores,
            family=family,
            rng=rng,
            target_total=target_total,
            target_jitter=target_jitter,
            target_blend=target_blend,
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


def _take_with_class_targets(
    depressed_pool: List[PersonaProfile],
    control_pool: List[PersonaProfile],
    target_total: int,
    *,
    require_both_classes: bool,
) -> List[PersonaProfile]:
    depressed_total = len(depressed_pool)
    control_total = len(control_pool)

    if target_total <= 0:
        return []

    if require_both_classes:
        if depressed_total == 0 or control_total == 0:
            raise ValueError(
                "Stratified strict split failed: both depressed and control personas are required in the pool."
            )
        if target_total < 2:
            raise ValueError(
                "Stratified strict split failed: holdout size must be >=2 to include both classes."
            )

    depressed_target = int(round(target_total * (depressed_total / max(1, depressed_total + control_total))))
    control_target = target_total - depressed_target

    if require_both_classes:
        depressed_target = max(1, depressed_target)
        control_target = max(1, control_target)

    depressed_target = min(depressed_target, depressed_total)
    control_target = min(control_target, control_total)

    while depressed_target + control_target < target_total:
        can_take_depressed = depressed_target < depressed_total
        can_take_control = control_target < control_total
        if can_take_depressed and (not can_take_control or depressed_target <= control_target):
            depressed_target += 1
        elif can_take_control:
            control_target += 1
        else:
            break

    while depressed_target + control_target > target_total:
        if depressed_target > control_target and depressed_target > (1 if require_both_classes else 0):
            depressed_target -= 1
        elif control_target > (1 if require_both_classes else 0):
            control_target -= 1
        else:
            break

    if require_both_classes and (depressed_target == 0 or control_target == 0):
        raise ValueError(
            "Stratified strict split failed: unable to satisfy both-class holdout with requested persona count."
        )

    selected = depressed_pool[:depressed_target] + control_pool[:control_target]
    if len(selected) != target_total:
        raise ValueError(
            "Stratified strict split failed: insufficient personas to satisfy requested split sizes."
        )
    return selected


def assign_splits(
    pool: List[PersonaProfile],
    seed: int,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> Dict[str, List[PersonaProfile]]:
    rng = random.Random(seed + 17)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    train_n, val_n, test_n = _split_counts(len(shuffled), ratios)

    strict = os.getenv("EVAL_STRATIFIED_STRICT", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
    holdout_n = val_n + test_n

    depressed_pool = [profile for profile in shuffled if bool(profile.depressed)]
    control_pool = [profile for profile in shuffled if not bool(profile.depressed)]

    holdout = _take_with_class_targets(
        depressed_pool,
        control_pool,
        holdout_n,
        require_both_classes=strict and holdout_n > 0,
    )
    holdout_ids = {profile.persona_id for profile in holdout}
    train_pool = [profile for profile in shuffled if profile.persona_id not in holdout_ids]

    if len(train_pool) != train_n:
        raise ValueError(
            "Stratified strict split failed: train split size mismatch; increase persona count or adjust ratios."
        )

    holdout_shuffled = holdout[:]
    rng.shuffle(holdout_shuffled)
    val_split = holdout_shuffled[:val_n]
    test_split = holdout_shuffled[val_n:]
    if len(val_split) != val_n or len(test_split) != test_n:
        raise ValueError("Stratified strict split failed: holdout allocation mismatch.")

    split_map: Dict[str, List[PersonaProfile]] = {
        "train": [replace(item, split="train", template_bank="train_bank_v1") for item in train_pool],
        "val": [replace(item, split="val", template_bank="val_bank_v1") for item in val_split],
        "test": [replace(item, split="test", template_bank="test_bank_v1") for item in test_split],
    }
    return split_map


def build_split_profiles(
    count: int,
    seed: int,
    split_seed: int | None = None,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> Dict[str, List[PersonaProfile]]:
    generator_version = os.getenv("SIM_GENERATOR_VERSION", "sim_v3").strip() or "sim_v3"
    pool = generate_persona_pool(count=count, seed=seed, generator_version=generator_version)
    effective_split_seed = int(split_seed) if split_seed is not None else int(seed)
    split_profiles = assign_splits(pool=pool, seed=effective_split_seed, ratios=ratios)

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
