from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.evaluation import compute_metrics
from core.runtime_policy import resolve_detector_backend, resolve_persona_backend
from persona import PersonaProfile

from app.cli_common import _parse_bool

MANIFEST_SCHEMA_VERSION = 2


def _objective(metrics: Dict[str, float], max_turns: int, latency_lambda: float = 0.15) -> float:
    if not metrics:
        return 0.0
    binary_f1 = float(metrics.get("binary_f1", 0.0))
    avg_turns = float(metrics.get("avg_turns_to_decision", 0.0))
    normalized_turns = min(1.0, avg_turns / max(1, max_turns))
    return round(binary_f1 - (latency_lambda * normalized_turns), 4)


def _with_objective(metrics: Dict[str, float], max_turns: int) -> Dict[str, float]:
    if not metrics:
        return {}
    return {**metrics, "objective": _objective(metrics, max_turns=max_turns)}


def _select_primary_metrics(
    synthetic_val: Dict[str, float],
    synthetic_test: Dict[str, float],
    overall_labeled: Dict[str, float],
) -> Tuple[str, Dict[str, float]]:
    if overall_labeled:
        return "overall_labeled", overall_labeled
    if synthetic_test:
        return "synthetic_test", synthetic_test
    if synthetic_val:
        return "synthetic_val", synthetic_val
    return "overall_labeled", {}


def _resolve_fit_calibrator_policy(policy: str) -> bool:
    value = str(policy).strip().lower()
    if value == "on":
        return True
    if value == "off":
        return False
    detector_backend = resolve_detector_backend()
    persona_backend = resolve_persona_backend()
    return not (detector_backend == "openrouter" or persona_backend == "openrouter_sim")


def _strict_split_lock_enabled() -> bool:
    return _parse_bool(os.getenv("STRICT_SPLIT_LOCK", "1"))


def _manifest_payload(
    *,
    persona_count: int,
    seed: int,
    generator_version: str,
    train_profiles: List[PersonaProfile],
    val_profiles: List[PersonaProfile],
    test_profiles: List[PersonaProfile],
) -> Dict[str, Any]:
    def _profile_dict(profile: PersonaProfile) -> Dict[str, Any]:
        return {
            "persona_id": profile.persona_id,
            "split": profile.split,
            "family": profile.family,
            "source": profile.source,
            "has_ground_truth": profile.has_ground_truth,
            "depressed": profile.depressed,
            "bdi_scores": dict(profile.bdi_scores),
            "bdi_total": profile.bdi_total,
            "key_symptoms": profile.key_symptoms,
            "risk_signal": profile.has_risk_signal,
            "behavior_params": dict(profile.behavior_params),
            "template_bank": profile.template_bank,
            "generation_seed": profile.generation_seed,
            "generator_version": profile.generator_version,
        }

    all_profiles = train_profiles + val_profiles + test_profiles
    return {
        "run_config": {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "persona_count": persona_count,
            "seed": seed,
            "generator_version": generator_version,
            "split_ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
        },
        "split_counts": {
            "train": len(train_profiles),
            "val": len(val_profiles),
            "test": len(test_profiles),
        },
        "profiles": [_profile_dict(profile) for profile in all_profiles],
    }


def _manifest_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _enforce_manifest_lock(manifest_path: Path, current_payload: Dict[str, Any], current_hash: str) -> None:
    if not manifest_path.exists():
        return
    try:
        previous_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return

    prev_config = previous_payload.get("run_config", {})
    curr_config = current_payload.get("run_config", {})
    if prev_config != curr_config:
        return

    previous_hash = _manifest_hash(previous_payload)
    if previous_hash != current_hash:
        raise ValueError(
            "STRICT_SPLIT_LOCK failed: split manifest hash mismatch for identical run_config. "
            "Possible leakage/non-deterministic generation detected. "
            "If this is an intentional simulator update, bump SIM_GENERATOR_VERSION "
            "or remove outputs/persona_manifest_run_local.json."
        )


def _split_overlap_count(a: List[str], b: List[str]) -> int:
    return len(set(a).intersection(set(b)))


def _template_overlap_counts(
    train_profiles: List[PersonaProfile],
    val_profiles: List[PersonaProfile],
    test_profiles: List[PersonaProfile],
) -> Dict[str, int]:
    train_banks = [profile.template_bank for profile in train_profiles]
    val_banks = [profile.template_bank for profile in val_profiles]
    test_banks = [profile.template_bank for profile in test_profiles]
    return {
        "train_val": _split_overlap_count(train_banks, val_banks),
        "train_test": _split_overlap_count(train_banks, test_banks),
        "val_test": _split_overlap_count(val_banks, test_banks),
    }


def _family_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("family", "unknown"))].append(row)

    family_count = {family: len(items) for family, items in grouped.items()}
    binary_f1_by_family: Dict[str, float] = {}
    bdi_mae_by_family: Dict[str, float] = {}
    for family, items in grouped.items():
        metrics = compute_metrics(items)
        binary_f1_by_family[family] = float(metrics.get("binary_f1", 0.0))
        bdi_mae_by_family[family] = float(metrics.get("bdi_mae", 0.0))
    return {
        "family_count": family_count,
        "binary_f1_by_family": binary_f1_by_family,
        "bdi_mae_by_family": bdi_mae_by_family,
    }


def _resolve_effective_eval_mode(eval_mode: str) -> tuple[str, str]:
    requested = str(eval_mode).strip().lower()
    if requested in {"mixed_holdout", "synthetic_only"}:
        return requested, "synthetic_holdout"
    return requested, "synthetic_holdout"


def _profile_meta(profile: PersonaProfile) -> Dict[str, Any]:
    return {
        "split": profile.split,
        "family": profile.family,
        "generator_version": profile.generator_version,
        "generation_seed": profile.generation_seed,
        "template_bank": profile.template_bank,
        "behavior_params": dict(profile.behavior_params),
        "bdi_total": profile.bdi_total,
        "risk_signal": profile.has_risk_signal,
    }
