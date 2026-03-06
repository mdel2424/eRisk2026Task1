from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.evaluation import compute_metrics
from persona import PersonaProfile

from app.cli_common import _parse_bool

MANIFEST_SCHEMA_VERSION = 2


def _objective(metrics: Dict[str, Any], max_turns: int, latency_lambda: float = 0.15) -> float:
    if not metrics:
        return 0.0
    headline_f1 = float(
        metrics.get("headline_f1", metrics.get("item_f1_macro_at_1", metrics.get("symptom_f1_at_4", 0.0))) or 0.0
    )
    avg_turns = float(metrics.get("avg_turns_to_decision", 0.0))
    normalized_turns = min(1.0, avg_turns / max(1, max_turns))
    return round(headline_f1 - (latency_lambda * normalized_turns), 4)


def _with_objective(metrics: Dict[str, Any], max_turns: int) -> Dict[str, Any]:
    if not metrics:
        return {}
    return {**metrics, "objective": _objective(metrics, max_turns=max_turns)}


def _select_primary_metrics(
    overall_labeled: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    if overall_labeled:
        return "overall_labeled", overall_labeled
    return "overall_labeled", {}


def _strict_split_lock_enabled() -> bool:
    return _parse_bool(os.getenv("STRICT_SPLIT_LOCK", "1"))


def _manifest_payload(
    *,
    persona_count: int,
    seed: int,
    generator_version: str,
    profiles: List[PersonaProfile],
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

    return {
        "run_config": {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "persona_count": persona_count,
            "seed": seed,
            "generator_version": generator_version,
        },
        "persona_count": len(profiles),
        "profiles": [_profile_dict(profile) for profile in profiles],
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


def _family_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("family", "unknown"))].append(row)

    family_count = {family: len(items) for family, items in grouped.items()}
    item_f1_by_family: Dict[str, float] = {}
    bdi_mae_by_family: Dict[str, float] = {}
    for family, items in grouped.items():
        metrics = compute_metrics(items)
        item_f1_by_family[family] = float(metrics.get("item_f1_macro_at_1", 0.0) or 0.0)
        bdi_mae_by_family[family] = float(metrics.get("bdi_mae", 0.0))
    return {
        "family_count": family_count,
        "item_f1_by_family": item_f1_by_family,
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
