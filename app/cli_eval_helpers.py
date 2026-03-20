from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.evaluation import compute_metrics
from persona import PersonaProfile

MANIFEST_SCHEMA_VERSION = 4


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


def _manifest_payload(
    *,
    persona_count: int,
    seed: int,
    profiles: List[PersonaProfile],
) -> Dict[str, Any]:
    def _profile_dict(profile: PersonaProfile) -> Dict[str, Any]:
        return {
            "persona_id": profile.persona_id,
            "split": profile.split,
            "family": profile.family,
            "severity_tier": profile.severity_tier,
            "subtype_tag": profile.subtype_tag,
            "context_tag": profile.context_tag,
            "style_tag": profile.style_tag,
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
        }

    return {
        "run_config": {
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "persona_count": persona_count,
            "seed": seed,
        },
        "persona_count": len(profiles),
        "profiles": [_profile_dict(profile) for profile in profiles],
    }


def _manifest_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_previous_manifest_info(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        return {
            "exists": False,
            "hash": None,
            "profile_count": 0,
            "read_error": None,
        }

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("manifest payload is not a JSON object")
        profiles = list(payload.get("profiles", []) or [])
        return {
            "exists": True,
            "hash": _manifest_hash(payload),
            "profile_count": len(profiles),
            "read_error": None,
        }
    except Exception as exc:
        return {
            "exists": True,
            "hash": None,
            "profile_count": 0,
            "read_error": str(exc),
        }


def _family_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    subtype_count: Dict[str, int] = defaultdict(int)
    context_count: Dict[str, int] = defaultdict(int)
    style_count: Dict[str, int] = defaultdict(int)
    for row in rows:
        grouped[str(row.get("family", "unknown"))].append(row)
        subtype = str(row.get("subtype_tag", "")).strip()
        context = str(row.get("context_tag", "")).strip()
        style = str(row.get("style_tag", "")).strip()
        if subtype:
            subtype_count[subtype] += 1
        if context:
            context_count[context] += 1
        if style:
            style_count[style] += 1

    family_count = {family: len(items) for family, items in grouped.items()}
    item_f1_by_family: Dict[str, float] = {}
    bdi_mae_by_family: Dict[str, float] = {}
    for family, items in grouped.items():
        metrics = compute_metrics(items)
        item_f1_by_family[family] = float(metrics.get("item_f1_macro_at_1", 0.0) or 0.0)
        bdi_mae_by_family[family] = float(metrics.get("bdi_mae", 0.0))
    return {
        "family_count": family_count,
        "subtype_count": dict(sorted(subtype_count.items(), key=lambda pair: pair[0])),
        "context_count": dict(sorted(context_count.items(), key=lambda pair: pair[0])),
        "style_count": dict(sorted(style_count.items(), key=lambda pair: pair[0])),
        "item_f1_by_family": item_f1_by_family,
        "bdi_mae_by_family": bdi_mae_by_family,
    }


def _profile_meta(profile: PersonaProfile) -> Dict[str, Any]:
    return {
        "split": profile.split,
        "family": profile.family,
        "severity_tier": profile.severity_tier,
        "subtype_tag": profile.subtype_tag,
        "context_tag": profile.context_tag,
        "style_tag": profile.style_tag,
        "generation_seed": profile.generation_seed,
        "template_bank": profile.template_bank,
        "behavior_params": dict(profile.behavior_params),
        "bdi_total": profile.bdi_total,
        "risk_signal": profile.has_risk_signal,
    }
