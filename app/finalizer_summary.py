from __future__ import annotations

from typing import Any, Dict, Mapping


FINALIZER_GUARDRAIL_FIELDS = [
    "low_signal_guardrail_active",
    "corroborated_core_hits",
    "imputed_points_before_guardrail",
    "imputed_points_after_guardrail",
    "suppressed_imputed_item_ids",
    "low_signal_observed_cap_item_ids",
    "low_signal_item9_cap_reason",
    "support_geometry_candidate_bypass",
    "anchor_gated_guardrail_blocked",
    "guardrail_bypass_source",
]

FINALIZER_SEVERE_RECOVERY_FIELDS = [
    "severe_recovery_mode_active",
    "corroborated_item_ids",
    "severe_anchor_item_ids",
    "severe_anchor_module_ids",
    "severe_recovery_reason",
]

FINALIZER_SEVERE_AMPLITUDE_FIELDS = [
    "severe_amplitude_observed_item_ids",
    "severe_amplitude_imputed_item_ids",
    "severe_item9_rescued",
]

FINALIZER_SUMMARY_FIELDS = (
    FINALIZER_GUARDRAIL_FIELDS
    + FINALIZER_SEVERE_RECOVERY_FIELDS
    + FINALIZER_SEVERE_AMPLITUDE_FIELDS
)

FINALIZER_SUMMARY_DEFAULTS: Dict[str, Any] = {
    "low_signal_guardrail_active": False,
    "corroborated_core_hits": 0,
    "imputed_points_before_guardrail": 0,
    "imputed_points_after_guardrail": 0,
    "suppressed_imputed_item_ids": [],
    "low_signal_observed_cap_item_ids": [],
    "low_signal_item9_cap_reason": "",
    "support_geometry_candidate_bypass": False,
    "anchor_gated_guardrail_blocked": False,
    "guardrail_bypass_source": "",
    "severe_recovery_mode_active": False,
    "corroborated_item_ids": [],
    "severe_anchor_item_ids": [],
    "severe_anchor_module_ids": [],
    "severe_recovery_reason": "",
    "severe_amplitude_observed_item_ids": [],
    "severe_amplitude_imputed_item_ids": [],
    "severe_item9_rescued": False,
}


def compact_finalizer_summary(module_imputation: Mapping[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(module_imputation, Mapping):
        return {}
    return {
        field: module_imputation[field]
        for field in FINALIZER_SUMMARY_FIELDS
        if field in module_imputation
    }


def default_finalizer_summary_value(field: str) -> Any:
    value = FINALIZER_SUMMARY_DEFAULTS[field]
    return list(value) if isinstance(value, list) else value
