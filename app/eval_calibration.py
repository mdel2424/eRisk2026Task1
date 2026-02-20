from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.calibration import clear_calibrator_cache, fit_calibrator, save_calibrator_bundle
from persona import PersonaProfile

from app.cli_runtime import _run_profile
from app.cli_runtime_helpers import _print_progress, _usage_snippet


def fit_calibrator_from_train_profiles(
    *,
    train_profiles: List[PersonaProfile],
    graph_app,
    fit_enabled: bool,
    min_train_records: int,
    output_dir: Path,
    verbose_console: bool,
    live_status: bool,
) -> Tuple[Dict[str, Any], List[str], Counter[str]]:
    run_failure_counters: Counter[str] = Counter()
    calibrator_status: Dict[str, Any] = {
        "requested_policy": "",
        "enabled": fit_enabled,
        "mode": "deterministic_default",
        "reason": "disabled_by_policy" if not fit_enabled else "",
        "train_records": 0,
        "saved_path": "",
    }

    calibrator_train_records: List[Dict[str, Any]] = []
    calibrator_train_ids: List[str] = []

    if train_profiles and fit_enabled:
        if len(train_profiles) < min_train_records:
            calibrator_status["reason"] = "small_train_split"
            calibrator_status["mode"] = "skipped_small_train"
            run_failure_counters["calibrator_fallback_small_train"] += 1
            if verbose_console:
                print(
                    f"Skipping calibrator fit: train split too small "
                    f"({len(train_profiles)} < {min_train_records})."
                )
        else:
            if verbose_console:
                print(f"Fitting calibrator from synthetic train split ({len(train_profiles)} personas)...")
            train_total = len(train_profiles)
            for idx, profile in enumerate(train_profiles, start=1):
                final_state, _, _ = _run_profile(
                    profile,
                    graph_app,
                    verbose=False,
                    progress_prefix=f"[fit {idx}/{train_total}]",
                    live_status=live_status and verbose_console,
                )
                calibrator_train_ids.append(profile.persona_id)
                feature_vector = dict(final_state.get("latest_feature_vector", {}))
                calibrator_train_records.append(
                    {
                        "features": feature_vector,
                        "bdi_true": profile.bdi_total,
                        "y_true": profile.depressed,
                    }
                )
                if verbose_console:
                    _print_progress(f"Calibrator fit {_usage_snippet()}", idx, train_total)

            calibrator_status["train_records"] = len(calibrator_train_records)
            if calibrator_train_records:
                bundle, fit_reason = fit_calibrator(calibrator_train_records, min_records=min_train_records)
                calibrator_status["mode"] = bundle.mode
                calibrator_status["reason"] = fit_reason or ""
                if bundle.mode == "sklearn_fitted":
                    calibrator_path = output_dir / "calibrator_bundle_local.json"
                    save_calibrator_bundle(calibrator_path, bundle)
                    os.environ["CALIBRATOR_PATH"] = str(calibrator_path)
                    calibrator_status["saved_path"] = str(calibrator_path)
                    if verbose_console:
                        print(f"Calibrator saved: {calibrator_path}")
                else:
                    if fit_reason == "small_train_split":
                        run_failure_counters["calibrator_fallback_small_train"] += 1
                    if verbose_console:
                        print(f"Calibrator fallback mode: {bundle.mode} ({fit_reason or 'n/a'})")
                clear_calibrator_cache()
    elif not fit_enabled:
        calibrator_status["reason"] = "disabled_by_policy"
    else:
        calibrator_status["reason"] = "no_train_profiles"

    return calibrator_status, calibrator_train_ids, run_failure_counters
