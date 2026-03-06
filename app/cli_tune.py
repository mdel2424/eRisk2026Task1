from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from app.cli_eval import run_eval
from app.cli_common import _write_json
from app.cli_tune_helpers import (
    _apply_guardrails,
    _build_best_thresholds_env,
    _pick_top_candidates,
    _safe_float,
    _temporary_env,
)
from app.tune_config import BASELINE_ENV_KEYS, ROUTING_GRID, STAGE1_GRID


def run_tune(
    *,
    tune_personas: int,
    tune_seed: int,
    tune_max_api_calls: int,
    tune_save_diagnostics: bool,
    tune_trace_level: str,
    tune_prompt_version: str,
    tune_top_k: int,
    debug_outputs: bool = False,
) -> None:
    output_dir = Path("outputs/tuning")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_overrides = {key: os.getenv(key, "").strip() for key in BASELINE_ENV_KEYS}
    baseline_overrides.update(
        {
            "MIN_TURNS": baseline_overrides.get("MIN_TURNS") or "18",
            "MAX_TURNS": baseline_overrides.get("MAX_TURNS") or "34",
            "STOP_CONFIDENCE": baseline_overrides.get("STOP_CONFIDENCE") or "0.66",
            "CONF_SUPPORT_TAU": baseline_overrides.get("CONF_SUPPORT_TAU") or "1.25",
            "CONF_UP_ALPHA": baseline_overrides.get("CONF_UP_ALPHA") or "0.55",
            "CONF_DEPTH_WEIGHT": baseline_overrides.get("CONF_DEPTH_WEIGHT") or "0.70",
            "SUPERVISOR_EVIDENCE_MIN_SCORE": baseline_overrides.get("SUPERVISOR_EVIDENCE_MIN_SCORE") or "0.30",
            "SUPERVISOR_EVIDENCE_RISK_THRESHOLD": baseline_overrides.get("SUPERVISOR_EVIDENCE_RISK_THRESHOLD") or "0.22",
            "SUPERVISOR_ESCAPE_EMPTY_STREAK": baseline_overrides.get("SUPERVISOR_ESCAPE_EMPTY_STREAK") or "2",
        }
    )

    tune_runs: List[Dict[str, Any]] = []
    expected_total_runs = 1 + len(STAGE1_GRID) + (max(1, int(tune_top_k)) * len(ROUTING_GRID)) + 2
    run_index = 0

    def _changed_overrides(overrides: Dict[str, Any]) -> str:
        changed: List[str] = []
        for key in sorted(BASELINE_ENV_KEYS):
            base = str(baseline_overrides.get(key, ""))
            current = str(overrides.get(key, base))
            if current != base:
                changed.append(f"{key}={current} (was {base})")
        return ", ".join(changed) if changed else "no threshold changes"

    print(
        f"Tune plan: total_candidates={expected_total_runs} "
        f"(baseline=1, stage1={len(STAGE1_GRID)}, stage2={max(1, int(tune_top_k)) * len(ROUTING_GRID)}, confirm=2)"
    )

    def _run_candidate(
        *,
        stage: str,
        candidate_id: str,
        overrides: Dict[str, Any],
        seed: int,
    ) -> Dict[str, Any]:
        nonlocal run_index
        run_index += 1
        candidate_dir = output_dir / candidate_id
        print(
            f"[tune {run_index}/{expected_total_runs}] stage={stage} candidate={candidate_id} "
            f"seed={seed} | {_changed_overrides(overrides)}"
        )
        with _temporary_env(overrides):
            result = run_eval(
                persona_count=tune_personas,
                seed=seed,
                eval_mode="mixed_holdout",
                prompt_version=tune_prompt_version,
                save_diagnostics=tune_save_diagnostics,
                max_api_calls=tune_max_api_calls,
                trace_level=tune_trace_level,
                output_dir=candidate_dir,
                debug_outputs=debug_outputs,
            )
        metrics = result.get("metrics", {}) or {}
        record = {
            "stage": stage,
            "candidate_id": candidate_id,
            "seed": seed,
            "overrides": {str(k): str(v) for k, v in overrides.items()},
            "metrics": {
                "objective": _safe_float(metrics.get("objective", 0.0)),
                "headline_f1": _safe_float(metrics.get("headline_f1", metrics.get("item_f1_macro_at_1", 0.0))),
                "item_f1_macro_at_1": _safe_float(metrics.get("item_f1_macro_at_1", metrics.get("symptom_f1_at_4", 0.0))),
                "item_mae": _safe_float(metrics.get("item_mae", 0.0)),
                "avg_turns_to_decision": _safe_float(metrics.get("avg_turns_to_decision", 0.0)),
                "bdi_mae": _safe_float(metrics.get("bdi_mae", 0.0)),
            },
            "profiles_evaluated": int(result.get("profiles_evaluated", 0)),
            "expected_eval_profiles": int(result.get("expected_eval_profiles", 0)),
            "llm_usage": (result.get("metrics", {}) or {}).get("llm_usage", {}),
            "route_distribution": result.get("route_distribution", {}),
            "failure_counters": result.get("failure_counters", {}),
            "output_dir": str(result.get("output_dir", candidate_dir)),
        }
        return record

    baseline_record = _run_candidate(
        stage="baseline",
        candidate_id=f"baseline_seed{tune_seed}",
        overrides=baseline_overrides,
        seed=tune_seed,
    )
    baseline_headline_f1 = _safe_float(
        baseline_record["metrics"].get("headline_f1", baseline_record["metrics"].get("item_f1_macro_at_1", 0.0))
    )
    baseline_item_f1 = _safe_float(baseline_record["metrics"].get("item_f1_macro_at_1", 0.0))
    baseline_record = _apply_guardrails(
        record=baseline_record,
        baseline_headline_f1=baseline_headline_f1,
        baseline_item_f1=baseline_item_f1,
    )
    tune_runs.append(baseline_record)

    stage1_records: List[Dict[str, Any]] = []
    for idx, config in enumerate(STAGE1_GRID, start=1):
        overrides = dict(baseline_overrides)
        overrides.update(config)
        record = _run_candidate(
            stage="stage1",
            candidate_id=f"stage1_{idx}_seed{tune_seed}",
            overrides=overrides,
            seed=tune_seed,
        )
        record = _apply_guardrails(
            record=record,
            baseline_headline_f1=baseline_headline_f1,
            baseline_item_f1=baseline_item_f1,
        )
        stage1_records.append(record)
        tune_runs.append(record)

    stage1_top = _pick_top_candidates(stage1_records, top_k=tune_top_k)

    stage2_records: List[Dict[str, Any]] = []
    for top_idx, stage1_best in enumerate(stage1_top, start=1):
        base_overrides = dict(baseline_overrides)
        base_overrides.update(stage1_best.get("overrides", {}))
        for idx, routing in enumerate(ROUTING_GRID, start=1):
            overrides = dict(base_overrides)
            overrides.update(routing)
            record = _run_candidate(
                stage="stage2",
                candidate_id=f"stage2_t{top_idx}_r{idx}_seed{tune_seed}",
                overrides=overrides,
                seed=tune_seed,
            )
            record = _apply_guardrails(
                record=record,
                baseline_headline_f1=baseline_headline_f1,
                baseline_item_f1=baseline_item_f1,
            )
            stage2_records.append(record)
            tune_runs.append(record)

    best_stage2 = _pick_top_candidates(stage2_records, top_k=1)
    selected_candidate = best_stage2[0] if best_stage2 else (stage1_top[0] if stage1_top else baseline_record)
    selected_overrides = dict(selected_candidate.get("overrides", {}))

    confirmation_records: List[Dict[str, Any]] = []
    for seed in (42, 43):
        confirm_record = _run_candidate(
            stage="confirm",
            candidate_id=f"confirm_seed{seed}",
            overrides=selected_overrides,
            seed=seed,
        )
        confirm_record = _apply_guardrails(
            record=confirm_record,
            baseline_headline_f1=baseline_headline_f1,
            baseline_item_f1=baseline_item_f1,
        )
        confirmation_records.append(confirm_record)
        tune_runs.append(confirm_record)

    valid_confirms = [row for row in confirmation_records if bool(row.get("valid", False))]
    final_candidate = selected_candidate
    confirmation_summary: Dict[str, Any] = {"valid_count": len(valid_confirms)}
    if len(valid_confirms) == len(confirmation_records):
        mean_objective = sum(_safe_float(row["metrics"].get("objective", 0.0)) for row in valid_confirms) / len(
            valid_confirms
        )
        mean_item_f1 = sum(
            _safe_float(row["metrics"].get("item_f1_macro_at_1", row["metrics"].get("symptom_f1_at_4", 0.0)))
            for row in valid_confirms
        ) / len(
            valid_confirms
        )
        mean_headline_f1 = sum(
            _safe_float(row["metrics"].get("headline_f1", row["metrics"].get("item_f1_macro_at_1", 0.0)))
            for row in valid_confirms
        ) / len(valid_confirms)
        mean_turns = sum(
            _safe_float(row["metrics"].get("avg_turns_to_decision", 0.0)) for row in valid_confirms
        ) / len(valid_confirms)
        confirmation_summary.update(
            {
                "mean_objective": round(mean_objective, 4),
                "mean_headline_f1": round(mean_headline_f1, 4),
                "mean_item_f1": round(mean_item_f1, 4),
                "mean_avg_turns": round(mean_turns, 4),
            }
        )
        if mean_headline_f1 >= max(0.35, baseline_headline_f1 - 0.03) and mean_item_f1 >= max(
            0.30, baseline_item_f1 - 0.03
        ):
            final_candidate = dict(selected_candidate)
            final_candidate["metrics"] = {
                **final_candidate.get("metrics", {}),
                "objective": round(mean_objective, 4),
                "headline_f1": round(mean_headline_f1, 4),
                "item_f1_macro_at_1": round(mean_item_f1, 4),
                "avg_turns_to_decision": round(mean_turns, 4),
            }
            final_candidate["source"] = "confirm_mean"
    else:
        confirmation_summary["reason"] = "confirmation_failed_guardrails_or_truncated"

    jsonl_path = output_dir / "tuning_runs.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in tune_runs:
            handle.write(json.dumps(row) + "\n")

    baseline_metrics = baseline_record.get("metrics", {})
    final_metrics = final_candidate.get("metrics", {})
    best_vs_baseline = {
        "baseline_candidate_id": baseline_record.get("candidate_id"),
        "final_candidate_id": final_candidate.get("candidate_id"),
        "baseline": baseline_metrics,
        "final": final_metrics,
        "delta": {
            "objective": round(
                float(final_metrics.get("objective", 0.0)) - float(baseline_metrics.get("objective", 0.0)),
                4,
            ),
            "headline_f1": round(
                _safe_float(final_metrics.get("headline_f1", final_metrics.get("item_f1_macro_at_1", 0.0)))
                - _safe_float(baseline_metrics.get("headline_f1", baseline_metrics.get("item_f1_macro_at_1", 0.0))),
                4,
            ),
            "item_f1_macro_at_1": round(
                _safe_float(final_metrics.get("item_f1_macro_at_1", 0.0))
                - _safe_float(baseline_metrics.get("item_f1_macro_at_1", 0.0)),
                4,
            ),
            "avg_turns_to_decision": round(
                float(final_metrics.get("avg_turns_to_decision", 0.0))
                - float(baseline_metrics.get("avg_turns_to_decision", 0.0)),
                4,
            ),
        },
    }
    _write_json(output_dir / "best_vs_baseline.json", best_vs_baseline)

    best_thresholds_env = _build_best_thresholds_env(final_candidate.get("overrides", {}))
    (output_dir / "best_thresholds.env").write_text(best_thresholds_env, encoding="utf-8")

    tuning_summary = {
        "tune_config": {
            "tune_personas": tune_personas,
            "tune_seed": tune_seed,
            "tune_max_api_calls": tune_max_api_calls,
            "tune_prompt_version": tune_prompt_version,
            "tune_top_k": tune_top_k,
            "debug_outputs": bool(debug_outputs),
        },
        "baseline": baseline_record,
        "stage1_top": stage1_top,
        "best_stage2": best_stage2[0] if best_stage2 else None,
        "confirmation": confirmation_summary,
        "final_candidate": final_candidate,
        "artifacts": {
            "tuning_runs_jsonl": str(jsonl_path),
            "tuning_summary_json": str(output_dir / "tuning_summary.json"),
            "best_thresholds_env": str(output_dir / "best_thresholds.env"),
            "best_vs_baseline_json": str(output_dir / "best_vs_baseline.json"),
        },
    }
    _write_json(output_dir / "tuning_summary.json", tuning_summary)

    print(
        f"tune_final objective={float(final_metrics.get('objective', 0.0)):.4f} "
        f"headline_f1={_safe_float(final_metrics.get('headline_f1', final_metrics.get('item_f1_macro_at_1', 0.0))):.4f} "
        f"item_f1={_safe_float(final_metrics.get('item_f1_macro_at_1', 0.0)):.4f} "
        f"avg_turns={_safe_float(final_metrics.get('avg_turns_to_decision', 0.0)):.2f}"
    )
    print(f"tune_artifacts={output_dir}")
