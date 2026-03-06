from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Dict, List

from app.cli_common import _write_json
from app.cli_eval import run_eval


def _parse_seed_list(raw: str, fallback_seed: int) -> List[int]:
    text = str(raw or "").strip()
    if not text:
        return [int(fallback_seed)]
    parsed: List[int] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            parsed.append(int(token))
        except ValueError:
            continue
    return parsed or [int(fallback_seed)]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _aggregate(records: List[Dict[str, Any]], keys: List[str]) -> Dict[str, Dict[str, float]]:
    payload: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = [_safe_float(row.get(key, 0.0)) for row in records]
        if not values:
            payload[key] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            continue
        payload[key] = {
            "mean": round(sum(values) / len(values), 4),
            "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }
    return payload


def run_eval_multi_seed(
    *,
    persona_count: int,
    seeds_raw: str,
    fallback_seed: int,
    eval_mode: str,
    prompt_version: str,
    save_diagnostics: bool,
    max_api_calls: int,
    trace_level: str,
    output_dir: str | Path,
    debug_outputs: bool = False,
) -> Dict[str, Any]:
    seeds = _parse_seed_list(seeds_raw, fallback_seed)
    root_output = Path(output_dir)
    root_output.mkdir(parents=True, exist_ok=True)

    print(f"Running multi-seed eval: seeds={seeds}, personas={persona_count}, eval_mode={eval_mode}")
    per_seed: List[Dict[str, Any]] = []

    for index, seed in enumerate(seeds, start=1):
        seed_output = root_output / f"seed_{seed}"
        print(f"[multi-seed {index}/{len(seeds)}] seed={seed} -> {seed_output}")
        result = run_eval(
            persona_count=persona_count,
            seed=seed,
            eval_mode=eval_mode,
            prompt_version=prompt_version,
            save_diagnostics=save_diagnostics,
            max_api_calls=max_api_calls,
            trace_level=trace_level,
            output_dir=seed_output,
            debug_outputs=debug_outputs,
        )
        metrics = dict(result.get("metrics", {}))
        primary = dict(metrics.get("primary_metrics", {}))
        per_seed.append(
            {
                "seed": seed,
                "profiles_evaluated": int(result.get("profiles_evaluated", 0)),
                "expected_eval_profiles": int(result.get("expected_eval_profiles", 0)),
                "metrics": {
                    "objective": _safe_float(primary.get("objective", metrics.get("objective", 0.0))),
                    "headline_f1": _safe_float(primary.get("headline_f1", metrics.get("headline_f1", 0.0))),
                    "item_f1_macro_at_1": _safe_float(
                        primary.get("item_f1_macro_at_1", metrics.get("item_f1_macro_at_1", 0.0))
                    ),
                    "item_mae": _safe_float(primary.get("item_mae", metrics.get("item_mae", 0.0))),
                    "bdi_mae": _safe_float(primary.get("bdi_mae", metrics.get("bdi_mae", 0.0))),
                    "avg_turns_to_decision": _safe_float(
                        primary.get("avg_turns_to_decision", metrics.get("avg_turns_to_decision", 0.0))
                    ),
                },
                "output_dir": str(seed_output),
            }
        )

    metric_keys = [
        "objective",
        "headline_f1",
        "item_f1_macro_at_1",
        "item_mae",
        "bdi_mae",
        "avg_turns_to_decision",
    ]
    aggregate = _aggregate([row.get("metrics", {}) for row in per_seed], metric_keys)

    summary = {
        "config": {
            "personas": persona_count,
            "seeds": seeds,
            "eval_mode": eval_mode,
            "prompt_version": prompt_version,
            "max_api_calls_per_seed": max_api_calls,
            "trace_level": trace_level,
            "save_diagnostics": bool(save_diagnostics),
            "debug_outputs": bool(debug_outputs),
        },
        "aggregate": aggregate,
        "per_seed": per_seed,
    }
    summary_path = root_output / "multi_seed_summary.json"
    _write_json(summary_path, summary)
    print(
        "multi-seed summary: "
        f"headline_f1_mean={aggregate['headline_f1']['mean']:.4f} "
        f"item_f1_mean={aggregate['item_f1_macro_at_1']['mean']:.4f}"
    )
    print(f"Wrote: {summary_path}")
    return summary
