# eRisk_Honours

Minimal PoC for eRisk 2026 Task 1:
- LangGraph detector (`extract_evidence -> update_beliefs -> assess_stop -> supervisor -> specialist`)
- BDI-SSI module-aware probing (deterministic target item/module selection)
- Final-time module-weighted imputation for unobserved BDI items (interpretable item-sum BDI)
- Swappable inference backends:
  - detector: `local_hf` or `openrouter`
  - persona runtime: `hf_adapter` or `openrouter_sim`
- Synthetic-only eval with leakage guards and traceability artifacts

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf auth login
cp .env.example .env
```

Default behavior is automatic:
- if CUDA VRAM >= `MIN_CUDA_VRAM_GB`, use `local_hf` + `hf_adapter`
- otherwise, use `openrouter` + `openrouter_sim`

Simulator behavior defaults:
- cooperative-but-hedged replies (answer-first, uncertainty-second)
- concrete day-to-day anchors (work/family/routine/messages)
- passive-distress risk language by default; explicit intent language only for high latent risk

## Run CLI

```bash
python -m app.cli --mode interactive
python -m app.cli --mode eval --personas 10 --seed 42 --eval_mode mixed_holdout --prompt_version v1 --save_diagnostics true --max_api_calls 380 --trace_level compact --fit_calibrator auto
python -m app.cli --mode tune --tune_personas 30 --tune_seed 42 --tune_max_api_calls 800 --tune_trace_level off --fit_calibrator auto
```

By default, eval prints only:
- `binary_f1`
- `objective`

Tune mode now logs:
- planned total candidate runs
- per-candidate stage/id/seed
- threshold deltas from baseline for that run

Tune run count:
- `1 + len(STAGE1_GRID) + (tune_top_k * len(ROUTING_GRID)) + 2`
- with defaults (`STAGE1=4`, `ROUTING=4`, `tune_top_k=1`) this is `11` eval runs

Set `CLI_VERBOSE=1` to print full backend/run summaries to stdout.

`--eval_mode`:
- `mixed_holdout`: synthetic val+test eval (train used for calibrator fit)
- `synthetic_only`: same as above

## Outputs

- `outputs/persona_manifest_run_local.json` (full synthetic persona metadata + BDI ground truth)
- `outputs/persona_manifest_hash_run_local.txt`
- `outputs/leakage_report_run_local.json`
- `outputs/interactions_run_local.json`
- `outputs/results_run_local.json`
- `outputs/metrics_run_local.json`
- `outputs/failure_report_run_local.json`
- `outputs/diagnostics_run_local.json` (if enabled)
- `outputs/config_used.json`
- `outputs/tuning/tuning_runs.jsonl` (all threshold candidates + validity/guardrail status)
- `outputs/tuning/tuning_summary.json`
- `outputs/tuning/best_thresholds.env`
- `outputs/tuning/best_vs_baseline.json`

Main tuning knobs live in `.env.example`:
- stop policy: `MIN_TURNS`, `MAX_TURNS`, `STOP_CONFIDENCE`, `MIN_EVIDENCE_FOR_CONF_STOP`
- supervisor routing: `SUPERVISOR_EVIDENCE_MIN_SCORE`, `SUPERVISOR_EVIDENCE_RISK_THRESHOLD`, `SUPERVISOR_ESCAPE_EMPTY_STREAK`
- simulator style: `SIM_HEDGE_RATE`, `SIM_NORMALIZATION_RATE`, `SIM_CONTEXT_ANCHOR_RATE`, `SIM_DIRECT_ANSWER_RATE`
