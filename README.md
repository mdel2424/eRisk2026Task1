# eRisk_Honours

Minimal PoC for eRisk 2026 Task 1:
- LangGraph detector (`ingest_turn -> risk_sentinel -> extract_likelihoods -> belief_update -> policy_metrics -> stop_decider -> target_selector -> question_generator -> finalize_outputs`)
- BDI-SSI module-aware probing (deterministic target item/module selection)
- Final-time module-weighted imputation for unobserved BDI items (interpretable item-sum BDI)
- Detector backends: `local_hf` or `openrouter`
- Persona runtime: deterministic simulator only (`openrouter_sim` path, no persona LLM calls)
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
- if CUDA VRAM >= `MIN_CUDA_VRAM_GB`, use `local_hf`
- otherwise, use `openrouter`
`OPENROUTER_API_KEY` is required only when detector resolves to `openrouter`.

Persona generation is always deterministic and consumes hidden probe intent (`target_item_id`, `route`, `style`, `mode`, `directness`, `priority`) from detector state.
Probe intent is stored only in `turn_trace`/diagnostics; transcripts remain natural-language only.
First detector message is fixed:
`Thank you for coming in today. What changes in your life or routine recently made you feel it was time to talk to someone?`

Simulator behavior defaults:
- cooperative-but-hedged replies (answer-first, uncertainty-second)
- concrete day-to-day anchors (work/family/routine/messages)
- passive-distress risk language by default; explicit intent language only for high latent risk

## Run CLI

```bash
python -m app.cli --mode interactive --personas 10 --seed 42 --interactive_persona_index 0
python -m app.cli --mode eval --personas 10 --seed 42 --eval_mode mixed_holdout --prompt_version v1 --max_api_calls 500 --fit_calibrator auto
python -m app.cli --mode eval --personas 10 --seed 42 --eval_mode mixed_holdout --prompt_version v1 --max_api_calls 500 --fit_calibrator auto --debug_outputs true --save_diagnostics true --trace_level compact
python -m app.cli --mode eval_multi --personas 30 --multi_seeds 42,43,44 --eval_mode mixed_holdout --prompt_version v1 --max_api_calls 380 --fit_calibrator auto
python -m app.cli --mode tune --tune_personas 30 --tune_seed 42 --tune_max_api_calls 800 --tune_trace_level off --fit_calibrator auto
```

`interactive` is a stepper:
- press Enter to alternate `detector -> persona -> detector -> ...`
- each step prints compact pipeline flow (ingest, risk, extraction, belief/policy, route, stop, usage)
- transcript stays natural-language only; probe intent is shown from hidden handoff metadata

`eval` randomizes holdout split membership per run (same generated pool, different val/test IDs) to improve persona coverage across repeated local runs.  
`eval_multi`/`tune` remain deterministic by seed.

By default, eval prints only:
- `item_f1`
- `objective`
(`objective` is computed from `headline_f1` with turn penalty)

Logging profiles:
- Lean (default): `--debug_outputs false` (default) keeps outputs simple.
- Debug (opt-in): `--debug_outputs true` enables heavy diagnostics artifacts.
- Backward compatibility: passing `--save_diagnostics true` or `--trace_level compact` also enables debug profile.

Tune mode now logs:
- planned total candidate runs
- per-candidate stage/id/seed
- threshold deltas from baseline for that run

Set `CLI_VERBOSE=1` to print full backend/run summaries to stdout.

## Outputs

Lean mode (default):
- `outputs/persona_manifest_run_local.json` (persona metadata + ground truth)
- `outputs/persona_manifest_hash_run_local.txt`
- `outputs/leakage_report_run_local.json`
- `outputs/interactions_run_local.json`
- `outputs/results_run_local.json`
- `outputs/metrics_run_local.json`
- `outputs/config_used.json`
- `outputs/multi_seed/multi_seed_summary.json` (from `--mode eval_multi`)
- `outputs/tuning/tuning_runs.jsonl` (all threshold candidates + validity/guardrail status)
- `outputs/tuning/tuning_summary.json`
- `outputs/tuning/best_thresholds.env`
- `outputs/tuning/best_vs_baseline.json`

Debug mode (`--debug_outputs true`) additionally writes:
- `outputs/error_report_run_local.json`
- `outputs/extract_parse_fail_log_run_local.json`
- `outputs/failure_report_run_local.json`
- `outputs/diagnostics_run_local.json` (if `--save_diagnostics true`)

Configuration:
- Use `.env.example` as the full source of tunable parameters.
- Most useful day-to-day knobs are `MIN_TURNS`, `MAX_TURNS`, `STOP_CONFIDENCE`, `max_api_calls`, and `--debug_outputs`.
- If strict split lock is enabled and persona generation logic changes, bump `SIM_GENERATOR_VERSION` (or remove prior manifest files) before rerunning eval.
