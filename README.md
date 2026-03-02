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
python -m app.cli --mode eval --personas 10 --seed 42 --eval_mode mixed_holdout --prompt_version v1 --save_diagnostics true --max_api_calls 500 --trace_level compact --fit_calibrator auto
python -m app.cli --mode eval_multi --personas 30 --multi_seeds 42,43,44 --eval_mode mixed_holdout --prompt_version v1 --save_diagnostics false --max_api_calls 380 --trace_level compact --fit_calibrator auto
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

`eval_multi`:
- runs `eval` across multiple seeds and writes aggregate mean/std metrics
- seed list via `--multi_seeds` (comma-separated)

## Outputs

- `outputs/persona_manifest_run_local.json` (full synthetic persona metadata + BDI ground truth)
- `outputs/persona_manifest_hash_run_local.txt`
- `outputs/leakage_report_run_local.json`
- `outputs/interactions_run_local.json`
- `outputs/results_run_local.json`
- `outputs/metrics_run_local.json`
- `outputs/error_report_run_local.json` (per-item/family error analysis + worst personas)
- `outputs/extract_parse_fail_log_run_local.json` (per-turn extractor parse-fail debug records with parser error kind/message, balance diagnostics, full raw extractor payload, and full persona message)
- `outputs/failure_report_run_local.json`
- `outputs/diagnostics_run_local.json` (if enabled)
- `outputs/config_used.json`
- `outputs/multi_seed/multi_seed_summary.json` (from `--mode eval_multi`)
- `outputs/tuning/tuning_runs.jsonl` (all threshold candidates + validity/guardrail status)
- `outputs/tuning/tuning_summary.json`
- `outputs/tuning/best_thresholds.env`
- `outputs/tuning/best_vs_baseline.json`

Main tuning knobs live in `.env.example`:
- stop policy: `MIN_TURNS`, `MAX_TURNS`, `STOP_CONFIDENCE`
- confidence model: `CONF_SUPPORT_TAU`, `CONF_DEPTH_WEIGHT`, `CONF_COVERAGE_WEIGHT`, `CONF_UP_ALPHA`, `CONF_DECAY_STREAK_START`, `CONF_DECAY_PER_TURN`, `CONF_DECAY_MAX`, `CONF_MAX_DROP_PER_TURN`
- extractor robustness: `EXTRACTOR_JSON_KEY_ALIASES`, `EXTRACTOR_STRICT_SCHEMA_COERCE`, `EXTRACTOR_MIN_RECORDS_TARGET`
- item-1 sadness guard: `EXTRACT_ITEM1_STRICT_GATE`, `EXTRACT_ITEM1_WEAK_MAX_CONF`, `EXTRACT_ITEM1_WEAK_MAX_INTENSITY`
- belief debiasing: `BELIEF_WEIGHT_LLM_EXTRACTOR`, `BELIEF_WEIGHT_LLM_SALVAGE`, `BELIEF_WEIGHT_LEXICAL_FALLBACK`, `BELIEF_WEIGHT_LEXICAL_PREFILTER`, `BELIEF_WEIGHT_DEFAULT`, `BELIEF_DUPLICATE_WEIGHT`, `BELIEF_DECAY_START_SUPPORT`, `BELIEF_DECAY_TAU`, `BELIEF_CONTRADICTION_WEIGHT`, `BELIEF_CONTRADICTION_NEUTRAL_BLEND`, `BELIEF_SUPPORT_MIN_WEIGHT`, `BELIEF_MEMORY_PER_ITEM`
- extractor generation budget: `DETECTOR_EXTRACTOR_MAX_NEW_TOKENS`, `DETECTOR_EXTRACTOR_TEMPERATURE`, `DETECTOR_EXTRACTOR_TOP_P`
- supervisor routing: `SUPERVISOR_EVIDENCE_MIN_SCORE`, `SUPERVISOR_EVIDENCE_RISK_THRESHOLD`, `SUPERVISOR_ESCAPE_EMPTY_STREAK`
- simulator style: `SIM_HEDGE_RATE`, `SIM_NORMALIZATION_RATE`, `SIM_CONTEXT_ANCHOR_RATE`, `SIM_DIRECT_ANSWER_RATE`
- simulator depressed severity target: `SIM_DEPRESSED_TARGET_BDI`, `SIM_DEPRESSED_TARGET_JITTER`, `SIM_DEPRESSED_TARGET_BLEND`
- finalization blending: `FINAL_OBS_BLEND_ENABLED`, `FINAL_OBS_BLEND_CONF_THRESHOLD`, `FINAL_OBS_BLEND_SUPPORT_MAX`, `FINAL_OBS_BLEND_MODULE_CONF_MIN`, `FINAL_OBS_BLEND_MAX_ALPHA`

Default stop behavior:
- risk cues no longer short-circuit stop policy.
- stop rule is simple: after `MIN_TURNS`, stop when `global_confidence >= STOP_CONFIDENCE` (or `MAX_TURNS` safety cap).
- `global_confidence` is the only confidence metric and uses support+coverage saturation with near-monotonic smoothing.
- confidence can decline only slightly and rarely after prolonged no-information streaks (bounded by `CONF_MAX_DROP_PER_TURN`).

Output semantics:
- `results_run_local.json`: includes `item-scores` for all 21 BDI items (`"1"`..`"21"`, values `0..3`) in addition to `bdi-score` and `key-symptoms`.
- `metrics_run_local.json`: primary quality metrics include `headline_f1`, `item_f1_macro_at_1`, and `item_mae`.
- `failure_report_run_local.json`: includes `extract_parse_fail_rate`, `extract_empty_rate`, extractor source/recovery distributions, post-floor productivity summaries, observed-item blend counters, duplicate/contradiction evidence rates, support increment rate, and method-weight usage.
- extractor applies a strict item-1 guard for LLM-derived rows: unsupported sadness-increase rows are dropped, weak sadness signals are confidence/intensity-clamped.
- binary depressed/control scoring is deprecated in local evaluation and no longer used for ranking/objective.
- `symptom_f1_at_4` is kept for compatibility and aliases `item_f1_macro_at_1`.

If strict split lock is on and persona generation logic changes, bump `SIM_GENERATOR_VERSION` (or remove prior manifest files) before rerunning eval.
