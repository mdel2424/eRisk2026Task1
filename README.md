# eRisk_Honours

Minimal PoC for eRisk 2026 Task 1:
- LangGraph detector (`supervisor -> specialist -> extract_evidence -> update_beliefs -> assess_stop`)
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
```

By default, eval prints only:
- `binary_f1`
- `objective`

Set `CLI_VERBOSE=1` to print full backend/run summaries to stdout.

`--eval_mode`:
- `mixed_holdout`: synthetic val+test eval (train used for calibrator fit)
- `synthetic_only`: same as above

## Key Outputs

- `outputs/persona_manifest_run_local.json` (full synthetic persona metadata + BDI ground truth)
- `outputs/persona_manifest_hash_run_local.txt`
- `outputs/leakage_report_run_local.json`
- `outputs/interactions_run_local.json`
- `outputs/results_run_local.json`
- `outputs/metrics_run_local.json`
- `outputs/failure_report_run_local.json`
- `outputs/diagnostics_run_local.json` (if enabled)
- `outputs/config_used.json`

Final decision behavior:
- Final BDI is computed from observed + module-imputed item scores.
- Final label uses BDI/risk plus sparse-evidence fallback:
  - `FINAL_CORE_ITEM_MIN_HITS` (default `2`)
  - `FINAL_CORE_SIGNAL_GATE` (default `1.0`)

Simulator tuning knobs (`.env` / `.env.example`):
- `SIM_HEDGE_RATE` (default `0.65`)
- `SIM_NORMALIZATION_RATE` (default `0.45`)
- `SIM_CONTEXT_ANCHOR_RATE` (default `0.55`)
- `SIM_DIRECT_ANSWER_RATE` (default `0.78`)

Diagnostics include persona style QA stats per run:
- `responses_total`
- `hedged_response_count`
- `deflect_response_count`
- `context_anchor_count`
- `avg_response_words`

## Run UI

```bash
streamlit run ui.py
```
