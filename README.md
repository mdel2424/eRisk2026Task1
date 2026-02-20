# eRisk_Honours

Minimal PoC for eRisk 2026 Task 1:
- LangGraph detector (`supervisor -> specialist -> extract_evidence -> update_beliefs -> assess_stop`)
- Swappable inference backends:
  - detector: `local_hf` or `openrouter`
  - persona: `hf_adapter` or `openrouter_sim`
- CLI for eval/iteration + Streamlit UI for explainability

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf auth login
cp .env.example .env
```

Note: you must have access to `meta-llama/Meta-Llama-3-8B-Instruct` on Hugging Face
(accept license on HF first).
This project expects local CUDA inference (recommended >=8 GB VRAM).
Default behavior is automatic:
- if CUDA VRAM >= `MIN_CUDA_VRAM_GB`, use `local_hf` + `hf_adapter`
- otherwise, use `openrouter` + `openrouter_sim` 
##### Using openrouter for simulated persona instead of huggingface inference endpoint since openrouter is pay per token, while inference endpoint is pay per GPU hour.
Set `AUTO_BACKEND_SWITCH=0` to disable auto and use manual backend env values.


## Run CLI

```bash
python -m app.cli --mode interactive
python -m app.cli --mode eval --personas 10 --seed 42 --eval_mode mixed_holdout --prompt_version v1 --save_diagnostics true --max_api_calls 180 --trace_level compact --fit_calibrator auto
```

CLI parameters:
- `--mode` (`interactive` or `eval`, default: `interactive`)
  - `interactive`: manual turn-by-turn local testing.
  - `eval`: automated persona runs + artifact generation.
- `--personas` (int, default: `10`)
  - Number of synthetic personas to generate (used in eval workflows with synthetic splits).
- `--seed` (int, default: `42`)
  - Random seed for synthetic persona generation/splitting reproducibility.
- `--eval_mode` (`mixed_holdout`, `official_only`, `synthetic_only`, default: `mixed_holdout`)
  - `mixed_holdout`: synthetic val/test + released official tracking personas.
  - `official_only`: only released official tracking personas from `OFFICIAL_RELEASED_PERSONAS`.
  - `synthetic_only`: only synthetic val/test splits.
- `--prompt_version` (string, default: `v1`)
  - Prompt registry key used by prompt loaders (e.g., `v1`).
- `--save_diagnostics` (bool-like string, default: `true`)
  - Accepts values like `true/false`, `1/0`, `yes/no`.
  - If enabled, writes `outputs/diagnostics_run_local.json`.
- `--max_api_calls` (int, default: `180`)
  - Hard budget for OpenRouter-backed calls. Run stops gracefully when budget is exhausted.
- `--trace_level` (`compact` or `off`, default: `compact`)
  - `compact`: store per-turn attribution traces in diagnostics.
  - `off`: disable per-turn timeline payload to keep output slim.
- `--fit_calibrator` (`auto`, `on`, `off`, default: `auto`)
  - `auto`: token-safe default (disabled when OpenRouter is active).
  - `on`: force synthetic-train calibrator fitting.
  - `off`: skip fitting and use deterministic calibrator.

Eval artifacts:
- `outputs/interactions_run_local.json`
- `outputs/results_run_local.json`
- `outputs/metrics_run_local.json`
- `outputs/failure_report_run_local.json`
- `outputs/diagnostics_run_local.json` (if enabled)
- `outputs/config_used.json`
- `outputs/calibrator_bundle_local.json` (if synthetic train split is available)

## Run UI

```bash
streamlit run ui.py
```