# eRisk_Honours

Minimal PoC for eRisk 2026 Task 1:
- LangGraph detector (`ingest_turn -> judgment_agent -> bayes_state_update -> diagnosis_agent -> stop_controller -> navigation_agent -> question_agent`)
- Probabilistic multi-agent BDI-II runtime with bound clinical assertions, Noisy-OR Bayes updates, and diagnosis synthesis
- Detector backends: `openrouter` or explicit local `ollama`
- Persona runtime: deterministic local simulator with atomic memory verification
- Synthetic-only evaluation with provenance and benchmark-integrity artifacts

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Default detector behavior uses OpenRouter. `OPENROUTER_API_KEY` is required unless you explicitly switch to Ollama.

To use Ollama for local inference:
- install and start Ollama outside the repo
- run `ollama pull qwen3.5:4b`
- set `DETECTOR_BACKEND=ollama`
- optionally set `OLLAMA_BASE_URL`
- `OLLAMA_THINK_MODE=auto` keeps thinking on when CUDA is available and turns it off on CPU-only runs

Persona generation is always deterministic and synthetic. Probe intent is stored only in diagnostics; transcripts remain natural-language only.

## Run CLI

```bash
python -m app.cli --mode interactive --personas 10 --seed 42 --interactive_persona_index 0
python -m app.cli --mode eval --personas 10 --seed 42 --max_api_calls 1200
python -m app.cli --mode eval --personas 10 --seed 42 --max_api_calls 1200 --debug_outputs true --save_diagnostics true --trace_level compact
python -m app.cli --mode eval_multi --personas 30 --multi_seeds 42,43,44 --max_api_calls 1200
python -m app.cli --mode tune --tune_personas 30 --tune_seed 42 --tune_max_api_calls 800
```

Notebook workflow:
- `notebooks/eval_item_error_analysis.ipynb` runs a fresh synthetic eval into `outputs/`, then renders summary, benchmark-integrity, persona-error, and item-bias tables.

`interactive` is a stepper:
- press Enter to alternate `detector -> persona -> detector -> ...`
- each step prints compact runtime flow (ingest, judgment, Bayes update, diagnosis, navigation, stop, usage)

`eval`, `eval_multi`, and `tune` all benchmark synthetic personas generated for the current seed.

By default, eval prints only:
- `item_f1`
- `objective`

Logging profiles:
- Lean (default): `--debug_outputs false`
- Debug (opt-in): `--debug_outputs true`
- Backward compatibility: passing `--save_diagnostics true` or `--trace_level compact` also enables debug profile

## Outputs

Lean mode (default):
- `outputs/persona_manifest_run_local.json`
- `outputs/persona_manifest_hash_run_local.txt`
- `outputs/benchmark_integrity_run_local.json`
- `outputs/interactions_run_local.json`
- `outputs/results_run_local.json`
- `outputs/metrics_run_local.json`
- `outputs/config_used.json`
- `outputs/multi_seed/multi_seed_summary.json` (from `--mode eval_multi`)
- `outputs/tuning/tuning_runs.jsonl`
- `outputs/tuning/tuning_summary.json`
- `outputs/tuning/best_thresholds.env`
- `outputs/tuning/best_vs_baseline.json`

Debug mode (`--debug_outputs true`) additionally writes:
- `outputs/error_report_run_local.json`
- `outputs/extract_parse_fail_log_run_local.json`
- `outputs/failure_report_run_local.json`
- `outputs/diagnostics_run_local.json` (if `--save_diagnostics true`)

Configuration:
- Use `.env.example` as the supported minimal config surface
- The env files keep only backend/model selection, simulator style, and the main benchmark knobs
- Advanced detector and extraction overrides still exist as code defaults; add them to `.env` manually only when you are intentionally doing expert tuning
