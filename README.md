# eRisk_Honours

Research codebase for a conversational BDI-style depression screening prototype used in an honours project. The system interviews a deterministic synthetic persona, gathers item-level evidence across the 21 BDI items, and produces item scores, a total BDI estimate, and evaluation artifacts for analysis.

This frozen repo snapshot keeps the **current heuristic pipeline** intact and treats **batch eval + notebook analysis** as the primary workflow.

## Architecture At A Glance

Active graph in [graph.py](graph.py):

`ingest_turn -> risk_sentinel -> extract_likelihoods -> belief_update -> policy_metrics -> stop_decider -> target_selector -> question_generator -> finalize_outputs`

Runtime nodes:
- `ingest_turn`: advance the dialogue state when a new persona reply arrives
- `risk_sentinel`: estimate immediate risk pressure and set risk flags
- `extract_likelihoods`: extract quote-grounded item evidence from the latest reply
- `belief_update`: update per-item posteriors and support counts
- `policy_metrics`: compute confidence, coverage, and uncertainty summaries
- `stop_decider`: decide whether the interview should stop
- `target_selector`: choose the next route/item/module to probe
- `question_generator`: realize the next detector question
- `finalize_outputs`: convert accumulated evidence into final item scores and total BDI

Canonical entrypoints:
- [app/cli.py](app/cli.py)
- [app/cli_eval.py](app/cli_eval.py)
- [notebooks/eval_item_error_analysis.ipynb](notebooks/eval_item_error_analysis.ipynb)
- [docs/architecture_overview.md](docs/architecture_overview.md)

## Recommended Workflow

1. Run a synthetic eval:

```bash
python -m app.cli --mode eval --personas 10 --seed 42 --max_api_calls 1200
```

2. Inspect the generated artifacts in `outputs/`.

3. Open [notebooks/eval_item_error_analysis.ipynb](notebooks/eval_item_error_analysis.ipynb) to review metrics, persona-level errors, item-level bias, and transcript patterns.

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
- `OLLAMA_THINK_MODE=off` disables reasoning modes required when using standard local models (like qwen3.5:4b) that don't natively support thinking tokens

Synthetic persona generation is always deterministic and local. Personas are used so the detector can be benchmarked repeatedly against fixed ground-truth BDI item scores.

## CLI Modes

Primary mode:
- `eval`: runs the benchmark used by the notebook and report workflow

Secondary workflows:
- `interactive`: step through one interview for live inspection/demo
- `eval_multi`: run the same synthetic benchmark across multiple seeds
- `tune`: search threshold combinations against the synthetic benchmark

Examples:

```bash
python -m app.cli --mode interactive --personas 10 --seed 42 --interactive_persona_index 0
python -m app.cli --mode eval --personas 10 --seed 42 --max_api_calls 1200
python -m app.cli --mode eval --personas 10 --seed 42 --max_api_calls 1200 --debug_outputs true --save_diagnostics true --trace_level compact
python -m app.cli --mode eval_multi --personas 30 --multi_seeds 42,43,44 --max_api_calls 1200
python -m app.cli --mode tune --tune_personas 30 --tune_seed 42 --tune_max_api_calls 800
```

## Outputs

Lean mode (default):
- `outputs/persona_manifest_run_local.json`
- `outputs/persona_manifest_hash_run_local.txt`
- `outputs/benchmark_integrity_run_local.json`
- `outputs/interactions_run_local.json`
- `outputs/results_run_local.json`
- `outputs/metrics_run_local.json`
- `outputs/config_used.json`
- `outputs/multi_seed/multi_seed_summary.json` from `--mode eval_multi`
- `outputs/tuning/tuning_runs.jsonl`
- `outputs/tuning/tuning_summary.json`
- `outputs/tuning/best_thresholds.env`
- `outputs/tuning/best_vs_baseline.json`

Debug mode (`--debug_outputs true`) additionally writes:
- `outputs/error_report_run_local.json`
- `outputs/extract_parse_fail_log_run_local.json`
- `outputs/failure_report_run_local.json`
- `outputs/diagnostics_run_local.json` when `--save_diagnostics true`

## Configuration

- Use `.env.example` as the supported minimal configuration surface.
- The main supported knobs are backend/model selection, simulator style, and benchmark controls.
- Additional detector and extraction thresholds exist in code defaults for research tuning, but they are intentionally not the main documented workflow for the frozen snapshot.
