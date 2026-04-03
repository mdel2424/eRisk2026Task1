# Architecture Overview

## What This Application Is

This project is a research prototype for conversational depression screening aligned to the 21 BDI items. It does not talk to real participants in this repo snapshot. Instead, it interviews deterministic synthetic personas with known ground-truth BDI item scores so the full pipeline can be evaluated repeatedly and compared across seeds and configurations.

## Why Synthetic Personas Are Used

Synthetic personas make the project reproducible:
- every run can be regenerated from a seed
- every persona has known item-level ground truth
- the detector can be benchmarked on item F1, BDI error, routing behavior, and transcript quality

The simulator is local and deterministic. This keeps evaluation stable and makes the system easier to explain in an honours report.

## Active Runtime Pipeline

The current runtime lives in [graph.py](../graph.py):

`ingest_turn -> risk_sentinel -> extract_likelihoods -> belief_update -> policy_metrics -> stop_decider -> target_selector -> question_generator -> finalize_outputs`

Stage responsibilities:
- `ingest_turn`: update turn state after each persona response
- `risk_sentinel`: estimate immediate risk pressure and decide whether risk probing should take priority
- `extract_likelihoods`: turn the latest reply into structured item-level evidence
- `belief_update`: update per-item beliefs and support counts from the extracted evidence
- `policy_metrics`: summarize confidence, uncertainty, and coverage
- `stop_decider`: decide whether the interview can stop
- `target_selector`: choose the next symptom area or item to probe
- `question_generator`: produce the next detector question
- `finalize_outputs`: convert accumulated evidence into final item scores and total BDI

## Main Repo Surfaces

The most important entrypoints are:
- [app/cli.py](../app/cli.py): command-line entrypoint
- [app/cli_eval.py](../app/cli_eval.py): canonical batch evaluation workflow
- [app/notebook_eval.py](../app/notebook_eval.py): helpers for loading and analyzing outputs
- [notebooks/eval_item_error_analysis.ipynb](../notebooks/eval_item_error_analysis.ipynb): main analysis notebook

## Artifacts Produced

The eval workflow writes:
- transcript artifacts
- per-persona prediction results
- aggregate metrics
- benchmark-integrity metadata
- optional debug diagnostics and failure summaries

These files are written under `outputs/` and are the main inputs to the notebook and the honours write-up.
