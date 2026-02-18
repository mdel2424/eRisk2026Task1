# eRisk_Honours

Minimal PoC for eRisk 2026 Task 1:
- LangGraph detector (`supervisor -> specialist -> assess_stop`)
- Hugging Face local inference for both detector and persona
- CLI for eval/iteration + Streamlit UI for demo/explainability (to come later)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf auth login
```

Note: you must have access to `meta-llama/Meta-Llama-3-8B-Instruct` on Hugging Face
(accept license on HF first).

## Environment (`.env`)

```env
# Shared auth
HF_TOKEN=...

# Detector model (used by supervisor/specialists prompts)
DETECTOR_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
DETECTOR_LOAD_IN_4BIT=1
DETECTOR_MAX_NEW_TOKENS=96
DETECTOR_TEMPERATURE=0.2
DETECTOR_TOP_P=0.9

# Persona adapter (released eRisk LoRA adapter)
ERISK_BASE_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
ERISK_ADAPTER_ID=Anxo/erisk26-task1-patient-00-adapter
ERISK_LOAD_IN_4BIT=1
ERISK_MAX_NEW_TOKENS=96
ERISK_TEMPERATURE=0.7
ERISK_TOP_P=0.9

# Stop policy
MIN_TURNS=4
MAX_TURNS=10
STOP_CONFIDENCE=0.75
```

## Run CLI

```bash
python -m app.cli --mode interactive
python -m app.cli --mode eval --personas 10 --seed 42
```

Eval artifacts:
- `outputs/interactions_run_local.json`
- `outputs/results_run_local.json`
- `outputs/metrics_run_local.json`

## Run UI

```bash
streamlit run ui.py
```