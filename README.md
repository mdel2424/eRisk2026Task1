# eRisk_Honours

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env` for remote inference.
Current PoC uses OpenRouter only.

## Run

Interactive debug:

```bash
python main.py --mode interactive
```

Automated evaluation batch:

```bash
python main.py --mode eval --personas 10 --seed 42
```

Simple UI (hot reload):

```bash
streamlit run ui.py
```
