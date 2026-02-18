from __future__ import annotations

import os
from typing import Dict, List

PROMPT_REGISTRY: Dict[str, Dict[str, str | Dict[str, List[str]]]] = {
    "v1": {
        "specialist_question": """
You are the {node_name} specialist in a supportive mental-health screening chat.
Given the persona's latest message and recent context, ask exactly one short, empathetic, indirect follow-up question.
Do not ask direct diagnostic questions.
Keep it under 20 words.
Anchor the question to one concrete detail from the persona's latest message.
Keep a clear "past two weeks" time window.
Target exactly one probe goal: {probe_goal}.
Probe goal definitions:
- frequency: how often this happens
- duration: how long episodes/states last
- impact: functional effect on work/study/home/social life
- exemplar: one concrete recent example
Avoid generic reassurance statements.
Return only the question text.
Avoid repeating the previous detector question.

Previous detector question:
{previous_question}

Recent conversation:
{recent_context}

Persona message:
{latest_message}
""",
        "opening_question": """
To ground us, what does a typical good week look like for you when things are going well?
Then, in the past two weeks, what has felt most different from your usual self?
""",
        "evidence_extraction": """
You extract interpretable depression evidence from one persona message.
Return strict JSON with schema:
{{
  "evidence": [
    {{
      "item_id": 1,
      "symptom_name": "Sadness",
      "direction": "increase|decrease|neutral",
      "intensity": 0.0,
      "confidence": 0.0,
      "evidence_text": "verbatim short quote/paraphrase",
      "reason": "short rationale"
    }}
  ]
}}

Constraints:
- If no credible evidence, return {{ "evidence": [] }}.
- item_id must be 1..21.
- intensity in [0, 3], confidence in [0, 1].
- use only BDI-II symptom labels.
- no markdown, no prose, JSON only.

Current specialist node: {node_name}
Recent conversation:
{recent_context}

Latest persona message:
{latest_message}
""",
        "fallback_questions": {
            "somatic": [
                "In the past two weeks, how has your sleep changed from usual?",
                "In the past two weeks, how has your energy changed across the day?",
                "In the past two weeks, have meals or appetite felt different?",
            ],
            "cognitive": [
                "In the past two weeks, what thought has been loudest in your mind?",
                "When things feel heavy lately, what do you tell yourself?",
                "In the past two weeks, what feels hardest to believe about tomorrow?",
            ],
            "risk": [
                "In the past two weeks, when things felt very heavy, what helped you stay safe?",
                "In the past two weeks, who or what helped you through your hardest moments?",
                "When thoughts feel overwhelming lately, what do you do first?",
            ],
        },
    }
}


def _prompt_version(version: str | None = None) -> str:
    resolved = (version or os.getenv("PROMPT_VERSION", "v1")).strip().lower()
    return resolved if resolved in PROMPT_REGISTRY else "v1"


def get_prompt(key: str, version: str | None = None) -> str:
    v = _prompt_version(version)
    value = PROMPT_REGISTRY[v].get(key)
    if isinstance(value, str):
        return value
    return ""


def get_fallback_questions(node_name: str, version: str | None = None) -> List[str]:
    v = _prompt_version(version)
    value = PROMPT_REGISTRY[v].get("fallback_questions", {})
    if isinstance(value, dict):
        options = value.get(node_name, [])
        if isinstance(options, list):
            return [str(item) for item in options]
    return []
