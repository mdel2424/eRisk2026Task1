from __future__ import annotations

import os
from typing import Dict, List

OPENING_MESSAGE_FIXED = (
    "Thank you for coming in today. What changes in your life or routine recently made you feel it was time to "
    "talk to someone?"
)

PROMPT_REGISTRY: Dict[str, Dict[str, str | Dict[str, List[str]]]] = {
    "v1": {
        "specialist_question": """
You are the {node_name} specialist in a supportive mental-health screening chat.
Write exactly one short, empathetic, indirect follow-up question.

Hard constraints:
- Do not ask direct diagnostic questions.
- Keep it under 30 words.
- Anchor to one concrete detail from the latest persona message.
- Keep a two-week frame implicitly; do not repeat "past two weeks" formulaically.
- Probe exactly one goal: {probe_goal} (frequency | duration | impact | exemplar).
- Stay inside the selected module and target-item intent.
- Avoid generic reassurance.
- Avoid repeating the previous detector question.
- Return only the question text.

Target module:
- module_id={target_module_id}
- module_name={target_module_name}
- module_goal={target_module_goal}
- module_items={target_module_items}
- target_item_id={target_item_id}
- target_item_name={target_item_name}

Previous detector question:
{previous_question}

Latest persona message:
{latest_message}

Recent context (most recent turns):
{recent_context}
""",
        "opening_question": OPENING_MESSAGE_FIXED,
        "evidence_extraction": """
You extract interpretable depression evidence from one persona message.
Return STRICT JSON ONLY, exactly one top-level object with this schema:
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
- symptom_name must be the exact canonical BDI label for the given item_id (no alternative labels).
- intensity in [0, 3], confidence in [0, 1].
- use only BDI-II symptom labels.
- Output MUST start with "{{" and end with "}}".
- Do not wrap output in markdown fences.
- Do not add commentary, notes, explanations, headings, or trailing text.
- Do not include fields outside the schema.
- Keep evidence list length <= 4 (unless strong concurrent risk+somatic evidence).

Invalid output examples (do NOT do these):
- ```json ... ```
- "Here is the JSON:" followed by object
- Objects missing item_id/intensity/confidence

Current specialist node: {node_name}
Recent conversation:
{recent_context}

Latest persona message:
{latest_message}
""",
        "fallback_questions": {
            "somatic": [
                "How has your sleep changed from your usual pattern?",
                "How has your energy shifted across a typical day recently?",
                "Have meals or appetite felt different lately?",
                "What part of your day has felt physically hardest to get through lately?",
                "When your body feels off, what does that usually look like for you?",
                "How refreshed do you feel when you wake up most mornings recently?",
                "Have you noticed changes in restlessness or feeling wound up lately?",
                "How often have you felt worn out before the day is even over?",
                "What has your evening routine felt like compared with your normal?",
                "How different has your concentration felt during routine tasks lately?",
                "How has your pace or motivation for basic tasks shifted recently?",
                "When you feel drained, what daily activities are most affected?",
            ],
            "cognitive": [
                "What thought has been loudest in your mind recently?",
                "When things feel heavy lately, what do you tell yourself?",
                "What feels hardest to believe about tomorrow right now?",
                "What kind of self-talk tends to show up when things get difficult lately?",
                "How has your sense of confidence in yourself changed recently?",
                "What worries have been looping the most in the past little while?",
                "When something goes wrong lately, where does your mind go first?",
                "How has your ability to decide small things felt compared with usual?",
                "What has felt most mentally exhausting in your day-to-day recently?",
                "How often have guilt or self-blame thoughts shown up lately?",
                "What has felt less meaningful or less rewarding than it used to?",
                "What has your mind been preoccupied with when you try to rest?",
            ],
            "risk": [
                "When things felt very heavy recently, what helped you stay safe?",
                "Who or what helped you through your hardest moments lately?",
                "When thoughts feel overwhelming lately, what do you do first?",
                "When distress spikes, what signs tell you that you need extra support?",
                "What has helped you get through moments that felt emotionally unsafe?",
                "When you feel close to your limit, who can you contact quickly?",
                "How have you been coping when thoughts feel darker than usual?",
                "What has made difficult moments feel even slightly safer recently?",
                "When you feel like withdrawing completely, what keeps you grounded?",
                "How often have you had to actively focus on staying safe lately?",
                "What has stopped things from escalating on your hardest days?",
                "What would be your first step if things felt unmanageable tonight?",
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
