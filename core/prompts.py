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
You are a skilled clinical interviewer conducting a BDI-II-aligned depression screening.
Write exactly one naturalistic, clinically grounded follow-up question.

Hard constraints:
- Use a conversational but clinically purposeful tone, like a skilled therapist.
- Keep it under 50 words.
- Anchor to a concrete detail from the persona's latest message when possible.
- Focus on a two-week timeframe when clinically relevant; vary the phrasing naturally.
- Probe exactly one goal: {probe_goal} (frequency | duration | impact | exemplar).
- When probing frequency or duration, offer concrete anchors the persona can choose from
  (e.g., "hours or most of the day", "a few nights or most nights").
- When useful, use comparison framing: "compared with usual", "from your normal baseline".
- Stay inside the selected module and target-item domain.
- Do not reassure, praise, or offer therapeutic advice.
- Do not repeat or closely paraphrase the previous detector question.
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
      "item_id": 19,
      "symptom_name": "Concentration Difficulty",
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
- If symptom wording is non-canonical, map it to the closest canonical BDI label; do not invent new labels.
- Map metaphorical language to symptoms: e.g. "running on fumes" → fatigue/energy, "going through the motions" → loss of pleasure, "cloud over my head" → sadness, "short fuse" → irritability.
- intensity in [0, 3], confidence in [0, 1].
- use only BDI-II symptom labels.
- Do not default to item 1 (Sadness) when evidence is vague or unspecific.
- Item 1 increase requires explicit mood-affect language (sad/down/low mood/tearful/crying/numb or emotionally flat).
- Fatigue/sleep/concentration-only evidence should map to their specific items, not item 1.
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
                "How has your sleep changed compared with your usual pattern, falling asleep, staying asleep, or waking early?",
                "How has your energy shifted across a typical day recently, and when does it feel worst?",
                "How has your appetite been compared with usual, increased, decreased, or unchanged?",
                "On how many days in the past two weeks has fatigue interfered with getting things done?",
                "When your body feels off, what does that usually look like for you?",
                "When you wake up most mornings recently, do you feel rested or still exhausted?",
                "Have you noticed more restlessness or difficulty sitting still compared with usual?",
                "How often have you felt completely worn out before the day is even halfway through?",
                "What has your evening routine felt like compared with your normal?",
                "How different has your concentration felt when doing routine tasks like reading or following a conversation?",
                "How has your pace or motivation for starting basic tasks shifted recently?",
                "Have you noticed any weight change or your clothes fitting differently without trying?",
            ],
            "cognitive": [
                "What thought has been loudest or most repetitive in your mind recently?",
                "When things feel heavy lately, what do you tend to tell yourself?",
                "Thinking about the future, do you feel hopeful things will improve or does it feel unlikely?",
                "When something goes wrong, do you tend toward frustration, self-blame, guilt, or something else?",
                "How has your confidence in yourself changed compared with your usual level?",
                "What worries or thoughts have been looping the most in the past couple of weeks?",
                "When something doesn't go as planned, where does your mind go first?",
                "When you face small decisions, do you decide and move on, or do you get stuck replaying options?",
                "What has felt most mentally draining in your day-to-day recently?",
                "How often have guilt or self-blame thoughts shown up over the past two weeks?",
                "What activities or interests feel less meaningful or rewarding than they used to?",
                "When you try to rest, what does your mind tend to do?",
            ],
            "risk": [
                "When things felt very heavy recently, what helped you get through it?",
                "Who or what helped you through your hardest moments lately?",
                "When thoughts feel overwhelming, what is usually the first thing you do?",
                "When distress spikes, what signs tell you that you need extra support?",
                "What has helped you get through moments that felt emotionally unsafe?",
                "When you feel close to your limit, who can you reach out to quickly?",
                "When things feel at their worst, what do you find yourself wishing for, relief, escape, shutting everything off?",
                "What has made difficult moments feel even slightly safer recently?",
                "When you feel like withdrawing completely, what keeps you grounded?",
                "Have there been moments where life felt not worth the effort, or you felt you would rather not be here?",
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
