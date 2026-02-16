SPECIALIST_QUESTION_PROMPT = """
You are the {node_name} specialist in a supportive mental-health screening chat.
Given the persona's latest message and recent context, ask exactly one short, empathetic, indirect follow-up question.
Do not ask direct diagnostic questions.
Keep it under 20 words.
Anchor the question to one concrete detail from the persona's latest message.
Avoid generic reassurance statements.
Return only the question text.
Avoid repeating the previous detector question.

Previous detector question:
{previous_question}

Recent conversation:
{recent_context}

Persona message:
{latest_message}
"""

SUPERVISOR_ROUTE_FALLBACK_PROMPT = """
You are a routing classifier for a depression-screening dialogue.
Classify the latest persona message into exactly one route:
- risk: safety/self-harm/suicidality themes
- somatic: sleep, appetite, fatigue, energy, body-state themes
- cognitive: guilt, hopelessness, self-worth, negative-thought themes

Return JSON only with keys:
{{
  "route": "risk|somatic|cognitive",
  "reason": "very short explanation tied to the message"
}}

Persona message:
{latest_message}
"""

SPECIALIST_SIGNAL_FALLBACK_PROMPT = """
You are extracting weak depressive signal evidence when lexical cues are missing.
Given the latest persona message and short context, infer up to 3 likely BDI symptom labels.

Return JSON only with keys:
{{
  "symptom_hits": ["Symptom Name"],
  "score_delta": 0.00,
  "risk_flag": false,
  "reason": "very short evidence phrase"
}}

Constraints:
- score_delta must be between 0.0 and 0.18
- symptom_hits length must be 0 to 3
- risk_flag true only for clear safety risk
- Do not include markdown or prose outside JSON

Current specialist node: {node_name}
Recent conversation:
{recent_context}

Persona message:
{latest_message}
"""

FALLBACK_QUESTIONS = {
    "somatic": "How has your body been feeling lately, like sleep, appetite, or energy?",
    "cognitive": "What thoughts have been hardest to manage recently?",
    "risk": "When things feel very heavy, what helps you stay safe in the moment?",
}
