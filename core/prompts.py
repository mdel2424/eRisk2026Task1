from __future__ import annotations

from typing import Dict, List

OPENING_MESSAGE_FIXED = (
    "Thank you for coming in today. What changes in your life or routine recently made you feel it was time to "
    "talk to someone?"
)

PROMPT_REGISTRY: Dict[str, str | Dict[str, List[str]]] = {
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
    "evidence_gate": """
You are a strict relevance gate for a targeted BDI-II evidence extractor.
Decide whether the latest persona message contains evidence relevant to the allowed target items.

Return STRICT JSON ONLY with exactly this schema:
{{
  "target_relevant": true,
  "candidate_item_ids": [14, 5],
  "anchor_quote": "I feel like a burden",
  "confidence": 0.72,
  "reason": "explicit self-worth language"
}}

Constraints:
- candidate_item_ids must be a subset of allowed_item_ids.
- If target_relevant is false, candidate_item_ids must be [].
- Return at most 2 candidate_item_ids.
- Use only canonical BDI-II item ids.
- confidence must be in [0, 1].
- Keep anchor_quote short and directly grounded in the latest message.
- Mark target_relevant=true for direct, indirect, hedged, metaphorical, change-from-baseline, or functional-impact evidence inside allowed_item_ids.
- Do not mark target_relevant=true for generic ambiguity alone, such as "it seems a little different", "there has been some shift", "it feels a bit off", or "it's hard to be exact", unless the same reply also includes symptom-specific content, a concrete functional example, or a direct symptom-linked change from baseline.
- For module-3 items (guilt, failure, self-dislike, self-criticalness, worthlessness), mark target_relevant=true when the reply contains explicit self-evaluation or self-blame language such as "my own fault", "guilt just shows up", "I don't measure up", "I feel like a burden", or "I've been hard on myself lately".
- If the allowed items include item 14, prefer worthlessness relevance when the reply is about worth, mattering, burden, contribution, or identity-level failure, such as "I feel like a burden", "I do not contribute anything that matters", "I feel like a failure", or "I do not like who I am right now."
- If the allowed items include item 14 and the reply is only "It is my own fault" without worth or identity language, prefer guilt/self-blame items rather than item 14.
- If the detector asked about sexual interest and the reply is "That side of things is a little lower than usual, not a big change though" or "To put it simply, that side of things is a little lower than usual, not a big change though", mark target_relevant=true for item 21.
- If the detector asked about sexual interest and the reply is "That side of things has been fine" or "That has not really been an issue", mark target_relevant=false for item 21.
- If the detector asked about appetite, require direct appetite/eating change evidence such as "I'm not eating at all", "I'm eating much less than usual", "I'm just grabbing junk because I can't be bothered", or explicit variability phrasing like "it's been up and down", "not clearly one direction", or "some days food sounds fine and other days I barely bother." Do not mark appetite relevant for generic fatigue, chores, or "everything feels heavier" alone.
- If the detector asked about sleep, mark target_relevant=true for mixed sleep-instability phrasing like "sleep is a mess", "I'm up all night or sleeping too much", or "I wake in the night and can't get back to sleep."
- For module-3 items, softer self-evaluation language like "I've lost confidence in myself", "I'm hard on myself", "I keep thinking I should be doing better", or "I second-guess everything" is still relevant evidence.
- If the detector asked about sadness and the reply says it is "more irritability than sadness", mark target_relevant=true for the irritability sibling if it is in allowed_item_ids.
- If the detector asked about energy or interest and the reply says "it is a bit of both" or "both show up", mark target_relevant=true for the in-scope sibling symptom named by the reply rather than leaving it unsupported.
- Reserve target_relevant=false for explicit denial, explicit no-change, or clearly unrelated/logistical replies.
- Do not treat generic stress, pressure, busyness, or feeling behind as module-3 evidence unless the reply also contains explicit self-judgment, guilt, shame, blame, failure, or worthlessness language.
- Do not emit markdown, prose, headings, or trailing text.
- Output must start with "{{" and end with "}}".

Examples:
- If the detector asked about sleep and the reply is "a few nights, more than usual", return target_relevant=true.
- If the detector asked about self-worth and the reply is "I guess I feel like a burden lately", return target_relevant=true.
- If the allowed items include item 14 and the reply is "I feel like a failure" or "I do not like who I am right now", return target_relevant=true.
- If the allowed items include module-3 and the reply is "I feel like I'm falling behind and it's my own fault", return target_relevant=true.
- If the allowed items include module-3 and the reply is "Most days, guilt just shows up out of nowhere", return target_relevant=true.
- If the detector asked about sexual interest and the reply is "That side of things is a little lower than usual, not a big change though", return target_relevant=true.
- If the detector asked about sexual interest and the reply is "That side of things has been fine" or "That has not really been an issue", return target_relevant=false.
- If the detector asked about appetite and the reply is "I'm not eating at all or just grabbing junk because I can't be bothered", return target_relevant=true.
- If the detector asked about appetite and the reply is "it's been up and down" or "some days food sounds fine and other days I barely bother", return target_relevant=true.
- If the detector asked about sleep and the reply is "sleep is a mess, I'm up all night or sleeping too much", return target_relevant=true.
- If the reply is "not really, things feel about normal", return target_relevant=false.
- If the detector asked about appetite and the reply is "I haven't noticed anything different there", return target_relevant=false.
- If the reply is only "work has just been busy" or "everything feels heavier" without self-blame, self-judgment, or direct appetite change language, return target_relevant=false.

Current specialist node: {node_name}
Current detector question: {current_detector_question}
Target item: {target_item_id} ({target_item_name})
Target module: {target_module_id} ({target_module_name})
Allowed item ids: {allowed_item_ids}

Recent conversation:
{recent_context}

Latest persona message:
{latest_message}
""",
    "evidence_extract_targeted": """
You score every allowed BDI-II item for support in the latest persona message.

Return STRICT JSON ONLY, exactly one top-level object with this schema:
{{
  "scores": [
    {{
      "item_id": 14,
      "symptom_name": "Worthlessness",
      "supported": true,
      "intensity": 0.0,
      "confidence": 0.0,
      "anchor_quote": "short grounded quote",
      "reason": "short rationale"
    }}
  ]
}}

Constraints:
- Score every item in allowed_item_ids exactly once.
- Never omit an allowed item from the scores list.
- Do not score any item outside allowed_item_ids.
- Prefer candidate_item_ids when supported by the message, but still score all allowed items.
- For indirect but plausible in-scope signals, set supported=true with low confidence/low intensity only when the reply still names or concretely describes the symptom. Generic uncertainty alone is not enough.
- For module-3 items, treat explicit self-blame, guilt, shame, failure, self-dislike, self-criticalness, burden, or worthlessness language as support even when phrased indirectly or conversationally.
- If allowed_item_ids include item 14, prefer worthlessness support when the reply is specifically about burden, mattering, worth, contribution, or identity-level failure, such as "I feel like a burden", "I do not contribute anything that matters", "I feel like a failure", or "I do not like who I am right now."
- If allowed_item_ids include item 14 and the reply is only "It is my own fault" without worth or identity language, prefer guilt/self-blame items rather than item 14.
- If allowed_item_ids include item 21 and the detector question is explicitly about sexual interest, treat direct mild decrease phrasing like "That side of things is a little lower than usual, not a big change though" or "To put it simply, that side of things is a little lower than usual, not a big change though" as supported=true with low confidence/intensity.
- If allowed_item_ids include item 21 and the detector question is explicitly about sexual interest, keep item 21 supported=false for direct no-change replies like "That side of things has been fine" or "That has not really been an issue."
- If allowed_item_ids include item 18, require direct appetite or eating change evidence. Keep item 18 supported=false for generic heaviness, fatigue, chores, or stress unless the reply explicitly describes appetite/eating change, including variability language like "it's been up and down", "not clearly one direction", or "some days food sounds fine and other days I barely bother."
- If allowed_item_ids include item 16 and the detector question is about sleep, treat mixed sleep-instability phrasing like "sleep is a mess", "I'm up all night or sleeping too much", or "I wake in the night and can't get back to sleep" as supported=true with low-to-moderate confidence/intensity rather than unsupported.
- For module-3 items, softer self-evaluation phrasing like "I've lost confidence in myself", "I'm hard on myself", "I keep thinking I should be doing better", or "I second-guess everything" still counts as support.
- If the detector asked about sadness and the reply says it is "more irritability than sadness", support the irritability sibling if it is in allowed_item_ids.
- If the detector asked about energy or interest and the reply says "it is a bit of both" or "both show up", support the in-scope sibling symptom named by the reply with low confidence rather than leaving it unsupported.
- For module-3 items, do not map generic stress, busyness, pressure, or "everything feels heavier" to support unless the message also contains explicit self-evaluation, guilt, blame, shame, failure, or worthlessness language.
- Do not treat phrases like "it seems a little different", "there has been some shift", "it feels a bit off compared with usual", or "it is hard to be exact" as support by themselves unless the reply also includes symptom-specific language, a concrete functional example, or a direct symptom-linked change from baseline.
- Reserve supported=false for true denial, no-change, or unrelated/logistical content for that item.
- symptom_name must be the exact canonical BDI label for the chosen item_id.
- intensity must be in [0, 3], confidence in [0, 1].
- If supported=false, use intensity=0 and confidence=0 unless the message contains weak contradictory context you need to mention.
- Do not emit markdown, prose, headings, or trailing text.
- Output must start with "{{" and end with "}}".

Examples:
- If the detector asked about sleep and the reply is "a few nights, more than usual", mark the sleep item supported=true with low-to-moderate confidence/intensity, and mark the other allowed items supported=false.
- If the allowed items include worthlessness/self-criticism and the reply is "I feel like a burden", mark worthlessness or self-critical items supported=true with low-to-moderate confidence rather than unsupported.
- If the allowed items include item 14 and the reply is "I do not contribute anything that matters anymore", mark item 14 supported=true.
- If the allowed items include item 14 and the reply is "I feel like a failure" or "I do not like who I am right now", mark item 14 supported=true.
- If the allowed items include item 14 and the reply is only "It is my own fault", prefer guilt/self-blame items over item 14.
- If the allowed items include module-3 and the reply is "I feel like I'm falling behind and it's my own fault", mark guilt/self-blame items supported=true with low-to-moderate confidence rather than unsupported.
- If the allowed items include module-3 and the reply is "Most days, guilt just shows up out of nowhere", mark guilt or related self-evaluation items supported=true.
- If the detector asked about sexual interest and the reply is "That side of things is a little lower than usual, not a big change though", mark item 21 supported=true with low confidence/intensity rather than unsupported.
- If the detector asked about sexual interest and the reply is "That side of things has been fine" or "That has not really been an issue", keep item 21 supported=false.
- If the detector asked about appetite and the reply is "I'm not eating at all or just grabbing junk because I can't be bothered", mark item 18 supported=true.
- If the detector asked about appetite and the reply is "it's been up and down rather than clearly one direction" or "some days food sounds fine and other days I barely bother", mark item 18 supported=true with low-to-moderate confidence/intensity.
- If the detector asked about appetite and the reply is "I haven't noticed anything different there", keep item 18 supported=false.
- If the detector asked about sleep and the reply is "sleep is a mess, I'm up all night or sleeping too much", mark item 16 supported=true with low-to-moderate confidence/intensity.
- If the allowed items include self-dislike or self-criticalness and the reply is "I've lost confidence in myself" or "I keep thinking I should be doing better", mark the matching module-3 item supported=true with low confidence rather than unsupported.
- If the detector asked about sadness and the reply is "it leans more toward irritability than outright sadness", support the irritability sibling if it is in allowed_item_ids.
- If the detector asked about energy or interest and the reply is "it is a bit of both, honestly; starting things takes more effort and I get less out of them once I do", support the in-scope sibling symptom named by the reply with low confidence rather than unsupported.
- If the allowed items include module-3 and the reply is only "work has been stressful" or "everything feels heavier" without self-judgment, keep the module-3 items supported=false.
- If the reply is "not really, everything feels about normal", return the full scores list with every allowed item marked supported=false.

Current specialist node: {node_name}
Current detector question: {current_detector_question}
Target item: {target_item_id} ({target_item_name})
Target module: {target_module_id} ({target_module_name})
Allowed item ids: {allowed_item_ids}
Candidate item ids: {candidate_item_ids}
Anchor quote: {anchor_quote}

Recent conversation:
{recent_context}

Latest persona message:
{latest_message}
""",
    "evidence_shortlist_opportunistic": """
You decide whether the latest persona message contains strong off-target BDI-II evidence outside the current scoped items.

Return STRICT JSON ONLY with exactly this schema:
{{
  "has_strong_offtarget_signal": true,
  "candidate_item_ids": [15, 20],
  "anchor_quote": "everything takes so much energy",
  "confidence": 0.72,
  "reason": "strong fatigue language outside the current target scope"
}}

Constraints:
- candidate_item_ids must only include BDI-II item ids outside scoped_allowed_item_ids.
- Return at most 4 candidate_item_ids.
- Use has_strong_offtarget_signal=true only for strong, specific evidence that clearly sits outside the current scope.
- Prefer has_strong_offtarget_signal=false with an empty candidate list for mild, vague, mixed, denied, or no-change replies.
- confidence must be in [0, 1].
- Keep anchor_quote short and directly grounded in the latest message.
- Do not emit markdown, prose, headings, or trailing text.
- Output must start with "{{" and end with "}}".

Examples:
- If the reply is "Getting out of bed is a battle and everything takes so much energy", shortlist fatigue/energy items outside the current scope.
- If the reply is "I wake up in the middle of the night and can't get back to sleep", shortlist the sleep item outside the current scope.
- If the reply is "some things don't feel quite as fun as they used to", return has_strong_offtarget_signal=false.
- If the reply is "not really, things feel about normal", return has_strong_offtarget_signal=false.

Current specialist node: {node_name}
Current detector question: {current_detector_question}
Current target item: {target_item_id} ({target_item_name})
Current target module: {target_module_id} ({target_module_name})
Scoped allowed item ids: {scoped_allowed_item_ids}

Recent conversation:
{recent_context}

Latest persona message:
{latest_message}
""",
    "evidence_score_opportunistic": """
You score shortlisted off-target BDI-II items for strong support in the latest persona message.

Return STRICT JSON ONLY, exactly one top-level object with this schema:
{{
  "scores": [
    {{
      "item_id": 16,
      "symptom_name": "Changes in Sleeping Pattern",
      "supported": false,
      "intensity": 0.0,
      "confidence": 0.0,
      "anchor_quote": "short grounded quote",
      "reason": "short rationale"
    }}
  ]
}}

Constraints:
- Score every item in candidate_item_ids exactly once.
- Never omit a candidate item from the scores list.
- Do not score items outside candidate_item_ids.
- Use supported=true only for strong, specific off-target evidence that clearly stands on its own.
- Prefer supported=false for mild, vague, mixed, denied, or generic emotional content.
- symptom_name must be the exact canonical BDI label for the chosen item_id.
- intensity must be in [0, 3], confidence in [0, 1].
- If supported=false, use intensity=0 and confidence=0 unless a weak contradictory cue needs to be noted.
- Do not emit markdown, prose, headings, or trailing text.
- Output must start with "{{" and end with "}}".

Examples:
- If candidate_item_ids includes sleep and the reply is "I wake up in the middle of the night and can't get back to sleep", mark the sleep item supported=true.
- If candidate_item_ids includes fatigue and the reply is "everything takes so much energy", mark the fatigue item supported=true.
- If candidate_item_ids includes worthlessness but the reply is only "I guess work has been stressful", keep all candidates supported=false.

Current specialist node: {node_name}
Current detector question: {current_detector_question}
Current target item: {target_item_id} ({target_item_name})
Current target module: {target_module_id} ({target_module_name})
Scoped allowed item ids: {scoped_allowed_item_ids}
Candidate item ids: {candidate_item_ids}
Shortlist anchor quote: {anchor_quote}
Shortlist reason: {shortlist_reason}

Recent conversation:
{recent_context}

Latest persona message:
{latest_message}
""",
}


def get_prompt(key: str) -> str:
    value = PROMPT_REGISTRY.get(key)
    if isinstance(value, str):
        return value
    return ""


def get_fallback_questions(node_name: str) -> List[str]:
    value = PROMPT_REGISTRY.get("fallback_questions", {})
    if isinstance(value, dict):
        options = value.get(node_name, [])
        if isinstance(options, list):
            return [str(item) for item in options]
    return []
