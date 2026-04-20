from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents.question_generator import question_generator


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, _messages):
        return SimpleNamespace(content=self._content)


class QuestionGeneratorTests(unittest.TestCase):
    def test_question_generator_falls_back_when_model_output_is_empty(self) -> None:
        state = {
            "turn_index": 1,
            "messages": [{"role": "assistant", "content": "I have been feeling more on edge lately."}],
            "next_action": {
                "target_item_id": 2,
                "route": "cognitive",
                "style": "gentle_probe",
                "rationale": "targeted follow-up",
            },
            "item_beliefs": {},
        }

        with patch("agents.question_generator.get_llm", return_value=_FakeLLM("   ")):
            result = question_generator(state)

        asked = result["messages"][0]["content"]
        trace = result["turn_trace"]["question_generator"]

        self.assertTrue(asked.endswith("?"))
        self.assertTrue(bool(asked.strip()))
        self.assertTrue(bool(trace["used_fallback"]))
        self.assertEqual(trace["fallback_reason"], "empty_model_output")

    def test_question_generator_uses_llm_output_when_present(self) -> None:
        state = {
            "turn_index": 1,
            "messages": [{"role": "assistant", "content": "I have been feeling more on edge lately."}],
            "next_action": {
                "target_item_id": 2,
                "route": "cognitive",
                "style": "gentle_probe",
                "rationale": "targeted follow-up",
            },
            "item_beliefs": {},
        }

        with patch("agents.question_generator.get_llm", return_value=_FakeLLM("What thought has been loudest lately")):
            result = question_generator(state)

        asked = result["messages"][0]["content"]
        trace = result["turn_trace"]["question_generator"]

        self.assertEqual(asked, "What thought has been loudest lately?")
        self.assertFalse(bool(trace["used_fallback"]))
        self.assertEqual(trace["fallback_reason"], "")

    def test_same_thread_followup_strips_repeated_timeframe_lead(self) -> None:
        state = {
            "turn_index": 2,
            "messages": [
                {"role": "user", "content": "Over the last couple of weeks, what thought has felt loudest lately?"},
                {"role": "assistant", "content": "I keep expecting the worst and it hangs over the whole day."},
            ],
            "next_action": {
                "target_item_id": 2,
                "route": "cognitive",
                "style": "clarify_frequency",
                "question_kind": "same_item_followup",
                "timeframe_mode": "carry",
                "thread_turn_index": 2,
                "anchor_text": "expecting the worst",
                "rationale": "threaded follow-up",
            },
            "item_beliefs": {},
        }

        with patch(
            "agents.question_generator.get_llm",
            return_value=_FakeLLM("In the last two weeks, when that happens, does it linger most of the day"),
        ):
            result = question_generator(state)

        asked = result["messages"][0]["content"]
        trace = result["turn_trace"]["question_generator"]

        self.assertEqual(asked, "When that happens, does it linger most of the day?")
        self.assertTrue(bool(trace["repeated_timeframe_lead_blocked"]))
        self.assertFalse(bool(trace["question_starts_with_timeframe"]))

    def test_followup_fallback_stays_conversational_without_timeframe_lead(self) -> None:
        state = {
            "turn_index": 3,
            "messages": [
                {"role": "user", "content": "Lately, what thought has been loudest?"},
                {"role": "assistant", "content": "I keep feeling like I let people down."},
            ],
            "next_action": {
                "target_item_id": 5,
                "route": "cognitive",
                "style": "clarify_frequency",
                "question_kind": "same_item_followup",
                "timeframe_mode": "carry",
                "thread_turn_index": 2,
                "anchor_text": "let people down",
                "rationale": "threaded follow-up",
            },
            "item_beliefs": {},
        }

        with patch("agents.question_generator.get_llm", return_value=_FakeLLM("   ")):
            result = question_generator(state)

        asked = result["messages"][0]["content"]
        trace = result["turn_trace"]["question_generator"]

        self.assertTrue(bool(trace["used_fallback"]))
        self.assertFalse(asked.lower().startswith("in the last two weeks"))
        self.assertEqual(str(trace["question_kind"]), "same_item_followup")

    def test_same_thread_followup_strips_stock_you_mentioned_lead(self) -> None:
        state = {
            "turn_index": 2,
            "messages": [
                {"role": "user", "content": "What has been weighing on you most lately?"},
                {"role": "assistant", "content": "I keep feeling like I let people down."},
            ],
            "next_action": {
                "target_item_id": 5,
                "route": "cognitive",
                "style": "functional_impact",
                "question_kind": "same_item_followup",
                "timeframe_mode": "carry",
                "thread_turn_index": 2,
                "anchor_text": "let people down",
                "rationale": "threaded follow-up",
            },
            "item_beliefs": {},
        }

        with patch(
            "agents.question_generator.get_llm",
            return_value=_FakeLLM(
                "You mentioned feeling like you let people down—how much does that shape the rest of your day"
            ),
        ):
            result = question_generator(state)

        asked = result["messages"][0]["content"]
        self.assertEqual(asked, "How much does that shape the rest of your day?")

    def test_same_module_fallback_no_longer_defaults_to_you_mentioned(self) -> None:
        state = {
            "turn_index": 3,
            "messages": [
                {"role": "user", "content": "What has that looked like lately?"},
                {"role": "assistant", "content": "I keep feeling like I let people down."},
            ],
            "next_action": {
                "target_item_id": 5,
                "route": "cognitive",
                "style": "gentle_probe",
                "question_kind": "same_module_followup",
                "timeframe_mode": "carry",
                "thread_turn_index": 2,
                "anchor_text": "let people down",
                "rationale": "threaded module follow-up",
            },
            "item_beliefs": {},
        }

        with patch("agents.question_generator.get_llm", return_value=_FakeLLM("   ")):
            result = question_generator(state)

        asked = result["messages"][0]["content"]
        self.assertTrue(bool(result["turn_trace"]["question_generator"]["used_fallback"]))
        self.assertFalse(asked.startswith("You mentioned"))

    def test_module3_self_critical_item_uses_explicit_self_evaluation_fallback(self) -> None:
        state = {
            "turn_index": 2,
            "messages": [
                {"role": "user", "content": "What has that looked like lately?"},
                {"role": "assistant", "content": "I keep replaying what I did wrong."},
            ],
            "next_action": {
                "target_item_id": 8,
                "route": "cognitive",
                "style": "gentle_probe",
                "question_kind": "topic_open",
                "timeframe_mode": "introduce",
                "thread_turn_index": 1,
                "anchor_text": "what I did wrong",
                "rationale": "module-3 follow-up",
            },
            "item_beliefs": {},
        }

        with patch("agents.question_generator.get_llm", return_value=_FakeLLM("   ")):
            result = question_generator(state)

        asked = result["messages"][0]["content"].lower()
        self.assertIn("hard", asked)
        self.assertIn("yourself", asked)
        self.assertNotIn("routine or responsibilities", asked)

    def test_module3_worth_item_uses_burden_or_mattering_wording(self) -> None:
        state = {
            "turn_index": 2,
            "messages": [
                {"role": "user", "content": "What has that looked like lately?"},
                {"role": "assistant", "content": "I keep feeling like I let people down."},
            ],
            "next_action": {
                "target_item_id": 14,
                "route": "cognitive",
                "style": "gentle_probe",
                "question_kind": "topic_open",
                "timeframe_mode": "introduce",
                "thread_turn_index": 1,
                "anchor_text": "let people down",
                "rationale": "module-3 follow-up",
            },
            "item_beliefs": {},
        }

        with patch("agents.question_generator.get_llm", return_value=_FakeLLM("   ")):
            result = question_generator(state)

        asked = result["messages"][0]["content"].lower()
        self.assertTrue("burden" in asked or "matters" in asked)
