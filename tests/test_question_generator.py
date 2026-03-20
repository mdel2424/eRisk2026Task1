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

