from __future__ import annotations

import unittest

from app.cli_eval import _extract_event_kind


class CliEvalFailureLoggingTests(unittest.TestCase):
    def test_valid_opportunistic_shortlist_no_candidates_is_not_parse_failure(self) -> None:
        kind = _extract_event_kind(
            source="llm_opportunistic",
            json_parse_ok=True,
            parse_fail_stage="",
            parse_error_kind="",
            genuine_no_signal_turn=False,
            opportunistic_shortlist_called=True,
            opportunistic_shortlist_parse_ok=True,
            opportunistic_has_strong_offtarget_signal=False,
        )

        self.assertEqual(kind, "opportunistic_no_candidate")

    def test_malformed_json_is_classified_as_true_parse_failure(self) -> None:
        kind = _extract_event_kind(
            source="llm_extractor",
            json_parse_ok=False,
            parse_fail_stage="detail",
            parse_error_kind="json_decode_error",
            genuine_no_signal_turn=False,
            opportunistic_shortlist_called=False,
            opportunistic_shortlist_parse_ok=False,
            opportunistic_has_strong_offtarget_signal=False,
        )

        self.assertEqual(kind, "json_parse_failure")

    def test_extractor_backend_exception_is_classified_as_runtime_failure(self) -> None:
        kind = _extract_event_kind(
            source="llm_extractor_error",
            json_parse_ok=False,
            parse_fail_stage="detail",
            parse_error_kind="llm_call_failed",
            genuine_no_signal_turn=False,
            opportunistic_shortlist_called=False,
            opportunistic_shortlist_parse_ok=False,
            opportunistic_has_strong_offtarget_signal=False,
        )

        self.assertEqual(kind, "extractor_call_failure")


if __name__ == "__main__":
    unittest.main()
