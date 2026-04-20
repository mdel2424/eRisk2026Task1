from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents.belief_update import update_beliefs
from agents.evidence_extraction import (
    _apply_precision_gate,
    _apply_precision_gate_batch,
    _coerce_evidence_record,
    _fallback_evidence_from_text,
    _records_from_scored_items,
    extract_likelihoods,
)
from core.state import BDI_ITEM_NAMES, build_initial_state


class _FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def invoke(self, _messages):
        if self.calls >= len(self._responses):
            raise AssertionError("Fake extractor LLM received more calls than expected")
        response = self._responses[self.calls]
        self.calls += 1
        return SimpleNamespace(content=response)


class _RaisingLLM:
    def __init__(self, message: str = "forced extractor failure") -> None:
        self._message = message

    def invoke(self, _messages):
        raise RuntimeError(self._message)


def _extract_state(
    *,
    route: str,
    target_item_id: int,
    target_module_id: int,
    latest_message: str,
    previous_question: str = "How have things felt lately?",
):
    state = build_initial_state(persona_id="extractor-test")
    state["has_new_persona_input"] = True
    state["turn_index"] = 1
    state["active_node"] = route
    state["turn"].turn_id = 1
    state["turn"].latest_text_raw = latest_message
    state["messages"] = [
        {"role": "user", "content": previous_question},
        {"role": "assistant", "content": latest_message},
    ]
    state["turn_trace"] = {
        "specialist": {
            "node": route,
            "target_item_id": target_item_id,
            "target_module_id": target_module_id,
        }
    }
    return state


def _scored_payload(
    allowed_item_ids: list[int],
    *,
    supported: dict[int, dict] | None = None,
    include_item_ids: list[int] | None = None,
    extra_items: list[dict] | None = None,
) -> str:
    supported = supported or {}
    include_ids = list(include_item_ids) if include_item_ids is not None else list(allowed_item_ids)
    scores = []
    for item_id in include_ids:
        spec = dict(supported.get(item_id, {}))
        is_supported = bool(spec.pop("supported", item_id in supported))
        scores.append(
            {
                "item_id": item_id,
                "symptom_name": BDI_ITEM_NAMES[item_id],
                "supported": is_supported,
                "confidence": float(spec.pop("confidence", 0.45 if is_supported else 0.0)),
                "intensity": float(spec.pop("intensity", 1.0 if is_supported else 0.0)),
                "anchor_quote": str(spec.pop("anchor_quote", "")),
                "reason": str(spec.pop("reason", "supported in-scope signal" if is_supported else "unsupported")),
                **spec,
            }
        )
    if extra_items:
        scores.extend(extra_items)
    return json.dumps({"scores": scores})


def _opportunistic_shortlist_payload(
    *,
    has_signal: bool,
    candidate_item_ids: list[int],
    anchor_quote: str = "",
    confidence: float = 0.0,
    reason: str = "",
) -> str:
    return json.dumps(
        {
            "has_strong_offtarget_signal": bool(has_signal),
            "candidate_item_ids": list(candidate_item_ids),
            "anchor_quote": anchor_quote,
            "confidence": float(confidence),
            "reason": reason,
        }
    )


class EvidenceExtractionPrecisionTests(unittest.TestCase):
    def test_vague_risk_phrase_no_longer_creates_item9_fallback(self) -> None:
        text = "Sometimes I just want to disappear and keep telling myself to stay safe."

        records = _fallback_evidence_from_text("risk", 1, text)

        self.assertFalse(any(int(record.item_id) == 9 for record in records))

    def test_direct_passive_death_phrase_still_creates_item9_fallback(self) -> None:
        text = "Some days I wish I wasn't here and I don't want to wake up."

        records = _fallback_evidence_from_text("risk", 1, text)
        gated, dropped_count, soft_clamped_count, _ = _apply_precision_gate_batch(records, latest_message=text)

        self.assertTrue(any(int(record.item_id) == 9 for record in gated))
        self.assertEqual(dropped_count, 0)
        self.assertEqual(soft_clamped_count, 0)

    def test_vague_module1_fallback_phrase_is_soft_clamped(self) -> None:
        text = "I feel like I'm just going through the motions lately."

        records = _fallback_evidence_from_text("cognitive", 1, text)
        gated, dropped_count, soft_clamped_count, item_counts = _apply_precision_gate_batch(records, latest_message=text)

        self.assertEqual(dropped_count, 0)
        self.assertEqual(soft_clamped_count, 1)
        self.assertEqual(item_counts["4"]["soft_clamped"], 1)
        self.assertEqual(len(gated), 1)
        self.assertEqual(int(gated[0].item_id), 4)
        self.assertLessEqual(float(gated[0].confidence), 0.35)
        self.assertLessEqual(float(gated[0].intensity), 1.0)
        self.assertTrue(bool(gated[0].support_increment_blocked))

    def test_vague_module3_and_module4_fallback_phrases_are_soft_clamped(self) -> None:
        text = "It's been a lot of self-doubt and brain fog."

        records = _fallback_evidence_from_text("cognitive", 1, text)
        gated, dropped_count, soft_clamped_count, item_counts = _apply_precision_gate_batch(records, latest_message=text)

        self.assertEqual(dropped_count, 0)
        self.assertEqual(soft_clamped_count, 2)
        self.assertEqual(item_counts["8"]["soft_clamped"], 1)
        self.assertEqual(item_counts["19"]["soft_clamped"], 1)
        self.assertEqual({int(record.item_id) for record in gated}, {8, 19})
        self.assertTrue(all(bool(record.support_increment_blocked) for record in gated))

    def test_explicit_reward_loss_self_blame_and_concentration_impairment_survive(self) -> None:
        reward_text = "Nothing feels good anymore and I don't enjoy anything."
        self_blame_text = "I blame myself for everything and feel like a failure."
        concentration_text = "I can't focus and it's hard to decide anything."

        reward_records, reward_dropped, reward_soft, _ = _apply_precision_gate_batch(
            _fallback_evidence_from_text("cognitive", 1, reward_text),
            latest_message=reward_text,
        )
        blame_records, blame_dropped, blame_soft, _ = _apply_precision_gate_batch(
            _fallback_evidence_from_text("cognitive", 1, self_blame_text),
            latest_message=self_blame_text,
        )
        concentration_records, concentration_dropped, concentration_soft, _ = _apply_precision_gate_batch(
            _fallback_evidence_from_text("cognitive", 1, concentration_text),
            latest_message=concentration_text,
        )

        self.assertTrue(any(int(record.item_id) == 4 for record in reward_records))
        self.assertTrue(any(int(record.item_id) in {3, 5} for record in blame_records))
        self.assertTrue(any(int(record.item_id) == 19 for record in concentration_records))
        self.assertEqual(reward_dropped + reward_soft, 0)
        self.assertEqual(blame_dropped + blame_soft, 0)
        self.assertEqual(concentration_dropped + concentration_soft, 0)

    def test_salvaged_weak_guarded_item_is_soft_clamped(self) -> None:
        latest_message = "Mostly I just feel on autopilot."
        record = _coerce_evidence_record(
            "cognitive",
            1,
            {
                "item_id": 4,
                "symptom_name": "Loss of Pleasure",
                "direction": "increase",
                "intensity": 1.0,
                "confidence": 0.4,
                "evidence_text": "on autopilot",
                "reason": "salvaged extractor output",
                "method": "llm_salvage",
            },
            latest_message,
        )

        gated_record, action = _apply_precision_gate(record, latest_message=latest_message)

        self.assertEqual(action, "soft_clamped")
        self.assertIsNotNone(gated_record)
        self.assertLessEqual(float(gated_record.confidence), 0.35)
        self.assertLessEqual(float(gated_record.intensity), 1.0)
        self.assertTrue(bool(gated_record.support_increment_blocked))

    def test_salvaged_direct_item9_evidence_is_kept(self) -> None:
        latest_message = "Sometimes I wish I wasn't here."
        record = _coerce_evidence_record(
            "risk",
            1,
            {
                "item_id": 9,
                "symptom_name": "Suicidal Thoughts or Wishes",
                "direction": "increase",
                "intensity": 1.0,
                "confidence": 0.4,
                "evidence_text": "wish I wasn't here",
                "reason": "salvaged extractor output",
                "method": "llm_salvage",
            },
            latest_message,
        )

        gated_record, action = _apply_precision_gate(record, latest_message=latest_message)

        self.assertEqual(action, "kept")
        self.assertIsNotNone(gated_record)
        self.assertFalse(bool(gated_record.support_increment_blocked))

    def test_passive_item9_phrase_stays_dropped_for_non_scoped_methods(self) -> None:
        latest_message = "Sometimes it feels like it'd be easier if I just disappeared."

        for method in ("llm_salvage", "lexical_fallback", "llm_opportunistic"):
            with self.subTest(method=method):
                record = _coerce_evidence_record(
                    "risk",
                    1,
                    {
                        "item_id": 9,
                        "symptom_name": "Suicidal Thoughts or Wishes",
                        "direction": "increase",
                        "intensity": 1.6,
                        "confidence": 0.7,
                        "evidence_text": "easier if I just disappeared",
                        "reason": "possible passive death phrasing",
                        "method": method,
                    },
                    latest_message,
                )

                gate_kwargs = {"latest_message": latest_message}
                if method == "llm_opportunistic":
                    gate_kwargs["guard_buckets"] = {"item9"}
                gated_record, action = _apply_precision_gate(record, **gate_kwargs)

                self.assertEqual(action, "dropped")
                self.assertIsNone(gated_record)


class EvidenceExtractionV2Tests(unittest.TestCase):
    def test_targeted_assertion_schema_exact_binding_emits_bound_evidence(self) -> None:
        allowed_item_ids = [5, 7, 8, 14]
        state = _extract_state(
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            latest_message="I feel like a burden lately.",
            previous_question="When things feel heavy, what do you tend to tell yourself?",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [14], "anchor_quote": "I feel like a burden", "confidence": 0.70, "reason": "worthlessness language"}',
                json.dumps(
                    {
                        "scores": [
                            {
                                "item_id": item_id,
                                "symptom_name": BDI_ITEM_NAMES[item_id],
                                "assertion": "present" if item_id == 14 else "absent",
                                "confidence": 0.72 if item_id == 14 else 0.0,
                                "intensity": 1.4 if item_id == 14 else 0.0,
                                "anchor_quote": "I feel like a burden" if item_id == 14 else "",
                                "reason": "worthlessness language" if item_id == 14 else "not supported",
                            }
                            for item_id in allowed_item_ids
                        ]
                    }
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        self.assertEqual([int(record.item_id) for record in result["latest_turn_evidence"]], [14])
        self.assertEqual(result["latest_turn_evidence"][0].assertion_label, "present")
        self.assertEqual(result["latest_turn_evidence"][0].binding_status, "exact")
        self.assertEqual(result["latest_turn_evidence"][0].evidence_text, "I feel like a burden")
        self.assertEqual(len(result["latest_turn_assertions"]), 4)
        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual(int(trace["detail_assertion_emitted_evidence_count"]), 1)
        self.assertEqual(int(trace["detail_assertion_counts"]["present"]), 1)
        self.assertEqual(int(trace["detail_assertion_binding_counts"]["exact"]), 1)

    def test_module3_scoped_recovery_maps_hard_on_myself_to_item8(self) -> None:
        state = _extract_state(
            route="cognitive",
            target_item_id=8,
            target_module_id=3,
            latest_message="I'm hard on myself lately.",
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=_RaisingLLM()):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        kept_ids = [int(record.item_id) for record in result["latest_turn_evidence"]]
        self.assertEqual(kept_ids, [8])
        self.assertEqual(str(trace["source"]), "module3_scoped_recovery")
        self.assertTrue(bool(trace["detail_module3_scoped_recovery_applied"]))
        self.assertEqual(trace["detail_module3_scoped_recovery_item_ids"], [8])
        self.assertEqual(str(trace["detail_module3_scoped_recovery_trigger"]), "llm_extractor_error")

    def test_module3_scoped_recovery_maps_letting_people_down_to_item14(self) -> None:
        state = _extract_state(
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            latest_message="I keep feeling like I'm letting people down lately.",
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=_RaisingLLM()):
                result = extract_likelihoods(state)

        kept_ids = [int(record.item_id) for record in result["latest_turn_evidence"]]
        self.assertEqual(kept_ids, [14])
        self.assertGreaterEqual(float(result["latest_turn_evidence"][0].confidence), 0.65)

    def test_module3_scoped_recovery_maps_noncontribution_phrase_to_item14(self) -> None:
        state = _extract_state(
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            latest_message="I don't contribute anything that matters anymore.",
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=_RaisingLLM()):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        kept_ids = [int(record.item_id) for record in result["latest_turn_evidence"]]
        self.assertEqual(kept_ids, [14])
        self.assertTrue(bool(trace["detail_module3_scoped_recovery_applied"]))

    def test_targeted_legacy_supported_payload_is_coerced_to_assertions(self) -> None:
        allowed_item_ids = [5, 7, 8, 14]
        state = _extract_state(
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            latest_message="I feel like a burden lately.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [14], "anchor_quote": "I feel like a burden", "confidence": 0.70, "reason": "worthlessness language"}',
                _scored_payload(
                    allowed_item_ids,
                    supported={
                        14: {
                            "confidence": 0.72,
                            "intensity": 1.4,
                            "anchor_quote": "I feel like a burden",
                            "reason": "worthlessness language",
                        }
                    },
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        self.assertEqual([int(record.item_id) for record in result["latest_turn_evidence"]], [14])
        self.assertEqual(result["latest_turn_evidence"][0].assertion_label, "present")
        self.assertEqual(result["latest_turn_assertions"][3].assertion_label, "present")
        trace = result["turn_trace"]["extract_evidence"]
        self.assertGreaterEqual(int(trace["detail_assertion_legacy_payload_coerce_count"]), 1)

    def test_targeted_assertion_normalized_binding_snaps_to_latest_message(self) -> None:
        records, stats = _records_from_scored_items(
            [
                {
                    "item_id": 14,
                    "symptom_name": "Worthlessness",
                    "assertion": "present",
                    "confidence": 0.7,
                    "intensity": 1.5,
                    "anchor_quote": "i feel   like a burden",
                    "reason": "worthlessness language",
                },
                {
                    "item_id": 5,
                    "symptom_name": "Guilty Feelings",
                    "assertion": "absent",
                    "confidence": 0.0,
                    "intensity": 0.0,
                    "anchor_quote": "",
                    "reason": "not supported",
                },
            ],
            allowed_item_ids=[5, 14],
            node_name="cognitive",
            turn=1,
            latest_message="I FEEL like a burden lately.",
            key_aliases_enabled=True,
            strict_schema_coerce=True,
            item1_strict_gate=False,
            item1_weak_max_conf=0.55,
            item1_weak_max_intensity=1.5,
            method_override="llm_extractor",
            stats_prefix="detail",
            current_detector_question="What do you tend to tell yourself when things feel heavy?",
        )

        self.assertEqual([int(record.item_id) for record in records], [14])
        self.assertEqual(records[0].binding_status, "normalized_exact")
        self.assertEqual(records[0].evidence_text, "I FEEL like a burden")
        self.assertEqual(int(stats["detail_assertion_binding_counts"]["normalized_exact"]), 1)

    def test_targeted_positive_unbound_assertion_is_preserved_but_not_emitted(self) -> None:
        records, stats = _records_from_scored_items(
            [
                {
                    "item_id": 14,
                    "symptom_name": "Worthlessness",
                    "assertion": "present",
                    "confidence": 0.7,
                    "intensity": 1.4,
                    "anchor_quote": "feeling like a burden somehow",
                    "reason": "worthlessness language",
                }
            ],
            allowed_item_ids=[14],
            node_name="cognitive",
            turn=1,
            latest_message="I feel like a burden lately.",
            key_aliases_enabled=True,
            strict_schema_coerce=True,
            item1_strict_gate=False,
            item1_weak_max_conf=0.55,
            item1_weak_max_intensity=1.5,
            method_override="llm_extractor",
            stats_prefix="detail",
            current_detector_question="What do you tend to tell yourself when things feel heavy?",
        )

        self.assertEqual(records, [])
        self.assertEqual(int(stats["detail_assertion_positive_unbound_dropped_count"]), 1)
        self.assertEqual(int(stats["detail_assertion_binding_counts"]["unbound"]), 1)
        assertions = list(stats["detail_assertions"])
        self.assertEqual(len(assertions), 1)
        self.assertEqual(assertions[0].assertion_label, "present")
        self.assertEqual(assertions[0].binding_status, "unbound")

    def test_targeted_absent_and_uncertain_assertions_are_preserved_only(self) -> None:
        records, stats = _records_from_scored_items(
            [
                {
                    "item_id": 16,
                    "symptom_name": "Changes in Sleeping Pattern",
                    "assertion": "absent",
                    "confidence": 0.0,
                    "intensity": 0.0,
                    "anchor_quote": "sleep has been fine",
                    "reason": "explicit denial",
                },
                {
                    "item_id": 20,
                    "symptom_name": "Tiredness or Fatigue",
                    "assertion": "uncertain",
                    "confidence": 0.0,
                    "intensity": 0.0,
                    "anchor_quote": "hard to be exact",
                    "reason": "vague answer",
                },
            ],
            allowed_item_ids=[16, 20],
            node_name="somatic",
            turn=1,
            latest_message="Sleep has been fine, honestly; it's hard to be exact about the rest.",
            key_aliases_enabled=True,
            strict_schema_coerce=True,
            item1_strict_gate=False,
            item1_weak_max_conf=0.55,
            item1_weak_max_intensity=1.5,
            method_override="llm_extractor",
            stats_prefix="detail",
            current_detector_question="How have sleep and fatigue changed compared with usual?",
        )

        self.assertEqual(records, [])
        assertions = list(stats["detail_assertions"])
        self.assertEqual({row.assertion_label for row in assertions}, {"absent", "uncertain"})
        self.assertEqual(int(stats["detail_assertion_emitted_evidence_count"]), 0)

    def test_targeted_contrastive_assertion_emits_when_bound(self) -> None:
        records, stats = _records_from_scored_items(
            [
                {
                    "item_id": 17,
                    "symptom_name": "Irritability",
                    "assertion": "contrastive",
                    "confidence": 0.58,
                    "intensity": 1.0,
                    "anchor_quote": "more toward irritability",
                    "reason": "contrastive answer",
                },
                {
                    "item_id": 1,
                    "symptom_name": "Sadness",
                    "assertion": "absent",
                    "confidence": 0.0,
                    "intensity": 0.0,
                    "anchor_quote": "",
                    "reason": "contrastive answer",
                },
            ],
            allowed_item_ids=[1, 17],
            node_name="cognitive",
            turn=1,
            latest_message="If I had to choose, it leans more toward irritability than outright sadness.",
            key_aliases_enabled=True,
            strict_schema_coerce=True,
            item1_strict_gate=False,
            item1_weak_max_conf=0.55,
            item1_weak_max_intensity=1.5,
            method_override="llm_extractor",
            stats_prefix="detail",
            current_detector_question="In the last two weeks, how often have you felt sad?",
        )

        self.assertEqual([int(record.item_id) for record in records], [17])
        self.assertEqual(records[0].assertion_label, "contrastive")
        self.assertEqual(int(stats["detail_assertion_counts"]["contrastive"]), 1)

    def test_targeted_conditional_assertion_emits_low_strength_positive_evidence(self) -> None:
        records, stats = _records_from_scored_items(
            [
                {
                    "item_id": 18,
                    "symptom_name": "Changes in Appetite",
                    "assertion": "conditional",
                    "confidence": 0.42,
                    "intensity": 1.0,
                    "anchor_quote": "up and down",
                    "reason": "appetite variability",
                }
            ],
            allowed_item_ids=[18],
            node_name="somatic",
            turn=1,
            latest_message="It has been up and down rather than clearly one direction the whole time.",
            key_aliases_enabled=True,
            strict_schema_coerce=True,
            item1_strict_gate=False,
            item1_weak_max_conf=0.55,
            item1_weak_max_intensity=1.5,
            method_override="llm_extractor",
            stats_prefix="detail",
            current_detector_question="How has your appetite changed compared with usual?",
        )

        self.assertEqual([int(record.item_id) for record in records], [18])
        self.assertEqual(records[0].assertion_label, "conditional")
        self.assertEqual(int(stats["detail_assertion_counts"]["conditional"]), 1)

    def test_v2_gate_false_still_runs_stage_two_for_indirect_relevant_reply(self) -> None:
        allowed_item_ids = [3, 5, 6, 7, 8, 14]
        state = _extract_state(
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            latest_message="I don't know, I guess I feel like a burden lately.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": false, "candidate_item_ids": [], "anchor_quote": "", "confidence": 0.22, "reason": "too indirect"}',
                _scored_payload(
                    allowed_item_ids,
                    supported={
                        14: {
                            "confidence": 0.45,
                            "intensity": 1.0,
                            "anchor_quote": "I feel like a burden",
                            "reason": "indirect self-worth statement",
                        }
                    },
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual(fake_llm.calls, 2)
        self.assertEqual(len(result["latest_turn_evidence"]), 1)
        self.assertEqual(int(result["latest_turn_evidence"][0].item_id), 14)
        self.assertTrue(bool(trace["gate_called"]))
        self.assertFalse(bool(trace["gate_target_relevant"]))
        self.assertTrue(bool(trace["stage2_called"]))
        self.assertTrue(bool(trace["gate_soft_false_overridden"]))
        self.assertTrue(bool(trace["detail_called_due_to_gate_false"]))
        self.assertEqual(trace["source"], "llm_extractor")
        self.assertEqual(trace["target_item_id"], 14)
        self.assertEqual(trace["target_module_id"], 3)
        self.assertEqual(trace["allowed_item_ids"], allowed_item_ids)
        self.assertEqual(int(trace["detail_scored_item_count"]), 6)
        self.assertEqual(int(trace["detail_supported_item_count"]), 1)
        self.assertEqual(int(trace["detail_missing_allowed_item_count"]), 0)

    def test_v2_clear_no_change_reply_skips_llm(self) -> None:
        state = _extract_state(
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            latest_message="Not really, things feel about normal.",
        )
        fake_llm = _FakeLLM([])

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual(fake_llm.calls, 0)
        self.assertEqual(result["latest_turn_evidence"], [])
        self.assertFalse(bool(trace["gate_called"]))
        self.assertTrue(bool(trace["clear_no_symptom_skip"]))
        self.assertFalse(bool(trace["opportunistic_called"]))
        self.assertEqual(trace["source"], "clear_no_symptom_skip")

    def test_v2_explicit_denial_with_symptom_word_still_skips_llm(self) -> None:
        state = _extract_state(
            route="somatic",
            target_item_id=16,
            target_module_id=6,
            latest_message="Sleep hasn't really been a problem.",
        )
        fake_llm = _FakeLLM([])

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual(fake_llm.calls, 0)
        self.assertTrue(bool(trace["clear_no_symptom_skip"]))
        self.assertTrue(bool(trace["genuine_no_signal_turn"]))
        self.assertEqual(result["latest_turn_evidence"], [])

    def test_v2_gate_true_stage_two_extracts_targeted_evidence(self) -> None:
        allowed_item_ids = [3, 5, 6, 7, 8, 14]
        state = _extract_state(
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            latest_message="I feel like a burden and honestly pretty worthless lately.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [14], "anchor_quote": "I feel like a burden", "confidence": 0.82, "reason": "explicit worthlessness language"}',
                _scored_payload(
                    allowed_item_ids,
                    supported={
                        14: {
                            "confidence": 0.8,
                            "intensity": 2.0,
                            "anchor_quote": "I feel like a burden",
                            "reason": "explicit burden statement",
                        }
                    },
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual(fake_llm.calls, 2)
        self.assertEqual(len(result["latest_turn_evidence"]), 1)
        self.assertEqual(int(result["latest_turn_evidence"][0].item_id), 14)
        self.assertEqual(trace["source"], "llm_extractor")
        self.assertTrue(bool(trace["gate_called"]))
        self.assertTrue(bool(trace["gate_parse_ok"]))
        self.assertTrue(bool(trace["gate_target_relevant"]))
        self.assertTrue(bool(trace["stage2_called"]))
        self.assertTrue(bool(trace["stage2_parse_ok"]))
        self.assertFalse(bool(trace["opportunistic_called"]))
        self.assertEqual(trace["gate_candidate_item_ids"], [14])
        self.assertEqual(int(trace["detail_scored_item_count"]), 6)
        self.assertEqual(trace["detail_supported_item_ids"], [14])

    def test_v2_gate_parse_failure_still_calls_stage_two(self) -> None:
        allowed_item_ids = [16, 18]
        state = _extract_state(
            route="somatic",
            target_item_id=16,
            target_module_id=6,
            latest_message="A few nights, more than usual.",
        )
        fake_llm = _FakeLLM(
            [
                "not valid json",
                _scored_payload(
                    allowed_item_ids,
                    supported={
                        16: {
                            "confidence": 0.45,
                            "intensity": 1.0,
                            "anchor_quote": "a few nights, more than usual",
                            "reason": "comparison to baseline implies sleep change",
                        }
                    },
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual(fake_llm.calls, 2)
        self.assertFalse(bool(trace["gate_parse_ok"]))
        self.assertTrue(bool(trace["stage2_called"]))
        self.assertTrue(bool(trace["detail_called_due_to_gate_parse_fail"]))
        self.assertTrue(bool(trace["stage2_parse_ok"]))
        self.assertEqual([int(record.item_id) for record in result["latest_turn_evidence"]], [16])

    def test_v2_out_of_scope_stage_two_items_are_dropped(self) -> None:
        allowed_item_ids = [3, 5, 6, 7, 8, 14]
        state = _extract_state(
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            latest_message="I keep thinking I do not contribute anything that matters anymore.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [14], "anchor_quote": "worthless", "confidence": 0.8, "reason": "explicit worthlessness language"}',
                _scored_payload(
                    allowed_item_ids,
                    supported={
                        14: {
                            "confidence": 0.8,
                            "intensity": 2.0,
                            "anchor_quote": "worthless",
                            "reason": "valid",
                        }
                    },
                    extra_items=[
                        {
                            "item_id": 20,
                            "symptom_name": "Tiredness or Fatigue",
                            "supported": True,
                            "confidence": 0.7,
                            "intensity": 2.0,
                            "anchor_quote": "tired",
                            "reason": "out of scope",
                        }
                    ],
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        kept_ids = [int(record.item_id) for record in result["latest_turn_evidence"]]
        self.assertEqual(kept_ids, [14])
        self.assertEqual(int(trace["out_of_scope_item_count"]), 1)
        self.assertEqual(int(trace["detail_scored_item_count"]), 6)

    def test_v2_missing_allowed_items_are_counted_as_unsupported(self) -> None:
        allowed_item_ids = [3, 5, 6, 7, 8, 14]
        state = _extract_state(
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            latest_message="I feel like a burden lately.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [14], "anchor_quote": "burden", "confidence": 0.7, "reason": "self-worth language"}',
                _scored_payload(
                    allowed_item_ids,
                    supported={
                        14: {
                            "confidence": 0.5,
                            "intensity": 1.0,
                            "anchor_quote": "burden",
                            "reason": "indirect worthlessness signal",
                        }
                    },
                    include_item_ids=[14],
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual([int(record.item_id) for record in result["latest_turn_evidence"]], [14])
        self.assertEqual(int(trace["detail_scored_item_count"]), 1)
        self.assertEqual(int(trace["detail_supported_item_count"]), 1)
        self.assertEqual(int(trace["detail_missing_allowed_item_count"]), 5)

    def test_v2_module3_recovery_preempts_out_of_scope_salvage_noise(self) -> None:
        state = _extract_state(
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            latest_message="I keep thinking I do not contribute anything that matters anymore.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [14], "anchor_quote": "worthless", "confidence": 0.8, "reason": "explicit worthlessness language"}',
                'item_id: 20; symptom_name: Tiredness or Fatigue; intensity: 2; confidence: 0.7; evidence_text: "tired"; reason: "out of scope"',
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual([int(record.item_id) for record in result["latest_turn_evidence"]], [14])
        self.assertEqual(str(trace["source"]), "module3_scoped_recovery")
        self.assertTrue(bool(trace["detail_module3_scoped_recovery_applied"]))
        self.assertFalse(bool(trace["salvage_used"]))
        self.assertEqual(int(trace["out_of_scope_item_count"]), 1)

    def test_v2_scoped_empty_triggers_opportunistic_rescue_for_strong_off_target_signal(self) -> None:
        allowed_item_ids = [3, 5, 6, 7, 8, 14]
        state = _extract_state(
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            latest_message="Getting out of bed is a battle and everything takes so much energy.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": false, "candidate_item_ids": [], "anchor_quote": "", "confidence": 0.2, "reason": "off-target signal"}',
                _scored_payload(allowed_item_ids),
                _opportunistic_shortlist_payload(
                    has_signal=True,
                    candidate_item_ids=[15, 20],
                    anchor_quote="everything takes so much energy",
                    confidence=0.72,
                    reason="strong fatigue language outside scope",
                ),
                _scored_payload(
                    [15, 20],
                    supported={
                        15: {
                            "confidence": 0.72,
                            "intensity": 2.0,
                            "anchor_quote": "everything takes so much energy",
                            "reason": "strong fatigue signal",
                        }
                    },
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual(fake_llm.calls, 4)
        self.assertTrue(bool(trace["opportunistic_called"]))
        self.assertTrue(bool(trace["opportunistic_shortlist_called"]))
        self.assertTrue(bool(trace["opportunistic_shortlist_parse_ok"]))
        self.assertTrue(bool(trace["opportunistic_has_strong_offtarget_signal"]))
        self.assertEqual(trace["opportunistic_candidate_item_ids"], [15, 20])
        self.assertTrue(bool(trace["opportunistic_score_called"]))
        self.assertTrue(bool(trace["opportunistic_score_parse_ok"]))
        self.assertTrue(bool(trace["opportunistic_parse_ok"]))
        self.assertEqual(int(trace["opportunistic_raw_items_count"]), 2)
        self.assertEqual(int(trace["opportunistic_scored_item_count"]), 2)
        self.assertEqual(int(trace["opportunistic_supported_item_count"]), 1)
        self.assertEqual(int(trace["opportunistic_missing_item_count"]), 0)
        self.assertEqual(int(trace["opportunistic_kept_items_count"]), 1)
        self.assertEqual(trace["opportunistic_item_ids"], [15])
        self.assertEqual(trace["opportunistic_supported_item_ids"], [15])
        self.assertEqual(trace["source"], "llm_opportunistic")
        self.assertEqual([int(record.item_id) for record in result["latest_turn_evidence"]], [15])
        self.assertEqual(str(result["latest_turn_evidence"][0].method), "llm_opportunistic")

    def test_v2_opportunistic_weak_off_target_evidence_is_dropped(self) -> None:
        allowed_item_ids = [3, 5, 6, 7, 8, 14]
        state = _extract_state(
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            latest_message="I have been tired, I guess.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": false, "candidate_item_ids": [], "anchor_quote": "", "confidence": 0.2, "reason": "off-target signal"}',
                _scored_payload(allowed_item_ids),
                _opportunistic_shortlist_payload(
                    has_signal=True,
                    candidate_item_ids=[15],
                    anchor_quote="tired",
                    confidence=0.6,
                    reason="possible fatigue signal outside scope",
                ),
                _scored_payload(
                    [15],
                    supported={
                        15: {
                            "confidence": 0.5,
                            "intensity": 1.0,
                            "anchor_quote": "tired",
                            "reason": "weak fatigue signal",
                        }
                    },
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual(fake_llm.calls, 4)
        self.assertEqual(result["latest_turn_evidence"], [])
        self.assertTrue(bool(trace["opportunistic_called"]))
        self.assertTrue(bool(trace["opportunistic_shortlist_called"]))
        self.assertTrue(bool(trace["opportunistic_score_called"]))
        self.assertTrue(bool(trace["opportunistic_parse_ok"]))
        self.assertEqual(int(trace["opportunistic_supported_item_count"]), 1)
        self.assertEqual(int(trace["opportunistic_kept_items_count"]), 0)
        self.assertEqual(int(trace["opportunistic_dropped_weak_count"]), 1)
        self.assertEqual(trace["source"], "llm_opportunistic")

    def test_v2_opportunistic_shortlist_with_no_candidates_skips_compact_scoring(self) -> None:
        allowed_item_ids = [3, 5, 6, 7, 8, 14]
        state = _extract_state(
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            latest_message="Work has just been stressful lately.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": false, "candidate_item_ids": [], "anchor_quote": "", "confidence": 0.2, "reason": "off-target signal"}',
                _scored_payload(allowed_item_ids),
                _opportunistic_shortlist_payload(
                    has_signal=False,
                    candidate_item_ids=[],
                    anchor_quote="",
                    confidence=0.2,
                    reason="no strong off-target evidence",
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual(fake_llm.calls, 3)
        self.assertEqual(result["latest_turn_evidence"], [])
        self.assertTrue(bool(trace["opportunistic_called"]))
        self.assertTrue(bool(trace["opportunistic_shortlist_called"]))
        self.assertTrue(bool(trace["opportunistic_shortlist_parse_ok"]))
        self.assertFalse(bool(trace["opportunistic_has_strong_offtarget_signal"]))
        self.assertEqual(trace["opportunistic_candidate_item_ids"], [])
        self.assertFalse(bool(trace["opportunistic_score_called"]))
        self.assertFalse(bool(trace["opportunistic_score_parse_ok"]))
        self.assertEqual(int(trace["opportunistic_raw_items_count"]), 0)

    def test_v2_opportunistic_shortlist_enforces_off_target_candidates(self) -> None:
        allowed_item_ids = [3, 5, 6, 7, 8, 14]
        state = _extract_state(
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            latest_message="Getting out of bed is a battle and everything takes so much energy.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": false, "candidate_item_ids": [], "anchor_quote": "", "confidence": 0.2, "reason": "off-target signal"}',
                _scored_payload(allowed_item_ids),
                _opportunistic_shortlist_payload(
                    has_signal=True,
                    candidate_item_ids=[14, 15, 20],
                    anchor_quote="everything takes so much energy",
                    confidence=0.72,
                    reason="mixed candidate list",
                ),
                _scored_payload(
                    [15, 20],
                    supported={
                        15: {
                            "confidence": 0.72,
                            "intensity": 2.0,
                            "anchor_quote": "everything takes so much energy",
                            "reason": "strong fatigue signal",
                        }
                    },
                    extra_items=[
                        {
                            "item_id": 14,
                            "symptom_name": "Worthlessness",
                            "supported": True,
                            "confidence": 0.9,
                            "intensity": 2.0,
                            "anchor_quote": "worthless",
                            "reason": "should be filtered out",
                        }
                    ],
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual(trace["opportunistic_candidate_item_ids"], [15, 20])
        self.assertEqual([int(record.item_id) for record in result["latest_turn_evidence"]], [15])

    def test_v2_lexical_short_circuit_keeps_target_aligned_prefilter(self) -> None:
        state = _extract_state(
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            latest_message="Lately I feel worthless and like a burden.",
        )
        fake_llm = _FakeLLM([])

        with patch.dict(
            os.environ,
            {"EVIDENCE_LLM_ON_LEXICAL_HIT": "0", "EXTRACTOR_MIN_RECORDS_TARGET": "1"},
            clear=False,
        ):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual(fake_llm.calls, 0)
        self.assertEqual(trace["source"], "lexical_prefilter")
        self.assertFalse(bool(trace["gate_called"]))
        self.assertTrue(any(int(record.item_id) == 14 for record in result["latest_turn_evidence"]))

    def test_v2_risk_route_is_restricted_to_item_nine(self) -> None:
        allowed_item_ids = [9]
        state = _extract_state(
            route="risk",
            target_item_id=9,
            target_module_id=9,
            latest_message="Sometimes I wish I wasn't here.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [9], "anchor_quote": "wish I wasn\'t here", "confidence": 0.8, "reason": "direct passive death language"}',
                _scored_payload(
                    allowed_item_ids,
                    supported={
                        9: {
                            "confidence": 0.7,
                            "intensity": 1.0,
                            "anchor_quote": "wish I wasn't here",
                            "reason": "in scope",
                        }
                    },
                    extra_items=[
                        {
                            "item_id": 14,
                            "symptom_name": "Worthlessness",
                            "supported": True,
                            "confidence": 0.7,
                            "intensity": 2.0,
                            "anchor_quote": "worthless",
                            "reason": "out of scope",
                        }
                    ],
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        kept_ids = [int(record.item_id) for record in result["latest_turn_evidence"]]
        self.assertEqual(trace["allowed_item_ids"], [9])
        self.assertEqual(kept_ids, [9])
        self.assertEqual(int(trace["out_of_scope_item_count"]), 1)
        self.assertTrue(bool(trace["item9_direct_match"]))
        self.assertFalse(bool(trace["item9_passive_risk_match"]))
        self.assertFalse(bool(trace["item9_routed_risk_recovery_applied"]))

    def test_v2_routed_risk_passive_item9_phrase_now_survives(self) -> None:
        allowed_item_ids = [9]
        state = _extract_state(
            route="risk",
            target_item_id=9,
            target_module_id=9,
            latest_message="Sometimes it feels like it'd be easier if I just disappeared.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [9], "anchor_quote": "easier if I just disappeared", "confidence": 0.76, "reason": "passive death phrasing"}',
                _scored_payload(
                    allowed_item_ids,
                    supported={
                        9: {
                            "confidence": 0.74,
                            "intensity": 1.9,
                            "anchor_quote": "easier if I just disappeared",
                            "reason": "passive death/non-existence phrasing",
                        }
                    },
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual([int(record.item_id) for record in result["latest_turn_evidence"]], [9])
        self.assertEqual(int(trace["detail_supported_rows_dropped_by_item9"]), 0)
        self.assertEqual(int(trace["detail_supported_rows_kept_post_validation"]), 1)
        self.assertFalse(bool(trace["item9_direct_match"]))
        self.assertTrue(bool(trace["item9_passive_risk_match"]))
        self.assertTrue(bool(trace["item9_routed_risk_recovery_applied"]))

    def test_v2_item1_gate_still_applies_to_scored_rows(self) -> None:
        allowed_item_ids = [1, 4, 10, 12, 17]
        item1_state = _extract_state(
            route="cognitive",
            target_item_id=1,
            target_module_id=1,
            latest_message="It's just been kind of weird lately, I guess.",
        )
        item1_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [1], "anchor_quote": "weird lately", "confidence": 0.4, "reason": "possible mood change"}',
                _scored_payload(
                    allowed_item_ids,
                    supported={
                        1: {
                            "confidence": 0.8,
                            "intensity": 2.0,
                            "anchor_quote": "weird lately",
                            "reason": "weak mood signal",
                        }
                    },
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=item1_llm):
                item1_result = extract_likelihoods(item1_state)

        item1_record = item1_result["latest_turn_evidence"][0]
        item1_trace = item1_result["turn_trace"]["extract_evidence"]
        self.assertLessEqual(float(item1_record.confidence), 0.55)
        self.assertLessEqual(float(item1_record.intensity), 1.5)
        self.assertEqual(int(item1_trace["detail_supported_rows_dropped_by_item1"]), 0)
        self.assertEqual(int(item1_trace["detail_supported_rows_kept_post_validation"]), 1)

    def test_v2_item1_strict_drop_is_counted_in_trace(self) -> None:
        allowed_item_ids = [1, 4, 10, 12, 17]
        state = _extract_state(
            route="cognitive",
            target_item_id=1,
            target_module_id=1,
            latest_message="It's just been kind of weird lately, I guess.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [1], "anchor_quote": "weird lately", "confidence": 0.4, "reason": "possible mood change"}',
                _scored_payload(
                    allowed_item_ids,
                    supported={
                        1: {
                            "confidence": 0.8,
                            "intensity": 2.0,
                            "anchor_quote": "weird lately",
                            "reason": "weak mood signal",
                        }
                    },
                ),
                _scored_payload(list(range(1, 22))),
            ]
        )

        with patch.dict(
            os.environ,
            {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1", "EXTRACT_ITEM1_STRICT_GATE": "1"},
            clear=False,
        ):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual(result["latest_turn_evidence"], [])
        self.assertEqual(int(trace["detail_supported_rows_dropped_by_item1"]), 1)
        self.assertEqual(int(trace["detail_supported_rows_kept_post_validation"]), 0)

    def test_v2_scoped_module_one_rows_survive_post_validation(self) -> None:
        allowed_item_ids = [1, 4, 10, 12, 17]
        state = _extract_state(
            route="cognitive",
            target_item_id=12,
            target_module_id=1,
            latest_message="Social stuff feels like more effort than it's worth lately.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [12], "anchor_quote": "more effort than it\'s worth", "confidence": 0.6, "reason": "social withdrawal signal"}',
                _scored_payload(
                    allowed_item_ids,
                    supported={
                        12: {
                            "confidence": 0.8,
                            "intensity": 2.0,
                            "anchor_quote": "more effort than it's worth",
                            "reason": "social effort feels unrewarding",
                        }
                    },
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual([int(record.item_id) for record in result["latest_turn_evidence"]], [12])
        self.assertEqual(int(trace["detail_supported_rows_kept_post_validation"]), 1)
        self.assertEqual(int(trace["detail_supported_rows_dropped_by_item9"]), 0)

    def test_v2_scoped_module_three_rows_survive_post_validation(self) -> None:
        allowed_item_ids = [3, 5, 6, 7, 8, 14]
        state = _extract_state(
            route="cognitive",
            target_item_id=7,
            target_module_id=3,
            latest_message="I don't like myself lately.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [7], "anchor_quote": "I don\'t like myself", "confidence": 0.6, "reason": "self-dislike language"}',
                _scored_payload(
                    allowed_item_ids,
                    supported={
                        7: {
                            "confidence": 0.72,
                            "intensity": 2.0,
                            "anchor_quote": "I don't like myself",
                            "reason": "supported self-dislike statement",
                        }
                    },
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual([int(record.item_id) for record in result["latest_turn_evidence"]], [7])
        self.assertEqual(int(trace["detail_supported_rows_kept_post_validation"]), 1)

    def test_v2_module_three_self_blame_phrase_gets_soft_supported(self) -> None:
        allowed_item_ids = [3, 5, 6, 7, 8, 14]
        state = _extract_state(
            route="cognitive",
            target_item_id=5,
            target_module_id=3,
            latest_message="I feel like I'm falling behind and it's my own fault.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [5], "anchor_quote": "my own fault", "confidence": 0.7, "reason": "self-blame language"}',
                _scored_payload(allowed_item_ids),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        kept_ids = [int(record.item_id) for record in result["latest_turn_evidence"]]
        self.assertEqual(kept_ids, [5])
        self.assertEqual(int(trace["detail_module3_soft_support_count"]), 1)
        self.assertEqual(trace["detail_module3_soft_support_item_ids"], [5])

    def test_v2_module_three_generic_stress_stays_unsupported(self) -> None:
        allowed_item_ids = [3, 5, 6, 7, 8, 14]
        state = _extract_state(
            route="cognitive",
            target_item_id=5,
            target_module_id=3,
            latest_message="Work has been stressful and busy, but that's about it.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": false, "candidate_item_ids": [], "anchor_quote": "", "confidence": 0.8, "reason": "generic stress only"}',
                _scored_payload(allowed_item_ids),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual(result["latest_turn_evidence"], [])
        self.assertEqual(int(trace["detail_module3_soft_support_count"]), 0)
        self.assertEqual(trace["detail_module3_soft_support_item_ids"], [])

    def test_v2_worthlessness_specific_language_retains_item14(self) -> None:
        allowed_item_ids = [3, 5, 6, 7, 8, 14]
        state = _extract_state(
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            latest_message="I feel like a burden and I do not contribute anything that matters.",
            previous_question="In the past two weeks, how often have you felt worthless or like you do not matter?",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [7], "anchor_quote": "I feel like a burden", "confidence": 0.7, "reason": "self-evaluation language"}',
                _scored_payload(
                    allowed_item_ids,
                    supported={
                        7: {
                            "confidence": 0.65,
                            "intensity": 1.5,
                            "anchor_quote": "I feel like a burden",
                            "reason": "self-dislike interpretation",
                        }
                    },
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        kept_ids = [int(record.item_id) for record in result["latest_turn_evidence"]]
        self.assertIn(14, kept_ids)
        self.assertTrue(bool(trace["detail_item14_worthlessness_hint_applied"]))
        self.assertTrue(bool(trace["detail_item14_latent_support_applied"]))

    def test_v2_item14_identity_change_phrase_recovers_observed_support(self) -> None:
        allowed_item_ids = [3, 5, 6, 7, 8, 14]
        state = _extract_state(
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            latest_message="I do not like who I am right now and I feel like a failure.",
            previous_question="In the past two weeks, how often have you felt worthless or like you do not matter?",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [14], "anchor_quote": "do not like who I am", "confidence": 0.7, "reason": "identity-level self-evaluation"}',
                _scored_payload(allowed_item_ids),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        records = result["latest_turn_evidence"]
        self.assertEqual([int(record.item_id) for record in records], [14])
        self.assertGreaterEqual(float(records[0].confidence), 0.45)
        self.assertGreaterEqual(float(records[0].intensity), 1.0)
        self.assertTrue(bool(trace["detail_item14_latent_support_applied"]))
        self.assertFalse(bool(trace["detail_module3_soft_support_count"]))

    def test_v2_item14_does_not_fire_on_pure_self_blame(self) -> None:
        allowed_item_ids = [3, 5, 6, 7, 8, 14]
        state = _extract_state(
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            latest_message="It feels like it's my own fault lately.",
            previous_question="In the past two weeks, how often have you felt worthless or like you do not matter?",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [5], "anchor_quote": "my own fault", "confidence": 0.72, "reason": "self-blame language"}',
                _scored_payload(allowed_item_ids),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual([int(record.item_id) for record in result["latest_turn_evidence"]], [5])
        self.assertFalse(bool(trace["detail_item14_latent_support_applied"]))

    def test_v2_explicit_self_dislike_stays_with_item7_not_item14(self) -> None:
        allowed_item_ids = [3, 5, 6, 7, 8, 14]
        state = _extract_state(
            route="cognitive",
            target_item_id=7,
            target_module_id=3,
            latest_message="I dislike myself lately.",
            previous_question="In the past two weeks, how often have you disliked yourself?",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [7], "anchor_quote": "I dislike myself", "confidence": 0.68, "reason": "explicit self-dislike"}',
                _scored_payload(allowed_item_ids),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual([int(record.item_id) for record in result["latest_turn_evidence"]], [7])
        self.assertFalse(bool(trace["detail_item14_latent_support_applied"]))
        self.assertEqual(int(trace["detail_module3_soft_support_count"]), 1)
        self.assertEqual(trace["detail_module3_soft_support_item_ids"], [7])

    def test_v2_item21_mild_direct_decrease_is_kept_under_explicit_question(self) -> None:
        allowed_item_ids = [20, 21]
        state = _extract_state(
            route="somatic",
            target_item_id=21,
            target_module_id=7,
            latest_message="That side of things is a little lower than usual, not a big change though.",
            previous_question="In the past two weeks, how often have you noticed a reduced interest in sexual activity compared with your usual level?",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [21], "anchor_quote": "lower than usual", "confidence": 0.65, "reason": "direct sexual-interest change"}',
                _scored_payload(allowed_item_ids),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual([int(record.item_id) for record in result["latest_turn_evidence"]], [21])
        self.assertTrue(bool(trace["detail_item21_mild_direct_keep_applied"]))

    def test_v2_item21_expanded_mild_phrase_is_kept_under_explicit_question(self) -> None:
        allowed_item_ids = [20, 21]
        state = _extract_state(
            route="somatic",
            target_item_id=21,
            target_module_id=7,
            latest_message="To put it simply, that side of things is a bit lower than usual, not a big change though.",
            previous_question="Over the past two weeks, how interested in sexual activity have you felt compared with your usual level?",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [21], "anchor_quote": "a bit lower than usual", "confidence": 0.62, "reason": "mild direct change"}',
                _scored_payload(allowed_item_ids),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        records = result["latest_turn_evidence"]
        self.assertEqual([int(record.item_id) for record in records], [21])
        self.assertGreaterEqual(float(records[0].confidence), 0.40)
        self.assertTrue(bool(trace["detail_item21_mild_direct_keep_applied"]))

    def test_v2_item21_direct_denial_blocks_mild_direct_keep(self) -> None:
        scored_items = json.loads(_scored_payload([20, 21]))["scores"]

        records, stats = _records_from_scored_items(
            scored_items,
            allowed_item_ids=[20, 21],
            node_name="somatic",
            turn=1,
            latest_message="That side of things has been fine, honestly.",
            key_aliases_enabled=True,
            strict_schema_coerce=True,
            item1_strict_gate=False,
            item1_weak_max_conf=0.55,
            item1_weak_max_intensity=1.5,
            method_override="llm_extractor",
            stats_prefix="detail",
            current_detector_question=(
                "In the past two weeks, how often have you noticed a reduced interest in sexual activity "
                "compared with your usual level?"
            ),
        )

        self.assertEqual(records, [])
        self.assertEqual(int(stats["detail_item21_direct_denial_blocked"]), 1)
        self.assertEqual(int(stats["detail_item21_mild_direct_keep_applied"]), 0)

    def test_generic_ambiguous_shift_alone_does_not_produce_support(self) -> None:
        records, stats = _records_from_scored_items(
            json.loads(
                _scored_payload(
                    [13, 19],
                    supported={
                        13: {
                            "confidence": 0.42,
                            "intensity": 1.0,
                            "anchor_quote": "seems a little different",
                            "reason": "generic shift only",
                        }
                    },
                )
            )["scores"],
            allowed_item_ids=[13, 19],
            node_name="cognitive",
            turn=1,
            latest_message="I haven't tracked it that closely, but it seems a little different.",
            key_aliases_enabled=True,
            strict_schema_coerce=True,
            item1_strict_gate=False,
            item1_weak_max_conf=0.55,
            item1_weak_max_intensity=1.5,
            method_override="llm_extractor",
            stats_prefix="detail",
            current_detector_question="In the past two weeks, how often has concentration been harder than usual?",
        )

        self.assertEqual(records, [])
        self.assertEqual(int(stats["detail_generic_shift_blocked_count"]), 1)

    def test_generic_ambiguous_shift_with_concrete_symptom_example_can_still_survive(self) -> None:
        records, stats = _records_from_scored_items(
            json.loads(
                _scored_payload(
                    [13, 19],
                    supported={
                        13: {
                            "confidence": 0.50,
                            "intensity": 1.2,
                            "anchor_quote": "keep rereading the same page",
                            "reason": "concentration difficulty",
                        }
                    },
                )
            )["scores"],
            allowed_item_ids=[13, 19],
            node_name="cognitive",
            turn=1,
            latest_message=(
                "I haven't tracked it that closely, but it seems a little different and I keep rereading the same page."
            ),
            key_aliases_enabled=True,
            strict_schema_coerce=True,
            item1_strict_gate=False,
            item1_weak_max_conf=0.55,
            item1_weak_max_intensity=1.5,
            method_override="llm_extractor",
            stats_prefix="detail",
            current_detector_question="In the past two weeks, how often has concentration been harder than usual?",
        )

        self.assertEqual({int(record.item_id) for record in records}, {13, 19})
        self.assertTrue(any(int(record.item_id) == 19 and float(record.confidence) >= 0.6 for record in records))
        self.assertEqual(int(stats["detail_generic_shift_with_symptom_kept_count"]), 2)

    def test_concentration_phrase_without_llm_support_recovers_item19(self) -> None:
        records, stats = _records_from_scored_items(
            [],
            allowed_item_ids=[13, 19],
            node_name="cognitive",
            turn=1,
            latest_message="I keep rereading the same page and zoning out halfway through.",
            key_aliases_enabled=True,
            strict_schema_coerce=True,
            item1_strict_gate=False,
            item1_weak_max_conf=0.55,
            item1_weak_max_intensity=1.5,
            method_override="llm_extractor",
            stats_prefix="detail",
            current_detector_question="In the past two weeks, how often has concentration been harder than usual?",
        )

        self.assertEqual([int(record.item_id) for record in records], [19])
        self.assertGreaterEqual(float(records[0].confidence), 0.6)

    def test_concentration_phrase_can_add_item19_alongside_indecisiveness(self) -> None:
        records, stats = _records_from_scored_items(
            json.loads(
                _scored_payload(
                    [13, 19],
                    supported={
                        13: {
                            "confidence": 0.50,
                            "intensity": 1.2,
                            "anchor_quote": "hard to decide anything",
                            "reason": "indecisiveness",
                        }
                    },
                )
            )["scores"],
            allowed_item_ids=[13, 19],
            node_name="cognitive",
            turn=1,
            latest_message="I keep rereading the same page and zoning out halfway through.",
            key_aliases_enabled=True,
            strict_schema_coerce=True,
            item1_strict_gate=False,
            item1_weak_max_conf=0.55,
            item1_weak_max_intensity=1.5,
            method_override="llm_extractor",
            stats_prefix="detail",
            current_detector_question="In the past two weeks, how often has concentration been harder than usual?",
        )

        self.assertEqual({int(record.item_id) for record in records}, {13, 19})
        self.assertTrue(any(int(record.item_id) == 19 and float(record.confidence) >= 0.6 for record in records))

    def test_generic_ambiguous_shift_does_not_support_module_three_without_self_evaluation_language(self) -> None:
        records, stats = _records_from_scored_items(
            json.loads(
                _scored_payload(
                    [5, 7, 8, 14],
                    supported={
                        14: {
                            "confidence": 0.40,
                            "intensity": 1.0,
                            "anchor_quote": "some shift",
                            "reason": "generic uncertainty only",
                        }
                    },
                )
            )["scores"],
            allowed_item_ids=[5, 7, 8, 14],
            node_name="cognitive",
            turn=1,
            latest_message="It's hard to be exact, though I think there has been some shift.",
            key_aliases_enabled=True,
            strict_schema_coerce=True,
            item1_strict_gate=False,
            item1_weak_max_conf=0.55,
            item1_weak_max_intensity=1.5,
            method_override="llm_extractor",
            stats_prefix="detail",
            current_detector_question="In the past two weeks, how often have you felt badly about yourself?",
        )

        self.assertEqual(records, [])
        self.assertEqual(int(stats["detail_generic_shift_blocked_count"]), 1)

    def test_v2_item18_heavier_language_without_appetite_change_is_rejected(self) -> None:
        allowed_item_ids = [16, 18]
        state = _extract_state(
            route="somatic",
            target_item_id=18,
            target_module_id=6,
            latest_message="Maybe that's why everything feels heavier.",
            previous_question="In the past two weeks, how often have you noticed a change in your appetite compared with your usual eating habits?",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [18], "anchor_quote": "everything feels heavier", "confidence": 0.55, "reason": "possible somatic change"}',
                _scored_payload(
                    allowed_item_ids,
                    supported={
                        18: {
                            "confidence": 0.6,
                            "intensity": 1.5,
                            "anchor_quote": "everything feels heavier",
                            "reason": "incorrect appetite inference",
                        }
                    },
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual(result["latest_turn_evidence"], [])
        self.assertTrue(bool(trace["detail_item18_change_signal_rejected"]))
        self.assertFalse(bool(trace["detail_item18_change_signal_match"]))

    def test_v2_item18_direct_no_change_reply_stays_unsupported(self) -> None:
        allowed_item_ids = [16, 18]
        state = _extract_state(
            route="somatic",
            target_item_id=18,
            target_module_id=6,
            latest_message="I haven't noticed anything different there with eating.",
            previous_question="In the past two weeks, how often have you noticed a change in your appetite compared with your usual eating habits?",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [18], "anchor_quote": "nothing different", "confidence": 0.6, "reason": "possible appetite response"}',
                _scored_payload(
                    allowed_item_ids,
                    supported={
                        18: {
                            "confidence": 0.55,
                            "intensity": 1.0,
                            "anchor_quote": "nothing different",
                            "reason": "incorrect appetite support",
                        }
                    },
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual(result["latest_turn_evidence"], [])
        self.assertTrue(bool(trace["detail_item18_change_signal_rejected"]))

    def test_v2_item18_strong_appetite_change_still_survives(self) -> None:
        allowed_item_ids = [16, 18]
        state = _extract_state(
            route="somatic",
            target_item_id=18,
            target_module_id=6,
            latest_message="I'm not eating at all or just grabbing junk because I can't be bothered.",
            previous_question="In the past two weeks, how often have you noticed a change in your appetite compared with your usual eating habits?",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [18], "anchor_quote": "not eating at all", "confidence": 0.8, "reason": "direct appetite change"}',
                _scored_payload(
                    allowed_item_ids,
                    supported={
                        18: {
                            "confidence": 0.75,
                            "intensity": 2.0,
                            "anchor_quote": "not eating at all",
                            "reason": "direct appetite change",
                        }
                    },
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual([int(record.item_id) for record in result["latest_turn_evidence"]], [18])
        self.assertTrue(bool(trace["detail_item18_change_signal_match"]))

    def test_v2_item18_variability_phrase_now_survives(self) -> None:
        allowed_item_ids = [16, 18]
        state = _extract_state(
            route="somatic",
            target_item_id=18,
            target_module_id=6,
            latest_message="A bit, it has been up and down rather than clearly one direction the whole time.",
            previous_question="In the past two weeks, how often have you noticed a change in your appetite compared with your usual eating habits?",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [18], "anchor_quote": "up and down", "confidence": 0.64, "reason": "explicit appetite variability"}',
                _scored_payload(allowed_item_ids),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        records = result["latest_turn_evidence"]
        self.assertEqual([int(record.item_id) for record in records], [18])
        self.assertTrue(bool(trace["detail_item18_change_signal_match"]))
        self.assertTrue(bool(trace["detail_item18_variability_match"]))

    def test_v2_item16_mixed_sleep_instability_phrase_survives(self) -> None:
        allowed_item_ids = [11, 15, 16, 20]
        state = _extract_state(
            route="somatic",
            target_item_id=16,
            target_module_id=6,
            latest_message="Sleep is a mess, I'm up all night or sleeping too much and still tired.",
            previous_question="In the past two weeks, how often have you had trouble falling asleep or staying asleep compared with your usual pattern?",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [16], "anchor_quote": "sleep is a mess", "confidence": 0.68, "reason": "mixed sleep instability"}',
                _scored_payload(allowed_item_ids),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual([int(record.item_id) for record in result["latest_turn_evidence"]], [16])
        self.assertTrue(bool(trace["detail_item16_sleep_instability_match"]))

    def test_v2_item7_soft_self_evaluation_phrase_survives(self) -> None:
        records, stats = _records_from_scored_items(
            json.loads(_scored_payload([5, 7, 8, 14]))["scores"],
            allowed_item_ids=[5, 7, 8, 14],
            node_name="cognitive",
            turn=1,
            latest_message="I've lost confidence in myself lately.",
            key_aliases_enabled=True,
            strict_schema_coerce=True,
            item1_strict_gate=False,
            item1_weak_max_conf=0.55,
            item1_weak_max_intensity=1.5,
            method_override="llm_extractor",
            stats_prefix="detail",
            current_detector_question="In the past two weeks, how often have you felt badly about yourself?",
        )

        self.assertEqual([int(record.item_id) for record in records], [7])
        self.assertEqual(int(stats["detail_item7_soft_self_evaluation_applied"]), 1)

    def test_v2_item8_soft_self_criticism_phrase_survives(self) -> None:
        records, stats = _records_from_scored_items(
            json.loads(_scored_payload([5, 7, 8, 14]))["scores"],
            allowed_item_ids=[5, 7, 8, 14],
            node_name="cognitive",
            turn=1,
            latest_message="I keep thinking I should be doing better and I second-guess everything.",
            key_aliases_enabled=True,
            strict_schema_coerce=True,
            item1_strict_gate=False,
            item1_weak_max_conf=0.55,
            item1_weak_max_intensity=1.5,
            method_override="llm_extractor",
            stats_prefix="detail",
            current_detector_question="In the past two weeks, how often have you been hard on yourself or critical of your choices?",
        )

        self.assertEqual([int(record.item_id) for record in records], [8])
        self.assertEqual(int(stats["detail_item8_soft_self_criticism_applied"]), 1)

    def test_v2_self_evaluation_quote_retargets_wrong_item_to_item7(self) -> None:
        records, stats = _records_from_scored_items(
            json.loads(
                _scored_payload(
                    [7, 8, 14],
                    supported={
                        14: {
                            "assertion": "present",
                            "confidence": 0.58,
                            "intensity": 1.0,
                            "anchor_quote": "I've lost confidence in myself",
                            "reason": "self-evaluation language",
                        }
                    },
                )
            )["scores"],
            allowed_item_ids=[7, 8, 14],
            node_name="cognitive",
            turn=1,
            latest_message="I've lost confidence in myself lately.",
            key_aliases_enabled=True,
            strict_schema_coerce=True,
            item1_strict_gate=False,
            item1_weak_max_conf=0.55,
            item1_weak_max_intensity=1.5,
            method_override="llm_extractor",
            stats_prefix="detail",
            current_detector_question="In the past two weeks, how often have you felt badly about yourself?",
        )

        self.assertEqual([int(record.item_id) for record in records], [7])
        self.assertEqual(int(stats["detail_self_evaluation_retarget_count"]), 1)
        self.assertEqual(int(stats["detail_self_evaluation_suppressed_count"]), 1)

    def test_v2_self_evaluation_quote_retargets_wrong_item_to_item8(self) -> None:
        records, stats = _records_from_scored_items(
            json.loads(
                _scored_payload(
                    [7, 8, 14],
                    supported={
                        14: {
                            "assertion": "present",
                            "confidence": 0.56,
                            "intensity": 1.0,
                            "anchor_quote": "I'm hard on myself",
                            "reason": "self-evaluative phrasing",
                        }
                    },
                )
            )["scores"],
            allowed_item_ids=[7, 8, 14],
            node_name="cognitive",
            turn=1,
            latest_message="I'm hard on myself lately.",
            key_aliases_enabled=True,
            strict_schema_coerce=True,
            item1_strict_gate=False,
            item1_weak_max_conf=0.55,
            item1_weak_max_intensity=1.5,
            method_override="llm_extractor",
            stats_prefix="detail",
            current_detector_question="In the past two weeks, how often have you been hard on yourself or critical of your choices?",
        )

        self.assertEqual([int(record.item_id) for record in records], [8])
        self.assertEqual(int(stats["detail_self_evaluation_retarget_count"]), 1)

    def test_v2_self_evaluation_quote_retargets_wrong_item_to_item14(self) -> None:
        records, stats = _records_from_scored_items(
            json.loads(
                _scored_payload(
                    [7, 8, 14],
                    supported={
                        8: {
                            "assertion": "present",
                            "confidence": 0.6,
                            "intensity": 1.0,
                            "anchor_quote": "I don't measure up",
                            "reason": "self-evaluative phrasing",
                        }
                    },
                )
            )["scores"],
            allowed_item_ids=[7, 8, 14],
            node_name="cognitive",
            turn=1,
            latest_message="I don't measure up lately.",
            key_aliases_enabled=True,
            strict_schema_coerce=True,
            item1_strict_gate=False,
            item1_weak_max_conf=0.55,
            item1_weak_max_intensity=1.5,
            method_override="llm_extractor",
            stats_prefix="detail",
            current_detector_question="In the past two weeks, how often have you felt worthless or like you do not matter?",
        )

        self.assertEqual([int(record.item_id) for record in records], [14])
        self.assertEqual(int(stats["detail_self_evaluation_retarget_count"]), 1)

    def test_v2_self_evaluation_cluster_suppresses_generic_stress_without_self_judgment(self) -> None:
        records, stats = _records_from_scored_items(
            json.loads(
                _scored_payload(
                    [7, 8, 14],
                    supported={
                        7: {
                            "assertion": "present",
                            "confidence": 0.55,
                            "intensity": 1.0,
                            "anchor_quote": "work has been stressful",
                            "reason": "generic stress language",
                        }
                    },
                )
            )["scores"],
            allowed_item_ids=[7, 8, 14],
            node_name="cognitive",
            turn=1,
            latest_message="Work has been stressful lately.",
            key_aliases_enabled=True,
            strict_schema_coerce=True,
            item1_strict_gate=False,
            item1_weak_max_conf=0.55,
            item1_weak_max_intensity=1.5,
            method_override="llm_extractor",
            stats_prefix="detail",
            current_detector_question="In the past two weeks, how often have you felt badly about yourself?",
        )

        self.assertEqual(records, [])
        self.assertEqual(int(stats["detail_self_evaluation_suppressed_count"]), 1)
        self.assertEqual(int(stats["detail_self_evaluation_retarget_count"]), 0)

    def test_v2_self_evaluation_cluster_allows_separable_multi_cue_support(self) -> None:
        records, stats = _records_from_scored_items(
            json.loads(
                _scored_payload(
                    [7, 8, 14],
                    supported={
                        7: {
                            "assertion": "present",
                            "confidence": 0.62,
                            "intensity": 1.0,
                            "anchor_quote": "I don't like myself",
                            "reason": "explicit self-dislike",
                        },
                        8: {
                            "assertion": "present",
                            "confidence": 0.6,
                            "intensity": 1.0,
                            "anchor_quote": "I should be doing better",
                            "reason": "self-critical phrasing",
                        },
                    },
                )
            )["scores"],
            allowed_item_ids=[7, 8, 14],
            node_name="cognitive",
            turn=1,
            latest_message="I don't like myself lately, and I keep thinking I should be doing better.",
            key_aliases_enabled=True,
            strict_schema_coerce=True,
            item1_strict_gate=False,
            item1_weak_max_conf=0.55,
            item1_weak_max_intensity=1.5,
            method_override="llm_extractor",
            stats_prefix="detail",
            current_detector_question="In the past two weeks, how often have you felt badly about yourself?",
        )

        self.assertEqual({int(record.item_id) for record in records}, {7, 8})
        self.assertEqual(int(stats["detail_self_evaluation_suppressed_count"]), 0)
        self.assertEqual(int(stats["detail_self_evaluation_multi_item_suppression_count"]), 0)

    def test_v2_sadness_contrastive_reply_supports_irritability_sibling_when_in_scope(self) -> None:
        allowed_item_ids = [1, 4, 10, 12, 17]
        state = _extract_state(
            route="cognitive",
            target_item_id=1,
            target_module_id=1,
            latest_message="If I had to choose, it leans more toward irritability than outright sadness.",
            previous_question="In the last two weeks, how often have you felt sad—like a few days, most days, or nearly every day?",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [17], "anchor_quote": "more toward irritability", "confidence": 0.58, "reason": "contrastive sibling answer"}',
                _scored_payload(allowed_item_ids),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual([int(record.item_id) for record in result["latest_turn_evidence"]], [17])
        self.assertTrue(bool(trace["detail_contrastive_sibling_support_applied"]))

    def test_v2_sadness_contrastive_reply_does_not_inject_irritability_when_out_of_scope(self) -> None:
        records, stats = _records_from_scored_items(
            json.loads(_scored_payload([1, 4, 10, 12]))["scores"],
            allowed_item_ids=[1, 4, 10, 12],
            node_name="cognitive",
            turn=1,
            latest_message="If I had to choose, it leans more toward irritability than outright sadness.",
            key_aliases_enabled=True,
            strict_schema_coerce=True,
            item1_strict_gate=False,
            item1_weak_max_conf=0.55,
            item1_weak_max_intensity=1.5,
            method_override="llm_extractor",
            stats_prefix="detail",
            current_detector_question="In the last two weeks, how often have you felt sad—like a few days, most days, or nearly every day?",
        )

        self.assertEqual(records, [])
        self.assertEqual(int(stats["detail_contrastive_sibling_support_applied"]), 0)

    def test_v2_scoped_module_four_rows_survive_post_validation(self) -> None:
        allowed_item_ids = [13, 19]
        state = _extract_state(
            route="cognitive",
            target_item_id=13,
            target_module_id=4,
            latest_message="Decisions take me longer than they used to.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [13], "anchor_quote": "decisions take me longer", "confidence": 0.62, "reason": "decision impairment"}',
                _scored_payload(
                    allowed_item_ids,
                    supported={
                        13: {
                            "confidence": 0.7,
                            "intensity": 1.7,
                            "anchor_quote": "Decisions take me longer",
                            "reason": "decision latency increased",
                        }
                    },
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual([int(record.item_id) for record in result["latest_turn_evidence"]], [13])
        self.assertEqual(int(trace["detail_supported_rows_kept_post_validation"]), 1)

    def test_v2_item9_scored_rows_still_obey_strict_guard(self) -> None:
        allowed_item_ids = [9]
        state = _extract_state(
            route="risk",
            target_item_id=9,
            target_module_id=9,
            latest_message="I don't know, things just feel bad lately.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [9], "anchor_quote": "feel bad", "confidence": 0.55, "reason": "possible risk"}',
                _scored_payload(
                    allowed_item_ids,
                    supported={
                        9: {
                            "confidence": 0.75,
                            "intensity": 2.0,
                            "anchor_quote": "feel bad lately",
                            "reason": "model marked possible risk",
                        }
                    },
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual(result["latest_turn_evidence"], [])
        self.assertEqual(int(trace["detail_supported_rows_dropped_by_item9"]), 1)
        self.assertEqual(int(trace["detail_supported_rows_kept_post_validation"]), 0)
        self.assertFalse(bool(trace["item9_direct_match"]))
        self.assertFalse(bool(trace["item9_passive_risk_match"]))
        self.assertFalse(bool(trace["item9_routed_risk_recovery_applied"]))
        self.assertTrue(bool(trace["opportunistic_skipped_on_risk"]))
        self.assertFalse(bool(trace["opportunistic_called"]))

    def test_v2_routed_risk_not_being_here_feels_easier_phrase_survives(self) -> None:
        allowed_item_ids = [9]
        state = _extract_state(
            route="risk",
            target_item_id=9,
            target_module_id=9,
            latest_message="I have had stretches where not being here feels easier, even if I am not planning anything.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [9], "anchor_quote": "not being here feels easier", "confidence": 0.76, "reason": "passive death phrasing"}',
                _scored_payload(
                    allowed_item_ids,
                    supported={
                        9: {
                            "confidence": 0.9,
                            "intensity": 1.7,
                            "anchor_quote": "not being here feels easier",
                            "reason": "passive death phrasing",
                        }
                    },
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual([int(record.item_id) for record in result["latest_turn_evidence"]], [9])
        self.assertEqual(int(trace["detail_supported_rows_dropped_by_item9"]), 0)
        self.assertTrue(bool(trace["item9_passive_risk_match"]))
        self.assertTrue(bool(trace["item9_routed_risk_recovery_applied"]))

    def test_v2_non_risk_route_still_drops_passive_item9_phrase(self) -> None:
        allowed_item_ids = [13, 19]
        state = _extract_state(
            route="cognitive",
            target_item_id=13,
            target_module_id=4,
            latest_message="Sometimes it feels like it'd be easier if I just disappeared.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [13], "anchor_quote": "easier if I just disappeared", "confidence": 0.4, "reason": "mixed signal"}',
                _scored_payload(
                    allowed_item_ids,
                    include_item_ids=[13, 19],
                    extra_items=[
                        {
                            "item_id": 9,
                            "symptom_name": "Suicidal Thoughts or Wishes",
                            "supported": True,
                            "confidence": 0.76,
                            "intensity": 2.0,
                            "anchor_quote": "easier if I just disappeared",
                            "reason": "passive death phrasing",
                        }
                    ],
                ),
                _opportunistic_shortlist_payload(
                    has_signal=True,
                    candidate_item_ids=[9],
                    anchor_quote="easier if I just disappeared",
                    confidence=0.8,
                    reason="possible off-target risk signal",
                ),
                _scored_payload(
                    [9],
                    supported={
                        9: {
                            "confidence": 0.76,
                            "intensity": 2.0,
                            "anchor_quote": "easier if I just disappeared",
                            "reason": "passive death phrasing",
                        }
                    },
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual(result["latest_turn_evidence"], [])
        self.assertEqual(int(trace["detail_supported_rows_dropped_by_item9"]), 0)
        self.assertEqual(int(trace["opportunistic_kept_items_count"]), 0)

    def test_v2_opportunistic_rows_remain_under_strict_guarded_precision(self) -> None:
        allowed_item_ids = [13, 19]
        state = _extract_state(
            route="cognitive",
            target_item_id=13,
            target_module_id=4,
            latest_message="Mostly I just feel on autopilot.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": false, "candidate_item_ids": [], "anchor_quote": "", "confidence": 0.2, "reason": "off-target or vague"}',
                _scored_payload(allowed_item_ids),
                _opportunistic_shortlist_payload(
                    has_signal=True,
                    candidate_item_ids=[4],
                    anchor_quote="on autopilot",
                    confidence=0.7,
                    reason="possible strong off-target pleasure-loss signal",
                ),
                _scored_payload(
                    [4],
                    supported={
                        4: {
                            "confidence": 0.72,
                            "intensity": 2.0,
                            "anchor_quote": "on autopilot",
                            "reason": "possible pleasure loss",
                        }
                    },
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                result = extract_likelihoods(state)

        trace = result["turn_trace"]["extract_evidence"]
        self.assertEqual(result["latest_turn_evidence"], [])
        self.assertTrue(bool(trace["opportunistic_called"]))
        self.assertEqual(int(trace["opportunistic_supported_item_count"]), 1)
        self.assertEqual(int(trace["opportunistic_kept_items_count"]), 0)
        self.assertEqual(int(trace["opportunistic_dropped_weak_count"]), 1)

    def test_belief_update_consumes_unified_rows_without_schema_change(self) -> None:
        allowed_item_ids = [3, 5, 6, 7, 8, 14]
        state = _extract_state(
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            latest_message="I feel like a burden and worthless lately.",
        )
        fake_llm = _FakeLLM(
            [
                '{"target_relevant": true, "candidate_item_ids": [14], "anchor_quote": "burden", "confidence": 0.8, "reason": "explicit worthlessness language"}',
                _scored_payload(
                    allowed_item_ids,
                    supported={
                        14: {
                            "confidence": 0.8,
                            "intensity": 2.0,
                            "anchor_quote": "burden",
                            "reason": "explicit worthlessness language",
                        }
                    },
                ),
            ]
        )

        with patch.dict(os.environ, {"EVIDENCE_LLM_ON_LEXICAL_HIT": "1"}, clear=False):
            with patch("agents.evidence_extraction.get_extractor_llm", return_value=fake_llm):
                extracted = extract_likelihoods(state)

        updated_state = _extract_state(
            route="cognitive",
            target_item_id=14,
            target_module_id=3,
            latest_message="I feel like a burden and worthless lately.",
        )
        updated_state["latest_turn_likelihoods"] = extracted["latest_turn_likelihoods"]
        updated = update_beliefs(updated_state)

        self.assertEqual(int(updated["item_beliefs"][14].support_count), 1)
        self.assertGreater(float(updated["item_beliefs"][14].expected_score), 1.0)


if __name__ == "__main__":
    unittest.main()
