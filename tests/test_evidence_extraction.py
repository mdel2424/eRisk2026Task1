from __future__ import annotations

import unittest

from agents.evidence_extraction import (
    _apply_precision_gate,
    _apply_precision_gate_batch,
    _coerce_evidence_record,
    _fallback_evidence_from_text,
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


if __name__ == "__main__":
    unittest.main()
