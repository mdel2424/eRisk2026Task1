from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from app.cli import parse_args
from app.cli_eval_helpers import _manifest_payload
from persona import PersonaProfile, create_persona
from persona.simulated_persona import SimulatedPersona


class RuntimeCleanupTests(unittest.TestCase):
    def test_cli_parse_args_no_eval_mode(self) -> None:
        with patch.object(sys, "argv", ["app.cli", "--mode", "eval", "--personas", "5"]):
            args = parse_args()

        self.assertEqual(args.mode, "eval")
        self.assertEqual(args.personas, 5)
        self.assertFalse(hasattr(args, "eval_mode"))

    def test_create_persona_returns_simulated_persona(self) -> None:
        profile = PersonaProfile(
            persona_id="1",
            split="eval",
            family="control_neutral",
            severity_tier="minimal",
            subtype_tag="routine_stable",
            context_tag="routine_stable",
            style_tag="open_but_flat",
            bdi_scores={item_id: 0 for item_id in range(1, 22)},
            depressed=False,
        )

        persona = create_persona(profile)

        self.assertIsInstance(persona, SimulatedPersona)

    def test_manifest_payload_excludes_generator_version(self) -> None:
        profile = PersonaProfile(
            persona_id="1",
            split="eval",
            family="control_neutral",
            severity_tier="minimal",
            subtype_tag="routine_stable",
            context_tag="routine_stable",
            style_tag="open_but_flat",
            bdi_scores={item_id: 0 for item_id in range(1, 22)},
            depressed=False,
        )

        payload = _manifest_payload(persona_count=1, seed=42, profiles=[profile])

        self.assertNotIn("generator_version", payload["run_config"])
        self.assertNotIn("generator_version", payload["profiles"][0])
        self.assertEqual(payload["profiles"][0]["severity_tier"], "minimal")
        self.assertEqual(payload["profiles"][0]["subtype_tag"], "routine_stable")


if __name__ == "__main__":
    unittest.main()
