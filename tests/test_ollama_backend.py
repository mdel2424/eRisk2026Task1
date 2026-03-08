from __future__ import annotations

import io
import json
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from core.llm import get_extractor_llm, get_llm
from core.llm_backends import OllamaChatLLM
from core.llm_usage import reset_llm_usage
from core.runtime_policy import resolve_detector_backend


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _DummyChatLLM:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _CaptureUrlopen:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.last_request = None

    def __call__(self, req, timeout=None):
        self.last_request = req
        return _FakeHTTPResponse(self.payload)


class OllamaBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        get_llm.cache_clear()
        get_extractor_llm.cache_clear()
        reset_llm_usage()

    def tearDown(self) -> None:
        get_llm.cache_clear()
        get_extractor_llm.cache_clear()

    def test_resolve_detector_backend_defaults_to_openrouter(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(resolve_detector_backend(), "openrouter")

    def test_resolve_detector_backend_allows_explicit_ollama(self) -> None:
        with patch.dict("os.environ", {"DETECTOR_BACKEND": "ollama"}, clear=False):
            self.assertEqual(resolve_detector_backend(), "ollama")

    def test_llm_builders_select_ollama(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "DETECTOR_BACKEND": "ollama",
                "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
                "OLLAMA_DETECTOR_MODEL": "qwen3.5:4b",
                "OLLAMA_TIMEOUT_SEC": "45",
                "DETECTOR_MAX_NEW_TOKENS": "96",
                "DETECTOR_EXTRACTOR_MAX_NEW_TOKENS": "321",
                "DETECTOR_TEMPERATURE": "0.2",
                "DETECTOR_TOP_P": "0.9",
                "DETECTOR_EXTRACTOR_TEMPERATURE": "0.0",
                "DETECTOR_EXTRACTOR_TOP_P": "1.0",
            },
            clear=False,
        ), patch("core.llm.OllamaChatLLM", _DummyChatLLM):
            detector_llm = get_llm()
            extractor_llm = get_extractor_llm()

        self.assertEqual(detector_llm.kwargs["model_id"], "qwen3.5:4b")
        self.assertEqual(detector_llm.kwargs["base_url"], "http://127.0.0.1:11434")
        self.assertEqual(detector_llm.kwargs["timeout_sec"], 45)
        self.assertEqual(detector_llm.kwargs["max_new_tokens"], 96)
        self.assertEqual(extractor_llm.kwargs["max_new_tokens"], 321)

    def test_ollama_chat_response_parses_content_and_usage(self) -> None:
        llm = OllamaChatLLM(
            model_id="qwen3.5:4b",
            max_new_tokens=96,
            temperature=0.2,
            top_p=0.9,
        )
        payload = {
            "message": {"role": "assistant", "content": "test reply"},
            "prompt_eval_count": 11,
            "eval_count": 5,
        }

        with patch("core.llm_backends.urllib_request.urlopen", return_value=_FakeHTTPResponse(payload)):
            response = llm.invoke([("system", "hello")])

        self.assertEqual(response.backend, "ollama")
        self.assertEqual(response.model_id, "qwen3.5:4b")
        self.assertEqual(response.content, "test reply")
        self.assertEqual(response.prompt_tokens, 11)
        self.assertEqual(response.completion_tokens, 5)
        self.assertEqual(response.total_tokens, 16)

    def test_ollama_chat_request_disables_thinking_on_cpu(self) -> None:
        llm = OllamaChatLLM(
            model_id="qwen3.5:4b",
            max_new_tokens=96,
            temperature=0.2,
            top_p=0.9,
        )
        capture = _CaptureUrlopen({"message": {"role": "assistant", "content": "ok"}})

        with patch.dict("os.environ", {"OLLAMA_THINK_MODE": "auto"}, clear=False), patch(
            "core.llm_backends.cuda_runtime", return_value=(False, 0.0)
        ), patch("core.llm_backends.urllib_request.urlopen", side_effect=capture):
            llm.invoke([("system", "hello")])

        request_payload = json.loads(capture.last_request.data.decode("utf-8"))
        self.assertIs(request_payload["think"], False)

    def test_ollama_chat_request_keeps_thinking_on_with_cuda(self) -> None:
        llm = OllamaChatLLM(
            model_id="qwen3.5:4b",
            max_new_tokens=96,
            temperature=0.2,
            top_p=0.9,
        )
        capture = _CaptureUrlopen({"message": {"role": "assistant", "content": "ok"}})

        with patch.dict("os.environ", {"OLLAMA_THINK_MODE": "auto"}, clear=False), patch(
            "core.llm_backends.cuda_runtime", return_value=(True, 12.0)
        ), patch("core.llm_backends.urllib_request.urlopen", side_effect=capture):
            llm.invoke([("system", "hello")])

        request_payload = json.loads(capture.last_request.data.decode("utf-8"))
        self.assertIs(request_payload["think"], True)

    def test_assert_detector_backend_ready_fails_when_ollama_unreachable(self) -> None:
        from app.cli_runtime_helpers import _assert_detector_backend_ready

        with patch.dict(
            "os.environ",
            {
                "DETECTOR_BACKEND": "ollama",
                "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
                "OLLAMA_DETECTOR_MODEL": "qwen3.5:4b",
            },
            clear=False,
        ), patch("app.cli_runtime_helpers.list_ollama_models", side_effect=RuntimeError("connection refused")):
            with self.assertRaisesRegex(ValueError, "Start Ollama and run `ollama pull qwen3.5:4b`"):
                _assert_detector_backend_ready()

    def test_assert_detector_backend_ready_fails_when_model_missing(self) -> None:
        from app.cli_runtime_helpers import _assert_detector_backend_ready

        with patch.dict(
            "os.environ",
            {
                "DETECTOR_BACKEND": "ollama",
                "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
                "OLLAMA_DETECTOR_MODEL": "qwen3.5:4b",
            },
            clear=False,
        ), patch("app.cli_runtime_helpers.list_ollama_models", return_value=["llama3.2:3b"]):
            with self.assertRaisesRegex(ValueError, "Run `ollama pull qwen3.5:4b`"):
                _assert_detector_backend_ready()

    def test_assert_detector_backend_ready_passes_when_model_available(self) -> None:
        from app.cli_runtime_helpers import _assert_detector_backend_ready

        with patch.dict(
            "os.environ",
            {
                "DETECTOR_BACKEND": "ollama",
                "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
                "OLLAMA_DETECTOR_MODEL": "qwen3.5:4b",
            },
            clear=False,
        ), patch("app.cli_runtime_helpers.list_ollama_models", return_value=["qwen3.5:4b"]):
            _assert_detector_backend_ready()

    def test_print_backend_info_reports_ollama_target(self) -> None:
        from app.cli_runtime_helpers import _print_backend_info

        buffer = io.StringIO()
        with patch.dict(
            "os.environ",
            {
                "DETECTOR_BACKEND": "ollama",
                "OLLAMA_DETECTOR_MODEL": "qwen3.5:4b",
            },
            clear=False,
        ), patch("app.cli_runtime_helpers.cuda_runtime", return_value=(False, 0.0)), redirect_stdout(buffer):
            _print_backend_info(max_api_calls=100, trace_level="compact")

        output = buffer.getvalue()
        self.assertIn("detector=ollama [qwen3.5:4b]", output)
        self.assertIn("persona=simulator [deterministic_local]", output)

    def test_write_eval_artifacts_records_integrity_and_live_env(self) -> None:
        from app.eval_artifacts import write_eval_artifacts

        manifest_payload = {
            "run_config": {"manifest_schema_version": 3, "persona_count": 1, "seed": 42},
            "persona_count": 1,
            "profiles": [
                {
                    "persona_id": "alpha",
                    "split": "eval",
                    "family": "control_neutral",
                    "source": "synthetic",
                    "has_ground_truth": True,
                    "depressed": False,
                    "bdi_scores": {"1": 0},
                    "bdi_total": 0,
                    "key_symptoms": [],
                    "risk_signal": False,
                    "behavior_params": {},
                    "template_bank": "default",
                    "generation_seed": 42001,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(
            "os.environ",
            {
                "DETECTOR_BACKEND": "ollama",
                "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
                "OLLAMA_DETECTOR_MODEL": "qwen3.5:4b",
                "OLLAMA_TIMEOUT_SEC": "120",
            },
            clear=False,
        ):
            output_dir = Path(tmp_dir)
            metrics_payload, _, integrity_payload, config_snapshot, _, _ = write_eval_artifacts(
                output_dir=output_dir,
                conversations=[],
                results=[],
                diagnostics_payload=[],
                overall_rows=[],
                route_distribution=Counter(),
                turns_total=0,
                evidence_turns_nonempty=0,
                evidence_records_total=0,
                extract_source_distribution=Counter(),
                extract_recovery_distribution=Counter(),
                route_policy_distribution=Counter(),
                duplicate_evidence_rows_total=0,
                contradiction_evidence_rows_total=0,
                support_increments_total=0,
                method_weight_usage=Counter(),
                post_floor_new_items_total=0,
                post_floor_nonempty_turns_total=0,
                post_floor_turns_total=0,
                min_turns_for_productivity=1,
                early_stop_reason_distribution=Counter(),
                extract_parse_fail_log_entries=[],
                run_failure_counters=Counter(),
                eval_ids=[],
                manifest_hash="abc123",
                manifest_payload=manifest_payload,
                prior_manifest_info={"exists": True, "hash": "oldhash", "profile_count": 1, "read_error": None},
                prompt_version="v1",
                seed=42,
                persona_count=1,
                processed_profiles=0,
                trace_level="off",
                max_api_calls=100,
                save_diagnostics=False,
                debug_outputs=False,
                run_profile="lean",
                requested_save_diagnostics=False,
                requested_trace_level="off",
                requested_debug_outputs=False,
                all_profiles=[],
            )

        self.assertEqual(metrics_payload["evaluation_mode"], "synthetic")
        self.assertEqual(config_snapshot["env"]["OLLAMA_BASE_URL"], "http://127.0.0.1:11434")
        self.assertEqual(config_snapshot["env"]["OLLAMA_DETECTOR_MODEL"], "qwen3.5:4b")
        self.assertEqual(config_snapshot["env"]["OLLAMA_THINK_MODE"], "auto")
        self.assertEqual(config_snapshot["resolved_backends"]["detector_backend"], "ollama")
        self.assertEqual(integrity_payload["evaluation_mode"], "synthetic")
        self.assertEqual(integrity_payload["persona_regeneration_policy"], "always_regenerate")
        self.assertFalse(integrity_payload["prior_manifest"]["matches_current"])


if __name__ == "__main__":
    unittest.main()
