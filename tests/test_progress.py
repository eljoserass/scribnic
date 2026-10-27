"""Progress is observable without exposing clinical text or stopping generation."""

import importlib.util
import io
import logging
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scribnic.backends.moss_transcribe_diarize import MossTranscribeDiarize
from scribnic.config import load_workflow, validate_settings
from scribnic.downloads import PreparedStep
from scribnic.telemetry import GenerationProgress, Telemetry


class ProgressTests(unittest.TestCase):
    def test_progress_reports_counts_at_intervals_and_records_first_token_time(self):
        with patch(
            "scribnic.telemetry.otel.monotonic", side_effect=[10, 12, 13, 18, 20]
        ):
            with self.assertLogs("scribnic.generation", logging.INFO) as logs:
                progress = GenerationProgress("MOSS")
                progress.update(1)
                progress.update(2)
                progress.update(50)
                timings = progress.timings()
        self.assertEqual(len(logs.output), 2)
        self.assertIn("1 tokens", logs.output[0])
        self.assertIn("50 tokens", logs.output[1])
        self.assertEqual(timings["time_to_first_token_seconds"], 2)
        self.assertEqual(timings["generation_seconds"], 10)

    def test_console_progress_and_otlp_logs_share_diagnostics_and_restore_logger(self):
        logger = logging.getLogger("scribnic")
        level, handlers = logger.level, tuple(logger.handlers)
        console = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with Telemetry(directory, progress=console):
                GenerationProgress("MOSS").update(42)
            saved = (Path(directory) / "logs.jsonl").read_text()
        self.assertIn("42 tokens", console.getvalue())
        self.assertIn("42 tokens", saved)
        self.assertEqual(logger.level, level)
        self.assertEqual(tuple(logger.handlers), handlers)


class CpuThreadsTests(unittest.TestCase):
    def backend(self, device="cpu"):
        step = load_workflow("moss_bonsai").steps[0]
        runtime = replace(step.runtime, settings={"threads": 6})
        backend = MossTranscribeDiarize(
            replace(step, runtime=runtime), PreparedStep(Path("unused"))
        )
        state = SimpleNamespace(threads=12)
        backend.torch = SimpleNamespace(
            get_num_threads=lambda: state.threads,
            set_num_threads=lambda value: setattr(state, "threads", value),
        )
        backend.device = SimpleNamespace(type=device)
        return backend, state

    def test_cpu_thread_limit_is_restored_when_inference_fails(self):
        backend, state = self.backend()
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            with backend._cpu_threads() as threads:
                self.assertEqual(threads, 6)
                self.assertEqual(state.threads, 6)
                raise RuntimeError("inference failed")
        self.assertEqual(state.threads, 12)

    def test_gpu_runtime_does_not_override_cpu_threads(self):
        backend, state = self.backend("cuda")
        with backend._cpu_threads() as threads:
            self.assertEqual(threads, 12)
            self.assertEqual(state.threads, 12)

    def test_thread_counts_must_be_positive_integers(self):
        step = load_workflow("moss_bonsai").steps[0]
        for value in (0, -1, 1.5, True, "6"):
            runtime = replace(step.runtime, settings={"threads": value})
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_settings(replace(step, runtime=runtime))


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
    "optional MOSS dependencies not installed",
)
class GenerationCallbackTests(unittest.TestCase):
    def test_progress_preserves_greedy_generation_and_never_requests_stopping(self):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM

        config = Qwen3Config(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            eos_token_id=None,
            pad_token_id=0,
        )
        model = Qwen3ForCausalLM(config).eval()
        inputs = torch.tensor([[1, 2]])
        with torch.inference_mode():
            expected = model.generate(inputs, max_new_tokens=3, do_sample=False)
            progress = GenerationProgress("MOSS")
            criteria = MossTranscribeDiarize._progress_criteria(2, progress)
            actual = model.generate(
                inputs, max_new_tokens=3, do_sample=False, stopping_criteria=criteria
            )
        self.assertTrue(torch.equal(expected, actual))
        self.assertIsNotNone(progress.first_token_at)
