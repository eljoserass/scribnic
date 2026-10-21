"""Optional CPU profiler checks; no Torch import in the base test environment."""

import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scribnic.profiling import profile_directory, torch_profile


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "optional Torch extra not installed"
)
class TorchProfilingTests(unittest.TestCase):
    def test_trace_labels_and_hooks_are_cleaned_up_even_on_error(self):
        import torch

        encoder = torch.nn.Linear(2, 2)
        decoder = torch.nn.Linear(2, 2)
        backbone = SimpleNamespace(whisper_encoder=encoder, language_model=decoder)
        model = SimpleNamespace(model=backbone)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with profile_directory(root):
                with self.assertRaisesRegex(ValueError, "test failure"):
                    with torch_profile(model, torch, torch.device("cpu")):
                        inputs = encoder(torch.ones(1, 2))
                        decoder(inputs)
                        decoder(inputs)
                        raise ValueError("test failure")
            summary = json.loads((root / "moss-profile.json").read_text())
            self.assertEqual(summary["status"], "error")
            with gzip.open(root / summary["traces"][0], "rt") as stream:
                events = json.load(stream)["traceEvents"]
            names = {event["name"] for event in events}
            self.assertTrue({"moss.encoder", "moss.prefill", "moss.decode"} <= names)
        for module in (encoder, decoder):
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    def test_windows_cover_every_forward_including_the_last_partial_window(self):
        import torch

        decoder = torch.nn.Linear(2, 2)
        model = SimpleNamespace(model=SimpleNamespace(language_model=decoder))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                profile_directory(root),
                torch_profile(
                    model, torch, torch.device("cpu"), window_steps=2
                ) as metadata,
            ):
                for _ in range(5):
                    decoder(torch.ones(1, 2))
            summary = json.loads((root / "moss-profile.json").read_text())
            phases = {row["name"]: row for row in summary["phases"]}
            self.assertEqual(phases["moss.prefill"]["calls"], 1)
            self.assertEqual(phases["moss.decode"]["calls"], 4)
            self.assertEqual(summary["status"], "success")
            self.assertEqual(len(summary["traces"]), 3)
            operators = {row["name"]: row for row in summary["operators"]}
            self.assertEqual(operators["aten::linear"]["calls"], 5)
            self.assertEqual(metadata["trace_count"], 3)
            for filename in summary["traces"]:
                with gzip.open(root / filename, "rt") as stream:
                    self.assertIn("traceEvents", json.load(stream))
        self.assertFalse(decoder._forward_hooks)
