"""Optional CPU profiler checks; no Torch import in the base test environment."""

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
            events = json.loads((root / "moss-torch-trace.json").read_text())[
                "traceEvents"
            ]
            names = {event["name"] for event in events}
            self.assertTrue({"moss.encoder", "moss.prefill", "moss.decode"} <= names)
        for module in (encoder, decoder):
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)
