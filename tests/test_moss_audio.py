"""Audio-to-note contracts without importing optional models or downloading weights."""

import json
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from test_workflow import write_audio

from scribnic import Note, open_workflow
from scribnic.backends.moss_audio import MossAudio, _final_note
from scribnic.config import load_workflow
from scribnic.downloads import PreparedStep, ResourceManager


class MossAudioTests(unittest.TestCase):
    """Check direct audio input, final-answer extraction and early rejection paths."""

    def test_unavailable_accelerator_fails_before_source_or_weights_are_loaded(
        self,
    ) -> None:
        """Explicit accelerator requests must not start an impossible model load."""
        for device in ("cuda", "mps"):
            with (
                self.subTest(device=device),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                step = load_workflow("moss_audio_note").steps[0]
                step = replace(
                    step,
                    runtime=replace(
                        step.runtime,
                        settings=step.runtime.settings | {"device": device},
                    ),
                )
                backend = MossAudio(step, PreparedStep(root, root, source_path=root))
                torch = ModuleType("torch")
                torch.device = lambda selected: SimpleNamespace(type=selected)
                torch.cuda = SimpleNamespace(is_available=lambda: False)
                torch.backends = SimpleNamespace(
                    mps=SimpleNamespace(is_available=lambda: False)
                )
                with (
                    patch.dict("sys.modules", {"torch": torch}),
                    patch(
                        "scribnic.backends.moss_audio.version", return_value="4.57.1"
                    ),
                    patch("scribnic.backends.moss_audio._provider") as provider,
                    patch.object(backend, "_memory_check") as memory_check,
                ):
                    with self.assertRaisesRegex(ValueError, "unavailable"):
                        backend.start()
                    provider.assert_not_called()
                    memory_check.assert_not_called()
                    backend.close()

    def test_audio_workflow_produces_note_without_transcription(self) -> None:
        """The existing executor can run one audio-to-note step and save raw output."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors.index.json").write_text("{}")
            prepared = PreparedStep(root, root, source_path=root)
            step = load_workflow("moss_audio_note").steps[0]
            for name in step.runtime.settings["required_source_files"]:
                source = root / name
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("# fake upstream code\n")
            backend = MossAudio(step, prepared)
            backend.model = SimpleNamespace(dtype="torch.float32")
            backend.device = SimpleNamespace(type="cpu")
            backend.torch = SimpleNamespace(__version__="fake")
            raw = "<think>private intermediate text</think>\nS — Tos.\nO — No consta."
            with (
                patch.object(backend, "start", return_value=0),
                patch.object(
                    backend, "_inference", return_value=(raw, 50, 20, {})
                ) as infer,
                patch("scribnic.backends.moss_audio.version", return_value="4.57.1"),
            ):
                with open_workflow(
                    "moss_audio_note", backend_factory=lambda step: backend
                ) as session:
                    result = session.run(write_audio(root), output_dir=root / "run")
                self.assertIsInstance(result.note, Note)
                self.assertIsNone(result.transcript)
                self.assertNotIn("private", result.note.text)
                self.assertEqual(len(result.steps), 1)
                self.assertEqual(infer.call_args.args[0].kind, "audio")
                raw_path = Path(result.steps[0].metadata["native_output_path"])
                self.assertEqual(raw_path.read_text(), raw)
                self.assertTrue((root / "run" / "report.html").is_file())

    def test_incomplete_thinking_is_not_a_note(self) -> None:
        """Empty answers or unterminated reasoning cannot become clinical output."""
        for text in ("", "<think>unfinished", "<think>done</think>"):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, "complete final note"):
                    _final_note(text)

    def test_context_overflow_fails_without_generation(self) -> None:
        """The adapter never silently clips audio to fit the language context."""
        step = load_workflow("moss_audio_note").steps[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = MossAudio(step, PreparedStep(root, root, source_path=root))
            backend.torch = SimpleNamespace(
                profiler=SimpleNamespace(record_function=lambda _: nullcontext())
            )
            generate = Mock()
            backend.model = SimpleNamespace(
                config=SimpleNamespace(
                    language_config=SimpleNamespace(max_position_embeddings=10)
                ),
                generate=generate,
            )
            inputs = {"input_ids": SimpleNamespace(shape=(1, 11))}
            with patch.object(backend, "_inputs", return_value=inputs):
                with self.assertRaisesRegex(ValueError, "nothing was truncated"):
                    backend._generate(write_audio(root))
            generate.assert_not_called()

    def test_weight_memory_guard_is_a_lower_bound_before_model_load(self) -> None:
        """Reject an actual insufficient-memory condition without allocating weights."""
        step = load_workflow("moss_audio_note").steps[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_parameters": 2**29}})
            )
            backend = MossAudio(step, PreparedStep(root, root, source_path=root))
            backend.device = SimpleNamespace(type="cpu")
            backend.torch = SimpleNamespace(
                empty=lambda *args, **kwargs: SimpleNamespace(element_size=lambda: 4)
            )
            with patch(
                "scribnic.backends.moss_audio.psutil.virtual_memory",
                return_value=SimpleNamespace(available=2**29),
            ):
                with self.assertRaisesRegex(ValueError, "weights need at least"):
                    backend._memory_check("float32")

    def test_wrong_extra_fails_before_downloading_assets(self) -> None:
        """The two incompatible Transformers versions do not mix implicitly."""
        step = load_workflow("moss_audio_note").steps[0]
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("importlib.metadata.version", return_value="5.20.2"),
                patch("scribnic.downloads.prepare_model") as prepare,
            ):
                with self.assertRaisesRegex(ImportError, "separate extra"):
                    ResourceManager(cache=Path(directory)).prepare(step)
                prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
