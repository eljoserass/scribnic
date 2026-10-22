"""NVIDIA artifact, managed runtime, lifecycle, and optional profiler contracts."""

import hashlib
import importlib.util
import json
import subprocess
import tarfile
import tempfile
import unittest
import wave
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scribnic import AudioFile, DiarizedTranscript, Transcript, open_workflow
from scribnic.backends.nemo import NemoSpeech, attribute_words
from scribnic.backends.nemo_cpp import NemoCPP
from scribnic.backends.nvidia_speech import parse_speech
from scribnic.config import RuntimeConfig, load_workflow
from scribnic.downloads import PreparedStep, prepare_binary
from scribnic.profiling import profile_directory
from scribnic.storage import artifact_directory


class NvidiaContractTests(unittest.TestCase):
    """Test actual interchange contracts without downloading weights or importing NeMo."""

    def test_presets_pin_both_models_and_validate_without_optional_imports(self):
        """All four experiment paths are discoverable through the ordinary API."""
        for runtime in ("cpp", "nemo"):
            for mode in ("offline", "streaming"):
                with open_workflow(
                    f"nvidia_{runtime}_{mode}", prepare=False
                ) as session:
                    step = session.config.steps[0]
                    self.assertEqual(step.model.settings["language"], "es-ES")
                    self.assertEqual(step.model.output, "diarized_transcript")
                    self.assertEqual(len(step.model.revision), 40)
                    self.assertEqual(len(step.companions["diarization"].sha256), 64)

    def test_missing_companions_and_unknown_modes_fail_before_preparation(self):
        """Configuration errors never trigger expensive model preparation."""
        config = load_workflow("nvidia_cpp_offline")
        step = config.steps[0]
        for invalid in (
            replace(step, companions={}),
            replace(step, model=replace(step.model, settings={"mode": "oops"})),
            replace(step, model=replace(step.model, settings={"max_speakers": True})),
        ):
            with self.assertRaises(ValueError):
                open_workflow(replace(config, steps=(invalid,)), prepare=False)

    def test_word_attribution_preserves_unknown_labels_and_turns(self):
        """Zero native tags stay unknown and speaker changes split adjacent words."""
        payload = {
            "text": "Hola doctor sí",
            "words": [
                {"word": "Hola", "start": 0, "end": 0.3, "speaker": 0},
                {"word": "doctor", "start": 0.4, "end": 0.6, "speaker": 1},
                {"word": "sí", "start": 0.7, "end": 0.9, "speaker": 2},
            ],
        }
        result = parse_speech(payload, duration_seconds=1, diarized=True)
        self.assertEqual([s.speaker for s in result.segments], [None, "1", "2"])
        plain = parse_speech(payload, duration_seconds=1, diarized=False)
        self.assertIsInstance(plain, Transcript)
        self.assertEqual(plain, result.without_speakers())
        payload["words"][0]["end"] = 4
        with self.assertRaises(ValueError):
            parse_speech(payload, duration_seconds=1, diarized=True)

    def test_seglst_overlap_and_zero_based_speakers_are_preserved(self):
        """Concurrent speakers remain separate even when their words overlap."""
        rows = [
            {"words": "Dos", "speaker": 1, "start_time": 0.2, "end_time": 0.8},
            {"words": "Uno", "speaker": 0, "start_time": 0.1, "end_time": 0.9},
        ]
        result = parse_speech(rows, duration_seconds=1, diarized=True)
        self.assertEqual([s.speaker for s in result.segments], ["0", "1"])
        self.assertLess(result.segments[1].start, result.segments[0].end)

    def test_offline_attribution_does_not_invent_speakers_for_ties_or_silence(self):
        """Only a unique positive overlap produces a speaker identity."""
        words = [{"word": "sí", "start": 1, "end": 2}]
        self.assertIsNone(attribute_words(words, ["0 1 A"])[0]["speaker"])
        self.assertIsNone(attribute_words(words, ["1 2 A", "1 2 B"])[0]["speaker"])
        self.assertEqual(
            attribute_words(words, ["1 2 speaker_0"])[0]["speaker"], "speaker_0"
        )

    def test_native_archive_preparation_verifies_and_installs_without_binary_path(self):
        """A fresh cache receives the pinned release; offline reuse needs no installer."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "release.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                info = tarfile.TarInfo("release/bin/nemo-speech")
                info.mode = 0o755
                info.size = 6
                bundle.addfile(info, BytesIO(b"native"))
            asset = {
                "url": archive.as_uri(),
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "binary": "release/bin/nemo-speech",
            }
            runtime = RuntimeConfig(
                "nemo_cpp", "nemo_cpp", {"release": "test"}, {"test-os": asset}
            )
            with patch("scribnic.downloads.platform_key", return_value="test-os"):
                binary = prepare_binary(runtime, root / "cache", True)
                self.assertEqual(binary.read_bytes(), b"native")
                self.assertEqual(prepare_binary(runtime, root / "cache", False), binary)


class NvidiaNativeLifecycleTests(unittest.TestCase):
    """Exercise CLI ownership, output persistence, and profile coverage boundaries."""

    def test_native_run_routes_streaming_and_keeps_diagnostics_inside_the_run(self):
        """An ordinary run gets raw output, native coverage, and an attributed artifact."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step = load_workflow("nvidia_cpp_streaming").steps[0]
            binary = root / "binary"
            binary.write_bytes(b"binary")
            model = root / "model"
            model.write_bytes(b"weights")
            prepared = PreparedStep(
                root, model, binary, companions={"diarization": model}
            )
            backend = NemoCPP(step, prepared)
            child = Mock(pid=456)
            child.poll.return_value = None

            def wait(**kwargs):
                """Simulate a child completing its file response while PID sampling is active."""
                self.assertEqual(backend.process_ids(), (456,))
                child.poll.return_value = 0
                return 0

            def popen(command, **kwargs):
                """Write the response the native CLI owns, using its real command arguments."""
                self.assertIn("--stream", command)
                self.assertIn(str(model), command)
                destination = Path(command[command.index("--output") + 1])
                destination.write_text(
                    json.dumps(
                        {
                            "text": "Hola",
                            "words": [
                                {"word": "Hola", "start": 0, "end": 0.5, "speaker": 1}
                            ],
                        }
                    )
                )
                child.wait.side_effect = wait
                return child

            with (
                artifact_directory(root / "run"),
                profile_directory(root / "run/profiles"),
                patch("scribnic.backends.nemo_cpp.subprocess.Popen", side_effect=popen),
            ):
                result = backend.run(AudioFile(root / "audio.wav", 16000, 1))
            self.assertIsInstance(result.artifact, DiarizedTranscript)
            self.assertTrue(
                Path(result.metadata["native_output_path"]).is_relative_to(root / "run")
            )
            coverage = json.loads(
                Path(result.metadata["profile"]["summary_path"]).read_text()
            )
            self.assertEqual(coverage["status"], "operator_capture_unavailable")
            self.assertFalse(backend.process_ids())

    def test_timeout_terminates_only_the_owned_process(self):
        """Timeouts leave no live native child behind."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step = load_workflow("nvidia_cpp_offline").steps[0]
            model = root / "model"
            model.write_bytes(b"weights")
            backend = NemoCPP(
                step,
                PreparedStep(root, model, model, companions={"diarization": model}),
            )
            child = Mock(pid=456)
            child.poll.return_value = None
            child.wait.side_effect = [subprocess.TimeoutExpired("native", 1), 0]
            with patch(
                "scribnic.backends.nemo_cpp.subprocess.Popen", return_value=child
            ):
                with self.assertRaises(subprocess.TimeoutExpired):
                    backend.run(AudioFile(root / "audio.wav", 16000, 1))
            child.terminate.assert_called_once()


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "optional Torch extra not installed"
)
class NvidiaPythonProfilingTests(unittest.TestCase):
    """Use tiny Torch models to verify the full Python adapter's reporting boundary."""

    def test_offline_profile_exports_traces_and_reuses_weights_without_state_leakage(
        self,
    ):
        """Each invocation profiles its own forwards and persists a separate raw result."""
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.write_bytes(b"weights")
            step = load_workflow("nvidia_nemo_offline").steps[0]
            backend = NemoSpeech(
                step, PreparedStep(root, model, companions={"diarization": model})
            )
            encoder = torch.nn.Linear(2, 2)
            with wave.open(str(root / "audio.wav"), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(b"\x00\x00" * 16000)

            def transcribe(*args, **kwargs):
                """Produce a real operator and one timestamped word."""
                self.assertEqual(kwargs["target_lang"], "es-ES")
                self.assertIsInstance(args[0][0], torch.Tensor)
                encoder(torch.ones(1, 2))
                return [
                    SimpleNamespace(
                        text="Hola",
                        timestamp={"word": [{"word": "Hola", "start": 0, "end": 0.5}]},
                    )
                ]

            backend.torch = torch
            backend.device = torch.device("cpu")
            backend.asr = SimpleNamespace(
                transcribe=transcribe,
                encoder=encoder,
                parameters=lambda: iter(encoder.parameters()),
            )
            backend.diarizer = SimpleNamespace(
                encoder=None, diarize=lambda *args, **kwargs: [["0 0.8 speaker_0"]]
            )
            previous_threads = torch.get_num_threads()
            with (
                patch.object(backend, "_load", return_value=0),
                patch(
                    "scribnic.backends.nemo.importlib.metadata.version",
                    return_value="test",
                ),
            ):
                for index in range(2):
                    directory = root / str(index)
                    with (
                        artifact_directory(directory),
                        profile_directory(directory / "profiles"),
                    ):
                        result = backend.run(AudioFile(root / "audio.wav", 16000, 1))
                    self.assertEqual(result.artifact.segments[0].speaker, "speaker_0")
                    summary = json.loads(
                        Path(result.metadata["profile"]["summary_path"]).read_text()
                    )
                    self.assertEqual(summary["status"], "success")
                    self.assertTrue(summary["traces"])
                    self.assertIn(
                        "nvidia.asr.encoder", {row["name"] for row in summary["phases"]}
                    )
            self.assertEqual(torch.get_num_threads(), previous_threads)
            self.assertFalse(encoder._forward_hooks)
            self.assertFalse(encoder._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
