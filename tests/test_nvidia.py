"""Speech service contracts without downloading models or importing Torch."""

import hashlib
import tarfile
import tempfile
import unittest
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from helpers import transcription_server
from test_whisper_cpp import FakeServer, prepared_files, write_audio

from scribnic import Transcript, open_workflow
from scribnic.backends.openai_transcription import OpenAITranscription
from scribnic.backends.transcription_formats import parse_segments, parse_words
from scribnic.config import RuntimeConfig, load_workflow
from scribnic.downloads import ResourceManager, prepare_binary
from scribnic.runtimes import NemoServer
from scribnic.storage import artifact_directory


class NvidiaContractTests(unittest.TestCase):
    """Exercise artifact capabilities, pinned weight variants and response schemas."""

    def test_presets_pin_both_models_without_optional_imports(self) -> None:
        """Native presets need no NeMo/PyTorch installation."""
        for mode in ("offline", "streaming"):
            with open_workflow(f"nvidia_cpp_{mode}", prepare=False) as session:
                step = session.config.steps[0]
                self.assertEqual(step.backend, "openai_transcription")
                self.assertEqual(step.model.settings["language"], "es-ES")
                self.assertIn("q8_0", step.model.name)
                self.assertEqual(step.model.quantization, "Q8_0")
                self.assertEqual(len(step.companions["diarization"].sha256), 64)

    def test_incompatible_contracts_fail_before_preparation(self) -> None:
        """Missing companions, unsupported modes and formats fail before downloads."""
        config = load_workflow("nvidia_cpp_offline")
        step = config.steps[0]
        for invalid in (
            replace(step, companions={}),
            replace(
                step,
                model=replace(
                    step.model, settings=step.model.settings | {"mode": "oops"}
                ),
            ),
            replace(step, model=replace(step.model, filename="model.nemo")),
        ):
            with self.assertRaises(ValueError):
                open_workflow(replace(config, steps=(invalid,)), prepare=False)

    def test_word_attribution_preserves_unknown_labels(self) -> None:
        """Native zero is unknown; labels never become clinical roles."""
        payload = {
            "text": "Hola doctor sí",
            "words": [
                {"word": "Hola", "start": 0, "end": 0.3, "speaker": 0},
                {"word": "doctor", "start": 0.4, "end": 0.6, "speaker": 1},
                {"word": "sí", "start": 0.7, "end": 0.9, "speaker": 2},
            ],
        }
        result = parse_words(payload, duration_seconds=1, diarized=True)
        self.assertEqual([s.speaker for s in result.segments], [None, "1", "2"])
        plain = parse_words(payload, duration_seconds=1, diarized=False)
        self.assertIsInstance(plain, Transcript)
        self.assertEqual(plain, result.without_speakers())

    def test_moss_segments_preserve_labels_and_unknown_values(self) -> None:
        """Interpret the service schema without importing a MOSS model library."""
        payload = {
            "segments": [
                {"start": 0, "end": 1, "text": "[S01] Hola"},
                {"start": 0.5, "end": 1.2, "text": "Sí"},
            ]
        }
        result = parse_segments(payload, duration_seconds=2, diarized=True)
        self.assertEqual([s.speaker for s in result.segments], ["S01", None])
        self.assertEqual(result.segments[0].text, "Hola")

    def test_external_moss_uses_shared_http_without_local_model_assets(self) -> None:
        """An existing service requires no local weights, Torch or SGLang install."""
        step = load_workflow("moss_bonsai_sglang").steps[0]
        model = step.model.settings["served_model"]
        payload = {
            "text": "Hola",
            "segments": [{"start": 0, "end": 0.5, "text": "[S01] Hola"}],
        }
        with (
            tempfile.TemporaryDirectory() as temporary,
            transcription_server(payload, models=(model,)) as (url, requests),
        ):
            root = Path(temporary)
            runtime = replace(step.runtime, settings={"base_url": url})
            step = replace(step, runtime=runtime)
            with patch("scribnic.downloads.download_file") as download:
                prepared = ResourceManager(cache=root, download=False).prepare(step)
                download.assert_not_called()
            backend = OpenAITranscription(step, prepared)
            try:
                result = backend.run(write_audio(root))
                self.assertEqual(backend.process_ids(), ())
            finally:
                backend.close()
            self.assertEqual(result.artifact.segments[0].speaker, "S01")
            self.assertEqual(requests[0][1]["model"], model)
            self.assertEqual(requests[0][1]["max_new_tokens"], "16384")
            self.assertNotIn("beam_size", requests[0][1])

    def test_pinned_archive_is_installed_and_reusable_offline(self) -> None:
        """A fresh cache receives the pinned archive without a manual binary path."""
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


class NvidiaServiceTests(unittest.TestCase):
    """Exercise persistent service commands and the shared HTTP client."""

    def test_server_loads_both_models_without_a_recording_argument(self) -> None:
        """The service owns weights; recordings belong to subsequent requests."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            step = load_workflow("nvidia_cpp_streaming").steps[0]
            prepared = prepared_files(root)
            prepared = replace(
                prepared, companions={"diarization": prepared.model_path}
            )
            server = NemoServer(step.runtime, prepared)
            command = server._command(1234)
            self.assertEqual(command[1], "serve")
            self.assertIn("--asr-model", command)
            self.assertIn("--diar-model", command)
            self.assertNotIn("transcribe", command)
            self.assertIn("--asr.batching.enabled=false", command)
            self.assertEqual(server.health_path, "/ready")

    def test_requests_reuse_service_and_preserve_diagnostics_and_strategy(self) -> None:
        """Separate requests retain their predictions and use one loaded service."""
        step = load_workflow("nvidia_cpp_offline").steps[0]
        payload = {
            "text": "Hola",
            "words": [{"word": "Hola", "start": 0, "end": 0.5, "speaker": 1}],
        }
        with (
            tempfile.TemporaryDirectory() as temporary,
            transcription_server(payload) as (url, requests),
        ):
            root = Path(temporary)
            service = FakeServer(url)
            with patch(
                "scribnic.backends.connection.create_runtime", return_value=service
            ):
                prepared = prepared_files(root)
                prepared = replace(
                    prepared, companions={"diarization": prepared.model_path}
                )
                backend = OpenAITranscription(step, prepared)
                try:
                    with artifact_directory(root / "run"):
                        first = backend.run(write_audio(root))
                        second = backend.run(write_audio(root))
                finally:
                    backend.close()
            self.assertEqual(service.starts, 1)
            self.assertEqual(second.metadata["load_seconds"], 0)
            self.assertEqual(
                first.metadata["speaker_attribution_strategy"], "word_time_alignment"
            )
            self.assertEqual(requests[0][1]["diarization"], "true")
            self.assertNotEqual(
                first.metadata["native_output_path"],
                second.metadata["native_output_path"],
            )
            self.assertTrue(
                Path(first.metadata["native_output_path"]).is_relative_to(root / "run")
            )


if __name__ == "__main__":
    unittest.main()
