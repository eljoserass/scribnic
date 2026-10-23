"""Whisper HTTP transport, VAD assets, timestamps and reusable workflow behavior."""

import os
import sys
import tempfile
import unittest
import wave
from contextlib import chdir
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from helpers import note_server, transcription_server

from scribnic import AudioFile, Transcript, open_workflow
from scribnic.backends.openai_chat import OpenAIChat
from scribnic.backends.openai_transcription import OpenAITranscription
from scribnic.backends.transcription_formats import parse_segments
from scribnic.config import load_workflow, validate_settings
from scribnic.downloads import (
    PreparedStep,
    prepare_binary,
    prepare_model,
    prepare_vad,
    sha256_file,
)
from scribnic.runtimes.endpoint import ExternalEndpoint

NATIVE = {
    "language": "Spanish",
    "task": "transcribe",
    "duration": 2,
    "text": "Tengo tos. ¿Desde cuándo?",
    "segments": [
        {"id": 0, "text": " Tengo tos. ", "start": 0.1, "end": 0.8},
        {"id": 1, "text": " ¿Desde cuándo?", "start": 0.9, "end": 1.5},
    ],
}


class FakeServer:
    """Stand in for a session-owned speech server without launching a model."""

    def __init__(self, url):
        self.url = url
        self.process = None
        self.starts = 0
        self.closed = False

    def start(self):
        self.starts += 1
        self.process = SimpleNamespace(pid=os.getpid(), poll=lambda: None)
        return self.url

    def metadata(self):
        return {"engine": "whisper_cpp", "pid": os.getpid()}

    def close(self):
        self.process = None
        self.closed = True

    def process_ids(self) -> tuple[int, ...]:
        """Expose the test process while the fake server is active."""
        return (self.process.pid,) if self.process else ()


def prepared_files(root: Path) -> PreparedStep:
    binary = root / "whisper-server"
    binary.write_bytes(b"fake runtime")
    model = root / "model.bin"
    model.write_bytes(b"fake weights")
    return PreparedStep(root, model, binary)


def write_audio(root: Path) -> AudioFile:
    audio = root / "sample.wav"
    with wave.open(str(audio), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 32000)
    return AudioFile(audio, duration=2)


class WhisperParserTests(unittest.TestCase):
    def test_native_times_are_preserved_without_inventing_speaker_labels(self):
        transcript = parse_segments(NATIVE, duration_seconds=2)
        self.assertIsInstance(transcript, Transcript)
        self.assertEqual(transcript.segments[0].start, 0.1)
        self.assertEqual(transcript.segments[-1].end, 1.5)
        self.assertEqual(transcript.segments[0].text, "Tengo tos.")
        self.assertFalse(hasattr(transcript.segments[0], "speaker"))

    def test_invalid_or_out_of_range_output_is_rejected(self):
        cases = (
            {},
            {"segments": [{"text": "missing times"}]},
            {"segments": [{"text": "bad", "start": 0.8, "end": 0.1}]},
            {"segments": [{"text": "bad", "start": 0, "end": 5}]},
        )
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse_segments(data, duration_seconds=2)

    def test_small_native_endpoint_overshoot_is_preserved(self):
        self.assertEqual(
            parse_segments(NATIVE, duration_seconds=1.391).segments[-1].end, 1.5
        )


class WhisperPreparationTests(unittest.TestCase):
    def test_explicit_relative_binary_is_not_replaced_by_a_path_search(self):
        runtime = load_workflow("whisper_bonsai").steps[0].runtime
        runtime = replace(runtime, settings={"binary": "./whisper-test.exe"})
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "whisper-test.exe"
            binary.write_bytes(b"fake executable")
            binary.chmod(0o755)
            with chdir(directory):
                self.assertEqual(prepare_binary(runtime, Path("unused"), False), binary)

    def test_binary_override_uses_an_existing_executable(self):
        runtime = load_workflow("whisper_bonsai").steps[0].runtime
        with patch.dict(os.environ, {"SCRIBNIC_WHISPER_CPP_BINARY": sys.executable}):
            self.assertEqual(
                prepare_binary(runtime, Path("unused"), False),
                Path(sys.executable).resolve(),
            )

    def test_pinned_whisper_weights_are_reusable_offline(self):
        step = load_workflow("whisper_bonsai").steps[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = (
                root
                / "models"
                / step.model.name
                / step.model.revision
                / step.model.filename
            )
            path.parent.mkdir(parents=True)
            path.write_bytes(b"cached weights")
            model = replace(step.model, sha256=sha256_file(path))
            self.assertEqual(prepare_model(model, step.runtime, root, False), path)

    def test_vad_download_is_optional_and_cached_assets_work_offline(self):
        step = load_workflow("whisper_bonsai").steps[0]
        with patch("scribnic.downloads.download_file") as download:
            self.assertIsNone(prepare_vad(step, Path("unused"), True))
        download.assert_not_called()
        step = load_workflow("whisper_bonsai_vad").steps[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = dict(step.model.settings["vad_model"])
            path = (
                root / "models" / asset["name"] / asset["revision"] / asset["filename"]
            )
            path.parent.mkdir(parents=True)
            path.write_bytes(b"vad weights")
            asset["sha256"] = sha256_file(path)
            model = replace(
                step.model, settings=step.model.settings | {"vad_model": asset}
            )
            self.assertEqual(prepare_vad(replace(step, model=model), root, False), path)

    def test_invalid_vad_units_and_thresholds_fail_before_preparation(self):
        step = load_workflow("whisper_bonsai").steps[0]
        for key, value in (
            ("vad", "true"),
            ("vad_threshold", 1.1),
            ("vad_min_speech_duration_ms", 0.5),
            ("vad_samples_overlap", -1),
            ("vad_speech_pad_ms", -1),
        ):
            model = replace(step.model, settings=step.model.settings | {key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_settings(replace(step, model=model))


class WhisperHTTPTests(unittest.TestCase):
    def test_http_error_closes_the_owned_runtime_and_is_not_retried(self):
        step = load_workflow("whisper_bonsai").steps[0]
        with (
            tempfile.TemporaryDirectory() as directory,
            transcription_server({"error": "failed"}, status=500) as (url, requests),
        ):
            root = Path(directory)
            server = FakeServer(url)
            with patch(
                "scribnic.backends.connection.create_runtime",
                side_effect=lambda step, prepared: (
                    server if step.name == "speech" else ExternalEndpoint(step.runtime)
                ),
            ):
                backend = OpenAITranscription(step, prepared_files(root))
                with self.assertRaisesRegex(ValueError, "transcription failed"):
                    backend.run(write_audio(root))
            self.assertTrue(server.closed)
            self.assertEqual(len(requests), 1)

    def test_audio_to_note_and_speech_reentry_reuse_one_server(self):
        config = load_workflow("whisper_bonsai_vad")
        models = [config.steps[1].model.settings["served_model"]]
        with (
            tempfile.TemporaryDirectory() as directory,
            transcription_server(NATIVE) as (speech_url, speech_requests),
            note_server(models=models) as (note_url, note_requests),
        ):
            root = Path(directory)
            prepared = prepared_files(root)
            server = FakeServer(speech_url)

            def factory(step):
                if step.name == "speech":
                    return OpenAITranscription(step, prepared)
                runtime = replace(
                    step.runtime,
                    engine="external_endpoint",
                    settings={"base_url": note_url},
                )
                return OpenAIChat(replace(step, runtime=runtime), prepared)

            with patch(
                "scribnic.backends.connection.create_runtime",
                side_effect=lambda step, prepared: (
                    server if step.name == "speech" else ExternalEndpoint(step.runtime)
                ),
            ):
                with open_workflow(config, backend_factory=factory) as session:
                    result = session.run(write_audio(root), output_dir=root / "full")
                    note_only = session.run(result.transcript)
                    early = session.run(write_audio(root), output="transcript")
            self.assertEqual([step.name for step in note_only.steps], ["note"])
            self.assertEqual([step.name for step in early.steps], ["speech"])
            self.assertEqual(server.starts, 1)
            self.assertTrue(server.closed)
            self.assertEqual(len(speech_requests), 2)
            self.assertEqual(speech_requests[0][0], "/v1/audio/transcriptions")
            fields = speech_requests[0][1]
            self.assertEqual(fields["vad"], "true")
            self.assertEqual(fields["vad_threshold"], "0.5")
            self.assertEqual(fields["token_timestamps"], "true")
            self.assertEqual(fields["vad_min_speech_duration_ms"], "250")
            self.assertEqual(
                note_requests[0]["messages"][1]["content"],
                "TRANSCRIPCIÓN:\nTengo tos.\n¿Desde cuándo?",
            )
            self.assertTrue((root / "full/transcript.json").is_file())
            self.assertEqual(early.steps[0].metadata["load_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
