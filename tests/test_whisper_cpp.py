"""Whisper.cpp artifact conversion, process ownership, and workflow integration."""

import json
import os
import sys
import tempfile
import unittest
import wave
from contextlib import chdir
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from helpers import note_server

from scribnic import AudioFile, Transcript, open_workflow
from scribnic.backends.http_generation import HTTPGeneration
from scribnic.backends.whisper_cpp import WhisperCPP, parse_transcript
from scribnic.config import load_workflow
from scribnic.downloads import PreparedStep, prepare_binary, prepare_model, sha256_file

NATIVE = {
    "result": {"language": "es"},
    "transcription": [
        {"text": " Tengo tos. ", "offsets": {"from": 100, "to": 800}},
        {"text": " ¿Desde cuándo?", "offsets": {"from": 900, "to": 1500}},
    ],
}


class FakeProcess:
    def __init__(self, command, *, cwd, stdout, stderr):
        self.pid = os.getpid()
        self.returncode = None
        self.terminated = False
        prefix = command[command.index("--output-file") + 1]
        Path(prefix + ".json").write_text(json.dumps(NATIVE), encoding="utf-8")
        stdout.write("whisper_print_timings: load time = 10.00 ms\n")

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15


def prepared_files(root: Path) -> PreparedStep:
    binary = root / "whisper-cli"
    binary.write_bytes(b"fake runtime")
    model = root / "model.bin"
    model.write_bytes(b"fake weights")
    return PreparedStep(root, model, binary)


class WhisperParserTests(unittest.TestCase):
    def test_native_times_are_converted_without_inventing_speaker_labels(self):
        transcript = parse_transcript(NATIVE, 2)
        self.assertIsInstance(transcript, Transcript)
        self.assertEqual(transcript.segments[0].start, 0.1)
        self.assertEqual(transcript.segments[1].end, 1.5)
        self.assertEqual(transcript.segments[0].text, "Tengo tos.")
        self.assertFalse(hasattr(transcript.segments[0], "speaker"))

    def test_invalid_or_out_of_range_output_is_rejected(self):
        cases = (
            {},
            {"transcription": [{"text": "no timestamps"}]},
            {"transcription": [{"text": "bad", "offsets": {"from": 800, "to": 100}}]},
            {"transcription": [{"text": "bad", "offsets": {"from": 0, "to": 5000}}]},
        )
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse_transcript(data, 2)

    def test_small_native_endpoint_overshoot_is_preserved(self):
        transcript = parse_transcript(NATIVE, 1.391)
        self.assertEqual(transcript.segments[-1].end, 1.5)


class WhisperPreparationTests(unittest.TestCase):
    def test_explicit_relative_binary_path_is_not_replaced_by_a_path_search(self):
        runtime = load_workflow("whisper_bonsai").steps[0].runtime
        runtime = replace(runtime, settings={"binary": "./whisper-test.exe"})
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "whisper-test.exe"
            binary.write_bytes(b"fake executable")
            binary.chmod(0o755)
            with chdir(directory):
                self.assertEqual(prepare_binary(runtime, Path("unused"), False), binary)

    def test_binary_override_uses_an_existing_executable_without_downloading(self):
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


class WhisperProcessTests(unittest.TestCase):
    def test_timeout_cleans_up_the_owned_process(self):
        step = load_workflow("whisper_bonsai").steps[0]
        with tempfile.TemporaryDirectory() as directory:
            backend = WhisperCPP(step, prepared_files(Path(directory)))
            processes = []

            def start(*args, **kwargs):
                process = FakeProcess(*args, **kwargs)
                processes.append(process)
                return process

            with patch("scribnic.backends.whisper_cpp.subprocess.Popen", start):
                with patch.object(
                    backend, "_wait", side_effect=TimeoutError("timeout")
                ):
                    with self.assertRaises(TimeoutError):
                        backend.run(AudioFile(Path("sample.wav"), duration=2))
                self.assertTrue(processes[0].terminated)
                self.assertEqual(backend.process_ids(), ())

    def test_nonzero_exit_reports_failure_instead_of_using_partial_json(self):
        step = load_workflow("whisper_bonsai").steps[0]
        with tempfile.TemporaryDirectory() as directory:
            backend = WhisperCPP(step, prepared_files(Path(directory)))

            def fail(*args, **kwargs):
                process = FakeProcess(*args, **kwargs)
                process.returncode = 3
                return process

            with patch("scribnic.backends.whisper_cpp.subprocess.Popen", fail):
                with self.assertRaisesRegex(RuntimeError, "code 3"):
                    backend.run(AudioFile(Path("sample.wav"), duration=2))
            self.assertEqual(backend.process_ids(), ())


class WhisperWorkflowTests(unittest.TestCase):
    def test_audio_to_plain_transcript_to_note_and_transcript_reentry(self):
        config = load_workflow("whisper_bonsai")
        models = [config.steps[1].model.settings["served_model"]]
        with (
            tempfile.TemporaryDirectory() as directory,
            note_server(models=models) as (url, requests),
        ):
            root = Path(directory)
            prepared = prepared_files(root)
            audio = root / "sample.wav"
            with wave.open(str(audio), "wb") as stream:
                stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                stream.writeframes(b"\0\0" * 32000)

            def factory(step):
                if step.name == "speech":
                    return WhisperCPP(step, prepared)
                runtime = replace(
                    step.runtime, engine="http", settings={"base_url": url}
                )
                return HTTPGeneration(replace(step, runtime=runtime), prepared)

            with patch("scribnic.backends.whisper_cpp.subprocess.Popen", FakeProcess):
                with open_workflow(config, backend_factory=factory) as session:
                    result = session.run(AudioFile(audio), output_dir=root / "full")
                    note_only = session.run(result.transcript)
                    early = session.run(AudioFile(audio), output="transcript")
            self.assertEqual([step.name for step in result.steps], ["speech", "note"])
            self.assertEqual([step.name for step in note_only.steps], ["note"])
            self.assertEqual([step.name for step in early.steps], ["speech"])
            self.assertTrue(result.note.text)
            self.assertIsInstance(result.transcript, Transcript)
            self.assertEqual(
                requests[0]["messages"][1]["content"],
                "TRANSCRIPCIÓN:\nTengo tos.\n¿Desde cuándo?",
            )
            self.assertTrue((root / "full/transcript.json").is_file())
            self.assertFalse((root / "full/diarized_transcript.json").exists())
            self.assertAlmostEqual(
                result.steps[0].metadata["native_timings"]["load_seconds"], 0.01
            )


if __name__ == "__main__":
    unittest.main()
