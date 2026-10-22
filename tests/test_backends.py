"""Check model adapters at their HTTP, parsing, and resource boundaries."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from helpers import MODEL, NOTE, note_server

from scribnic import DiarizedTranscript, SpeakerSegment, Transcript, open_workflow
from scribnic.backends.http_generation import HTTPGeneration
from scribnic.backends.moss_transcribe_diarize import parse_transcript
from scribnic.config import load_workflow
from scribnic.downloads import PreparedStep, download_file


def http_step(url: str):
    config = load_workflow("bonsai_note")
    step = config.steps[0]
    runtime = replace(
        step.runtime,
        engine="http",
        settings={"base_url": url, "base_url_env": "TEST_SCRIBNIC_UNUSED"},
    )
    model = replace(step.model, settings=step.model.settings | {"served_model": MODEL})
    return replace(step, runtime=runtime, model=model)


class HTTPBackendTests(unittest.TestCase):
    def test_deferred_note_loading_checks_service_only_when_note_is_requested(self):
        """A deferred service consumes no resources while the speech step loads."""
        with note_server() as (url, _):
            step = http_step(url)
            step = replace(
                step,
                runtime=replace(
                    step.runtime, settings=step.runtime.settings | {"lazy_load": True}
                ),
            )
            with patch.object(HTTPGeneration, "_check_available") as check:
                backend = HTTPGeneration(step, PreparedStep(Path("unused")))
                self.assertIsNone(backend.client)
                check.assert_not_called()
                try:
                    backend.run(Transcript.from_text("Tengo tos."))
                    backend.run(Transcript.from_text("Tengo fiebre."))
                    check.assert_called_once()
                finally:
                    backend.close()

    def test_bonsai_sampling_profile_and_supplied_plain_text(self):
        with note_server() as (url, requests):
            backend = HTTPGeneration(http_step(url), PreparedStep(Path("unused")))
            try:
                result = backend.run(Transcript.from_text("Paciente: tos desde ayer."))
            finally:
                backend.close()
        self.assertEqual(result.artifact.text, NOTE)
        request = requests[0]
        self.assertEqual(request["temperature"], 0.7)
        self.assertEqual(request["top_p"], 0.8)
        self.assertEqual(request["top_k"], 20)
        self.assertEqual(request["presence_penalty"], 1.5)
        self.assertFalse(request["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(
            request["messages"][1]["content"],
            "TRANSCRIPCIÓN:\nPaciente: tos desde ayer.",
        )

    def test_model_identity_and_incomplete_notes_are_checked(self):
        with note_server(models=[]) as (url, _):
            with self.assertRaisesRegex(ValueError, "unavailable"):
                HTTPGeneration(http_step(url), PreparedStep(Path("unused")))
        with note_server(finish_reason="length") as (url, _):
            backend = HTTPGeneration(http_step(url), PreparedStep(Path("unused")))
            try:
                with self.assertRaisesRegex(ValueError, "output limit"):
                    backend.run(Transcript.from_text("hello"))
            finally:
                backend.close()

    def test_removing_speakers_changes_the_actual_model_input(self):
        transcript = DiarizedTranscript((SpeakerSegment("Tengo tos.", "S02"),))
        with note_server() as (url, requests):
            backend = HTTPGeneration(http_step(url), PreparedStep(Path("unused")))
            try:
                backend.run(transcript)
                backend.run(transcript.without_speakers())
            finally:
                backend.close()
        self.assertIn("[S02]", requests[0]["messages"][1]["content"])
        self.assertNotIn("S02", requests[1]["messages"][1]["content"])

    def test_long_inputs_fail_without_sending_a_generation_request(self):
        with note_server() as (url, requests):
            step = http_step(url)
            step = replace(
                step,
                model=replace(
                    step.model, settings=step.model.settings | {"max_input_chars": 1}
                ),
            )
            backend = HTTPGeneration(step, PreparedStep(Path("unused")))
            try:
                with self.assertRaisesRegex(ValueError, "Nothing was truncated"):
                    backend.run(Transcript.from_text("hello"))
            finally:
                backend.close()
        self.assertFalse(requests)


class PreparationTests(unittest.TestCase):
    def test_cached_downloads_verify_the_checksum_without_network(self):
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset"
            path.write_bytes(b"example")
            checksum = hashlib.sha256(b"example").hexdigest()
            self.assertEqual(
                download_file("unused", path, checksum, allowed=False), path
            )
            path.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "invalid cached asset"):
                download_file("unused", path, checksum, allowed=False)

    def test_bad_download_never_replaces_an_existing_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_bytes(b"bad")
            target = root / "target"
            target.write_bytes(b"existing")
            with self.assertRaisesRegex(ValueError, "checksum"):
                download_file(source.as_uri(), target, "0" * 64, allowed=True)
            self.assertEqual(target.read_bytes(), b"existing")
            self.assertEqual(len(list(root.iterdir())), 2)

    def test_presets_are_valid_without_preparing_external_runtimes(self):
        for name in (
            "moss_bonsai",
            "moss_bonsai_http",
            "whisper_bonsai",
            "whisper_bonsai_vad",
            "bonsai_note",
        ):
            with open_workflow(name, prepare=False):
                pass


class MossParserTests(unittest.TestCase):
    def test_native_speaker_labels_and_timestamps_are_preserved(self):
        result = parse_transcript("[0.48][S01]Hola[1.66][2.26][S02]Buenos días[3.81]")
        self.assertEqual(result.segments[0].speaker, "speaker_0")
        self.assertEqual(result.segments[1].end, 3.81)
        for bad in ("unstructured", "[2][S01]bad[1]"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_transcript(bad)


if __name__ == "__main__":
    unittest.main()
