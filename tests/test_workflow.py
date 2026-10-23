"""Exercise public workflow routing with tiny, deterministic backend fakes."""

import subprocess
import sys
import tempfile
import unittest
import wave
from dataclasses import replace
from pathlib import Path

from scribnic import (
    AudioFile,
    DiarizedTranscript,
    Note,
    SpeakerSegment,
    Transcript,
    WorkflowRunError,
    open_workflow,
)
from scribnic.artifacts import StepOutput
from scribnic.config import (
    ModelConfig,
    RuntimeConfig,
    StepConfig,
    WorkflowConfig,
    discover_workflow,
)


def configuration() -> WorkflowConfig:
    runtime = RuntimeConfig("fake", "fake")
    speech = ModelConfig("speech", ("audio",), "diarized_transcript")
    note = ModelConfig("note", ("transcript", "diarized_transcript"), "note")
    return WorkflowConfig(
        "example",
        (
            StepConfig("speech", speech, runtime, "fake"),
            StepConfig("note", note, runtime, "fake"),
        ),
    )


def write_audio(directory: Path, name: str = "sample.wav") -> AudioFile:
    path = directory / name
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * 16000)
    return AudioFile(path)


class FakeBackend:
    def __init__(self, step: StepConfig, calls: list, *, fail: bool = False):
        self.step = step
        self.calls = calls
        self.fail = fail
        self.closed = False

    def start(self) -> float:
        """Represent an already available fake model without producing artifacts."""
        return 0.0

    def run(self, source):
        self.calls.append((self.step.name, source))
        if self.fail:
            raise ValueError("Note server unavailable")
        if isinstance(source, AudioFile):
            transcript = DiarizedTranscript(
                (SpeakerSegment(source.path.name, "S01", 0, 1),)
            )
            return StepOutput(transcript)
        return StepOutput(Note(source.text))

    def close(self):
        self.closed = True

    def process_ids(self):
        return ()


class WorkflowTests(unittest.TestCase):
    """Verify reusable sessions independently of speech engines and interfaces."""

    def setUp(self):
        self.calls = []
        self.constructed = []

    def factory(self, step):
        backend = FakeBackend(step, self.calls)
        self.constructed.append(backend)
        return backend

    def test_audio_defaults_to_note_and_keeps_intermediate_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_audio(Path(directory))
            with open_workflow(
                configuration(), backend_factory=self.factory
            ) as session:
                result = session.run(source)
        self.assertEqual(result.note.text, "sample.wav")
        self.assertIsInstance(result.transcript, DiarizedTranscript)
        self.assertEqual([name for name, _ in self.calls], ["speech", "note"])
        self.assertTrue(all(backend.closed for backend in self.constructed))

    def test_explicit_start_loads_selected_steps_without_producing_artifacts(
        self,
    ) -> None:
        """Evaluation can separate startup from requests and bypass speech for text."""
        with open_workflow(configuration(), backend_factory=self.factory) as session:
            startup = session.start(source_kind="transcript")
            self.assertEqual(set(startup["step_load_seconds"]), {"note"})
            self.assertFalse(self.calls)
            result = session.run(Transcript.from_text("consulta independiente"))
        self.assertEqual(len(self.constructed), 1)
        self.assertEqual(result.note.text, "consulta independiente")
        self.assertIsNone(
            result.transcript if result.get("diarized_transcript") else None
        )

    def test_stop_at_speech_then_supply_modified_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            with open_workflow(
                configuration(), backend_factory=self.factory
            ) as session:
                speech = session.run(
                    write_audio(Path(directory)), output="diarized_transcript"
                )
                self.assertEqual(len(self.constructed), 1)
                plain = speech.transcript.without_speakers()
                note = session.run(plain)
        self.assertEqual([name for name, _ in self.calls], ["speech", "note"])
        self.assertIs(self.calls[-1][1], plain)
        self.assertEqual(note.note.text, speech.transcript.text)
        self.assertEqual(speech.transcript.segments[0].speaker, "S01")
        self.assertEqual(plain.segments[0].start, 0)

    def test_arbitrary_text_needs_only_note_backend(self):
        with open_workflow(configuration(), backend_factory=self.factory) as session:
            result = session.run(
                Transcript.from_text("texto desordenado sin etiquetas")
            )
        self.assertEqual(result.note.text, "texto desordenado sin etiquetas")
        self.assertEqual([backend.step.name for backend in self.constructed], ["note"])

    def test_runs_reuse_models_without_reusing_artifacts(self):
        with open_workflow(configuration(), backend_factory=self.factory) as session:
            first = session.run(Transcript.from_text("first"))
            second = session.run(Transcript.from_text("second"))
        self.assertEqual(len(self.constructed), 1)
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(first.note.text, "first")
        self.assertEqual(second.note.text, "second")
        self.assertIsNone(second.get("diarized_transcript"))

    def test_unsupported_output_fails_before_models_or_audio_loading(self):
        with open_workflow(configuration(), backend_factory=self.factory) as session:
            with self.assertRaisesRegex(ValueError, "does not produce 'transcript'"):
                session.run(AudioFile("missing.wav"), output="transcript")
        self.assertFalse(self.constructed)

    def test_generation_error_retains_completed_speech(self):
        def factory(step):
            return FakeBackend(step, self.calls, fail=step.name == "note")

        with tempfile.TemporaryDirectory() as directory:
            with open_workflow(configuration(), backend_factory=factory) as session:
                with self.assertRaises(WorkflowRunError) as failed:
                    session.run(write_audio(Path(directory)))
        self.assertIsInstance(failed.exception.result.transcript, DiarizedTranscript)
        self.assertIsNone(failed.exception.result.note)
        self.assertEqual(failed.exception.result.status, "error")

    def test_closed_session_cannot_run(self):
        session = open_workflow(configuration(), backend_factory=self.factory)
        session.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.run(Transcript.from_text("hello"))

    def test_missing_producer_is_a_configuration_error(self):
        config = configuration()
        config = replace(config, steps=(config.steps[1],), inputs=("audio",))
        with self.assertRaisesRegex(ValueError, "No configured producer"):
            open_workflow(config, backend_factory=self.factory)
        self.assertFalse(self.constructed)

    def test_ambiguous_input_needs_explicit_source(self):
        config = configuration()
        other = replace(config.steps[0], name="other_speech")
        ambiguous = replace(config, steps=(*config.steps, other))
        with self.assertRaisesRegex(ValueError, "Ambiguous input"):
            open_workflow(ambiguous, backend_factory=self.factory)
        bound_note = replace(config.steps[1], source="speech")
        resolved = replace(ambiguous, steps=(config.steps[0], bound_note, other))
        with open_workflow(resolved, backend_factory=self.factory):
            self.assertFalse(self.constructed)

    def test_cycles_are_rejected_before_models(self):
        config = configuration()
        looping = replace(config.steps[0].model, inputs=("note",))
        config = replace(
            config, steps=(replace(config.steps[0], model=looping), config.steps[1])
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            open_workflow(config, backend_factory=self.factory)
        self.assertFalse(self.constructed)

    def test_importing_the_public_api_does_not_import_torch(self):
        code = "import scribnic, sys; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True)


class DiscoveryTests(unittest.TestCase):
    def test_directory_discovery_requires_one_file_or_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "one.toml"
            first.write_text('name = "one"\n')
            self.assertEqual(discover_workflow(root), first)
            (root / "two.toml").write_text('name = "two"\n')
            with self.assertRaisesRegex(ValueError, "Select a workflow"):
                discover_workflow(root)
            default = root / "default.toml"
            default.write_text('name = "default"\n')
            self.assertEqual(discover_workflow(root), default)


if __name__ == "__main__":
    unittest.main()
