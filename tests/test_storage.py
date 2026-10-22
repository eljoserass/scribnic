"""Persistence, recovery, and real OpenTelemetry export contracts."""

import json
import logging
import tempfile
import unittest
from pathlib import Path

from opentelemetry import trace
from test_workflow import FakeBackend, configuration, write_audio

from scribnic import (
    DiarizedTranscript,
    SpeakerSegment,
    Transcript,
    WorkflowRunError,
    open_workflow,
)
from scribnic.observability import Telemetry
from scribnic.reporting import render_workflow_report
from scribnic.storage import artifact_from_dict, artifact_to_dict, load_artifact


class StorageTests(unittest.TestCase):
    def test_operator_summary_is_rendered_and_operator_names_are_escaped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with open_workflow(
                configuration(), backend_factory=lambda step: FakeBackend(step, [])
            ) as session:
                result = session.run(Transcript.from_text("Example"))
            profiles = root / "profiles"
            profiles.mkdir()
            row = {
                "name": "<operator>",
                "calls": 2,
                "cpu_seconds": 1.0,
                "self_cpu_seconds": 0.5,
                "self_cpu_memory_net_bytes": -1024,
            }
            (profiles / "moss-profile.json").write_text(
                json.dumps(
                    {
                        "operators": [row],
                        "phases": [row],
                        "traces": ["example.json.gz"],
                    }
                )
            )
            report = render_workflow_report(result, root)
        self.assertIn("Profiling de MOSS", report)
        self.assertIn("&lt;operator&gt;", report)
        self.assertNotIn("<operator>", report)
        self.assertIn("100.0%", report)

    def test_roundtrip_preserves_unknown_times_and_independent_roles(self):
        original = DiarizedTranscript((SpeakerSegment("Hola", "S02", role="doctor"),))
        restored = artifact_from_dict(
            json.loads(json.dumps(artifact_to_dict(original)))
        )
        self.assertEqual(original, restored)
        self.assertIsNone(restored.segments[0].start)
        self.assertEqual(restored.without_speakers(), Transcript.from_text("Hola"))
        self.assertEqual(original.segments[0].speaker, "S02")

    def test_openai_and_legacy_transcripts_can_be_reentered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "speech.json"
            turn = {"text": "Hola", "speaker": "A", "start": 0.2, "end": 1.3}
            for key in ("segments", "utterances"):
                path.write_text(json.dumps({key: [turn]}))
                restored = load_artifact(path)
                self.assertEqual(restored.segments[0].speaker, "A")
                self.assertEqual(restored.segments[0].start, 0.2)
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                load_artifact(path)

    def test_failure_keeps_speech_and_an_error_report(self):
        def factory(step):
            return FakeBackend(step, [], fail=step.name == "note")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_audio(root)
            output = root / "result"
            with open_workflow(configuration(), backend_factory=factory) as session:
                with self.assertRaises(WorkflowRunError) as failure:
                    session.run(source, output_dir=output)
            speech = load_artifact(output / "diarized_transcript.json")
            self.assertEqual(speech, failure.exception.result.transcript)
            saved = json.loads((output / "result.json").read_text())
            self.assertEqual(saved["status"], "error")
            self.assertIn("Note server unavailable", saved["error"])
            self.assertTrue((output / "report.html").is_file())
            self.assertFalse((output / "note.txt").exists())

    def test_html_escapes_supplied_text_and_never_invents_timestamps(self):
        source = Transcript.from_text("<script>alert('x')</script>")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            with open_workflow(
                configuration(), backend_factory=lambda step: FakeBackend(step, [])
            ) as session:
                session.run(source, output_dir=output)
            report = (output / "report.html").read_text()
        self.assertNotIn("<script>", report)
        self.assertIn("&lt;script&gt;", report)
        self.assertNotIn("00:00 —", report)


class TelemetryTests(unittest.TestCase):
    def test_otlp_files_correlate_the_run_without_replacing_the_host_sdk(self):
        original_provider = trace.get_tracer_provider()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with Telemetry(output):
                with open_workflow(
                    configuration(), backend_factory=lambda step: FakeBackend(step, [])
                ) as session:
                    result = session.run(Transcript.from_text("Example"))
                logging.getLogger("scribnic").warning("Test diagnostic")
            records = [
                json.loads(line)
                for line in (output / "traces.jsonl").read_text().splitlines()
            ]
            spans = [
                span
                for record in records
                for resource in record["resourceSpans"]
                for scope in resource["scopeSpans"]
                for span in scope["spans"]
            ]
            root = next(
                span for span in spans if span["name"] == "scribnic.workflow.run"
            )
            self.assertEqual(root["traceId"].lower(), result.trace_id)
            self.assertTrue(all(span["traceId"] == root["traceId"] for span in spans))
            metrics = json.loads((output / "metrics.jsonl").read_text().splitlines()[0])
            self.assertIn("resourceMetrics", metrics)
            logs = (output / "logs.jsonl").read_text()
            self.assertIn("Test diagnostic", logs)
            self.assertTrue(result.resources["processes"])
        self.assertIs(trace.get_tracer_provider(), original_provider)
