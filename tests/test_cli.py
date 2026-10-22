"""The CLI delegates to the same application and workflow APIs as Python callers."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from test_workflow import FakeBackend

from scribnic import open_workflow
from scribnic.cli import main


class CLITests(unittest.TestCase):
    def test_json_stdout_is_one_document_and_the_run_has_local_telemetry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "conversation.txt"
            source.write_text("Paciente: tengo tos.", encoding="utf-8")
            output = root / "run"
            stdout, stderr = io.StringIO(), io.StringIO()

            def test_session(config, **kwargs):
                return open_workflow(
                    config, backend_factory=lambda step: FakeBackend(step, [])
                )

            with (
                redirect_stdout(stdout),
                redirect_stderr(stderr),
                patch("scribnic.application.open_workflow", test_session),
            ):
                main(
                    [
                        "run",
                        str(source),
                        "--workflow",
                        "bonsai_note",
                        "--output-dir",
                        str(output),
                        "--format",
                        "json",
                        "--offline",
                    ]
                )
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["requested_output"], "note")
            self.assertEqual(len(result["steps"]), 1)
            self.assertTrue((output / "note.txt").is_file())
            self.assertTrue((output / "report.html").is_file())
            self.assertTrue((output / "telemetry/traces.jsonl").is_file())
            self.assertIn("note", stderr.getvalue())

    def test_prepare_never_invokes_inference(self):
        with patch("scribnic.cli.prepare_workflow") as prepare:
            with redirect_stdout(io.StringIO()):
                main(["prepare", "--workflow", "bonsai_note", "--offline"])
        prepare.assert_called_once_with("bonsai_note", offline=True)

    def test_existing_results_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.txt"
            source.write_text("Hello")
            existing = root / "run"
            existing.mkdir()
            (existing / "note.txt").write_text("Keep me")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(
                    [
                        "run",
                        str(source),
                        "--workflow",
                        "bonsai_note",
                        "--output-dir",
                        str(existing),
                    ]
                )
            self.assertEqual((existing / "note.txt").read_text(), "Keep me")

    def test_bad_wav_is_rejected_before_resource_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.wav"
            source.write_bytes(b"invalid")
            with patch("scribnic.application.open_workflow") as open_session:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(["run", str(source), "--workflow", "moss_bonsai"])
            open_session.assert_not_called()
