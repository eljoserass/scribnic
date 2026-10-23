"""Verify CLI presentation and scoped notifications without live inference services."""

import hashlib
import io
import json
import logging
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from test_workflow import FakeBackend, configuration, write_audio

from scribnic import open_workflow
from scribnic.cli import main
from scribnic.downloads import download_file
from scribnic.observability import Telemetry, tracer
from scribnic.progress import current_observer, observe_progress, phase
from scribnic.terminal import TerminalView, present_result


class _TTY(io.StringIO):
    """Emulate an interactive terminal without owning any operating-system handles."""

    def isatty(self) -> bool:
        """Request the Rich terminal path for rendering tests."""
        return True


class TerminalTests(unittest.TestCase):
    """Exercise terminal cleanup, literal content and stable redirected output."""

    def test_progress_has_plain_boundaries_without_ansi_or_updates(self) -> None:
        """Keep redirected logs small while preserving phase and elapsed information."""
        stream = io.StringIO()
        with TerminalView(stream):
            with phase("Descargar", detail="model[Q8].gguf") as task:
                task.update(123)
                task.update(456)
        text = stream.getvalue()
        self.assertIn("… Descargar · model[Q8].gguf", text)
        self.assertIn("✓ Descargar · model[Q8].gguf", text)
        self.assertEqual(len(text.splitlines()), 2)
        self.assertNotIn("\x1b", text)
        self.assertIsNone(current_observer())

    def test_rich_terminal_draws_color_and_restores_cursor_on_error(self) -> None:
        """Close the live display and restore observer state when execution fails."""
        stream = _TTY()
        with patch.dict(os.environ, {"TERM": "xterm-256color"}):
            with self.assertRaises(ValueError):
                with TerminalView(stream):
                    with phase("Cargar", detail="model[Q8]"):
                        raise ValueError("broken")
        text = stream.getvalue()
        self.assertIn("\x1b[", text)
        self.assertIn("model[Q8]", text)
        self.assertIn("✗", text)
        self.assertIn("\x1b[?25h", text)
        self.assertIsNone(current_observer())

    def test_no_color_has_no_foreground_color_sequences(self) -> None:
        """Respect NO_COLOR on interactive terminals while retaining progress updates."""
        stream = _TTY()
        with patch.dict(os.environ, {"TERM": "xterm", "NO_COLOR": "1"}):
            with TerminalView(stream) as view:
                self.assertTrue(view.console.no_color)
                with phase("Validar entrada"):
                    pass
        text = stream.getvalue()
        self.assertNotIn("[36m", text)
        self.assertNotIn("[32m", text)

    def test_pretty_result_preserves_speaker_labels_and_note_brackets(self) -> None:
        """Never interpret model output as Rich markup or remove existing timestamps."""
        with tempfile.TemporaryDirectory() as directory:
            with open_workflow(
                configuration(), backend_factory=lambda step: FakeBackend(step, [])
            ) as session:
                result = session.run(write_audio(Path(directory)))
        stream = _TTY()
        present_result(result, stream)
        self.assertIn("[S01]", stream.getvalue())
        self.assertIn("00:00", stream.getvalue())

    def test_warning_logs_visible_and_information_requires_verbose(self) -> None:
        """Hide routine diagnostics without hiding actionable warning messages."""
        stream = io.StringIO()
        logger = logging.getLogger("scribnic.ui-test")
        with TerminalView(stream):
            logger.warning("Check model [Q8]")
            logger.info("Quiet detail")
        self.assertIn("Check model [Q8]", stream.getvalue())
        self.assertNotIn("Quiet detail", stream.getvalue())

    def test_nested_observers_are_restored_after_interrupt(self) -> None:
        """Do not leak presentation into later API executions or sibling callers."""
        events = []
        with observe_progress(events.append):
            with self.assertRaises(KeyboardInterrupt):
                with TerminalView(io.StringIO()):
                    with phase("Inferencia"):
                        raise KeyboardInterrupt
            with phase("Outer"):
                pass
        self.assertEqual([event.label for event in events], ["Outer", "Outer"])
        self.assertIsNone(current_observer())

    def test_execution_span_reports_interruption_instead_of_success(self) -> None:
        """Reflect cancellation even when OTel leaves the span status unset."""
        events = []
        with tempfile.TemporaryDirectory() as directory:
            with observe_progress(events.append), Telemetry(Path(directory)):
                with self.assertRaises(KeyboardInterrupt):
                    with tracer().start_as_current_span("scribnic.step.run"):
                        raise KeyboardInterrupt
        self.assertEqual([event.state for event in events], ["start", "interrupted"])

    def test_service_phase_identifies_its_parent_steps_model(self) -> None:
        """Distinguish recognition and generation service loads in the live display."""
        events = []
        with tempfile.TemporaryDirectory() as directory:
            with observe_progress(events.append), Telemetry(Path(directory)):
                with tracer().start_as_current_span(
                    "scribnic.step.run",
                    attributes={"scribnic.model.name": "Bonsai-PQ2_0"},
                ):
                    with tracer().start_as_current_span("scribnic.service.load"):
                        pass
        loads = [event for event in events if event.label == "Cargar servicio"]
        self.assertEqual([event.detail for event in loads], ["Bonsai-PQ2_0"] * 2)


class CLICompatibilityTests(unittest.TestCase):
    """Keep public commands, structured stdout and execution semantics compatible."""

    def test_help_and_invalid_choices_return_expected_exit_codes(self) -> None:
        """Retain argparse's successful help and usage-error exit code two."""
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            main(["--help"])
            with self.assertRaises(SystemExit) as caught:
                main(["run", "audio.wav", "--workflow", "x", "--format", "bad"])
        self.assertEqual(caught.exception.code, 2)

    def test_cancel_closes_ui_and_preserves_130(self) -> None:
        """Report cancellation without a traceback or a leaked observer."""
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            patch("scribnic.cli.execute", side_effect=KeyboardInterrupt),
            self.assertRaises(SystemExit) as caught,
        ):
            main(["run", "audio.wav", "--workflow", "bonsai_note"])
        self.assertEqual(caught.exception.code, 130)
        self.assertIn("interrumpida", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")
        self.assertIsNone(current_observer())

    def test_json_early_stop_runs_only_speech_and_names_actual_phases(self) -> None:
        """Honor the core's selected output, without showing a fictitious note phase."""
        calls = []

        def test_session(config: object, **kwargs: object) -> object:
            """Reuse the real planner with fake backends, without preparing assets."""
            return open_workflow(
                configuration(), backend_factory=lambda step: FakeBackend(step, calls)
            )

        with tempfile.TemporaryDirectory() as directory:
            source = write_audio(Path(directory)).path
            stdout, stderr = io.StringIO(), io.StringIO()
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
                        "moss_bonsai",
                        "--output",
                        "diarized_transcript",
                        "--format",
                        "json",
                        "--output-dir",
                        str(Path(directory) / "run"),
                    ]
                )
        data = json.loads(stdout.getvalue())
        self.assertEqual(data["requested_output"], "diarized_transcript")
        self.assertEqual(len(calls), 1)
        self.assertIn("Transcribir y diarizar", stderr.getvalue())
        self.assertNotIn("Generar nota", stderr.getvalue())
        self.assertNotIn("\x1b", stdout.getvalue())


class DownloadProgressTests(unittest.TestCase):
    """Observe actual byte counts while preserving verified atomic downloads."""

    def test_transfer_reports_bytes_and_cached_second_call_does_not_download(
        self,
    ) -> None:
        """Use a local file URL to validate byte progress without contacting a service."""
        payload = b"model contents" * 100
        checksum = hashlib.sha256(payload).hexdigest()
        events = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source", root / "cache/model.bin"
            source.write_bytes(payload)
            with observe_progress(events.append):
                download_file(source.as_uri(), target, checksum, allowed=True)
                download_file("https://unused.invalid", target, checksum, allowed=False)
            self.assertEqual(target.read_bytes(), payload)
        transferred = [event for event in events if event.unit == "bytes"]
        self.assertEqual(transferred[-1].state, "done")
        self.assertEqual(transferred[-1].completed, len(payload))
        self.assertEqual(transferred[-1].total, len(payload))
        self.assertIn("Verificar caché", [event.label for event in events])

    def test_unknown_length_uses_no_total_and_checksum_error_removes_temporary(
        self,
    ) -> None:
        """Keep indeterminate transfer progress honest and never cache corrupt data."""
        response = io.BytesIO(b"wrong model")
        response.headers = {}
        events = []
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model.bin"
            with (
                observe_progress(events.append),
                patch(
                    "scribnic.downloads.urllib.request.urlopen", return_value=response
                ),
                self.assertRaises(ValueError),
            ):
                download_file("https://unused.invalid", target, "bad", allowed=True)
            self.assertEqual(list(target.parent.iterdir()), [])
        self.assertTrue(all(event.total is None for event in events))
        self.assertEqual(events[-1].state, "error")
