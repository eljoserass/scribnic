"""Check actual adapter progress boundaries without weights or inference services."""

import importlib.util
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from helpers import note_server
from test_backends import http_step
from test_whisper_cpp import FakeServer, prepared_files, write_audio

from scribnic import Transcript
from scribnic.backends.moss_transcribe_diarize import MossTranscribeDiarize
from scribnic.backends.openai_chat import OpenAIChat
from scribnic.backends.openai_transcription import OpenAITranscription
from scribnic.config import load_workflow
from scribnic.observability import Telemetry
from scribnic.progress import ProgressEvent, observe_progress


def _load_fake_moss(backend: MossTranscribeDiarize) -> float:
    """Allocate harmless stand-ins for resources used after optional model loading."""
    backend.model = SimpleNamespace(dtype="float32")
    backend.device = SimpleNamespace(type="cpu")
    backend.torch = SimpleNamespace(__version__="test")
    return 0.01


class BackendProgressTests(unittest.TestCase):
    """Cover provider phases, explicit startup and warm requests through real adapters."""

    def assert_phases(
        self, events: list[ProgressEvent], *, load: str, action: str, model: str
    ) -> None:
        """Require one initial load and two independent inference notifications."""
        self.assertEqual(
            [(event.label, event.state) for event in events],
            [
                (load, "start"),
                (load, "done"),
                (action, "start"),
                (action, "done"),
                (action, "start"),
                (action, "done"),
            ],
        )
        self.assertTrue(all(event.detail == model for event in events))

    def test_all_speech_presets_report_loading_before_inference(self) -> None:
        """Exercise Whisper, NeMo HTTP/streaming and external MOSS without real requests."""
        presets = (
            "whisper_bonsai",
            "whisper_bonsai_vad",
            "nvidia_cpp_offline",
            "nvidia_cpp_streaming",
            "nvidia_cpp_asr_offline",
            "nvidia_cpp_asr_streaming",
            "moss_bonsai_sglang",
        )
        payload = {
            "text": "Hola",
            "segments": [{"text": "Hola", "start": 0, "end": 1}],
            "words": [{"word": "Hola", "start": 0, "end": 1, "speaker": 1}],
        }
        for preset in presets:
            with (
                self.subTest(preset=preset),
                tempfile.TemporaryDirectory() as temporary,
            ):
                step = load_workflow(preset).steps[0]
                self.assertEqual(step.backend, "openai_transcription")
                if step.model.settings.get("mode") == "streaming":
                    if importlib.util.find_spec("websockets") is None:
                        self.skipTest("streaming extra not installed")
                root = Path(temporary)
                prepared = prepared_files(root)
                prepared = replace(
                    prepared,
                    companions={role: prepared.model_path for role in step.companions},
                )
                service = FakeServer("http://127.0.0.1:1234/v1")
                events: list[ProgressEvent] = []
                with patch(
                    "scribnic.backends.connection.create_runtime", return_value=service
                ):
                    backend = OpenAITranscription(step, prepared)
                with (
                    patch.object(backend, "_request", return_value=payload),
                    patch.object(backend.connection, "check_model"),
                    observe_progress(events.append),
                    Telemetry(root / "telemetry"),
                ):
                    try:
                        backend.run(write_audio(root))
                        second = backend.run(write_audio(root))
                    finally:
                        backend.close()
                action = "Transcribir y diarizar" if backend.diarized else "Transcribir"
                self.assert_phases(
                    events, load="Cargar servicio", action=action, model=step.model.name
                )
                self.assertEqual(second.metadata["load_seconds"], 0)
                self.assertEqual(service.starts, 1)

    def test_chat_reports_generation_after_loading_and_reuses_the_service(self) -> None:
        """Measure the real chat adapter against a tiny local HTTP service fake."""
        with (
            tempfile.TemporaryDirectory() as temporary,
            note_server() as (url, _),
        ):
            root = Path(temporary)
            step = http_step(url)
            backend = OpenAIChat(step, prepared_files(root))
            events: list[ProgressEvent] = []
            with observe_progress(events.append), Telemetry(root / "telemetry"):
                try:
                    backend.run(Transcript.from_text("Tengo tos."))
                    second = backend.run(Transcript.from_text("Desde ayer."))
                finally:
                    backend.close()
            self.assert_phases(
                events,
                load="Cargar servicio",
                action="Generar nota",
                model=step.model.name,
            )
            self.assertEqual(second.metadata["load_seconds"], 0)

    def test_moss_reports_cold_and_explicit_loading_without_reloading(self) -> None:
        """Use fake tensors to verify local MOSS phases without importing Torch."""
        for preset, preload in (("moss_bonsai", False), ("moss_bonsai_http", True)):
            with (
                self.subTest(preset=preset, preload=preload),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                step = load_workflow(preset).steps[0]
                backend = MossTranscribeDiarize(step, prepared_files(root))
                events: list[ProgressEvent] = []
                with (
                    patch.object(
                        backend, "_load", side_effect=lambda: _load_fake_moss(backend)
                    ) as load,
                    patch.object(backend, "_cpu_threads", return_value=nullcontext(2)),
                    patch.object(backend, "_describe_runtime"),
                    patch.object(
                        backend,
                        "_generate",
                        return_value=("[0][S01]Hola[1]", 1, 1, {}),
                    ),
                    patch(
                        "scribnic.backends.moss_transcribe_diarize.torch_profile",
                        return_value=nullcontext({}),
                    ),
                    patch(
                        "scribnic.backends.moss_transcribe_diarize.version",
                        return_value="test",
                    ),
                    observe_progress(events.append),
                    Telemetry(root / "telemetry"),
                ):
                    try:
                        if preload:
                            backend.start()
                        backend.run(write_audio(root))
                        second = backend.run(write_audio(root))
                    finally:
                        backend.close()
                load.assert_called_once()
                self.assert_phases(
                    events,
                    load="Cargar MOSS",
                    action="Transcribir y diarizar",
                    model=step.model.name,
                )
                self.assertEqual(second.metadata["load_seconds"], 0)
