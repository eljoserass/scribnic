"""Exercise direct audio notes and realtime lifecycle without optional packages."""

import base64
import json
import sys
import tempfile
import threading
import unittest
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from test_workflow import write_audio

from scribnic import WorkflowRunError, open_workflow
from scribnic.audio import inspect_audio
from scribnic.backends.llama_audio_note import LlamaAudioNote
from scribnic.config import load_workflow
from scribnic.downloads import PreparedStep, ResourceManager
from scribnic.runtimes.llama_cpp import LlamaServer, server_arguments


class SocketFailure(OSError):
    """Represent a broken WebSocket independently of the optional dependency."""


class FakeSocket:
    """Emulate the pinned server's session, audio commit and completion protocol."""

    def __init__(self, *, tokens: int = 30, status: str = "completed") -> None:
        """Keep one consultation's events and received frames until it closes."""
        self.events = deque(
            [
                {
                    "type": "session.created",
                    "session": {
                        "audio": {"input": {"format": {"rate": 16000}}},
                    },
                },
            ]
        )
        self.sent: list[dict[str, object]] = []
        self.audio: list[bytes] = []
        self.closed = False
        self.tokens = tokens
        self.status = status
        self.fail_on_audio = False
        self.cancel_on_response: threading.Event | None = None
        self.disconnect_on_response = False

    def __enter__(self) -> "FakeSocket":
        """Open this single-use fake consultation."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the consultation even after source or server failures."""
        self.closed = True

    def send(self, message: str) -> None:
        """Schedule acknowledgements only after their corresponding requests."""
        event = json.loads(message)
        self.sent.append(event)
        kind = event["type"]
        if kind == "session.update":
            self.events.append({"type": "session.updated"})
        elif kind == "input_audio_buffer.append":
            self.audio.append(base64.b64decode(event["audio"]))
            if self.fail_on_audio:
                self.events.append(
                    {"type": "error", "error": {"message": "encoder failed"}}
                )
        elif kind == "input_audio_buffer.commit":
            self.events.append({"type": "input_audio_buffer.committed"})
            self.events.append({"type": "conversation.item.done"})
        elif kind == "response.create":
            self.events.extend(
                [
                    {"type": "response.output_text.delta", "delta": "S — Tos."},
                    {
                        "type": "response.done",
                        "response": {
                            "status": self.status,
                            "usage": {"output_tokens": self.tokens},
                            "output": [
                                {
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": f"S — Tos. Bloques: {len(self.audio)}.",
                                        },
                                    ],
                                }
                            ],
                        },
                    },
                ]
            )
            if self.cancel_on_response is not None:
                self.cancel_on_response.set()

    def recv(self, *, timeout: float) -> str:
        """Return available events; absence while recording is ordinary silence."""
        if self.disconnect_on_response and any(
            event["type"] == "response.create" for event in self.sent
        ):
            raise SocketFailure("Disconnected before the final note")
        if not self.events:
            raise TimeoutError
        return json.dumps(self.events.popleft())


@contextmanager
def sockets(*instances: FakeSocket) -> Iterator[Mock]:
    """Inject socket fakes so ordinary tests never require websockets or a server."""
    exceptions = ModuleType("websockets.exceptions")
    exceptions.WebSocketException = SocketFailure
    client = ModuleType("websockets.sync.client")
    connect = Mock(side_effect=instances)
    client.connect = connect
    with patch.dict(
        sys.modules,
        {
            "websockets": ModuleType("websockets"),
            "websockets.sync": ModuleType("websockets.sync"),
            "websockets.exceptions": exceptions,
            "websockets.sync.client": client,
        },
    ):
        yield connect


def backend(root: Path) -> LlamaAudioNote:
    """Configure real adapter code around fake local assets and a resident service."""
    model = root / "model.gguf"
    projector = root / "mmproj.gguf"
    binary = root / "llama-server"
    for path in (model, projector, binary):
        path.write_bytes(b"fake asset")
    prepared = PreparedStep(root, model, binary, companions={"mmproj": projector})
    adapter = LlamaAudioNote(load_workflow("lfm3_audio_note").steps[0], prepared)
    adapter.connection = SimpleNamespace(
        start=lambda: 0.0,
        close=lambda: None,
        client=SimpleNamespace(base_url="http://127.0.0.1:1234/v1/"),
        runtime=SimpleNamespace(
            metadata=lambda: {},
            metrics=lambda: None,
            process_ids=lambda: (),
        ),
    )
    return adapter


class AudioNoteTests(unittest.TestCase):
    """Cover public artifact boundaries, configuration and failure ownership."""

    def test_file_and_live_audio_share_protocol_and_isolate_consultations(self) -> None:
        """One resident backend produces isolated notes without inventing transcripts."""
        first, second = FakeSocket(), FakeSocket()
        updates = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = backend(root)
            with sockets(first, second) as connect:
                with open_workflow(
                    "lfm3_audio_note",
                    backend_factory=lambda step: adapter,
                ) as session:
                    file_result = session.run(
                        write_audio(root), output_dir=root / "file"
                    )
                    live_result = session.run_stream(
                        [b"\x01\x00" * 3200],
                        output_dir=root / "live",
                        on_update=updates.append,
                    )
            self.assertEqual(connect.call_count, 2)
            self.assertEqual(file_result.note.text, "S — Tos. Bloques: 5.")
            self.assertEqual(live_result.note.text, "S — Tos. Bloques: 1.")
            self.assertIsNone(live_result.transcript)
            self.assertEqual([record.name for record in live_result.steps], ["note"])
            self.assertEqual(live_result.source.duration, 0.2)
            self.assertTrue(live_result.source.sha256)
            self.assertTrue((root / "live/report.html").is_file())
            metadata = live_result.steps[0].metadata
            self.assertEqual(metadata["provenance"]["mmproj"]["quantization"], "Q8_0")
            events = Path(metadata["native_events_path"]).read_text()
            self.assertIn("response.done", events)
            self.assertNotIn("input_audio_buffer.append", events)
        self.assertTrue(first.closed and second.closed)
        self.assertFalse(updates)
        kinds = [event["type"] for event in first.sent]
        self.assertEqual(
            kinds[-3:],
            [
                "input_audio_buffer.commit",
                "conversation.item.create",
                "response.create",
            ],
        )
        self.assertEqual(kinds.count("input_audio_buffer.commit"), 1)
        request = first.sent[0]["session"]
        self.assertEqual(request["output_modalities"], ["text"])
        self.assertEqual(
            request["instructions"], adapter.step.model.settings["system_prompt"]
        )
        question = first.sent[-2]["item"]["content"][0]["text"]
        self.assertIn("SOAP", question)

    def test_audio_is_sent_before_input_ends_and_source_failure_closes_socket(
        self,
    ) -> None:
        """Producer exceptions preserve the received recording and never commit a note."""
        socket = FakeSocket()

        def chunks() -> Iterator[bytes]:
            """Assert eager consumption and fail after one recorded fragment."""
            yield b"\0\0" * 3200
            self.assertEqual(len(socket.audio), 1)
            raise OSError("Microphone failed")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = backend(root)
            with (
                sockets(socket),
                open_workflow(
                    "lfm3_audio_note",
                    backend_factory=lambda step: adapter,
                ) as session,
            ):
                with self.assertRaisesRegex(
                    WorkflowRunError, "Microphone failed"
                ) as error:
                    session.run_stream(chunks(), output_dir=root / "failed")
            self.assertEqual(error.exception.result.source.duration, 0.2)
            self.assertIsNone(error.exception.result.note)
        self.assertTrue(socket.closed)
        self.assertNotIn(
            "input_audio_buffer.commit", [event["type"] for event in socket.sent]
        )

    def test_encoder_error_stops_input_and_preserves_its_cause(self) -> None:
        """Native server errors surface during ingestion instead of yielding partial notes."""
        socket = FakeSocket()
        socket.fail_on_audio = True
        with tempfile.TemporaryDirectory() as directory, sockets(socket):
            adapter = backend(Path(directory))
            with self.assertRaisesRegex(ValueError, "encoder failed"):
                adapter._generate([b"\0\0" * 3200] * 5)
        self.assertEqual(len(socket.audio), 1)
        self.assertTrue(socket.closed)

    def test_cancel_during_generation_closes_socket_and_returns_no_note(self) -> None:
        """Cancellation remains distinct from a successful end of input."""
        cancel = threading.Event()
        socket = FakeSocket()
        socket.cancel_on_response = cancel
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = backend(root)
            with (
                sockets(socket),
                open_workflow(
                    "lfm3_audio_note",
                    backend_factory=lambda step: adapter,
                ) as session,
            ):
                with self.assertRaises(WorkflowRunError) as error:
                    session.run_stream([b"\0\0" * 3200], cancel=cancel)
            self.assertEqual(error.exception.result.status, "interrupted")
            self.assertIsNone(error.exception.result.note)
        self.assertTrue(socket.closed)

    def test_disconnect_or_output_limit_never_accepts_an_incomplete_note(self) -> None:
        """The pinned server's completed status alone does not establish a full answer."""
        for failure in ("disconnect", "limit", "cancelled"):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as directory,
            ):
                socket = FakeSocket(
                    tokens=2048 if failure == "limit" else 30,
                    status="cancelled" if failure == "cancelled" else "completed",
                )
                socket.disconnect_on_response = failure == "disconnect"
                with sockets(socket):
                    adapter = backend(Path(directory))
                    with self.assertRaises(ValueError):
                        adapter._generate([b"\0\0" * 3200])
                self.assertTrue(socket.closed)

    def test_duration_and_context_limits_fail_before_model_start(self) -> None:
        """Oversized WAV requests cannot load a model or silently shorten the input."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = backend(root)
            audio = inspect_audio(write_audio(root))
            for limit in ("duration", "context"):
                with self.subTest(limit=limit), patch.object(adapter, "start") as start:
                    source = (
                        replace(audio, duration=1201) if limit == "duration" else audio
                    )
                    if limit == "context":
                        adapter.step = replace(
                            adapter.step,
                            runtime=replace(
                                adapter.step.runtime,
                                settings=adapter.step.runtime.settings
                                | {"context_size": 2000},
                            ),
                        )
                    with self.assertRaisesRegex(ValueError, "nothing was truncated"):
                        adapter.run(source)
                    start.assert_not_called()

    def test_invalid_workflow_fails_before_preparing_assets(self) -> None:
        """Missing audio capabilities and unsafe context policies fail before downloads."""
        config = load_workflow("lfm3_audio_note")
        step = config.steps[0]
        cases = [
            replace(step, companions={}),
            replace(step, model=replace(step.model, inputs=("transcript",))),
            replace(
                step,
                runtime=replace(
                    step.runtime,
                    settings=step.runtime.settings
                    | {
                        "server_options": {"context-shift": True},
                    },
                ),
            ),
        ]
        for invalid in cases:
            with (
                self.subTest(step=invalid),
                patch.object(ResourceManager, "prepare") as prepare,
            ):
                with self.assertRaises(ValueError):
                    open_workflow(replace(config, steps=(invalid,)))
                prepare.assert_not_called()


class ServerArgumentsTests(unittest.TestCase):
    """Check configurable llama-server flags and managed resource paths."""

    def test_scalar_options_and_companion_are_passed_as_literal_argv(self) -> None:
        """Spaces in values remain one argument and flags use explicit boolean forms."""
        options = {
            "jinja": True,
            "context-shift": False,
            "temp": 0.0,
            "prompt": "a b; $(literal)",
        }
        self.assertEqual(
            server_arguments(options),
            [
                "--jinja",
                "--no-context-shift",
                "--temp",
                "0.0",
                "--prompt",
                "a b; $(literal)",
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            adapter = backend(Path(directory))
            server = LlamaServer(adapter.step.runtime, adapter.prepared, "lfm3")
            command = server._command(1234)
            self.assertEqual(
                command[command.index("--mmproj") + 1],
                str(adapter.prepared.companions["mmproj"]),
            )
            self.assertIn("--no-context-shift", command)

    def test_invalid_options_and_managed_overrides_fail_early(self) -> None:
        """Malformed option tables cannot override process ownership or context settings."""
        for options in (
            [],
            {"--temp": 0},
            {"temp": []},
            {"temp": float("inf")},
            {"host": "example.com"},
            {"no-ctx-size": True},
            {"mmproj": "other.gguf"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                server_arguments(options)
