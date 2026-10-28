"""Generate notes with llama-server's realtime audio protocol and incremental prefill."""

import base64
import json
import math
import os
import threading
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass
from time import monotonic, perf_counter
from typing import TYPE_CHECKING, TextIO
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from ..artifacts import Artifact, AudioFile, Note, StepOutput
from ..audio import BYTES_PER_SECOND, SAMPLE_RATE
from ..audio_stream import CHUNK_FRAMES, PCMRecording, wav_chunks
from ..config import StepConfig
from ..downloads import PreparedStep, model_provenance
from ..notes import note_instruction
from ..storage import diagnostic_path
from ..telemetry import tracer
from .connection import OpenAIConnection

if TYPE_CHECKING:
    from websockets.sync.client import ClientConnection


def _object(value: object, label: str) -> dict[str, object]:
    """Require a JSON object at a named protocol boundary."""
    if not isinstance(value, dict):
        raise ValueError(f"llama realtime returned invalid {label}.")
    return value


def _websocket_url(base_url: str) -> str:
    """Address llama-server's realtime endpoint using the existing API prefix."""
    address = urlsplit(base_url)
    scheme = "wss" if address.scheme == "https" else "ws"
    return urlunsplit(
        (scheme, address.netloc, address.path.rstrip("/") + "/realtime", "", "")
    )


def _check_cancelled(cancel: threading.Event | None) -> None:
    """Interrupt input or response waiting when the caller cancels this execution."""
    if cancel is not None and cancel.is_set():
        raise InterruptedError("Audio note generation was cancelled.")


def _event(
    socket: "ClientConnection", log: TextIO, *, timeout_seconds: float
) -> dict[str, object]:
    """Receive one event and persist it without retaining the event history."""
    event = _object(json.loads(socket.recv(timeout=timeout_seconds)), "event")
    log.write(json.dumps(event, ensure_ascii=False) + "\n")
    log.flush()
    if event.get("type") == "error":
        raise ValueError(f"llama realtime failed: {event.get('error')}")
    return event


def _wait_event(
    socket: "ClientConnection",
    log: TextIO,
    expected: str,
    *,
    timeout_seconds: float,
    cancel: threading.Event | None = None,
) -> dict[str, object]:
    """Wait for a handshake event with a deadline and responsive cancellation."""
    deadline = monotonic() + timeout_seconds
    while True:
        _check_cancelled(cancel)
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for llama realtime {expected}.")
        try:
            event = _event(socket, log, timeout_seconds=min(remaining, 0.25))
        except TimeoutError:
            continue
        if event.get("type") != expected:
            raise ValueError(f"Expected llama realtime {expected}, received {event}.")
        return event


def _final_note(event: dict[str, object], *, max_tokens: int) -> tuple[str, int]:
    """Validate final text, rejecting failures, output limits and reasoning leakage."""
    response = _object(event.get("response"), "response")
    if response.get("status") != "completed":
        raise ValueError(f"Audio note did not complete: {response.get('status')}.")
    usage = _object(response.get("usage"), "response usage")
    tokens = usage.get("output_tokens")
    if type(tokens) is not int or tokens < 0:
        raise ValueError("llama realtime returned invalid output token usage.")
    # This release reports 'completed' even when generation hits n_predict.
    if tokens >= max_tokens:
        raise ValueError("The note reached its output limit; incomplete note rejected.")
    output = response.get("output")
    if not isinstance(output, list):
        raise ValueError("llama realtime returned no note output.")
    parts = []
    for item in output:
        message = _object(item, "output message")
        content = message.get("content")
        if message.get("role") != "assistant" or not isinstance(content, list):
            raise ValueError("llama realtime returned an invalid note message.")
        for part in content:
            block = _object(part, "output content")
            text = block.get("text")
            if block.get("type") != "output_text" or not isinstance(text, str):
                raise ValueError("llama realtime did not return text-only output.")
            parts.append(text)
    note = "".join(parts).strip()
    if not note or "<think>" in note or "</think>" in note:
        raise ValueError("llama realtime returned no complete final note.")
    return note, tokens


@dataclass
class _Response:
    """Retain only this execution's completion and first-text timestamp."""

    done: dict[str, object] | None = None
    first_text_at: float | None = None

    def accept(self, event: dict[str, object]) -> None:
        """Track response boundaries without accumulating provisional text."""
        kind = event.get("type")
        if kind == "response.output_text.delta" and self.first_text_at is None:
            self.first_text_at = perf_counter()
        if kind == "response.output_audio.delta":
            raise ValueError(
                "llama realtime generated speech despite text-only output."
            )
        if kind == "response.done":
            self.done = event


class LlamaAudioNote:
    """Retain a model server per workflow session; own one fresh socket per run.

    WAV and live sources share incremental prefill. This adapter starts no sender
    workers: the caller's thread sends bounded blocks and polls incoming errors.
    Socket buffers apply transport backpressure; the server's decoder context
    and encoder backlog may grow up to the configured consultation limit.
    """

    def __init__(self, step: StepConfig, prepared: PreparedStep) -> None:
        """Keep verified asset provenance without loading optional model libraries."""
        self.step = step
        self.prepared = prepared
        self.connection = OpenAIConnection(step, prepared)
        self.model = str(step.model.settings.get("served_model", step.model.name))
        self.provenance = model_provenance(step, prepared)
        self.max_tokens = int(step.model.settings["max_tokens"])
        self.timeout_seconds = float(step.runtime.settings.get("timeout", 600))

    def start(self) -> float:
        """Load and verify the configured model once, retaining only owned resources."""
        elapsed = self.connection.start()
        if elapsed:
            try:
                self.connection.check_model(self.model)
            except ValueError:
                self.close()
                raise
        return elapsed

    def _check_duration(self, seconds: float) -> None:
        """Reject duration or estimated context overflow without clipping audio.

        The configured frontend rate estimates audio positions. UTF-8 bytes plus
        a template reserve conservatively budget the prompt; the server also runs
        with context shifting disabled. This is not a claim of long-audio quality.
        """
        settings = self.step.model.settings
        maximum = float(settings["max_audio_seconds"])
        if seconds > maximum:
            raise ValueError(
                f"Audio exceeds max_audio_seconds={maximum}; nothing was truncated."
            )
        positions = math.ceil(seconds * float(settings["audio_positions_per_second"]))
        system_prompt = str(settings["system_prompt"])
        prompt_budget = (
            len(system_prompt.encode("utf-8"))
            + len(note_instruction(self.step).encode("utf-8"))
            + 256
        )
        context = int(self.step.runtime.settings["context_size"])
        if positions + prompt_budget + self.max_tokens > context:
            raise ValueError(
                "Audio and note budget exceed context; nothing was truncated."
            )

    def _configure(
        self, socket: "ClientConnection", log: TextIO, cancel: threading.Event | None
    ) -> None:
        """Require compatible audio and configure the prompt before sending any input."""
        created = _wait_event(
            socket,
            log,
            "session.created",
            timeout_seconds=self.timeout_seconds,
            cancel=cancel,
        )
        session = _object(created.get("session"), "session")
        audio = _object(session.get("audio"), "session audio")
        audio_input = _object(audio.get("input"), "session audio input")
        audio_format = _object(audio_input.get("format"), "session audio format")
        if audio_format.get("rate") != SAMPLE_RATE:
            raise ValueError("llama realtime requires mono PCM16 audio at 16 kHz.")
        request = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": str(self.step.model.settings["system_prompt"]),
                "output_modalities": ["text"],
                "max_output_tokens": self.max_tokens,
            },
        }
        socket.send(json.dumps(request, ensure_ascii=False))
        _wait_event(
            socket,
            log,
            "session.updated",
            timeout_seconds=self.timeout_seconds,
            cancel=cancel,
        )

    def _send_audio(
        self,
        chunks: Iterable[bytes],
        socket: "ClientConnection",
        log: TextIO,
        *,
        cancel: threading.Event | None,
    ) -> float:
        """Send bounded audio blocks and check errors between them; return audio seconds."""
        received = 0
        for chunk in chunks:
            _check_cancelled(cancel)
            if not isinstance(chunk, bytes) or len(chunk) % 2:
                raise ValueError("Audio chunks must contain complete PCM16 frames.")
            for offset in range(0, len(chunk), CHUNK_FRAMES * 2):
                _check_cancelled(cancel)
                block = chunk[offset : offset + CHUNK_FRAMES * 2]
                received += len(block)
                self._check_duration(received / BYTES_PER_SECOND)
                socket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(block).decode("ascii"),
                        }
                    )
                )
                try:
                    event = _event(socket, log, timeout_seconds=0)
                except TimeoutError:
                    continue
                if event.get("type", "").startswith("response."):
                    raise ValueError("llama realtime responded before audio ended.")
        if not received:
            raise ValueError("Cannot generate a note from empty audio.")
        return received / BYTES_PER_SECOND

    def _complete(
        self, socket: "ClientConnection", log: TextIO, cancel: threading.Event | None
    ) -> tuple[_Response, float]:
        """Commit only at EOF and await a complete response with a separate deadline."""
        _check_cancelled(cancel)
        socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        _wait_event(
            socket,
            log,
            "input_audio_buffer.committed",
            timeout_seconds=self.timeout_seconds,
            cancel=cancel,
        )
        # An audio-only user turn defaults toward ASR. Ask the actual task in the
        # same user turn, matching the model's audio-understanding examples.
        socket.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": note_instruction(self.step)}
                        ],
                    },
                },
                ensure_ascii=False,
            )
        )
        requested_at = perf_counter()
        socket.send(json.dumps({"type": "response.create"}))
        response = _Response()
        deadline = monotonic() + self.timeout_seconds
        while response.done is None:
            _check_cancelled(cancel)
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for the final audio note.")
            try:
                event = _event(socket, log, timeout_seconds=min(remaining, 0.25))
            except TimeoutError:
                continue
            response.accept(event)
        return response, requested_at

    def _generate(
        self, chunks: Iterable[bytes], *, cancel: threading.Event | None = None
    ) -> StepOutput:
        """Own a fresh bounded socket and diagnostic log for one consultation.

        Keep connection setup, inference and result timing in one resource scope
        so every failure closes the socket before returning or validating a note.
        """
        try:
            from websockets.exceptions import WebSocketException
            from websockets.sync.client import connect
        except ImportError as error:
            raise ImportError("LFM3 audio notes need scribnic[streaming].") from error

        _check_cancelled(cancel)
        self._check_duration(0)
        load_seconds = self.start()
        base_url = str(self.connection.client.base_url)
        key_name = str(
            self.step.runtime.settings.get("api_key_env", "SCRIBNIC_API_KEY")
        )
        headers = {"Authorization": "Bearer " + (os.getenv(key_name) or "local")}
        path = diagnostic_path(
            "llama-realtime-" + uuid4().hex + ".jsonl", cache=self.prepared.cache
        )
        started = perf_counter()
        try:
            with (
                tracer().start_as_current_span("scribnic.note.request"),
                path.open("w", encoding="utf-8") as log,
                connect(
                    _websocket_url(base_url),
                    additional_headers=headers,
                    open_timeout=self.timeout_seconds,
                    close_timeout=2,
                    compression=None,
                    max_size=1024 * 1024,
                    max_queue=4,
                ) as socket,
            ):
                self._configure(socket, log, cancel)
                seconds = self._send_audio(chunks, socket, log, cancel=cancel)
                response, requested_at = self._complete(socket, log, cancel)
        except WebSocketException as error:
            raise ValueError(f"llama realtime connection failed: {error}") from error
        text, tokens = _final_note(response.done, max_tokens=self.max_tokens)
        first_text_seconds = None
        if response.first_text_at is not None:
            first_text_seconds = response.first_text_at - requested_at
        metadata = {
            "model": self.model,
            "base_url": base_url,
            "protocol": "llama_cpp_realtime",
            "incremental_prefill": True,
            "audio_seconds": seconds,
            "generated_tokens": tokens,
            "load_seconds": load_seconds,
            "request_seconds": perf_counter() - started,
            "first_text_after_response_request_seconds": first_text_seconds,
            "native_events_path": str(path),
            "provenance": self.provenance,
            "runtime": self.connection.runtime.metadata(),
            "server_metrics": self.connection.runtime.metrics(),
        }
        return StepOutput(Note(text), metadata)

    def run(self, source: Artifact) -> StepOutput:
        """Read a validated WAV incrementally and generate one final note."""
        if not isinstance(source, AudioFile) or source.duration is None:
            raise ValueError("llama_audio_note needs a validated audio artifact.")
        self._check_duration(source.duration)
        with closing(wav_chunks(source.path)) as chunks:
            return self._generate(chunks)

    def run_stream(self, source: PCMRecording) -> StepOutput:
        """Prefill arriving audio and return a note without transcript updates."""
        return self._generate(source, cancel=source.cancel)

    def process_ids(self) -> tuple[int, ...]:
        """Expose only the model server processes owned by this workflow session."""
        return self.connection.runtime.process_ids()

    def close(self) -> None:
        """Release the client and owned server; individual runs close their own sockets."""
        self.connection.close()
