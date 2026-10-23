"""Replay WAV PCM through NeMo-Speech.cpp's specific streaming WebSocket API."""

import json
import threading
import wave
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

from ..storage import write_json

if TYPE_CHECKING:
    from websockets.sync.client import ClientConnection


def _send_audio(socket: "ClientConnection", path: Path) -> None:
    """Send bounded PCM chunks as fast as accepted, then flush the recording."""
    from websockets.exceptions import ConnectionClosed

    try:
        with wave.open(str(path), "rb") as audio:
            while chunk := audio.readframes(3200):
                socket.send(chunk)
        socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
    except (OSError, EOFError, wave.Error, ConnectionClosed):
        socket.close()


def _event(socket: "ClientConnection", *, deadline: float) -> dict:
    """Read one server event within the request deadline and surface errors."""
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("NeMo streaming exceeded its request timeout.")
    event = json.loads(socket.recv(timeout=remaining))
    if event.get("type") == "error":
        raise ValueError(f"NeMo streaming failed: {event.get('error')}")
    return event


def _collect_finals(
    socket: "ClientConnection", *, deadline: float, events: list[dict]
) -> dict:
    """Collect finalized words until commit acknowledgement; ignore provisional text."""
    words, texts = [], []
    while True:
        event = _event(socket, deadline=deadline)
        events.append(event)
        if event["type"].endswith(".completed"):
            words.extend(event.get("words", []))
            texts.append(event.get("transcript", ""))
        if event["type"] == "input_audio_buffer.committed":
            return {"text": " ".join(texts), "words": words}


def _websocket_url(base_url: str) -> str:
    """Map a configured transcription API prefix to NeMo's specific socket path."""
    address = urlsplit(base_url)
    scheme = "wss" if address.scheme == "https" else "ws"
    path = address.path.rstrip("/") + "/audio/transcriptions/realtime"
    return urlunsplit((scheme, address.netloc, path, "", ""))


def transcribe_stream(
    path: Path,
    *,
    base_url: str,
    language: str,
    diarized: bool,
    timeout_seconds: float,
    event_path: Path,
    api_key: str,
) -> dict:
    """Own a fresh recognition stream and persist its events, including on failure.

    Replay bounded WAV PCM chunks without realtime sleeps. Receive concurrently
    so server output cannot block long uploads. This is not OpenAI Realtime.
    """
    try:
        from websockets.exceptions import ConnectionClosed
        from websockets.sync.client import connect
    except ImportError as error:
        raise ImportError("NeMo streaming needs scribnic[streaming].") from error

    deadline = monotonic() + timeout_seconds
    session = {
        "sample_rate": 16000,
        "language": language,
        "word_timestamps": True,
        "speaker_diarization": diarized,
    }
    events: list[dict] = []
    try:
        with connect(
            _websocket_url(base_url),
            open_timeout=timeout_seconds,
            additional_headers={"Authorization": "Bearer " + api_key},
        ) as socket:
            created = _event(socket, deadline=deadline)
            events.append(created)
            if created["type"] != "session.created":
                raise ValueError("NeMo did not create a fresh recognition session.")
            socket.send(json.dumps({"type": "session.update", "session": session}))
            updated = _event(socket, deadline=deadline)
            events.append(updated)
            if updated["type"] != "session.updated":
                raise ValueError("NeMo did not accept the recognition settings.")
            sender = threading.Thread(
                target=_send_audio, args=(socket, path), daemon=True
            )
            sender.start()
            try:
                return _collect_finals(socket, deadline=deadline, events=events)
            finally:
                socket.close()
                sender.join(timeout=2)
    except ConnectionClosed as error:
        raise ValueError(
            "NeMo closed the stream before acknowledging final audio."
        ) from error
    finally:
        write_json(event_path, events)
