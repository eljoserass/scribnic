"""Feed PCM to NeMo's native WebSocket, preserving one diarizer per recording."""

import json
import queue
import threading
from collections.abc import Callable, Iterable
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING, TextIO
from urllib.parse import urlsplit, urlunsplit

from ..audio_stream import SAMPLE_RATE, wav_chunks

if TYPE_CHECKING:
    from websockets.sync.client import ClientConnection


def _send_audio(
    socket: "ClientConnection",
    chunks: Iterable[bytes],
    errors: "queue.SimpleQueue[BaseException]",
    stopped: threading.Event,
) -> None:
    """Send with backpressure and transfer producer failures to the receiver."""
    try:
        for chunk in chunks:
            if stopped.is_set():
                return
            socket.send(chunk)
        # Commit destroys recognition and diarizer state: only commit at the end.
        socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
    except (Exception, KeyboardInterrupt) as error:
        # A thread cannot raise into its caller. Wake the receiver and transfer
        # the original exception rather than returning an incomplete transcript.
        errors.put(error)
        socket.close()


def _event(socket: "ClientConnection", *, timeout_seconds: float, log: TextIO) -> dict:
    """Read and persist one event with an inactivity timeout, not a session limit."""
    event = json.loads(socket.recv(timeout=timeout_seconds))
    log.write(json.dumps(event, ensure_ascii=False) + "\n")
    log.flush()
    if event.get("type") == "error":
        raise ValueError(f"NeMo streaming failed: {event.get('error')}")
    return event


def _collect_finals(
    socket: "ClientConnection",
    *,
    timeout_seconds: float,
    log: TextIO,
    on_update: Callable[[str], None] | None,
) -> dict:
    """Retain final words; provisional deltas remain in the diagnostic log."""
    words, texts = [], []
    while True:
        event = _event(socket, timeout_seconds=timeout_seconds, log=log)
        if event["type"].endswith(".completed"):
            words.extend(event.get("words", []))
            texts.append(event.get("transcript", ""))
            if on_update is not None:
                on_update(" ".join(texts))
        if event["type"] == "input_audio_buffer.committed":
            return {"text": " ".join(texts), "words": words}


def _websocket_url(base_url: str) -> str:
    """Map a transcription API prefix to NeMo's specific socket path."""
    address = urlsplit(base_url)
    scheme = "wss" if address.scheme == "https" else "ws"
    path = address.path.rstrip("/") + "/audio/transcriptions/realtime"
    return urlunsplit((scheme, address.netloc, path, "", ""))


def _recognize(
    socket: "ClientConnection",
    chunks: Iterable[bytes],
    *,
    timeout_seconds: float,
    log: TextIO,
    on_update: Callable[[str], None] | None,
) -> dict:
    """Receive concurrently and stop the owned sender on failure or cancellation."""
    from websockets.exceptions import ConnectionClosed

    errors: queue.SimpleQueue[BaseException] = queue.SimpleQueue()
    stopped = threading.Event()
    sender = threading.Thread(
        target=_send_audio, args=(socket, chunks, errors, stopped), daemon=True
    )
    sender.start()
    try:
        result = _collect_finals(
            socket, timeout_seconds=timeout_seconds, log=log, on_update=on_update
        )
    except ConnectionClosed as error:
        if not errors.empty():
            raise errors.get() from error
        raise ValueError("NeMo closed before acknowledging final audio.") from error
    finally:
        stopped.set()
        socket.close()
        sender.join(timeout=2)
    if not errors.empty():
        raise errors.get()
    return result


def transcribe_pcm(
    chunks: Iterable[bytes],
    *,
    base_url: str,
    language: str,
    diarized: bool,
    timeout_seconds: float,
    event_path: Path,
    api_key: str,
    on_update: Callable[[str], None] | None = None,
) -> dict:
    """Own one PCM16/16 kHz session and stream diagnostic events to JSONL.

    The source can replay WAV or receive live capture. This is NeMo's protocol,
    not OpenAI Realtime. Only finalized hypotheses update the callback: native
    deltas do not unambiguously distinguish all hypothesis revisions.
    """
    try:
        from websockets.sync.client import connect
    except ImportError as error:
        raise ImportError("NeMo streaming needs scribnic[streaming].") from error

    session = {
        "sample_rate": SAMPLE_RATE,
        "language": language,
        "word_timestamps": True,
        "speaker_diarization": diarized,
    }
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("w", encoding="utf-8") as log:
        with connect(
            _websocket_url(base_url),
            open_timeout=timeout_seconds,
            additional_headers={"Authorization": "Bearer " + api_key},
        ) as socket:
            created = _event(socket, timeout_seconds=timeout_seconds, log=log)
            if created["type"] != "session.created":
                raise ValueError("NeMo did not create a fresh recognition session.")
            socket.send(json.dumps({"type": "session.update", "session": session}))
            updated = _event(socket, timeout_seconds=timeout_seconds, log=log)
            if updated["type"] != "session.updated":
                raise ValueError("NeMo did not accept the recognition settings.")
            result = _recognize(
                socket,
                chunks,
                timeout_seconds=timeout_seconds,
                log=log,
                on_update=on_update,
            )
            return result | {"native_events_path": str(event_path)}


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
    """Keep WAV callers compatible while using the shared, unpaced PCM source."""
    with closing(wav_chunks(path)) as chunks:
        return transcribe_pcm(
            chunks,
            base_url=base_url,
            language=language,
            diarized=diarized,
            timeout_seconds=timeout_seconds,
            event_path=event_path,
            api_key=api_key,
        )
