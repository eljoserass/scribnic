"""PCM sources and single-use recordings, independent of inference protocols."""

import hashlib
import threading
import time
import wave
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .artifacts import AudioFile
from .audio import BYTES_PER_SECOND, SAMPLE_RATE

CHUNK_FRAMES = 3200


@dataclass(frozen=True)
class TranscriptUpdate:
    """A replaceable transcript snapshot; final means the input has finished."""

    text: str
    audio_seconds: float
    final: bool = False


UpdateCallback = Callable[[TranscriptUpdate], None]


def wav_chunks(path: Path, *, realtime: bool = False) -> Iterator[bytes]:
    """Read mono PCM16/16 kHz WAV in 200 ms blocks, optionally pacing ingestion.

    Pacing simulates an arriving recording. Without pacing, backpressure comes
    from the consumer. The file is closed on exhaustion or generator closure.
    """
    with wave.open(str(path), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (
            1,
            2,
            SAMPLE_RATE,
        ):
            raise ValueError("Streaming audio must be mono PCM16 WAV at 16 kHz.")
        started = time.monotonic()
        frames = 0
        while chunk := audio.readframes(CHUNK_FRAMES):
            frames += len(chunk) // 2
            if realtime:
                delay = started + frames / SAMPLE_RATE - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            yield chunk


class PCMRecording:
    """Own one WAV writer while consuming PCM once, retaining no audio history.

    The producer supplies frame-aligned mono PCM16 at 16 kHz. Blocking device
    capture and its shutdown belong to the caller. Cancel between incoming
    chunks with the supplied event; no audio is silently dropped.
    """

    def __init__(
        self,
        chunks: Iterable[bytes],
        path: Path,
        *,
        on_update: UpdateCallback | None = None,
        cancel: threading.Event | None = None,
    ) -> None:
        """Configure a recording without opening a file or consuming the producer."""
        self.chunks = chunks
        self.path = path
        self.on_update = on_update
        self.cancel = cancel
        self.bytes_received = 0
        self._used = False
        self._writer: wave.Wave_write | None = None
        self._sha256: str | None = None

    def __enter__(self) -> "PCMRecording":
        """Open the durable recording owned by this execution."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = wave.open(str(self.path), "wb")
        self._writer.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        return self

    def __exit__(self, *exc: object) -> None:
        """Finalize the WAV header even when input or inference fails."""
        self.finish()

    @property
    def duration(self) -> float:
        """Return the number of source seconds consumed so far."""
        return self.bytes_received / BYTES_PER_SECOND

    def __iter__(self) -> Iterator[bytes]:
        """Validate, record and yield bounded blocks with consumer backpressure."""
        if self._used or self._writer is None:
            raise ValueError("A PCM recording must be open and consumed only once.")
        self._used = True
        for chunk in self.chunks:
            if self.cancel is not None and self.cancel.is_set():
                raise InterruptedError("Audio input was cancelled.")
            if not isinstance(chunk, bytes) or len(chunk) % 2:
                raise ValueError("PCM chunks must contain complete PCM16 frames.")
            for offset in range(0, len(chunk), CHUNK_FRAMES * 2):
                # A capture source may yield a large block: stop between its parts.
                if self.cancel is not None and self.cancel.is_set():
                    raise InterruptedError("Audio input was cancelled.")
                block = chunk[offset : offset + CHUNK_FRAMES * 2]
                self._writer.writeframesraw(block)
                self.bytes_received += len(block)
                yield block
        if not self.bytes_received:
            raise ValueError("Audio input is empty.")

    def finish(self) -> AudioFile:
        """Close the writer and return a reusable audio artifact, including on error."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
            with self.path.open("rb") as audio:
                self._sha256 = hashlib.file_digest(audio, "sha256").hexdigest()
        return AudioFile(self.path, SAMPLE_RATE, self.duration, self._sha256)

    def notify(self, text: str, *, final: bool = False) -> None:
        """Deliver a transcript snapshot without owning any terminal presentation."""
        if self.on_update is not None:
            self.on_update(
                TranscriptUpdate(text, audio_seconds=self.duration, final=final)
            )
