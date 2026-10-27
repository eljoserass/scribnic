"""Bounded overlapping PCM windows over a resident Whisper HTTP service."""

import io
import json
import math
import re
import wave
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

from ..artifacts import TextSegment
from ..audio_stream import BYTES_PER_SECOND, SAMPLE_RATE
from .transcription_formats import parse_segments


@dataclass(frozen=True, kw_only=True)
class WindowSettings:
    """Recognition windows in seconds; transport fragments are independent."""

    window_seconds: float = 10
    overlap_seconds: float = 2

    def __post_init__(self) -> None:
        """Reject invalid geometry before any input or model is consumed."""
        values = (self.window_seconds, self.overlap_seconds)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("Whisper window settings must be finite numbers.")
        if not 0 <= self.overlap_seconds < self.window_seconds:
            raise ValueError("Require 0 <= overlap_seconds < window_seconds.")
        if self.window_seconds < 1:
            raise ValueError("Whisper window_seconds must be at least one second.")
        if int(self.overlap_seconds * SAMPLE_RATE) >= int(
            self.window_seconds * SAMPLE_RATE
        ):
            raise ValueError("Whisper window stride must contain at least one sample.")


def _wav_bytes(pcm: bytes) -> bytes:
    """Encode one bounded mono PCM16 window without a temporary file."""
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
        audio.writeframes(pcm)
    return output.getvalue()


def _normalized(text: str) -> str:
    """Compare overlap words without punctuation/case differences."""
    return "".join(re.findall(r"\w+", text.casefold()))


def _merge(
    previous: list[TextSegment],
    current: list[TextSegment],
    *,
    start: float,
    previous_end: float,
) -> list[TextSegment]:
    """Reconcile overlap words, falling back to temporal ownership at its midpoint.

    Matching is restricted to the overlapping time range. Without a usable anchor,
    intervals crossing the ownership midpoint are clipped rather than reordering
    words. Original estimates remain in per-window diagnostics for evaluation.
    """
    if not previous or start >= previous_end:
        return previous + current
    old_overlap = [segment for segment in previous if segment.end >= start]
    new_overlap = [segment for segment in current if segment.start <= previous_end]
    maximum = min(len(old_overlap), len(new_overlap))
    for size in range(maximum, 0, -1):
        old_words = [_normalized(segment.text) for segment in old_overlap[-size:]]
        new_words = [_normalized(segment.text) for segment in new_overlap[:size]]
        if all(old_words) and old_words == new_words:
            prefix = previous[:-size]
            if not prefix or prefix[-1].start <= current[0].start:
                return prefix + current
            suffix = current[size:]
            if not suffix or previous[-1].start <= suffix[0].start:
                return previous + suffix
    boundary = (start + previous_end) / 2
    left = [
        TextSegment(segment.text, segment.start, min(segment.end, boundary))
        for segment in previous
        if (segment.start + segment.end) / 2 < boundary
    ]
    right = [
        TextSegment(segment.text, max(segment.start, boundary), segment.end)
        for segment in current
        if (segment.start + segment.end) / 2 >= boundary
    ]
    return left + right


class _Recognition:
    """Own transcript stitching and per-window diagnostics for one recording."""

    def __init__(
        self,
        request: Callable[[bytes], dict],
        *,
        log: TextIO,
        on_update: Callable[[str], None] | None,
    ) -> None:
        """Retain the resident service request function without loading a model."""
        self.request = request
        self.log = log
        self.on_update = on_update
        self.segments: list[TextSegment] = []
        self.end = 0.0
        self.count = 0

    def process(self, pcm: bytes, *, start_seconds: float) -> None:
        """Recognize one window and publish a replaceable transcript snapshot."""
        duration = len(pcm) / BYTES_PER_SECOND
        payload = self.request(_wav_bytes(pcm))
        transcript = parse_segments(payload, duration_seconds=duration)
        current = [
            TextSegment(
                segment.text, segment.start + start_seconds, segment.end + start_seconds
            )
            for segment in transcript.segments
        ]
        self.segments = _merge(
            self.segments, current, start=start_seconds, previous_end=self.end
        )
        self.end = start_seconds + duration
        self.count += 1
        self.log.write(
            json.dumps(
                {
                    "start_seconds": start_seconds,
                    "audio_seconds": duration,
                    "response": payload,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        self.log.flush()
        if self.on_update is not None:
            self.on_update(self.text())

    def text(self) -> str:
        """Render the current stitched word segments for diagnostics and callbacks."""
        return " ".join(segment.text.strip() for segment in self.segments)


def transcribe_windows(
    chunks: Iterable[bytes],
    *,
    request: Callable[[bytes], dict],
    settings: WindowSettings,
    event_path: Path,
    on_update: Callable[[str], None] | None = None,
) -> dict:
    """Recognize while PCM arrives, retaining at most one window of audio.

    Each request re-encodes its window; model weights stay in the existing HTTP
    server. The caller owns capture and pacing. The final transcript may grow
    with recording length, while native per-window responses go to JSONL.
    """
    window_bytes = int(settings.window_seconds * SAMPLE_RATE) * 2
    overlap_bytes = int(settings.overlap_seconds * SAMPLE_RATE) * 2
    stride = window_bytes - overlap_bytes
    buffer = bytearray()
    start = 0
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("w", encoding="utf-8") as log:
        recognition = _Recognition(request, log=log, on_update=on_update)
        for chunk in chunks:
            offset = 0
            while offset < len(chunk):
                length = min(window_bytes - len(buffer), len(chunk) - offset)
                buffer.extend(chunk[offset : offset + length])
                offset += length
                if len(buffer) == window_bytes:
                    recognition.process(
                        bytes(buffer), start_seconds=start / BYTES_PER_SECOND
                    )
                    del buffer[:stride]
                    start += stride
        end_seconds = (start + len(buffer)) / BYTES_PER_SECOND
        if buffer and end_seconds > recognition.end + 1 / SAMPLE_RATE:
            recognition.process(bytes(buffer), start_seconds=start / BYTES_PER_SECOND)
        return {
            "text": recognition.text(),
            "segments": [asdict(segment) for segment in recognition.segments],
            "windowing": {
                "timestamp_policy": "overlap_anchor_or_midpoint_ownership_clipping",
                "window_seconds": settings.window_seconds,
                "overlap_seconds": settings.overlap_seconds,
                "requests": recognition.count,
                "native_windows_path": str(event_path),
            },
        }
