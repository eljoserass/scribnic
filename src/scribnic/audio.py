"""Canonical WAV validation and optional waveform decoding for model frontends."""

import hashlib
import wave
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from .artifacts import AudioFile

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

SAMPLE_RATE = 16000
BYTES_PER_SECOND = SAMPLE_RATE * 2


def _duration(path: Path) -> float:
    """Validate all PCM frames using bounded reads and return source seconds."""
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (
            1,
            2,
            SAMPLE_RATE,
        ):
            raise ValueError("Audio must be mono PCM16 WAV at 16 kHz.")
        frames = wav.getnframes()
        received = 0
        while chunk := wav.readframes(SAMPLE_RATE):
            received += len(chunk)
        if frames == 0 or received != frames * 2:
            raise ValueError("Audio is empty or incomplete.")
    return frames / SAMPLE_RATE


def inspect_audio(source: AudioFile) -> AudioFile:
    """Validate a complete mono PCM16 WAV at 16 kHz without loading models."""
    path = source.path.resolve(strict=True)
    try:
        duration = _duration(path)
    except (EOFError, wave.Error) as error:
        raise ValueError(f"Invalid WAV file: {path}") from error
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return replace(
        source, path=path, sample_rate=SAMPLE_RATE, duration=duration, sha256=digest
    )


def read_waveform(path: Path) -> "NDArray[np.float32]":
    """Decode a validated mono PCM16 WAV to normalized float32 samples.

    Offline neural frontends need the complete array. NumPy is imported only when
    one of those optional models calls this function; streaming uses PCM blocks.
    """
    import numpy as np

    with wave.open(str(path), "rb") as audio:
        pcm = audio.readframes(audio.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768
