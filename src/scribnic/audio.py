"""Input audio validation."""

import hashlib
import wave
from dataclasses import replace
from pathlib import Path

from .artifacts import AudioFile


def _duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (
            1,
            2,
            16000,
        ):
            raise ValueError("Audio must be mono PCM16 WAV at 16 kHz.")
        frames = wav.getnframes()
        if frames == 0 or len(wav.readframes(frames)) != frames * 2:
            raise ValueError("Audio is empty or incomplete.")
    return frames / 16000


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
        source, path=path, sample_rate=16000, duration=duration, sha256=digest
    )
