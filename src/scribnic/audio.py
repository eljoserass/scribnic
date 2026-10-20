"""Input audio validation."""

import wave
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

from .artifacts import AudioFile
from .models import Audio


def inspect_audio(source: AudioFile) -> AudioFile:
    """Validate a complete mono PCM16 WAV at 16 kHz without loading models."""
    path = source.path.resolve(strict=True)
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise ValueError("Audio must be mono PCM16 WAV at 16 kHz.")
        frames = wav.getnframes()
        if frames == 0 or len(wav.readframes(frames)) != frames * 2:
            raise ValueError("Audio is empty or incomplete.")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return replace(source, path=path, sample_rate=16000, duration=frames / 16000, sha256=digest)


@dataclass(frozen=True)
class FileSource:
    path: Path

    def load(self) -> Audio:
        path = Path(self.path).expanduser().resolve(strict=True)
        with wave.open(str(path), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
                raise ValueError("Audio must be mono PCM16 WAV at 16 kHz.")
            frames = wav.getnframes()
            if frames == 0 or len(wav.readframes(frames)) != frames * 2:
                raise ValueError("Audio is empty or incomplete.")
            return Audio(path, 16000, frames / 16000)
