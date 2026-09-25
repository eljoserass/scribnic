"""Input audio validation."""

import wave
from dataclasses import dataclass
from pathlib import Path

from .models import Audio


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
