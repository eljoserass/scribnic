"""Data shared by the pipeline and every model implementation."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Audio:
    path: Path
    sample_rate: int
    duration: float


@dataclass(frozen=True)
class TextSpan:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SpeakerSpan:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class Utterance:
    start: float
    end: float
    text: str
    speaker: str | None
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class PipelineResult:
    transcription: tuple[TextSpan, ...]
    diarization: tuple[SpeakerSpan, ...]
    utterances: tuple[Utterance, ...]
    generated_text: str | None
