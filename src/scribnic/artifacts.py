"""Immutable inputs and outputs shared by every workflow backend."""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, TypeAlias

ArtifactKind: TypeAlias = Literal[
    "audio", "transcript", "diarized_transcript", "note"
]
ARTIFACT_KINDS = ("audio", "transcript", "diarized_transcript", "note")


def validate_interval(start: float | None, end: float | None) -> None:
    """Intervals use seconds; both bounds may be unknown."""
    if start is None and end is None:
        return
    if start is None or end is None:
        raise ValueError("Provide both start and end, or neither.")
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
        raise ValueError(f"Invalid interval in seconds: {start!r}, {end!r}")


@dataclass(frozen=True)
class AudioFile:
    path: Path
    sample_rate: int | None = None
    duration: float | None = None
    sha256: str | None = None
    kind: ClassVar[ArtifactKind] = "audio"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).expanduser())


@dataclass(frozen=True)
class TextSegment:
    text: str
    start: float | None = None
    end: float | None = None

    def __post_init__(self) -> None:
        validate_interval(self.start, self.end)


@dataclass(frozen=True)
class SpeakerSegment:
    text: str
    speaker: str | None
    start: float | None = None
    end: float | None = None
    role: str | None = None

    def __post_init__(self) -> None:
        validate_interval(self.start, self.end)


@dataclass(frozen=True)
class Transcript:
    segments: tuple[TextSegment, ...]
    kind: ClassVar[ArtifactKind] = "transcript"

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", tuple(self.segments))

    @classmethod
    def from_text(cls, text: str) -> "Transcript":
        return cls((TextSegment(text),))

    @property
    def text(self) -> str:
        return "\n".join(segment.text for segment in self.segments)


@dataclass(frozen=True)
class DiarizedTranscript:
    segments: tuple[SpeakerSegment, ...]
    kind: ClassVar[ArtifactKind] = "diarized_transcript"

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", tuple(self.segments))
        previous_start = -1.0
        for segment in self.segments:
            if segment.start is not None:
                if segment.start < previous_start:
                    raise ValueError("Transcript segments must be ordered by start time.")
                previous_start = segment.start

    @property
    def text(self) -> str:
        return "\n".join(segment.text for segment in self.segments)

    def without_speakers(self) -> Transcript:
        """Explicit projection that preserves text, turn boundaries, and times."""
        segments = tuple(
            TextSegment(segment.text, segment.start, segment.end)
            for segment in self.segments
        )
        return Transcript(segments)


@dataclass(frozen=True)
class Note:
    text: str
    kind: ClassVar[ArtifactKind] = "note"


Artifact: TypeAlias = AudioFile | Transcript | DiarizedTranscript | Note


@dataclass(frozen=True)
class StepOutput:
    artifact: Artifact
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class StepRecord:
    name: str
    backend: str
    elapsed_seconds: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    run_id: str
    workflow: str
    requested_output: ArtifactKind
    source: Artifact
    artifacts: tuple[Artifact, ...]
    steps: tuple[StepRecord, ...]
    configuration: dict[str, object]
    started_at: str
    elapsed_seconds: float
    status: str = "success"
    error: str | None = None
    trace_id: str | None = None

    def get(self, kind: ArtifactKind) -> Artifact | None:
        for artifact in reversed(self.artifacts):
            if artifact.kind == kind:
                return artifact
        return None

    @property
    def transcript(self) -> Transcript | DiarizedTranscript | None:
        for artifact in reversed(self.artifacts):
            if isinstance(artifact, (Transcript, DiarizedTranscript)):
                return artifact
        return None

    @property
    def note(self) -> Note | None:
        artifact = self.get("note")
        return artifact if isinstance(artifact, Note) else None

    @property
    def output(self) -> Artifact | None:
        return self.get(self.requested_output)
