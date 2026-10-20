"""Invented deterministic outputs for exercising workflow wiring offline."""

from ..artifacts import (
    Artifact,
    AudioFile,
    DiarizedTranscript,
    Note,
    SpeakerSegment,
    StepOutput,
)
from ..config import StepConfig


class DemoBackend:
    """Deterministic invented outputs for exercising workflow wiring offline."""

    def __init__(self, step: StepConfig):
        self.step = step

    def run(self, source: Artifact) -> StepOutput:
        if self.step.model.output == "diarized_transcript":
            if not isinstance(source, AudioFile):
                raise ValueError("The demo speech step expects audio.")
            duration = source.duration or 1
            transcript = DiarizedTranscript(
                (
                    SpeakerSegment("¿Qué le ocurre?", "S01", 0, duration / 2),
                    SpeakerSegment(
                        "Tengo tos desde ayer.", "S02", duration / 2, duration
                    ),
                )
            )
            return StepOutput(transcript, {"demo": True})
        text = (
            "DEMO · salida inventada\n\n"
            "S — Subjetivo: Tengo tos desde ayer.\n"
            "O — Objetivo: No consta.\n"
            "A — Análisis: No consta.\n"
            "P — Plan: No consta."
        )
        return StepOutput(Note(text), {"demo": True})

    def close(self) -> None:
        pass

    def process_ids(self) -> tuple[int, ...]:
        return ()
