"""Small deterministic adapters for trying the pipeline without model downloads."""

from ..models import Audio, SpeakerSpan, TextSpan, Utterance


class DemoTranscriber:
    def transcribe(self, audio: Audio, speakers: tuple[SpeakerSpan, ...]) -> tuple[TextSpan, ...]:
        half = audio.duration / 2
        return (TextSpan(0, half, "Hola, ¿cómo estás?"),
                TextSpan(half, audio.duration, "Hoy estoy mejor."))


class DemoDiarizer:
    def diarize(self, audio: Audio) -> tuple[SpeakerSpan, ...]:
        half = audio.duration / 2
        return (SpeakerSpan(0, half, "speaker_0"),
                SpeakerSpan(half, audio.duration, "speaker_1"))


class DemoTextGenerator:
    def generate(self, utterances: tuple[Utterance, ...], instruction: str) -> str:
        conversation = "\n".join(
            f"{turn.speaker or 'sin atribuir'}: {turn.text}" for turn in utterances
        )
        return f"[DEMO: texto inventado, sin LLM]\nInstrucción: {instruction}\n{conversation}"
from ..artifacts import (
    Artifact, AudioFile, DiarizedTranscript, Note, SpeakerSegment, StepOutput,
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
            transcript = DiarizedTranscript((
                SpeakerSegment("¿Qué le ocurre?", "S01", 0, duration / 2),
                SpeakerSegment("Tengo tos desde ayer.", "S02", duration / 2, duration),
            ))
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
