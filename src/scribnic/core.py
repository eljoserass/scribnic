"""Pipeline contracts and model-independent speaker attribution."""

import math
from dataclasses import dataclass
from typing import Protocol

from .models import Audio, PipelineResult, SpeakerSpan, TextSpan, Utterance


class AudioSource(Protocol):
    def load(self) -> Audio: ...


class Transcriber(Protocol):
    def transcribe(self, audio: Audio, speakers: tuple[SpeakerSpan, ...]) -> tuple[TextSpan, ...]: ...


class Diarizer(Protocol):
    def diarize(self, audio: Audio) -> tuple[SpeakerSpan, ...]: ...


class TextGenerator(Protocol):
    def generate(self, utterances: tuple[Utterance, ...], instruction: str) -> str: ...


def attribute_speakers(
    text: tuple[TextSpan, ...], speakers: tuple[SpeakerSpan, ...]
) -> tuple[Utterance, ...]:
    """Attribute a text region when exactly one speaker is active at its midpoint."""
    result = []
    for span in text:
        midpoint = (span.start + span.end) / 2
        candidates = tuple(sorted({
            turn.speaker for turn in speakers if turn.start <= midpoint < turn.end
        }))
        result.append(Utterance(
            span.start, span.end, span.text,
            candidates[0] if len(candidates) == 1 else None, candidates,
        ))
    return tuple(result)


def validate_timeline(spans: tuple[TextSpan, ...] | tuple[SpeakerSpan, ...], audio: Audio) -> None:
    previous_start = -1.0
    for span in spans:
        if not (math.isfinite(span.start) and math.isfinite(span.end)
                and 0 <= span.start <= span.end <= audio.duration + 0.1
                and span.start >= previous_start):
            raise ValueError(f"Invalid timeline interval: {span!r}")
        previous_start = span.start


@dataclass
class Pipeline:
    transcriber: Transcriber
    diarizer: Diarizer
    generator: TextGenerator | None = None

    def run(self, source: AudioSource, instruction: str = "Summarize the conversation.") -> PipelineResult:
        audio = source.load()
        speakers = self.diarizer.diarize(audio)
        validate_timeline(speakers, audio)
        text = self.transcriber.transcribe(audio, speakers)
        validate_timeline(text, audio)
        utterances = attribute_speakers(text, speakers)
        generated = self.generator.generate(utterances, instruction) if self.generator else None
        return PipelineResult(text, speakers, utterances, generated)
