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


class JointRecognizer(Protocol):
    def recognize(self, audio: Audio) -> tuple[Utterance, ...]: ...


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
    transcriber: Transcriber | None = None
    diarizer: Diarizer | None = None
    generator: TextGenerator | None = None
    recognizer: JointRecognizer | None = None

    def run(self, source: AudioSource, instruction: str = "Summarize the conversation.") -> PipelineResult:
        audio = source.load()
        if self.recognizer is not None:
            if self.transcriber is not None or self.diarizer is not None:
                raise ValueError("A joint recognizer cannot be combined with separate stages.")
            utterances = self.recognizer.recognize(audio)
            text = tuple(TextSpan(turn.start, turn.end, turn.text) for turn in utterances)
            speakers = tuple(SpeakerSpan(turn.start, turn.end, turn.speaker)
                             for turn in utterances if turn.speaker is not None)
        else:
            if self.transcriber is None or self.diarizer is None:
                raise ValueError("Both transcriber and diarizer are required.")
            speakers = self.diarizer.diarize(audio)
            validate_timeline(speakers, audio)
            text = self.transcriber.transcribe(audio, speakers)
            utterances = attribute_speakers(text, speakers)
        validate_timeline(text, audio)
        validate_timeline(speakers, audio)
        for turn in utterances:
            if turn.speaker is not None and turn.speaker not in turn.candidates:
                raise ValueError(f"Speaker is absent from candidates: {turn!r}")
        generated = self.generator.generate(utterances, instruction) if self.generator else None
        return PipelineResult(text, speakers, utterances, generated)
