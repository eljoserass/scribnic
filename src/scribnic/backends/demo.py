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
