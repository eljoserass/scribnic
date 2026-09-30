"""ASR implementations. Both return text in the original audio's clock."""

import wave

from ..models import Audio, SpeakerSpan, TextSpan


def speech_regions(speakers: tuple[SpeakerSpan, ...], duration: float) -> tuple[tuple[float, float], ...]:
    """Partition speech at speaker boundaries, including overlapping speech once."""
    if not speakers:
        return ((0.0, duration),)
    boundaries = sorted({point for span in speakers for point in (span.start, span.end)})
    regions: list[tuple[float, float]] = []
    active_before: tuple[str, ...] = ()
    for start, end in zip(boundaries, boundaries[1:]):
        active = tuple(sorted({span.speaker for span in speakers
                               if span.start < end and span.end > start}))
        if not active or start == end:
            active_before = ()
            continue
        if regions and active == active_before and regions[-1][1] == start:
            regions[-1] = (regions[-1][0], end)
        else:
            regions.append((start, end))
        active_before = active
    return tuple(regions)


def audio_chunks(audio: Audio, speakers: tuple[SpeakerSpan, ...], max_seconds: int = 30):
    """Yield bounded mono float32 chunks with offsets in the original audio clock."""
    import numpy as np

    with wave.open(str(audio.path), "rb") as wav:
        for region_start, region_end in speech_regions(speakers, audio.duration):
            start_frame = min(round(region_start * audio.sample_rate), wav.getnframes())
            end_frame = min(round(region_end * audio.sample_rate), wav.getnframes())
            for frame in range(start_frame, end_frame, max_seconds * audio.sample_rate):
                frames = min(end_frame - frame, max_seconds * audio.sample_rate)
                wav.setpos(frame)
                samples = np.frombuffer(wav.readframes(frames), dtype="<i2").astype("float32") / 32768
                yield frame / audio.sample_rate, (frame + frames) / audio.sample_rate, samples


class NemotronTranscriber:
    """Transcribe diarized regions with the multilingual Nemotron 3.5 ASR model.

    This checkpoint's documented API returns text, so timings are region-level.
    """

    MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"

    def __init__(self, device: str, language: str = "es-ES"):
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.MODEL)
        self.model = AutoModelForRNNT.from_pretrained(self.MODEL).to(device).eval()
        self.language = language

    def transcribe(self, audio: Audio, speakers: tuple[SpeakerSpan, ...]) -> tuple[TextSpan, ...]:
        spans = []
        for start, end, samples in audio_chunks(audio, speakers):
            inputs = self.processor(
                samples, sampling_rate=audio.sample_rate,
                language=self.language, return_tensors="pt",
            ).to(self.model.device, dtype=self.model.dtype)
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, return_dict_in_generate=True)
            text = self.processor.batch_decode(
                output.sequences, skip_special_tokens=True,
            )[0].strip()
            if text:
                spans.append(TextSpan(start, end, text))
        return tuple(spans)


class QwenTranscriber:
    """Qwen3-ASR produces text, while the selected diarizer supplies its regions."""

    MODEL = "Qwen/Qwen3-ASR-0.6B-hf"

    def __init__(self, device: str, language: str = "es-ES"):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.MODEL)
        self.model = AutoModelForMultimodalLM.from_pretrained(self.MODEL).to(device).eval()
        self.language = language.split("-")[0]

    def transcribe(self, audio: Audio, speakers: tuple[SpeakerSpan, ...]) -> tuple[TextSpan, ...]:
        spans = []
        for start, end, samples in audio_chunks(audio, speakers):
            inputs = self.processor.apply_transcription_request(
                audio=samples, language=self.language,
            ).to(self.model.device, self.model.dtype)
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
            generated = output[:, inputs["input_ids"].shape[1]:]
            text = self.processor.decode(generated, return_format="transcription_only")[0].strip()
            if text:
                spans.append(TextSpan(start, end, text))
        return tuple(spans)


class CanaryTranscriber:
    """The previous word-timestamp ASR implementation."""

    MODEL = "nvidia/canary-1b-v2"

    def __init__(self, device: str, language: str = "es"):
        from nemo.collections.asr.models import ASRModel

        self.model = ASRModel.from_pretrained(self.MODEL, map_location=device).eval()
        self.language = language.split("-")[0]

    def transcribe(self, audio: Audio, speakers: tuple[SpeakerSpan, ...]) -> tuple[TextSpan, ...]:
        hypothesis = self.model.transcribe(
            [str(audio.path)], batch_size=1, timestamps=True,
            source_lang=self.language, target_lang=self.language,
        )[0]
        timestamps = getattr(hypothesis, "timestamp", None)
        if not isinstance(timestamps, dict) or "word" not in timestamps:
            raise ValueError("Canary did not return word timestamps.")
        if hypothesis.text.strip() and not timestamps["word"]:
            raise ValueError("Canary returned text without aligned words.")
        return tuple(TextSpan(float(w["start"]), float(w["end"]), w["word"])
                     for w in timestamps["word"])
