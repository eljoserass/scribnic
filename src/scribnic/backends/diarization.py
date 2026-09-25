"""Diarization adapters with the same output contract."""

import wave

from ..models import Audio, SpeakerSpan


class NemotronDiarizer:
    """Nemotron 3 through Transformers, on the selected PyTorch device."""

    MODEL = "nvidia/Nemotron-3-Diarization"

    def __init__(self, device: str):
        import torch
        from transformers import AutoModelForAudioFrameClassification, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.MODEL)
        self.model = AutoModelForAudioFrameClassification.from_pretrained(self.MODEL).to(device).eval()

    def diarize(self, audio: Audio) -> tuple[SpeakerSpan, ...]:
        import numpy as np

        with wave.open(str(audio.path), "rb") as wav:
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype("float32") / 32768
        inputs = self.processor(samples, sampling_rate=audio.sample_rate).to(
            self.model.device, dtype=self.model.dtype,
        )
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
        segments = self.processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
        return tuple(sorted(
            (SpeakerSpan(float(segment["Start"]), float(segment["End"]),
                         f"speaker_{segment['Speaker']}") for segment in segments),
            key=lambda span: span.start,
        ))


class SortformerDiarizer:
    """Previous NeMo Sortformer v2 implementation."""

    MODEL = "nvidia/diar_streaming_sortformer_4spk-v2"

    def __init__(self, device: str):
        from nemo.collections.asr.models import SortformerEncLabelModel

        self.model = SortformerEncLabelModel.from_pretrained(
            self.MODEL, map_location=device,
        ).eval()
        config = self.model.sortformer_modules
        config.chunk_len = 340
        config.chunk_right_context = 40
        config.fifo_len = 40
        config.spkcache_update_period = 300
        # The existing ROCm setup uses PyTorch convolutions instead of Triton.
        conv = getattr(self.model.encoder.pre_encode, "conv", None)
        if conv is not None and hasattr(conv, "fuse_triton"):
            conv.fuse_triton = False

    def diarize(self, audio: Audio) -> tuple[SpeakerSpan, ...]:
        segments = self.model.diarize(audio=[str(audio.path)], batch_size=1)[0]
        spans = []
        for segment in segments:
            start, end, speaker = segment.split()
            spans.append(SpeakerSpan(float(start), float(end), speaker))
        return tuple(sorted(spans, key=lambda span: span.start))
