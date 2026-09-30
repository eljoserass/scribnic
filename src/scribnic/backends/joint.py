"""Adapters that produce timed, speaker-attributed turns in one inference pass."""

import re
import wave

from ..models import Audio, Utterance


_MOSS_SEGMENT = re.compile(r"\[(\d+(?:\.\d+)?)\]\[S(\d+)\](.*?)\[(\d+(?:\.\d+)?)\]", re.DOTALL)
_MOSS_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，"
    "正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。"
)


def parse_moss_transcript(output: str) -> tuple[Utterance, ...]:
    """Convert MOSS's compact [start][speaker]text[end] format to turns."""
    turns = []
    position = 0
    for match in _MOSS_SEGMENT.finditer(output):
        if output[position:match.start()].strip():
            raise ValueError("MOSS returned an unparseable transcript fragment.")
        start, speaker_number, text, end = match.groups()
        text = text.strip()
        if text:
            speaker = f"speaker_{int(speaker_number) - 1}"
            turns.append(Utterance(float(start), float(end), text, speaker, (speaker,)))
        position = match.end()
    if output[position:].strip() or (output.strip() and not turns):
        raise ValueError("MOSS returned an unparseable transcript fragment.")
    return tuple(turns)


class MossRecognizer:
    """MOSS Transcribe-Diarize via its Transformers remote-code interface."""

    MODEL = "OpenMOSS-Team/MOSS-Transcribe-Diarize"

    def __init__(self, device: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.torch = torch
        self.device = torch.device(device)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL, trust_remote_code=True, dtype="auto", attn_implementation="sdpa",
        ).to(dtype=dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(self.MODEL, trust_remote_code=True)

    def recognize(self, audio: Audio) -> tuple[Utterance, ...]:
        import numpy as np

        with wave.open(str(audio.path), "rb") as wav:
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype("float32") / 32768
        messages = [{"role": "user", "content": [
            {"type": "audio", "audio": str(audio.path)},
            {"type": "text", "text": _MOSS_PROMPT},
        ]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=prompt, audio=[samples], return_tensors="pt",
            audio_kwargs={"device": str(self.device)} if self.device.type == "cuda" else {},
        ).to(self.device)
        prompt_length = int(inputs["attention_mask"][0].sum().item())
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=min(16384, max(2048, round(audio.duration * 12))),
                do_sample=False,
            )
        transcript = self.processor.tokenizer.decode(
            output[0][prompt_length:], skip_special_tokens=True,
        ).strip()
        return parse_moss_transcript(transcript)
