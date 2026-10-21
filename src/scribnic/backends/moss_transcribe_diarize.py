"""MOSS-Transcribe-Diarize through its pinned Hugging Face implementation."""

import logging
import re
import wave
from contextlib import contextmanager, nullcontext
from importlib.metadata import version
from time import perf_counter

from ..artifacts import (
    Artifact,
    AudioFile,
    DiarizedTranscript,
    SpeakerSegment,
    StepOutput,
)
from ..config import StepConfig
from ..downloads import PreparedStep
from ..observability import GenerationProgress, tracer
from ..profiling import torch_profile

SEGMENT = re.compile(
    r"\[(\d+(?:\.\d+)?)\]\[S(\d+)\](.*?)\[(\d+(?:\.\d+)?)\]", re.DOTALL
)
PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，"
    "正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。"
)


def parse_transcript(text: str) -> DiarizedTranscript:
    segments = []
    position = 0
    for match in SEGMENT.finditer(text):
        if text[position : match.start()].strip():
            raise ValueError("MOSS returned an unparseable transcript fragment.")
        start, speaker, content, end = match.groups()
        if content.strip():
            segments.append(
                SpeakerSegment(
                    content.strip(),
                    f"speaker_{int(speaker) - 1}",
                    float(start),
                    float(end),
                )
            )
        position = match.end()
    if text[position:].strip() or (text.strip() and not segments):
        raise ValueError("MOSS returned an unparseable transcript fragment.")
    return DiarizedTranscript(tuple(segments))


class MossTranscribeDiarize:
    def __init__(self, step: StepConfig, prepared: PreparedStep):
        self.step = step
        self.prepared = prepared
        self.model = None
        self.processor = None
        self.device = None

    def _load(self) -> float:
        if self.model is not None:
            return 0.0
        logging.getLogger(__name__).info("MOSS · cargando runtime y pesos")
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        started = perf_counter()
        self.torch = torch
        self.device = self._device(torch)
        dtype_name = self.step.runtime.settings.get("dtype", "auto")
        if dtype_name == "auto":
            dtype_name = "bfloat16" if self.device.type == "cuda" else "float32"
        dtype = getattr(torch, str(dtype_name))
        location = str(self.prepared.model_path)
        model = (
            AutoModelForCausalLM.from_pretrained(
                location,
                trust_remote_code=True,
                local_files_only=True,
                dtype=dtype,
                attn_implementation=self.step.runtime.settings.get("attention", "sdpa"),
            )
            .to(self.device)
            .eval()
        )
        processor = AutoProcessor.from_pretrained(
            location, trust_remote_code=True, local_files_only=True
        )
        self.model = model
        self.processor = processor
        return perf_counter() - started

    def _device(self, torch):
        selected = str(self.step.runtime.settings.get("device", "auto"))
        if selected == "auto":
            if torch.cuda.is_available():
                selected = "cuda:0"
            else:
                selected = "cpu"
        device = torch.device(selected)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("No CUDA/ROCm device is available; select a CPU runtime.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("The selected MPS device is unavailable.")
        return device

    def _inputs(self, audio: AudioFile):
        import numpy as np

        with wave.open(str(audio.path), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
        samples = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio.path)},
                    {
                        "type": "text",
                        "text": self.step.model.settings.get("prompt", PROMPT),
                    },
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audio_kwargs = (
            {"device": str(self.device)} if self.device.type == "cuda" else {}
        )
        return self.processor(
            text=prompt, audio=[samples], return_tensors="pt", audio_kwargs=audio_kwargs
        ).to(self.device)

    def run(self, source: Artifact) -> StepOutput:
        if not isinstance(source, AudioFile) or source.duration is None:
            raise ValueError("MOSS needs a validated WAV audio artifact.")
        with tracer().start_as_current_span("scribnic.moss.load"):
            load_seconds = self._load()
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)
        include_gpu = self.step.runtime.settings.get("profile_gpu", False)
        with self._cpu_threads() as threads:
            self._describe_runtime(source, threads)
            with torch_profile(
                self.model, self.torch, self.device, include_gpu=include_gpu
            ) as profile:
                text, prompt_length, generated_tokens, timings = self._generate(source)
        transcript = parse_transcript(text)
        for segment in transcript.segments:
            if segment.end > source.duration + 0.1:
                raise ValueError("MOSS returned a timestamp beyond the audio duration.")
        metadata = {
            "repository": self.step.model.repository,
            "revision": self.step.model.revision,
            "device": str(self.device),
            "dtype": str(self.model.dtype),
            "load_seconds": load_seconds,
            "torch_version": str(self.torch.__version__),
            "transformers_version": version("transformers"),
            "cpu_threads": threads,
            "prompt_tokens": prompt_length,
            "generated_tokens": generated_tokens,
            "generation_timings": timings,
            "profile": profile,
        }
        if self.device.type == "cuda":
            metadata["gpu_memory_scope"] = "inference including resident model weights"
            metadata["peak_gpu_allocated_bytes"] = self.torch.cuda.max_memory_allocated(
                self.device
            )
            metadata["peak_gpu_reserved_bytes"] = self.torch.cuda.max_memory_reserved(
                self.device
            )
        return StepOutput(transcript, metadata)

    @contextmanager
    def _cpu_threads(self):
        previous = self.torch.get_num_threads()
        threads = previous
        if self.device.type == "cpu":
            threads = int(self.step.runtime.settings.get("threads", previous))
        try:
            if threads != previous:
                self.torch.set_num_threads(threads)
            yield threads
        finally:
            if threads != previous:
                self.torch.set_num_threads(previous)

    def _describe_runtime(self, source: AudioFile, threads: int) -> None:
        logger = logging.getLogger(__name__)
        logger.info(
            "MOSS · audio %.1f s · dispositivo %s · %s · CPU: %d hilos · Torch %s",
            source.duration,
            self.device,
            self.model.dtype,
            threads,
            self.torch.__version__,
        )
        if self.device.type == "cpu":
            logger.info(
                "MOSS · inferencia en CPU; un audio largo puede tardar varios minutos"
            )
            if "+cpu" in str(self.torch.__version__):
                logger.info("MOSS · este build de PyTorch es CPU; no utiliza la GPU")

    @staticmethod
    def _progress_criteria(prompt_length: int, progress: GenerationProgress):
        from transformers import StoppingCriteria

        class ReportProgress(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                progress.update(input_ids.shape[-1] - prompt_length)
                return False

        return [ReportProgress()]

    def _generate(self, source: AudioFile) -> tuple[str, int, int, dict]:
        logger = logging.getLogger(__name__)
        logger.info("MOSS · preparando audio")
        with self.torch.profiler.record_function("moss.frontend"):
            inputs = self._inputs(source)
        prompt_length = int(inputs["attention_mask"][0].sum().item())
        settings = self.step.model.settings
        maximum = int(settings.get("max_tokens", 16384))
        minimum = int(settings.get("min_tokens_budget", 2048))
        duration_budget = round(
            source.duration * float(settings.get("tokens_per_second", 12))
        )
        budget = min(maximum, max(minimum, duration_budget))
        convolution = nullcontext()
        if self.step.runtime.settings.get("cudnn_enabled") is False:
            convolution = self.torch.backends.cudnn.flags(enabled=False)
        progress = GenerationProgress("MOSS")
        logger.info("MOSS · encoder y prefill · %d tokens de entrada", prompt_length)
        with convolution, self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=budget,
                do_sample=False,
                stopping_criteria=self._progress_criteria(prompt_length, progress),
            )
        if output.shape[1] - prompt_length >= budget:
            raise ValueError(
                "MOSS reached its output limit; incomplete transcript rejected."
            )
        text = self.processor.tokenizer.decode(
            output[0][prompt_length:], skip_special_tokens=True
        ).strip()
        return (
            text,
            prompt_length,
            int(output.shape[1] - prompt_length),
            progress.timings(),
        )

    def process_ids(self) -> tuple[int, ...]:
        return ()

    def close(self) -> None:
        self.model = None
        self.processor = None
        if hasattr(self, "torch") and self.device is not None:
            if self.device.type == "cuda":
                self.torch.cuda.empty_cache()
            elif self.device.type == "mps":
                self.torch.mps.empty_cache()
