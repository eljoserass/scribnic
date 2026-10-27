"""Offline audio-to-note inference using pinned, optional MOSS-Audio code."""

import importlib
import json
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import TYPE_CHECKING
from uuid import uuid4

import psutil

from ..artifacts import Artifact, AudioFile, Note, StepOutput
from ..audio import read_waveform
from ..config import StepConfig
from ..downloads import PreparedStep, sha256_file
from ..notes import note_instruction
from ..observability import tracer
from ..profiling import moss_audio_ranges, operator_profile
from ..storage import diagnostic_path

if TYPE_CHECKING:
    from torch import dtype as TorchDtype
    from transformers import BatchEncoding, PreTrainedModel


def _provider(source: Path) -> tuple[ModuleType, ModuleType]:
    """Import the pinned upstream package without installing its demo dependencies.

    Upstream uses the package name ``src``. Check its origin so an unrelated
    package or a different code revision cannot silently supply the implementation.
    Python's ordinary module cache retains code, never model instances.
    """
    sys.path.insert(0, str(source))
    try:
        modules = []
        for name in (
            "configuration_moss_audio",
            "modeling_moss_audio",
            "processing_moss_audio",
        ):
            module = importlib.import_module("src." + name)
            if not Path(module.__file__).resolve().is_relative_to(source.resolve()):
                raise ImportError(
                    "A different 'src' package is loaded; use a separate MOSS-Audio process."
                )
            modules.append(module)
    finally:
        sys.path.remove(str(source))
    return modules[1], modules[2]


def _final_note(text: str) -> str:
    """Keep the final answer separate from optional model reasoning."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    if "<think>" in text or not text.strip():
        raise ValueError("MOSS-Audio returned no complete final note.")
    return text.strip()


class MossAudio:
    """Own an optional model and processor until the workflow session closes."""

    def __init__(self, step: StepConfig, prepared: PreparedStep) -> None:
        """Retain configuration without importing Torch or loading model weights."""
        if prepared.model_path is None or prepared.source_path is None:
            raise ValueError(
                "MOSS-Audio needs prepared weights and pinned source code."
            )
        self.step = step
        self.prepared = prepared
        self.model: PreTrainedModel | None = None
        self.processor = None
        self.torch: ModuleType | None = None
        self.device = None

    def _memory_check(self, dtype: "TorchDtype") -> None:
        """Reject CPU/unified-memory loads whose weights alone exceed free memory."""
        if self.device.type == "cuda":
            return
        index = self.prepared.model_path / "model.safetensors.index.json"
        parameters = json.loads(index.read_text())["metadata"]["total_parameters"]
        minimum_bytes = parameters * self.torch.empty((), dtype=dtype).element_size()
        available = psutil.virtual_memory().available
        if minimum_bytes > available:
            raise ValueError(
                f"MOSS-Audio weights need at least {minimum_bytes / 2**30:.2f} GiB "
                f"at this dtype; {available / 2**30:.2f} GiB is available. "
                "Activations need additional memory. Use a larger machine or change dtype."
            )

    def _compute_settings(self) -> "TorchDtype":
        """Select an available device and return the configured floating-point dtype."""
        torch = self.torch
        selected = self.step.runtime.settings.get("device", "auto")
        if selected == "auto":
            selected = "cpu"
            if torch.cuda.is_available():
                selected = "cuda:0"
            elif torch.backends.mps.is_available():
                selected = "mps"
        self.device = torch.device(selected)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("The selected CUDA/ROCm device is unavailable.")
        if self.device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("The selected Apple MPS device is unavailable.")
        dtype_name = self.step.runtime.settings.get("dtype", "auto")
        if dtype_name == "auto":
            dtype_name = {"cuda": "bfloat16", "mps": "float16"}.get(
                self.device.type, "float32"
            )
        if dtype_name not in {"float32", "float16", "bfloat16"}:
            raise ValueError(
                "Select float32, float16, bfloat16 or auto for MOSS-Audio."
            )
        return getattr(torch, dtype_name)

    def start(self) -> float:
        """Load the pinned model once, choosing CPU, CUDA or Apple MPS explicitly."""
        if self.model is not None:
            return 0.0
        started = perf_counter()
        with tracer().start_as_current_span(
            "scribnic.model.load",
            attributes={"scribnic.model.name": self.step.model.name},
        ):
            if version("transformers") != "4.57.1":
                raise ImportError(
                    "Use scribnic[moss-audio] in its separate environment."
                )
            import torch

            self.torch = torch
            dtype = self._compute_settings()
            self._memory_check(dtype)
            model_code, processor_code = _provider(self.prepared.source_path)
            model = (
                model_code.MossAudioModel.from_pretrained(
                    str(self.prepared.model_path),
                    local_files_only=True,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                )
                .to(self.device)
                .eval()
            )
            processor = processor_code.MossAudioProcessor.from_pretrained(
                str(self.prepared.model_path),
                local_files_only=True,
                enable_time_marker=True,
            )
            self.model, self.processor = model, processor
        return perf_counter() - started

    def _inputs(self, source: AudioFile) -> "BatchEncoding":
        """Prepare the complete recording and the workflow's audio-note instruction."""
        samples = read_waveform(source.path)
        inputs = self.processor(
            text=note_instruction(self.step), audios=[samples], return_tensors="pt"
        ).to(self.device)
        inputs["audio_data"] = inputs["audio_data"].to(self.model.dtype)
        inputs["audio_input_mask"] = (
            inputs["input_ids"] == self.processor.audio_token_id
        )
        return inputs

    def _generate(self, source: AudioFile) -> tuple[str, int, int]:
        """Run offline generation and reject context overflow or incomplete output."""
        with self.torch.profiler.record_function("moss_audio.frontend"):
            inputs = self._inputs(source)
        prompt_tokens = inputs["input_ids"].shape[-1]
        settings = self.step.model.settings
        maximum = int(settings.get("max_tokens", 2048))
        context = self.model.config.language_config.max_position_embeddings
        if prompt_tokens + maximum > context:
            raise ValueError(
                "Audio and note budget exceed MOSS-Audio context; nothing was truncated."
            )
        temperature = float(settings.get("temperature", 0))
        generation = {
            "max_new_tokens": maximum,
            "do_sample": temperature > 0,
            "use_cache": True,
        }
        if temperature > 0:
            generation.update(
                temperature=temperature,
                top_p=float(settings.get("top_p", 1)),
                top_k=int(settings.get("top_k", 50)),
            )
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation)
        tokens = output[0, prompt_tokens:]
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        if len(tokens) >= maximum and int(tokens[-1]) not in eos_ids:
            raise ValueError(
                "MOSS-Audio reached its output limit; incomplete note rejected."
            )
        text = self.processor.decode(tokens, skip_special_tokens=True)
        return text, prompt_tokens, len(tokens)

    def _inference(self, source: AudioFile) -> tuple[str, int, int, dict]:
        """Scope thread tuning and bounded operator traces to this inference call."""
        torch = self.torch
        previous_threads = torch.get_num_threads()
        settings = self.step.runtime.settings
        try:
            if self.device.type == "cpu":
                torch.set_num_threads(int(settings.get("threads", 6)))
            with operator_profile(
                torch,
                self.device,
                name="moss_audio",
                phase_prefix="moss_audio.",
                scope="MOSS-Audio frontend, encoder, adapter and language model; loading excluded",
                include_gpu=bool(settings.get("profile_gpu", False)),
                window_steps=int(settings.get("profile_window_steps", 16)),
            ) as capture:
                with moss_audio_ranges(self.model, torch, capture.advance):
                    text, prompt_tokens, generated_tokens = self._generate(source)
            return text, prompt_tokens, generated_tokens, capture.metadata
        finally:
            if torch.get_num_threads() != previous_threads:
                torch.set_num_threads(previous_threads)

    def run(self, source: Artifact) -> StepOutput:
        """Generate one note directly from audio, persisting raw output and profiling."""
        if not isinstance(source, AudioFile) or source.duration is None:
            raise ValueError("MOSS-Audio needs a validated audio artifact.")
        load_seconds = self.start()
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)
        started = perf_counter()
        with tracer().start_as_current_span(
            "scribnic.note.request",
            attributes={"scribnic.model.name": self.step.model.name},
        ):
            text, prompt_tokens, generated_tokens, profile = self._inference(source)
        path = diagnostic_path(
            "moss-audio-" + uuid4().hex + ".txt", cache=self.prepared.cache
        )
        path.write_text(text, encoding="utf-8")
        metadata = {
            "repository": self.step.model.repository,
            "revision": self.step.model.revision,
            "weight_format": self.step.model.quantization,
            "dtype": str(self.model.dtype),
            "device": str(self.device),
            "load_seconds": load_seconds,
            "request_seconds": perf_counter() - started,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "torch_version": str(self.torch.__version__),
            "transformers_version": version("transformers"),
            "source_revision": self.step.runtime.settings["release"],
            "source_archive_sha256": self.step.runtime.assets["source"]["sha256"],
            "source_files_sha256": {
                name: sha256_file(self.prepared.source_path / name)
                for name in self.step.runtime.settings["required_source_files"]
            },
            "weights_index_sha256": sha256_file(
                self.prepared.model_path / "model.safetensors.index.json"
            ),
            "native_output_path": str(path),
            "profile": profile,
        }
        if self.device.type == "cuda":
            metadata["peak_gpu_allocated_bytes"] = self.torch.cuda.max_memory_allocated(
                self.device
            )
            metadata["peak_gpu_reserved_bytes"] = self.torch.cuda.max_memory_reserved(
                self.device
            )
        return StepOutput(Note(_final_note(text)), metadata)

    def process_ids(self) -> tuple[int, ...]:
        """Inference runs inside the sampled application process."""
        return ()

    def close(self) -> None:
        """Release owned weights/processor and accelerator allocation caches."""
        self.model = self.processor = None
        if self.torch is not None and self.device is not None:
            if self.device.type == "cuda" and self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
            elif self.device.type == "mps" and self.torch.backends.mps.is_available():
                self.torch.mps.empty_cache()
