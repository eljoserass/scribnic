"""Construct a pipeline from model registries without importing unused backends."""

from importlib import import_module
import os
import tempfile
from pathlib import Path

from .core import Pipeline

ASR_MODELS = {
    "nemotron": ("asr", "NemotronTranscriber", False),
    "canary": ("asr", "CanaryTranscriber", True),
    "qwen": ("asr", "QwenTranscriber", False),
}
DIARIZERS = {
    "nemotron": ("diarization", "NemotronDiarizer", False),
    "sortformer": ("diarization", "SortformerDiarizer", True),
}
JOINT_MODELS = {
    "moss": ("joint", "MossRecognizer", False),
}


def _backend(spec: tuple[str, str, bool]):
    module, name, _ = spec
    return getattr(import_module(f".backends.{module}", package=__package__), name)


def build_pipeline(
    *, demo: bool = False, device: str = "auto", language: str = "es-ES",
    asr: str = "nemotron", diarizer: str = "nemotron",
    model: str | None = None,
    llm_model: str | None = None, base_url: str = "http://localhost:1234/v1",
    skip_generation: bool = False,
) -> Pipeline:
    if demo:
        from .backends.demo import DemoDiarizer, DemoTextGenerator, DemoTranscriber

        return Pipeline(DemoTranscriber(), DemoDiarizer(),
                        None if skip_generation else DemoTextGenerator())
    if asr not in ASR_MODELS or diarizer not in DIARIZERS or (model and model not in JOINT_MODELS):
        raise ValueError("Unknown model implementation.")

    # NeMo unpacks large model archives. Avoid a small /tmp tmpfs on WSL.
    if not model and (ASR_MODELS[asr][2] or DIARIZERS[diarizer][2]) and "TMPDIR" not in os.environ:
        cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        model_tmpdir = cache_root / "scribnic" / "tmp"
        model_tmpdir.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(model_tmpdir)
        tempfile.tempdir = str(model_tmpdir)

    import torch

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        try:
            device = str(torch.device(device))
        except RuntimeError as exc:
            raise ValueError(f"Invalid device: {device}") from exc
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("No CUDA or ROCm GPU is available; use --device cpu.")
    if torch.version.hip is not None and device.startswith("cuda"):
        torch.backends.cudnn.enabled = False

    from .backends.generation import LMStudioTextGenerator

    generator = (LMStudioTextGenerator(llm_model, base_url)
                 if llm_model and not skip_generation else None)
    if model:
        return Pipeline(generator=generator, recognizer=_backend(JOINT_MODELS[model])(device))
    return Pipeline(_backend(ASR_MODELS[asr])(device, language),
                    _backend(DIARIZERS[diarizer])(device), generator)
