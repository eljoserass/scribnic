"""Choose the small set of supported model implementations."""

import os
import tempfile
from pathlib import Path

from .core import Pipeline


def build_pipeline(
    *, demo: bool = False, device: str = "auto", language: str = "es-ES",
    asr: str = "nemotron", diarizer: str = "nemotron",
    llm_model: str | None = None, base_url: str = "http://localhost:1234/v1",
    skip_generation: bool = False,
) -> Pipeline:
    if demo:
        from .backends.demo import DemoDiarizer, DemoTextGenerator, DemoTranscriber

        return Pipeline(DemoTranscriber(), DemoDiarizer(),
                        None if skip_generation else DemoTextGenerator())
    if asr not in ("nemotron", "canary") or diarizer not in ("nemotron", "sortformer"):
        raise ValueError("Unknown ASR or diarizer implementation.")

    # NeMo unpacks large model archives. Avoid a small /tmp tmpfs on WSL.
    if (asr == "canary" or diarizer == "sortformer") and "TMPDIR" not in os.environ:
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

    from .backends.asr import CanaryTranscriber, NemotronTranscriber
    from .backends.diarization import NemotronDiarizer, SortformerDiarizer
    from .backends.generation import LMStudioTextGenerator

    generator = (LMStudioTextGenerator(llm_model, base_url)
                 if llm_model and not skip_generation else None)
    transcriber = (NemotronTranscriber(device, language) if asr == "nemotron"
                   else CanaryTranscriber(device, language))
    diarization = NemotronDiarizer(device) if diarizer == "nemotron" else SortformerDiarizer(device)
    return Pipeline(transcriber, diarization, generator)
