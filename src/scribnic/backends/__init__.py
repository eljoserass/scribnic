"""Construct only the backend selected by a workflow step."""

from typing import TYPE_CHECKING

from ..config import StepConfig
from ..downloads import PreparedStep

if TYPE_CHECKING:
    from ..workflow import Backend


def create_backend(step: StepConfig, prepared: PreparedStep) -> "Backend":
    """Import only the protocol adapter explicitly selected by this workflow step."""
    from importlib import import_module

    adapters = {
        "openai_chat": ("openai_chat", "OpenAIChat"),
        "openai_transcription": ("openai_transcription", "OpenAITranscription"),
        "moss_transcribe_diarize": ("moss_transcribe_diarize", "MossTranscribeDiarize"),
        "moss_audio": ("moss_audio", "MossAudio"),
        "llama_audio_note": ("llama_audio_note", "LlamaAudioNote"),
    }
    if step.backend not in adapters:
        raise ValueError(f"Unknown workflow backend: {step.backend}")
    module_name, class_name = adapters[step.backend]
    module = import_module("." + module_name, package=__package__)
    return getattr(module, class_name)(step, prepared)
