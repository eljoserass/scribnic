"""Construct only the backend selected by a workflow step."""

from ..config import StepConfig
from ..downloads import PreparedStep


def create_backend(step: StepConfig, prepared: PreparedStep):
    """Construct the selected adapter without importing unrelated model libraries."""
    if step.model.backend == "nvidia_speech":
        if step.runtime.engine == "nemo_cpp":
            from .nemo_cpp import NemoCPP

            return NemoCPP(step, prepared)
        if step.runtime.engine == "nemo_python":
            from .nemo import NemoSpeech

            return NemoSpeech(step, prepared)
        raise ValueError("NVIDIA speech needs nemo_cpp or nemo_python.")
    if step.model.backend == "whisper_cpp":
        from .whisper_cpp import WhisperCPP

        return WhisperCPP(step, prepared)
    if step.model.backend == "http_generation":
        from .http_generation import HTTPGeneration

        return HTTPGeneration(step, prepared)
    if step.model.backend == "moss_transcribe_diarize":
        from .moss_transcribe_diarize import MossTranscribeDiarize

        if step.runtime.engine != "transformers":
            raise ValueError(
                "MOSS-Transcribe-Diarize currently uses the Transformers runtime."
            )
        return MossTranscribeDiarize(step, prepared)
    raise ValueError(f"Unknown workflow backend: {step.model.backend}")
