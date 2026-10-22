"""Construct only the backend selected by a workflow step."""

from ..config import StepConfig
from ..downloads import PreparedStep


def create_backend(step: StepConfig, prepared: PreparedStep):
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
    if step.model.backend == "demo":
        from .demo import DemoBackend

        return DemoBackend(step)
    raise ValueError(f"Unknown workflow backend: {step.model.backend}")
