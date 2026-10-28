"""Validate incremental execution requests without owning provider algorithms."""

import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable
from uuid import uuid4

from .artifacts import ArtifactKind, StepOutput
from .audio_stream import PCMRecording, UpdateCallback
from .config import WorkflowConfig
from .downloads import cache_directory
from .planning import resolve_plan


@runtime_checkable
class StreamingBackend(Protocol):
    """An optional backend capability consuming one execution-owned PCM source."""

    def run_stream(self, source: PCMRecording) -> StepOutput:
        """Consume the input and return the step's declared final artifact."""
        ...


def prepare_recording(
    chunks: Iterable[bytes],
    config: WorkflowConfig,
    *,
    output: ArtifactKind,
    output_dir: str | Path | None,
    on_update: UpdateCallback | None,
    cancel: threading.Event | None,
) -> PCMRecording:
    """Validate stream support and select a durable recording path before input.

    Opening and closing the recording belong to the execution, after its output
    directory and telemetry have been initialized. No model or file is opened here.
    """
    plan = resolve_plan(config, "audio", output)
    if not plan:
        raise ValueError("Streaming needs a transcript or note output.")
    step = plan[0].step
    if (
        step.backend not in {"openai_transcription", "llama_audio_note"}
        or step.model.settings.get("mode") != "streaming"
    ):
        raise ValueError("The workflow must select a streaming audio step.")
    directory = (
        Path(output_dir) if output_dir is not None else cache_directory() / "logs"
    )
    return PCMRecording(
        chunks,
        directory.resolve() / "runtime" / f"recording-{uuid4().hex}.wav",
        on_update=on_update,
        cancel=cancel,
    )
