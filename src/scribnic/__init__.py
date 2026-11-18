"""Reusable local scribe workflows, independent of CLI and application servers."""

from importlib.metadata import version as _distribution_version

from .artifacts import (
    Artifact,
    ArtifactKind,
    AudioFile,
    DiarizedTranscript,
    Note,
    RunResult,
    SpeakerSegment,
    TextSegment,
    Transcript,
)
from .audio_stream import TranscriptUpdate, wav_chunks
from .config import (
    ModelAsset,
    ModelConfig,
    RuntimeConfig,
    StepConfig,
    WorkflowConfig,
    load_workflow,
)
from .storage import load_artifact, save_artifact
from .workflow import WorkflowRunError, WorkflowSession, open_workflow

# The version lives only in pyproject.toml; read it from the installed metadata.
__version__ = _distribution_version("scribnic")

__all__ = [
    "Artifact",
    "ArtifactKind",
    "AudioFile",
    "DiarizedTranscript",
    "ModelAsset",
    "ModelConfig",
    "Note",
    "RunResult",
    "RuntimeConfig",
    "SpeakerSegment",
    "StepConfig",
    "TextSegment",
    "Transcript",
    "TranscriptUpdate",
    "WorkflowConfig",
    "WorkflowRunError",
    "WorkflowSession",
    "load_artifact",
    "load_workflow",
    "open_workflow",
    "save_artifact",
    "wav_chunks",
]
