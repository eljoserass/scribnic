"""Application entry points: saved runs and local telemetry around the core API."""

from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import TextIO
from uuid import uuid4

from .artifacts import ArtifactKind, AudioFile, RunResult
from .audio import inspect_audio
from .audio_stream import wav_chunks
from .config import WorkflowConfig, load_workflow
from .observability import Telemetry
from .planning import resolve_plan, validate_workflow
from .progress import phase
from .storage import load_artifact, write_json
from .workflow import open_workflow


def new_run_directory() -> Path:
    """Choose a unique local directory name without creating it."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("runs") / f"{stamp}-{uuid4().hex[:8]}"


def setup_workflow(
    workflow: str | Path | WorkflowConfig, *, offline: bool = False
) -> None:
    """Populate/check the configured asset cache without loading models into memory."""
    with phase("Preparar workflow", detail=str(workflow)):
        with open_workflow(workflow, download=not offline):
            pass


def execute(
    workflow: str | Path | WorkflowConfig,
    input_path: str | Path,
    directory: str | Path,
    *,
    output: ArtifactKind | None = None,
    offline: bool = False,
    profile: bool = False,
    progress: TextIO | None = None,
    realtime: bool = False,
) -> RunResult:
    """Run from a file and save artifacts, a report, and local OTLP files.

    Args:
        realtime: Pace WAV fragments at source time. Requires a streaming workflow;
            inference still runs as soon as the configured input window is ready.
    """
    with phase("Validar entrada y workflow"):
        config = load_workflow(workflow)
        validate_workflow(config)
        source = load_artifact(input_path)
        plan = resolve_plan(config, source.kind, output or config.default_output)
        if realtime and (
            not isinstance(source, AudioFile)
            or not plan
            or plan[0].step.model.settings.get("mode") != "streaming"
        ):
            raise ValueError("Realtime replay needs WAV and a streaming workflow.")
        if isinstance(source, AudioFile):
            source = inspect_audio(source)
        directory = Path(directory).expanduser().resolve()
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Output directory must be empty: {directory}")
    with Telemetry(directory / "telemetry", progress=progress):
        write_json(
            directory / "telemetry" / "invocation.json",
            {
                "workflow": config.name,
                "input_path": str(Path(input_path).expanduser().resolve()),
                "realtime_replay": realtime,
                "profile": profile,
            },
        )
        with open_workflow(config, download=not offline) as session:
            if realtime:
                with closing(wav_chunks(source.path, realtime=True)) as chunks:
                    return session.run_stream(
                        chunks,
                        output=output,
                        output_dir=directory,
                        profile=profile,
                    )
            return session.run(
                source, output=output, output_dir=directory, profile=profile
            )
