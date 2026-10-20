"""Public reusable workflow sessions, independent of CLI and evaluation tools."""

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Protocol
from uuid import uuid4

from .artifacts import (
    Artifact,
    ArtifactKind,
    AudioFile,
    RunResult,
    StepOutput,
    StepRecord,
)
from .config import StepConfig, WorkflowConfig, artifact_kind, load_workflow
from .planning import PlannedStep, resolve_plan, validate_workflow


class Backend(Protocol):
    def run(self, source: Artifact) -> StepOutput: ...

    def close(self) -> None: ...


BackendFactory = Callable[[StepConfig], Backend]


class WorkflowRunError(RuntimeError):
    """A failed execution retains every completed artifact."""

    def __init__(self, message: str, result: RunResult):
        super().__init__(message)
        self.result = result


class WorkflowSession:
    def __init__(
        self,
        config: WorkflowConfig,
        *,
        backend_factory: BackendFactory | None = None,
    ):
        validate_workflow(config)
        self.config = config
        self._factory = backend_factory
        self._backends: dict[str, Backend] = {}
        self._closed = False

    def __enter__(self) -> "WorkflowSession":
        if self._closed:
            raise RuntimeError("This workflow session has been closed.")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        for backend in reversed(tuple(self._backends.values())):
            backend.close()
        self._backends.clear()
        self._closed = True

    def _backend(self, step: StepConfig) -> Backend:
        if step.name not in self._backends:
            factory = self._factory
            if factory is None:
                from .backends import create_backend

                factory = create_backend
            self._backends[step.name] = factory(step)
        return self._backends[step.name]

    def run(
        self, source: Artifact, *, output: ArtifactKind | None = None
    ) -> RunResult:
        if self._closed:
            raise RuntimeError("This workflow session has been closed.")
        target = artifact_kind(output or self.config.default_output)
        plan = resolve_plan(self.config, source.kind, target)
        if isinstance(source, AudioFile):
            from .audio import inspect_audio

            source = inspect_audio(source)
        return self._execute(source, target, plan)

    def _execute(
        self, source: Artifact, target: ArtifactKind, plan: tuple[PlannedStep, ...]
    ) -> RunResult:
        started = perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        run_id = uuid4().hex
        artifacts = {source.kind: source}
        records = []
        try:
            for planned in plan:
                step_started = perf_counter()
                backend = self._backend(planned.step)
                produced = backend.run(artifacts[planned.input_kind])
                if produced.artifact.kind != planned.step.model.output:
                    raise ValueError(f"Backend {planned.step.name!r} returned the wrong artifact.")
                artifacts[produced.artifact.kind] = produced.artifact
                records.append(StepRecord(
                    planned.step.name,
                    planned.step.model.backend,
                    perf_counter() - step_started,
                    produced.metadata,
                ))
        except Exception as error:
            result = self._result(
                run_id, target, source, artifacts, records, started_at,
                perf_counter() - started, str(error),
            )
            raise WorkflowRunError(str(error), result) from error
        return self._result(
            run_id, target, source, artifacts, records, started_at,
            perf_counter() - started,
        )

    def _result(
        self, run_id, target, source, artifacts, records, started_at, elapsed, error=None
    ) -> RunResult:
        return RunResult(
            run_id=run_id,
            workflow=self.config.name,
            requested_output=target,
            source=source,
            artifacts=tuple(artifacts.values()),
            steps=tuple(records),
            configuration=self.config.to_dict(),
            started_at=started_at,
            elapsed_seconds=elapsed,
            status="error" if error else "success",
            error=error,
        )


def open_workflow(
    workflow: str | Path | WorkflowConfig,
    *,
    backend_factory: BackendFactory | None = None,
) -> WorkflowSession:
    """Load a required workflow; run defaults to its configured note output."""
    return WorkflowSession(load_workflow(workflow), backend_factory=backend_factory)
