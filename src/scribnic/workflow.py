"""Public workflow sessions and isolated, checkpointed executions."""

import logging
import threading
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Protocol
from uuid import uuid4

from .artifacts import Artifact, ArtifactKind, AudioFile, RunResult, StepOutput, StepRecord
from .config import StepConfig, WorkflowConfig, artifact_kind, load_workflow
from .observability import trace_id, tracer
from .planning import PlannedStep, resolve_plan, validate_workflow
from .profiling import profile_directory
from .resources import ResourceMonitor


class Backend(Protocol):
    def run(self, source: Artifact) -> StepOutput: ...

    def process_ids(self) -> tuple[int, ...]: ...

    def close(self) -> None: ...


BackendFactory = Callable[[StepConfig], Backend]


class WorkflowRunError(RuntimeError):
    """A failed execution retains every completed artifact."""

    def __init__(self, message: str, result: RunResult):
        super().__init__(message)
        self.result = result


@dataclass
class _Execution:
    config: WorkflowConfig
    source: Artifact
    target: ArtifactKind
    run_id: str = field(default_factory=lambda: uuid4().hex)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started: float = field(default_factory=perf_counter)
    artifacts: dict[ArtifactKind, Artifact] = field(default_factory=dict)
    records: list[StepRecord] = field(default_factory=list)
    resources: dict[str, object] = field(default_factory=dict)
    trace_id: str | None = None

    def snapshot(self, status: str = "running", error: str | None = None) -> RunResult:
        return RunResult(
            run_id=self.run_id, workflow=self.config.name, requested_output=self.target,
            source=self.source, artifacts=tuple(self.artifacts.values()),
            steps=tuple(self.records), configuration=self.config.to_dict(),
            started_at=self.started_at, elapsed_seconds=perf_counter() - self.started,
            status=status, error=error, trace_id=self.trace_id, resources=self.resources,
        )


class WorkflowSession:
    def __init__(
        self, config: WorkflowConfig, *, backend_factory: BackendFactory | None = None,
        prepare: bool = True, download: bool | None = None, cache: Path | None = None,
    ):
        from .downloads import ResourceManager

        validate_workflow(config)
        self.config = config
        self._factory = backend_factory
        self._resources = ResourceManager(
            cache=cache, download=config.download_missing if download is None else download
        )
        self._cleanup = ExitStack()
        self._backends: dict[str, Backend] = {}
        self._closed = False
        self._lock = threading.Lock()
        if prepare and backend_factory is None:
            for step in config.steps:
                self._resources.prepare(step)

    def __enter__(self) -> "WorkflowSession":
        if self._closed:
            raise RuntimeError("This workflow session has been closed.")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            try:
                self._cleanup.close()
            finally:
                self._backends.clear()
                self._closed = True

    def _backend(self, step: StepConfig) -> Backend:
        if step.name not in self._backends:
            if self._factory is None:
                from .backends import create_backend

                backend = create_backend(step, self._resources.prepare(step))
            else:
                backend = self._factory(step)
            self._backends[step.name] = backend
            self._cleanup.callback(backend.close)
        return self._backends[step.name]

    def _runtime_pids(self) -> tuple[int, ...]:
        identifiers = []
        for backend in tuple(self._backends.values()):
            identifiers.extend(backend.process_ids())
        return tuple(identifiers)

    def run(
        self, source: Artifact, *, output: ArtifactKind | None = None,
        output_dir: str | Path | None = None, profile: bool = False,
    ) -> RunResult:
        """Produce the requested artifact; supplied transcripts bypass speech."""
        with self._lock:
            if self._closed:
                raise RuntimeError("This workflow session has been closed.")
            target = artifact_kind(output or self.config.default_output)
            plan = resolve_plan(self.config, source.kind, target)
            if profile and output_dir is None:
                raise ValueError("Profiling needs an output_dir for its trace artifacts.")
            from .storage import RunStore

            store = RunStore(output_dir) if output_dir is not None else None
            state = _Execution(self.config, source, target)
            attributes = {
                "scribnic.run.id": state.run_id,
                "scribnic.workflow.name": self.config.name,
                "scribnic.input.kind": source.kind,
                "scribnic.output.kind": target,
            }
            with tracer().start_as_current_span("scribnic.workflow.run", attributes=attributes):
                return self._execute(state, plan, store, profile)

    def _execute(self, state: _Execution, plan, store, profile: bool) -> RunResult:
        state.trace_id = trace_id()
        monitor = ResourceMonitor(state.run_id, self._runtime_pids)
        profiles = store.directory / "profiles" if profile else None
        try:
            with monitor, profile_directory(profiles):
                self._validate_source(state)
                if store is not None:
                    store.save(state.snapshot())
                self._prepare_backends(plan)
                self._run_steps(state, plan, store)
        except (Exception, KeyboardInterrupt) as error:
            state.resources = monitor.summary()
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
            result = state.snapshot(status, str(error) or status)
            logging.getLogger(__name__).error("Workflow failed: %s", error)
            self._save_failure(store, result, error)
            if isinstance(error, KeyboardInterrupt):
                raise
            raise WorkflowRunError(str(error), result) from error
        state.resources = monitor.summary()
        result = state.snapshot("success")
        if store is not None:
            store.save(result)
        return result

    @staticmethod
    def _validate_source(state: _Execution) -> None:
        if isinstance(state.source, AudioFile):
            from .audio import inspect_audio

            with tracer().start_as_current_span("scribnic.audio.validate"):
                state.source = inspect_audio(state.source)
        state.artifacts[state.source.kind] = state.source

    def _prepare_backends(self, plan: tuple[PlannedStep, ...]) -> None:
        # Preflight reachable services before loading speech model weights.
        for planned in plan:
            with tracer().start_as_current_span(
                "scribnic.backend.prepare",
                attributes={"scribnic.step.name": planned.step.name},
            ):
                self._backend(planned.step)

    def _run_steps(self, state: _Execution, plan, store) -> None:
        for planned in plan:
            attributes = {
                "scribnic.step.name": planned.step.name,
                "scribnic.model.name": planned.step.model.name,
                "scribnic.runtime.engine": planned.step.runtime.engine,
            }
            started = perf_counter()
            with tracer().start_as_current_span("scribnic.step.run", attributes=attributes):
                produced = self._backend(planned.step).run(state.artifacts[planned.input_kind])
                if produced.artifact.kind != planned.step.model.output:
                    raise ValueError(f"Backend {planned.step.name!r} returned the wrong artifact.")
            state.artifacts[produced.artifact.kind] = produced.artifact
            state.records.append(StepRecord(
                planned.step.name, planned.step.model.backend,
                perf_counter() - started, produced.metadata,
            ))
            if store is not None:
                store.save(state.snapshot())

    @staticmethod
    def _save_failure(store, result: RunResult, original: BaseException) -> None:
        if store is None:
            return
        try:
            store.save(result)
        except OSError as error:
            original.add_note(f"Could not persist the final checkpoint: {error}")


def open_workflow(
    workflow: str | Path | WorkflowConfig, *, backend_factory: BackendFactory | None = None,
    prepare: bool = True, download: bool | None = None, cache: Path | None = None,
) -> WorkflowSession:
    """Select a workflow and prepare cached resources without loading model weights."""
    return WorkflowSession(
        load_workflow(workflow), backend_factory=backend_factory,
        prepare=prepare, download=download, cache=cache,
    )
