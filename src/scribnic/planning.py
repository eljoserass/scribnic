"""Resolve a requested artifact using only declared workflow producers."""

from dataclasses import dataclass

from .artifacts import ArtifactKind
from .config import StepConfig, WorkflowConfig


@dataclass(frozen=True)
class PlannedStep:
    step: StepConfig
    input_kind: ArtifactKind


def _producers(config: WorkflowConfig, target: ArtifactKind) -> list[StepConfig]:
    return [step for step in config.steps if step.model.output == target]


def _upstream(config: WorkflowConfig, step: StepConfig) -> StepConfig:
    candidates = [
        producer for producer in config.steps
        if producer.model.output in step.model.inputs
    ]
    if step.source is not None:
        candidates = [item for item in candidates if item.name == step.source]
    if not candidates:
        raise ValueError(f"No configured producer can provide input for {step.name!r}.")
    if len(candidates) != 1:
        names = ", ".join(item.name for item in candidates)
        raise ValueError(
            f"Ambiguous input for {step.name!r}: {names}. Set its source explicitly."
        )
    return candidates[0]


def resolve_plan(
    config: WorkflowConfig, source: ArtifactKind, target: ArtifactKind
) -> tuple[PlannedStep, ...]:
    """Supplied input satisfies a dependency before any model is considered."""
    if source not in config.inputs:
        raise ValueError(f"Workflow {config.name!r} does not accept {source!r} input.")
    if source == target:
        return ()
    producers = _producers(config, target)
    if not producers:
        raise ValueError(f"Workflow {config.name!r} does not produce {target!r}.")
    if len(producers) != 1:
        raise ValueError(f"Ambiguous producers for {target!r} in {config.name!r}.")
    return _resolve_step(config, producers[0], source, ())


def _resolve_step(
    config: WorkflowConfig,
    step: StepConfig,
    source: ArtifactKind,
    visiting: tuple[str, ...],
) -> tuple[PlannedStep, ...]:
    if step.name in visiting:
        chain = " -> ".join((*visiting, step.name))
        raise ValueError(f"Workflow dependency cycle: {chain}")
    if source in step.model.inputs:
        return (PlannedStep(step, source),)
    upstream = _upstream(config, step)
    previous = _resolve_step(config, upstream, source, (*visiting, step.name))
    return (*previous, PlannedStep(step, upstream.model.output))


def validate_workflow(config: WorkflowConfig) -> None:
    if not config.steps:
        raise ValueError("A workflow needs at least one step.")
    names = [step.name for step in config.steps]
    if len(names) != len(set(names)):
        raise ValueError("Workflow step names must be unique.")
    for step in config.steps:
        if not step.model.inputs:
            raise ValueError(f"Step {step.name!r} declares no accepted inputs.")
        if step.source is not None:
            _upstream(config, step)
        _validate_cycles(config, step, ())
    for source in config.inputs:
        resolve_plan(config, source, config.default_output)


def _validate_cycles(
    config: WorkflowConfig, step: StepConfig, visiting: tuple[str, ...]
) -> None:
    if step.name in visiting:
        raise ValueError(f"Workflow dependency cycle involving {step.name!r}.")
    parents = [
        item for item in config.steps
        if item.model.output in step.model.inputs
        and (step.source is None or item.name == step.source)
    ]
    for parent in parents:
        _validate_cycles(config, parent, (*visiting, step.name))
