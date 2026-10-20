"""Small TOML configuration objects and deterministic preset discovery."""

import json
import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .artifacts import ARTIFACT_KINDS, ArtifactKind

PRESETS = Path(__file__).parent / "presets"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    backend: str
    inputs: tuple[ArtifactKind, ...]
    output: ArtifactKind
    repository: str | None = None
    revision: str | None = None
    filename: str | None = None
    sha256: str | None = None
    settings: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeConfig:
    name: str
    engine: str
    settings: dict[str, object] = field(default_factory=dict)
    assets: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class StepConfig:
    name: str
    model: ModelConfig
    runtime: RuntimeConfig
    template: Path | None = None
    source: str | None = None
    instruction: str | None = None


@dataclass(frozen=True)
class WorkflowConfig:
    name: str
    steps: tuple[StepConfig, ...]
    inputs: tuple[ArtifactKind, ...] = ("audio", "transcript", "diarized_transcript")
    default_output: ArtifactKind = "note"
    download_missing: bool = True

    def to_dict(self) -> dict[str, object]:
        return json.loads(json.dumps(asdict(self), default=str))


def artifact_kind(value: str) -> ArtifactKind:
    if value not in ARTIFACT_KINDS:
        raise ValueError(f"Unknown artifact kind: {value!r}")
    return value


def validate_settings(step: StepConfig) -> None:
    """Reject invalid numeric settings before preparing expensive resources."""
    positive = (
        "max_tokens",
        "max_input_chars",
        "min_tokens_budget",
        "tokens_per_second",
        "context_size",
        "threads",
        "threads_batch",
        "timeout",
        "startup_timeout",
    )
    for settings in (step.model.settings, step.runtime.settings):
        for key in positive:
            if key not in settings:
                continue
            value = settings[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{step.name}.{key} must be a positive number.")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{step.name}.{key} must be a positive number.")
    for key in ("top_p", "min_p"):
        value = step.model.settings.get(key, 0)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{step.name}.{key} must be between 0 and 1.")


def _read_toml(path: Path) -> dict:
    with path.open("rb") as source:
        return tomllib.load(source)


def discover_workflow(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        default = candidate / "default.toml"
        if default.is_file():
            return default.resolve()
        choices = sorted(candidate.glob("*.toml"))
        if len(choices) != 1:
            raise ValueError(
                f"Expected one workflow or default.toml in {candidate}; "
                f"found {len(choices)}. Select a workflow file explicitly."
            )
        return choices[0].resolve()
    if candidate.is_file():
        return candidate.resolve()
    preset = PRESETS / "workflows" / f"{path}.toml"
    if preset.is_file():
        return preset
    raise ValueError(f"Workflow not found: {path}")


def _preset_path(name: str, category: str, directory: Path) -> Path:
    candidate = directory / name
    if candidate.is_file():
        return candidate.resolve()
    suffix = ".md" if category == "notes" else ".toml"
    filename = name if name.endswith(suffix) else name + suffix
    preset = PRESETS / category / filename
    if not preset.is_file():
        raise ValueError(f"Unknown {category} preset or file: {name}")
    return preset


def _load_model(name: str, directory: Path, overrides: dict) -> ModelConfig:
    data = _read_toml(_preset_path(name, "models", directory))
    settings = data.get("settings", {}) | overrides
    return ModelConfig(
        name=data.get("name", name),
        backend=data["backend"],
        inputs=tuple(artifact_kind(kind) for kind in data["inputs"]),
        output=artifact_kind(data["output"]),
        repository=data.get("repository"),
        revision=data.get("revision"),
        filename=data.get("filename"),
        sha256=data.get("sha256"),
        settings=settings,
    )


def _load_runtime(name: str, directory: Path, overrides: dict) -> RuntimeConfig:
    data = _read_toml(_preset_path(name, "runtimes", directory))
    return RuntimeConfig(
        name=data.get("name", name),
        engine=data["engine"],
        settings=data.get("settings", {}) | overrides,
        assets=data.get("assets", {}),
    )


def _load_step(name: str, data: dict, directory: Path) -> StepConfig:
    template = None
    if data.get("template"):
        template = _preset_path(data["template"], "notes", directory)
    return StepConfig(
        name=name,
        model=_load_model(data["model"], directory, data.get("model_settings", {})),
        runtime=_load_runtime(
            data["runtime"], directory, data.get("runtime_settings", {})
        ),
        template=template,
        source=data.get("source"),
        instruction=template.read_text(encoding="utf-8").strip() if template else None,
    )


def load_workflow(path: str | Path | WorkflowConfig) -> WorkflowConfig:
    if isinstance(path, WorkflowConfig):
        return path
    location = discover_workflow(path)
    data = _read_toml(location)
    try:
        steps = tuple(
            _load_step(name, step, location.parent)
            for name, step in data["steps"].items()
        )
        inputs = data.get("inputs", ["audio", "transcript", "diarized_transcript"])
        return WorkflowConfig(
            name=data.get("name", location.stem),
            steps=steps,
            inputs=tuple(artifact_kind(kind) for kind in inputs),
            default_output=artifact_kind(data.get("default_output", "note")),
            download_missing=data.get("download_missing", True),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Invalid workflow configuration in {location}: {error}"
        ) from error
