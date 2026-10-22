"""Small TOML configuration objects and deterministic preset discovery."""

import json
import math
import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .artifacts import ARTIFACT_KINDS, ArtifactKind

PRESETS = Path(__file__).parent / "presets"


@dataclass(frozen=True)
class ModelAsset:
    """A pinned companion checkpoint, without a standalone artifact contract."""

    name: str
    repository: str
    revision: str
    filename: str
    sha256: str


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
    companions: dict[str, ModelAsset] = field(default_factory=dict)


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


def _validate_number(step, key, value, *, integer: bool, allow_zero: bool) -> None:
    label = f"{step.name}.{key}"
    if integer and type(value) is not int:
        raise ValueError(f"{label} must be an integer.")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number.")
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(
            f"{label} must be {'nonnegative' if allow_zero else 'positive'}."
        )


def validate_settings(step: StepConfig) -> None:
    """Reject invalid settings before preparing expensive resources."""
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
        "profile_window_steps",
        "beam_size",
        "best_of",
        "vad_min_speech_duration_ms",
        "vad_min_silence_duration_ms",
        "vad_max_speech_duration_s",
    )
    nonnegative = ("vad_speech_pad_ms", "vad_samples_overlap")
    integers = {
        "threads",
        "threads_batch",
        "profile_window_steps",
        "beam_size",
        "best_of",
        "vad_min_speech_duration_ms",
        "vad_min_silence_duration_ms",
        "vad_speech_pad_ms",
    }
    for settings in (step.model.settings, step.runtime.settings):
        for key in (*positive, *nonnegative):
            if key not in settings:
                continue
            _validate_number(
                step,
                key,
                settings[key],
                integer=key in integers,
                allow_zero=key in nonnegative,
            )
    for key in ("top_p", "min_p", "vad_threshold"):
        value = step.model.settings.get(key, 0)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{step.name}.{key} must be between 0 and 1.")
    for key in ("vad", "word_timestamps"):
        if key in step.model.settings and type(step.model.settings[key]) is not bool:
            raise ValueError(f"{step.name}.{key} must be true or false.")
    if step.model.backend == "nvidia_speech":
        _validate_nvidia(step)


def _validate_nvidia(step: StepConfig) -> None:
    """Check NVIDIA capability and mode choices before downloading weights."""
    if step.runtime.engine not in {"nemo_cpp", "nemo_python"}:
        raise ValueError("NVIDIA speech requires nemo_cpp or nemo_python.")
    if step.model.settings.get("mode", "offline") not in {"offline", "streaming"}:
        raise ValueError("NVIDIA speech mode must be offline or streaming.")
    if step.model.output not in {"transcript", "diarized_transcript"}:
        raise ValueError("NVIDIA speech produces transcript or diarized_transcript.")
    if (
        step.model.output == "diarized_transcript"
        and "diarization" not in step.companions
    ):
        raise ValueError("Diarized NVIDIA speech needs a diarization companion model.")
    maximum = step.model.settings.get("max_speakers", 8)
    if type(maximum) is not int or not 1 <= maximum <= 8:
        raise ValueError("max_speakers must be an integer between 1 and 8.")
    if step.runtime.engine == "nemo_cpp" and maximum != 8:
        raise ValueError(
            "NeMo-Speech.cpp uses the model's fixed eight-speaker capacity; max_speakers is adjustable only in nemo_python."
        )
    if (
        step.runtime.engine == "nemo_python"
        and step.model.settings.get("mode") == "streaming"
        and step.model.output != "diarized_transcript"
    ):
        raise ValueError(
            "NeMo masked streaming needs diarized_transcript output and a diarization companion."
        )
    suffix = ".gguf" if step.runtime.engine == "nemo_cpp" else ".nemo"
    for model in (step.model, *step.companions.values()):
        if not model.filename or not model.filename.endswith(suffix):
            raise ValueError(f"{step.runtime.engine} requires {suffix} model files.")
    options = step.model.settings.get("streaming", {})
    allowed = {
        "fifo_len",
        "spkcache_len",
        "spkcache_update_period",
        "cache_gating",
        "cache_gating_buffer_size",
        "binary_diar_preds",
        "fix_prev_words_count",
        "update_prev_words_sentence",
        "ignored_initial_frame_steps",
    }
    if not isinstance(options, dict) or options.keys() - allowed:
        raise ValueError(
            "Unknown NeMo streaming settings; mask strategy and input paths are fixed by the adapter."
        )
    for key, value in options.items():
        if key in {"cache_gating", "binary_diar_preds"}:
            if type(value) is not bool:
                raise ValueError(f"{key} must be true or false.")
        else:
            _validate_number(step, key, value, integer=True, allow_zero=False)
    gap = step.model.settings.get("sentence_gap_seconds", 1.0)
    _validate_number(step, "sentence_gap_seconds", gap, integer=False, allow_zero=False)


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


def _load_asset(name: str, directory: Path) -> ModelAsset:
    """Read a companion's weight descriptor without inventing an output artifact."""
    data = _read_toml(_preset_path(name, "models", directory))
    return ModelAsset(
        name=data.get("name", name),
        repository=data["repository"],
        revision=data["revision"],
        filename=data["filename"],
        sha256=data["sha256"],
    )


def _load_step(name: str, data: dict, directory: Path) -> StepConfig:
    """Resolve a step and its explicitly named companion model presets."""
    template = None
    if data.get("template"):
        template = _preset_path(data["template"], "notes", directory)
    model = _load_model(data["model"], directory, data.get("model_settings", {}))
    if "output" in data:
        model = replace(model, output=artifact_kind(data["output"]))
    return StepConfig(
        name=name,
        model=model,
        runtime=_load_runtime(
            data["runtime"], directory, data.get("runtime_settings", {})
        ),
        template=template,
        source=data.get("source"),
        instruction=template.read_text(encoding="utf-8").strip() if template else None,
        companions={
            role: _load_asset(preset, directory)
            for role, preset in data.get("companions", {}).items()
        },
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
