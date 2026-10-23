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
    quantization: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    """Identify model weights, artifact capabilities and model-specific settings."""

    name: str
    inputs: tuple[ArtifactKind, ...]
    output: ArtifactKind
    repository: str | None = None
    revision: str | None = None
    filename: str | None = None
    sha256: str | None = None
    settings: dict[str, object] = field(default_factory=dict)
    quantization: str | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    """Describe an inference engine distribution or an externally owned endpoint."""

    name: str
    engine: str
    settings: dict[str, object] = field(default_factory=dict)
    assets: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class StepConfig:
    """Bind a protocol adapter to weights, a service and an artifact contract."""

    # TODO(lifecycle): configure retention independently of explicit preload.
    # Apply policy in WorkflowSession and respect service ownership/capabilities;
    # ASR and companion diarization can share one resident process. See docs/nvidia.md.
    name: str
    model: ModelConfig
    runtime: RuntimeConfig
    backend: str
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
    _validate_adapter(step)


def _validate_adapter(step: StepConfig) -> None:
    """Reject incompatible protocols, formats and outputs before downloading assets."""
    engine = step.runtime.engine
    if step.backend == "openai_chat":
        if engine not in {"llama_cpp", "external_endpoint"}:
            raise ValueError("openai_chat needs llama_cpp or external_endpoint.")
        return
    if step.backend == "moss_transcribe_diarize":
        if engine != "transformers":
            raise ValueError("The local MOSS backend needs the Transformers runtime.")
        return
    if step.backend != "openai_transcription":
        return
    if engine not in {"whisper_cpp", "nemo_cpp", "external_endpoint"}:
        raise ValueError(
            "openai_transcription needs a speech server or external endpoint."
        )
    if step.model.output not in {"transcript", "diarized_transcript"}:
        raise ValueError("Transcription produces transcript or diarized_transcript.")
    response = step.model.settings.get("response_schema")
    if response not in {"whisper_segments", "nemo_words", "moss_segments"}:
        raise ValueError("Select an explicit transcription response_schema.")
    mode = step.model.settings.get("mode", "offline")
    if mode not in {"offline", "streaming"}:
        raise ValueError("Speech mode must be offline or streaming.")
    if response == "whisper_segments" and step.model.output != "transcript":
        raise ValueError("Whisper segments do not provide speaker diarization.")
    if mode == "streaming" and response != "nemo_words":
        raise ValueError(
            "Streaming input currently requires the NeMo WebSocket protocol."
        )
    if engine == "nemo_cpp":
        if response != "nemo_words":
            raise ValueError("NeMo-Speech.cpp requires the nemo_words response schema.")
        if (
            step.model.output == "diarized_transcript"
            and "diarization" not in step.companions
        ):
            raise ValueError(
                "Diarized NVIDIA speech needs a diarization companion model."
            )
        for model in (step.model, *step.companions.values()):
            if not model.filename or not model.filename.endswith(".gguf"):
                raise ValueError("NeMo-Speech.cpp requires .gguf model files.")
    if engine == "whisper_cpp" and response != "whisper_segments":
        raise ValueError("Whisper.cpp requires the whisper_segments response schema.")
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
    """Read weight identity and artifact capabilities independently of the adapter."""
    data = _read_toml(_preset_path(name, "models", directory))
    settings = data.get("settings", {}) | overrides
    return ModelConfig(
        name=data.get("name", name),
        inputs=tuple(artifact_kind(kind) for kind in data["inputs"]),
        output=artifact_kind(data["output"]),
        repository=data.get("repository"),
        revision=data.get("revision"),
        filename=data.get("filename"),
        sha256=data.get("sha256"),
        settings=settings,
        quantization=data.get("quantization"),
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
        quantization=data.get("quantization"),
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
        backend=data["backend"],
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
