"""Versioned asset preparation with atomic, checksum-verified downloads."""

import hashlib
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .config import ModelAsset, ModelConfig, RuntimeConfig, StepConfig
from .progress import ProgressTask, phase


def cache_directory() -> Path:
    if os.getenv("SCRIBNIC_CACHE"):
        return Path(os.environ["SCRIBNIC_CACHE"]).expanduser()
    if platform.system() == "Windows":
        return (
            Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData/Local")) / "scribnic"
        )
    if platform.system() == "Darwin":
        return Path.home() / "Library/Caches/scribnic"
    return Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "scribnic"


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_file(url: str, path: Path, checksum: str, *, allowed: bool) -> Path:
    """Download atomically, reporting verified cache reuse and actual transferred bytes."""
    if path.is_file():
        with phase("Verificar caché", detail=path.name):
            valid = sha256_file(path) == checksum
        if valid:
            return path
    if not allowed:
        raise ValueError(f"Missing or invalid cached asset: {path}. Enable downloads.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(url, timeout=60) as response:
                size = response.headers.get("Content-Length")
                total = int(size) if size and size.isdigit() else None
                progress = ProgressTask(
                    "Descargar", detail=path.name, total=total, unit="bytes"
                )
                try:
                    transferred = 0
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        transferred += len(chunk)
                        progress.update(transferred)
                except BaseException as error:
                    progress.finish(
                        "interrupted"
                        if isinstance(error, KeyboardInterrupt)
                        else "error"
                    )
                    raise
                else:
                    progress.finish()
        with phase("Verificar descarga", detail=path.name):
            if sha256_file(temporary) != checksum:
                raise ValueError(
                    f"Downloaded asset checksum does not match: {path.name}"
                )
            temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def platform_key() -> str:
    machine = platform.machine().lower()
    aliases = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}
    return f"{platform.system().lower()}-{aliases.get(machine, machine)}"


def _extract_archive(archive: Path, directory: Path) -> None:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for name in bundle.namelist():
                destination = (directory / name).resolve()
                if not destination.is_relative_to(directory.resolve()):
                    raise ValueError("Runtime archive contains an unsafe path.")
            bundle.extractall(directory)
        return
    with tarfile.open(archive) as bundle:
        bundle.extractall(directory, filter="data")


def prepare_binary(runtime: RuntimeConfig, cache: Path, download: bool) -> Path:
    """Install a pinned release or build its pinned source when no binary ships."""
    environment = str(runtime.settings.get("binary_env", ""))
    explicit = os.getenv(environment) or runtime.settings.get("binary")
    if explicit:
        executable = shutil.which(os.path.expanduser(str(explicit)))
        if executable is None:
            raise ValueError(
                f"Runtime binary not found: {explicit}. Install {runtime.name} or "
                "set runtime_settings.binary to its executable path."
            )
        return Path(executable).resolve(strict=True)
    key = platform_key()
    if key not in runtime.assets:
        raise ValueError(f"No packaged {runtime.name} runtime for {key}.")
    asset = runtime.assets[key]
    release = str(runtime.settings["release"])
    directory = cache / "runtimes" / runtime.engine.replace("_", "-") / release
    binary = directory / asset["binary"]
    if binary.is_file():
        return binary
    archive = directory / Path(asset["url"]).name
    download_file(asset["url"], archive, asset["sha256"], allowed=download)
    with phase("Descomprimir runtime", detail=runtime.name):
        _extract_archive(archive, directory)
    if runtime.settings.get("build_target"):
        with phase("Compilar runtime", detail=runtime.name):
            _build_runtime(runtime, directory, download=download)
    if not binary.is_file():
        raise ValueError(f"Runtime archive did not contain {asset['binary']}.")
    return binary


def _build_runtime(runtime: RuntimeConfig, directory: Path, *, download: bool) -> None:
    """Build a selected native target, keeping tools out of Python client dependencies.

    A compiler must be installed. Use a system CMake or uv's isolated pinned CMake
    tool environment. Keep build output in the versioned cache for diagnostics.
    """
    cmake = shutil.which("cmake")
    if cmake:
        command = [cmake]
    elif download and shutil.which("uvx"):
        command = [shutil.which("uvx"), "--from", "cmake==4.1.0", "cmake"]
    else:
        raise ValueError(
            "Building this runtime needs CMake and a C/C++ compiler. Install them, then repeat scribnic setup."
        )
    source = directory / str(runtime.settings["source_root"])
    build = source / "build"
    configure = [
        *command,
        "-S",
        str(source),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        *runtime.settings.get("cmake_options", []),
    ]
    compile_command = [
        *command,
        "--build",
        str(build),
        "--target",
        str(runtime.settings["build_target"]),
        "--config",
        "Release",
        "--parallel",
        "6",
    ]
    log = directory / "build.log"
    with log.open("w", encoding="utf-8") as output:
        try:
            subprocess.run(
                configure,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=300,
            )
            subprocess.run(
                compile_command,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1200,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"Native runtime build failed. See {log}") from error


def _hub_cache() -> Path:
    if os.getenv("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = Path(
        os.getenv(
            "HF_HOME",
            Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "huggingface",
        )
    )
    return home / "hub"


def _prepare_snapshot(model: ModelConfig, download: bool) -> Path:
    if not model.repository or not model.revision:
        raise ValueError(f"Model {model.name} needs a repository and pinned revision.")
    repository = "models--" + model.repository.replace("/", "--")
    cached = _hub_cache() / repository / "snapshots" / model.revision
    required = model.settings.get("required_files", ["config.json", model.filename])
    if cached.is_dir() and all((cached / name).is_file() for name in required if name):
        return cached
    if not download:
        raise ValueError(f"Model {model.name} is not cached. Enable downloads.")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError(
            "Install scribnic[moss] to prepare the MOSS runtime."
        ) from error
    return Path(
        snapshot_download(
            repo_id=model.repository,
            revision=model.revision,
            allow_patterns=["*.json", "*.py", "*.safetensors", "*.txt", "*.jinja"],
            max_workers=2,
        )
    )


def prepare_model(
    model: ModelConfig | ModelAsset, runtime: RuntimeConfig, cache: Path, download: bool
) -> Path | None:
    """Prepare the format selected explicitly by a model/runtime configuration."""
    if runtime.engine == "transformers":
        with phase("Preparar modelo", detail=model.name):
            return _prepare_snapshot(model, download)
    if runtime.engine not in {"llama_cpp", "whisper_cpp", "nemo_cpp"}:
        return None
    if not all((model.repository, model.revision, model.filename, model.sha256)):
        raise ValueError(f"Model {model.name} needs a pinned file and checksum.")
    path = cache / "models" / model.name / model.revision / model.filename
    if not path.exists():
        # Variant names can change without downloading identical pinned weights.
        for cached in (cache / "models").glob(f"*/{model.revision}/{model.filename}"):
            if sha256_file(cached) == model.sha256:
                return cached
    url = (
        f"https://huggingface.co/{model.repository}/resolve/"
        f"{model.revision}/{model.filename}"
    )
    return download_file(url, path, model.sha256, allowed=download)


@dataclass(frozen=True)
class PreparedStep:
    """Keep verified resource paths without loading models or owning processes."""

    cache: Path
    model_path: Path | None = None
    binary: Path | None = None
    vad_model_path: Path | None = None
    companions: dict[str, Path] = field(default_factory=dict)


def prepare_vad(step: StepConfig, cache: Path, download: bool) -> Path | None:
    """Prepare the optional Silero asset only when the speech step enables VAD."""
    if not step.model.settings.get("vad", False):
        return None
    asset = step.model.settings.get("vad_model")
    required = ("name", "repository", "revision", "filename", "sha256")
    if not isinstance(asset, dict) or not all(asset.get(key) for key in required):
        raise ValueError("VAD needs a versioned vad_model asset with its SHA256.")
    path = cache / "models" / asset["name"] / asset["revision"] / asset["filename"]
    url = f"https://huggingface.co/{asset['repository']}/resolve/{asset['revision']}/{asset['filename']}"
    return download_file(url, path, asset["sha256"], allowed=download)


class ResourceManager:
    """Prepare only configured assets and retain their paths for one session."""

    def __init__(self, *, cache: Path | None = None, download: bool = True) -> None:
        """Select a versioned cache and whether missing assets may be downloaded."""
        self.cache = cache or cache_directory()
        self.download = download
        self._prepared: dict[str, PreparedStep] = {}

    def prepare(self, step: StepConfig) -> PreparedStep:
        """Download versioned weights, companions, and native runtimes as needed."""
        if step.name not in self._prepared:
            with phase("Preparar recursos", detail=step.model.name):
                model = prepare_model(
                    step.model, step.runtime, self.cache, self.download
                )
                binary = None
                if step.runtime.engine in {"llama_cpp", "whisper_cpp", "nemo_cpp"}:
                    binary = prepare_binary(step.runtime, self.cache, self.download)
                vad = prepare_vad(step, self.cache, self.download)
                companions = {
                    role: prepare_model(asset, step.runtime, self.cache, self.download)
                    for role, asset in step.companions.items()
                }
                self._prepared[step.name] = PreparedStep(
                    self.cache, model, binary, vad, companions
                )
        return self._prepared[step.name]


def model_provenance(step: StepConfig, prepared: PreparedStep) -> dict:
    """Describe actual prepared weights, including the companion diarizer."""
    if prepared.model_path is None:
        return {
            "model": {
                "name": step.model.name,
                "repository": step.model.repository,
                "revision": step.model.revision,
                "quantization": step.model.quantization,
                "verification_scope": "configuration only; weights owned by external service",
            }
        }
    models = {"primary": (step.model, prepared.model_path)}
    models.update(
        {
            role: (model, prepared.companions[role])
            for role, model in step.companions.items()
        }
    )
    return {
        role: {
            "name": model.name,
            "repository": model.repository,
            "revision": model.revision,
            "filename": path.name,
            "sha256": sha256_file(path),
            "path": str(path),
            "quantization": model.quantization,
        }
        for role, (model, path) in models.items()
    }
