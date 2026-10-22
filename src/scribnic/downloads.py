"""Versioned asset preparation with atomic, checksum-verified downloads."""

import hashlib
import os
import platform
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .config import ModelAsset, ModelConfig, RuntimeConfig, StepConfig


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
    if path.is_file() and sha256_file(path) == checksum:
        return path
    if not allowed:
        raise ValueError(f"Missing or invalid cached asset: {path}. Enable downloads.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            with urllib.request.urlopen(url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        if sha256_file(temporary) != checksum:
            raise ValueError(f"Downloaded asset checksum does not match: {path.name}")
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
    _extract_archive(archive, directory)
    if not binary.is_file():
        raise ValueError(f"Runtime archive did not contain {asset['binary']}.")
    return binary


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
        return _prepare_snapshot(model, download)
    if runtime.engine not in {"llama_cpp", "whisper_cpp", "nemo_cpp", "nemo_python"}:
        return None
    if not all((model.repository, model.revision, model.filename, model.sha256)):
        raise ValueError(f"Model {model.name} needs a pinned file and checksum.")
    path = cache / "models" / model.name / model.revision / model.filename
    url = (
        f"https://huggingface.co/{model.repository}/resolve/"
        f"{model.revision}/{model.filename}"
    )
    return download_file(url, path, model.sha256, allowed=download)


@dataclass(frozen=True)
class PreparedStep:
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
    def __init__(self, *, cache: Path | None = None, download: bool = True):
        self.cache = cache or cache_directory()
        self.download = download
        self._prepared: dict[str, PreparedStep] = {}

    def prepare(self, step: StepConfig) -> PreparedStep:
        """Download versioned weights, companions, and native runtimes as needed."""
        if step.name not in self._prepared:
            model = prepare_model(step.model, step.runtime, self.cache, self.download)
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
