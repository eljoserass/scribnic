"""Manage an owned local llama-server without Python native bindings."""

import json
import platform
import socket
import subprocess
import time
import urllib.error
import urllib.request
from uuid import uuid4

from .config import RuntimeConfig
from .downloads import PreparedStep


def _available_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _health(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url + "/health", timeout=0.5) as response:
            return json.load(response).get("status") == "ok"
    except (OSError, ValueError):
        return False


class LlamaServer:
    def __init__(self, runtime: RuntimeConfig, prepared: PreparedStep, alias: str):
        if prepared.binary is None or prepared.model_path is None:
            raise ValueError("llama-server needs prepared binary and model files.")
        self.runtime = runtime
        self.prepared = prepared
        self.alias = alias
        self.process: subprocess.Popen | None = None
        self._log = None
        self.log_path = prepared.cache / "logs" / f"llama-{uuid4().hex}.log"
        self.base_url = ""
        self.command: list[str] = []

    def _command(self, port: int) -> list[str]:
        options = self.runtime.settings
        gpu_layers = options.get("gpu_layers", "auto")
        if gpu_layers == "auto":
            gpu_layers = "all" if platform.system() == "Darwin" else "0"
        return [
            str(self.prepared.binary),
            "-m",
            str(self.prepared.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-c",
            str(options.get("context_size", 8192)),
            "-np",
            "1",
            "-ngl",
            str(gpu_layers),
            "-t",
            str(options.get("threads", 6)),
            "-tb",
            str(options.get("threads_batch", options.get("threads", 6))),
            "--alias",
            self.alias,
            "--metrics",
            "--perf",
        ]

    def start(self) -> str:
        if self.process is not None and self.process.poll() is None:
            return self.base_url
        port = int(self.runtime.settings.get("port", 0)) or _available_port()
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.command = self._command(port)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                self.command,
                cwd=self.prepared.binary.parent,
                stdout=self._log,
                stderr=subprocess.STDOUT,
            )
            self._wait_ready()
        except (OSError, RuntimeError, TimeoutError):
            self.close()
            raise
        return self.base_url

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + float(
            self.runtime.settings.get("startup_timeout", 90)
        )
        endpoint = self.base_url.removesuffix("/v1")
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                tail = self.log_path.read_text(encoding="utf-8")[-2000:]
                raise RuntimeError(f"llama-server exited during startup:\n{tail}")
            if _health(endpoint):
                return
            time.sleep(0.1)
        raise TimeoutError(f"llama-server startup timed out. See {self.log_path}")

    def metadata(self) -> dict[str, object]:
        return {
            "engine": "llama_cpp",
            "release": self.runtime.settings.get("release"),
            "gpu_layers_requested": self.command[self.command.index("-ngl") + 1],
            "pid": self.process.pid if self.process else None,
            "argv": self.command,
            "log_path": str(self.log_path),
        }

    def metrics(self) -> str | None:
        endpoint = self.base_url.removesuffix("/v1") + "/metrics"
        try:
            with urllib.request.urlopen(endpoint, timeout=2) as response:
                return response.read().decode("utf-8")
        except OSError:
            return None

    def close(self) -> None:
        try:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None
