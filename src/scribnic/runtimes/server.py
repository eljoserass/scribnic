"""Lifecycle and diagnostics for local inference server processes owned by Scribnic."""

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from uuid import uuid4

from ..config import RuntimeConfig
from ..downloads import PreparedStep, sha256_file
from ..storage import diagnostic_path


def _available_port() -> int:
    """Select a loopback TCP port for a session-owned service."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _health(base_url: str) -> bool:
    """Check the readiness endpoint without importing an inference SDK."""
    try:
        with urllib.request.urlopen(base_url, timeout=0.5) as response:
            payload = json.load(response)
            return payload.get("ready") is True or payload.get("status") == "ok"
    except (OSError, ValueError):
        return False


class ManagedServer:
    """Own one resident server process and its log until the session closes."""

    def __init__(
        self, runtime: RuntimeConfig, prepared: PreparedStep, alias: str = ""
    ) -> None:
        """Retain prepared local files without starting a process or loading weights."""
        if prepared.binary is None or prepared.model_path is None:
            raise ValueError(f"{runtime.name} needs prepared binary and model files.")
        self.runtime = runtime
        self.prepared = prepared
        self.alias = alias
        self.process: subprocess.Popen | None = None
        self._log = None
        self.log_path = prepared.cache / "logs" / f"{runtime.name}-{uuid4().hex}.log"
        self.base_url = ""
        self.command: list[str] = []
        self.binary_sha256 = sha256_file(prepared.binary)
        self.startup_seconds = 0.0
        self.health_path = "/health"

    def _command(self, port: int) -> list[str]:
        """Build the selected engine's argument list."""
        raise NotImplementedError

    def process_ids(self) -> tuple[int, ...]:
        """Return the live owned PID for application resource sampling."""
        process = self.process
        return (process.pid,) if process and process.poll() is None else ()

    def metrics(self) -> str | None:
        """Return optional native metrics; engines must opt into this capability."""
        return None

    def start(self) -> str:
        """Start the owned service once and wait for its model to become ready."""
        if self.process is not None and self.process.poll() is None:
            return self.base_url
        started = time.monotonic()
        port = int(self.runtime.settings.get("port", 0)) or _available_port()
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.command = self._command(port)
        self.log_path = diagnostic_path(
            f"{self.runtime.name}-{uuid4().hex}.log", cache=self.prepared.cache
        )
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
        except (OSError, RuntimeError, TimeoutError, KeyboardInterrupt):
            self.close()
            raise
        self.startup_seconds = time.monotonic() - started
        return self.base_url

    def _wait_ready(self) -> None:
        """Wait for readiness, reporting early process exit or startup timeout."""
        deadline = time.monotonic() + float(
            self.runtime.settings.get("startup_timeout", 90)
        )
        endpoint = self.base_url.removesuffix("/v1")
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                tail = self.log_path.read_text(encoding="utf-8")[-2000:]
                raise RuntimeError(
                    f"{self.runtime.name} exited during startup:\n{tail}"
                )
            if _health(endpoint + self.health_path):
                return
            time.sleep(0.1)
        raise TimeoutError(
            f"{self.runtime.name} startup timed out. See {self.log_path}"
        )

    def metadata(self) -> dict[str, object]:
        """Describe the exact executable, distribution, startup, PID and log."""
        return {
            "engine": self.runtime.engine,
            "name": self.runtime.name,
            "distribution": self.runtime.settings.get("distribution", "upstream"),
            "release_requested": self.runtime.settings.get("release"),
            "binary_sha256": self.binary_sha256,
            "startup_seconds": self.startup_seconds,
            "pid": self.process.pid if self.process else None,
            "argv": self.command,
            "log_path": str(self.log_path),
        }

    def close(self) -> None:
        """Terminate only the owned process and release the log handle."""
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
