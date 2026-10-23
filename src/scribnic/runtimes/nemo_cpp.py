"""Resident official NeMo-Speech.cpp service with explicit ASR/diarizer assets."""

from ..config import RuntimeConfig
from ..downloads import PreparedStep
from .server import ManagedServer


class NemoServer(ManagedServer):
    """Keep the ASR and optional diarizer loaded for this workflow session."""

    def __init__(self, runtime: RuntimeConfig, prepared: PreparedStep) -> None:
        """Retain pinned assets and use the service's model readiness endpoint."""
        super().__init__(runtime, prepared)
        self.health_path = "/ready"

    def _command(self, port: int) -> list[str]:
        """Configure the native engine; inference inputs belong to its client."""
        settings = self.runtime.settings
        command = [
            str(self.prepared.binary),
            "serve",
            "--asr-model",
            str(self.prepared.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--device",
            str(settings.get("device", "cpu")),
            "--asr.backend.threads",
            str(settings.get("threads", 6)),
            "--asr.batching.enabled=false",
            "--no-ui",
        ]
        diarizer = self.prepared.companions.get("diarization")
        if diarizer is not None:
            command.extend(["--diar-model", str(diarizer)])
        for key, value in settings.get("asr", {}).items():
            if isinstance(value, bool):
                command.append(f"--asr.{key}={str(value).lower()}")
            else:
                command.extend(["--asr." + key, str(value)])
        return command
