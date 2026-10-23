"""Resident Whisper.cpp server command and its accelerator/VAD settings."""

import platform

from .server import ManagedServer


class WhisperServer(ManagedServer):
    """Resident Whisper model exposed through a transcription HTTP endpoint."""

    def _command(self, port: int) -> list[str]:
        """Build the resident transcription service command and optional VAD path."""
        settings = self.runtime.settings
        command = [
            str(self.prepared.binary),
            "--model",
            str(self.prepared.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--threads",
            str(settings.get("threads", 6)),
            "--inference-path",
            "/v1/audio/transcriptions",
            "--print-progress",
        ]
        use_gpu = settings.get("use_gpu", "auto")
        if type(use_gpu) is not bool and use_gpu != "auto":
            raise ValueError("Whisper.cpp use_gpu must be true, false, or 'auto'.")
        if use_gpu is False or (use_gpu == "auto" and platform.system() != "Darwin"):
            command.append("--no-gpu")
        if self.prepared.vad_model_path is not None:
            command.extend(["--vad-model", str(self.prepared.vad_model_path)])
        return command
