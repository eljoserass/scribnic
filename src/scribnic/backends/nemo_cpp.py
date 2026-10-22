"""Official NeMo-Speech.cpp CLI with managed assets and owned process cleanup."""

import json
import subprocess
from time import perf_counter
from uuid import uuid4

from ..artifacts import Artifact, AudioFile, StepOutput
from ..config import StepConfig
from ..downloads import PreparedStep, sha256_file
from ..observability import tracer
from ..profiling import native_profile
from ..storage import diagnostic_path
from .nvidia_speech import model_provenance, parse_speech, speech_metrics


class NemoCPP:
    """Run one native file/replay request; weights live for this CLI invocation."""

    def __init__(self, step: StepConfig, prepared: PreparedStep):
        """Retain prepared files without loading models or launching processes."""
        if prepared.binary is None or prepared.model_path is None:
            raise ValueError("NeMo-Speech.cpp needs its prepared binary and ASR model.")
        self.step = step
        self.prepared = prepared
        self.process: subprocess.Popen | None = None
        self.provenance = model_provenance(step, prepared)
        self.binary_sha256 = sha256_file(prepared.binary)

    def _command(self, source: AudioFile, output) -> list[str]:
        """Map explicit model/runtime settings to the pinned CLI's argument API."""
        settings = self.step.runtime.settings
        command = [
            str(self.prepared.binary),
            "transcribe",
            str(source.path),
            "--model",
            str(self.prepared.model_path),
            "--language",
            str(self.step.model.settings.get("language", "es-ES")),
            "--device",
            str(settings.get("device", "cpu")),
            "--asr.backend.threads",
            str(settings.get("threads", 6)),
            "--json",
            "--output",
            str(output),
            "--verbose",
        ]
        if self.step.model.output == "diarized_transcript":
            command.extend(
                ["--diar-model", str(self.prepared.companions["diarization"])]
            )
        if self.step.model.settings.get("mode", "offline") == "streaming":
            command.append("--stream")
        for key, value in settings.get("asr", {}).items():
            command.extend(["--asr." + key, str(value).lower()])
        return command

    def run(self, source: Artifact) -> StepOutput:
        """Persist native output and return an artifact with resource/profile scope."""
        if not isinstance(source, AudioFile) or source.duration is None:
            raise ValueError("NVIDIA speech requires a validated WAV audio artifact.")
        identifier = uuid4().hex
        output = diagnostic_path(
            f"nemo-cpp-{identifier}.json", cache=self.prepared.cache
        )
        log = output.with_suffix(".log")
        command = self._command(source, output)
        profile = native_profile(self.step.name, log_path=log)
        started = perf_counter()
        with tracer().start_as_current_span("scribnic.nemo_cpp.inference"):
            self._invoke(command, log)
        elapsed = perf_counter() - started
        payload = json.loads(output.read_text(encoding="utf-8"))
        artifact = parse_speech(
            payload,
            duration_seconds=source.duration,
            diarized=self.step.model.output == "diarized_transcript",
            sentence_gap_seconds=float(
                self.step.model.settings.get("sentence_gap_seconds", 1.0)
            ),
        )
        speech_metrics(
            self.step, elapsed_seconds=elapsed, audio_seconds=source.duration
        )
        return StepOutput(
            artifact,
            {
                "models": self.provenance,
                "mode": self.step.model.settings.get("mode", "offline"),
                "integration": "independent ASR and diarization, timestamp-based word attribution",
                "runtime": {
                    "engine": "nemo_cpp",
                    "distribution": "NVIDIA/NeMo-Speech.cpp",
                    "release_requested": self.step.runtime.settings.get("release"),
                    "binary_sha256": self.binary_sha256,
                    "argv": command,
                    "pid": self.process.pid,
                    "log_path": str(log),
                },
                "native_output_path": str(output),
                "request_seconds": elapsed,
                "real_time_factor": elapsed / source.duration,
                "timing_scope": "Cold CLI: model loading, warmup, inference and serialization; downloads excluded",
                "model_reuse": "One CLI process per run; no loaded weights retained between runs",
                "profile": profile,
            },
        )

    def _invoke(self, command: list[str], log) -> None:
        """Wait for our child and terminate it on timeout, interruption, or failure."""
        try:
            with log.open("w", encoding="utf-8") as stream:
                self.process = subprocess.Popen(
                    command, stdout=stream, stderr=subprocess.STDOUT
                )
                code = self.process.wait(
                    timeout=float(self.step.runtime.settings.get("timeout", 600))
                )
            if code:
                raise RuntimeError(
                    f"NeMo-Speech.cpp exited with {code}. See {log}\n{log.read_text()[-2000:]}"
                )
        except (OSError, subprocess.TimeoutExpired, KeyboardInterrupt):
            self.close()
            raise

    def process_ids(self) -> tuple[int, ...]:
        """Expose only the live process owned by this backend to resource sampling."""
        process = self.process
        return (process.pid,) if process and process.poll() is None else ()

    def close(self) -> None:
        """Terminate an active owned invocation; completed processes need no action."""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
