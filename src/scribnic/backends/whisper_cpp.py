"""Whisper.cpp CLI inference without Python bindings or a Torch dependency."""

import json
import logging
import platform
import re
import subprocess
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from ..artifacts import Artifact, AudioFile, StepOutput, TextSegment, Transcript
from ..config import StepConfig
from ..downloads import PreparedStep, sha256_file
from ..observability import tracer

END_TOLERANCE_SECONDS = 0.5


def parse_transcript(data: dict, duration: float) -> Transcript:
    """Preserve native times, allowing at most 0.5 s beyond the final WAV sample."""
    segments = []
    previous_start = -1.0
    try:
        for item in data["transcription"]:
            start = item["offsets"]["from"] / 1000
            end = item["offsets"]["to"] / 1000
            text = item["text"]
            if not isinstance(text, str):
                raise ValueError("Whisper.cpp returned non-text segment content.")
            if start < previous_start or end > duration + END_TOLERANCE_SECONDS:
                raise ValueError(
                    "Whisper.cpp returned out-of-order or out-of-range times."
                )
            segment = TextSegment(text.strip(), start, end)
            previous_start = start
            if segment.text:
                segments.append(segment)
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid Whisper.cpp transcription JSON.") from error
    return Transcript(tuple(segments))


def _native_timings(log: str) -> dict[str, float]:
    pattern = r"whisper_print_timings:\s+([a-z]+) time\s*=\s*([\d.]+) ms"
    return {
        name + "_seconds": float(ms) / 1000 for name, ms in re.findall(pattern, log)
    }


class WhisperCPP:
    def __init__(self, step: StepConfig, prepared: PreparedStep):
        if step.runtime.engine != "whisper_cpp":
            raise ValueError("The Whisper.cpp backend needs the whisper_cpp runtime.")
        if prepared.binary is None or prepared.model_path is None:
            raise ValueError("Whisper.cpp needs prepared binary and model files.")
        self.step = step
        self.prepared = prepared
        self.process: subprocess.Popen | None = None
        self.binary_sha256 = sha256_file(prepared.binary)

    def _command(self, source: AudioFile, prefix: Path) -> list[str]:
        model = self.step.model.settings
        runtime = self.step.runtime.settings
        command = [
            str(self.prepared.binary),
            "--model",
            str(self.prepared.model_path),
            "--file",
            str(source.path),
            "--language",
            str(model.get("language", "es")),
            "--threads",
            str(runtime.get("threads", 6)),
            "--beam-size",
            str(model.get("beam_size", 5)),
            "--best-of",
            str(model.get("best_of", 5)),
            "--temperature",
            str(model.get("temperature", 0)),
            "--output-json-full",
            "--output-file",
            str(prefix),
            "--print-progress",
        ]
        use_gpu = runtime.get("use_gpu", "auto")
        if type(use_gpu) is not bool and use_gpu != "auto":
            raise ValueError("Whisper.cpp use_gpu must be true, false, or 'auto'.")
        if use_gpu is False or (use_gpu == "auto" and platform.system() != "Darwin"):
            command.append("--no-gpu")
        return command

    def _wait(self, started: float) -> None:
        timeout = float(self.step.runtime.settings.get("timeout", 600))
        logger = logging.getLogger(__name__)
        while True:
            remaining = timeout - (perf_counter() - started)
            if remaining <= 0:
                raise TimeoutError(f"Whisper.cpp timed out after {timeout:g} seconds.")
            try:
                self.process.wait(timeout=min(5.0, remaining))
                return
            except subprocess.TimeoutExpired:
                logger.info(
                    "Whisper.cpp · procesando audio · %.0f s", perf_counter() - started
                )

    def _invoke(self, command: list[str], log_path: Path) -> tuple[float, int]:
        started = perf_counter()
        try:
            with log_path.open("w", encoding="utf-8") as log:
                self.process = subprocess.Popen(
                    command,
                    cwd=self.prepared.binary.parent,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                pid = self.process.pid
                self._wait(started)
                if self.process.returncode != 0:
                    raise RuntimeError(
                        f"Whisper.cpp exited with code {self.process.returncode}. "
                        f"See {log_path}."
                    )
        finally:
            self.close()
        return perf_counter() - started, pid

    def run(self, source: Artifact) -> StepOutput:
        if not isinstance(source, AudioFile) or source.duration is None:
            raise ValueError("Whisper.cpp needs a validated WAV audio artifact.")
        prefix = self.prepared.cache / "logs" / f"whisper-{uuid4().hex}"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        log_path = prefix.with_suffix(".log")
        command = self._command(source, prefix)
        logging.getLogger(__name__).info(
            "Whisper.cpp · audio %.1f s · idioma %s · %s",
            source.duration,
            self.step.model.settings.get("language", "es"),
            self.step.model.name,
        )
        with tracer().start_as_current_span("scribnic.whisper_cpp.transcribe"):
            elapsed, pid = self._invoke(command, log_path)
        native_path = prefix.with_suffix(".json")
        try:
            data = json.loads(native_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Cannot read Whisper.cpp output. See {log_path}."
            ) from error
        transcript = parse_transcript(data, source.duration)
        return StepOutput(
            transcript,
            self._metadata(
                data, command, log_path, native_path, elapsed, source.duration, pid
            ),
        )

    def _metadata(
        self, data, command, log_path, native_path, elapsed, duration, pid
    ) -> dict:
        return {
            "repository": self.step.model.repository,
            "revision": self.step.model.revision,
            "model_sha256": self.step.model.sha256,
            "quantization": self.step.model.settings.get("quantization"),
            "timestamp_end_tolerance_seconds": END_TOLERANCE_SECONDS,
            "language": data.get("result", {}).get("language"),
            "native_model": data.get("model"),
            "systeminfo": data.get("systeminfo"),
            "runtime": {
                "engine": "whisper_cpp",
                "pid": pid,
                "release_requested": self.step.runtime.settings.get("release"),
                "binary_sha256": self.binary_sha256,
                "argv": command,
                "log_path": str(log_path),
                "native_output_path": str(native_path),
            },
            "process_seconds": elapsed,
            "real_time_factor": elapsed / duration,
            "native_timings": _native_timings(log_path.read_text(encoding="utf-8")),
        }

    def process_ids(self) -> tuple[int, ...]:
        process = self.process
        if process is not None and process.poll() is None:
            return (process.pid,)
        return ()

    def close(self) -> None:
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        finally:
            self.process = None
