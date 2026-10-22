"""Whisper.cpp transcription over a session-owned native HTTP server."""

import logging
from time import perf_counter
from uuid import uuid4

from ..artifacts import Artifact, AudioFile, StepOutput, TextSegment, Transcript
from ..config import StepConfig
from ..downloads import PreparedStep
from ..observability import meter, tracer
from ..runtimes import WhisperServer
from ..storage import write_json

END_TOLERANCE_SECONDS = 0.5
VAD_PARAMETERS = (
    "vad",
    "vad_threshold",
    "vad_min_speech_duration_ms",
    "vad_min_silence_duration_ms",
    "vad_max_speech_duration_s",
    "vad_speech_pad_ms",
    "vad_samples_overlap",
)


def parse_transcript(data: dict, duration: float) -> Transcript:
    """Preserve verbose_json times, with a 0.5 s endpoint estimation tolerance."""
    segments = []
    previous_start = -1.0
    try:
        for item in data["segments"]:
            start, end, text = item["start"], item["end"], item["text"]
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
        raise ValueError("Invalid Whisper.cpp verbose_json transcription.") from error
    return Transcript(tuple(segments))


class WhisperCPP:
    def __init__(self, step: StepConfig, prepared: PreparedStep):
        if step.runtime.engine != "whisper_cpp":
            raise ValueError("The Whisper.cpp backend needs the whisper_cpp runtime.")
        self.step = step
        self.prepared = prepared
        self.server = WhisperServer(step.runtime, prepared)
        self.client = None

    def _load(self) -> float:
        if self.client is not None:
            return 0.0
        from openai import OpenAI

        started = perf_counter()
        with tracer().start_as_current_span("scribnic.whisper_cpp.load"):
            base_url = self.server.start()
            self.client = OpenAI(
                base_url=base_url,
                api_key="local",
                timeout=float(self.step.runtime.settings.get("timeout", 600)),
                max_retries=0,
            )
        return perf_counter() - started

    def _parameters(self) -> dict[str, str]:
        settings = self.step.model.settings
        parameters = {
            "beam_size": settings.get("beam_size", 5),
            "best_of": settings.get("best_of", 5),
            "token_timestamps": settings.get("word_timestamps", True),
            # The server maps zero to 60 characters; negative disables wrapping.
            "max_len": -1,
            "no_language_probabilities": True,
        }
        parameters.update(
            {key: settings[key] for key in VAD_PARAMETERS if key in settings}
        )
        return {key: str(value).lower() for key, value in parameters.items()}

    def _transcribe(self, source: AudioFile) -> tuple[dict, float]:
        from openai import OpenAIError

        started = perf_counter()
        settings = self.step.model.settings
        try:
            with source.path.open("rb") as audio:
                with tracer().start_as_current_span("scribnic.whisper_cpp.transcribe"):
                    response = self.client.audio.transcriptions.create(
                        model=self.step.model.name,
                        file=audio,
                        language=str(settings.get("language", "es")),
                        response_format="verbose_json",
                        temperature=float(settings.get("temperature", 0)),
                        extra_body=self._parameters(),
                    )
        except OpenAIError as error:
            self.close()
            raise ValueError(f"Whisper.cpp transcription failed: {error}") from error
        return response.model_dump(mode="json"), perf_counter() - started

    def run(self, source: Artifact) -> StepOutput:
        if not isinstance(source, AudioFile) or source.duration is None:
            raise ValueError("Whisper.cpp needs a validated WAV audio artifact.")
        load_seconds = self._load()
        logging.getLogger(__name__).info(
            "Whisper.cpp · audio %.1f s · %s · VAD %s",
            source.duration,
            self.step.model.name,
            self.step.model.settings.get("vad", False),
        )
        data, elapsed = self._transcribe(source)
        native_path = self.prepared.cache / "logs" / f"whisper-{uuid4().hex}.json"
        write_json(native_path, data)
        transcript = parse_transcript(data, source.duration)
        self._record_metrics(source.duration, elapsed)
        metadata = {
            "repository": self.step.model.repository,
            "revision": self.step.model.revision,
            "model_sha256": self.step.model.sha256,
            "quantization": self.step.model.settings.get("quantization"),
            "timestamp_end_tolerance_seconds": END_TOLERANCE_SECONDS,
            "language": data.get("language"),
            "runtime": self.server.metadata(),
            "load_seconds": load_seconds,
            "request_seconds": elapsed,
            "real_time_factor": elapsed / source.duration,
            "request_parameters": self._parameters(),
            "vad_model_path": str(self.prepared.vad_model_path)
            if self.prepared.vad_model_path
            else None,
            "native_output_path": str(native_path),
            "native_timings": data.get("timings"),
            "native_timings_scope": "Server v1.9.5 prints aggregate timings at shutdown; see runtime log.",
        }
        return StepOutput(transcript, metadata)

    def _record_metrics(self, duration: float, elapsed: float) -> None:
        attributes = {
            "scribnic.model.name": self.step.model.name,
            "scribnic.vad.enabled": bool(self.step.model.settings.get("vad")),
        }
        for name, value, unit in (
            ("scribnic.speech.processing.duration", elapsed, "s"),
            ("scribnic.speech.audio.duration", duration, "s"),
            ("scribnic.speech.real_time_factor", elapsed / duration, "1"),
        ):
            meter().create_histogram(name, unit=unit).record(value, attributes)

    def process_ids(self) -> tuple[int, ...]:
        process = self.server.process
        if process is not None and process.poll() is None:
            return (process.pid,)
        return ()

    def close(self) -> None:
        try:
            if self.client is not None:
                self.client.close()
                self.client = None
        finally:
            self.server.close()
