"""Speech protocol client with explicit provider parameters and response schemas."""

from pathlib import Path
from time import perf_counter
from uuid import uuid4

from ..artifacts import Artifact, AudioFile, StepOutput
from ..config import StepConfig
from ..downloads import PreparedStep, model_provenance
from ..observability import speech_metrics, tracer
from ..profiling import native_profile
from ..storage import diagnostic_path, write_json
from .connection import OpenAIConnection
from .transcription_formats import parse_segments, parse_words

WHISPER_PARAMETERS = (
    "vad",
    "vad_threshold",
    "vad_min_speech_duration_ms",
    "vad_min_silence_duration_ms",
    "vad_max_speech_duration_s",
    "vad_speech_pad_ms",
    "vad_samples_overlap",
)


class OpenAITranscription:
    """Retain one service/client while keeping recognition state local to each run."""

    def __init__(self, step: StepConfig, prepared: PreparedStep) -> None:
        """Retain protocol settings and prepared assets without starting inference."""
        self.step = step
        self.prepared = prepared
        self.connection = OpenAIConnection(step, prepared)
        self.schema = str(step.model.settings["response_schema"])
        self.diarized = step.model.output == "diarized_transcript"
        self.provenance = model_provenance(step, prepared)

    def start(self) -> float:
        """Load a resident service once and return initial startup seconds."""
        if self.step.model.settings.get("mode") == "streaming":
            from importlib.util import find_spec

            if find_spec("websockets") is None:
                raise ImportError("NeMo streaming needs scribnic[streaming].")
        started = perf_counter()
        elapsed = self.connection.start()
        if elapsed and self.step.runtime.engine == "external_endpoint":
            model = str(
                self.step.model.settings.get("served_model", self.step.model.name)
            )
            try:
                self.connection.check_model(model)
            except ValueError:
                self.close()
                raise
        return perf_counter() - started if elapsed else 0.0

    def _parameters(self) -> dict[str, str]:
        """Map only the selected provider's supported multipart extension fields."""
        settings = self.step.model.settings
        if self.schema == "whisper_segments":
            values = {
                "beam_size": settings.get("beam_size", 5),
                "best_of": settings.get("best_of", 5),
                "token_timestamps": settings.get("word_timestamps", True),
                "max_len": -1,
                "no_language_probabilities": True,
            }
            values.update(
                {key: settings[key] for key in WHISPER_PARAMETERS if key in settings}
            )
        elif self.schema == "nemo_words":
            values = {"diarization": self.diarized}
        else:
            values = {
                key: settings[key]
                for key in ("max_new_tokens", "prompt")
                if key in settings
            }
        return {
            key: str(value).lower() if isinstance(value, bool) else str(value)
            for key, value in values.items()
        }

    def _request(self, source: AudioFile) -> dict:
        """Upload a WAV or replay its PCM with the provider's streaming protocol."""
        client = self.connection.client
        settings = self.step.model.settings
        language = str(settings.get("language", "es"))
        if settings.get("mode", "offline") == "streaming":
            from .nemo_streaming import transcribe_stream

            return transcribe_stream(
                source.path,
                base_url=str(client.base_url),
                language=language,
                diarized=self.diarized,
                timeout_seconds=float(self.step.runtime.settings.get("timeout", 600)),
                event_path=diagnostic_path(
                    "nemo-events-" + uuid4().hex + ".json", cache=self.prepared.cache
                ),
                api_key=client.api_key,
            )
        with source.path.open("rb") as audio:
            response = client.audio.transcriptions.create(
                model=str(settings.get("served_model", self.step.model.name)),
                file=audio,
                language=language,
                response_format="verbose_json",
                temperature=float(settings.get("temperature", 0)),
                extra_body=self._parameters(),
            )
        return response.model_dump(mode="json")

    def run(self, source: Artifact) -> StepOutput:
        """Persist provider output, convert its declared schema and record CPU/RSS scope."""
        from openai import OpenAIError

        if not isinstance(source, AudioFile) or source.duration is None:
            raise ValueError("Speech needs a validated WAV audio artifact.")
        load_seconds = self.start()
        started = perf_counter()
        try:
            with tracer().start_as_current_span("scribnic.speech.request"):
                payload = self._request(source)
        except (OpenAIError, OSError, TimeoutError, ValueError) as error:
            self.close()
            raise ValueError(f"Speech transcription failed: {error}") from error
        elapsed = perf_counter() - started
        path = diagnostic_path(
            "speech-" + uuid4().hex + ".json", cache=self.prepared.cache
        )
        write_json(path, payload)
        if self.schema == "nemo_words":
            artifact = parse_words(
                payload,
                duration_seconds=source.duration,
                diarized=self.diarized,
                sentence_gap_seconds=float(
                    self.step.model.settings.get("sentence_gap_seconds", 1)
                ),
            )
        else:
            artifact = parse_segments(
                payload, duration_seconds=source.duration, diarized=self.diarized
            )
        speech_metrics(
            self.step, elapsed_seconds=elapsed, audio_seconds=source.duration
        )
        return StepOutput(
            artifact, self._metadata(path, elapsed, source.duration, load_seconds)
        )

    def _metadata(
        self, path: Path, elapsed: float, audio_seconds: float, load_seconds: float
    ) -> dict[str, object]:
        """Separate service load, recognition, protocol and weight provenance."""
        runtime = self.connection.runtime.metadata()
        profile = {}
        if runtime.get("log_path"):
            profile = native_profile(self.step.name, log_path=Path(runtime["log_path"]))
        strategy = "none"
        if self.schema == "nemo_words" and self.diarized:
            strategy = "word_time_alignment"
        elif self.diarized:
            strategy = "model_joint"
        return {
            "models": self.provenance,
            "runtime": runtime,
            "response_schema": self.schema,
            "speaker_attribution_strategy": strategy,
            "mode": self.step.model.settings.get("mode", "offline"),
            "load_seconds": load_seconds,
            "request_seconds": elapsed,
            "real_time_factor": elapsed / audio_seconds,
            "request_parameters": self._parameters(),
            "native_output_path": str(path),
            "model_reuse": "Resident service; fresh recognition state for each recording",
            "profile": profile,
        }

    def process_ids(self) -> tuple[int, ...]:
        """Expose live owned service PIDs; externally hosted services have none."""
        return self.connection.runtime.process_ids()

    def close(self) -> None:
        """Release the client and stop the service only when Scribnic owns it."""
        self.connection.close()
