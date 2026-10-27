"""Speech protocol client with explicit provider parameters and response schemas."""

from collections.abc import Callable, Iterable
from contextlib import closing
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from ..artifacts import Artifact, AudioFile, DiarizedTranscript, StepOutput, Transcript
from ..audio_stream import PCMRecording, wav_chunks
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
        if (
            self.step.model.settings.get("mode") == "streaming"
            and self.schema == "nemo_words"
        ):
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
            with closing(wav_chunks(source.path)) as chunks:
                return self._stream_request(chunks)
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

    def _stream_request(
        self, chunks: Iterable[bytes], *, on_update: Callable[[str], None] | None = None
    ) -> dict:
        """Select a protocol strategy without changing the resident runtime."""
        settings = self.step.model.settings
        if self.schema == "nemo_words":
            from .nemo_streaming import transcribe_pcm

            client = self.connection.client
            return transcribe_pcm(
                chunks,
                base_url=str(client.base_url),
                language=str(settings.get("language", "es-ES")),
                diarized=self.diarized,
                timeout_seconds=float(self.step.runtime.settings.get("timeout", 600)),
                event_path=diagnostic_path(
                    "nemo-events-" + uuid4().hex + ".jsonl", cache=self.prepared.cache
                ),
                api_key=client.api_key,
                on_update=on_update,
            )
        from .whisper_windowed import WindowSettings, transcribe_windows

        return transcribe_windows(
            chunks,
            request=self._window_request,
            settings=WindowSettings(
                window_seconds=float(settings.get("window_seconds", 10)),
                overlap_seconds=float(settings.get("overlap_seconds", 2)),
            ),
            event_path=diagnostic_path(
                "whisper-windows-" + uuid4().hex + ".jsonl", cache=self.prepared.cache
            ),
            on_update=on_update,
        )

    def _window_request(self, wav: bytes) -> dict:
        """Request timestamped words so overlap reconciliation has temporal anchors."""
        settings = self.step.model.settings
        parameters = self._parameters() | {
            "max_len": "1",
            "split_on_word": "true",
            "token_timestamps": "true",
        }
        response = self.connection.client.audio.transcriptions.create(
            model=str(settings.get("served_model", self.step.model.name)),
            file=("window.wav", wav, "audio/wav"),
            language=str(settings.get("language", "es")),
            response_format="verbose_json",
            temperature=float(settings.get("temperature", 0)),
            extra_body=parameters,
        )
        return response.model_dump(mode="json")

    def run(self, source: Artifact) -> StepOutput:
        """Persist provider output, convert its declared schema and record CPU/RSS scope."""
        if not isinstance(source, AudioFile) or source.duration is None:
            raise ValueError("Speech needs a validated WAV audio artifact.")
        return self._recognize(
            lambda: self._request(source), audio_seconds=lambda: source.duration
        )

    def run_stream(self, source: PCMRecording) -> StepOutput:
        """Recognize an arriving PCM source and finalize one transcript artifact."""
        if self.step.model.settings.get("mode") != "streaming":
            raise ValueError("Select mode='streaming' before submitting live PCM.")
        on_update = source.notify if source.on_update is not None else None
        result = self._recognize(
            lambda: self._stream_request(source, on_update=on_update),
            audio_seconds=lambda: source.duration,
        )
        if source.on_update is not None:
            source.notify(result.artifact.text, final=True)
        return result

    def _recognize(
        self, request: Callable[[], dict], *, audio_seconds: Callable[[], float]
    ) -> StepOutput:
        """Measure, persist and convert file or incremental recognition consistently."""
        from openai import OpenAIError

        load_seconds = self.start()
        started = perf_counter()
        try:
            with tracer().start_as_current_span(
                "scribnic.speech.request",
                attributes={
                    "scribnic.model.name": self.step.model.name,
                    "scribnic.output.kind": self.step.model.output,
                },
            ):
                payload = request()
        except InterruptedError:
            raise
        except (OpenAIError, OSError, TimeoutError, ValueError) as error:
            self.close()
            raise ValueError(f"Speech transcription failed: {error}") from error
        elapsed = perf_counter() - started
        duration_seconds = audio_seconds()
        if duration_seconds <= 0:
            raise ValueError("Audio input is empty.")
        path = diagnostic_path(
            "speech-" + uuid4().hex + ".json", cache=self.prepared.cache
        )
        write_json(path, payload)
        artifact = self._convert_response(payload, audio_seconds=duration_seconds)
        speech_metrics(
            self.step, elapsed_seconds=elapsed, audio_seconds=duration_seconds
        )
        return StepOutput(
            artifact,
            self._metadata(
                path,
                request_seconds=elapsed,
                audio_seconds=duration_seconds,
                load_seconds=load_seconds,
            )
            | {
                "windowing": payload.get("windowing"),
                "native_events_path": payload.get("native_events_path"),
            },
        )

    def _convert_response(
        self, payload: dict, *, audio_seconds: float
    ) -> Transcript | DiarizedTranscript:
        """Apply only the configured provider's transcript and speaker policy."""
        if self.schema == "nemo_words":
            return parse_words(
                payload,
                duration_seconds=audio_seconds,
                diarized=self.diarized,
                sentence_gap_seconds=float(
                    self.step.model.settings.get("sentence_gap_seconds", 1)
                ),
            )
        return parse_segments(
            payload, duration_seconds=audio_seconds, diarized=self.diarized
        )

    def _metadata(
        self,
        path: Path,
        *,
        request_seconds: float,
        audio_seconds: float,
        load_seconds: float,
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
        parameters = self._parameters()
        streaming_strategy = None
        if self.step.model.settings.get("mode") == "streaming":
            streaming_strategy = "cache_aware_websocket"
            if self.schema == "whisper_segments":
                streaming_strategy = "windowed_http"
                parameters |= {
                    "max_len": "1",
                    "split_on_word": "true",
                    "token_timestamps": "true",
                }
        return {
            "models": self.provenance,
            "runtime": runtime,
            "response_schema": self.schema,
            "speaker_attribution_strategy": strategy,
            "mode": self.step.model.settings.get("mode", "offline"),
            "load_seconds": load_seconds,
            "request_seconds": request_seconds,
            "real_time_factor": request_seconds / audio_seconds,
            "request_parameters": parameters,
            "streaming_strategy": streaming_strategy,
            "timing_scope": "request wall time, including input arrival and optional profiling overhead",
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
