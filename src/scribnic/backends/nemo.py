"""NeMo reference inference: offline attribution or speaker-masked streaming replay."""

import importlib.metadata
import wave
from contextlib import contextmanager
from time import perf_counter
from typing import TYPE_CHECKING
from uuid import uuid4

from ..artifacts import Artifact, AudioFile, StepOutput
from ..config import StepConfig
from ..downloads import PreparedStep
from ..observability import tracer
from ..profiling import OperatorCapture, label_modules, operator_profile
from ..storage import diagnostic_path, write_json
from .nvidia_speech import model_provenance, parse_speech, speech_metrics

if TYPE_CHECKING:
    from omegaconf import DictConfig


class NemoSpeech:
    """Retain loaded models while creating fresh caches and speaker state per run."""

    def __init__(self, step: StepConfig, prepared: PreparedStep):
        """Retain pinned weight paths; import and load NeMo lazily on first use."""
        self.step = step
        self.prepared = prepared
        self.asr = None
        self.diarizer = None
        self.torch = None
        self.device = None
        self.provenance = model_provenance(step, prepared)
        self.phase_seconds: dict[str, float] = {}

    def _load(self) -> float:
        """Restore local checkpoints once, without fetching unpinned hub defaults."""
        if self.asr is not None:
            return 0.0
        started = perf_counter()
        try:
            import torch
            from nemo.collections.asr.models import ASRModel, SortformerEncLabelModel
        except ImportError as error:
            raise ImportError(
                "Install scribnic[nvidia], or use uv run --extra nvidia."
            ) from error
        actual = importlib.metadata.version("nemo-toolkit")
        expected = self.step.runtime.settings.get("package_version")
        if expected and actual != expected:
            raise ValueError(
                f"Expected NeMo {expected}, found {actual}. Install the pinned nvidia extra."
            )
        self.torch = torch
        requested = str(self.step.runtime.settings.get("device", "cpu"))
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested)
        try:
            with self._threads(), tracer().start_as_current_span("scribnic.nemo.load"):
                self.asr = (
                    ASRModel.restore_from(
                        str(self.prepared.model_path), map_location=self.device
                    )
                    .eval()
                    .to(self.device)
                )
                if self.step.model.output == "diarized_transcript":
                    self.diarizer = (
                        SortformerEncLabelModel.restore_from(
                            str(self.prepared.companions["diarization"]),
                            map_location=self.device,
                        )
                        .eval()
                        .to(self.device)
                    )
                self.asr.set_inference_prompt(
                    str(self.step.model.settings.get("language", "es-ES"))
                )
                self.asr.decoding.set_strip_lang_tags(True)
                self.asr.encoder.set_default_att_context_size(
                    list(self.step.model.settings.get("attention_context", [56, 13]))
                )
        except BaseException:
            self.close()
            raise
        return perf_counter() - started

    @contextmanager
    def _threads(self):
        """Restore the caller's Torch thread setting after inference or loading."""
        previous = self.torch.get_num_threads()
        self.torch.set_num_threads(int(self.step.runtime.settings.get("threads", 6)))
        try:
            yield
        finally:
            self.torch.set_num_threads(previous)

    def run(self, source: Artifact) -> StepOutput:
        """Execute a WAV experiment and persist raw output plus operator profiles."""
        if not isinstance(source, AudioFile) or source.duration is None:
            raise ValueError("NeMo needs a validated WAV audio artifact.")
        load_seconds = self._load()
        mode = self.step.model.settings.get("mode", "offline")
        scope = "NeMo preprocessing, ASR, diarization and attribution; loading excluded"
        started = perf_counter()
        self.phase_seconds = {}
        with self._threads(), self.torch.inference_mode():
            with operator_profile(
                self.torch,
                self.device,
                name=self.step.name + "-nemo",
                scope=scope,
                phase_prefix="nvidia.",
                include_gpu=bool(self.step.runtime.settings.get("profile_gpu", False)),
                window_steps=int(
                    self.step.runtime.settings.get("profile_window_steps", 16)
                ),
            ) as capture:
                with label_modules(self._modules(), self.torch):
                    with tracer().start_as_current_span("scribnic.nemo.inference"):
                        payload = (
                            self._stream(source, capture)
                            if mode == "streaming"
                            else self._offline(source, capture)
                        )
        elapsed = perf_counter() - started
        output = diagnostic_path(f"nemo-{uuid4().hex}.json", cache=self.prepared.cache)
        write_json(output, payload)
        artifact = parse_speech(
            payload,
            duration_seconds=source.duration,
            diarized=self.diarizer is not None,
            sentence_gap_seconds=float(
                self.step.model.settings.get("sentence_gap_seconds", 1.0)
            ),
        )
        overhead = capture.metadata.get("export_and_aggregation_seconds", 0)
        speech_metrics(
            self.step, elapsed_seconds=elapsed, audio_seconds=source.duration
        )
        return StepOutput(
            artifact,
            {
                "models": self.provenance,
                "load_seconds": load_seconds,
                "request_seconds": elapsed,
                "real_time_factor": elapsed / source.duration,
                "profile_export_seconds": overhead,
                "phase_wall_seconds": self.phase_seconds,
                "mode": mode,
                "streaming_scope": "recorded WAV replay, no microphone or network ingestion",
                "integration": "speaker-masked parallel ASR"
                if mode == "streaming"
                else "independent offline ASR and diarization with word-time attribution",
                "runtime": {
                    "engine": "nemo_python",
                    "version": importlib.metadata.version("nemo-toolkit"),
                    "source_revision": self.step.runtime.settings.get(
                        "source_revision"
                    ),
                    "torch_version": self.torch.__version__,
                    "device": str(self.device),
                    "dtype": str(next(self.asr.parameters()).dtype),
                    "threads": self.step.runtime.settings.get("threads", 6),
                },
                "native_output_path": str(output),
                "profile": capture.metadata,
            },
        )

    def _modules(self) -> tuple[tuple[object, str], ...]:
        """Identify expensive forwards without modifying NVIDIA's implementation."""
        modules = tuple(
            (getattr(self.asr, name, None), "nvidia.asr." + name)
            for name in ("preprocessor", "encoder", "decoder", "joint")
        )
        if self.diarizer is not None:
            modules += ((self.diarizer.encoder, "nvidia.diarization.encoder"),)
        return modules

    @contextmanager
    def _phase(self, name: str):
        """Record host wall time, an OTel span, and a PyTorch phase label."""
        started = perf_counter()
        with (
            tracer().start_as_current_span("scribnic.nemo." + name),
            self.torch.profiler.record_function("nvidia." + name),
        ):
            yield
        self.phase_seconds[name] = (
            self.phase_seconds.get(name, 0.0) + perf_counter() - started
        )

    def _offline(self, source: AudioFile, capture: OperatorCapture) -> list[dict]:
        """Transcribe a whole file and attribute words using diarization intervals."""
        # Tensor input makes the explicit language prompt authoritative. The pinned
        # NeMo file/Lhotse loader otherwise supplies a missing per-cut language.
        with self._phase("frontend"):
            with wave.open(str(source.path), "rb") as audio:
                pcm = bytearray(audio.readframes(audio.getnframes()))
            samples = self.torch.frombuffer(pcm, dtype=self.torch.int16).float()
            samples.mul_(1 / 32768)
        with self._phase("asr"):
            hypotheses = self.asr.transcribe(
                [samples],
                target_lang=str(self.step.model.settings.get("language", "es-ES")),
                batch_size=1,
                return_hypotheses=True,
                timestamps=True,
                num_workers=0,
                verbose=False,
            )
        capture.advance()
        if isinstance(hypotheses, tuple):
            hypotheses = hypotheses[0]
        hypothesis = hypotheses[0]
        intervals = []
        if self.diarizer is not None:
            self.diarizer.streaming_mode = False
            with self._phase("diarization"):
                intervals = self.diarizer.diarize(
                    [str(source.path)], batch_size=1, num_workers=0, verbose=False
                )[0]
            capture.advance()
        words = hypothesis.timestamp.get("word", [])
        if hypothesis.text.strip() and not words:
            raise ValueError("NeMo returned text without requested word timestamps.")
        with self._phase("attribution"):
            return attribute_words(words, intervals)

    def _stream(self, source: AudioFile, capture: OperatorCapture) -> list[dict]:
        """Use NVIDIA's stateful masked-ASR helpers with fresh per-file caches."""
        from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
            SpeakerTaggedASR,
        )
        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer,
        )

        if self.diarizer is None:
            raise ValueError("NeMo masked streaming requires a diarization companion.")
        cfg = self._stream_configuration(source)
        buffer = CacheAwareStreamingAudioBuffer(
            model=self.asr,
            online_normalization=False,
            pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
        )
        with self._phase("frontend"):
            buffer.append_audio_file(audio_filepath=str(source.path), stream_id=-1)
        streamer = SpeakerTaggedASR(cfg, self.asr, self.diarizer)
        for index, (audio, lengths, diar_audio, diar_lengths) in enumerate(
            buffer.iter_with_right_context(0)
        ):
            drop = (
                self.asr.encoder.streaming_cfg.drop_extra_pre_encoded
                if index or cfg.pad_and_drop_preencoded
                else 0
            )
            with self._phase("chunk"):
                streamer.perform_parallel_streaming_stt_spk(
                    step_num=index,
                    chunk_audio=audio,
                    chunk_lengths=lengths,
                    diar_chunk_audio=diar_audio,
                    diar_chunk_lengths=diar_lengths,
                    is_buffer_empty=buffer.is_buffer_empty(),
                    drop_extra_pre_encoded=drop,
                )
            capture.advance()
        return streamer.generate_seglst_dicts_from_parallel_streaming(
            samples=[{"audio_filepath": str(source.path)}]
        )

    def _stream_configuration(self, source: AudioFile) -> "DictConfig":
        """Apply the documented Nemotron 3.5 masked-ASR geometry and cache settings."""
        from nemo.collections.asr.parts.submodules.subsampling import FeatureStacking
        from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
            configure_diar_streaming,
            validate_feature_frame_strides,
        )
        from omegaconf import OmegaConf

        settings = self.step.model.settings
        values = {
            "audio_file": str(source.path),
            "manifest_file": None,
            "deploy_mode": False,
            "batch_size": 1,
            "streaming_mode": True,
            "max_num_of_spks": settings.get("max_speakers", 8),
            "parallel_speaker_strategy": True,
            "masked_asr": True,
            "mask_preencode": False,
            "single_speaker_mode": False,
            "cache_gating": True,
            "cache_gating_buffer_size": 2,
            "binary_diar_preds": True,
            "att_context_size": list(settings.get("attention_context", [56, 13])),
            "fifo_len": 264,
            "spkcache_len": None,
            "spkcache_update_period": 222,
            "diar_right_context": 0,
            "spk_supervision": "diar",
            "log": False,
            "fix_prev_words_count": 5,
            "update_prev_words_sentence": 5,
            "ignored_initial_frame_steps": 5,
            "word_window": 50,
            "sent_break_sec": settings.get("sentence_gap_seconds", 1.0),
            "left_frame_shift": -1,
            "right_frame_shift": 0,
            "min_sigmoid_val": 1e-2,
            "discarded_frames": 8,
            "generate_realtime_scripts": False,
            "colored_text": False,
            "print_time": False,
            "verbose": False,
            "print_sample_indices": [0],
            "pad_and_drop_preencoded": isinstance(
                self.diarizer.encoder.pre_encode, FeatureStacking
            ),
        }
        values.update(settings.get("streaming", {}))
        cfg = OmegaConf.create(values)
        validate_feature_frame_strides(self.asr, self.diarizer)
        geometry = self.asr.encoder.streaming_cfg
        configure_diar_streaming(
            self.diarizer,
            cfg,
            output_subsampling_factor=self.asr.encoder.subsampling_factor,
            diar_chunk_len=geometry.valid_out_len + geometry.cache_drop_size,
        )
        return cfg

    def process_ids(self) -> tuple[int, ...]:
        """NeMo executes in the application process, already monitored by the core."""
        return ()

    def close(self) -> None:
        """Release model references; no caches or artifacts survive into another run."""
        self.asr = None
        self.diarizer = None


def attribute_words(words: list[dict], intervals: list[str]) -> list[dict]:
    """Assign the largest positive temporal overlap, preserving unknown or tied IDs."""
    activity = []
    for interval in intervals:
        start, end, speaker = interval.split()
        activity.append((float(start), float(end), speaker))
    result = []
    for word in words:
        start, end = float(word["start"]), float(word["end"])
        overlaps: dict[str, float] = {}
        for left, right, speaker in activity:
            overlaps[speaker] = overlaps.get(speaker, 0) + max(
                0.0, min(end, right) - max(start, left)
            )
        best = max(overlaps.values(), default=0)
        winners = [
            speaker
            for speaker, overlap in overlaps.items()
            if best > 0 and overlap == best
        ]
        result.append(
            {
                "words": word["word"],
                "speaker": winners[0] if len(winners) == 1 else None,
                "start_time": start,
                "end_time": end,
            }
        )
    return result
