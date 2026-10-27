"""OpenTelemetry instrumentation and application-owned local export."""

import logging
import os
import platform
import sys
from contextlib import ExitStack
from contextvars import ContextVar
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, TextIO

from opentelemetry import metrics, trace

from .progress import ProgressTask, current_observer

if TYPE_CHECKING:
    from opentelemetry.context import Context
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

    from ..config import StepConfig

_providers: ContextVar[tuple[trace.TracerProvider, metrics.MeterProvider] | None] = (
    ContextVar("scribnic_otel_providers", default=None)
)


def tracer() -> trace.Tracer:
    """Use this call's provider or the host's global OpenTelemetry provider."""
    providers = _providers.get()
    if providers is not None:
        return providers[0].get_tracer("scribnic", "0.1.0")
    return trace.get_tracer("scribnic", "0.1.0")


def meter() -> metrics.Meter:
    """Use this call's metric provider without replacing the host's global one."""
    providers = _providers.get()
    if providers is not None:
        return providers[1].get_meter("scribnic", "0.1.0")
    return metrics.get_meter("scribnic", "0.1.0")


def trace_id() -> str | None:
    """Return the active trace ID, preserving the absence of a recording span."""
    current = trace.get_current_span().get_span_context()
    return f"{current.trace_id:032x}" if current.is_valid else None


class GenerationProgress:
    """Rate-limited diagnostics without logging model input or generated text."""

    def __init__(self, name: str, *, interval: float = 5.0) -> None:
        """Configure generation logging with a minimum interval in seconds."""
        self.name = name
        self.interval = interval
        self.started = monotonic()
        self.next_report = self.started
        self.first_token_at: float | None = None
        self.logger = logging.getLogger("scribnic.generation")

    def update(self, generated_tokens: int) -> None:
        """Record the first observed token and periodically log cumulative counts."""
        now = monotonic()
        if self.first_token_at is None:
            self.first_token_at = now
        if now < self.next_report:
            return
        self.next_report = now + self.interval
        self.logger.info(
            "%s · generando · %d tokens · %.0f s",
            self.name,
            generated_tokens,
            now - self.started,
            extra={
                "scribnic.generation.tokens": generated_tokens,
                "scribnic.generation.elapsed_seconds": now - self.started,
            },
        )

    def timings(self) -> dict[str, float]:
        """Return observed first-token and total generation times in seconds."""
        if self.first_token_at is None:
            return {}
        return {
            "time_to_first_token_seconds": self.first_token_at - self.started,
            "generation_seconds": monotonic() - self.started,
        }


def _progress_processor(stream: TextIO | None) -> "SpanProcessor":
    """Reuse execution spans for optional UI notifications and legacy text output."""
    from opentelemetry.sdk.trace import SpanProcessor

    class ProgressProcessor(SpanProcessor):
        """Retain only active span/task pairs for the lifetime of one trace provider."""

        def __init__(self) -> None:
            """Capture the caller's observer before SDK callbacks run."""
            self.observer = current_observer()
            self.tasks: dict[int, ProgressTask] = {}
            self.steps: dict[int, Span] = {}

        def on_start(
            self, span: "Span", parent_context: "Context | None" = None
        ) -> None:
            """Notify selected phases without polling or reading exported traces."""
            if span.name == "scribnic.step.run":
                self.steps[span.context.span_id] = span
            attributes = dict(span.attributes or {})
            parent = self.steps.get(span.parent.span_id) if span.parent else None
            if parent is not None:
                attributes = {**(parent.attributes or {}), **attributes}
            speech_label = (
                "Transcribir y diarizar"
                if attributes.get("scribnic.output.kind") == "diarized_transcript"
                else "Transcribir"
            )
            label = {
                "scribnic.service.load": "Cargar servicio",
                "scribnic.model.load": "Cargar modelo",
                "scribnic.speech.request": speech_label,
                "scribnic.note.request": "Generar nota",
            }.get(span.name)
            if self.observer is not None and label is not None:
                detail = attributes.get("scribnic.model.name", "")
                self.tasks[span.context.span_id] = ProgressTask(
                    label, detail=detail, observer=self.observer
                )
            if stream is not None and span.name == "scribnic.step.run":
                name = span.attributes.get("scribnic.step.name", "step")
                print(f"  … {name}", file=stream, flush=True)

        def on_end(self, span: "ReadableSpan") -> None:
            """Close matching notifications using the recorded span status."""
            self.steps.pop(span.context.span_id, None)
            task = self.tasks.pop(span.context.span_id, None)
            if task is not None:
                # OTel does not mark KeyboardInterrupt as an error. The synchronous
                # callback still runs while that exception unwinds the span.
                if isinstance(sys.exception(), KeyboardInterrupt):
                    task.finish("interrupted")
                elif span.status.status_code.name == "ERROR":
                    task.finish("error")
                else:
                    task.finish()
            if stream is not None and span.name == "scribnic.step.run":
                name = span.attributes.get("scribnic.step.name", "step")
                seconds = (span.end_time - span.start_time) / 1e9
                mark = "✗" if span.status.status_code.name == "ERROR" else "✓"
                print(f"  {mark} {name} · {seconds:.2f} s", file=stream, flush=True)

    return ProgressProcessor()


class Telemetry:
    """Export real OTLP JSONL locally, without replacing a host's global SDK."""

    def __init__(
        self, directory: str | Path, *, progress: TextIO | None = None
    ) -> None:
        """Configure local export without opening files or starting SDK workers."""
        self.directory = Path(directory)
        self.progress = progress
        self._token = None
        self._instrumentor = None
        self._cleanup = ExitStack()

    def __enter__(self) -> "Telemetry":
        """Start local exporters and scope their providers to the current caller."""
        from opentelemetry.sdk.resources import Resource

        self.directory.mkdir(parents=True, exist_ok=True)
        resource = Resource.create(
            {
                "service.name": "scribnic",
                "service.version": "0.1.0",
                "process.pid": os.getpid(),
                "os.type": platform.system().lower(),
                "host.arch": platform.machine(),
            }
        )
        try:
            self._configure_traces(resource)
            self._configure_metrics(resource)
            self._configure_logs(resource)
        except Exception:
            self._cleanup.close()
            raise
        self._token = _providers.set((self._traces, self._metrics))
        return self

    def _configure_traces(self, resource: "Resource") -> None:
        """Export spans locally and notify the caller's optional progress observer."""
        from opentelemetry.exporter.otlp.json.file import FileSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        self._traces = TracerProvider(resource=resource)
        self._cleanup.callback(self._traces.shutdown)
        self._traces.add_span_processor(
            BatchSpanProcessor(FileSpanExporter(self.directory / "traces.jsonl"))
        )
        if self.progress is not None or current_observer() is not None:
            self._traces.add_span_processor(_progress_processor(self.progress))

    def _configure_metrics(self, resource: "Resource") -> None:
        """Own periodic local metric export and system/process instrumentation."""
        from opentelemetry.exporter.otlp.json.file import FileMetricExporter
        from opentelemetry.instrumentation.system_metrics import (
            SystemMetricsInstrumentor,
        )
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        reader = PeriodicExportingMetricReader(
            FileMetricExporter(self.directory / "metrics.jsonl"),
            export_interval_millis=500,
        )
        self._metrics = MeterProvider(resource=resource, metric_readers=[reader])
        self._cleanup.callback(self._metrics.shutdown)
        self._instrumentor = SystemMetricsInstrumentor(
            config={
                "system.cpu.utilization": ["user", "system", "idle"],
                "system.memory.usage": ["used", "free"],
                "process.cpu.time": ["user", "system"],
                "process.memory.usage": None,
                "process.thread.count": None,
            }
        )
        self._instrumentor.instrument(meter_provider=self._metrics)
        self._cleanup.callback(self._instrumentor.uninstrument)

    def _configure_logs(self, resource: "Resource") -> None:
        """Attach scoped log exporters and restore the logger when closing."""
        from opentelemetry.exporter.otlp.json.file._log_exporter import FileLogExporter
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor

        self._logs = LoggerProvider(resource=resource)
        self._cleanup.callback(self._logs.shutdown)
        self._logs.add_log_record_processor(
            SimpleLogRecordProcessor(FileLogExporter(self.directory / "logs.jsonl"))
        )
        self._handler = LoggingHandler(logger_provider=self._logs)
        self._logger = logging.getLogger("scribnic")
        previous_level = self._logger.level
        self._logger.setLevel(logging.INFO)
        self._cleanup.callback(self._logger.setLevel, previous_level)
        self._logger.addHandler(self._handler)
        self._cleanup.callback(self._handler.close)
        self._cleanup.callback(self._logger.removeHandler, self._handler)
        if self.progress is not None:
            console = logging.StreamHandler(self.progress)
            console.setFormatter(logging.Formatter("  %(message)s"))
            self._logger.addHandler(console)
            self._cleanup.callback(console.close)
            self._cleanup.callback(self._logger.removeHandler, console)

    def __exit__(self, *exc: object) -> None:
        """Flush telemetry, restore the previous providers and close owned workers."""
        try:
            self._traces.force_flush()
            self._metrics.force_flush()
            self._logs.force_flush()
        finally:
            if self._token is not None:
                _providers.reset(self._token)
                self._token = None
            self._cleanup.close()


def speech_metrics(
    step: "StepConfig", *, elapsed_seconds: float, audio_seconds: float
) -> None:
    """Export comparable speech duration and RTF through the caller's OTel SDK."""
    attributes = {
        "scribnic.model.name": step.model.name,
        "scribnic.runtime.engine": step.runtime.engine,
        "scribnic.speech.mode": step.model.settings.get("mode", "offline"),
    }
    for name, value, unit in (
        ("processing.duration", elapsed_seconds, "s"),
        ("audio.duration", audio_seconds, "s"),
        ("real_time_factor", elapsed_seconds / audio_seconds, "1"),
    ):
        meter().create_histogram("scribnic.speech." + name, unit=unit).record(
            value, attributes
        )
