"""OpenTelemetry instrumentation and application-owned local export."""

import logging
import os
import platform
from contextlib import ExitStack
from contextvars import ContextVar
from pathlib import Path
from typing import TextIO

from opentelemetry import metrics, trace

_providers: ContextVar[tuple | None] = ContextVar("scribnic_otel_providers", default=None)


def tracer():
    providers = _providers.get()
    if providers is not None:
        return providers[0].get_tracer("scribnic", "0.1.0")
    return trace.get_tracer("scribnic", "0.1.0")


def meter():
    providers = _providers.get()
    if providers is not None:
        return providers[1].get_meter("scribnic", "0.1.0")
    return metrics.get_meter("scribnic", "0.1.0")


def trace_id() -> str | None:
    current = trace.get_current_span().get_span_context()
    return f"{current.trace_id:032x}" if current.is_valid else None


def _progress_processor(stream: TextIO):
    from opentelemetry.sdk.trace import SpanProcessor

    class ProgressProcessor(SpanProcessor):
        def on_start(self, span, parent_context=None):
            if span.name == "scribnic.step.run":
                name = span.attributes.get("scribnic.step.name", "step")
                print(f"  … {name}", file=stream, flush=True)

        def on_end(self, span):
            if span.name == "scribnic.step.run":
                name = span.attributes.get("scribnic.step.name", "step")
                seconds = (span.end_time - span.start_time) / 1e9
                mark = "✗" if span.status.status_code.name == "ERROR" else "✓"
                print(f"  {mark} {name} · {seconds:.2f} s", file=stream, flush=True)

    return ProgressProcessor()


class Telemetry:
    """Export real OTLP JSONL locally, without replacing a host's global SDK."""

    def __init__(self, directory: str | Path, *, progress: TextIO | None = None):
        self.directory = Path(directory)
        self.progress = progress
        self._token = None
        self._instrumentor = None
        self._cleanup = ExitStack()

    def __enter__(self) -> "Telemetry":
        from opentelemetry.sdk.resources import Resource

        self.directory.mkdir(parents=True, exist_ok=True)
        resource = Resource.create({
            "service.name": "scribnic",
            "service.version": "0.1.0",
            "process.pid": os.getpid(),
            "os.type": platform.system().lower(),
            "host.arch": platform.machine(),
        })
        try:
            self._configure_traces(resource)
            self._configure_metrics(resource)
            self._configure_logs(resource)
        except Exception:
            self._cleanup.close()
            raise
        self._token = _providers.set((self._traces, self._metrics))
        return self

    def _configure_traces(self, resource) -> None:
        from opentelemetry.exporter.otlp.json.file import FileSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        self._traces = TracerProvider(resource=resource)
        self._cleanup.callback(self._traces.shutdown)
        self._traces.add_span_processor(BatchSpanProcessor(
            FileSpanExporter(self.directory / "traces.jsonl")
        ))
        if self.progress is not None:
            self._traces.add_span_processor(_progress_processor(self.progress))

    def _configure_metrics(self, resource) -> None:
        from opentelemetry.exporter.otlp.json.file import FileMetricExporter
        from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        reader = PeriodicExportingMetricReader(
            FileMetricExporter(self.directory / "metrics.jsonl"),
            export_interval_millis=500,
        )
        self._metrics = MeterProvider(resource=resource, metric_readers=[reader])
        self._cleanup.callback(self._metrics.shutdown)
        self._instrumentor = SystemMetricsInstrumentor(config={
            "system.cpu.utilization": ["user", "system", "idle"],
            "system.memory.usage": ["used", "free"],
            "process.cpu.time": ["user", "system"],
            "process.memory.usage": None,
            "process.thread.count": None,
        })
        self._instrumentor.instrument(meter_provider=self._metrics)
        self._cleanup.callback(self._instrumentor.uninstrument)

    def _configure_logs(self, resource) -> None:
        from opentelemetry.exporter.otlp.json.file._log_exporter import FileLogExporter
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor

        self._logs = LoggerProvider(resource=resource)
        self._cleanup.callback(self._logs.shutdown)
        self._logs.add_log_record_processor(SimpleLogRecordProcessor(
            FileLogExporter(self.directory / "logs.jsonl")
        ))
        self._handler = LoggingHandler(logger_provider=self._logs)
        self._logger = logging.getLogger("scribnic")
        self._logger.addHandler(self._handler)
        self._cleanup.callback(self._handler.close)
        self._cleanup.callback(self._logger.removeHandler, self._handler)

    def __exit__(self, *exc: object) -> None:
        try:
            self._traces.force_flush()
            self._metrics.force_flush()
            self._logs.force_flush()
        finally:
            if self._token is not None:
                _providers.reset(self._token)
                self._token = None
            self._cleanup.close()
