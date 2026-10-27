"""Expose OpenTelemetry instrumentation without loading model profilers or a UI."""

from .otel import GenerationProgress, Telemetry, meter, speech_metrics, trace_id, tracer

__all__ = [
    "GenerationProgress",
    "Telemetry",
    "meter",
    "speech_metrics",
    "trace_id",
    "tracer",
]
