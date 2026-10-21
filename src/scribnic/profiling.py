"""Optional PyTorch operator traces, kept outside normal inference imports."""

import gzip
import logging
import shutil
import tempfile
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path
from time import monotonic

from .storage import write_json

_directory: ContextVar[Path | None] = ContextVar(
    "scribnic_profile_directory", default=None
)


@contextmanager
def profile_directory(directory: Path | None):
    token = _directory.set(directory)
    try:
        yield
    finally:
        _directory.reset(token)


def _module_ranges(model, torch, stack: ExitStack, advance=None) -> None:
    """Label existing forwards without changing the checkpoint implementation."""
    decoder_calls = 0

    def install(module, label):
        contexts = []

        def before(*args):
            nonlocal decoder_calls
            name = label
            if label == "moss.decoder":
                name = "moss.prefill" if decoder_calls == 0 else "moss.decode"
                decoder_calls += 1
            active = torch.profiler.record_function(name)
            active.__enter__()
            contexts.append(active)

        def after(*args):
            contexts.pop().__exit__(None, None, None)
            if label == "moss.decoder" and advance is not None:
                advance()

        stack.callback(module.register_forward_pre_hook(before).remove)
        stack.callback(module.register_forward_hook(after, always_call=True).remove)

    backbone = model.model
    for attribute, label in (
        ("whisper_encoder", "moss.encoder"),
        ("vq_adaptor", "moss.adaptor"),
        ("language_model", "moss.decoder"),
    ):
        module = getattr(backbone, attribute, None)
        if module is not None:
            install(module, label)


class _ProfileWindows:
    """Flush completed windows and aggregate statistics without retaining events."""

    def __init__(self, directory: Path, torch, steps: int, activities):
        self.directory = directory
        self.actions = torch.profiler.ProfilerAction
        self.steps = steps
        self.activities = [activity.name for activity in activities]
        self.traces: list[str] = []
        self.statistics: dict[str, dict] = {}
        self.summary_path = directory / "moss-profile.json"
        self.export_and_aggregation_seconds = 0.0

    def schedule(self, step: int):
        # Continuous coverage, including the first encoder/prefill and final tail.
        if (step + 1) % self.steps == 0:
            return self.actions.RECORD_AND_SAVE
        return self.actions.RECORD

    def flush(self, profiler) -> None:
        started = monotonic()
        name = f"moss-trace-{len(self.traces):04d}.json.gz"
        logging.getLogger(__name__).info(
            "Profiling MOSS · exportando bloque %d", len(self.traces) + 1
        )
        self._export_trace(profiler, self.directory / name)
        for event in profiler.key_averages():
            row = self.statistics.setdefault(
                event.key,
                {
                    "name": event.key,
                    "calls": 0,
                    "cpu_seconds": 0.0,
                    "self_cpu_seconds": 0.0,
                    "self_cpu_memory_net_bytes": 0,
                    "device_seconds": 0.0,
                },
            )
            row["calls"] += event.count
            row["cpu_seconds"] += event.cpu_time_total / 1e6
            row["self_cpu_seconds"] += event.self_cpu_time_total / 1e6
            row["self_cpu_memory_net_bytes"] += event.self_cpu_memory_usage
            row["device_seconds"] += getattr(event, "device_time_total", 0) / 1e6
        self.traces.append(name)
        self.export_and_aggregation_seconds += monotonic() - started
        self.save("running")
        logging.getLogger(__name__).info(
            "Profiling MOSS · bloque guardado · %.1f s", monotonic() - started
        )

    def _export_trace(self, profiler, destination: Path) -> None:
        # Stream with low compression; PyTorch's gzip exporter prioritizes size.
        with tempfile.NamedTemporaryFile(
            dir=self.directory, suffix=".json", delete=False
        ) as stream:
            temporary = Path(stream.name)
        try:
            profiler.export_chrome_trace(str(temporary))
            wait = getattr(profiler, "wait_for_exports", None)
            if wait is not None:
                wait()
            with temporary.open("rb") as source:
                with gzip.open(destination, "wb", compresslevel=1) as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        finally:
            temporary.unlink(missing_ok=True)

    def save(self, status: str) -> None:
        phases = []
        operators = []
        for name, row in self.statistics.items():
            if name.startswith("moss."):
                phases.append(row)
            elif not name.startswith("ProfilerStep"):
                operators.append(row)
        operators.sort(key=lambda row: row["self_cpu_seconds"], reverse=True)
        write_json(
            self.summary_path,
            {
                "schema_version": 1,
                "status": status,
                "scope": "MOSS frontend, encoder, adaptor, prefill and decode; model loading excluded",
                "activities": self.activities,
                "window_steps": self.steps,
                "record_shapes": True,
                "profile_memory": True,
                "with_stack": False,
                "export_and_aggregation_seconds": self.export_and_aggregation_seconds,
                "traces": self.traces,
                "phases": phases,
                "operators": operators,
                "memory_semantics": "Signed net tensor allocations observed per operator; not peak RSS or peak VRAM.",
                "timing_semantics": "Inclusive phase times may overlap; operator self times exclude children. Export overhead is outside captured windows.",
            },
        )


@contextmanager
def torch_profile(
    model, torch, device, *, include_gpu: bool = False, window_steps: int = 16
):
    directory = _directory.get()
    metadata = {}
    if directory is None:
        yield metadata
        return
    directory.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if include_gpu and device.type == "cuda":
        supported = torch.profiler.supported_activities()
        if torch.profiler.ProfilerActivity.CUDA in supported:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
    if type(window_steps) is not int or window_steps < 1:
        raise ValueError("profile_window_steps must be a positive integer.")
    windows = _ProfileWindows(directory, torch, window_steps, activities)
    metal = (
        torch.mps.profiler.profile()
        if include_gpu and device.type == "mps"
        else nullcontext()
    )
    profiler = torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        schedule=windows.schedule,
        on_trace_ready=windows.flush,
        acc_events=False,
    )
    completed = False
    try:
        with ExitStack() as stack:
            _module_ranges(model, torch, stack, profiler.step)
            stack.enter_context(profiler)
            with metal:
                yield metadata
                completed = True
    finally:
        windows.save("success" if completed else "error")
    metadata.update(
        {
            "trace_path": str(directory / windows.traces[0])
            if windows.traces
            else None,
            "summary_path": str(windows.summary_path),
            "trace_count": len(windows.traces),
            "format": "chrome_trace_gzip_windows",
            "activities": [activity.name for activity in activities],
            "scope": "MOSS frontend, encoder, adaptor, prefill and decode",
        }
    )
    if include_gpu and device.type == "mps":
        metadata["metal_capture"] = (
            "Use Instruments to capture the emitted Metal signposts."
        )
