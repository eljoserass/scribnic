"""Optional PyTorch operator traces, kept outside normal inference imports."""

from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path

_directory: ContextVar[Path | None] = ContextVar("scribnic_profile_directory", default=None)


@contextmanager
def profile_directory(directory: Path | None):
    token = _directory.set(directory)
    try:
        yield
    finally:
        _directory.reset(token)


def _module_ranges(model, torch, stack: ExitStack) -> None:
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


@contextmanager
def torch_profile(model, torch, device):
    directory = _directory.get()
    metadata = {}
    if directory is None:
        yield metadata
        return
    directory.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        supported = torch.profiler.supported_activities()
        if torch.profiler.ProfilerActivity.CUDA in supported:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
    trace_path = directory / "moss-torch-trace.json"
    metal = torch.mps.profiler.profile() if device.type == "mps" else nullcontext()
    with ExitStack() as stack:
        _module_ranges(model, torch, stack)
        profiler = stack.enter_context(torch.profiler.profile(
            activities=activities, record_shapes=True, profile_memory=True
        ))
        with metal:
            yield metadata
    profiler.export_chrome_trace(str(trace_path))
    metadata.update({
        "trace_path": str(trace_path),
        "format": "chrome_trace",
        "activities": [activity.name for activity in activities],
        "scope": "MOSS frontend, encoder, adaptor, prefill and decode",
    })
    if device.type == "mps":
        metadata["metal_capture"] = "Use Instruments to capture the emitted Metal signposts."
