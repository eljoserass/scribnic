"""Select a service manager independently of the inference protocol client."""

from ..config import StepConfig
from ..downloads import PreparedStep
from .endpoint import ExternalEndpoint
from .llama_cpp import LlamaServer
from .nemo_cpp import NemoServer
from .server import ManagedServer
from .whisper_cpp import WhisperServer


def create_runtime(
    step: StepConfig, prepared: PreparedStep
) -> ManagedServer | ExternalEndpoint:
    """Create an unstarted service manager for the configured engine and assets."""
    engine = step.runtime.engine
    if engine == "external_endpoint":
        return ExternalEndpoint(step.runtime)
    if engine == "llama_cpp":
        alias = str(step.model.settings.get("served_model", step.model.name))
        return LlamaServer(step.runtime, prepared, alias)
    if engine == "whisper_cpp":
        return WhisperServer(step.runtime, prepared)
    if engine == "nemo_cpp":
        return NemoServer(step.runtime, prepared)
    raise ValueError(f"No managed service for runtime {engine!r}.")
