"""Own a protocol client and the selected service manager for one session."""

import os
from time import perf_counter
from typing import TYPE_CHECKING

from ..config import StepConfig
from ..downloads import PreparedStep
from ..runtimes import create_runtime
from ..telemetry import tracer

if TYPE_CHECKING:
    from openai import OpenAI


class OpenAIConnection:
    """Retain one SDK client and release only the service processes we own."""

    def __init__(self, step: StepConfig, prepared: PreparedStep) -> None:
        """Configure the service without importing model libraries or loading weights."""
        self.model_name = step.model.name
        self.settings = step.runtime.settings
        self.runtime = create_runtime(step, prepared)
        self.client: OpenAI | None = None

    def start(self) -> float:
        """Start the selected service and client once; return initial loading seconds."""
        if self.client is not None:
            return 0.0
        from openai import OpenAI

        started = perf_counter()
        key_name = str(self.settings.get("api_key_env", "SCRIBNIC_API_KEY"))
        with tracer().start_as_current_span(
            "scribnic.service.load",
            attributes={"scribnic.model.name": self.model_name},
        ):
            base_url = self.runtime.start()
            try:
                self.client = OpenAI(
                    base_url=base_url,
                    api_key=os.getenv(key_name) or "local",
                    timeout=float(self.settings.get("timeout", 600)),
                    max_retries=0,
                )
            except (ValueError, OSError):
                self.runtime.close()
                raise
        return perf_counter() - started

    def close(self) -> None:
        """Release the HTTP client and stop an owned service, leaving external ones."""
        try:
            if self.client is not None:
                self.client.close()
                self.client = None
        finally:
            self.runtime.close()

    def check_model(self, model: str) -> None:
        """Check endpoint availability and explicit model identity when configured."""
        from openai import OpenAIError

        if not self.settings.get("check_models", True):
            return
        try:
            models = self.client.with_options(timeout=5).models.list()
        except OpenAIError as error:
            raise ValueError(
                f"Cannot reach inference endpoint {self.client.base_url}: {error}"
            ) from error
        identifiers = [item.id for item in models]
        if model not in identifiers:
            raise ValueError(
                f"Model {model!r} is unavailable. Available IDs: {identifiers}"
            )
