"""Existing inference endpoints whose processes are owned by another application."""

import os
from urllib.parse import urlparse

from ..config import RuntimeConfig


class ExternalEndpoint:
    """Describe a service address without installing, launching or stopping it."""

    def __init__(self, runtime: RuntimeConfig) -> None:
        """Resolve the configured address; authentication remains in the client."""
        self.runtime = runtime
        environment = str(runtime.settings.get("base_url_env", "SCRIBNIC_BASE_URL"))
        self.base_url = os.getenv(environment) or runtime.settings.get("base_url")
        if not isinstance(self.base_url, str):
            raise ValueError("An external endpoint needs a base_url.")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an HTTP(S) inference endpoint.")

    def start(self) -> str:
        """Return the existing endpoint; the protocol client checks availability."""
        return self.base_url

    def metadata(self) -> dict[str, object]:
        """Identify external ownership without claiming local resource coverage."""
        return {
            "engine": "external_endpoint",
            "name": self.runtime.name,
            "base_url": self.base_url,
            "resource_scope": "external service; CPU/RSS require server telemetry",
        }

    def process_ids(self) -> tuple[int, ...]:
        """External processes are never claimed or sampled as owned resources."""
        return ()

    def metrics(self) -> str | None:
        """No universal metrics endpoint is assumed for external services."""
        return None

    def close(self) -> None:
        """Leave the external service running; no process is owned here."""
