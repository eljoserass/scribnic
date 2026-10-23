"""Commands and native metrics for a pinned llama-server distribution."""

import platform
import urllib.request

from .server import ManagedServer


class LlamaServer(ManagedServer):
    """A llama-server distribution selected explicitly by its runtime preset."""

    def _command(self, port: int) -> list[str]:
        """Build llama-server arguments independently of the selected LLM weights."""
        options = self.runtime.settings
        gpu_layers = options.get("gpu_layers", "auto")
        if gpu_layers == "auto":
            gpu_layers = "all" if platform.system() == "Darwin" else "0"
        return [
            str(self.prepared.binary),
            "-m",
            str(self.prepared.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-c",
            str(options.get("context_size", 8192)),
            "-np",
            "1",
            "-ngl",
            str(gpu_layers),
            "-t",
            str(options.get("threads", 6)),
            "-tb",
            str(options.get("threads_batch", options.get("threads", 6))),
            "--alias",
            self.alias,
            "--metrics",
            "--perf",
        ]

    def metadata(self) -> dict[str, object]:
        """Include the requested accelerator offload setting."""
        return super().metadata() | {
            "gpu_layers_requested": self.command[self.command.index("-ngl") + 1]
        }

    def metrics(self) -> str | None:
        """Fetch the server's native Prometheus snapshot when available."""
        endpoint = self.base_url.removesuffix("/v1") + "/metrics"
        try:
            with urllib.request.urlopen(endpoint, timeout=2) as response:
                return response.read().decode("utf-8")
        except OSError:
            return None
