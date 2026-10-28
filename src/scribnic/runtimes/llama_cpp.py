"""Commands and native metrics for a pinned llama-server distribution."""

import math
import platform
import re
import urllib.request

from .server import ManagedServer


def server_arguments(options: object) -> list[str]:
    """Translate TOML server_options to argv without a shell.

    Keys are long llama-server option names without leading dashes. Booleans
    select --name or --no-name; strings and finite numbers supply one value.
    Scribnic owns model paths, networking and the explicitly managed settings.
    """
    if not isinstance(options, dict):
        raise ValueError("llama_cpp.server_options must be a TOML table.")
    owned = {
        "model",
        "model-url",
        "hf-repo",
        "hf-file",
        "mmproj",
        "mmproj-url",
        "host",
        "port",
        "alias",
        "parallel",
        "ctx-size",
        "threads",
        "threads-batch",
        "gpu-layers",
        "n-gpu-layers",
        "metrics",
        "perf",
    }
    arguments = []
    for name, value in options.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", name):
            raise ValueError("server_options keys must be long option names.")
        if name.removeprefix("no-") in owned:
            raise ValueError(f"Cannot override Scribnic-managed --{name}.")
        if isinstance(value, bool):
            arguments.append("--" + (name if value else "no-" + name))
        elif isinstance(value, (str, int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"server_options.{name} must be finite.")
            arguments.extend(("--" + name, str(value)))
        else:
            raise ValueError(f"server_options.{name} needs a scalar value.")
    return arguments


class LlamaServer(ManagedServer):
    """A llama-server distribution selected explicitly by its runtime preset."""

    def _command(self, port: int) -> list[str]:
        """Build llama-server arguments independently of the selected LLM weights."""
        options = self.runtime.settings
        gpu_layers = options.get("gpu_layers", "auto")
        if gpu_layers == "auto":
            gpu_layers = "all" if platform.system() == "Darwin" else "0"
        command = [
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
        if "mmproj" in self.prepared.companions:
            command.extend(("--mmproj", str(self.prepared.companions["mmproj"])))
        return command + server_arguments(options.get("server_options", {}))

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
