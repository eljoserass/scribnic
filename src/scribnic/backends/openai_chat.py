"""OpenAI-compatible note generation for local or explicitly remote endpoints."""

from time import perf_counter
from typing import TYPE_CHECKING

from ..artifacts import Artifact, DiarizedTranscript, Note, StepOutput, Transcript
from ..config import StepConfig
from ..downloads import PreparedStep
from ..notes import format_conversation, note_instruction
from ..observability import tracer
from .connection import OpenAIConnection

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion


class OpenAIChat:
    """Generate notes through Chat Completions, retaining one connection per session."""

    def __init__(self, step: StepConfig, prepared: PreparedStep) -> None:
        """Configure weights, service and note settings without loading resources."""
        self.step = step
        self.connection = OpenAIConnection(step, prepared)
        self.model = str(step.model.settings.get("served_model", step.model.name))

    @property
    def client(self) -> "OpenAI | None":
        """Expose the initialized protocol client, or None before startup."""
        return self.connection.client

    def start(self) -> float:
        """Load/check the configured service once and return initial startup time."""
        started = perf_counter()
        elapsed = self.connection.start()
        if elapsed:
            try:
                self._check_available()
            except ValueError:
                self.close()
                raise
        return perf_counter() - started if elapsed else 0.0

    def _check_available(self) -> None:
        """Verify the selected model ID through the service model inventory."""
        self.connection.check_model(self.model)

    def _request(self, conversation: str) -> dict[str, object]:
        """Build a bounded note request with explicit model sampling settings."""
        settings = self.step.model.settings
        maximum = int(settings.get("max_input_chars", 12000))
        if len(conversation) > maximum:
            raise ValueError(
                f"Transcript has {len(conversation)} characters; limit is {maximum}. "
                "Nothing was truncated. Change the workflow model settings."
            )
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": note_instruction(self.step)},
                {"role": "user", "content": "TRANSCRIPCIÓN:\n" + conversation},
            ],
            "max_tokens": int(settings.get("max_tokens", 768)),
        }
        for key in (
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "seed",
        ):
            if key in settings:
                request[key] = settings[key]
        extra = dict(self.step.runtime.settings.get("request_extra", {}))
        for key in ("top_k", "min_p", "repeat_penalty", "repeat_last_n"):
            if key in settings:
                extra[key] = settings[key]
        if "thinking" in settings:
            extra["chat_template_kwargs"] = {"enable_thinking": settings["thinking"]}
        request["extra_body"] = extra
        return request

    def run(self, source: Artifact) -> StepOutput:
        """Generate from supplied text, loading a deferred service only when needed."""
        from openai import OpenAIError

        if not isinstance(source, (Transcript, DiarizedTranscript)):
            raise ValueError(
                "HTTP note generation expects a plain or diarized transcript."
            )
        conversation = format_conversation(source)
        if not conversation.strip():
            raise ValueError("Cannot generate a note from an empty transcript.")
        request = self._request(conversation)
        load_seconds = self.start()
        try:
            with tracer().start_as_current_span(
                "scribnic.note.request",
                attributes={"scribnic.model.name": self.step.model.name},
            ):
                response = self.client.chat.completions.create(**request)
        except OpenAIError as error:
            raise ValueError(f"Note generation failed: {error}") from error
        note = self._content(response)
        metadata = {
            "model": response.model,
            "base_url": str(self.client.base_url),
            "runtime": self.connection.runtime.metadata(),
            "usage": response.usage.model_dump() if response.usage else None,
            "timings": (response.model_extra or {}).get("timings"),
            "load_seconds": load_seconds,
            "server_metrics": self.connection.runtime.metrics(),
        }
        return StepOutput(Note(note), metadata)

    def process_ids(self) -> tuple[int, ...]:
        """Expose only the service processes owned by this session."""
        return self.connection.runtime.process_ids()

    @staticmethod
    def _content(response: "ChatCompletion") -> str:
        """Reject missing, truncated or thinking output rather than accepting a partial note."""
        if not response.choices:
            raise ValueError("The note model returned no choices.")
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise ValueError(
                "The note reached its output limit; incomplete note rejected."
            )
        text = choice.message.content
        if not text or not text.strip():
            raise ValueError("The note model returned an empty response.")
        if "<think>" in text or "</think>" in text:
            raise ValueError(
                "Thinking text appeared in the note. Check the runtime profile."
            )
        return text.strip()

    def close(self) -> None:
        """Release the client and the service processes owned by this session."""
        self.connection.close()
