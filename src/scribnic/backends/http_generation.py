"""OpenAI-compatible note generation for local or explicitly remote endpoints."""

import os
from time import perf_counter

from ..artifacts import Artifact, DiarizedTranscript, Note, StepOutput, Transcript
from ..config import StepConfig
from ..downloads import PreparedStep
from ..notes import format_conversation, note_instruction
from ..runtimes import LlamaServer


class HTTPGeneration:
    def __init__(self, step: StepConfig, prepared: PreparedStep):
        from openai import OpenAI

        self.step = step
        self.server: LlamaServer | None = None
        self.model = str(step.model.settings.get("served_model", step.model.name))
        self.load_seconds = 0.0
        started = perf_counter()
        base_url = self._endpoint(prepared)
        settings = step.runtime.settings
        key_name = str(settings.get("api_key_env", "SCRIBNIC_API_KEY"))
        self.client = OpenAI(
            base_url=base_url,
            api_key=os.getenv(key_name) or "local",
            timeout=float(settings.get("timeout", 120)),
            max_retries=0,
        )
        try:
            self._check_available()
        except Exception:
            self.close()
            raise
        self.load_seconds = perf_counter() - started

    def _endpoint(self, prepared: PreparedStep) -> str:
        if self.step.runtime.engine == "llama_cpp":
            self.server = LlamaServer(self.step.runtime, prepared, self.model)
            return self.server.start()
        if self.step.runtime.engine != "http":
            raise ValueError(f"Unsupported generation runtime: {self.step.runtime.engine}")
        settings = self.step.runtime.settings
        environment = str(settings.get("base_url_env", "SCRIBNIC_BASE_URL"))
        base_url = os.getenv(environment) or settings.get("base_url")
        if not base_url:
            raise ValueError("An HTTP runtime needs a base_url.")
        return str(base_url)

    def _check_available(self) -> None:
        from openai import OpenAIError

        if not self.step.runtime.settings.get("check_models", True):
            return
        try:
            models = self.client.with_options(timeout=5).models.list()
            identifiers = [model.id for model in models]
        except OpenAIError as error:
            raise ValueError(f"Cannot reach note endpoint {self.client.base_url}: {error}") from error
        if self.model not in identifiers:
            raise ValueError(
                f"Note model {self.model!r} is unavailable. Available IDs: {identifiers}"
            )

    def _request(self, conversation: str) -> dict:
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
        for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "seed"):
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
        from openai import OpenAIError

        if not isinstance(source, (Transcript, DiarizedTranscript)):
            raise ValueError("HTTP note generation expects a plain or diarized transcript.")
        conversation = format_conversation(source)
        if not conversation.strip():
            raise ValueError("Cannot generate a note from an empty transcript.")
        try:
            response = self.client.chat.completions.create(**self._request(conversation))
        except OpenAIError as error:
            raise ValueError(f"Note generation failed: {error}") from error
        note = self._content(response)
        metadata = {
            "model": response.model,
            "base_url": str(self.client.base_url),
            "runtime": self.server.metadata() if self.server else {"engine": "http"},
            "usage": response.usage.model_dump() if response.usage else None,
            "timings": response.model_extra.get("timings"),
            "load_seconds": self.load_seconds,
            "server_metrics": self.server.metrics() if self.server else None,
        }
        self.load_seconds = 0.0
        return StepOutput(Note(note), metadata)

    def process_ids(self) -> tuple[int, ...]:
        if self.server is not None and self.server.process is not None:
            return (self.server.process.pid,)
        return ()

    @staticmethod
    def _content(response) -> str:
        if not response.choices:
            raise ValueError("The note model returned no choices.")
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise ValueError("The note reached its output limit; incomplete note rejected.")
        text = choice.message.content
        if not text or not text.strip():
            raise ValueError("The note model returned an empty response.")
        if "<think>" in text or "</think>" in text:
            raise ValueError("Thinking text appeared in the note. Check the runtime profile.")
        return text.strip()

    def close(self) -> None:
        try:
            if hasattr(self, "client"):
                self.client.close()
        finally:
            if self.server is not None:
                self.server.close()
