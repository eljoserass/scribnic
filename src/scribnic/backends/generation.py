"""Optional LM Studio text generation."""

import os
import re

from ..models import Utterance
from ..prompts import TRANSCRIPT_RULES


def format_conversation(utterances: tuple[Utterance, ...]) -> str:
    """Give small models readable dialogue without duplicated timeline metadata."""
    return "\n".join(
        f"[{turn.speaker or 'hablante sin atribuir'}] {turn.text.strip()}"
        for turn in utterances
    )


class LMStudioTextGenerator:
    def __init__(
        self, model: str, base_url: str = "http://localhost:1234/v1", *,
        max_tokens: int = 768, max_input_chars: int = 12000,
    ):
        from openai import OpenAI

        self.client = OpenAI(
            base_url=base_url, api_key=os.getenv("LM_STUDIO_API_KEY") or "lm-studio",
            timeout=120, max_retries=0,
        )
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.max_input_chars = max_input_chars
        if max_tokens < 1 or max_input_chars < 1:
            raise ValueError("LLM token and input limits must be positive.")

    def check_available(self) -> str:
        """Fail before speech inference; auto selects only the requested Bonsai family."""
        from openai import OpenAIError

        try:
            models = [model.id for model in self.client.with_options(timeout=5).models.list()]
        except OpenAIError as exc:
            raise ValueError(
                f"Cannot query the note server at {self.base_url}. Start LM Studio's server "
                "and check --base-url (Windows/WSL may need the Windows host IP)."
            ) from exc
        if self.model == "auto":
            required = {"ternary", "bonsai", "2", "2b"}
            matches = [model for model in models
                       if required <= set(re.sub(r"[^a-z0-9]+", " ", model.lower()).split())]
            if len(matches) != 1:
                raise ValueError(
                    "Expected one Ternary Bonsai 2 2B model. "
                    "Load it in LM Studio or choose its exact ID with --llm-model. "
                    f"Available IDs: {', '.join(models) or '(none)'}"
                )
            self.model = matches[0]
        elif self.model not in models:
            raise ValueError(
                f"Note model {self.model!r} is unavailable. "
                f"Available IDs: {', '.join(models) or '(none)'}"
            )
        return self.model

    def generate(self, utterances: tuple[Utterance, ...], instruction: str) -> str:
        if not utterances:
            return "No hay transcripción para generar texto."
        conversation = format_conversation(utterances)
        if len(conversation) > self.max_input_chars:
            raise ValueError(
                f"Transcript has {len(conversation)} characters; the simple note limit is "
                f"{self.max_input_chars}. Use a shorter recording or increase "
                "--llm-max-input-chars after checking the server context. Nothing was truncated."
            )
        from openai import OpenAIError

        try:
            response = self.client.chat.completions.create(
                model=self.model, temperature=0.2, max_tokens=self.max_tokens,
                reasoning_effort="none",
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                messages=[
                    {"role": "system", "content": instruction + "\n" + TRANSCRIPT_RULES},
                    {"role": "user", "content": "TRANSCRIPCIÓN:\n" + conversation},
                ],
            )
        except OpenAIError as exc:
            raise ValueError(f"Note generation failed at {self.base_url}: {exc}") from exc
        if not response.choices:
            raise ValueError("The generator returned no choices.")
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise ValueError(
                "The note hit its output limit. Disable thinking in LM Studio or "
                "increase --llm-max-tokens; an incomplete note was not accepted."
            )
        content = choice.message.content
        if not content or not content.strip():
            raise ValueError("The generator returned an empty response. Check that thinking is off.")
        if "<think>" in content or "</think>" in content:
            raise ValueError("The model returned thinking text. Disable thinking in LM Studio.")
        return content.strip()
