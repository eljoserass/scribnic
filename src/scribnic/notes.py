"""Prepare concise note instructions and render only the supplied conversation."""

from .artifacts import DiarizedTranscript, Transcript
from .config import StepConfig


def note_instruction(step: StepConfig) -> str:
    if step.instruction is not None:
        return step.instruction
    if step.template is None:
        raise ValueError("A note-generation step needs a note template.")
    return step.template.read_text(encoding="utf-8").strip()


def format_conversation(source: Transcript | DiarizedTranscript) -> str:
    if isinstance(source, Transcript):
        return source.text
    lines = []
    for segment in source.segments:
        label = segment.speaker or "hablante sin atribuir"
        if segment.role:
            label += f"; {segment.role}"
        lines.append(f"[{label}] {segment.text}")
    return "\n".join(lines)
