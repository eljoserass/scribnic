"""NVIDIA result conversion and observability shared by native and NeMo engines."""

from ..artifacts import DiarizedTranscript, SpeakerSegment, Transcript
from ..config import StepConfig
from ..downloads import PreparedStep, sha256_file
from ..observability import meter


def parse_speech(
    payload: dict | list,
    *,
    duration_seconds: float,
    diarized: bool,
    sentence_gap_seconds: float = 1.0,
) -> Transcript | DiarizedTranscript:
    """Convert native word JSON or NeMo SegLST, preserving original audio times.

    Unknown native speaker zero stays unknown; SegLST speaker zero is an identity.
    Native words are grouped into turns by speaker and pause. SegLST turns are
    preserved, including overlapping speakers. Invalid or truncated times fail.
    """
    if isinstance(payload, list):
        segments = _seglst_segments(payload)
    elif isinstance(payload, dict) and isinstance(payload.get("words"), list):
        segments = _word_segments(payload["words"], sentence_gap_seconds)
        if payload.get("text", "").strip() and not segments:
            raise ValueError("NVIDIA returned text without word timestamps.")
    else:
        raise ValueError("Expected NVIDIA word JSON or NeMo SegLST.")
    segments.sort(key=lambda segment: segment.start)
    if any(segment.end > duration_seconds + 0.5 for segment in segments):
        raise ValueError("NVIDIA returned timestamps outside the source recording.")
    result = DiarizedTranscript(tuple(segments))
    return result if diarized else result.without_speakers()


def _seglst_segments(rows: list[dict]) -> list[SpeakerSegment]:
    """Preserve upstream SegLST identities, text, and second-based bounds."""
    try:
        return [
            SpeakerSegment(
                row["words"].strip(),
                str(row["speaker"]) if row.get("speaker") is not None else None,
                float(row["start_time"]),
                float(row["end_time"]),
            )
            for row in rows
            if row["words"].strip()
        ]
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Invalid NeMo SegLST output.") from error


def _word_segments(rows: list[dict], gap_seconds: float) -> list[SpeakerSegment]:
    """Join adjacent timestamped words without merging different speakers."""
    segments: list[SpeakerSegment] = []
    try:
        for row in rows:
            word = row["word"].strip()
            if not word:
                continue
            speaker = row.get("speaker")
            label = str(speaker) if speaker not in (None, 0, "0") else None
            current = SpeakerSegment(
                word, label, float(row["start"]), float(row["end"])
            )
            previous = segments[-1] if segments else None
            if previous and current.start < previous.start:
                raise ValueError("NVIDIA returned out-of-order word timestamps.")
            if (
                previous
                and previous.speaker == label
                and current.start - previous.end <= gap_seconds
            ):
                segments[-1] = SpeakerSegment(
                    previous.text + " " + word,
                    label,
                    previous.start,
                    max(previous.end, current.end),
                )
            else:
                segments.append(current)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Invalid NVIDIA word transcription.") from error
    return segments


def model_provenance(step: StepConfig, prepared: PreparedStep) -> dict:
    """Describe actual prepared weights, including the companion diarizer."""
    models = {"asr": (step.model, prepared.model_path)}
    models.update(
        {
            role: (model, prepared.companions[role])
            for role, model in step.companions.items()
        }
    )
    return {
        role: {
            "name": model.name,
            "repository": model.repository,
            "revision": model.revision,
            "filename": path.name,
            "sha256": sha256_file(path),
            "path": str(path),
        }
        for role, (model, path) in models.items()
    }


def speech_metrics(
    step: StepConfig, *, elapsed_seconds: float, audio_seconds: float
) -> None:
    """Export comparable speech duration and RTF through the caller's OTel SDK."""
    attributes = {
        "scribnic.model.name": step.model.name,
        "scribnic.runtime.engine": step.runtime.engine,
        "scribnic.speech.mode": step.model.settings.get("mode", "offline"),
    }
    for name, value, unit in (
        ("processing.duration", elapsed_seconds, "s"),
        ("audio.duration", audio_seconds, "s"),
        ("real_time_factor", elapsed_seconds / audio_seconds, "1"),
    ):
        meter().create_histogram("scribnic.speech." + name, unit=unit).record(
            value, attributes
        )
