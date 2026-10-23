"""Convert explicit speech response schemas into passive Scribnic artifacts."""

import re

from ..artifacts import DiarizedTranscript, SpeakerSegment, TextSegment, Transcript

END_TOLERANCE_SECONDS = 0.5


def parse_segments(
    data: dict, *, duration_seconds: float, diarized: bool = False
) -> Transcript | DiarizedTranscript:
    """Preserve segment timestamps and optional MOSS speaker prefixes.

    A half-second endpoint estimation tolerance preserves native output. Missing
    identities remain unknown; the result format never implies clinical roles.
    """
    segments = []
    previous_start = -1.0
    try:
        for item in data["segments"]:
            start, end, text = item["start"], item["end"], item["text"]
            if not isinstance(text, str):
                raise ValueError("Speech returned non-text segment content.")
            if start < previous_start or end > duration_seconds + END_TOLERANCE_SECONDS:
                raise ValueError("Speech returned out-of-order or out-of-range times.")
            previous_start = start
            if not text.strip():
                continue
            if diarized:
                prefix = re.match(r"^\s*\[(S\d+)\]\s*", text)
                speaker = prefix[1] if prefix else item.get("speaker")
                if prefix:
                    text = text[prefix.end() :]
                segments.append(SpeakerSegment(text.strip(), speaker, start, end))
            else:
                segments.append(TextSegment(text.strip(), start, end))
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid speech verbose_json segments.") from error
    if data.get("text", "").strip() and not segments:
        raise ValueError("Speech returned text without requested timestamped segments.")
    return (
        DiarizedTranscript(tuple(segments)) if diarized else Transcript(tuple(segments))
    )


def parse_words(
    data: dict,
    *,
    duration_seconds: float,
    diarized: bool,
    sentence_gap_seconds: float = 1.0,
) -> Transcript | DiarizedTranscript:
    """Group native words into turns, preserving unknown speaker zero and overlaps."""
    segments: list[SpeakerSegment] = []
    try:
        for row in data["words"]:
            word = row["word"].strip()
            if not word:
                continue
            identity = row.get("speaker")
            speaker = str(identity) if identity not in (None, 0, "0") else None
            current = SpeakerSegment(
                word, speaker, float(row["start"]), float(row["end"])
            )
            previous = segments[-1] if segments else None
            if current.end > duration_seconds + END_TOLERANCE_SECONDS:
                raise ValueError("Speech returned timestamps outside the recording.")
            if previous and current.start < previous.start:
                raise ValueError("Speech returned out-of-order word timestamps.")
            if (
                previous
                and previous.speaker == speaker
                and current.start - previous.end <= sentence_gap_seconds
            ):
                segments[-1] = SpeakerSegment(
                    previous.text + " " + word,
                    speaker,
                    previous.start,
                    max(previous.end, current.end),
                )
            else:
                segments.append(current)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Invalid speech word transcription.") from error
    if data.get("text", "").strip() and not segments:
        raise ValueError("Speech returned text without requested word timestamps.")
    result = DiarizedTranscript(tuple(segments))
    return result if diarized else result.without_speakers()
