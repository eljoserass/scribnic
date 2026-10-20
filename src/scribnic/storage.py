"""Versioned artifact JSON and durable per-run checkpoints."""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from .artifacts import (
    Artifact, AudioFile, DiarizedTranscript, Note, RunResult, SpeakerSegment,
    TextSegment, Transcript,
)

SCHEMA_VERSION = 1


def artifact_to_dict(artifact: Artifact) -> dict:
    data = {"schema_version": SCHEMA_VERSION, "kind": artifact.kind, **asdict(artifact)}
    if isinstance(artifact, AudioFile):
        data["path"] = str(artifact.path)
    return data


def artifact_from_dict(data: dict) -> Artifact:
    if data.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError("Unsupported artifact schema version.")
    kind = data.get("kind")
    if kind == "audio":
        return AudioFile(
            Path(data["path"]), data.get("sample_rate"), data.get("duration"), data.get("sha256")
        )
    if kind == "note":
        return Note(data["text"])
    if kind == "transcript":
        if "segments" not in data:
            return Transcript.from_text(data["text"])
        return Transcript(tuple(TextSegment(**segment) for segment in data["segments"]))
    if kind == "diarized_transcript":
        return DiarizedTranscript(tuple(
            SpeakerSegment(**segment) for segment in data["segments"]
        ))
    raise ValueError(f"Unsupported artifact kind: {kind!r}")


def _import_transcript(data: dict) -> Transcript | DiarizedTranscript:
    if "utterances" in data:
        segments = data["utterances"]
    elif "segments" in data:
        segments = data["segments"]
    elif isinstance(data.get("text"), str):
        return Transcript.from_text(data["text"])
    else:
        raise ValueError("Expected a Scribnic artifact or a transcript with text/segments.")
    if any("speaker" in segment for segment in segments):
        return DiarizedTranscript(tuple(
            SpeakerSegment(
                segment["text"], segment.get("speaker"), segment.get("start"),
                segment.get("end"), segment.get("role"),
            )
            for segment in segments
        ))
    return Transcript(tuple(
        TextSegment(segment["text"], segment.get("start"), segment.get("end"))
        for segment in segments
    ))


def load_artifact(path: str | Path) -> Artifact:
    """Read Scribnic JSON, plain text, or OpenAI-style transcript segments."""
    path = Path(path).expanduser().resolve(strict=True)
    if path.suffix.lower() == ".wav":
        return AudioFile(path)
    if path.suffix.lower() in {".txt", ".md"}:
        return Transcript.from_text(path.read_text(encoding="utf-8"))
    with path.open(encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    try:
        if "kind" in data:
            return artifact_from_dict(data)
        return _import_transcript(data)
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid artifact in {path}: {error}") from error


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2, default=str)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_artifact(artifact: Artifact, path: str | Path) -> None:
    write_json(Path(path), artifact_to_dict(artifact))


def result_to_dict(result: RunResult) -> dict:
    data = asdict(result)
    data["schema_version"] = SCHEMA_VERSION
    data["source"] = artifact_to_dict(result.source)
    data["artifacts"] = [artifact_to_dict(artifact) for artifact in result.artifacts]
    return data


class RunStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser().resolve()
        if self.directory.exists():
            existing = {path.name for path in self.directory.iterdir()}
            if existing - {"telemetry"}:
                raise ValueError(f"Output directory must be empty: {self.directory}")
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, result: RunResult) -> None:
        from .reporting import render_workflow_report

        for artifact in result.artifacts:
            save_artifact(artifact, self.directory / f"{artifact.kind}.json")
        if result.note is not None:
            (self.directory / "note.txt").write_text(result.note.text + "\n", encoding="utf-8")
        data = result_to_dict(result)
        write_json(self.directory / "result.json", data)
        metadata = {key: value for key, value in data.items() if key != "artifacts"}
        write_json(self.directory / "run.json", metadata)
        self._runtime_attachments(result)
        report = render_workflow_report(result, self.directory)
        (self.directory / "report.html").write_text(report, encoding="utf-8")

    def _runtime_attachments(self, result: RunResult) -> None:
        for record in result.steps:
            native_metrics = record.metadata.get("server_metrics")
            if native_metrics:
                (self.directory / f"{record.name}-metrics.prom").write_text(
                    str(native_metrics), encoding="utf-8"
                )
