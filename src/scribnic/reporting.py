"""Readable terminal output and local run artifacts, without a web service."""

import html
import json
import os
from pathlib import Path
from string import Template
from typing import TextIO
from urllib.parse import quote

from .artifacts import AudioFile, DiarizedTranscript, RunResult


def _report_cards(result: RunResult) -> str:
    transcript = result.transcript
    segments = len(transcript.segments) if transcript else 0
    speakers = set()
    if isinstance(transcript, DiarizedTranscript):
        speakers = {
            segment.speaker for segment in transcript.segments if segment.speaker
        }
    values = [
        (f"{result.elapsed_seconds:.2f} s", "Tiempo total"),
        (str(segments), "Segmentos"),
        (str(len(speakers)), "Hablantes con etiqueta"),
    ]
    if isinstance(result.source, AudioFile) and result.source.duration is not None:
        values.insert(0, (f"{result.source.duration:.1f} s", "Audio"))
    return "".join(
        f'<div class="metric"><b>{value}</b><small>{label}</small></div>'
        for value, label in values
    )


def _report_turns(result: RunResult) -> str:
    if result.transcript is None:
        return '<p class="muted">No hay transcripción disponible.</p>'
    rows = []
    colors = {}
    for segment in result.transcript.segments:
        speaker = getattr(segment, "speaker", None) or "Sin etiqueta"
        color = colors.setdefault(speaker, len(colors) % 4)
        bounds = ""
        if segment.start is not None:
            bounds = f"{timestamp(segment.start)} — {timestamp(segment.end)}"
        rows.append(
            f'<article class="turn speaker-{color}"><div class="turn-meta">'
            f"<strong>{html.escape(speaker)}</strong><time>{bounds}</time></div>"
            f"<p>{html.escape(segment.text)}</p></article>"
        )
    return "".join(rows) or '<p class="muted">No se detectó voz.</p>'


def _report_steps(result: RunResult) -> str:
    rows = []
    for step in result.steps:
        timings = step.metadata.get("timings") or {}
        prompt = timings.get("prompt_per_second")
        decode = timings.get("predicted_per_second")
        prompt_text = f"{prompt:.1f} tok/s" if prompt else "—"
        decode_text = f"{decode:.1f} tok/s" if decode else "—"
        rows.append(
            f"<tr><td>{html.escape(step.name)}</td><td>{step.elapsed_seconds:.2f} s</td>"
            f"<td>{prompt_text}</td><td>{decode_text}</td></tr>"
        )
    return "".join(rows)


def _report_resources(result: RunResult) -> str:
    rows = []
    for process in result.resources.get("processes", []):
        gib = process["peak_sampled_rss_bytes"] / 1024**3
        role = html.escape(process["role"])
        cores = process["average_cpu_cores"]
        rows.append(
            f"<tr><td>{role} · PID {process['pid']}</td>"
            f"<td>{gib:.2f} GiB</td><td>{cores:.2f}</td></tr>"
        )
    return "".join(rows)


def _relative_url(path: Path, directory: Path) -> str:
    try:
        return quote(Path(os.path.relpath(path, directory)).as_posix())
    except ValueError:
        return path.resolve().as_uri()


def _report_links(result: RunResult, directory: Path) -> str:
    paths = [Path("result.json"), Path("run.json")]
    paths.extend(Path(f"{artifact.kind}.json") for artifact in result.artifacts)
    telemetry = directory / "telemetry"
    if telemetry.is_dir():
        paths.extend(
            path.relative_to(directory) for path in sorted(telemetry.glob("*.jsonl"))
        )
    paths.extend(
        path.relative_to(directory)
        for path in sorted(directory.glob("profiles/*.json"))
    )
    return " · ".join(
        f'<a href="{html.escape(path.as_posix(), quote=True)}">{html.escape(path.name)}</a>'
        for path in paths
    )


def render_workflow_report(result: RunResult, directory: Path) -> str:
    assets = Path(__file__).parent / "assets"
    template = Template((assets / "report.html").read_text(encoding="utf-8"))
    audio = ""
    if isinstance(result.source, AudioFile) and result.source.path.is_file():
        url = html.escape(_relative_url(result.source.path, directory), quote=True)
        audio = f'<audio controls preload="none" src="{url}"></audio>'
    error = f'<p class="error">{html.escape(result.error)}</p>' if result.error else ""
    status = "DEMO · salidas inventadas" if result.workflow == "demo" else result.status
    return template.substitute(
        styles=(assets / "report.css").read_text(encoding="utf-8"),
        workflow=html.escape(result.workflow),
        status=html.escape(status),
        started_at=html.escape(result.started_at),
        run_id=result.run_id,
        trace_id=result.trace_id or "Sin exportador configurado",
        cards=_report_cards(result),
        error=error,
        note=html.escape(result.note.text) if result.note else "Nota pendiente.",
        audio=audio,
        transcript=_report_turns(result),
        steps=_report_steps(result),
        resources=_report_resources(result),
        links=_report_links(result, directory),
        configuration=html.escape(
            json.dumps(result.configuration, ensure_ascii=False, indent=2)
        ),
    )


def timestamp(seconds: float) -> str:
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes:02}:{remainder:02}"


def print_result(result: RunResult, stream: TextIO) -> None:
    """Present completed artifacts; timestamps/roles are shown only when supplied."""
    if result.workflow == "demo":
        print("DEMO · salidas inventadas", file=stream)
    if result.transcript is not None:
        print("\nTRANSCRIPCIÓN\n", file=stream)
        for segment in result.transcript.segments:
            speaker = getattr(segment, "speaker", None)
            label = f"[{speaker}] " if speaker else ""
            bounds = f"{timestamp(segment.start)} " if segment.start is not None else ""
            print(f"{bounds}{label}{segment.text}", file=stream)
    if result.note is not None:
        print(f"\nNOTA\n\n{result.note.text}", file=stream)
    print(f"\n{result.elapsed_seconds:.2f} s · {result.run_id}", file=stream)
