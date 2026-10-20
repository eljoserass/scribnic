"""Readable terminal output and local run artifacts, without a web service."""

import html
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import TextIO
from urllib.parse import quote
from string import Template

from .models import PipelineResult
from .artifacts import AudioFile, DiarizedTranscript, RunResult


STAGES = {
    "server": "Comprobar servidor y modelo de notas",
    "models": "Cargar modelos de voz",
    "audio": "Validar audio",
    "recognition": "Transcribir y separar hablantes",
    "diarization": "Separar hablantes",
    "transcription": "Transcribir",
    "transcript": "Guardar transcripción",
    "generation": "Generar nota",
}


def _report_cards(result: RunResult) -> str:
    transcript = result.transcript
    segments = len(transcript.segments) if transcript else 0
    speakers = set()
    if isinstance(transcript, DiarizedTranscript):
        speakers = {segment.speaker for segment in transcript.segments if segment.speaker}
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
    for index, segment in enumerate(result.transcript.segments):
        speaker = getattr(segment, "speaker", None) or "Sin etiqueta"
        bounds = ""
        if segment.start is not None:
            bounds = f"{timestamp(segment.start)} — {timestamp(segment.end)}"
        rows.append(
            f'<article class="turn speaker-{index % 4}"><div class="turn-meta">'
            f'<strong>{html.escape(speaker)}</strong><time>{bounds}</time></div>'
            f'<p>{html.escape(segment.text)}</p></article>'
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
        gib = process["peak_sampled_rss_bytes"] / 1024 ** 3
        role = html.escape(process["role"])
        cores = process["average_cpu_cores"]
        rows.append(
            f'<tr><td>{role} · PID {process["pid"]}</td>'
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
        paths.extend(path.relative_to(directory) for path in sorted(telemetry.glob("*.jsonl")))
    for step in result.steps:
        profile = step.metadata.get("profile", {})
        if profile.get("trace_path"):
            paths.append(Path(_relative_url(Path(profile["trace_path"]), directory)))
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
        workflow=html.escape(result.workflow), status=html.escape(status),
        started_at=html.escape(result.started_at), run_id=result.run_id,
        trace_id=result.trace_id or "Sin exportador configurado", cards=_report_cards(result),
        error=error, note=html.escape(result.note.text) if result.note else "Nota pendiente.",
        audio=audio, transcript=_report_turns(result), steps=_report_steps(result),
        resources=_report_resources(result), links=_report_links(result, directory),
        configuration=html.escape(json.dumps(result.configuration, ensure_ascii=False, indent=2)),
    )


class Progress:
    def __init__(self, stream: TextIO = sys.stderr):
        self.stream = stream
        self.timings: dict[str, float] = {}
        self.active: str | None = None
        self.started = perf_counter()

    def __call__(self, stage: str) -> None:
        now = perf_counter()
        if self.active:
            elapsed = now - self.started
            self.timings[self.active] = round(elapsed, 3)
            print(f"  ✓ {STAGES[self.active]} · {elapsed:.1f} s", file=self.stream, flush=True)
        self.active = None if stage == "done" else stage
        self.started = now
        if self.active:
            print(f"  … {STAGES[stage]}", file=self.stream, flush=True)

    def fail(self) -> None:
        if self.active:
            self.timings[self.active] = round(perf_counter() - self.started, 3)
            print(f"  ✗ {STAGES[self.active]}", file=self.stream, flush=True)
            self.active = None


def timestamp(seconds: float) -> str:
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes):02}:{seconds:04.1f}"


def print_transcript(result: PipelineResult, stream: TextIO) -> None:
    print("\nTRANSCRIPCIÓN\n" + "─" * 64, file=stream)
    for turn in result.utterances:
        print(f"[{timestamp(turn.start)}–{timestamp(turn.end)}] "
              f"{turn.speaker or 'sin atribuir'}\n  {turn.text}\n", file=stream)
    if not result.utterances:
        print("Sin segmentos de voz.", file=stream)
    stream.flush()


def print_note(result: PipelineResult, stream: TextIO) -> None:
    if result.generated_text:
        print("\nNOTA GENERADA\n" + "─" * 64, file=stream)
        print(result.generated_text, file=stream, flush=True)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_run(directory: Path, result: PipelineResult, metadata: dict, *, complete: bool) -> None:
    """A speech checkpoint survives a later LLM error."""
    if complete:
        write_json(directory / "result.json", asdict(result))
        if result.generated_text is not None:
            (directory / "note.txt").write_text(result.generated_text + "\n", encoding="utf-8")
    else:
        write_json(directory / "transcript.json", asdict(result))
    write_json(directory / "run.json", metadata)
    (directory / "report.html").write_text(render_report(result, metadata, directory), encoding="utf-8")


def render_report(result: PipelineResult, metadata: dict, directory: Path) -> str:
    esc = html.escape
    audio = Path(metadata["audio"])
    audio_url = quote(Path(os.path.relpath(audio, directory)).as_posix())
    speakers = sorted({turn.speaker for turn in result.utterances if turn.speaker is not None})
    rows = []
    for turn in result.utterances:
        color = speakers.index(turn.speaker) % 4 if turn.speaker in speakers else 4
        rows.append(
            f'<article class="turn speaker-{color}"><div class="turn-meta">'
            f'<strong>{esc(turn.speaker or "sin atribuir")}</strong>'
            f'<time>{timestamp(turn.start)} — {timestamp(turn.end)}</time></div>'
            f'<p>{esc(turn.text)}</p></article>'
        )
    timings = "".join(
        f'<div class="timing"><span>{esc(STAGES.get(stage, stage))}</span><b>{seconds:.1f} s</b></div>'
        for stage, seconds in metadata["timings"].items()
    )
    note = esc(result.generated_text) if result.generated_text else "No se ha generado una nota."
    error = f'<p class="error">{esc(metadata["error"])}</p>' if metadata.get("error") else ""
    status = "DEMO · salidas inventadas" if metadata["demo"] else metadata["status"]
    note_model = metadata.get("llm_model") or "Sin generador"
    return f'''<!doctype html>
<html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scribnic · {esc(audio.stem)}</title>
<style>
:root{{color-scheme:light;--ink:#193b36;--muted:#64756d;--paper:#f3f4ed;--line:#dce4d9}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 system-ui,sans-serif}}
main{{max-width:1240px;margin:auto;padding:48px 28px}}header{{border-bottom:1px solid var(--line);padding-bottom:28px}}
.eyebrow{{text-transform:uppercase;letter-spacing:.18em;font-size:12px;font-weight:700;color:var(--muted)}}
h1{{font-size:clamp(30px,5vw,54px);line-height:1.15;letter-spacing:-.04em;margin:16px 0}}
h2{{font-size:23px;margin:0 0 22px;letter-spacing:-.02em}}p{{margin:10px 0}}.muted,time{{color:var(--muted)}}
.pill{{display:inline-block;border:1px solid #b6cfc0;border-radius:30px;padding:4px 12px;font-size:12px;background:#e7f1e7}}
.flow{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:25px 0}}.flow span{{background:white;border:1px solid var(--line);border-radius:8px;padding:8px 12px;font-size:13px;overflow-wrap:anywhere}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0}}.metric{{border-top:2px solid #80a38d;padding:14px 0}}.metric b{{display:block;font-size:27px;font-weight:550}}.metric small{{color:var(--muted)}}
.columns{{display:grid;grid-template-columns:1fr 1.12fr;gap:24px;align-items:start}}section{{padding:28px;background:#fff;border:1px solid var(--line);border-radius:16px;min-width:0}}
.note{{white-space:pre-wrap;overflow-wrap:anywhere;font:17px/1.85 Georgia,serif}}audio{{width:100%;margin-bottom:20px}}
.turn{{padding:8px 0 8px 16px;border-left:3px solid #71a68b;margin:0 0 22px}}.turn p{{margin-bottom:0;white-space:pre-wrap;overflow-wrap:anywhere}}
.speaker-1{{border-color:#8a98c7}}.speaker-2{{border-color:#c9a468}}.speaker-3{{border-color:#bd8d9f}}.speaker-4{{border-color:#aaa}}
.turn-meta{{display:flex;gap:12px;justify-content:space-between;font-size:12px;flex-wrap:wrap}}.timing{{display:flex;justify-content:space-between;gap:20px;padding:8px 0;border-bottom:1px solid var(--line);font-size:13px}}
details{{margin:28px 0}}summary{{cursor:pointer;font-weight:600}}.error{{background:#fff0eb;border-radius:8px;padding:16px;overflow-wrap:anywhere}}
footer{{font-size:12px;color:var(--muted);margin-top:28px}}a{{color:inherit}}@media(max-width:760px){{main{{padding:28px 16px}}.columns{{grid-template-columns:1fr}}.metrics{{grid-template-columns:repeat(2,1fr)}}section{{padding:20px}}}}
</style>
<main><header><div class="eyebrow">Scribnic / Registro de ejecución</div>
<h1>De la conversación<br>a la nota clínica.</h1><span class="pill">{esc(status)}</span>
<p class="muted">{esc(audio.name)} · {esc(metadata["started_at"])} · {esc(metadata["device"])}</p>
<div class="flow"><span>Audio</span> → <span>{esc(metadata["speech_model"])}</span> → <span>{esc(note_model)}</span> → <span>Nota</span></div></header>
<div class="metrics"><div class="metric"><b>{metadata["audio_seconds"]:.1f} s</b><small>Audio</small></div>
<div class="metric"><b>{len(speakers)}</b><small>Hablantes con etiqueta</small></div>
<div class="metric"><b>{len(result.utterances)}</b><small>Segmentos</small></div>
<div class="metric"><b>{sum(metadata["timings"].values()):.1f} s</b><small>Tiempo de etapas</small></div></div>
{error}<div class="columns"><section><div class="eyebrow">Generación / Borrador</div><h2>Nota clínica</h2><div class="note">{note}</div></section>
<section><div class="eyebrow">Voz / Evidencia</div><h2>Transcripción</h2><audio controls preload="none" src="{esc(audio_url, quote=True)}"></audio>
{"".join(rows) or "<p>Sin segmentos de voz.</p>"}</section></div>
<details><summary>Etapas y tiempos</summary>{timings}</details>
<details><summary>Instrucción del generador</summary><div class="note">{esc(metadata["instruction"])}</div></details>
<footer>Resultado local · <a href="transcript.json">Transcripción JSON</a> · <a href="run.json">Datos de ejecución</a></footer></main></html>'''
