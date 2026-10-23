"""Rich terminal presentation; importing the inference API never imports this module."""

import logging
from contextlib import ExitStack
from time import monotonic
from typing import TextIO

from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TextColumn
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from .artifacts import RunResult
from .progress import ProgressEvent, observe_progress
from .reporting import print_result, timestamp


class _StatusColumn(ProgressColumn):
    """Render phase state without deriving completion from a token budget."""

    def __init__(self) -> None:
        """Retain animation timing across frames for the life of the column."""
        super().__init__()
        self.spinner = Spinner("dots", style="cyan")

    def render(self, task: Task) -> RenderableType:
        """Return a spinner while active, or a colored final state marker."""
        status = task.fields.get("status", "start")
        if status == "done":
            return Text("✓", style="green")
        if status == "error":
            return Text("✗", style="red")
        if status == "interrupted":
            return Text("■", style="yellow")
        return self.spinner.render(task.get_time())


class _BytesBar(BarColumn):
    """Show a determinate bar only for downloads with a known byte count."""

    def render(self, task: Task) -> RenderableType:
        """Keep inference and unknown-length transfers free of fake percentages."""
        if task.fields.get("unit") != "bytes" or task.total is None:
            return Text("")
        return super().render(task)


class _DetailsColumn(ProgressColumn):
    """Display elapsed time and actual transfer counters for the current task."""

    def render(self, task: Task) -> Text:
        """Describe bytes and sampled transfer speed, or an unbounded token count."""
        seconds = task.finished_time if task.finished else task.elapsed
        detail = f"{seconds or 0:,.1f} s"
        unit = task.fields.get("unit")
        if unit == "bytes":
            detail += f" · {task.completed / 1024**2:.1f} MiB"
            if task.total is not None:
                detail += f" / {task.total / 1024**2:.1f} MiB"
            if task.speed:
                detail += f" · {task.speed / 1024**2:.1f} MiB/s"
        elif unit == "tokens":
            detail += f" · {int(task.completed)} tokens"
        return Text(detail, style="dim")


class _ConsoleLogs(logging.Handler):
    """Present actionable Scribnic logs through the active Rich console."""

    def __init__(self, console: Console, *, verbose: bool) -> None:
        """Own a presentation handler for the duration of one CLI invocation."""
        super().__init__(logging.INFO if verbose else logging.WARNING)
        self.console = console

    def emit(self, record: logging.LogRecord) -> None:
        """Print literal messages safely, including paths or bracketed identifiers."""
        style = "yellow" if record.levelno == logging.WARNING else "dim"
        if record.levelno >= logging.ERROR:
            style = "red"
        self.console.print(Text(record.getMessage(), style=style))


class TerminalView:
    """Own one CLI's live progress and log handler; never own inference resources."""

    def __init__(self, stream: TextIO, *, verbose: bool = False) -> None:
        """Use a live display only for a TTY; otherwise emit stable phase lines."""
        self.console = Console(file=stream, highlight=False)
        self.verbose = verbose
        self._cleanup = ExitStack()
        self._tasks: dict[str, int] = {}
        self._started: dict[str, float] = {}
        self._progress = Progress(
            _StatusColumn(),
            TextColumn("{task.description}", markup=False),
            _BytesBar(bar_width=16),
            _DetailsColumn(),
            console=self.console,
            refresh_per_second=8,
            disable=not self.console.is_terminal,
            redirect_stdout=False,
            redirect_stderr=False,
        )

    def __enter__(self) -> "TerminalView":
        """Start presentation and attach the observer only to this calling context."""
        self._cleanup.enter_context(self._progress)
        self._cleanup.enter_context(observe_progress(self.update))
        handler = _ConsoleLogs(self.console, verbose=self.verbose)
        logger = logging.getLogger("scribnic")
        logger.addHandler(handler)
        self._cleanup.callback(handler.close)
        self._cleanup.callback(logger.removeHandler, handler)
        return self

    def __exit__(self, *exc: object) -> None:
        """Restore observer, terminal cursor and logging on all exit paths."""
        self._cleanup.close()

    def header(self, workflow: str, *, destination: str | None = None) -> None:
        """Print the selected workflow and destination without interpreting markup."""
        self.console.print(Text(f"SCRIBNIC · {workflow}", style="bold cyan"))
        if destination is not None:
            self.console.print(Text(f"Resultados: {destination}", style="dim"))

    def update(self, event: ProgressEvent) -> None:
        """Present a real execution notification, with no dependency on its producer."""
        description = event.label
        if event.detail:
            description += " · " + event.detail
        if event.state == "start":
            self._started[event.task_id] = monotonic()
            self._tasks[event.task_id] = self._progress.add_task(
                description,
                total=event.total,
                unit=event.unit,
                status=event.state,
            )
        identifier = self._tasks[event.task_id]
        self._progress.update(identifier, completed=event.completed, status=event.state)
        if event.state in {"done", "error", "interrupted"}:
            self._progress.stop_task(identifier)
        if not self.console.is_terminal:
            self._plain_event(event, description)

    def _plain_event(self, event: ProgressEvent, description: str) -> None:
        """Write only phase boundaries to redirected output, never animation frames."""
        if event.state == "update":
            return
        if event.state == "start":
            self.console.print(Text(f"  … {description}"))
            return
        mark = {"done": "✓", "error": "✗", "interrupted": "■"}[event.state]
        seconds = monotonic() - self._started[event.task_id]
        self.console.print(Text(f"  {mark} {description} · {seconds:.2f} s"))

    def error(self, message: str) -> None:
        """Render an actionable error without printing a local Python traceback."""
        self.console.print(Text(message, style="bold red"))

    def summary(self, result: RunResult) -> None:
        """Show measured per-step times after progress has completed."""
        table = Table(box=None, show_header=False, padding=(0, 2))
        for step in result.steps:
            table.add_row(Text(step.name), f"{step.elapsed_seconds:.2f} s")
        table.add_row(Text("Total", style="bold"), f"{result.elapsed_seconds:.2f} s")
        self.console.print(table)


def present_result(result: RunResult, stream: TextIO) -> None:
    """Render literal note text in a panel on TTYs, retaining plain redirected output."""
    console = Console(file=stream, highlight=False)
    if not console.is_terminal:
        print_result(result, stream)
        return
    transcript = result.transcript
    if transcript is not None:
        lines = []
        for segment in transcript.segments:
            speaker = getattr(segment, "speaker", None)
            prefix = f"[{speaker}] " if speaker else ""
            bounds = f"{timestamp(segment.start)} " if segment.start is not None else ""
            lines.append(bounds + prefix + segment.text)
        console.print(
            Panel(Text("\n".join(lines)), title="Transcripción", border_style="dim")
        )
    if result.note is not None:
        console.print(Panel(Text(result.note.text), title="Nota", border_style="cyan"))
