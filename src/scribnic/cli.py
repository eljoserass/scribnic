"""Typer commands delegate execution to the public API and presentation to Rich."""

import json
import sys
from contextlib import redirect_stdout
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from .application import execute, new_run_directory, setup_workflow
from .config import artifact_kind
from .storage import result_to_dict
from .terminal import TerminalView, present_result
from .workflow import WorkflowRunError

app = typer.Typer(
    help="Run a local clinical note workflow or prepare its cached resources.",
    add_completion=True,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


class OutputKind(str, Enum):
    """Expose supported artifact choices without implementing workflow routing."""

    audio = "audio"
    transcript = "transcript"
    diarized_transcript = "diarized_transcript"
    note = "note"


class OutputFormat(str, Enum):
    """Choose human-readable presentation or a single machine-readable document."""

    pretty = "pretty"
    json = "json"


@app.command("run")
def run_command(
    input_path: Annotated[
        Path, typer.Argument(help="WAV, transcript JSON, or plain text")
    ],
    *,
    workflow: Annotated[str, typer.Option(help="Preset name, TOML file, or directory")],
    output: Annotated[
        OutputKind | None, typer.Option(help="Requested artifact")
    ] = None,
    output_dir: Annotated[
        Path | None, typer.Option(help="Empty results directory")
    ] = None,
    output_format: Annotated[
        OutputFormat, typer.Option("--format")
    ] = OutputFormat.pretty,
    offline: Annotated[
        bool, typer.Option("--offline", help="Require cached resources")
    ] = False,
    profile: Annotated[
        bool, typer.Option("--profile", help="Save optional runtime profiles")
    ] = False,
    realtime: Annotated[
        bool, typer.Option("--realtime", help="Pace WAV input in a streaming workflow")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show diagnostics")
    ] = False,
) -> None:
    """Generate a note or stop at a configured intermediate artifact."""
    directory = (output_dir or new_run_directory()).expanduser().resolve()
    view = TerminalView(sys.stderr, verbose=verbose)
    try:
        with view, redirect_stdout(sys.stderr):
            view.header(workflow, destination=str(directory))
            result = execute(
                workflow,
                input_path,
                directory,
                output=artifact_kind(output.value) if output is not None else None,
                offline=offline,
                profile=profile,
                realtime=realtime,
            )
        if output_format == OutputFormat.json:
            print(json.dumps(result_to_dict(result), ensure_ascii=False))
        else:
            present_result(result, sys.stdout)
        view.summary(result)
        view.console.print(f"Informe: {directory / 'report.html'}", markup=False)
    except WorkflowRunError as error:
        view.error(f"Error: {error}")
        view.console.print(
            "Los artefactos completados se conservan en el directorio de resultados."
        )
        raise typer.Exit(1) from error
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        view.error(f"Error: {error}")
        raise typer.Exit(1) from error
    except KeyboardInterrupt:
        view.error("Ejecución interrumpida.")
        raise typer.Exit(130) from None


@app.command("prepare", hidden=True)
@app.command("setup")
def setup_command(
    *,
    workflow: Annotated[str, typer.Option(help="Preset name, TOML file, or directory")],
    offline: Annotated[
        bool, typer.Option("--offline", help="Require cached resources")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show diagnostics")
    ] = False,
) -> None:
    """Download and verify workflow assets without loading models or starting servers."""
    view = TerminalView(sys.stderr, verbose=verbose)
    try:
        with view, redirect_stdout(sys.stderr):
            view.header(workflow)
            setup_workflow(workflow, offline=offline)
        print("Recursos preparados. Los modelos se cargarán al ejecutar el workflow.")
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        view.error(f"Error: {error}")
        raise typer.Exit(1) from error
    except KeyboardInterrupt:
        view.error("Preparación interrumpida.")
        raise typer.Exit(130) from None


def main(argv: list[str] | None = None) -> None:
    """Invoke Typer while preserving normal return on success for Python callers."""
    try:
        app(args=argv, prog_name="scribnic")
    except SystemExit as error:
        if error.code not in (None, 0):
            raise


if __name__ == "__main__":
    main()
