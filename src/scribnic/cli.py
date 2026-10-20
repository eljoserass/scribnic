"""Run a configured local scribe workflow or prepare its cached resources."""

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from .application import execute, new_run_directory, prepare_workflow
from .artifacts import ARTIFACT_KINDS
from .reporting import print_result
from .storage import result_to_dict
from .workflow import WorkflowRunError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Generate a note or an intermediate artifact")
    run.add_argument("input", type=Path, help="WAV, transcript JSON, or plain text")
    run.add_argument(
        "--workflow", required=True, help="Preset name, TOML file, or directory"
    )
    run.add_argument(
        "--output", choices=ARTIFACT_KINDS, help="Default: workflow's note output"
    )
    run.add_argument(
        "--output-dir", type=Path, help="Empty directory; default: runs/<timestamp>"
    )
    run.add_argument("--format", choices=("pretty", "json"), default="pretty")
    run.add_argument(
        "--offline", action="store_true", help="Require already cached resources"
    )
    run.add_argument(
        "--profile", action="store_true", help="Also save a detailed Torch trace"
    )
    prepare = commands.add_parser(
        "prepare", help="Download/check assets without inference"
    )
    prepare.add_argument("--workflow", required=True)
    prepare.add_argument("--offline", action="store_true")
    return parser


def _run(args: argparse.Namespace) -> None:
    directory = (args.output_dir or new_run_directory()).expanduser().resolve()
    print(f"\nSCRIBNIC · {args.workflow}\nResultados: {directory}", file=sys.stderr)
    with redirect_stdout(sys.stderr):
        result = execute(
            args.workflow,
            args.input,
            directory,
            output=args.output,
            offline=args.offline,
            profile=args.profile,
            progress=sys.stderr,
        )
    if args.format == "json":
        print(json.dumps(result_to_dict(result), ensure_ascii=False))
    else:
        print_result(result, sys.stdout)
    print(f"Informe: {directory / 'report.html'}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            prepare_workflow(args.workflow, offline=args.offline)
            print(
                "Recursos preparados. Los modelos se cargarán al ejecutar el workflow."
            )
            return
        _run(args)
    except WorkflowRunError as error:
        print(f"Error: {error}", file=sys.stderr)
        print(
            "Los artefactos completados se conservan en el directorio de resultados.",
            file=sys.stderr,
        )
        raise SystemExit(1) from error
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    except KeyboardInterrupt:
        print("Ejecución interrumpida.", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
