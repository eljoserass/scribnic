"""Transcribe a WAV file, optionally generate a note, and inspect the run."""

import argparse
import json
import os
import sys
import wave
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .audio import FileSource
from .factory import ASR_MODELS, DIARIZERS, JOINT_MODELS, build_pipeline
from .prompts import SOAP_INSTRUCTION
from .reporting import Progress, print_note, print_transcript, save_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Mono PCM16 WAV at 16 kHz")
    parser.add_argument("--demo", action="store_true", help="Use deterministic example outputs")
    parser.add_argument("--device", default="auto", help="Both models: auto, cpu, or cuda:0 (ROCm also uses cuda:0)")
    parser.add_argument("--language", default="es-ES", help="ASR locale, e.g. es-ES or en-US")
    ### TODO check for later a better way to indicate the models, joint model v diarizer v asr dont like it much
    parser.add_argument("--asr", choices=ASR_MODELS, default="nemotron")
    parser.add_argument("--diarizer", choices=DIARIZERS, default="nemotron")
    parser.add_argument("--model", choices=JOINT_MODELS, help="Joint transcription and diarization model")
    parser.add_argument("--llm-model", help="Model identifier served by LM Studio")
    #####
    parser.add_argument("--base-url", default=os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"))
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--instruction", help="Override the generation instruction")
    parser.add_argument("--soap", action="store_true",
                        help="MOSS + auto-detected Ternary Bonsai 2 2B, Spanish SOAP, pretty output and saved report")
    parser.add_argument("--format", choices=("json", "pretty"), help="Default: json; pretty with --soap")
    parser.add_argument("--output-dir", type=Path, help="Empty directory for transcript, note, timings and HTML report")
    parser.add_argument("--llm-max-tokens", type=int, default=768)
    parser.add_argument("--llm-max-input-chars", type=int, default=12000,
                        help="Reject longer transcripts without truncation (default: 12000 characters)")
    args = parser.parse_args()
    if args.llm_max_tokens < 1 or args.llm_max_input_chars < 1:
        parser.error("LLM token and input limits must be positive.")
    if args.soap and (args.asr != "nemotron" or args.diarizer != "nemotron"):
        parser.error("--soap uses a joint speech model; omit --asr and --diarizer.")
    model = args.model or ("moss" if args.soap else None)
    llm_model = args.llm_model or ("auto" if args.soap else None)
    instruction = args.instruction or (SOAP_INSTRUCTION if args.soap else
                                      "Resume la conversación sin inventar información.")
    output_format = args.format or ("pretty" if args.soap else "json")
    started = datetime.now().astimezone()
    output_dir = args.output_dir
    if args.soap and output_dir is None:
        output_dir = Path("runs") / started.strftime("%Y%m%d-%H%M%S-%f")
    if output_dir:
        output_dir = output_dir.expanduser().resolve()
    stream = sys.stdout
    progress = Progress(sys.stderr)
    metadata = None
    transcript = None

    def checkpoint(result):
        nonlocal transcript
        transcript = result
        metadata["status"] = "Transcripción lista · nota pendiente" if pipeline.generator else "Transcripción lista"
        if output_dir:
            save_run(output_dir, result, metadata, complete=False)
        if output_format == "pretty":
            print_transcript(result, stream)

    try:
        source = FileSource(args.audio)
        audio = source.load()  # Check the input before loading any large model.
        if output_dir:
            if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
                raise ValueError(f"Output directory must be empty: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
        if output_format == "pretty":
            print(f"\nSCRIBNIC{' · DEMO (salidas inventadas)' if args.demo else ''}\n"
                  f"{audio.path.name} · {audio.duration:.1f} s\n"
                  f"{model or args.diarizer + ' + ' + args.asr} → "
                  f"{llm_model if not args.skip_generation else 'sin generación'}\n",
                  file=sys.stderr, flush=True)
        with redirect_stdout(sys.stderr):
            if llm_model and not args.skip_generation and not args.demo:
                from .backends.generation import LMStudioTextGenerator

                progress("server")
                llm_model = LMStudioTextGenerator(llm_model, args.base_url).check_available()
                print(f"  Modelo de notas: {llm_model}", file=sys.stderr, flush=True)
            progress("models")
            pipeline = build_pipeline(
                demo=args.demo, device=args.device, language=args.language,
                asr=args.asr, diarizer=args.diarizer, model=model, llm_model=llm_model,
                base_url=args.base_url, skip_generation=args.skip_generation,
                llm_max_tokens=args.llm_max_tokens, llm_max_input_chars=args.llm_max_input_chars,
            )
            recognizer = pipeline.recognizer
            metadata = {
                "started_at": started.isoformat(timespec="seconds"),
                "audio": str(audio.path), "audio_seconds": audio.duration,
                "speech_model": "demo" if args.demo else model or f"{args.diarizer} + {args.asr}",
                "speech_checkpoint": getattr(recognizer, "MODEL", None),
                "device": str(getattr(recognizer, "device", args.device)),
                "llm_model": "demo" if args.demo and pipeline.generator else
                             llm_model if pipeline.generator else None,
                "base_url": args.base_url if not args.demo and pipeline.generator else None,
                "instruction": instruction, "llm_max_tokens": args.llm_max_tokens,
                "llm_max_input_chars": args.llm_max_input_chars, "demo": args.demo,
                "timings": progress.timings, "status": "En ejecución", "error": None,
            }
            result = pipeline.run(source, instruction, on_progress=progress, on_transcript=checkpoint)
        metadata["status"] = "Completado"
        if output_dir:
            save_run(output_dir, result, metadata, complete=True)
    except (OSError, ValueError, wave.Error, EOFError, ImportError) as exc:
        progress.fail()
        if output_dir and transcript is not None and metadata is not None:
            metadata["status"] = "Error · transcripción conservada"
            metadata["error"] = str(exc)
            try:
                save_run(output_dir, transcript, metadata, complete=False)
            except OSError:
                pass  # Preserve the original failure if the output disk also fails.
            print(f"Transcripción guardada en {output_dir / 'transcript.json'}", file=sys.stderr)
        parser.exit(1, f"Error: {exc}\n")
    except KeyboardInterrupt:
        progress.fail()
        parser.exit(130, "\nEjecución interrumpida.\n")
    if output_format == "json":
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print_note(result, stream)
    if output_dir:
        print(f"\nInforme: {output_dir / 'report.html'}", file=sys.stderr)


if __name__ == "__main__":
    main()
