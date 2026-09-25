"""Run the pipeline on one WAV file and emit its JSON result."""

import argparse
import json
import sys
import wave
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path

from .audio import FileSource
from .factory import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="Mono PCM16 WAV at 16 kHz")
    parser.add_argument("--demo", action="store_true", help="Use deterministic example outputs")
    parser.add_argument("--device", default="auto", help="Both models: auto, cpu, or cuda:0 (ROCm also uses cuda:0)")
    parser.add_argument("--language", default="es-ES", help="ASR locale, e.g. es-ES or en-US")
    parser.add_argument("--asr", choices=("nemotron", "canary"), default="nemotron")
    parser.add_argument("--diarizer", choices=("nemotron", "sortformer"), default="nemotron")
    parser.add_argument("--llm-model", help="Model identifier served by LM Studio")
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--instruction", default="Resume la conversación sin inventar información.")
    args = parser.parse_args()
    try:
        source = FileSource(args.audio)
        source.load()  # Check the input before loading any large model.
        with redirect_stdout(sys.stderr):
            pipeline = build_pipeline(
                demo=args.demo, device=args.device, language=args.language,
                asr=args.asr, diarizer=args.diarizer, llm_model=args.llm_model,
                base_url=args.base_url, skip_generation=args.skip_generation,
            )
            result = pipeline.run(source, args.instruction)
    except (OSError, ValueError, wave.Error, EOFError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
