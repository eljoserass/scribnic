# Speech presets

Whisper and Parakeet presets run on the existing runtimes through
`openai_transcription`: whisper.cpp v1.9.5 (`whisper_cpp`) and NeMo-Speech.cpp
v0.2.0 (`nemo_cpp`). Notes use Bonsai 2 2B PQ2_0 on the Prism llama-server. For the
Nemotron ASR and diarizer presets, see [NVIDIA NeMo](nvidia.md).

## Models and workflows

All Whisper models are multilingual (no `.en` suffix) and use Silero VAD, Spanish,
beam 5, and timestamps. Sizes are weight files in decimal units, not inference RAM.

| ASR | Parameters | Weights | Workflow |
| --- | ---: | ---: | --- |
| Whisper tiny Q8_0 | 39M | 43.5 MB | `whisper_tiny_q8_0_bonsai` |
| Whisper base Q8_0 | 74M | 81.8 MB | `whisper_base_q8_0_bonsai` |
| Whisper small Q8_0 | 244M | 264 MB | `whisper_small_q8_0_bonsai` |
| Whisper medium Q8_0 | 769M | 823 MB | `whisper_medium_q8_0_bonsai` |
| Whisper large-v3-turbo Q8_0 | 809M | 874 MB | `whisper_turbo_q8_0_bonsai` |
| Whisper large-v3 Q5_0 | 1,550M | 1.081 GB | `whisper_large_v3_q5_0_bonsai` |
| Parakeet TDT 0.6B v3 Q8_0 | 600M | 714 MB | `parakeet_v3_q8_0_bonsai` |
| Parakeet v3 + Nemotron 3 Diarization Q8_0 | ASR + diarizer | 714 MB + companion | `parakeet_v3_q8_0_diarized_bonsai` |

`whisper_bonsai`, `whisper_bonsai_vad`, `whisper_bonsai_streaming`, and
`whisper_qwen35_08b_q8_0` also use Whisper small Q8_0 and share its model preset.

Whisper weights come from the
[official whisper.cpp distribution](https://huggingface.co/ggerganov/whisper.cpp/tree/5359861c739e955e79d9a303bcbc70fb988958b1)
(MIT), an approved exception in the [source policy](model_sources.md). That
revision has no `ggml-large-v3-q8_0.bin`, so large-v3 uses Q5_0 rather than
large-v2 or a third-party conversion. Turbo Q8_0 is a different checkpoint and
keeps its full identity. Parakeet comes from
[NVIDIA's repository](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/tree/541d1f99c6b0c3cd0b11a95167540bb8edefd82b)
(CC BY 4.0). Every preset pins repository, revision, file, format, and SHA256.

Parakeet detects the language; `es-ES` is kept for compatible post-processing and
metadata, not as a prompt forcing the ASR language. NeMo-Speech.cpp serves Parakeet
v3 offline only. The diarized variant reuses the Nemotron companion and NeMo's
time-based attribution; the plain variant does not load the companion.

## Run and compare

```bash
uv run scribnic run consultation.wav --workflow whisper_turbo_q8_0_bonsai \
  --output-dir runs/whisper-turbo

uv run scribnic run consultation.wav --workflow parakeet_v3_q8_0_diarized_bonsai \
  --output-dir runs/parakeet-v3-diarized
```

Add `--output transcript` (or `--output diarized_transcript` for the diarized
workflow) to stop before the note. These workflows also accept text or saved
transcripts and then run only the note step; the session still prepares all
configured resources when it opens.

To compare note generators, recognize once and pass the same `transcript.json` to
the [text workflows](note_models.md). To change the generator of a speech pipeline,
copy the workflow and replace its `note` step; there is no need to ship every
combination.

## Validation scope

The local check used the first 15 s of the same recording on WSL/Linux x86_64 CPU,
six threads, one cold run per workflow with models already downloaded. It covered
real loading, HTTP, timings, text, the companion, and note generation. It is a short
integration check without warmup or repetitions: it picks no clinical winner and
says nothing about full consultations or quantization equivalence. Some Bonsai
notes added claims the clip did not contain.
