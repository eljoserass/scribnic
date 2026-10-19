# Speech pipeline, evaluation, and inference options

Decision note, not an implementation plan. The immediate goal is to compare local multi-speaker models; the longer-term goal is an application that downloads suitable models, accepts audio through a UI, and produces a transcript and, optionally, a medical note.

## Where we are

`Pipeline` currently accepts four optional components and selects either `recognizer` or `diarizer + transcriber` at runtime (`src/scribnic/core.py`). `PipelineResult` exposes parallel `transcription`, `diarization`, and `utterances` fields (`src/scribnic/models.py`). For MOSS/VibeVoice, the first two are reconstructed from the attributed turns; for separate models, they are actual stage outputs. Those are not equivalent provenance. The CLI also exposes separate `--model` versus `--asr`/`--diarizer` choices (`src/scribnic/cli.py`).

The stable *product* flow should be:

```text
audio -> speaker-attributed transcript -> optional medical note
           |                         |
           +-- stage evidence        +-- separate note evaluation
```

An utterance may have an unknown speaker; that is preferable to inventing a label. Preserve overlap and distinguish native, aligned, and approximate timestamps. [NeMo describes both cascaded and end-to-end diarization](https://github.com/NVIDIA-NeMo/Speech/blob/main/docs/source/asr/speaker_diarization/models.rst); [WhisperX](https://github.com/m-bain/whisperX) is an example of ASR, alignment, diarization, then speaker assignment. Neither requires our application to expose those stages as mutually exclusive top-level pipelines.

## Three architecture directions

### A. Transcript-first strategy (recommended)

Define one required speech interface, for example `ConversationTranscriber.transcribe(audio) -> TranscriptResult`. `TranscriptResult` owns ordered speaker-attributed utterances plus explicit evidence/provenance. MOSS and VibeVoice implement it directly. A `DiarizeThenTranscribe(diarizer, asr, attributor)` adapter implements the same interface for Qwen/Nemotron/Canary combinations. `Pipeline` then has a required speech component and an optional note generator; it does not know whether inference was joint or cascaded.

Keep `TextSpan` and `SpeakerSpan` as useful *internal stage outputs*. Record which were actually produced, model ID/revision, duration, and timestamp origin in a small typed trace; do not synthesize a separate diarizer result for a joint model. A display-only diarization view can be derived and labelled as such. The speech result should be useful without a note, and note generation should consume the same canonical transcript for every backend.

- **Good:** clear output contract, simple UI/API integration, testable composition, honest provenance.
- **Cost:** one migration of result schema, adapters, CLI selection, and tests. Diarizer-first is one composition strategy, not a rule for every future ASR.
- **Selection:** user-facing speech presets such as `moss`, `vibevoice`, `qwen+nemotron`; advanced component overrides may remain, but invalid combinations should fail during construction.

### B. Typed artifact graph

Model processing as nodes such as `Audio -> Diarization`, `Audio -> Text`, `(Text, Diarization) -> AttributedTranscript`, and `AttributedTranscript -> Note`. Joint models emit the attributed transcript directly. A graph runner resolves dependencies and may cache or expose every stage.

- **Good:** naturally handles ASR-first versus diarizer-first, forced alignment, alternative attribution, multiple outputs, and reusable intermediate artifacts.
- **Cost:** scheduler, artifact typing, invalidation, and cache semantics become application concerns. A general plugin/DAG framework would be premature for four speech backends.
- **Choose when:** real workflows need branching, resumable long-running jobs, or re-evaluating stages without rerunning models. Until then, A can expose enough stage evidence without a graph engine.

### C. Minimal strategy wrapper

Introduce one `TranscriptProducer` around the existing joint and separate paths, but keep `PipelineResult` and most internals unchanged. This removes the four-optionals problem with the smallest diff.

- **Good:** low migration risk and fastest route to a consistent call site.
- **Cost:** the parallel result fields still blur native and derived outputs; stage-order assumptions and misleading names can survive behind the wrapper.
- **Choose when:** we want an intermediate release before deciding the result format. It should not become the long-term schema by accident.

**Choice:** A now, with typed provenance rather than a generic bag of arbitrary objects. B remains a possible evolution; C is only a transitional step. Keep the speech contract independent of the runtime engine: a Transformers adapter and a future vLLM service can produce the same `TranscriptResult`.

## Evaluation tools: different jobs

| Tool | What it is good for | Fit here |
| --- | --- | --- |
| [OpenBench](https://github.com/argmaxinc/OpenBench) | Full benchmark runner: datasets, pipeline adapters, aggregate metrics, reproducible reports; supports diarization, ASR, and combined systems. | Useful once we have multiple annotated recordings and want standardized comparisons. Its combined-task schema expects word-level speaker labels; our current script JSON is not directly that schema. Keep it as an optional adapter, not the core application dependency. |
| [MeetEval](https://github.com/fgnt/meeteval) | Scoring library/CLI for meeting transcripts: WER, speaker-permutation-aware **cpWER**, time-constrained variants, format conversion, and DER via its mdeval wrapper. | Best first lightweight scorer for our segment-level speaker/text references. cpWER does not require exact timestamps; time-constrained scores and DER do. It scores saved outputs without rerunning a model. |
| [pyannote.metrics](https://pyannote.github.io/pyannote-metrics/) | Detailed diarization metrics and error analysis, including DER components. | Good once reference speaker intervals are manually aligned. It is a metric library, not a model runner or UI harness. |

WER measures words without speaker identity; cpWER tests speaker-attributed words while allowing arbitrary speaker-ID permutations; DER measures *who spoke when*. Track speaker count, missing/unknown attribution, inference wall time, and real-time factor (wall time / audio duration) alongside quality. Fix text normalization and DER collar/overlap policy before comparing runs. A single score on one synthetic consultation is a regression signal, **not** a general model ranking. [MeetEval's formats and metrics](https://github.com/fgnt/meeteval) and [OpenBench's local dataset and orchestration schemas](https://github.com/argmaxinc/OpenBench) document the different annotation requirements.

`data/examples/original.json` contains speaker turns and *estimated* durations, not verified audio boundaries. First confirm that its words match the recording, and score the full recording or create an exact reference for an excerpt. WER/cpWER can then be tried without times. Do not report DER until a human-checked RTTM/interval reference exists. Do not score the medical note with WER: later, evaluate factual coverage and unsupported claims separately, ideally with human-reviewed cases.

## How evaluation fits the application

Keep inference and scoring connected by **saved artifacts**, not by coupling the application to OpenBench:

```text
CLI or UI -> application service -> model manager -> speech backend -> transcript -> note
                                              |                |
                                              +---- run record -+
                                                       |
                                    optional evaluator + reference -> report
```

1. A shared application service runs the same backend for CLI and future UI. It records input ID, model/checkpoint revision, runtime, options, environment, stage timing, result, and errors. Evaluation reads that record; it must not secretly download or repeat inference.
2. Model management is separate: catalog supported backends and hardware, explicit download/verify/cache states, progress, cancellation/resume, and offline mode. Default to conservative download concurrency, with weights outside Git. Hugging Face's [`snapshot_download`](https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download) supports `max_workers=1` for one file at a time and `local_files_only`; [Xet separately controls ranges per file](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/environment_variables) with `HF_XET_NUM_CONCURRENT_RANGE_GETS=1`. Neither is an exact bandwidth cap; that needs network-level rate limiting. Pin model revisions for repeatable comparisons and surface remote-code trust requirements.
3. Test in layers: fast synthetic unit/contract tests; offline scoring of saved predictions; opt-in live model tests with locally cached weights; model-download tests with a fake/local repository; and later UI/API end-to-end tests using a fake backend. Real network and heavy models should not run in ordinary CI. A small, versioned reference set and a report per model/configuration are more valuable than fragile pass/fail accuracy thresholds on one sample.
4. For a client application, show **available / download required / unsupported hardware / failed** before Run. Inference must use the selected installed checkpoint and return the same transcript contract whether started from CLI or UI. A note is optional, and sensitive recordings/results should remain local unless the user explicitly configures a remote provider.

## Are we missing an inference engine?

An *inference engine* executes a model; an *evaluation harness* scores its output. Changing engines does not automatically improve model accuracy, and serving APIs may expose less structured output than the native model. Compare engines with the same checkpoint, audio, decoding parameters, output parser, and hardware.

| Candidate | What the model authors document | Decision for this repo |
| --- | --- | --- |
| Transformers / NeMo Python | Our current direct adapters; [MOSS](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize), [Qwen](https://github.com/QwenLM/Qwen3-ASR), and [VibeVoice-HF](https://huggingface.co/docs/transformers/model_doc/vibevoice_asr) document native inference paths. | Keep as correctness baseline; it gives access to native structured outputs and runs in our existing ROCm environment. |
| Model-specific packages | [Qwen's `qwen-asr`](https://github.com/QwenLM/Qwen3-ASR) wraps Transformers/vLLM and offers an optional forced aligner; [MOSS's package](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) includes inference helpers and a subtitle UI. | Try wrappers where they replace fragile custom glue, but compare output and dependency compatibility first. Qwen's aligner adds timestamps, **not** speaker labels. |
| vLLM | Official serving paths exist for [MOSS](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize), [Qwen](https://github.com/QwenLM/Qwen3-ASR), and [VibeVoice](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-vllm-asr.md). [vLLM lists gfx1151 ROCm support](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/). | Worth a separate benchmark for throughput/latency, not an immediate replacement. Its ROCm wheels bundle a specific PyTorch/ROCm stack that may conflict with our pinned environment; use an isolated runtime/process and verify each model/plugin/checkpoint on this GPU. |
| SGLang Omni | [MOSS recommends it for serving](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize), currently documenting a CUDA 13 path. | Consider for clients with matching NVIDIA hardware; not the first experiment on our AMD setup. |
| NeMo-Speech.cpp | [Native GGUF runtime](https://github.com/NVIDIA/NeMo-Speech.cpp) supports several Nemotron ASR and diarization models, standalone or combined. | Interesting small-footprint alternative for those models; validate checkpoint conversion, backend support, and output parity before adopting. It does not replace MOSS/Qwen/VibeVoice support. |

The immediate experiment should therefore be **better reference data and reproducible measurements**, not another runtime installation. VibeVoice's current CPU offload did not complete our live sample on this machine; a different engine may help, but that is a hypothesis to test, not a claimed speedup.

## Suggested increments, if we proceed

1. Agree on A/B/C and the transcript/provenance schema; migrate one existing backend and its contract tests.
2. Migrate the remaining backends and CLI presets without changing model behavior; test native versus derived traces.
3. Add saved-run format and an optional MeetEval scorer with a verified full-audio reference; report WER/cpWER and timing separately.
4. Add aligned diarization references/DER and more recordings; only then assess OpenBench integration or alternative runtimes.
5. Add explicit model lifecycle and UI on the same application service, with fake-backend end-to-end tests before heavyweight client tests.

No architecture or evaluation code is changed by this document.
