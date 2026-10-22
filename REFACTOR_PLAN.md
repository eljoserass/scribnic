# Workflow core refactor

## Whisper.cpp experiment — 22 October 2026

First compare plain transcription with the existing note generator:
`audio → Whisper.cpp → Transcript → Bonsai → Note`. Keep the public API, resolver,
CLI and SOAP prompt unchanged. The new workflow alone uses greedy Bonsai decoding
(temperature 0) after the recommended temperature 0.7 emitted an unmatched
`</think>` on this plain transcript. Preserve the failed run and existing presets.

- [x] Add a Whisper.cpp CLI adapter without Python bindings or new client packages.
- [x] Add pinned multilingual small Q8_0 weights and a `whisper_bonsai` workflow.
- [x] Reuse checked downloads, transcript persistence, OTel and process monitoring.
- [x] Document executable installation, explicit paths, CPU/Metal selection and
      the difference between native windows, external chunks and batching.
- [x] Validate a real full `dialogue8.wav` audio-to-note run and record resources.
- [x] Complete unit/style/package checks and commit/push the bounded integration.

The runtime executable is external in this first adapter: find `whisper-cli` on
PATH, set `SCRIBNIC_WHISPER_CPP_BINARY`, or set runtime `binary`. Record its SHA256;
do not claim an arbitrary installed binary matches the requested release. Source
used here is tag v1.9.5 / d1be6fde11ac6e0407606b4e42fe72d34add8037, source archive
SHA256 `645445da884df6dbd52b30f82d7be7c01d1767c938714e477e786b88d529e9f4`.
Automatic source builds and platform runtime bundles remain packaging work.

Later experiments remain separate: shorter external windows with explicit overlap
merging; pyannote diarization plus timestamp alignment; and bounded-context MOSS
with speaker correspondence across windows. Independent transcription/diarization
steps need an explicit multi-input join in the planner. Windowed MOSS can keep the
existing backend contract, with internal speaker reconciliation. Quantized MOSS ports
require their own backend/quality validation; Whisper GGML quantization does not
automatically apply to the joint MOSS checkpoint.

Validation: `runs/dialogue8-whisper-bonsai-20261022/report.html` contains the full
297.891-second recording, 108 timestamped plain segments, and a Spanish SOAP note.
Wall time is 48.04 s: speech 36.78 s (native load 0.24 s), note 9.11 s; speech RTF
is 0.123. No installation/downloads or detailed profiler are included. On the WSL
CPU, Whisper uses six threads (5.93 average cores) and peaks at 704,577,536 bytes
sampled RSS (671.9 MiB). Python peaks at 90,599,424 bytes (86.4 MiB); the owned
Bonsai server peaks at 1,379,778,560 bytes (1.29 GiB), measured separately. Both
owned native processes exited. These are CPU measurements; Mac/Windows hardware
inference is unvalidated.

Keep failures for inspection: `runs/dialogue8-whisper-small-q8-20261022/` exposed
a 109 ms native timestamp overshoot; the adapter now preserves native timestamps
with an explicit 0.5 s endpoint tolerance. Its sibling ending `-complete/` keeps
the successful transcript and the failed Bonsai temperature-0.7 generation.
Greedy decoding allows execution to finish, but note quality remains inadequate:
Bonsai turns a conditional chest-pain warning into a confirmed symptom, mixes
treatment into Objective, and appends repeated “No consta”. This smoke test is not
a quality or diarization evaluation.

Checks: 48 tests (47 pass, one opt-in live skip), lint/format and wheel/sdist build.
The built base wheel runs the suite without Torch: 44 pass, four optional skips.
No Python dependency changes. User notes and their README edit are unstaged.

## Original refactor scope

The first milestone is a readable local scribe core shared by Python, the CLI,
future evaluation adapters, and a future application server. Evaluation datasets,
judges, metrics, databases, microphone capture, and live streaming remain separate
work. Preserve `TODO`, `ARCHITECTURE_EVALUATION.md`, recordings, and existing runs.

## Public behavior

```python
with open_workflow("moss_bonsai") as scribe:
    full_run = scribe.run(AudioFile("consultation.wav"))
    speech = scribe.run(AudioFile("consultation.wav"), output="diarized_transcript")
    note = scribe.run(speech.transcript)
    manual = scribe.run(Transcript.from_text("An arbitrary conversation"))
```

- Workflow selection is required; the default output is a note.
- A workflow path may identify a TOML file or directory. Directory discovery uses
  `default.toml` or the sole workflow; multiple candidates are an error.
- Steps declare accepted inputs and their actual output. MOSS produces a
  diarized transcript; note generation accepts plain or diarized transcripts.
- The resolver uses supplied artifacts and configured steps, validates ambiguous
  or impossible paths, and stops once the requested output exists.
- Artifacts are immutable data, serializable, and reusable across executions.
  Removing speaker labels is an explicit transformation. No synthetic timestamps
  or hidden recovery of information from an original transcript.
- A session retains loaded models; each run owns its own artifacts and metadata.
- Opening a workflow checks/prepares its versioned resources. Model construction
  is lazy. Downloads can be disabled for offline execution.
- Persist each completed intermediate artifact before starting the next step, so
  a note-generation failure preserves the speech result.

## Intended layout

```text
src/scribnic/
  __init__.py               Public Python API
  artifacts.py              Audio, transcripts, note, run results
  config.py                 TOML configuration and preset discovery
  planning.py               Producer selection and dependency validation
  workflow.py               Planning, session lifecycle, execution
  audio.py                  WAV validation/loading
  notes.py                  Template loading and conversation formatting
  runtimes.py               Cached resources and owned llama-server lifecycle
  observability.py          OpenTelemetry and process resource measurements
  profiling.py              Optional PyTorch operation traces
  storage.py                Artifact and run serialization/checkpoints
  reporting.py              Terminal and HTML presentation
  cli.py                    Argument parsing and public API invocation
  backends/
    moss_transcribe_diarize.py
    http_generation.py
    demo.py
  presets/
    models/                 MOSS/Bonsai revisions and model defaults
    runtimes/               Transformers, local llama-server, external HTTP
    workflows/              Real and deterministic demo workflows
    notes/                  Plain-text note instructions
  assets/                   Local report template/styles
```

Keep files cohesive; split download/platform helpers only if their concrete
responsibility becomes hard to follow. Avoid a generic plugin framework.

## Checkpoints

- [x] Record agreed coding conventions in `AGENTS.md`.
- [x] Inspect current behavior, dependencies, Git state, and backend boundaries.
- [x] Commit/push the plan on `refactor/workflow-core`.
- [x] Implement typed artifacts, configuration, planning, and the reusable API.
- [x] Test default note output, early stopping, transcript re-entry, isolation,
      unsupported outputs, missing producers, ambiguity, and cycles; commit/push.

Core checkpoint: 31 tests pass, including the 19-test baseline; 3 live checks are
opt-in. Tests demonstrate speech is neither constructed nor called for supplied
text, loaded backend reuse does not leak artifacts, and removing speakers keeps
the original unchanged.
- [x] Port MOSS and generic HTTP generation; add versioned local llama-server
      preparation/lifecycle and model sampling presets; commit/push.

Backend checkpoint: 39 tests pass (3 live checks opt-in). A real `bonsai_note`
session starts the cached Prism server, generates through the new HTTP adapter,
preserves native timings, and closes its process. Sampling uses Prism's published
small-model non-thinking profile. The short synthetic note still has semantic and
formatting mistakes; execution success is not a quality assessment.
- [x] Add OTel file export, resources, optional Torch profiling, persistence,
      partial-failure recovery, and a standalone HTML report; commit/push.

Observability checkpoint: local OTLP traces/metrics/logs export successfully in a
clean installation without Torch. Recovery tests verify that a failed note keeps
the diarized transcript and an error report. Process sampling covers Scribnic and
owned local runtimes; remote service resources require service-side telemetry.
- [x] Replace the CLI with a thin workflow interface; remove discarded NeMo,
      Nemotron, Qwen-ASR, and VibeVoice paths and obsolete scripts/tests.
- [x] Separate portable client dependencies, optional MOSS dependencies, and dev
      tools; configure Ruff, refresh the lock, and document platforms; commit/push.

Cleanup checkpoint: the public API and CLI no longer import the old pipeline.
Removed the superseded source, copied NeMo scripts, legacy command wrapper, and
obsolete backend tests. Preserved notes and local data. A clean locked MOSS install
uses standard CPU Torch on Linux; the original custom ROCm environment is intact.
Mac/Windows archive hashes and executable layouts were checked; hardware inference
on those platforms remains untested.
- [x] Run unit/style/package checks and a small real MOSS/Bonsai workflow.
- [x] Demonstrate saved transcript -> explicit modification -> note generation;
      inspect note/transcript quality and record actual measurements.
- [x] Update this plan and README with results, limitations, and execution commands;
      final commit/push.

Final validation on 20 October 2026:

- A clean locked install with MOSS uses Torch 2.14.1+cpu and Transformers 5.20.2.
  Its suite passes 30 tests; the separate real-model test is opt-in and skipped.
  The built wheel also passes the base suite without Torch (29 passed, 2 skipped).
- Ruff lint/format, lock consistency, whitespace checks, wheel and sdist build pass.
  Packaged presets and HTML/CSS are included. No production source function exceeds
  the 50-body-line review threshold. User notes remain unchanged from the baseline.
- Actual CPU run: `runs/refactor-real-20261020-154439/full/report.html`.
  A 35-second WAV produced 13 diarized segments with 2 speaker IDs and a Spanish
  note. Total run: 108.68 s; speech: 100.40 s (model loading 6.01 s); note: 4.61 s.
  Sampled peak RSS: application 5.18 GiB, owned llama-server 1.25 GiB, measured
  separately. CPU thread setting: 6; system: 24 logical CPUs, 11.56 GiB WSL memory.
- The saved diarized transcript was explicitly converted, saved, loaded and
  resubmitted without speaker labels. Only the note step ran (3.37 s), preserving
  the original. Its report is in the sibling `without-speakers/` directory. The
  server/prefix cache was reused, so timings are not a controlled comparison.
- A separate four-second real MOSS run exported a CPU operator trace and report
  under `runs/refactor-profile-20261020-154812/run/`. The trace includes frontend,
  encoder, adaptor, prefill and decode labels; it is 213 MiB. Detailed profiling
  is opt-in and GPU activity requires an additional explicit runtime setting.
- The first AMD/ROCm GPU run failed with a MIOpen convolution error. A second
  profiled attempt was incomplete when WSL restarted. Neither GPU performance
  nor the workaround is considered validated. CPU completed using the portable
  lock; macOS and Windows inference still need their own hardware checks.
- Execution succeeds, but quality is not established. MOSS transcribed "valares"
  for "valores"; Bonsai misplaced content in Objective, inferred unsupported
  assessment wording and repeated "No consta". Preserve these predictions for
  the future external evaluator; do not equate the smoke test with clinical
  accuracy.

## Dependency and runtime decisions

- Keep HTTP generation reusable for LM Studio, llama-server, and other compatible
  endpoints. Do not add llama.cpp Python bindings.
- Pin Bonsai weights and a compatible Prism llama.cpp release, including hashes.
  Download the configured model/runtime, not the full demo UI/tooling stack.
- MOSS uses its Hugging Face checkpoint through Transformers. Keep Torch imports
  confined to that optional backend. No NeMo or machine-specific ROCm stack in
  the portable installation.
- Use OpenTelemetry directly, with local OTLP JSONL export. The application owns
  SDK/exporter setup; the core emits spans and metrics.
- Detailed Torch profiling is opt-in and linked as a run artifact. Native
  llama.cpp timings are preserved alongside process resource measurements.
- Prefer local execution. Remote endpoints require an explicit configuration.
  Record effective models, runtimes, settings, and any configured fallback.
- Validate Linux locally. Package Windows/macOS runtime choices without claiming
  unperformed hardware or model compatibility tests.

## Follow-on work

### Detailed profiling report, 21 October 2026

`--profile` now flushes compressed traces every 16 language-model forwards and
aggregates every window in `profiles/moss-profile.json`. The first encoder,
adaptor and prefill plus every decode call are captured without sampling; the
final partial window and failures are flushed too. Decoder events are released
between windows. This does not place a fixed bound on encoder/prefill memory.
The HTML report includes inclusive phase times, the top operators by self CPU
time and signed net tensor allocations. Export/aggregation overhead is recorded
separately. Stack recording is disabled; shapes and tensor memory are enabled.

A real 12-second excerpt of `dialogue8` completed MOSS -> Bonsai under this profiler:
`runs/profile-check-20261021/fast-export/report.html`.

- 9 compressed trace blocks, 50.4 MiB total; 136 generated tokens.
- Encoder 3.44 s; adaptor 0.006 s; prefill 0.77 s; 135 decode calls totaling 16.64 s.
- Matrix multiplication (`aten::mm` plus `aten::addmm`) accounts for 82.5% of
  operator self CPU time in this excerpt. These percentages are not measurements
  of the complete five-minute recording or of GPU execution.
- Profiled run total 163.59 s; export/aggregation callbacks account for 118.45 s.
  Detailed profiling is intrusive; use the earlier unprofiled recording for
  performance comparisons. Low-compression streaming reduces trace export cost.
- Peak sampled application RSS 5.33 GiB. This includes profiling overhead and
  is not a minimum client memory requirement.
- 38 tests pass and one live test is opt-in; the built base wheel passes 35
  tests with four optional checks skipped. Tests check full coverage across window
  boundaries, the final partial window, error cleanup, and escaped HTML rendering.

The README gives the full-recording command. This capture covers Torch operators
inside MOSS; llama-server contributes native timings/metrics and process resources,
but its internal C++/GGML operators still require native tooling.

### CPU execution diagnosis, 21 October 2026

The normal `uv run --extra moss ... --workflow moss_bonsai` command selected the
portable Torch 2.14.1+cpu installation. The earlier 297.891-second recording had
used `cuda:0` with ROCm (106.2 seconds of speech inference). The two interrupted
CPU attempts stopped during speech after 38.5 and 447.7 seconds; the note service
was idle. The CLI only exposed stage boundaries, hiding ongoing generation.

The follow-up fix reports loading, effective device/dtype/CPU threads and Torch
version, then encoder/prefill and generated-token counts about every five seconds.
Progress is also exported as OTLP logs without clinical text. The Transformers
preset sets six CPU threads temporarily and restores the previous setting. Run
metadata now records runtime versions, thread count and time to first token.

Full reproduction completed with the user's command and `dialogue8.wav`:

- Report: `runs/dialogue8-cpu-progress-20261021/report.html`.
- Status `success`; total 770.93 seconds (12 minutes 51 seconds).
- MOSS: 755.07 seconds, 3967 input tokens, 2778 generated tokens, 81 segments and
  two speaker IDs. Time to first token: 47.67 seconds. Float32 CPU, six threads.
- Bonsai: 13.23 seconds; 1966 input tokens, 169 output tokens. Both models finish
  and the managed llama-server exits normally. Deep profiling was disabled.
- Unit suite: 36 passed, one opt-in live test skipped. Built-wheel base suite:
  34 passed, three optional tests skipped. Lint, formatting, lock and build pass.
- Tests verify progress rate limiting/log export, restoration of host logger and
  CPU-thread settings, and unchanged greedy output with the progress callback.

This fixes missing runtime/progress diagnostics. It does not restore the previous
ROCm acceleration: the portable Linux installation remains CPU by design. CPU
performance must not be presented as equivalent to the earlier GPU run.

After this core is stable: create the separate `scribe_eval` adapter, start with
saved predictions and a few PriMock57 samples, add automatic metrics and paper
protocols, then scale datasets/experiments. Keep quality evaluation independent of
inference and reuse recorded predictions for rescoring.
