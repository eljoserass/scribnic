# Workflow core refactor

The first milestone is a readable local scribe core shared by Python, the CLI,
future evaluation adapters, and a future application server. Evaluation datasets,
judges, metrics, databases, microphone capture, and live streaming remain separate
work. Preserve `TODO`, `ARCHITECTURE_EVALUATION.md`, recordings, and existing runs.

## Public behavior

```python
with open_workflow("moss_bonsai") as scribe:
    full_run = scribe.run(AudioFile("consultation.wav"))
    speech = scribe.run(
        AudioFile("consultation.wav"), output="diarized_transcript"
    )
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
- [ ] Port MOSS and generic HTTP generation; add versioned local llama-server
      preparation/lifecycle and model sampling presets; commit/push.
- [ ] Add OTel file export, resources, optional Torch profiling, persistence,
      partial-failure recovery, and a standalone HTML report; commit/push.
- [ ] Replace the CLI with a thin workflow interface; remove discarded NeMo,
      Nemotron, Qwen-ASR, and VibeVoice paths and obsolete scripts/tests.
- [ ] Separate portable client dependencies, optional MOSS dependencies, and dev
      tools; configure Ruff, refresh the lock, and document platforms; commit/push.
- [ ] Run unit/style/package checks and a small real MOSS/Bonsai workflow.
- [ ] Demonstrate saved transcript -> explicit modification -> note generation;
      inspect note/transcript quality and record actual measurements.
- [ ] Update this plan and README with results, limitations, and execution commands;
      final commit/push.

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

After this core is stable: create the separate `scribe_eval` adapter, start with
saved predictions and a few PriMock57 samples, add automatic metrics and paper
protocols, then scale datasets/experiments. Keep quality evaluation independent of
inference and reuse recorded predictions for rescoring.
