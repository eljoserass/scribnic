# Changelog

All notable changes to Scribnic are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is
`0.x`, a minor release may contain breaking changes; they are listed under
**Breaking changes**.

Draft entries with `uvx git-cliff --unreleased` and edit them before committing;
see [docs/releasing.md](docs/releasing.md).

## [Unreleased]

First tagged release. It summarizes development from 2026-09-25 to 2026-11-18.

### Features

- Workflow core: artifacts (`AudioFile`, `Transcript`, `DiarizedTranscript`,
  `Note`), declared step inputs and outputs, dependency resolution before model
  preparation, and a reusable `WorkflowSession` that keeps models loaded and
  artifacts isolated per request.
- Model and runtime presets with pinned repositories, revisions, SHA256 checksums,
  and weight formats, prepared from a versioned cache by `scribnic setup`.
- Managed local runtimes for `whisper-server`, `llama-server`, and NVIDIA
  `nemo-speech`, plus external OpenAI-compatible endpoints.
- Speech recognition presets: Whisper tiny through large-v3 and turbo, Parakeet
  TDT 0.6B v3, Nemotron streaming ASR and diarization, and MOSS Transcribe-Diarize.
- Note generation presets: Ternary Bonsai 2B/4B, Qwen3.5 0.8B/2B/4B, and LFM3 1.7B
  in Q4/Q8 quantizations, with Spanish SOAP templates.
- Direct audio-to-note workflows with LFM3-Audio (incremental prefill) and
  MOSS-Audio-4B-Thinking.
- Incremental audio: `run_stream` with windowed Whisper over HTTP and NVIDIA
  WebSocket streaming, including transcript snapshots and a final note.
- Note generation from supplied plain or diarized transcripts, without speech
  recognition.
- Typer CLI with Rich progress, `run` and `setup` commands, shell completion, and
  `--version`.
- Observability: per-step checkpoints, OpenTelemetry spans and metrics with
  optional local JSONL export, CPU/RSS sampling of owned processes, optional
  operator profiles, and an HTML run report.

### Bug fixes

- Reject unknown keys in workflow files instead of ignoring them, and reject note
  steps without a template before any server starts.
- Keep engine timestamps instead of rejecting transcripts with unusual timing.
- Group windowed Whisper words into sentences.
- Read the MOSS checkpoint index as UTF-8 instead of the locale encoding
  (Windows).
- Show service loading before inference progress, across all adapters.
- Replace quantized weights from personal accounts with sources from identified
  organizations.

### Documentation

- English documentation for the CLI, Python API, presets, streaming, model sources,
  and the validated model/runtime/platform combinations.
- Contribution, security, and release guides.
- README starts with the most exercised workflows (`bonsai_note`, `whisper_bonsai`),
  lists models by role and family, and summarizes how a model gets a preset.

### Maintenance

- Model presets are grouped by role and family under `presets/models/`
  (`speech/`, `note/`, `audio_note/`). Preset names are unchanged, and duplicate
  names are rejected.
- Release under the MIT license, with citation metadata in `CITATION.cff`.
- Continuous integration for linting, formatting, base installation without Torch,
  and unit tests; Dependabot updates; pre-commit hooks.

[Unreleased]: https://github.com/eljoserass/scribnic/commits/HEAD
