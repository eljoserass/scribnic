# NVIDIA speech experiments

The same workflow API supports Nemotron 3.5 ASR (multilingual, including Spanish)
and Nemotron 3 Diarization through two implementations. Both return the existing
`DiarizedTranscript` artifact and can feed Bonsai without a CLI dependency.

| Workflow | Runtime | Speech execution |
| --- | --- | --- |
| `nvidia_cpp_offline` | Official NeMo-Speech.cpp v0.2.0, Q8 GGUF | Whole-file ASR and word-time speaker attribution |
| `nvidia_cpp_streaming` | Same native runtime and weights | Recorded WAV replay using native streaming caches |
| `nvidia_nemo_offline` | Pinned NeMo/PyTorch, original checkpoints | Whole-file ASR, diarization, and temporal word attribution |
| `nvidia_nemo_streaming` | Same NeMo/PyTorch runtime and checkpoints | NVIDIA's parallel, speaker-masked ASR integration |

The native and Python streaming implementations are different experiments.
The native word-tagging path does not implement the reference Python helper's
speaker-conditioned ASR. `streaming` here means replaying a recorded WAV through
stateful inference; this API does not capture a microphone or accept live chunks.
No artificial real-time sleeps are inserted.

## Install and run

The native workflows need only the base installation. Resource preparation
downloads a pinned official release and the two pinned GGUF files, verifies
SHA-256, and installs the runtime into Scribnic's cache. No external executable
path, bindings, compiler, NeMo toolkit, or Torch is needed.

```bash
uv run scribnic prepare --workflow nvidia_cpp_streaming
uv run scribnic run consulta.wav --workflow nvidia_cpp_streaming \
  --profile --output-dir runs/nvidia-cpp
```

Release assets are configured for Linux x86_64/arm64 CPU, Windows x86_64 CPU, and
Apple Silicon Metal. `device = "auto"` selects an available backend. Linux CPU is
the locally tested path; Mac and Windows execution still need platform validation.
The managed archive contains the runtime's libraries alongside the executable.

The reference Python backend is an optional extra, pinned to NeMo Speech commit
`16b0ff2584c6355c53cbd6c28f0f431a76b24956`. It is not part of the client base:

```bash
uv sync --locked --extra nvidia
uv run --extra nvidia scribnic prepare --workflow nvidia_nemo_streaming
uv run --extra nvidia scribnic run consulta.wav --workflow nvidia_nemo_streaming \
  --profile --output-dir runs/nvidia-nemo
```

Use `uv sync --extra moss --extra nvidia` when developing both Python backends.
`uv run --no-sync` preserves an already synchronized environment. The portable
lock selects CPU Torch on Linux/Windows; it does not install GPU acceleration.
Use a separate environment for a CUDA/ROCm build rather than overwriting it with
the portable lock. NeMo defaults to CPU; MPS remains an explicit, unvalidated choice.

`prepare` populates/checks resources without loading weights. `run` also prepares
missing resources automatically. Use `--offline` once cached to disallow resource
downloads. Original `.nemo` checkpoints are larger than their Q8 GGUF counterparts:
the ASR file is about 2.37 GB versus 742 MB; diarization is 199 MB versus 107 MB.
Loading original weights can temporarily require substantially more RAM than
their on-disk size. Run heavy experiments sequentially on constrained machines.

## Configuration and reuse

Copy a preset workflow and change its TOML settings:

```toml
[steps.speech]
model = "nemotron_3_5_asr_nemo"
runtime = "nemo_python"
output = "diarized_transcript"

[steps.speech.companions]
diarization = "nemotron_3_diarization_nemo"

[steps.speech.model_settings]
language = "es-ES" # "auto" requests language inference.
mode = "streaming" # "offline" uses independent whole-file inference.
max_speakers = 8
attention_context = [56, 13]
sentence_gap_seconds = 1.0

[steps.speech.model_settings.streaming]
fifo_len = 264
spkcache_update_period = 222
cache_gating = true

[steps.speech.runtime_settings]
device = "cpu"
threads = 6
profile_window_steps = 16
profile_gpu = false
```

The Python streaming adapter uses `masked_asr=true`, parallel speaker streams,
binary diarization predictions, and fresh speaker/decoder caches for each run.
Advanced streaming overrides are validated; input paths and the integration
strategy cannot be overwritten through this settings table. In the native path,
speaker capacity is fixed by the loaded model (eight); `max_speakers` cannot
silently pretend to reduce it. Native engine controls can be set in
`steps.speech.runtime_settings.asr`, for example
`"streaming.rnnt_right_context" = 13` or `"diar.preset" = "v3-offline"`.

Python sessions retain loaded models, but each `run` creates fresh per-recording
state. The native CLI currently starts one process per run and releases its weights
when that process ends. Native timings therefore include loading and warmup;
Python metadata records loading separately. Note loading is deferred in the NVIDIA
workflows so Bonsai is not resident during the NeMo checkpoint-loading peak.

To obtain only speech, use `--output diarized_transcript`. For an ASR-only offline
workflow, set the speech output to `transcript` and remove the diarization companion.
To test note generation without speaker labels after a paired run:

```python
from scribnic import AudioFile, open_workflow

with open_workflow("nvidia_nemo_streaming") as scribe:
    result = scribe.run(AudioFile("consulta.wav"), output="diarized_transcript")
    ablation = scribe.run(result.transcript.without_speakers())
```

## Reports and profiling

Every persisted run produces `report.html`, typed artifacts, effective configuration,
both model revisions/hashes, native outputs under `runtime/`, and OTLP traces,
metrics, and logs under `telemetry/`. RSS/CPU sampling includes owned C++ processes;
NeMo executes in the monitored application process. RTF is processing time divided
by source-audio duration. It is not a quality score.

`--profile` additionally captures NeMo Torch operators in compressed Perfetto
traces under `profiles/`, including ASR encoder/decoder/joint ranges, the diarization
encoder, and chunk/component phases. Exports happen at configured chunk boundaries;
an offline component can still produce a large capture. Export overhead, net tensor
allocations, activities, coverage, and aggregate operator totals are recorded in
`*-profile.json` and displayed in HTML. Loading is outside the operator capture.
Profiled times include instrumentation overhead and should not be speed benchmarks.

For the native engine, `--profile` writes an explicit coverage declaration: process
resources and OTel spans are available, C++ operators are not captured by Torch.
The native diagnostic log is retained. Use a native profiler for operation-level
analysis of that executable. No synthetic operator timings are reported.

The initial native test assigned all words to one speaker. Successful execution
does not establish diarization quality, transcription accuracy, or clinical note
quality. Reference-based scoring and controlled comparisons belong to the future
evaluation harness.

Reference implementation: [NVIDIA's ASR integration guide](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/main/ASR_INTEGRATION_GUIDE.md).
Native runtime: [NVIDIA/NeMo-Speech.cpp v0.2.0](https://github.com/NVIDIA/NeMo-Speech.cpp/tree/v0.2.0).
