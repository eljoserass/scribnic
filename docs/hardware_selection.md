# Hardware and compute selection (proposal)

Proposal from 2026-10-28. This document describes possible changes; nothing here
is implemented, and current automatic selection is unchanged.

## Current state

Scribnic selects native binaries by OS/architecture in `downloads.py` and checks
Torch devices inside the optional backends. Dtype and acceleration decisions are
spread across backends and use conservative rules:

| Runtime | Current automatic choice | Gap |
| --- | --- | --- |
| Torch: `moss_transcribe_diarize` | CUDA if available; BF16 on CUDA, FP32 on CPU | Ignores CPU BF16; CUDA availability does not prove native BF16 |
| Torch: `moss_audio` | CUDA, then MPS, then CPU; BF16/FP16/FP32 | Same CPU gap; dtype not checked per device and model |
| llama-server upstream and Prism | Pinned download per platform; `gpu_layers=auto` offloads everything on Darwin and nothing elsewhere | The OS does not prove build capabilities or enough memory |
| whisper-server | `use_gpu=auto` enables GPU by default only on Darwin | Compiled backend, usable device, and effective choice are not checked |
| NeMo-Speech.cpp | `device=cpu` unless configured | Distribution capabilities are not queried |
| External endpoint | Connects to the given service | The client neither knows nor owns its hardware |

The [MOSS CPU BF16 vs FP32 test](moss.md#in-process-transformers) shows the case
worth solving: the CPU supports BF16 and the model path works, but `auto` chose
FP32. That does not mean every CPU or backend benefits.

## Proposed decision flow

1. **Read the request.** Separate explicit device/dtype from `auto`. Keep the
   selected model, quantization, and revision; choosing BF16 compute never means
   switching a Q8 model to Q4 or downloading other weights.
2. **Query what the process can see.** OS, architecture, affinity, container/VM
   limits, available memory, and exposed capabilities. A host GPU may be
   unavailable inside WSL or to the installed library.
3. **Query the selected runtime.** Torch CPU, CUDA/ROCm, and MPS; or the
   capabilities and usable devices of the pinned native build. Use each release's
   own mechanism rather than assuming a common `--list-devices`.
4. **Intersect capabilities with model support.** BF16 hardware, available kernels,
   and a compatible model path are separate conditions. Select only combinations
   backed by that evidence.
5. **Prepare and record.** Validate an explicit request before loading weights; read
   the effective device and compute from the runtime when it exposes them; record
   any difference from the request and why.

An incompatible explicit request must fail with a clear error. For `auto`, an
unknown capability allows a conservative choice with a recorded reason. An
arbitrary inference error does not justify silently changing dtype and retrying.
External endpoints report `unknown` when the server exposes nothing, without
inspecting local hardware as a substitute.

## Useful signals per runtime

| Scope | Signals and proposed criterion |
| --- | --- |
| Torch on CPU | Public `torch.cpu.get_capabilities()` where available; BF16 specifically, not just the AVX512 level. oneDNN availability and an optional functional probe; older versions report unknown |
| CUDA/ROCm | Installed build, availability, device, memory, and dtype support; `torch.cuda.is_bf16_supported(including_emulation=False)` where supported, separating native from emulated compute |
| Apple MPS | Real MPS availability and operator/dtype support for the macOS, Torch, and model combination; do not carry CUDA rules over |
| GGML engines | Distribution/release, compiled backends, devices, and effective offload reported by the process; keep the tensor quantization and separate KV cache and operation dtypes |
| Memory | Lower-bound estimate of weights plus reserves for context, audio, and buffers; shared memory, mmap, and RSS are not the file size |

On the validation machine, Torch reports `avx512_bf16=True` and `avx512_fp16=False`,
so BF16 and FP16 capabilities must not be treated as equivalent. The `2.14.1+cpu`
build offers no CUDA/ROCm even though the chip includes a Radeon GPU, which has not
been validated with any selected runtime.

Capability queries do not prove BF16 is faster for the whole workload. A short
opt-in probe could live in a future diagnostics command, without downloading models
or turning every session start into a benchmark. Cached diagnostics would be
invalidated when visible hardware, OS, library version, build, or model changes.

References: [CPU capabilities](https://docs.pytorch.org/docs/2.14/generated/torch.cpu.get_capabilities.html),
[BF16 support on CUDA/ROCm](https://docs.pytorch.org/docs/2.14/generated/torch.cuda.is_bf16_supported.html).

## Files a future implementation would touch

| File | Responsibility |
| --- | --- |
| New `src/scribnic/runtimes/compute.py` | Small capability and decision dataclasses; testable selection separating requested, supported, and effective. No Torch import at module load |
| `src/scribnic/downloads.py` | Share platform identity and the prepared distribution's capabilities; keep pinned downloads and builds |
| Torch backends (`moss_transcribe_diarize.py`, `moss_audio.py`) | Query Torch only when the backend loads; apply the shared dtype decision while keeping each backend's version and memory constraints |
| `src/scribnic/runtimes/{llama_cpp,whisper_cpp,nemo_cpp}.py` | Translate a compatible selection into server arguments and collect the effective choice, without prompt logic |
| `src/scribnic/config.py` | Validate explicit values and represent only settings actually used |
| `src/scribnic/telemetry/otel.py` and persistence | Attach requested, selected, effective, evidence, and reason to the run, separate from the weight format |
| CLI and docs | Show diagnostics and allow overrides; the decision must also work through the Python API |
| New `tests/test_compute.py` | Fakes for CPU with/without BF16, unknown capabilities, GPU without a compatible runtime, overrides, external endpoints, and decision provenance |

Suggested first step: resolve only BF16/FP32 in the Torch backends through one
shared, recorded decision. Native server detection would follow. No new inference
backend or runtime is needed for this.
