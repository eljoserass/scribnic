# Model sources and quantizations

Source review: 2026-10-28. This is the selection policy for the presets bundled
with Scribnic.

## Accepted sources

Prefer weights published by the model author's organization. For conversions,
accept identifiable organizations and document who trained the model and who
published the conversion. Do not add quantizations from personal accounts.

One explicit exception: weights from the official
[`ggerganov/whisper.cpp`](https://huggingface.co/ggerganov/whisper.cpp) repository,
maintained under the project creator's account. The exception does not extend to
other accounts.

An organization account does not certify weight quality. Pinning revision and
SHA256 makes a download reproducible; it does not validate the model, its
quantization, or its clinical quality. A preset configures a possible run; measured
platforms and results are documented separately.

If no acceptable source exists for a quantization, keep the original checkpoint
when the backend supports it, or leave the variant out. Never substitute another
conversion silently. Adding a new organization requires reviewing its identity,
relationship to the model, files, license, and runtime.

Handy weights and the transcribe.cpp runtime are out of scope. Parakeet Ultra and
Redux would come in a separate branch using Photon and Moondream's original
weights; native preset additions stay separate from those integrations.

## Catalog

| Family | Publisher of selected weights | Decision |
| --- | --- | --- |
| Qwen3.5 0.8B, 2B, 4B, Q4_K_M / Q8_0 | [Unsloth AI](https://huggingface.co/unsloth) | Organization conversions of Qwen's models; Qwen publishes no GGUF for these sizes |
| Bonsai 2 2B / 4B, PQ2_0 | [Prism ML](https://huggingface.co/prism-ml) | Author's weights |
| LFM3-1.7B and LFM3-Audio-2B, including QAD and mmproj | [LiquidAI](https://huggingface.co/LiquidAI) | Author's weights |
| Nemotron 3.5 ASR, Nemotron 3 Diarization, Parakeet TDT v3, Q8_0 | [NVIDIA](https://huggingface.co/nvidia) | Author's weights |
| Whisper tiny, base, small, medium, turbo Q8_0; large-v3 Q5_0 | Official whisper.cpp repository | Approved runtime-distribution exception; see [speech presets](asr_presets.md) |
| Silero VAD for whisper.cpp | [ggml-org/whisper-vad](https://huggingface.co/ggml-org/whisper-vad) | Runtime organization |
| MOSS-Transcribe-Diarize, MOSS-Audio-4B-Thinking | [OpenMOSS-Team](https://huggingface.co/OpenMOSS-Team) | Original BF16 checkpoints; no third-party quant |

Earlier Qwen presets used conversions from a personal account; the switch to
Unsloth kept model and workflow names. Older cached files are not deleted, so a
comparison with historical runs must tell both publishers apart even when they
share a Q4/Q8 label.

## Own quantization

A local quant made from an official checkpoint is our own experiment, not an
official author quant. Before adding it as a preset, record the source revision,
tool and version, method, converted modules, compute dtype, files, and evaluation
results. It requires neither publishing the weights nor changing the base client
installation. No own quantization is implemented yet.

Fewer bits shrink the converted tensors, not activations, KV cache, or the whole
process in the same proportion, and they do not guarantee speed without efficient
kernels. Changing `dtype` to `int8` in Transformers does not replace layers or
enable quantized kernels; that needs an explicit `quantization_config` method that
is compatible with the custom model. A GGUF export is only useful if the pinned
runtime implements the architecture; the pinned llama.cpp does not implement MOSS.

## Publisher notes

**Unsloth** builds tools for running and fine-tuning models and also publishes
converted weights. Using one of its GGUF files in our llama-server requires neither
its library nor its Studio interface. Its catalog mixes GGUF and safetensors, some
quantized with bitsandbytes and some floating point; neither the publisher name nor
the extension identifies precision. Dynamic 2.0 mixes per-layer precisions chosen by
calibration; its published MMLU and KL results do not validate Spanish clinical
notes, and our presets use plain `Q4_K_M` and `Q8_0` files.
[Docs](https://unsloth.ai/docs), [catalog](https://unsloth.ai/docs/get-started/unsloth-model-catalog),
[Dynamic 2.0](https://unsloth.ai/blog/dynamic-v2).

**Whisper from Unsloth.** `unsloth` publishes Whisper safetensors, which have no
local Whisper/Transformers backend in Scribnic. The secondary `unslothai` account
(linked from Unsloth's own code at commit
[`4b71ee3`](https://github.com/unslothai/unsloth/blob/4b71ee393d12addf7ee1bc5952273ca05f404d88/studio/backend/core/inference/stt_ggml_sidecar.py#L88-L101))
publishes whisper.cpp GGML `.bin` files in F16, despite the `-GGUF` repository
names. They would work with `whisper_cpp`, not `llama_cpp`, after pinning and
testing, but they are a different precision from our Q8_0 files. Their cards declare
Apache-2.0 while OpenAI's original Whisper is MIT and the repositories contain no
`LICENSE` file, so origin notices would need checking before distribution. We keep
the official whisper.cpp Q8_0 files.

**Other organizations.** [ggml-org](https://huggingface.co/ggml-org) is preferred
when it publishes the needed model and format.
[LM Studio Community](https://huggingface.co/lmstudio-community) GGUF files can run
on our llama-server if the architecture is supported.
[MLX Community](https://huggingface.co/mlx-community) targets Apple Silicon; Scribnic
has no MLX backend. Being listed here does not approve an organization's whole
catalog: each addition identifies the original author, converter, license,
precision, and a verifiable file, and is tested with the chosen runtime.

Weight formats map to runtimes, not publishers: GGUF chat models to `openai_chat` +
`llama_cpp`, whisper.cpp `.bin` to `openai_transcription` + `whisper_cpp`,
bitsandbytes or MLX weights to nothing yet. A compatible HTTP chat server can reuse
`openai_chat`, but that does not establish audio, timestamp, or diarization support.
