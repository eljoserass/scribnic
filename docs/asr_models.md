# Speech and audio-to-note candidates

Research from 2026-10-28. The project goal is to keep Spanish clinical information
and its attribution at a reasonable local cost; an ASR score alone does not show
that the final note is good. Integrated models are listed in the
[README](../README.md#workflows-by-backend-and-runtime) and
[speech presets](asr_presets.md). Everything below is a proposal for future
branches under the [source policy](model_sources.md).

## Speech candidates

Order of interest: Qwen3-ASR-0.6B as the next separate integration; Parakeet Ultra
for its public quality results; Parakeet Redux for memory and CPU cost, at the price
of clearer degradation with noise; Canary last, for integration effort and the
available medical evidence.

| Candidate | Spanish and speakers | Work needed in Scribnic |
| --- | --- | --- |
| [Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | 30 languages and 22 Chinese dialects; no diarization | Reuse the llama-server manager; adapt the audio request and ASR response conversion; add presets and a text contract without invented timestamps |
| [Parakeet Ultra](https://huggingface.co/moondream/parakeet-ultra) | 25 languages; no diarization | Integrate Photon, or convert and validate its safetensors including the new VAD head; the current `.nemo` converter does not cover it |
| [Parakeet Redux](https://huggingface.co/moondream/parakeet-redux) | Same 25 languages; ternary encoder | Integrate the author's runtime and packed format; not a GGUF swap in our NeMo runtime |
| [Canary 1B v2](https://huggingface.co/nvidia/canary-1b-v2) | 25 languages; transcription and translation; no diarization | Backend for its NeMo/Transformers path or extra native support; the pinned C++ runtime lacks its AED decoder |
| [Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | English only | Not a priority for Spanish consultations despite a good English medical score |

Parakeet is a TDT transducer that predicts tokens and durations; Canary is an
attention encoder-decoder that adds translation. Sharing the NVIDIA name does not
make architectures or protocols interchangeable. Whisper, Qwen, Parakeet, and
Canary all need an explicit speaker-attribution strategy: a transcript of a
conversation is not a diarized transcript.

**Qwen3-ASR.** llama.cpp b11946 lists it, and
[ggml-org publishes a Q8 GGUF and companion](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF),
so the manager can be reused; our transcription backend does not yet convert its
language-and-text response. The official model streams through vLLM, which does
not mean our llama-server client has that protocol. Word timestamps need an extra
alignment path such as ForcedAligner.
[Pinned engine capabilities](https://github.com/ggml-org/llama.cpp/blob/b11946/docs/multimodal.md),
[author's contracts](https://huggingface.co/Qwen/Qwen3-ASR-0.6B#quickstart).

**Ultra and Redux.** Moondream post-trained Ultra from NVIDIA's Parakeet v3; Redux
is the ternary variant (not to be confused with Useful Sensors'
[Moonshine](https://huggingface.co/moonshine-ai/moonshine-tiny)). Both split long
recordings at pauses into segments of up to 30 s with a built-in VAD head, which is
not native streaming ASR with a cache. NeMo-Speech.cpp v0.2.0 expects its own GGUF
schema and its converter reads `.nemo`, YAML, `.ckpt`, and a NeMo tokenizer, while
Ultra ships `config.json`, safetensors, and a JSON tokenizer
([pinned converter](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/v0.2.0/conversion/source.py)).
Options are a compatible converter with validated execution, or the author's Photon
runtime, which Ultra and Redux could share. Redux also needs kernels for its
[`thrush-ternary-v2`](https://huggingface.co/moondream/parakeet-redux/blob/main/ternary.json)
format (five ternary values per byte with group scales), which is neither Bonsai's
PQ2_0 nor a generic GGUF Q2.

**Licenses.** Qwen3-ASR declares Apache-2.0. Parakeet v2/v3, Ultra, Redux, and
Canary v2 declare CC BY 4.0. The integrated Nemotron models use
[OpenMDW-1.1](https://openmdw.ai/license/1-1/), a different license; do not assume
one license for every NVIDIA model, or for third-party conversions using their names.

## Published weights

Decimal file sizes, not inference RAM.

| Model | Verified source and format | Weights |
| --- | --- | ---: |
| Parakeet Ultra | Moondream safetensors, F16 with some F32 tensors; no author Q4/Q8 | 1.255 GB |
| Parakeet Redux | Moondream packed ternary encoder, other weights F16 | 178 MB |
| Qwen3-ASR-0.6B | ggml-org GGUF Q8_0 text model + Q8_0 audio companion | 805 + 214 MB |
| Canary 1B v2 | NVIDIA safetensors F32 and NeMo checkpoint; no author Q4/Q8 | 3.916 GB |

Counting only Qwen's text file omits the audio encoder. A
[4-bit MLX Qwen3-ASR](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-4bit)
(about 708 MB) exists but needs MLX on Apple Silicon. Redux's small size is a
trained ternary variant, not the expected result of converting any Parakeet to two
bits, and does not guarantee lower error.

## Benchmarks and their limits

Moondream's own measurements, same audio and normalizer; WER in %, lower is better.
Ultra and Redux run on Photon, the original on NeMo BF16.

| Test | Parakeet v3 | Ultra | Redux |
| --- | ---: | ---: | ---: |
| FLEURS Spanish, read sentences | 3.12 | 2.72 | 3.71 |
| FLEURS Spanish + MUSAN noise, 0 dB SNR | 5.44 | 4.64 | 7.22 |
| AMI, English meetings | 10.86 | 9.77 | 10.80 |
| TED-LIUM 3, eleven 10–20 min English talks | 2.71 | 1.94 | 2.51 |

Sources: [Ultra](https://huggingface.co/moondream/parakeet-ultra#benchmarks),
[Redux](https://huggingface.co/moondream/parakeet-redux#benchmarks). FLEURS is read
Wikipedia sentences; AMI WER does not measure speaker attribution; TED-LIUM is long
audio but not Spanish consultations. Ultra's B200 speed figures do not predict CPU
latency.

[Omi Health](https://github.com/Omi-Health/medical-STT-eval#leaderboard) measured 55
simulated PriMock57 consultations in **British English** (updated 2026-04-29), in %:

| Model | WER | Medical M-WER | Medication M-WER |
| --- | ---: | ---: | ---: |
| Qwen3-ASR-0.6B | 9.83 | 4.95 | 13.7 |
| Parakeet TDT 0.6B v2 | 10.75 | 4.60 | 15.7 |
| Parakeet TDT 0.6B v3 | 9.35 | 5.90 | 20.6 |
| Canary 1B v2 | 14.32 | 9.40 | 18.0 |

This picks no Spanish medical winner and includes none of Ultra, Redux, MOSS, or our
Nemotron 3.5. Timings come from different hardware and the quants were not measured.

## Audio-to-note candidates

| Model | Fit and cost | Status and license |
| --- | --- | --- |
| [LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B) | Spanish; official Q4 GGUF and Q8 companion; incremental prefill | Integrated as [`lfm3_audio_note`](lfm3.md); LFM Open License 1.0 |
| [MOSS-Audio-4B-Thinking](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Thinking) | General audio-to-text; about 9.73 GiB BF16 weights | Integrated as [`moss_audio_note`](moss.md); full checkpoint not run here; Apache-2.0 |
| [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) | 2.3B effective, about 5.1B with embeddings; at most 30 s of audio | No workflow; needs chunking and audio protocol validation; Apache-2.0 |
| [Voxtral Mini 3B 2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | Spanish and direct summarization; about 4.68B total | Candidate; Apache-2.0; needs audio chat adaptation, prompt format, response conversion |
| [Nemotron-Labs-Audex-2B](https://huggingface.co/nvidia/Nemotron-Labs-Audex-2B) | General audio, English | Rejected for the commercial path: NVIDIA OneWay Noncommercial |

Voxtral documents up to 30 minutes for transcription and 40 for understanding with
about 9.5 GB of GPU memory in BF16/FP16. The pinned llama.cpp lists Voxtral and
Gemma 4, but the LFM3 WebSocket adapter is model-specific, so neither is a TOML-only
change, and Voxtral does not accept our template's `system` usage as-is.
[LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) is smaller
but English-only.

When comparing ASR locally, keep audio, reference, generator, and prompt fixed; when
comparing generators, reuse exactly the same transcript. Record figures, doses,
negations, follow-up, and attribution, plus load/inference time and per-process
memory.
