# MOSS

MOSS was the first speech model Scribnic integrated. It is now one model family
among the others, available through these presets:

| Workflow | Backend | Runtime preset | Model preset | Extra |
| --- | --- | --- | --- | --- |
| `moss_bonsai` | `moss_transcribe_diarize` | `transformers` (in-process Torch) | `moss_transcribe_diarize` (BF16) | `[moss]` |
| `moss_bonsai_http` | `moss_transcribe_diarize` | `transformers`; note on `external_endpoint` | `moss_transcribe_diarize` (BF16) | `[moss]` |
| `moss_bonsai_sglang` | `openai_transcription` | `sglang_omni_endpoint` (external) | `moss_transcribe_diarize` | — |
| `moss_audio_note` | `moss_audio` | `moss_audio_torch` (in-process Torch) | `MOSS-Audio-4B-Thinking-bf16` | `[moss-audio]` |

MOSS-Transcribe-Diarize (0.9B) and MOSS-Audio-4B-Thinking are different models with
different architectures. Both checkpoints come from
[OpenMOSS-Team](https://huggingface.co/OpenMOSS-Team) in BF16; there is no official
Q4/Q8 and no third-party quant has been added (see [model sources](model_sources.md)).
Neither full checkpoint nor the SGLang service has been run on the validation
machine.

## MOSS-Transcribe-Diarize

The model produces text, timestamps, and speakers jointly, without an external
diarizer. [Model card](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize),
[authors' usage](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize).

### In-process Transformers

`moss_bonsai` loads the pinned snapshot
(`704aa4a9c304e8520be88901e0d1960158ef5b15`, 683 BF16 tensors, about 1.82 GB) with
pinned model code. Torch and Transformers are imported only inside the backend.
`dtype = "auto"` currently picks BF16 on CUDA and FP32 on CPU, so CPU runs widen
the weights at load time. To request BF16 on CPU:

```toml
[steps.speech.runtime_settings]
device = "cpu"
dtype = "bfloat16"
```

A short comparison on the validation CPU (AMD Ryzen AI Max PRO 390, which exposes
`avx512_bf16`; Torch 2.14.1+cpu, six threads, SDPA, first 15 s of one recording,
one cold pass per dtype in separate processes):

| Measure | BF16 | FP32 |
| --- | ---: | ---: |
| Generation, including prefill | 13.775 s | 30.521 s |
| Time to first token within generation | 2.147 s | 4.193 s |
| Full run, including load | 21.382 s | 35.874 s |
| Peak sampled RSS of the Python process | 2.412 GB | 5.675 GB |
| Generated tokens | 180 | 180 |

Both produced identical artifacts. A 15 s sample does not establish equal quality
or performance on full consultations, and the defaults were not changed; see the
[hardware selection proposal](hardware_selection.md). The Transformers backend
supports Torch operator profiling with `--profile`.

### SGLang Omni endpoint

`moss_bonsai_sglang` sends audio to an existing SGLang Omni service through the
shared `openai_transcription` client with `response_schema = "moss_segments"`. Set
the base URL with `SCRIBNIC_MOSS_BASE_URL` and, if needed, the key with
`SCRIBNIC_MOSS_API_KEY`. `setup` downloads nothing for this route; the service is a
separate installation that Scribnic neither owns nor measures.

The client requests `verbose_json`, keeps timestamps, and reads labels such as
`[S01]` from each segment prefix; missing identities stay unknown. `prompt`,
`language`, `temperature`, and `max_new_tokens` are request settings; no Whisper or
NVIDIA parameters are sent. MOSS hotwords are vocabulary hints inside the prompt,
not a post-correction. A custom `prompt` replaces the default instructions, so it
must keep the request for timestamps and speaker labels.
[OpenMOSS prompt recipes](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/main/examples/prompts.md).

OpenMOSS recommends CUDA 13 for SGLang Omni, and the SGLang cookbook also covers
Intel GPUs. Its Apple Silicon install covers other models, not MOSS; AMD and Mac
are unvalidated. The SGLang Omni
[profiler](https://github.com/sgl-project/sglang-omni/blob/f073b1c6e3e6a5143f16ced1b9bd5d0c24ef9d76/docs/developer_reference/profiler.md)
writes its files on the server host; Scribnic does not control those captures, and
the transcription route in that revision does not propagate `traceparent`.
[Cookbook](https://github.com/sgl-project/sglang-omni/blob/f073b1c6e3e6a5143f16ced1b9bd5d0c24ef9d76/docs/cookbook/moss_transcribe_diarize.md).

Keeping the service loaded does not make MOSS a streaming-input ASR: progressive
text output and incremental audio input are different capabilities. Splitting MOSS
input externally would require resolving speaker consistency across segments.

## MOSS-Audio-4B-Thinking

`moss_audio_note` runs one `Audio → Note` step with MOSS-Audio's own encoder and a
Qwen3 decoder; it does not use MOSS-Transcribe-Diarize or Bonsai. The planner runs
the direct step and does not invent an intermediate transcript. The Spanish SOAP
template, sampling, and output budget belong to the workflow.
[Model card](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Thinking).

```bash
UV_PROJECT_ENVIRONMENT=.venv-moss-audio uv run --extra moss-audio \
  scribnic run consultation.wav --workflow moss_audio_note --profile
```

The extra pins Torch/torchaudio 2.9.1 and Transformers 4.57.1, which conflicts with
`[moss]` (Transformers 5); use a separate environment. The resource manager
downloads the official source by commit and SHA256 and the weights by revision.

The checkpoint declares about 9.73 GiB of weights before activations and KV cache.
Its published context is 40,960 positions and the encoder emits 12.5 positions per
second, so one hour of audio exceeds the context before prompt and answer.
[Weight index](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Thinking/blob/5163125c56c9d66eb3887221ef28e4ee506d59ef/model.safetensors.index.json),
[configuration](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-4B-Thinking/blob/5163125c56c9d66eb3887221ef28e4ee506d59ef/config.json).

The backend selects CUDA, MPS, or CPU; set `device` and `dtype` in
`[steps.note.runtime_settings]` to override. Automatic compute is BF16 on CUDA, FP16
on MPS, and FP32 on CPU, recorded separately from the weight format. Before loading
on CPU/MPS it checks that the weights alone fit in available memory, which is only a
lower bound. Profiles cover frontend, encoder, adapters, prefill, and decode;
`profile_gpu = true` additionally requests CUDA activities or Metal signposts.

The backend separates thinking from the final text and rejects incomplete output or
context budgets that would require truncating audio. It does not stream.

Validation so far: fake-based tests of the public boundary, and inference with a
small randomly initialized instance of the upstream code to check formats,
generation, and trace export. Parameter names and counts of the full architecture
on the meta device match the pinned index. The full checkpoint was not run on the
11 GiB WSL machine; MPS and CUDA are not validated.
