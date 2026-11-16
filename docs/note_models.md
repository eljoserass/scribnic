# Note models

Text note generators use backend `openai_chat` on a local llama-server. Qwen3.5 and
LFM3 text select runtime preset `llama_cpp` (upstream b11946); Bonsai selects
`llama_cpp_prism`. Each workflow accepts a plain or diarized transcript and writes a
Spanish SOAP note, so several generators can be evaluated on the same input,
including reference text, without repeating audio recognition.

## Variants

Sizes are the published weight files in decimal GB; runtime memory adds cache,
activations, and buffers. Only the selected workflow's model is prepared.

| Model | Format | Weights | Workflow |
| --- | --- | ---: | --- |
| Bonsai 2 2B | PQ2_0 | 0.52 GB | `bonsai_note` |
| Bonsai 2 4B | PQ2_0 | 1.15 GB | `bonsai_4b_note` |
| Qwen3.5-0.8B | Q4_K_M | 0.53 GB | `qwen35_08b_q4_k_m_note` |
| Qwen3.5-0.8B | Q8_0 | 0.81 GB | `qwen35_08b_q8_0_note` |
| Qwen3.5-2B | Q4_K_M | 1.28 GB | `qwen35_2b_q4_k_m_note` |
| Qwen3.5-2B | Q8_0 | 2.01 GB | `qwen35_2b_q8_0_note` |
| Qwen3.5-4B | Q4_K_M | 2.74 GB | `qwen35_4b_q4_k_m_note` |
| Qwen3.5-4B | Q8_0 | 4.48 GB | `qwen35_4b_q8_0_note` |
| LFM3-1.7B | Q4_K_M | 1.10 GB | `lfm3_17b_q4_k_m_note` |
| LFM3-1.7B | Q8_0 | 1.85 GB | `lfm3_17b_q8_0_note` |
| LFM3-1.7B QAD | Q4_0 | 1.05 GB | `lfm3_17b_qad_q4_0_note` |

QAD is a checkpoint trained to compensate for quantization, stored as Q4_0; it is
kept separate from the ordinary Q4_K_M. PQ2_0 and Q4_K_M are different weight
types too.

Sources (each preset pins repository, revision, and SHA256):

- Qwen: [0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF),
  [2B](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF), and
  [4B](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF), converted by Unsloth AI
  from Qwen's models. Qwen publishes no GGUF for these sizes.
- Bonsai: [2B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf) and
  [4B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf) from Prism ML,
  using the Prism fork's kernels.
- [LFM3-1.7B GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF) from Liquid. The
  direct audio workflow is [`lfm3_audio_note`](lfm3.md).

Qwen and Bonsai declare Apache-2.0. LFM3 uses the
[LFM Open License v1.0](https://www.liquid.ai/lfm-license), with an annual revenue
threshold for free commercial use.

## Run from text or a saved transcript

```bash
# Download and verify one variant without loading it.
uv run scribnic setup --workflow qwen35_08b_q4_k_m_note

# Generate from plain text, then reuse exactly the same text with another model.
uv run scribnic run conversation.txt \
  --workflow qwen35_08b_q4_k_m_note --output-dir runs/qwen08-q4
uv run scribnic run conversation.txt \
  --workflow lfm3_17b_qad_q4_0_note --output-dir runs/lfm3-text-qad

# Recognize audio once and reuse the transcript.
uv run scribnic run consultation.wav --workflow whisper_bonsai_vad \
  --output transcript --output-dir runs/shared-transcript
uv run scribnic run runs/shared-transcript/transcript.json \
  --workflow qwen35_08b_q4_k_m_note --output-dir runs/qwen08-from-whisper
```

`whisper_qwen35_08b_q8_0` runs Whisper small Q8 with VAD and Qwen3.5-0.8B Q8 in one
command. To use any of these generators in another pipeline, keep that pipeline's
speech steps and set the note step's model and runtime:

```toml
[steps.note]
backend = "openai_chat"
model = "Qwen3.5-0.8B-Q4_K_M"
runtime = "llama_cpp"
template = "soap_es"
source = "speech"

[steps.note.model_settings]
temperature = 0.0

[steps.note.runtime_settings]
context_size = 8192
```

For LFM3, keep the workflow's `thinking = true` together with its `server_options`
table and zero reasoning budget. The template leaves the protocol block open and
the server closes it immediately, with no free reasoning tokens; prefilling a
closed block produced an orphan `</think>` with b11946. Override tables replace the
whole table. The client still rejects thinking tags in the final content rather
than stripping them.

## Comparison conditions

The workflows share the `soap_es` template, seed 42, at most 768 output tokens, a
12,000-character input limit, and an 8,192-position context. Each family keeps its
author's sampling profile:

| Family | Profile |
| --- | --- |
| Qwen3.5 | Temperature 0 |
| Bonsai 4B | Prism's non-thinking profile: temperature 0.7, top-p 0.8, top-k 20, min-p 0, presence penalty 1.5, repetition 1.0 over 64 tokens ([source](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/SMALL-MODELS.md)) |
| LFM3 text | Temperature 0.1, top-k 50, repetition 1.1, top-p 1, min-p 0 ([source](https://huggingface.co/LiquidAI/LFM3-1.7B#inference)) |

Comparing families therefore means recording sampling settings as well as prompt,
input, and quantization; each run already saves its effective configuration.
Bonsai 4B and LFM3-1.7B Q8_0 were run on the same reference text with Bonsai 2B as
a control; see the [text note evaluation](text_note_evaluation.md). The other
variants still need real inference. When choosing, compare figures, doses,
intervals, negations, follow-up, and fact attribution alongside resources.
