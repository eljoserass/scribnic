# Text note evaluation: LFM3 vs Bonsai 4B

Local evaluation from 2026-10-28. It compares note generation only, without audio
recognition: one case and one run per profile, reviewed qualitatively against the
script. It is not a validated clinical scale.

## Input and method

The source is a scenario JSON with 33 turns and 5,154 characters of conversation. A
temporary script extracted the turns into a plain `Transcript`, without the
scenario summary, context, voices, or estimated durations. That same artifact went
to three text workflows, each in its own session with one `session.run` call;
`session.start(source_kind="transcript")` measured loading separately. Weights were
downloaded and verified beforehand, and models ran sequentially on WSL/Linux x86_64
CPU with six threads.

Settings: `soap_es` template, seed 42, context 8192, each author's sampling
profile (see [note models](note_models.md#comparison-conditions)). The experimental
limit was 2048 tokens with a 250-word request; every answer finished under 768
tokens, the preset limit. Neither input nor output was truncated.

## Results

| Model | Generation incl. prefill | Service load | Peak client + server RSS | Approx. words |
| --- | ---: | ---: | ---: | ---: |
| Bonsai 2B PQ2_0, control | 9.41 s | 2.22 s | 1.46 GB | 120 |
| Bonsai 4B PQ2_0 | 26.91 s | 2.37 s | 3.03 GB | 184 |
| LFM3-1.7B Q8_0 | 30.41 s | 0.63 s | 2.20 GB | 242 |

RSS is a sum sampled every 250 ms in the same pass and may count shared pages more
than once. Times are single runs without downloads, with no prompt tokens reused
from an earlier generation. Formats, models, and runtimes differ, so this does not
isolate the effect of quantization.

Provisional preference: **Bonsai 4B for this case**. It synthesizes and organizes
better than the 2B control and keeps the warning not to stop treatment abruptly.
LFM3 covers more details and used less observed RSS than Bonsai 4B, but its wording
and some instructions are less precise. No note is fully faithful to the script.

| Fact in the script | Bonsai 2B | Bonsai 4B | LFM3 |
| --- | --- | --- | --- |
| Propranolol 40 mg every 12 hours | Kept | Kept | Kept |
| Measured cholesterol 240 mg/dL | Kept | Kept | Kept |
| Cholesterol target below 200 | Omitted | Omitted | Kept |
| Review in one month | Kept | Kept | Kept |
| Blood test within 15 days | In an imprecise closing line | Kept | Kept |
| Do not stop the medication abruptly | Omitted | Kept | Omitted |
| Consider cholesterol drug if no improvement in three months | Omitted | Omitted | Kept |
| Coffee: two cups a day for a week, then one | Partial | Distorted | Kept |
| Stop immediately if symptoms appear during exercise | Omitted | Omitted | Distorted |

Notable errors:

- Bonsai 2B adds a "favorable response to drug treatment" when treatment is only
  starting, and introduces an unestablished metabolic cause.
- Bonsai 4B writes "from 3–4 to 2 cups/week", changing the coffee plan's frequency,
  and presents coffee and tobacco as triggers with more certainty than the script.
- LFM3 merges the general advice to avoid extreme intensity with what to do on
  palpitations, dizziness, or pain, losing "stop immediately". It omits gradual
  withdrawal and contains "Constantes: EPCOTEGRAMO" in Objective.

## LFM3 reasoning format

The first profile, `enable_thinking=false` with `reasoning-budget=0`, produced an
orphan `</think>` at the start of `message.content` with llama.cpp b11946, and the
existing guard rejected those runs. Changing only `reasoning-format` to `deepseek`
did not help.

The validated setting leaves the protocol prefix open with `thinking=true` and
keeps `reasoning-budget=0`: the server forces an immediate close with no free
reasoning tokens, and the final answer arrives clean. Tags are not stripped and the
client guard is not weakened. All three LFM3 text workflows use this setting; real
inference was checked with Q8_0. The behavior comes from the pinned
[parser](https://github.com/ggml-org/llama.cpp/blob/b11946/common/parsers/qwen3-coder.cpp)
and [reasoning budget](https://github.com/ggml-org/llama.cpp/blob/b11946/common/reasoning-budget.cpp);
sampling recommendations were not the cause.

## Reproduction

Generated reports, inputs, and the temporary script are local artifacts, not
source. With an exported plain transcript, the CLI reproduces each run:

```bash
uv run scribnic run plain.json --workflow bonsai_4b_note --output-dir runs/bonsai4
uv run scribnic run plain.json --workflow lfm3_17b_q8_0_note --output-dir runs/lfm3-text
```
