# NVIDIA NeMo

Backend `openai_transcription` with runtime preset `nemo_cpp`. Scribnic uses the
official NeMo-Speech.cpp v0.2.0 release and two GGUF files pinned by Hugging Face
revision and SHA256: `nemotron-3.5-asr-streaming-0.6b.q8_0.gguf` and
`Nemotron-3-Diarization.q8_0.gguf`. The ASR is multilingual and selects `es-ES`.
The diarizer is Sortformer V3 with up to eight speakers. Q8_0 is the weight format;
it does not mean every tensor or operation runs in 8-bit integers.

The same runtime also serves Parakeet TDT 0.6B v3; see
[speech presets](asr_presets.md).

## Setup and lifecycle

```bash
uv run scribnic setup --workflow nvidia_cpp_offline
uv run scribnic run consultation.wav --workflow nvidia_cpp_offline
uv run --extra streaming scribnic run consultation.wav --workflow nvidia_cpp_streaming
```

`setup` downloads only the configured assets and verifies hashes; it needs no
NeMo/PyTorch and no manual path to the executable. The runtime starts
`nemo-speech serve` on loopback, loads ASR and diarizer once, and waits for
`/ready`. The manager is `runtimes/nemo_cpp.py`; the client and response conversion
are in `backends/openai_transcription.py` and `backends/transcription_formats.py`.

A session keeps the models between requests; each file gets its own recognition
state and labels. Processes are closed on exit, even if startup is interrupted.
For evaluation, call `session.start()` before `session.run(...)`, store its load
measurements separately, and omit `--profile` when measuring latency.

## Input modes

- `nvidia_cpp_offline` uploads the WAV to `/v1/audio/transcriptions`, requests
  `verbose_json` with `diarization=true`, and keeps words with timestamps and
  speakers.
- `nvidia_cpp_streaming` opens a new connection to the protocol-specific
  `/v1/audio/transcriptions/realtime`, sends PCM in 200 ms chunks, and finishes with
  `input_audio_buffer.commit`. It receives concurrently to avoid blocking on partial
  responses. There is no playback pacing unless `--realtime` is given.
  `session.run_stream(...)` also accepts PCM from other sources through the
  [common API](streaming.md).

Both modes use `OpenAITranscription`. `backends/nemo_streaming.py` is a small client
for the WebSocket protocol, with no model loading or step planning. HTTP
compatibility does not imply Realtime compatibility. OpenAI's documentation uses
"streaming" for two different contracts:

| Contract | Input and output | Scribnic support for NVIDIA |
| --- | --- | --- |
| Transcriptions HTTP, `stream=true` | Upload a recording; text arrives as SSE events | Not wired; offline mode asks for one final response |
| OpenAI Realtime transcription | Incremental audio over a persistent session | Not implemented as a client or proxy |
| NeMo-Speech.cpp WebSocket | Incremental PCM and NVIDIA's own events | Implemented by `nemo_streaming.py` inside the shared backend |

OpenAI Realtime and NeMo share some event names but differ in URL, session
configuration, and metadata. NeMo accepts binary PCM or base64 JSON; Scribnic sends
binary PCM. Changing `base_url` in the OpenAI SDK is not enough. A protocol adapter
could unify them later; the small dedicated client avoids a proxy and keeps NeMo's
timestamps and speaker attribution.

Sources: [OpenAI files and streamed output](https://developers.openai.com/api/docs/guides/speech-to-text),
[OpenAI Realtime input](https://developers.openai.com/api/docs/guides/realtime-transcription),
[NeMo v0.2.0 protocol](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/v0.2.0/docs/api.md).

The public output is `DiarizedTranscript`. The note step accepts it or its
`without_speakers()` projection. The client never infers clinical roles from
speaker numbers.

## Variants without diarization

`nvidia_cpp_asr_offline` and `nvidia_cpp_asr_streaming` use the same ASR, runtime,
and note generator without declaring or preparing the diarizer. The server starts
with ASR only; HTTP requests send `diarization=false` and the WebSocket session sends
`speaker_diarization=false`. They produce a `Transcript` with timestamps and no
speakers.

```bash
uv run scribnic run consultation.wav --workflow nvidia_cpp_asr_offline
uv run scribnic run consultation.wav --workflow nvidia_cpp_asr_offline --output transcript
```

Disabling diarization avoids running that model. To isolate only the effect of
labels on a note, use the same diarized transcript and its `without_speakers()`
projection instead.

## Diarization strategy

The C++ runtime recognizes words and attributes each to a speaker from the
diarizer's activity over time, recorded as
`speaker_attribution_strategy=word_time_alignment`. The former NeMo/Python reference
used `masked_asr=true`: acoustic features masked by speaker activity and one ASR
state per speaker. That is not source separation, and overlapping speech can keep
the same mixture. The Python reference and its `[nvidia]` extra were removed from
the core; the strategy difference and quantization mean results are not equivalent.
Its code remains in Git history.

## Profiling

Runs record load, request, RTF, configuration, hashes, PID, native response, and
logs. Python and the C++ server are sampled separately. Streaming also writes the
received event stream to JSONL as it arrives. The client timeout measures waiting
for server events, not total request duration. `--profile` explicitly declares that
C++ operators are not captured through Torch; resource sampling and spans stay on.

Model cache size, Python process memory, and native process memory are different
quantities. Keeping the server loaded avoids reloads but keeps its weights
resident. Do not add individual RSS peaks as if they were simultaneous.

## Platforms and sources

Validated on WSL/Linux x86_64 CPU. Presets also pin the official files for Linux
arm64, Windows x86_64 CPU, and macOS arm64 Metal, which have not been validated
locally. Transcription and diarization quality are still to be evaluated.

- [Official server v0.2.0](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/v0.2.0/docs/server.md)
- [API and WebSocket protocol](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/v0.2.0/docs/api.md)
- [Models and quantization](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/v0.2.0/docs/asr/models.md)
- [Per-speaker ASR reference](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/main/ASR_INTEGRATION_GUIDE.md)
