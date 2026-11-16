# LFM3-Audio: audio directly to note

Backend `llama_audio_note` with runtime preset `llama_cpp`. `lfm3_audio_note`
generates a Spanish SOAP note directly from audio, without an intermediate
transcript. It uses `LiquidAI/LFM3-Audio-2B-GGUF` in `Q4_K_M` (1.10 GB) with its
`mmproj` companion in `Q8_0` (338 MB), both pinned by revision and SHA256. It needs
no Torch or Transformers.

```bash
uv run --extra streaming scribnic setup --workflow lfm3_audio_note
uv run --extra streaming scribnic run consultation.wav --workflow lfm3_audio_note

# Replay the WAV at recording pace.
uv run --extra streaming scribnic run consultation.wav --workflow lfm3_audio_note --realtime
```

The output directory holds the note, effective configuration, provenance of both
checkpoints, the server log, WebSocket events as JSONL (without the audio blocks
sent), and the usual telemetry.

## Resources and configuration

The `llama_cpp` preset selects upstream `b11946`, which includes LFM3 support
([#31257](https://github.com/ggml-org/llama.cpp/pull/31257)) and realtime
conversation ([#31269](https://github.com/ggml-org/llama.cpp/pull/31269)). It uses
the existing `LlamaServer` manager; Bonsai keeps its own Prism distribution. No
llama.cpp Python bindings are installed.

The workflow binds the companion in `[steps.note.companions]`; the runtime receives
its verified path as `--mmproj`. Weights stay loaded for the Python session and
each run opens a new WebSocket to isolate requests.

Extra executable options can be passed in TOML:

```toml
[steps.note.runtime_settings.server_options]
jinja = true
context-shift = false
temp = 0.0
top-k = 50
top-p = 1.0
min-p = 0.0
```

Keys are llama-server long option names without `--`. A boolean becomes `--name`
or `--no-name`; a number or string becomes an option with a value. They are passed
as literal arguments, never through a shell. An override table replaces the
preset's whole table, so include every option you need, especially
`context-shift = false`. Model paths, host, port, alias, context size, and other
settings Scribnic manages use their own fields and cannot be overridden here.

Realtime sampling is configured on the server: this version does not accept
temperature through `session.update`. `max_tokens` belongs to the model and is sent
as `max_output_tokens`. The preset limits each request to 1,200 seconds with a
32,768 context. Changing these values requires evaluating memory and long-audio
quality.

## Incremental prefill and limits

`run(AudioFile)` and `run_stream` share one implementation. The backend sends PCM16
mono 16 kHz blocks and the server runs the encoder and prefill as they arrive. A WAV
without `--realtime` is sent as fast as the connection allows; live capture
consumes the caller's iterable.

Only the explicit end of input triggers `commit`, adds the template instruction to
the same user turn, and requests a note. The system message is LiquidAI's published
audio-understanding prompt, set in `model_settings.system_prompt`; the SOAP
template is the user request. Using the template as the system message produced
transcription and repetition in the real test, over both HTTP and WebSocket. A
pause does not end the request. `output_modalities = ["text"]` disables voice
output. No `TranscriptUpdate` is published because there is no intermediate text.

```python
from scribnic import open_workflow

with open_workflow("lfm3_audio_note") as session:
    session.start()  # Load before capture begins.
    result = session.run_stream(audio_chunks, output_dir="runs/lfm3-consultation")
    print(result.note.text)
```

The caller owns capture and must be able to stop a blocking source. `cancel` takes
a `threading.Event`, checked between blocks and while waiting for the response. The
run closes the socket on cancel or failure and keeps the WAV received so far. The
backend creates no sender threads or audio queues in Python; transport buffers
apply backpressure. The server's encoder queue and context can grow up to the
configured limit.

Incremental prefill reduces pending work when speech ends; it does not remove the
context limit or guarantee constant RAM. The budget is estimated from the
frontend's 12.5 audio positions per second plus a conservative reserve for prompt
and note. The server runs with `--no-context-shift`. Inputs over the limits are
rejected, as are empty, cancelled, or max-token responses. `b11946` reports
`completed` even when it hits the maximum, so the client also checks token usage.

Request time includes waiting for the source plus processing.
`first_text_after_response_request_seconds` measures from the final request to the
first text delta; it is not total processing time.

## Local check

Checked on 2026-10-28 on WSL/Linux x86_64, CPU, six threads, with the pinned
Q4_K_M checkpoint and Q8_0 companion:

- A 297.891 s consultation produced a note in 60.93 s total: 57.76 s request and
  1.92 s service load.
- Peak per-sample RSS sum: 2.96 GB across client and server. The sum may count
  shared pages more than once.
- First text 0.94 s after requesting the response. Chunks were sent without
  capture pacing, so this is not a live-microphone measurement.
- A 24 s clip with `--realtime` produced a note in 30.40 s total, with first text
  0.80 s after the response request.

The note was generated but mixed treatment and recommendations into Objective and
left an uninformative Plan. This run does not show clinical quality or full audio
coverage. Ten continuous minutes, GPU, macOS, and Windows have not been validated.

Sources: [model](https://huggingface.co/LiquidAI/LFM3-Audio-2B),
[GGUF](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF),
[pinned protocol](https://github.com/ggml-org/llama.cpp/blob/b11946/tools/server/README.md#realtime-conversation).
