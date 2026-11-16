# Incremental audio

A speech step opts in with `mode = "streaming"`. The backend picks the strategy from
the declared protocol; the runtime is the same resident server as offline.
`WorkflowSession` orchestrates and saves artifacts; it does not implement windows
or WebSockets.

| Workflow | Processing | Service input |
| --- | --- | --- |
| `nvidia_cpp_streaming` | Cache-aware ASR + diarization | PCM over the native WebSocket |
| `nvidia_cpp_asr_streaming` | Same ASR without the diarizer | PCM over the native WebSocket |
| `whisper_bonsai_streaming` | Overlapping windows, encoder recomputed | Small WAV files over HTTP |
| `lfm3_audio_note` | Causal encoder and incremental prefill; direct note | PCM over the llama-server WebSocket |

Whisper and NVIDIA produce one final transcript and then one note. Use
`--output transcript` with Whisper and NVIDIA ASR, or `--output diarized_transcript`
with diarized NVIDIA. LFM3 produces a note directly, with no transcript or
transcript updates; see [LFM3-Audio](lfm3.md).

```bash
# Consume the file as fast as inference allows.
uv run scribnic run consultation.wav --workflow whisper_bonsai_streaming
uv run --extra streaming scribnic run consultation.wav --workflow nvidia_cpp_streaming

# Simulate audio arriving during a consultation, through the same pipeline.
uv run scribnic run consultation.wav --workflow whisper_bonsai_streaming --realtime
```

## Sources and Python API

`run_stream` takes an iterable of `bytes`: PCM16 little-endian, mono, 16 kHz, in
whole frames. The source can replay a WAV, read a capture queue, or receive audio
from another application. Scribnic installs no microphone library; the caller owns
the device and must be able to stop a blocking source.

```python
from pathlib import Path
from scribnic import open_workflow, wav_chunks

with open_workflow("whisper_bonsai_streaming") as session:
    session.start(output="transcript")  # Load speech before opening the device.
    result = session.run_stream(
        wav_chunks(Path("consultation.wav"), realtime=True),
        output_dir="runs/consultation-incremental",
    )
    print(result.note.text)
```

Use `session.start(output="diarized_transcript")` for diarized NVIDIA. Preloading
only the speech step lets the note generator load later.

`on_update` is an optional callback receiving
`TranscriptUpdate(text, audio_seconds, final)`. Each update replaces the previous
text rather than appending to it. Whisper may revise the overlap. NVIDIA publishes
finalized blocks and keeps native deltas in the log, because its contract does not
identify every revision unambiguously. `final=True` means the input ended.

`cancel` takes a `threading.Event`, checked between input blocks. It cannot unblock
an external device; that belongs to the capture code. Reading applies consumer
backpressure and never drops audio silently. A hardware callback must manage its
own queue or write to disk; it cannot assume Python always keeps pace.

The run saves `runtime/recording-*.wav`, records it as an `AudioFile` with duration
and hash, and reuses the usual storage, reports, CPU/RSS sampling, and spans. Audio
already received is kept if the run is interrupted. Without `output_dir`, the
recording goes to the diagnostics cache.

## Whisper windows

The preset uses `window_seconds = 10` and `overlap_seconds = 2`, so each window
advances 8 s; override them in `[steps.speech.model_settings]`. The client requests
per-word timestamps, shifts them to the original audio, and reconciles the overlap
by text. Without a compatible anchor it uses the temporal midpoint and clips the
intervals that cross it, keeping word order. The policy is recorded in the result
and native timestamps stay in the JSONL log. The last window with new audio is
always processed. The final transcript groups words into sentences at pauses longer
than `sentence_gap_seconds` (1 s by default), as with NeMo.

Boundary errors and repetitions still need evaluation. The result is not
guaranteed to match offline inference, nor to use less compute: the model stays
loaded but the encoder reprocesses every window. The official `whisper-stream`
example also uses windows; its SDL capture is not needed for this PCM source and
does not turn the HTTP server into OpenAI Realtime.

## NVIDIA and network contracts

NeMo receives 200 ms chunks and keeps the acoustic cache and diarizer state for the
connection; chunk size is not the encoder's internal context. The client sends one
`commit` at the end, which in v0.2.0 finishes and resets recognition and speaker
state. Internal endpointing can produce intermediate finals without opening a new
request. The native protocol differs from OpenAI Realtime in session, events, IDs,
and diarization; see [NVIDIA NeMo](nvidia.md#input-modes).

## Memory and measurement

WAV validation reads in blocks and native events are written to JSONL rather than
kept in a list. Whisper retains one window of PCM; transcript text and the
diarizer's internal state can still grow with the consultation. Weights and compute
buffers stay resident, so total RAM is not constant. Streaming transcription does
not shrink the note generator's context: for long recordings, request only the
transcript and evaluate the note separately.

Request time includes waiting for the source. A `--realtime` replay and an
unpaced one measure different things; label them separately when comparing RTF.
The CLI records those flags in `telemetry/invocation.json`; a Python application
must record how it paces its source. Window/event JSONL files and final responses
are in `runtime/`. Run latency benchmarks without `--profile`.

Checked on WSL/Linux CPU with the same 24 s WAV for diarized NVIDIA and Whisper,
both followed by Bonsai. That shows execution and artifact generation; quality,
physical capture, and hours-long stability need their own validation.
