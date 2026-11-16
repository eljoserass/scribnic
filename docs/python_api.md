# Python API and sessions

`open_workflow` returns a `WorkflowSession`. The session owns the loaded models and
the runtime processes it started; every `run` call gets its own artifacts.

```python
from scribnic import AudioFile, Transcript, open_workflow

with open_workflow("nvidia_cpp_offline") as scribe:
    startup = scribe.start()
    result = scribe.run(AudioFile("consultation.wav"), output_dir="runs/consultation")

    speech = scribe.run(AudioFile("other.wav"), output="diarized_transcript")
    ablation = scribe.run(speech.transcript.without_speakers())
    manual = scribe.run(Transcript.from_text("Paciente: tengo tos desde ayer."))
```

## Lifecycle

- `open_workflow(..., prepare=False)` skips preparing resources for steps that may
  never run. `download=False` requires the cache.
- `start()` preloads the selected steps and returns startup timings and resources,
  so an evaluator can store them separately from inference.
  `start(source_kind="transcript")` loads only what that input needs.
- `run()` starts services on demand if `start()` was not called. It validates the
  input, converts outputs, and saves results when `output_dir` is given.
- `close()` (or leaving the `with` block) stops only processes the session owns,
  never external services.

`start()` is an explicit preload; Scribnic does not send a dummy request to warm up
every operation. The CLI opens and closes one session per command, so preloading
there only moves loading to another phase. Reuse pays off in an application or
evaluator that keeps the same session open across recordings.

## Artifacts

Artifacts are passive dataclasses that can be saved and passed back as input.
`without_speakers()` returns an explicit copy that keeps text and timestamps and
drops speaker IDs and roles. Speaker IDs describe attribution; clinical roles are
separate optional information. The note model receives exactly the supplied
transcript.

To isolate the effect of speaker labels on a note, run the same diarized
transcript and its `without_speakers()` projection through the same generator.
Comparing two different speech systems measures something else.

## Residency (not implemented yet)

There is no `keep_loaded` setting and no selective unloading after a step. A
session keeps every loaded step until `close()`. Files prepared by `setup` stay in
the disk cache, which is not the same as models loaded in RAM.

A future policy should separate *when to load* from *how long to keep*: for
example, keep speech resident and release the note generator after each note, or
the reverse. Releasing after a streaming step means closing the whole request, not
reloading between chunks. `WorkflowSession` would apply the common policy, and each
backend/runtime would release resources through what its service supports; a
supported unload command and stopping an owned process are different mechanisms.
Managing models on external endpoints needs an explicit ownership contract.

The residency unit must match what the engine actually groups. NeMo loads ASR and
diarizer in one process with no operation to unload only one of them, so keeping
only diarization resident would need granular server support or separate services.
Within one CLI command, releasing speech before loading the note generator could
lower peak RAM.

## Parallel evaluation

Calls on one session are serialized, even from several threads. For parallel
evaluation, give each worker its own Python process and resident session. Sessions
share the disk cache, but each starts its own servers on demand, with distinct PIDs
and ports unless a port is fixed. Inference state and artifacts stay separate; RAM
grows with the number of replicas. Each sample needs its own output directory.

Workers configured with the same external URL share that server: creating clients
does not replicate the model. The harness must limit in-flight requests and tell
per-request latency apart from total throughput. Owned servers are not tuned for
high concurrency: Bonsai starts with one slot and NVIDIA with batching disabled.

Run `setup` for each workflow first, then open workers with `download=False`.
Runtime extraction and builds do not yet lock across processes. Scribnic has no
worker pool or automatic CPU/RAM partitioning; the harness sets workers and threads
per runtime. Record whole-machine resources for these experiments, because
individual run reports do not represent total concurrent use.
