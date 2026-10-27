# Working on Scribnic

Keep the core easy to read, navigate, test, and reuse from Python, the CLI,
evaluation tools, and a future application server. Apply these rules to new and
modified code; avoid unrelated formatting changes. These are refactor guidelines,
not a claim that the current implementation already follows them.

## Reference guides and engineering review

- Use the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
  for Python language and style decisions, and Google's
  [code review guidance](https://google.github.io/eng-practices/review/reviewer/looking-for.html)
  for design, complexity, concurrency, tests, and documentation. The explicit
  Scribnic rules below take precedence, including Ruff, 88 columns, and documenting
  private functions.
- Review responsibilities and resource ownership as well as formatting. Keep
  abstractions tied to a concrete current need. A shorter file count or a passing
  linter does not establish a simpler or correct design.
- For concurrent or streaming code, review buffer growth, backpressure, normal
  completion, cancellation, worker shutdown, and failure propagation explicitly.
  Silence, end of an input source, and cancellation are separate events.

## Readable Python

- Follow PEP 8 with four-space indentation and an 88-character formatting target.
  Use Ruff for formatting, import sorting, and linting when adding development
  tooling. Development tools do not belong in client runtime dependencies.
- Give each module one clear responsibility and each function one coherent task.
  Prefer concrete names that explain the domain and include units where relevant.
- Aim for functions below 50 body lines, excluding blank lines, comments, and
  docstrings. Crossing that threshold is a reason to review the responsibility,
  not to compress statements or extract meaningless helpers. Keep a longer
  function when its cohesive sequence is easier to follow, and explain why.
- Put executable statements on separate lines. Avoid semicolon-separated
  statements and inline bodies for `if`, `for`, `with`, and function definitions.
  Protocol declarations and typing stubs may use conventional ellipsis bodies.
- Use comprehensions for simple mappings or filters. Use ordinary loops when
  there are nested iterations, several conditions, side effects, or error handling.
- Avoid nested conditional expressions and long expressions with several jobs.
  Introduce named intermediate values when they clarify the calculation.
- Prefer guard clauses and early returns to deep nesting. Ordinary `else` and
  `elif` branches are fine when they make the alternatives clearer.
- Keep whitespace between logical blocks. Let the formatter handle wrapping;
  never squeeze code to satisfy a line-count guideline.
- Add type annotations to function parameters, return values, and shared data.
  Avoid `Any` when a concrete type or a small Protocol describes the contract.
- Document behavior, assumptions, units, and non-obvious constraints using the
  function documentation rules below. Comments should explain decisions rather
  than repeat the adjacent statement.
- Catch specific exceptions at the boundary that can handle them. Preserve the
  original cause and provide actionable context. Avoid blanket exception handling
  and silent fallbacks.
- Use context managers or `try`/`finally` for resource ownership and cleanup.

## Function signatures and documentation

- Give every named function and method a docstring, including private helpers
  and constructors. A short, obvious function needs only a one-line summary.
  Apply this when adding or modifying a function; do not launch an unrelated
  documentation sweep.
- Use Google-style docstrings: start with what the function does, then add
  `Args`, `Returns`, `Yields`, and `Raises` sections where they clarify its
  contract. Explain parameter meaning, units, accepted values, and important
  defaults; keep types in annotations rather than repeating them in prose.
- Document observable side effects and ownership: whether a call downloads,
  loads or retains a model, starts a process, writes files, or closes a resource.
  Describe errors callers need to handle, not every incidental library exception.
- Put the main input first. Make optional behavior switches and tuning settings
  keyword-only with `*`, especially booleans and numeric values with similar types.
  Use named arguments at call sites when their meaning would otherwise be unclear.
- Keep signatures focused on the operation. Group related settings in an existing
  typed configuration object when that improves clarity. Avoid both long lists of
  unrelated options and catch-all configuration objects passed through every layer.
- Use explicit defaults and avoid mutable default arguments. Distinguish `None`
  from valid false or zero values. Prefer names with units such as `timeout_seconds`.
- Let Ruff wrap long signatures and calls with one argument per line and a trailing
  comma. Use modern annotations such as `list[str]` and `str | None`.

For example:

```python
def real_time_factor(
    processing_seconds: float,
    *,
    audio_seconds: float,
) -> float:
    """Calculate processing time per second of source audio.

    Args:
        processing_seconds: Elapsed processing time, excluding model preparation.
        audio_seconds: Positive duration of the original recording.

    Returns:
        The time ratio; values below one mean processing faster than real time.

    Raises:
        ValueError: If the recording duration is not positive.
    """
    if audio_seconds <= 0:
        raise ValueError("Audio duration must be positive.")
    return processing_seconds / audio_seconds
```

## Logical organization

- Give modules a short docstring explaining their responsibility. Group related
  constants, types, and operations so readers can find the public entry points and
  follow their helpers without jumping between unrelated concerns.
- Separate cohesive phases with blank lines. A short comment such as
  `# Prepare the input` can name a phase when it helps readers follow a longer
  sequence. Avoid decorative banners and comments that narrate every statement.
- Group configuration fields by purpose and document units or constraints near
  the fields. Keep CLI parsing, model preparation, inference, and persistence in
  their respective layers even when a script presents them as consecutive phases.
- A phase that owns separate state, error handling, or reusable behavior is a
  candidate for a helper. A section comment does not make a large function cohesive.
- Pass dependencies explicitly or keep them on the owning session. Avoid mutable
  module globals for models, clients, inference contexts, and per-run state.

## Functions, classes, and module boundaries

- Give every new or modified class a docstring stating its responsibility,
  owned resources, and lifetime. Document behavior rather than repeating names.
- A backend adapts an inference protocol to Scribnic artifacts. A runtime manager
  prepares/starts/checks/stops an inference service; it does not format prompts or
  parse clinical artifacts. Model presets identify weights and their precision.
- Share protocol clients where contracts match. Keep provider-specific parameters,
  response conversion, and streaming protocols explicit; HTTP compatibility does
  not establish diarization, timestamp, or streaming capabilities.
- Client installation includes only dependencies and assets of the selected
  workflow. Additional provider modules must remain importable without loading
  their optional libraries. External endpoints are not owned local processes.
- Identify model variants with their actual weight format/quantization, record
  effective compute dtype separately, and keep runtime distribution/version pins.

- Use dataclasses for configuration and artifacts. Prefer immutable artifacts;
  create explicit copies for transformations.
- Use classes for stateful resources such as loaded models, runtime sessions,
  and clients. Use functions for transformations and calculations.
- Use small Protocols where implementations need a shared interface. Introduce
  inheritance, factories, and helper layers only when they simplify a concrete use.
- Keep CLI responsibilities to argument parsing, invoking the public API, and
  presenting results or errors. Workflow logic must work without the CLI.
- Keep model-specific loading, preprocessing, inference, and response conversion
  in the corresponding backend. Keep generic artifact handling in the core.
- Keep persistence, report rendering, and observability distinct from inference.
  Keep dataset loading, perturbations, judges, and evaluation metrics outside the
  Scribnic inference core.

## Workflow and artifact contracts

- Require a selected workflow before inference. Default `run` output is a note.
  Workflow discovery must be deterministic: use an explicit file, a declared
  default, or the sole workflow in the requested directory. Report ambiguity.
- Declare each step's accepted input kinds and produced output kinds. Resolve
  dependencies only among configured steps and supplied artifacts. Validate
  missing producers, ambiguous choices, and cycles before model preparation.
- Support note generation from supplied plain or diarized transcripts without
  running audio recognition. Note generation uses the supplied artifact's content.
- Preserve the distinction between plain and diarized transcripts. Speaker IDs
  describe speaker attribution; clinical roles are separate optional information.
  Make projections and information removal explicit, preserving unknown values.
- Keep artifacts as passive data that can be saved, loaded, and reused as input.
  Do not make artifacts discover or execute their own producers.
- Keep intermediate artifacts scoped to one execution. A reusable session may
  retain loaded models; it must not mix artifacts between independent requests.
- Keep model, runtime, prompt/template, and generation settings in configuration.
  Record the effective configuration and provenance in execution results.

## Dependencies, runtimes, and platforms

- Keep client dependencies small and scoped to the configured capabilities.
  Import optional model libraries inside the backend that uses them. Importing
  Scribnic or generating a note over HTTP must not require importing Torch.
- Prepare configured resources from a versioned cache and download missing
  assets as configured. Instantiate models in memory only when needed.
- Use llama-server through the existing HTTP approach; do not add Python
  llama.cpp bindings to the client. Manage only processes owned by Scribnic.
- Centralize platform-specific binary selection, cache paths, and process
  lifecycle handling. Use `pathlib` and argument lists with `subprocess`; avoid
  platform shell scripts and shell-constructed commands in the core.
- Keep platform-specific accelerator dependencies out of the universal base
  installation. State which model/runtime/platform combinations were validated.
- Use OpenTelemetry APIs for execution spans and metrics. The calling application
  configures the SDK and exporters. Keep detailed runtime profilers optional and
  associate their output artifacts with the execution.

## Verification and repository hygiene

- Run `uv run ruff check src tests` and `uv run ruff format --check src tests`.
  Ruff checks imports, basic errors, and Google-style docstring formatting.
  Its formatter handles layout; do not add competing formatters.
- Run `uv run ruff check src/scribnic/telemetry --extend-select D,ANN` for
  docstring presence and annotated signatures in the migrated telemetry package.
  Expand that coverage as other modules are brought into compliance; do not
  suppress inherited violations globally or add meaningless annotations to pass.
  These checks do not validate type compatibility, documentation accuracy,
  function responsibility, or architecture; those still need review.
- Test meaningful behavior and public boundaries with small fakes. For the
  refactor, cover default audio-to-note execution, early stopping, transcript input,
  artifact isolation, and clear errors for unsupported workflows.
- Keep live model checks opt-in. Do not download weights, require a GPU, or contact
  inference services during ordinary unit tests.
- Run checks appropriate to the change. Documentation-only edits need no model
  runs. Do not repeat expensive checks after they pass without a concrete reason.
- Preserve the user's notes, including `TODO` and `ARCHITECTURE_EVALUATION.md`.
  Generated runs, model caches, and local recordings are not source code.
