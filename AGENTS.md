# Working on Scribnic

Keep the core easy to read, navigate, test, and reuse from Python, the CLI,
evaluation tools, and a future application server. Apply these rules to new and
modified code; avoid unrelated formatting changes. These are refactor guidelines,
not a claim that the current implementation already follows them.

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
- Add type annotations to public interfaces and shared data. Avoid `Any` when a
  concrete type or a small Protocol describes the contract.
- Document public behavior, assumptions, units, and non-obvious constraints.
  Comments should explain decisions rather than repeat the adjacent statement.
- Catch specific exceptions at the boundary that can handle them. Preserve the
  original cause and provide actionable context. Avoid blanket exception handling
  and silent fallbacks.
- Use context managers or `try`/`finally` for resource ownership and cleanup.

## Functions, classes, and module boundaries

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

- Test meaningful behavior and public boundaries with small fakes. For the
  refactor, cover default audio-to-note execution, early stopping, transcript input,
  artifact isolation, and clear errors for unsupported workflows.
- Keep live model checks opt-in. Do not download weights, require a GPU, or contact
  inference services during ordinary unit tests.
- Run checks appropriate to the change. Documentation-only edits need no model
  runs. Do not repeat expensive checks after they pass without a concrete reason.
- Preserve the user's notes, including `TODO` and `ARCHITECTURE_EVALUATION.md`.
  Generated runs, model caches, and local recordings are not source code.
