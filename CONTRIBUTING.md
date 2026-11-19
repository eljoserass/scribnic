# Contributing to Scribnic

Thanks for helping. This guide covers the workflow; design, style, and
documentation rules are in [AGENTS.md](AGENTS.md) and apply to everyone, human or
agent.

## Ground rules

- Never commit or attach patient audio, transcripts, notes, model weights, or
  generated runs. Use synthetic or public recordings.
- Model presets must use weights from the model authors' or identified
  organizations; see [docs/model_sources.md](docs/model_sources.md).
- Discuss large changes in an issue before opening a pull request.

## Development setup

```bash
uv sync --locked --extra moss --extra streaming
uvx pre-commit install
```

## Checks

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run ruff check src/scribnic/telemetry --extend-select D,ANN
uv run python -m unittest discover -s tests -v
```

Unit tests use fakes. They must not download weights, require a GPU, or contact
inference services. Live checks are opt-in through `SCRIBNIC_LIVE_TESTS=1`.

## Commits and pull requests

- Write commit messages as [Conventional Commits](https://www.conventionalcommits.org/):
  `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `perf:`, `chore:`, `ci:`.
  Mark breaking changes with `!` (for example `feat!:`) and explain them in the
  body. The changelog is drafted from these messages.
- Branch from `main` with a descriptive name such as `fix/whisper-timestamps`.
- Record user-visible changes under `## [Unreleased]` in
  [CHANGELOG.md](CHANGELOG.md).
- State which model/runtime/platform combinations you validated.

Releases are described in [docs/releasing.md](docs/releasing.md).
