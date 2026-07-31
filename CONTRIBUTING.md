# Contributing

Thanks for your interest in improving ldetect-lite.

## Setup

```bash
git clone https://github.com/adamyhe/ldetect-lite.git
cd ldetect-lite
uv sync --group dev
```

Run CLI commands and tests through `uv run` so they use the managed environment.

## Development workflow

```bash
# Run unit tests only (fast)
uv run pytest -m "not integration"

# Run integration tests (downloads a few files from BitBucket on first run, cached to tests/data/)
uv run pytest -m integration

# Run all tests
uv run pytest

# Lint and type-check (must pass before opening a PR)
uv run ruff check src tests
uv run mypy src
```

CI (`.github/workflows/tests.yml`) runs the same lint, type-check, and test steps across Python 3.11–3.14, plus a build check. `examples/` scripts are not covered by CI lint/type-check, but should still follow the style of the rest of the codebase.

## Making changes

- Keep PRs scoped to one concern — prefer several small, independently-mergeable PRs over one that mixes unrelated fixes/features.
- Add or update tests for any behavior change. New modules under `src/ldetect_lite/` should have corresponding tests under `tests/`.
- If a change affects reproduction behavior against `examples/ldetect_original` or `examples/MacDonald2022`, note the finding in `notes/findings/` (confirmed summaries) or `notes/logs/` (dated experiment notes), whichever fits.
- Update `README.md`/`AGENTS.md` if you add a CLI flag, module, or change documented behavior.

## Versioning and releases

This project follows semantic versioning. Version bumps are part of the PR that merges the corresponding change, not a separate release PR:

1. Bump the `version` field in `pyproject.toml` and run `uv lock` to update `uv.lock` in the same PR.
2. After merging, tag the merge commit `vX.Y.Z` and publish a GitHub Release from that tag — this triggers `.github/workflows/publish.yml`, which builds and publishes to PyPI via trusted publishing.

Use a patch bump for fixes/internal changes and a minor bump for new user-facing functionality. If multiple PRs are in flight for closely related work, it's fine to defer a single combined bump to whichever merges last — just say so in that PR's description.

## Reporting issues

Open an issue at https://github.com/adamyhe/ldetect-lite/issues with enough detail to reproduce: command run, input data shape (not the data itself, unless it's public), and full error output.
