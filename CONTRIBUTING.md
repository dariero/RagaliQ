# Contributing to RagaliQ

Thank you for your interest in contributing! RagaliQ is an open-source LLM/RAG evaluation framework and we welcome bug reports, feature ideas, and pull requests.

## Before You Start

- Check the [GitHub Issues](https://github.com/dariero/RagaliQ/issues) to see if your bug or feature is already tracked.
- For significant changes, open an issue first to discuss the approach before writing code.

## Development Setup

**Requirements:** Python 3.14+, uv

```bash
git clone https://github.com/dariero/RagaliQ.git
cd RagaliQ
uv venv --python 3.14
uv pip sync pylock.toml
uv pip install -e . --no-deps
```

## Quality Gates

All contributions must pass these checks before merging:

```bash
.venv/bin/ruff check src/ tests/ scripts/   # style and imports
.venv/bin/mypy src/ scripts/                # strict mode
.venv/bin/pytest                            # with coverage
.venv/bin/coverage run --source=src/ragaliq -m pytest tests/unit/ tests/meta/test_meta_metrics.py -o addopts="" -q && .venv/bin/coverage report   # CI's test form
```

Run them together:

```bash
.venv/bin/ruff check src/ tests/ scripts/ && .venv/bin/mypy src/ scripts/ && .venv/bin/pytest
.venv/bin/coverage run --source=src/ragaliq -m pytest tests/unit/ tests/meta/test_meta_metrics.py -o addopts="" -q && .venv/bin/coverage report
```

> **Dependencies are locked** in `pylock.toml` (PEP 751, cross-platform). After
> changing dependencies in `pyproject.toml`, regenerate it with:
> `uv pip compile pyproject.toml --extra dev --universal --python-version 3.14 --generate-hashes --upgrade-package <name> -o pylock.toml`.
> **One package per regeneration.** `--upgrade` (no `-package`) pulls *every*
> floor to latest, which silently drags unrelated bumps into a slice — a
> tooling PR would ship an `anthropic` bump with it. Always diff the lock
> before committing and confirm only the named package moved:
> `git diff pylock.toml | grep -E '^[+-]name|^[+-]version'`.
> A platform-specific lock breaks Linux CI, so the lock must be `--universal`.
> CI installs strictly from this lock via uv.

**2026-10-11 override:** the owner's instruction for the `deps-latest-2026-10-11` run, approved at its Checkpoint 1, overrides the one-package-per-regeneration rule above for that branch only: `pylock.toml` was regenerated with a bare `--upgrade` (every package to latest, majors included) and `[project.dependencies]` floors were left unchanged.

**Amended 2026-10-12:** `faec8d7` raised the `anthropic` floor in `[project.dependencies]` from `>=0.120` to `>=1.13`, and its mirrors-mypy copy in `.pre-commit-config.yaml` with it, because CI tests only the locked 1.13 and 0.x support was untested. The other `[project.dependencies]` floors are unchanged.

**Amended 2026-10-24:** the `anthropic` floor in `[project.dependencies]` rose from `>=1.13` to `>=2.3`, and its mirrors-mypy copy in `.pre-commit-config.yaml` with it, when `pylock.toml` moved to anthropic 2.3.0: CI tests only the locked 2.3, and 1.x sends a different API version (`2023-06-01`, not `2026-10-21`). See ADR-149's amendment of the same date.

## Keeping pre-commit revs current

Dependabot's `pre-commit` ecosystem can update the `rev:` values in
`.pre-commit-config.yaml`, but this repo does not enable it: a PR that moves
ruff's `rev:` to a new minor on its own fails `check_ruff_atom` (below). Run
**`.venv/bin/pre-commit autoupdate` quarterly** and review the diff before
staging; `git checkout -- .pre-commit-config.yaml` reverts it.

> **ruff lives in three places and they move together:** the `dev` extra floor
> in `pyproject.toml`, the `rev:` in `.pre-commit-config.yaml`, and the pin in
> `pylock.toml`. `autoupdate` moves only the second. Left split, the pre-commit
> hook and `.venv/bin/ruff` format the same tree differently.
>
> A split does not even need `autoupdate`: the floor is a `>=`, so a lock
> regenerated with `--upgrade-package ruff` can pull a newer ruff than the
> pinned hook. Run `.venv/bin/python scripts/verify_release.py --consistency-only`
> after any of the three changes — `check_ruff_atom` fails if they disagree.

The `additional_dependencies` under `mirrors-mypy` are a further copy of the
runtime floors. If they lag, mypy types against a different version than the
code imports.

## Code Standards

- **Type hints** are required on all public functions and methods.
- **Docstrings** must use [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
- **Evaluator pattern**: each new metric MUST be a separate `Evaluator` subclass with an `evaluate()` method. See `src/ragaliq/evaluators/` for examples.
- **Async-first**: all LLM calls must be `async def`.
- **Pydantic models** for all data structures (no raw dicts).

## Submitting a Pull Request

1. Fork the repository and create a branch: `feat/<short-description>`.
2. Make your changes and ensure all quality gates pass.
3. Write or update tests in `tests/` mirroring the `src/` structure.
4. Open a PR against `main` with a clear description of what changed and why.

## Reporting Bugs

Open a [GitHub Issue](https://github.com/dariero/RagaliQ/issues/new) with:
- Python version and OS
- RagaliQ version (`pip show ragaliq`)
- A minimal reproducible example
- The full error traceback

## Security Vulnerabilities

Please do **not** open a public issue for security vulnerabilities. See [SECURITY.md](SECURITY.md) for the responsible disclosure process.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
