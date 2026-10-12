# Project Constants

Shared configuration referenced by `/start-work` and `/ship`.

```
OWNER:          dariero
PROJECT_ID:     PVT_kwHODR8J4s4BNe_Y
PROJECT_NUM:    2
STATUS_FIELD:   PVTSSF_lAHODR8J4s4BNe_Yzg8dwP8
PRIORITY_FIELD: PVTSSF_lAHODR8J4s4BNe_Yzg8dwQc
SIZE_FIELD:     PVTSSF_lAHODR8J4s4BNe_Yzg8dwQg

Board statuses:
  Todo:   98236657
  Doing:  47fc9ee4
  Done:   caff0873

Priority options:
  Critical: 79628723
  High:     0a877460
  Medium:   da944a9c
  Low:      56c1c445

Size options:
  XS: 6c6483d2
  S:  f784b110
  M:  7515a9f1
  L:  817d0097
  XL: db339eb2
```

**Board URL:** https://github.com/users/dariero/projects/2/views/1

## Branch Naming

Format: `<prefix>/<issue>-<description>`

| Title Prefix | Branch Prefix |
|--------------|---------------|
| `[FEAT]` | `feat/` |
| `[FIX]` | `fix/` |
| `[REFACTOR]` | `refactor/` |
| `[DOCS]` | `docs/` |
| `[CHORE]` | `chore/` |
| (none) | `feat/` |

## Commit Type Mapping

Inferred from branch prefix — used by `/ship` when building `[TYPE #N]` commit messages.

| Branch Prefix | Commit TYPE |
|---------------|-------------|
| `feat/`       | `FEAT`      |
| `fix/`        | `FIX`       |
| `refactor/`   | `REFACTOR`  |
| `docs/`       | `DOCS`      |
| `chore/`      | `CHORE`     |
| (none)        | `FEAT`      |

## Issue Type Defaults

Inferred from title prefix — used by `/start-work` to set Priority and Size on the board.

| Title prefix | Priority | Priority ID | Size | Size ID    |
|--------------|----------|-------------|------|------------|
| `[FIX]`      | Medium   | `da944a9c`  | S    | `f784b110` |
| `[FEAT]`     | Medium   | `da944a9c`  | M    | `7515a9f1` |
| `[REFACTOR]` | Low      | `56c1c445`  | M    | `7515a9f1` |
| `[DOCS]`     | Low      | `56c1c445`  | S    | `f784b110` |
| (none)       | Medium   | `da944a9c`  | M    | `7515a9f1` |

## Commit Format

`[TYPE #issue] Description`

## Quality Gates

Agents on the owner's machine do not run hatch (the 2026-10-12 override in `CLAUDE.md`). The gates run from the repo's `.venv`:

1. If `.venv/bin/python` does not exist: report it and **STOP**. Do not create the environment.
2. Sync `.venv` to this branch's `pylock.toml`. Hatch synced its environment on every run; without this step the gates test whichever lock was synced last. `--python` is the install target here, so it is always `.venv/bin/python`, never a system or Homebrew interpreter. If the command fails: report it and **STOP**; do not run the gates.
   ```bash
   uv pip sync --no-python-downloads --python .venv/bin/python pylock.toml \
     && uv pip install --no-python-downloads --python .venv/bin/python --no-deps -e .
   ```
3. Run the gates with `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and `RAGALIQ_RUN_META` unset, so no test can call a paid API. These are two commands, and both must exit 0:
   ```bash
   env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY -u RAGALIQ_RUN_META .venv/bin/ruff check src/ tests/ scripts/ \
     && env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY -u RAGALIQ_RUN_META .venv/bin/mypy src/ scripts/ \
     && env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY -u RAGALIQ_RUN_META .venv/bin/pytest
   env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY -u RAGALIQ_RUN_META .venv/bin/coverage run --source=src/ragaliq -m pytest tests/unit/ tests/meta/test_meta_metrics.py -o addopts="" -q \
     && env -u ANTHROPIC_API_KEY -u OPENAI_API_KEY -u RAGALIQ_RUN_META .venv/bin/coverage report
   ```

If lint fails: run the `format` script of `[tool.hatch.envs.default.scripts]` in `pyproject.toml` from `.venv` (`.venv/bin/ruff format src/ tests/ scripts/`, then `.venv/bin/ruff check --fix src/ tests/ scripts/`), commit the files it changed as in `/ship` step 2, then re-run the gates. If typecheck or test fails: report which failed and **STOP**.
