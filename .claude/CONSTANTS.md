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
| (none) | `feat/` |

## Commit Type Mapping

Inferred from branch prefix — used by `/ship` when building `[TYPE #N]` commit messages.

| Branch Prefix | Commit TYPE |
|---------------|-------------|
| `feat/`       | `FEAT`      |
| `fix/`        | `FIX`       |
| `refactor/`   | `REFACTOR`  |
| `docs/`       | `DOCS`      |
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

```bash
hatch run lint && hatch run typecheck && hatch run test
```
