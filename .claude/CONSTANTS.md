# Project Constants

Shared configuration referenced by `/start-work` and `/ship`.

```
OWNER:         dariero
PROJECT_ID:    PVT_kwHODR8J4s4BNe_Y
PROJECT_NUM:   2
STATUS_FIELD:  PVTSSF_lAHODR8J4s4BNe_Yzg8dwP8

Board statuses:
  Todo:   98236657
  Doing:  47fc9ee4
  Done:   caff0873
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

## Commit Format

`[TYPE #issue] Description`

## Quality Gates

```bash
hatch run lint && hatch run typecheck && hatch run test
```
