# RagaliQ Claude Commands

5 commands. That's it.

## Workflow (2)

| Command | Purpose |
|---------|---------|
| `/start-work <issue>` | Begin work (Todo → Doing) |
| `/ship` | Check, PR, merge, cleanup (Doing → Done) |

## Scaffolds (3)

| Command | Purpose |
|---------|---------|
| `/new-evaluator` | Scaffold evaluator |
| `/new-judge` | Scaffold judge |
| `/optimize-prompts` | A/B test prompts |

## The Solo Dev Workflow

```
/start-work 42    ← Pick a task
   ...code...     ← Do the work
/ship             ← Ship it
```

That's the whole process. Two commands.

## What /ship Does

1. Commits uncommitted changes
2. Runs lint, typecheck, tests
3. Creates PR with proper format
4. Quick self-review (Claude checks for issues)
5. Merges to main
6. Moves board card to Done
7. Cleans up local branch

All automatic. All in one command.

## Quality Gates

Still available manually:
```bash
hatch run lint       # ruff
hatch run format     # ruff (auto-fix)
hatch run typecheck  # mypy
hatch run test       # pytest
```

These run automatically during `/ship`.
