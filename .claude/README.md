# RagaliQ Claude Commands

9 focused commands for task-driven development of RagaliQ.

## Commands

### Workflow (6)

| Command | Phase | Purpose |
|---------|-------|---------|
| `/start-work <issue>` | Init | Branch, board update, show context |
| `/commit` | Impl | `[TYPE #issue]` format commit |
| `/check` | Valid | lint + typecheck + test |
| `/pr` | Submit | PR with template |
| `/fix` | Iter | Address review feedback |
| `/complete-issue <issue>` | Done | Post-merge cleanup |

### Feature (3)

| Command | Purpose |
|---------|---------|
| `/new-evaluator` | Scaffold evaluator (async, 0-1 score, registry) |
| `/new-judge` | Scaffold judge (async, retry, token tracking) |
| `/optimize-prompts` | A/B test judge prompts |

## Typical Session

```bash
/start-work 42       # → feat/42-description branch
# implement
/commit              # → [FEAT #42] message
/check               # → lint, typecheck, test
/pr                  # → PR with template
/complete-issue 42   # → cleanup
```

## When NOT to Use Commands

Most tasks don't need a command. Just describe what you need:
- "Add a CLI command for exporting results"
- "Set up GitHub Actions for testing"
- "Refactor this to async"
- "Write documentation for the evaluators"

Commands exist for **repetitive decisions** (branch naming, commit format) and **RagaliQ-specific patterns** (evaluator/judge scaffolding), not for general coding tasks.

## Quality Standards

```bash
hatch run lint       # ruff
hatch run format     # ruff (auto-fix)
hatch run typecheck  # mypy
hatch run test       # pytest (80% coverage)
```

## See Also

- [commands.md](commands.md) - Full command reference
- [WORKFLOW.md](WORKFLOW.md) - Development workflow guide
