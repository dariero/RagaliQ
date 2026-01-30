# RagaliQ Development Commands

9 focused commands for task-driven development.

## Workflow Commands (6)

| Command | Purpose |
|---------|---------|
| `/start-work <issue>` | Initialize: branch, board update, context |
| `/commit` | Standardized commit with issue linking |
| `/check` | Pre-push quality gates (lint, typecheck, test) |
| `/pr` | Create PR with template and issue link |
| `/fix` | Address review feedback |
| `/complete-issue <issue>` | Post-merge cleanup |

## Feature Commands (3)

| Command | Purpose |
|---------|---------|
| `/new-evaluator` | Scaffold evaluator with required pattern |
| `/new-judge` | Scaffold judge with async/retry/token tracking |
| `/optimize-prompts` | A/B test and improve judge prompts |

---

## Typical Session

```bash
/start-work 42              # Branch + board update
# implement (use /new-evaluator if needed)
/commit                      # [FEAT #42] message
/check                       # lint + typecheck + test
/pr                          # Create PR
/fix                         # Handle feedback (if any)
/complete-issue 42           # Cleanup after merge
```

---

## Workflow Details

### /start-work

```bash
gh issue view $ISSUE
git checkout main && git pull
git checkout -b <prefix>/$ISSUE-<description>
# Update board to "In Progress"
```

Branch prefixes: `feat/`, `fix/`, `refactor/`, `arch/`, `docs/`

### /commit

Format: `[TYPE #issue] Description`

```
[FEAT #42] Add ToxicityEvaluator

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

### /check

```bash
hatch run lint
hatch run typecheck
hatch run test
```

### /pr

Runs `/check` first, then:
```bash
git push -u origin <branch>
gh pr create --title "[TYPE #issue] Title" --body "template"
```

### /fix

```bash
gh pr view --json reviews,comments
# Apply fixes
git commit && git push
```

### /complete-issue

```bash
git checkout main && git pull
git branch -d <feature-branch>
git fetch --prune
```

---

## Feature Details

### /new-evaluator

Creates:
- `src/ragaliq/evaluators/{name}.py`
- `tests/unit/evaluators/test_{name}.py`
- `tests/integration/evaluators/test_{name}.py`

Requirements:
- Async `evaluate()` method
- Score 0.0-1.0
- Handle empty input
- Register with `@register_evaluator`

### /new-judge

Creates:
- `src/ragaliq/judges/{provider}.py`
- `tests/unit/judges/test_{provider}.py`

Requirements:
- All calls async
- Retry with exponential backoff
- Track token usage
- Handle malformed JSON

### /optimize-prompts

Workflow:
1. Create evaluation dataset with expected outputs
2. Measure baseline accuracy
3. Modify prompts in `src/ragaliq/judges/prompts/`
4. A/B test against baseline
5. Validate no parsing regressions

---

## Project Conventions

**Board:** https://github.com/users/dariero/projects/2/views/1

**Branch naming:** `<prefix>/<issue>-<description>`

**Commit format:** `[TYPE #issue] Description`

**Quality gates:** lint, typecheck, test (80% coverage)
