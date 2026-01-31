# GitHub Labels Guide

## Label Structure

RagaliQ uses a **2-dimensional labeling system**:

```
Every PR should have: 1 TYPE + 0-2 SCOPE labels
```

### Type Labels (Pick 1)

These describe **what kind of change** this is:

| Label | Description | Color | When to use |
|-------|-------------|-------|-------------|
| `feat` | New functionality | 🟢 Green | Adding new features or capabilities |
| `bug` | Bugs and fixes | 🔴 Red | Fixing broken behavior or errors |
| `refactor` | Code improvements | 🟠 Orange | Restructuring code without changing behavior |
| `chore` | Maintenance | 🟣 Purple | Dependencies, tooling, config updates |
| `docs` | Documentation | 🔵 Blue | Docs, examples, docstrings |
| `research` | Exploration | 🟢 Lime | Investigating new methods or approaches |

### Scope Labels (Pick 0-2)

These describe **which part of the codebase** is affected:

| Label | Description | Color | Component |
|-------|-------------|-------|-----------|
| `judge` | LLM judge logic | 🟡 Yellow | `src/ragaliq/judges/` |
| `evaluator` | Evaluator implementations | 🟠 Orange | `src/ragaliq/evaluators/` |
| `core` | Base classes & architecture | ⚪ Gray | `src/ragaliq/core/` |
| `cli` | Command-line interface | 🟢 Green | `src/ragaliq/cli/` |
| `dataset` | Test data & generation | 🟢 Lime | `src/ragaliq/datasets/` |
| `report` | Output formatting | 🟡 Amber | `src/ragaliq/reports/` |
| `pytest` | Pytest plugin | 🔵 Cyan | `src/ragaliq/integrations/pytest/` |
| `async` | Async/await patterns | 🔵 Sky | Async logic across codebase |
| `infra` | CI/CD & deployment | 🟢 Olive | `.github/`, `Dockerfile`, `pyproject.toml` |
| `testing` | Test infrastructure | 🔵 Indigo | `tests/` |

## Examples

### Good Labeling

```
PR: "Add faithfulness evaluator with async support"
Labels: feat, evaluator, async
       ↑     ↑         ↑
     type   scope1   scope2
```

```
PR: "Fix bug in Claude judge API timeout"
Labels: bug, judge
       ↑     ↑
     type   scope
```

```
PR: "Refactor core runner architecture"
Labels: refactor, core
       ↑         ↑
     type      scope
```

```
PR: "Update dependencies to latest versions"
Labels: chore
       ↑
     type only (no specific scope)
```

### Auto-Generated Labels

The `/ship` command automatically infers labels from:
- Branch name (e.g., `feat/12-add-faithfulness-evaluator`)
- Commit messages
- Changed files

You can override by manually editing the PR labels after creation.

## Migration Notes

### Removed Labels

The following labels were removed in favor of milestones:
- ~~`phase-1-foundation`~~ → Use milestone instead
- ~~`phase-2-evaluators`~~ → Use milestone instead
- ~~`phase-3-usability`~~ → Use milestone instead
- ~~`phase-4-reports`~~ → Use milestone instead

### Merged Labels

- ~~`feature`~~ → Merged into `feat`
- ~~`prompt`~~ → Use `judge` or `core` instead

## Visual Color Scheme

**Type labels** use bright, saturated colors:
- Easy to spot the change type at a glance
- One per PR maximum

**Scope labels** use muted, pastel colors:
- Less visually prominent
- Can combine multiple scopes

This creates a visual hierarchy: **TYPE** stands out, scope provides context.
