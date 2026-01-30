# RagaliQ Development Workflow

## Core Principle

Every change links to a GitHub issue and follows:

```
Issue → Branch → Implement → Validate → PR → Review → Merge → Cleanup
```

## Daily Flow

### 1. Pick a Task

```bash
gh issue list --assignee @me --state open
```

### 2. Start Work

```
/start-work 42
```
- Creates `feat/42-description` branch
- Updates board to "In Progress"
- Shows issue context

### 3. Implement

Use feature commands if scaffolding:
- `/new-evaluator` - New evaluator
- `/new-judge` - New judge backend

Otherwise, just code normally.

### 4. Commit

```
/commit
```
- Extracts issue from branch
- Formats: `[FEAT #42] Description`
- Adds co-author

### 5. Validate

```
/check
```
- Runs lint, typecheck, test
- Must pass before PR

### 6. Submit

```
/pr
```
- Runs /check first
- Creates PR with template
- Links issue

### 7. Handle Feedback

```
/fix
```
- Fetches review comments
- Guides fixes
- Pushes update

### 8. Cleanup

```
/complete-issue 42
```
- Switches to main
- Deletes local branch
- Prunes remotes

## Conventions

| Item | Format |
|------|--------|
| Branch | `<prefix>/<issue>-<description>` |
| Commit | `[TYPE #issue] Description` |
| Prefixes | `feat/`, `fix/`, `refactor/`, `arch/`, `docs/` |

## Quality Gates

All must pass before PR:

```bash
hatch run lint        # Code style
hatch run typecheck   # Type checking
hatch run test        # Tests (80% coverage)
```

## Board

https://github.com/users/dariero/projects/2/views/1

Columns: 📋 Backlog → 🔄 In Progress → 🤖 AI Review → ✅ Approved → 🚀 Deployed → ✨ Done

Most transitions are automated. `/start-work` handles "In Progress".
