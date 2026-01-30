# pr

Create a pull request with proper template, quality gates, and issue linking.

## Arguments

`$ARGUMENTS` - Additional PR context or description (optional)

## Pre-requisites

Quality checks must pass before PR creation. If not already run, this command executes them first.

## Process

### 1. Run Quality Gates

```bash
hatch run lint
hatch run test
```

**If checks fail:**
```
✗ Quality gates failed. Cannot create PR.

Issues found:
  - Lint: 2 errors
  - Tests: 1 failure

Run /check to see details and fix issues.
```

Abort PR creation if any gate fails.

### 2. Gather Branch Information

```bash
git branch --show-current
git log main..HEAD --oneline
git diff main --stat
```

Collect:
- Current branch name: `feat/42-add-toxicity-evaluator`
- Commits since main: list of commit messages
- Files changed: count and names

### 3. Extract Issue Context

Parse from branch name:
- Issue number: `42`
- PR type: `FEAT`

Fetch issue details:
```bash
gh issue view 42 --json title,body
```

### 4. Check Remote Status

```bash
git fetch origin
git rev-list --left-right --count main...HEAD
```

**If branch not pushed:**
```bash
git push -u origin feat/42-add-toxicity-evaluator
```

**If behind remote:**
```
Warning: Remote branch has commits not in local.
Pull and resolve before creating PR:
  git pull origin feat/42-add-toxicity-evaluator
```

### 5. Generate PR Content

**Title format:**
```
[TYPE #issue] Concise description
```
Example: `[FEAT #42] Add ToxicityEvaluator for response safety checking`

**Body template:**
```markdown
## Summary

Brief description of what this PR does and why.

## Changes

- List of significant changes
- Another change
- Technical decisions made

## Test Plan

- [ ] Unit tests added for new evaluator
- [ ] Edge cases covered (empty input, boundary scores)
- [ ] Integration test with real API (skipped in CI without key)
- [ ] Manual testing completed

## Checklist

- [ ] Code follows project conventions (Pydantic v2, async-first)
- [ ] Type hints on all public APIs
- [ ] Google-style docstrings added
- [ ] No secrets or credentials exposed
- [ ] Lint and type checks pass

## RagaliQ-Specific

- [ ] Evaluator returns 0.0-1.0 scores
- [ ] Judge calls are async
- [ ] Token usage tracked (if applicable)
- [ ] Error handling for LLM failures

Closes #42

🤖 Generated with Claude Code
```

### 6. Create Pull Request

```bash
gh pr create \
  --title "[FEAT #42] Add ToxicityEvaluator for response safety checking" \
  --body "$(cat <<'EOF'
## Summary

Implement ToxicityEvaluator that detects harmful, toxic, or inappropriate
content in LLM responses using claim-based scoring.

## Changes

- Add `ToxicityEvaluator` class inheriting from `Evaluator` base
- Implement configurable toxicity categories
- Add claim-level breakdown in evaluation metadata
- Register evaluator in global registry

## Test Plan

- [x] Unit tests with mocked judge responses
- [x] Edge cases: empty input, all toxic, all safe
- [x] Score boundary tests (0.0, 1.0, threshold)
- [ ] Integration test with Anthropic API

## Checklist

- [x] Code follows project conventions
- [x] Type hints on all public APIs
- [x] Google-style docstrings
- [x] No exposed credentials

## RagaliQ-Specific

- [x] Returns 0.0-1.0 scores (0=safe, 1=toxic)
- [x] Async evaluate() method
- [x] Metadata includes per-claim scores

Closes #42

🤖 Generated with Claude Code
EOF
)" \
  --assignee dariero
```

### 7. Report Result

```
✓ Pull request created successfully

  PR #47: [FEAT #42] Add ToxicityEvaluator for response safety checking
  URL: https://github.com/dariero/RagaliQ/pull/47

  Branch: feat/42-add-toxicity-evaluator → main
  Commits: 3
  Files changed: 4 (+234, -2)

GitHub automation will:
  • Move PR to "🤖 AI Review" column
  • Run CI checks

Next steps:
  • Wait for CI to pass
  • Address any review feedback with /fix
  • After merge, run /complete-issue 42
```

## PR Best Practices

**Title:**
- Under 72 characters
- Imperative mood ("Add" not "Added")
- Include type and issue number

**Summary:**
- What changed and why (not how)
- Link to related issues/discussions
- Call out breaking changes

**Test Plan:**
- Specific, actionable checklist
- Include both automated and manual testing
- Note any tests that require API keys

**Avoid:**
- Huge PRs (split if > 500 lines)
- Multiple unrelated changes
- Missing tests for new code

## Error Handling

**No commits to push:**
```
Error: No commits between main and current branch.
Make changes and /commit before creating PR.
```

**PR already exists:**
```
A PR already exists for this branch:
  PR #45: [FEAT #42] Add ToxicityEvaluator
  URL: https://github.com/dariero/RagaliQ/pull/45

Options:
  1. View existing PR: gh pr view 45
  2. Update existing PR: git push (commits auto-added)
```

**Conflicts with main:**
```
Warning: Branch has conflicts with main.
Resolve before creating PR:
  git fetch origin main
  git rebase origin/main
  # resolve conflicts
  git push --force-with-lease
```

## Success Criteria

- [ ] All quality gates pass
- [ ] Branch pushed to remote
- [ ] PR created with proper format
- [ ] Issue linked with "Closes #X"
- [ ] PR URL displayed
