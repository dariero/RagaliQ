# commit

Create a commit following project conventions with automatic issue linking.

## Arguments

`$ARGUMENTS` - Commit message or change description (optional, will be prompted if not provided)

## Process

### 1. Review Current State

```bash
git status
git diff --staged
git diff
```

Display:
```
Staged changes:
  modified:   src/ragaliq/evaluators/toxicity.py
  new file:   tests/unit/evaluators/test_toxicity.py

Unstaged changes:
  modified:   src/ragaliq/evaluators/__init__.py
```

### 2. Extract Issue Context from Branch

Parse current branch name:
```bash
git branch --show-current
```

Example: `feat/42-add-toxicity-evaluator`
- Issue number: `42`
- Type: `FEAT` (from `feat/` prefix)

**Branch prefix mapping:**

| Branch Prefix | Commit Type |
|---------------|-------------|
| `feat/` | `FEAT` |
| `fix/` | `FIX` |
| `refactor/` | `REFACTOR` |
| `arch/` | `ARCH` |
| `docs/` | `DOCS` |

### 3. Stage Changes (if needed)

If there are unstaged changes:
```
You have unstaged changes. What would you like to do?

  1. Stage all changes (git add -A)
  2. Stage specific files
  3. Commit only staged changes
  4. Cancel

Choose [1-4]:
```

**If staging specific files:**
```
Select files to stage:
  [ ] src/ragaliq/evaluators/__init__.py
  [x] src/ragaliq/evaluators/toxicity.py
  [x] tests/unit/evaluators/test_toxicity.py
```

Prefer explicit file staging over `git add -A` to avoid committing unintended files.

### 4. Generate Commit Message

**If `$ARGUMENTS` provided:**
Use as description, format appropriately.

**If not provided:**
Analyze staged changes and suggest:
```
Suggested commit message based on changes:

  [FEAT #42] Add ToxicityEvaluator with claim-based scoring

  - Implement evaluate() method with configurable threshold
  - Add unit tests for edge cases
  - Register evaluator in registry

Accept this message? [Y/n/edit]
```

### 5. Format and Create Commit

**Commit format:**
```
[TYPE #issue] Concise description (imperative mood)

- Bullet point details (optional)
- Another detail

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

**Rules:**
- First line: max 72 characters
- Type matches branch prefix
- Issue number from branch
- Imperative mood ("Add" not "Added")
- Body wrapped at 72 characters
- Co-author on separate line at end

**Execute commit:**
```bash
git commit -m "$(cat <<'EOF'
[FEAT #42] Add ToxicityEvaluator with claim-based scoring

- Implement evaluate() method with configurable threshold
- Add unit tests for edge cases (empty input, perfect score)
- Register evaluator in registry
- Add Google-style docstrings

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

### 6. Confirm Success

```bash
git log -1 --oneline
```

Display:
```
✓ Commit created successfully

  a1b2c3d [FEAT #42] Add ToxicityEvaluator with claim-based scoring

Files committed:
  src/ragaliq/evaluators/toxicity.py (+145)
  tests/unit/evaluators/test_toxicity.py (+89)

Next steps:
  • Continue working and /commit again
  • /check - Run quality gates
  • /pr - Create pull request (after /check)
```

## Commit Message Examples

**Feature:**
```
[FEAT #42] Add ToxicityEvaluator for response safety checking

- Score responses 0.0-1.0 based on harmful content detection
- Support configurable toxicity categories
- Include claim-level breakdown in metadata

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

**Bug fix:**
```
[FIX #58] Handle empty context in FaithfulnessEvaluator

- Return score 0.0 when context is empty instead of raising
- Add explicit test case for empty context scenario

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

**Refactor:**
```
[REFACTOR #71] Convert ClaudeJudge to async/await pattern

- Replace synchronous client with AsyncAnthropic
- Use asyncio.gather for parallel claim verification
- Maintain backward compatibility with sync wrapper

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

## Error Handling

**Not on feature branch:**
```
Warning: You're on 'main' branch, not a feature branch.
Commits should be made on feature branches.

Options:
  1. Create branch first: /start-work <issue-number>
  2. Continue anyway (not recommended)
```

**No changes to commit:**
```
Nothing to commit. Working tree is clean.

Make changes first, then run /commit.
```

**Commit hook failed:**
```
Pre-commit hook failed:

  Lint check failed - 2 issues found
  ...

Fix issues and try again. Do NOT use --no-verify.
```

## Success Criteria

- [ ] Branch name parsed correctly for issue/type
- [ ] Changes staged appropriately
- [ ] Commit message follows format
- [ ] Commit created successfully
- [ ] No hooks bypassed
