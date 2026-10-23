---
description: Ship the current branch to main through commit, quality gate, PR, CI, squash merge, board Done and cleanup. Use when the user asks for /ship.
---
# ship

Ship current work: commit, check, PR, review, merge, cleanup -- all in one command.

<critical>
## MANDATORY Pre-Merge Checklist

Before ANY merge, verify ALL of the following via `gh pr diff`:
- No secrets, credentials, API keys, or .env content
- No debug code (print statements, console.log, breakpoints)
- No TODO/FIXME/HACK comments introduced
- Tests exist for new functionality
- All quality gates passed (lint, typecheck, test)

If ANY item fails: STOP. Report the issue. DO NOT merge.
</critical>

## Arguments

`$ARGUMENTS` - Optional. Interpreted as:

| Input | Behavior |
|-------|----------|
| (empty) | Auto-generate commit message from diff |
| `draft` | Create PR but skip merge, board update, cleanup |
| any text | Use as commit message |

## Assumptions

Solo developer. Passing checks = ready to merge. No approval gates beyond Claude's review.

## Process

### 1. Validate State

Extract branch and issue number. Abort if on main.

```bash
BRANCH=$(git branch --show-current)
ISSUE_NUMBER=$(echo "$BRANCH" | sed 's|.*/||' | grep -oE '^[0-9]+')
COMMIT_TYPE=$(echo "$BRANCH" | grep -oE '^[^/]+')  # feat, fix, refactor, docs, chore
```

Map COMMIT_TYPE to the uppercase form using the **Commit Type Mapping** table in `.claude/CONSTANTS.md` (e.g. `feat` → `FEAT`, `fix` → `FIX`).

Validate that ISSUE_NUMBER is non-empty:

```bash
if [ -z "$ISSUE_NUMBER" ]; then
  echo "ERROR: Could not extract issue number from branch '$BRANCH'. Expected format: <prefix>/<number>-<description>"
  # STOP. Do not proceed.
fi
```

Shell variables do not persist between tool calls. Write the values of BRANCH and ISSUE_NUMBER literally in every later command.

### 2. Commit (if needed)

If `git status --porcelain` is empty, run `git log main..$BRANCH --oneline`. If that is empty too, report "Nothing to ship. Working tree is clean and no commits ahead of main." and **STOP**.

If `git status --porcelain` shows changes:

1. Review changed files:
   ```bash
   git diff --name-only
   git diff --cached --name-only
   ```

2. Stage files explicitly. NEVER use `git add -A`. Exclude: `.env*`, `*.pem`, `*credentials*`, `*secret*`, `.DS_Store`, `__pycache__/`, build artifacts.

3. Commit, with no attribution line:
   ```bash
   git commit -m "[TYPE #$ISSUE_NUMBER] message"
   ```

### 3. Quality Gates

These are the quality gate in `AGENTS.md`. The gates run from the repo's `.venv`:

1. If `.venv/bin/python` does not exist: report it and **STOP**. Do not create the environment.
2. Sync `.venv` to this branch's `pylock.toml`. Without this step the gates test whichever lock was synced last. `--python` is the install target here, so it is always `.venv/bin/python`, never a system or Homebrew interpreter. If the command fails: report it and **STOP**; do not run the gates.
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

If lint fails: run `.venv/bin/ruff format src/ tests/ scripts/`, then `.venv/bin/ruff check --fix src/ tests/ scripts/`, commit the files it changed as in `/ship` step 2, then re-run the gates. If typecheck or test fails: report which failed and **STOP**.

### 4. Push and Create PR

```bash
GIT_TERMINAL_PROMPT=0 git push -u origin $BRANCH
gh pr create --base main --head $BRANCH \
  --title "[TYPE #$ISSUE_NUMBER] $(gh issue view $ISSUE_NUMBER --json title -q .title | sed -E 's/^\[[A-Z]+\] +//')" \
  --body "Closes #$ISSUE_NUMBER ..."
```

The PR title is `[TYPE #N] ` plus the issue title without its leading `[TYPE] `.

Include: change list from `git log main..$BRANCH --oneline`, checks passed confirmation.

Then wait for CI on the PR, where `<PR>` is the number at the end of the URL that `gh pr create` printed:

```bash
gh pr checks <PR> --watch
```

Every check must be passed or skipped. If any check fails or is cancelled: report it and **STOP**. DO NOT merge.

### 5. Self-Review

Run `gh pr diff` and execute the MANDATORY Pre-Merge Checklist at the top of this document.

If issues found: report them and **STOP**. DO NOT merge.

### 6. Merge and Cleanup

Right before the merge, run `gh pr checks <PR>` again. Every check must still be passed or skipped; if not, report it and **STOP**.

Then record the tip with `git rev-parse HEAD` and run `gh pr view <PR> --json state,headRefOid,mergeable,mergeStateStatus`: `state` must be `OPEN`, `headRefOid` the recorded tip, and `mergeable` `MERGEABLE`. `CONFLICTING` is a merge conflict (see Error Handling): report it and **STOP**.

```bash
gh pr merge <PR> --squash --delete-branch --match-head-commit <TIP>
```

Update the board to Done. All IDs (PROJECT_ID, STATUS_FIELD, the Done option ID) come from `.claude/CONSTANTS.md`. First, retrieve the project item ID for this issue:

```bash
gh api graphql -f query='
  query {
    repository(owner: "dariero", name: "RagaliQ") {
      issue(number: '$ISSUE_NUMBER') {
        projectItems(first: 10) {
          nodes { id }
        }
      }
    }
  }
'
```

Then set Status to Done:

```bash
gh api graphql \
  -f projectId="<PROJECT_ID>" \
  -f itemId="<ITEM_ID>" \
  -f statusField="<STATUS_FIELD>" \
  -f statusValue="<DONE_ID>" \
  -f query='
    mutation($projectId: ID!, $itemId: ID!, $statusField: ID!, $statusValue: String!) {
      setStatus: updateProjectV2ItemFieldValue(input: {
        projectId: $projectId, itemId: $itemId, fieldId: $statusField
        value: { singleSelectOptionId: $statusValue }
      }) { projectV2Item { id } }
    }
  '
```

Validate that `projectV2Item.id` in the response is non-null. If it is null, report the failure.

Before `git checkout main`: `git status --porcelain` must be empty and `gh pr view <PR> --json state` must say `MERGED`; if not, report it and **STOP**.

```bash
git checkout main && git pull --ff-only origin main
git branch --list $BRANCH
git fetch --prune
```

`gh pr merge --delete-branch` already deletes the local branch. Run `git branch -D $BRANCH` only if `git branch --list $BRANCH` still shows it.

### 7. Report

Show: PR number, branch deleted, board status Done. List open issues: `gh issue list --state open`.

## Draft Mode

If `$ARGUMENTS` is "draft": create PR but skip steps 6-7. Useful for discussion before merge.

## Error Handling

- **On main:** "Cannot ship from main. Use /start-work first."
- **No changes and no commits ahead of main:** "Nothing to ship. Working tree is clean and no commits ahead of main."
- **PR exists:** Show options: push updates, merge existing, or close and recreate.
- **Merge conflicts:** "Cannot merge: conflicts with main." Report the conflict and **STOP**. Do not rebase.
