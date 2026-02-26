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
COMMIT_TYPE=$(echo "$BRANCH" | grep -oE '^[^/]+')  # feat, fix, refactor, docs
```

Map COMMIT_TYPE to the uppercase form using the **Commit Type Mapping** table in `.claude/CONSTANTS.md` (e.g. `feat` → `FEAT`, `fix` → `FIX`).

Validate that ISSUE_NUMBER is non-empty:

```bash
if [ -z "$ISSUE_NUMBER" ]; then
  echo "ERROR: Could not extract issue number from branch '$BRANCH'. Expected format: <prefix>/<number>-<description>"
  # STOP. Do not proceed.
fi
```

### 2. Commit (if needed)

If `git status --porcelain` shows changes:

1. Review changed files:
   ```bash
   git diff --name-only
   git diff --cached --name-only
   ```

2. Stage files explicitly. NEVER use `git add -A`. Exclude: `.env*`, `*.pem`, `*credentials*`, `*secret*`, `.DS_Store`, `__pycache__/`, build artifacts.

3. Commit:
   ```bash
   git commit -m "[TYPE #$ISSUE_NUMBER] message

   Co-Authored-By: Claude <noreply@anthropic.com>"
   ```

### 3. Quality Gates

```bash
hatch run lint && hatch run typecheck && hatch run test
```

If lint fails: run `hatch run format` to auto-fix, then re-run gates. If typecheck or test fails: report which failed and **STOP**.

### 4. Push and Create PR

```bash
git push -u origin $BRANCH
gh pr create \
  --title "[TYPE #$ISSUE_NUMBER] $(gh issue view $ISSUE_NUMBER --json title -q .title)" \
  --body "Closes #$ISSUE_NUMBER ..."
```

Include: change list from `git log main..$BRANCH --oneline`, checks passed confirmation.

### 5. Self-Review

Run `gh pr diff` and execute the MANDATORY Pre-Merge Checklist at the top of this document.

If issues found: report them and **STOP**. DO NOT merge.

### 6. Merge and Cleanup

```bash
gh pr merge --squash --delete-branch
```

Update board to "Done" via GraphQL (see `.claude/CONSTANTS.md` for IDs).

```bash
git checkout main && git pull origin main
git branch -D $BRANCH && git fetch --prune
```

### 7. Report

Show: PR number, branch deleted, board status Done. List open issues: `gh issue list --state open`.

## Draft Mode

If `$ARGUMENTS` is "draft": create PR but skip steps 6-7. Useful for discussion before merge.

## Error Handling

- **On main:** "Cannot ship from main. Use /start-work first."
- **No changes:** "Nothing to ship. Working tree is clean."
- **PR exists:** Show options: push updates, merge existing, or close and recreate.
- **Merge conflicts:** Rebase onto origin/main: `git rebase origin/main`, then `/ship` again.
