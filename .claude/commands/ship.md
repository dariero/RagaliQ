# ship

Ship current work: commit, check, PR, review, merge, cleanup -- all in one command.

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
ISSUE_NUMBER=$(echo $BRANCH | grep -oE '[0-9]+' | head -1)
```

### 2. Commit (if needed)

If `git status --porcelain` shows changes:

```bash
git add -A
git commit -m "[TYPE #$ISSUE_NUMBER] message

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### 3. Quality Gates

```bash
hatch run lint && hatch run typecheck && hatch run test
```

If any fail: report which failed, suggest `hatch run format` for auto-fixable lint. **Stop here.**

### 4. Push and Create PR

```bash
git push -u origin $BRANCH
gh pr create \
  --title "[TYPE #$ISSUE_NUMBER] $(gh issue view $ISSUE_NUMBER --json title -q .title)" \
  --body "Closes #$ISSUE_NUMBER ..."
```

Include: change list from `git log main..$BRANCH --oneline`, checks passed confirmation.

### 5. Self-Review

Run `gh pr diff` and check:
- No secrets or credentials
- No debug code left behind
- Tests cover new functionality

If issues found: report them and **stop**. Do not merge.

### 6. Merge and Cleanup

```bash
gh pr merge --squash --delete-branch
```

Update board to "Done" via GraphQL (see WORKFLOW.md § Project Constants for IDs).

```bash
git checkout main && git pull origin main
git branch -d $BRANCH && git fetch --prune
```

### 7. Report

Show: PR number, branch deleted, board status Done. Suggest next task via `gh issue list`.

## Draft Mode

If `$ARGUMENTS` is "draft": create PR but skip steps 6-7. Useful for discussion before merge.

## Error Handling

- **On main:** "Cannot ship from main. Use /start-work first."
- **No changes:** "Nothing to ship. Working tree is clean."
- **PR exists:** Show options: push updates, merge existing, or close and recreate.
- **Merge conflicts:** Suggest `git rebase origin/main`, then `/ship` again.
