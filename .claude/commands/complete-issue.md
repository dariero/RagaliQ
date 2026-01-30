# complete-issue

Cleanup local environment after a PR is merged. Removes feature branch and syncs with main.

## Arguments

`$ARGUMENTS` - Issue number (required)

## Process

### 1. Verify PR is Merged

Find the PR associated with this issue:
```bash
gh pr list --state merged --search "Closes #$ARGUMENTS in:body" --json number,title,mergedAt
```

**If no merged PR found:**
```
No merged PR found for issue #$ARGUMENTS.

Possible reasons:
  • PR hasn't been merged yet
  • PR doesn't reference this issue

Check PR status:
  gh pr list --search "head:*/$ARGUMENTS-*"
```

**If PR found:**
```
Found merged PR:
  PR #47: [FEAT #42] Add ToxicityEvaluator
  Merged: 2024-01-15 14:32:00

Proceeding with cleanup...
```

### 2. Switch to Main Branch

```bash
git checkout main
```

**If on feature branch with uncommitted changes:**
```
Warning: You have uncommitted changes on feat/42-add-toxicity-evaluator

Options:
  1. Stash changes: git stash
  2. Discard changes: git checkout .
  3. Cancel cleanup

Choose [1-3]:
```

### 3. Pull Latest Main

```bash
git pull origin main
```

Ensure local main includes the merged PR.

### 4. Delete Local Feature Branch

Determine branch name from issue:
```bash
git branch --list "*/$ARGUMENTS-*"
```

Delete the branch:
```bash
git branch -d feat/42-add-toxicity-evaluator
```

**If branch has unmerged changes:**
```
Warning: Branch has unmerged changes.

This usually means:
  • Additional commits were made after PR merge
  • Changes weren't included in the PR

Options:
  1. Force delete (lose changes): git branch -D feat/42-add-toxicity-evaluator
  2. Keep branch and investigate
  3. Cancel

Choose [1-3]:
```

### 5. Prune Remote Tracking Branches

```bash
git fetch --prune
```

Remove references to deleted remote branches.

### 6. Verify Cleanup

```bash
git branch -a
```

Confirm:
- Feature branch removed from local
- Remote tracking reference pruned
- Currently on main branch

### 7. Report Completion

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Issue #42 Completed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ PR #47 was merged
✓ Local branch deleted: feat/42-add-toxicity-evaluator
✓ Remote tracking pruned
✓ Main branch updated

GitHub automation handled:
  • PR moved to 🚀 Deployed
  • Issue #42 closed
  • Issue moved to ✨ Done

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Current branch: main
Local branches:
  * main

Ready for next task!

View backlog:
  gh issue list --assignee @me --state open

Start new work:
  /start-work <issue-number>
```

## Batch Cleanup

If multiple issues completed:
```
/complete-issue 42
/complete-issue 43
/complete-issue 44
```

Or use git to clean all merged branches:
```bash
git branch --merged main | grep -v main | xargs git branch -d
```

## What GitHub Automation Handles

You don't need to manually:
- Close the issue (done when PR merged with "Closes #X")
- Move PR to "Deployed" column (automation rule)
- Move issue to "Done" column (automation rule)

The `/complete-issue` command only handles local cleanup that automation can't do.

## Error Handling

**Issue still open:**
```
Issue #42 is still open.

Status: In Progress
PR: #47 (Open, awaiting review)

The PR hasn't been merged yet. Wait for merge, then run /complete-issue.

To check PR status:
  gh pr view 47
```

**Branch not found:**
```
No local branch found for issue #42.

Expected pattern: */42-*

Possible reasons:
  • Branch already deleted
  • Work was done on a differently-named branch

Nothing to clean up locally.
```

**Not on main after checkout:**
```
Failed to switch to main branch.

Current branch: feat/42-add-toxicity-evaluator

Error: Your local changes would be overwritten.

Options:
  1. Stash: git stash
  2. Discard: git checkout .
  3. Commit: /commit

Then retry /complete-issue 42.
```

## Success Criteria

- [ ] PR merge verified
- [ ] Switched to main branch
- [ ] Main branch up to date
- [ ] Feature branch deleted
- [ ] Remote tracking pruned
- [ ] Ready for next task
