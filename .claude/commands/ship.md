# ship

Ship current work: run checks, create PR, review, merge, and cleanup — all in one command.

## Arguments

`$ARGUMENTS` - Optional: commit message or "draft" to create draft PR

## The Solo Developer Contract

This command assumes:
- You are the only developer
- Passing checks = ready to merge
- No approval gates needed beyond Claude's review
- Merged to main = deployed (CI/CD handles the rest)

## Process

### 1. Validate State

```bash
BRANCH=$(git branch --show-current)
```

**Guard rails:**
```
if [ "$BRANCH" = "main" ]; then
  echo "Error: Cannot ship from main. Use /start-work first."
  exit 1
fi
```

Extract issue number from branch:
```bash
ISSUE_NUMBER=$(echo $BRANCH | grep -oE '[0-9]+' | head -1)
```

### 2. Stage and Commit (if needed)

```bash
git status --porcelain
```

**If uncommitted changes exist:**
```bash
git add -A
git commit -m "[$(echo $BRANCH | cut -d'/' -f1 | tr '[:lower:]' '[:upper:]') #$ISSUE_NUMBER] $ARGUMENTS

Co-Authored-By: Claude <noreply@anthropic.com>"
```

If no `$ARGUMENTS`, generate message from diff analysis.

### 3. Run Quality Gates

```bash
hatch run lint && hatch run typecheck && hatch run test
```

**If any fail:**
```
✗ Quality gates failed. Fix issues before shipping.

  Lint:      ✗ 2 errors
  Typecheck: ✓
  Tests:     ✗ 1 failure

Run `hatch run format` for auto-fixable lint issues.
```
Stop here. Do not create PR.

### 4. Push and Create PR

```bash
git push -u origin $BRANCH

# Auto-generate PR metadata (labels, assignees, milestone)
PR_META=$(python .github/pr_metadata.py --gh-flags)

gh pr create \
  --title "[$(echo $BRANCH | cut -d'/' -f1 | tr '[:lower:]' '[:upper:]') #$ISSUE_NUMBER] $(gh issue view $ISSUE_NUMBER --json title -q .title)" \
  --body "$(cat <<EOF
Closes #$ISSUE_NUMBER

## Changes
$(git log main..$BRANCH --oneline | sed 's/^/- /')

## Checks
- [x] Lint passed
- [x] Type check passed
- [x] Tests passed

🤖 Shipped with Claude Code
EOF
)" \
  $PR_META
```

### 5. Self-Review (Claude)

Fetch and analyze the diff:
```bash
gh pr diff
```

**Review checklist (quick pass):**
- [ ] No secrets or credentials
- [ ] No debug code left behind
- [ ] Tests cover new functionality
- [ ] No obvious logic errors

**If issues found:**
```
⚠ Review found issues:

  1. Possible credential in src/config.py:42
  2. TODO comment should be resolved

Fix these issues and run /ship again.
```
Stop here. Do not merge.

**If clean:**
```
✓ Review passed. Proceeding to merge.
```

### 6. Merge PR

```bash
gh pr merge --squash --delete-branch
```

### 7. Update Project Board

Move issue to "Done":
```bash
# Get project item ID for this issue
ITEM_ID=$(gh api graphql -f query='
  query($owner: String!, $number: Int!) {
    user(login: $owner) {
      projectV2(number: $number) {
        items(first: 100) {
          nodes {
            id
            content { ... on Issue { number } }
          }
        }
      }
    }
  }
' -f owner="dariero" -F number=2 | jq -r ".data.user.projectV2.items.nodes[] | select(.content.number == $ISSUE_NUMBER) | .id")

# Move to Done
gh api graphql -f query='
  mutation($project: ID!, $item: ID!, $field: ID!, $value: String!) {
    updateProjectV2ItemFieldValue(
      input: {
        projectId: $project
        itemId: $item
        fieldId: $field
        value: { singleSelectOptionId: $value }
      }
    ) { projectV2Item { id } }
  }
' -f project="PVT_kwHODR8J4s4BNe_Y" \
  -f item="$ITEM_ID" \
  -f field="PVTSSF_lAHODR8J4s4BNe_Yzg8dwP8" \
  -f value="caff0873"  # Done✨
```

### 8. Local Cleanup

```bash
git checkout main
git pull origin main
git branch -d $BRANCH
git fetch --prune
```

### 9. Report Success

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 Shipped #$ISSUE_NUMBER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PR:     #XX (merged)
Branch: $BRANCH (deleted)
Status: Done ✅

Time from start-work: X minutes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Next task?
  gh issue list --assignee @me --state open --limit 5
  /start-work <number>
```

## Draft Mode

If `$ARGUMENTS` is "draft":
- Create PR but don't merge
- Don't move board status
- Don't cleanup branch

```
/ship draft
```

Useful when you want a PR for discussion but aren't ready to merge.

## Escape Hatches

**Skip checks (dangerous):**
```
/ship --force
```
Only use when you know what you're doing (hotfix, etc.)

**Partial ship (PR only):**
```
/ship --pr-only
```
Creates PR but stops before merge. Use when you actually want review.

## Error Handling

**No changes to ship:**
```
Nothing to ship. Working tree is clean and no commits ahead of main.
```

**PR already exists:**
```
PR #47 already exists for this branch.

Options:
  1. Update PR: git push (already done)
  2. Merge existing: gh pr merge 47 --squash
  3. Close and recreate: gh pr close 47; /ship
```

**Merge conflicts:**
```
Cannot merge: conflicts with main.

Fix:
  git fetch origin main
  git rebase origin/main
  # resolve conflicts
  /ship
```

## Success Criteria

- [ ] All uncommitted changes committed
- [ ] Quality gates pass (lint, typecheck, tests)
- [ ] PR created with proper format
- [ ] Self-review passes
- [ ] PR merged and branch deleted
- [ ] Project board updated to Done
- [ ] Local environment cleaned up
- [ ] Back on main branch
