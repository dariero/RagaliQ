# start-work

Begin work on a GitHub issue. Creates branch, updates board, assigns you.

## Arguments

`$ARGUMENTS` - GitHub issue number (required)

## Process

### 1. Get Issue

```bash
gh issue view $ARGUMENTS --json title,body,number,state
```

Parse type from title prefix: `[FEAT]`, `[FIX]`, `[REFACTOR]`, `[DOCS]`

### 2. Sync and Branch

```bash
git checkout main && git pull origin main
git checkout -b <prefix>/$ARGUMENTS-<short-description>
```

Branch prefix is derived from title prefix (see WORKFLOW.md § Project Constants).

If uncommitted changes exist, stop and ask: stash or discard?
If branch already exists, switch to it.

### 3. Update Board to Doing

Use GraphQL to move the project item to "Doing" status (see WORKFLOW.md § Project Constants for IDs).

### 4. Assign Self

```bash
gh issue edit $ARGUMENTS --add-assignee dariero
```

### 5. Show Context

Display: branch name, board status, issue title, and first 500 chars of issue body.

End with: `When done: /ship`
