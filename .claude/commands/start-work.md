# start-work

Begin work on a GitHub issue. Creates branch, updates board, assigns you.

## Arguments

`$ARGUMENTS` - GitHub issue number (required)

## Process

### 1. Get Issue

```bash
gh issue view $ARGUMENTS --json title,body,number,state
```

Validate issue state: if `state` is `closed`, warn the user and ask whether to reopen or abort. DO NOT proceed on a closed issue without explicit confirmation.

Parse type from title prefix: `[FEAT]`, `[FIX]`, `[REFACTOR]`, `[DOCS]`

### 2. Sync and Branch

```bash
git checkout main && git pull origin main
git checkout -b <prefix>/$ARGUMENTS-<short-description>
```

Branch prefix is derived from title prefix (see `.claude/CONSTANTS.md`).

If uncommitted changes exist: STOP and ask the user whether to stash or discard. DO NOT make this decision autonomously.

If branch already exists, switch to it.

### 3. Update Board to Doing

Use GraphQL to move the project item to "Doing" status (see `.claude/CONSTANTS.md` for IDs).

### 4. Assign Self

```bash
gh issue edit $ARGUMENTS --add-assignee dariero
```

### 5. Show Context

Display: branch name, board status, issue title, and first 500 chars of issue body.

End with: `When done: /ship`
