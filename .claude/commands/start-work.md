# start-work

Begin work on a GitHub issue. Creates branch, moves card to "Doing".

## Arguments

`$ARGUMENTS` - GitHub issue number (required)

## Process

### 1. Get Issue

```bash
gh issue view $ARGUMENTS --json title,body,number,state
```

Parse issue type from title prefix: `[FEAT]`, `[FIX]`, `[REFACTOR]`, `[DOCS]`

### 2. Sync Main

```bash
git checkout main
git pull origin main
```

### 3. Create Branch

```bash
git checkout -b <prefix>/$ARGUMENTS-<short-description>
```

| Title Prefix | Branch Prefix |
|--------------|---------------|
| `[FEAT]` | `feat/` |
| `[FIX]` | `fix/` |
| `[REFACTOR]` | `refactor/` |
| `[DOCS]` | `docs/` |
| (none) | `feat/` |

### 4. Update Board → Doing

```bash
# Get item ID
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
' -f owner="dariero" -F number=2 | jq -r ".data.user.projectV2.items.nodes[] | select(.content.number == $ARGUMENTS) | .id")

# Move to Doing
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
  -f value="47fc9ee4"  # Doing♟️
```

### 5. Assign Self

```bash
gh issue edit $ARGUMENTS --add-assignee dariero
```

### 6. Show Context

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔄 Started #$ARGUMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Branch: feat/$ARGUMENTS-description
Status: Doing

[Issue title here]

[Issue body here - first 500 chars]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When done: /ship
```

## Error Handling

**Uncommitted changes:**
```
You have uncommitted changes.
  1. Stash: git stash
  2. Discard: git checkout .
```

**Branch exists:**
```
Branch feat/$ARGUMENTS-* already exists.
Switching to it: git checkout feat/$ARGUMENTS-*
```

## Success Criteria

- [ ] On feature branch
- [ ] Main is up to date
- [ ] Board shows "Doing"
- [ ] Issue assigned
