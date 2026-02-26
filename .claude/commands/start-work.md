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

### 3. Update Board to Doing + Set Priority and Size

All IDs (PROJECT_ID, STATUS_FIELD, PRIORITY_FIELD, SIZE_FIELD, option IDs) come from `.claude/CONSTANTS.md`. Look them up before constructing mutations. Do not use inline values.

Infer Priority and Size from the issue title prefix using the **Issue Type Defaults** table in `.claude/CONSTANTS.md`.

First, retrieve the project item ID for this issue:

```bash
gh api graphql -f query='
  query {
    repository(owner: "dariero", name: "RagaliQ") {
      issue(number: '$ARGUMENTS') {
        projectItems(first: 10) {
          nodes { id }
        }
      }
    }
  }
'
```

Then execute all three field updates in a single batched mutation, substituting values from CONSTANTS.md:

```bash
gh api graphql \
  -f projectId="<PROJECT_ID>" \
  -f itemId="<ITEM_ID>" \
  -f statusField="<STATUS_FIELD>" \
  -f priorityField="<PRIORITY_FIELD>" \
  -f sizeField="<SIZE_FIELD>" \
  -f statusValue="<DOING_ID>" \
  -f priorityValue="<PRIORITY_OPTION_ID>" \
  -f sizeValue="<SIZE_OPTION_ID>" \
  -f query='
    mutation(
      $projectId: ID!, $itemId: ID!,
      $statusField: ID!, $priorityField: ID!, $sizeField: ID!,
      $statusValue: String!, $priorityValue: String!, $sizeValue: String!
    ) {
      setStatus: updateProjectV2ItemFieldValue(input: {
        projectId: $projectId, itemId: $itemId, fieldId: $statusField
        value: { singleSelectOptionId: $statusValue }
      }) { projectV2Item { id } }

      setPriority: updateProjectV2ItemFieldValue(input: {
        projectId: $projectId, itemId: $itemId, fieldId: $priorityField
        value: { singleSelectOptionId: $priorityValue }
      }) { projectV2Item { id } }

      setSize: updateProjectV2ItemFieldValue(input: {
        projectId: $projectId, itemId: $itemId, fieldId: $sizeField
        value: { singleSelectOptionId: $sizeValue }
      }) { projectV2Item { id } }
    }
  '
```

Validate that all three `projectV2Item.id` fields in the response are non-null. If any is null, report the failure and abort.

### 4. Assign Self + Add Label

```bash
gh issue edit $ARGUMENTS --add-assignee @me --add-label <LABEL>
```

Infer label from title prefix:

| Title prefix | Label     |
|---|---|
| `[FIX]`      | `bug`     |
| `[FEAT]`     | `feat`    |
| `[REFACTOR]` | `refactor`|
| `[DOCS]`     | `docs`    |
| (none)       | `feat`    |

### 5. Show Context

Display: branch name, board status (Doing), priority, size, label applied, issue title, and first 500 chars of issue body.

End with: `When done: /ship`
