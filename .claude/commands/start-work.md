# start-work

Initialize work on a GitHub issue. Creates branch, updates project board, prepares context.

## Arguments

`$ARGUMENTS` - GitHub issue number (required)

## Process

### 1. Retrieve Issue Metadata

```bash
gh issue view $ARGUMENTS --json title,body,labels,number,state
```

Parse the issue to extract:
- Issue type from title prefix: `[FEAT]`, `[FIX]`, `[REFACTOR]`, `[ARCH]`, `[DOCS]`
- Labels for area and priority
- Acceptance criteria from body

### 1.5. Determine Task Metadata

**Priority Mapping** (from labels to project field):

| Label Pattern | Priority | Option ID |
|---------------|----------|-----------|
| `priority-critical`, `bug` | Critical | `79628723` |
| `priority-high`, `[FIX]` prefix | High | `0a877460` |
| `priority-medium` | Medium | `da944a9c` |
| `priority-low`, `docs` | Low | `56c1c445` |
| *No matching label* | Medium (default) | `da944a9c` |

**Size Detection** (from labels):

| Label | Size | Option ID |
|-------|------|-----------|
| `size-XS` | XS | `6c6483d2` |
| `size-S` | S | `f784b110` |
| `size-M` | M | `7515a9f1` |
| `size-L` | L | `817d0097` |
| `size-XL` | XL | `db339eb2` |

**If no size label exists**, prompt for T-shirt size:
```
Size not found in issue labels. Please estimate task size:
[S] Small (1-2 hours)
[M] Medium (3-5 hours)
[L] Large (1-2 days)
[XL] Extra Large (3+ days)

Enter size (S/M/L/XL):
```

Store the determined values:
- `PRIORITY_OPTION_ID`
- `SIZE_OPTION_ID`

### 2. Determine Branch Prefix

| Issue Title Prefix | Branch Prefix |
|--------------------|---------------|
| `[FEAT]` | `feat/` |
| `[FIX]` | `fix/` |
| `[REFACTOR]` | `refactor/` |
| `[ARCH]` | `arch/` |
| `[DOCS]` | `docs/` |

### 3. Synchronize Main Branch

```bash
git checkout main
git pull origin main
```

Ensure we're starting from the latest code.

### 4. Create Feature Branch

```bash
git checkout -b <prefix>/<issue-number>-<short-description>
```

Branch naming rules:
- Use issue number for traceability
- Short description: 2-4 words, kebab-case
- Example: `feat/42-add-toxicity-evaluator`

### 5. Update Project Board and Assignee

**5.1. Assign Issue to @dariero**

```bash
gh issue edit $ARGUMENTS --add-assignee dariero
```

**5.2. Retrieve Project Field IDs**

Query for project item and all required field information:

```bash
gh api graphql -f query='
  query($owner: String!, $number: Int!) {
    user(login: $owner) {
      projectV2(number: $number) {
        id
        items(first: 100) {
          nodes {
            id
            content {
              ... on Issue {
                number
              }
            }
          }
        }
        fields(first: 20) {
          nodes {
            ... on ProjectV2SingleSelectField {
              id
              name
              options {
                id
                name
              }
            }
          }
        }
      }
    }
  }
' -f owner="dariero" -F number=2
```

Extract the following IDs from the response:
- `PROJECT_ID`: `PVT_kwHODR8J4s4BNe_Y`
- `ITEM_ID`: Match issue number to get project item ID
- `STATUS_FIELD_ID`: `PVTSSF_lAHODR8J4s4BNe_Yzg8dwP8`
- `PRIORITY_FIELD_ID`: `PVTSSF_lAHODR8J4s4BNe_Yzg8dwQc`
- `SIZE_FIELD_ID`: `PVTSSF_lAHODR8J4s4BNe_Yzg8dwQg`
- `IN_PROGRESS_OPTION_ID`: `47fc9ee4` (for "🔄 In progress")

**5.3. Update Status to "In Progress"**

```bash
gh api graphql -f query='
  mutation($project: ID!, $item: ID!, $field: ID!, $value: String!) {
    updateProjectV2ItemFieldValue(
      input: {
        projectId: $project
        itemId: $item
        fieldId: $field
        value: { singleSelectOptionId: $value }
      }
    ) {
      projectV2Item { id }
    }
  }
' -f project="PVT_kwHODR8J4s4BNe_Y" \
  -f item="$ITEM_ID" \
  -f field="PVTSSF_lAHODR8J4s4BNe_Yzg8dwP8" \
  -f value="47fc9ee4"
```

**5.4. Update Priority** (only if not already set)

Check if Priority field is already populated. If empty or needs update:

```bash
gh api graphql -f query='
  mutation($project: ID!, $item: ID!, $field: ID!, $value: String!) {
    updateProjectV2ItemFieldValue(
      input: {
        projectId: $project
        itemId: $item
        fieldId: $field
        value: { singleSelectOptionId: $value }
      }
    ) {
      projectV2Item { id }
    }
  }
' -f project="PVT_kwHODR8J4s4BNe_Y" \
  -f item="$ITEM_ID" \
  -f field="PVTSSF_lAHODR8J4s4BNe_Yzg8dwQc" \
  -f value="$PRIORITY_OPTION_ID"
```

**5.5. Update Size** (only if not already set)

Check if Size field is already populated. If empty or needs update:

```bash
gh api graphql -f query='
  mutation($project: ID!, $item: ID!, $field: ID!, $value: String!) {
    updateProjectV2ItemFieldValue(
      input: {
        projectId: $project
        itemId: $item
        fieldId: $field
        value: { singleSelectOptionId: $value }
      }
    ) {
      projectV2Item { id }
    }
  }
' -f project="PVT_kwHODR8J4s4BNe_Y" \
  -f item="$ITEM_ID" \
  -f field="PVTSSF_lAHODR8J4s4BNe_Yzg8dwQg" \
  -f value="$SIZE_OPTION_ID"
```

**Field Update Logic:**
- Always update Status to "In Progress"
- Always set Assignee to @dariero
- Update Priority if currently empty or if inferred value differs from existing
- Update Size if currently empty (prompt if no label found)
- Skip updates if field already has the correct value

### 6. Summarize and Prepare

Display issue context with metadata confirmation:
```
✓ Switched to branch: feat/42-add-toxicity-evaluator
✓ Issue #42 assigned to @dariero
✓ Project board updated:
  - Status: In Progress
  - Priority: High
  - Size: M

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Issue #42: [FEAT] Add ToxicityEvaluator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Metadata:
  Priority: High (from label: priority-high)
  Size: M (from label: size-M)
  Estimate: 3-5 hours

Description:
Implement an evaluator that detects toxic, harmful, or inappropriate
content in LLM responses.

Acceptance Criteria:
- [ ] Evaluator returns 0.0-1.0 score (higher = more toxic)
- [ ] Configurable toxicity categories
- [ ] Unit tests with mocked judge
- [ ] Integration test with real API

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Suggested: /new-evaluator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Relevant files:
  src/ragaliq/evaluators/
  src/ragaliq/core/evaluator.py
  tests/unit/evaluators/
```

Suggest appropriate `/command` based on issue type:
- `[FEAT]` with "evaluator" → `/new-evaluator`
- `[FEAT]` with "judge" → `/new-judge`
- `[FEAT]` with "cli" → `/add-cli-command`
- `[FIX]` → analyze files mentioned in issue
- `[REFACTOR]` with "async" → `/refactor-async`

## Error Handling

**Issue not found:**
```
Error: Issue #$ARGUMENTS not found.
Check the issue number and try again.
```

**Uncommitted changes:**
```
Warning: You have uncommitted changes on current branch.
Options:
  1. Stash changes: git stash
  2. Commit changes: /commit
  3. Discard changes: git checkout .

Proceed after resolving.
```

**Branch already exists:**
```
Warning: Branch feat/42-add-toxicity-evaluator already exists.
Options:
  1. Switch to existing branch: git checkout feat/42-add-toxicity-evaluator
  2. Delete and recreate: git branch -D feat/42-add-toxicity-evaluator

Choose an option to proceed.
```

**Fields already set:**
```
ℹ Priority already set to "High" - keeping existing value.
ℹ Size already set to "M" - keeping existing value.
✓ Status updated to "In Progress"
```

When Priority or Size fields are already populated:
- Display current values
- Skip GraphQL update for those fields
- Only update if explicitly needed (e.g., label change detected)

## Success Criteria

- [ ] Issue exists and is open
- [ ] Branch created with correct naming
- [ ] Main branch is up to date
- [ ] Issue assigned to @dariero
- [ ] Project board updated:
  - [ ] Status set to "In Progress"
  - [ ] Priority determined and set (from labels or default)
  - [ ] Size determined and set (from labels or prompt)
- [ ] Context displayed with metadata confirmation
