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

### 5. Update Project Board

Move the issue to "🔄 In Progress" column.

```bash
# Get project item ID
gh api graphql -f query='
  query($owner: String!, $number: Int!) {
    user(login: $owner) {
      projectV2(number: $number) {
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
        field(name: "Status") {
          ... on ProjectV2SingleSelectField {
            id
            options {
              id
              name
            }
          }
        }
      }
    }
  }
' -f owner="dariero" -F number=2

# Update status to "In Progress"
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
' -f project="<PROJECT_ID>" -f item="<ITEM_ID>" -f field="<FIELD_ID>" -f value="<IN_PROGRESS_OPTION_ID>"
```

### 6. Summarize and Prepare

Display issue context:
```
✓ Switched to branch: feat/42-add-toxicity-evaluator
✓ Issue #42 moved to "In Progress"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Issue #42: [FEAT] Add ToxicityEvaluator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

## Success Criteria

- [ ] Issue exists and is open
- [ ] Branch created with correct naming
- [ ] Main branch is up to date
- [ ] Project board updated to "In Progress"
- [ ] Context displayed to developer
