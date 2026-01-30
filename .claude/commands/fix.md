# fix

Address feedback from PR review. Fetches review comments, guides fixes, and updates PR.

## Arguments

`$ARGUMENTS` - PR number (optional, defaults to current branch's PR)

## Process

### 1. Identify PR

If `$ARGUMENTS` provided:
```bash
gh pr view $ARGUMENTS --json number,headRefName
```

If not provided, find PR for current branch:
```bash
BRANCH=$(git branch --show-current)
gh pr list --head $BRANCH --json number,title
```

### 2. Fetch Review Feedback

```bash
gh pr view $ARGUMENTS --json reviews,comments,reviewRequests
gh api repos/{owner}/{repo}/pulls/$ARGUMENTS/comments
```

Parse and display:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Review Feedback for PR #47
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Reviewer: dariero
Status: Changes Requested

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Required Changes (3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Missing integration test
   File: (new file needed)
   Location: tests/integration/evaluators/test_toxicity.py

   Add integration test with real API call (skippable without key)

2. Score interpretation unclear
   File: src/ragaliq/evaluators/toxicity.py
   Line: 8

   Add docstring clarifying if higher score = more toxic or safer

3. Edge case handling
   File: src/ragaliq/evaluators/toxicity.py
   Line: 52

   evaluate() should handle test_case.response = "" gracefully

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Suggestions (1) - Optional
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Consider caching extracted claims for repeated evaluations
```

### 3. Check CI Status

```bash
gh pr checks $ARGUMENTS
```

Display any failures:
```
CI Status:
  ✓ lint
  ✓ typecheck
  ✗ tests - 1 failure
    test_toxicity_empty_input failed

Address CI failures as part of fixes.
```

### 4. Guide Implementation

For each required change, offer to help:
```
How would you like to proceed?

  1. Fix all issues automatically (recommended)
  2. Fix issues one by one
  3. Show me what needs to change
  4. I'll fix manually

Choose [1-4]:
```

**If automatic fix requested:**
Apply fixes using appropriate tools:
- Read the file
- Make the necessary edits
- Create/update tests as needed

**Example fix for issue #2 (docstring):**
```python
# Before
class ToxicityEvaluator(Evaluator):
    """Evaluates response toxicity."""

# After
class ToxicityEvaluator(Evaluator):
    """Evaluates response toxicity and harmful content.

    Scores responses from 0.0 to 1.0 where:
    - 0.0 = completely safe, no toxic content detected
    - 1.0 = highly toxic, multiple harmful elements found

    Higher scores indicate MORE toxic content.
    """
```

**Example fix for issue #3 (edge case):**
```python
async def evaluate(self, test_case: RAGTestCase, judge: LLMJudge) -> EvaluationResult:
    # Handle empty response
    if not test_case.response or not test_case.response.strip():
        return EvaluationResult(
            score=0.0,
            passed=True,
            reasoning="Empty response - no toxic content possible",
            metadata={"empty_input": True}
        )

    # ... rest of implementation
```

### 5. Validate Fixes

```bash
hatch run lint
hatch run test
```

**If lint fails:**
```
Lint issues found after fixes:

  src/ragaliq/evaluators/toxicity.py:15: E302 expected 2 blank lines

Run `hatch run format` to auto-fix? [Y/n]
```

**If tests fail:**
```
Test failures after fixes:

  FAILED tests/unit/evaluators/test_toxicity.py::test_empty_input
    AssertionError: Expected score 0.0, got None

Review the fix and try again.
```

### 6. Create Fix Commit

```bash
git add -A
git commit -m "$(cat <<'EOF'
[FIX #42] Address review feedback for ToxicityEvaluator

- Add docstring clarifying score interpretation (0=safe, 1=toxic)
- Handle empty response edge case gracefully
- Add integration test with API skip marker

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

### 7. Push Updates

```bash
git push
```

The PR automatically updates with new commits.

### 8. Notify Reviewer (Optional)

```
Would you like to notify the reviewer that fixes are ready? [Y/n]
```

If yes:
```bash
gh pr comment $ARGUMENTS --body "$(cat <<'EOF'
Addressed all review feedback:

- ✓ Added docstring clarifying score interpretation
- ✓ Handle empty response edge case (returns 0.0)
- ✓ Added integration test (skipped without API key)

Ready for re-review!
EOF
)"
```

### 9. Confirm Completion

```
✓ Review feedback addressed

  Fixes committed: a1b2c3d
  PR updated: https://github.com/dariero/RagaliQ/pull/47

Changes made:
  src/ragaliq/evaluators/toxicity.py (+15, -3)
  tests/integration/evaluators/test_toxicity.py (+42, new file)

PR Status: Ready for re-review

Next steps:
  • Wait for CI to pass
  • Reviewer will re-review
  • After approval, PR will be merged
  • Then run /complete-issue 42
```

## Handling Multiple Review Rounds

If this isn't the first fix:
```
This is review round #2 for PR #47.

Previous fixes:
  Round 1: Added missing type hints

Current feedback:
  ...
```

Keep fixes focused and atomic. Each round should address specific feedback.

## Error Handling

**No review feedback:**
```
PR #47 has no pending review comments.

Status: Awaiting review

Nothing to fix yet. Wait for reviewer feedback.
```

**PR already merged:**
```
PR #47 has already been merged.

Use /complete-issue 42 for cleanup instead.
```

**Conflicts after push:**
```
Push failed due to conflicts.

The PR has been updated by another collaborator.
Pull latest changes first:
  git pull origin feat/42-add-toxicity-evaluator
  # resolve conflicts
  git push
```

## Success Criteria

- [ ] All review feedback fetched
- [ ] Required changes addressed
- [ ] Tests pass after fixes
- [ ] Lint passes after fixes
- [ ] Changes committed with proper message
- [ ] PR updated via push
- [ ] Reviewer notified (optional)
