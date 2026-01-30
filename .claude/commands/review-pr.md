# review-pr

Review a pull request against RagaliQ standards.

## Arguments

`$ARGUMENTS` - PR number or URL (required)

## Process

1. Fetch PR details and diff:
   ```bash
   gh pr view $ARGUMENTS --json title,body,files
   gh pr diff $ARGUMENTS
   gh pr checks $ARGUMENTS
   ```

2. Get linked issue requirements:
   ```bash
   gh issue view <issue-number> --json title,body
   ```

3. **Apply RagaliQ-specific checklist:**

### Evaluators
- [ ] Inherits from `Evaluator` base class
- [ ] `evaluate()` is async
- [ ] Returns score in 0.0-1.0 range
- [ ] Score interpretation documented (higher = better or worse?)
- [ ] Registered with `@register_evaluator`
- [ ] Handles empty input gracefully

### Judges
- [ ] All API calls are async
- [ ] Retry logic with exponential backoff
- [ ] Token usage tracked
- [ ] Handles malformed JSON responses

### General
- [ ] Type hints on public APIs
- [ ] No secrets exposed
- [ ] Tests cover new code

4. Submit review:

   **Approve:**
   ```bash
   gh pr review $ARGUMENTS --approve
   gh pr merge $ARGUMENTS --squash --delete-branch
   ```

   **Request changes:**
   ```bash
   gh pr review $ARGUMENTS --request-changes --body "feedback"
   ```
   Direct author to `/fix`
