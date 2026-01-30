# check

Pre-push quality validation.

## Arguments

`$ARGUMENTS` - Optional test path or pytest flags

## Process

```bash
hatch run lint
hatch run typecheck
hatch run test $ARGUMENTS
```

## Output

**Pass:**
```
✓ Lint passed
✓ Type check passed
✓ Tests passed (N passed)
Ready to push. Use /pr to create pull request.
```

**Fail:**
```
✗ [stage] failed
[error details]
Fix issues and re-run /check.
```

If lint fails, offer `hatch run format` to auto-fix.

## Success Criteria

- [ ] All three stages pass
- [ ] Coverage ≥ 80%
