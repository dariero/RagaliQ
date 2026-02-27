## Summary

<!-- What does this PR do? 1-3 sentences. -->

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor
- [ ] Documentation
- [ ] CI/CD / infrastructure

## Related Issue

Closes #

## Quality Gates

All three must pass before merging:

```bash
hatch run lint       # ruff — style and imports
hatch run typecheck  # mypy strict
hatch run test       # pytest + coverage
```

- [ ] `hatch run lint` passes
- [ ] `hatch run typecheck` passes
- [ ] `hatch run test` passes (no regressions)

## Checklist

- [ ] New evaluators are a separate `Evaluator` subclass with an `evaluate()` method
- [ ] All public functions have type hints (required) and Google-style docstrings
- [ ] Tests added or updated in `tests/` mirroring the `src/` structure
- [ ] `CHANGELOG.md` updated for any user-facing change
- [ ] No secrets, API keys, `.env` content, or debug code (`print`, breakpoints) in this PR
