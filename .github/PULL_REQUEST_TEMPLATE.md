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
.venv/bin/ruff check src/ tests/ scripts/ && .venv/bin/mypy src/ scripts/ && .venv/bin/pytest
.venv/bin/coverage run --source=src/ragaliq -m pytest tests/unit/ tests/meta/test_meta_metrics.py -o addopts="" -q && .venv/bin/coverage report
```

- [ ] `ruff check` passes
- [ ] `mypy` passes
- [ ] `pytest` passes (no regressions), on its own and in CI's test form under `coverage`

## Checklist

- [ ] New evaluators are a separate `Evaluator` subclass with an `evaluate()` method
- [ ] All public functions have type hints (required) and Google-style docstrings
- [ ] Tests added or updated in `tests/` mirroring the `src/` structure
- [ ] `CHANGELOG.md` updated for any user-facing change
- [ ] No secrets, API keys, `.env` content, or debug code (`print`, breakpoints) in this PR
