# new-evaluator

Scaffold a new LLM response evaluator.

## Files to Create

```
src/ragaliq/evaluators/{name}.py
tests/unit/evaluators/test_{name}.py
tests/integration/evaluators/test_{name}.py
```

## Required Pattern

```python
from ragaliq.core import Evaluator, EvaluationResult, RAGTestCase
from ragaliq.judges import LLMJudge
from ragaliq.evaluators.registry import register_evaluator

@register_evaluator("{name}")
class {Name}Evaluator(Evaluator):
    """One-line description."""

    name: str = "{name}"

    async def evaluate(
        self,
        test_case: RAGTestCase,
        judge: LLMJudge
    ) -> EvaluationResult:
        # Handle empty input
        if not test_case.response:
            return EvaluationResult(score=0.0, reasoning="Empty response")

        # Implementation: extract units, score via judge, aggregate

        return EvaluationResult(
            score=float,        # 0.0-1.0
            passed=bool,        # score >= threshold
            reasoning=str,      # Human-readable explanation
            metadata=dict       # Debugging details
        )
```

## Requirements

- Score must be 0.0-1.0
- Document score interpretation (higher = better? worse?)
- Handle edge cases: empty input, perfect score, complete failure
- Unit tests mock the judge
- Integration test uses `@pytest.mark.skipif(not ANTHROPIC_API_KEY)`

## Update Exports

Add to `src/ragaliq/evaluators/__init__.py`

## Verify

```bash
hatch run test tests/unit/evaluators/test_{name}.py
hatch run typecheck
```
