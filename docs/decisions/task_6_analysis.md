# Task 6: FaithfulnessEvaluator - Design Decision Record

**Issue**: [#6 - Task 5: FaithfulnessEvaluator](https://github.com/dariero/RagaliQ/issues/6)
**Date**: 2026-02-05
**Status**: Implemented

---

## 1. Problem Statement

Implement an evaluator that measures whether an LLM response is **faithful** to the provided context - meaning every claim is grounded in the context with no hallucinations or fabricated information.

### Acceptance Criteria (from Issue #6)

| Scenario | Expected Score |
|----------|---------------|
| All claims supported | 1.0 |
| Half supported | 0.5 |
| No claims (empty) | 1.0 (vacuously faithful) |
| All unsupported | 0.0 |

---

## 2. Chosen Approach: Claim-Level Decomposition

### Architecture

```
FaithfulnessEvaluator
    │
    ├─── 1. judge.extract_claims(response) → ClaimsResult
    │         └─── Returns: ["claim 1", "claim 2", ...]
    │
    ├─── 2. For each claim:
    │         judge.verify_claim(claim, context) → ClaimVerdict
    │         └─── Returns: {verdict: SUPPORTED|CONTRADICTED|NOT_ENOUGH_INFO, evidence: "..."}
    │
    └─── 3. Calculate: score = supported_count / total_claims
              └─── Returns: EvaluationResult with metadata
```

### Why This Approach

| Criterion | Claim-Level | Holistic Alternative |
|-----------|-------------|----------------------|
| **Debuggability** | See exactly which claims failed | Black-box score only |
| **Actionability** | Users can fix specific sentences | Must rewrite entire response |
| **Granularity** | Fine-grained 0.0-1.0 scoring | Often binary-feeling |
| **Metadata** | Detailed claim breakdown in `raw_response` | None |
| **Testability** | Each step independently testable | Single integration test |

---

## 3. Alternatives Considered

### Alternative A: Holistic Evaluation (Rejected)

**Approach**: Use existing `judge.evaluate_faithfulness()` method that returns a single score.

```python
# Would have been simple:
result = await judge.evaluate_faithfulness(response, context)
return EvaluationResult(score=result.score, ...)
```

**Why Rejected**:
- No claim-level metadata (`metadata["claims"]` was required per issue spec)
- Users can't understand why they got 0.6 vs 0.8
- Violates issue specification requiring claim extraction algorithm
- Less testable - can only verify final score

### Alternative B: Evaluator Calls Claude Directly (Rejected)

**Approach**: Have the evaluator make direct API calls using prompt templates, bypassing the judge abstraction.

```python
class FaithfulnessEvaluator(Evaluator):
    def __init__(self, api_key: str):
        self.client = AsyncAnthropic(api_key=api_key)

    async def evaluate(self, test_case, judge):  # judge ignored
        # Direct API calls here
```

**Why Rejected**:
- Violates **Dependency Inversion Principle** - evaluator depends on concrete Claude, not abstraction
- Untestable without mocking Anthropic client
- Breaks architecture - judge exists to encapsulate LLM calls
- Duplicate API key management
- Can't swap LLM providers (tightly coupled to Claude)

---

## 4. Design Patterns Applied

### 4.1 Strategy Pattern

The `LLMJudge` interface acts as the strategy. Evaluators work with any judge implementation.

```python
class LLMJudge(ABC):
    @abstractmethod
    async def extract_claims(self, response: str) -> ClaimsResult: ...

    @abstractmethod
    async def verify_claim(self, claim: str, context: list[str]) -> ClaimVerdict: ...
```

### 4.2 Dependency Injection

Judge is injected into `evaluate()` method, not constructed internally:

```python
async def evaluate(self, test_case: RAGTestCase, judge: LLMJudge) -> EvaluationResult:
    # Judge is provided, not created
    claims = await judge.extract_claims(test_case.response)
```

### 4.3 Value Objects (Pydantic)

All data models are immutable Pydantic models:

```python
class ClaimVerdict(BaseModel):
    verdict: Literal["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"]
    evidence: str = ""
    model_config = {"frozen": True}
```

### 4.4 Template Method

Base `Evaluator` provides `is_passing()` helper; subclasses implement `evaluate()`:

```python
class Evaluator(ABC):
    def is_passing(self, score: float) -> bool:
        return score >= self.threshold  # Provided by base

    @abstractmethod
    async def evaluate(self, ...) -> EvaluationResult:
        pass  # Implemented by subclass
```

---

## 5. SOLID Principles

| Principle | Application |
|-----------|-------------|
| **S**ingle Responsibility | Evaluator only orchestrates; Judge handles LLM calls |
| **O**pen/Closed | New verdict types don't require Evaluator changes |
| **L**iskov Substitution | Any `LLMJudge` subclass works with FaithfulnessEvaluator |
| **I**nterface Segregation | Judge methods are focused, not bloated |
| **D**ependency Inversion | Evaluator depends on `LLMJudge` abstraction, not `ClaudeJudge` |

---

## 6. Test-Driven Development

### Test Pyramid

```
        /\
       /  \  Integration (existing runner tests)
      /────\
     /      \
    /────────\  Unit Tests (21 new tests)
   /          \ - Evaluator with mock judge
  /────────────\ - Each edge case covered
 /              \
```

### Test Categories (21 tests)

1. **Attributes** (5 tests): name, description, threshold, custom threshold, repr
2. **Acceptance Criteria** (4 tests): all supported, half supported, no claims, all unsupported
3. **Verdict Handling** (3 tests): contradicted, not_enough_info, mixed verdicts
4. **Metadata** (2 tests): claim details, summary stats
5. **Result Structure** (3 tests): EvaluationResult type, evaluator name, reasoning
6. **Edge Cases** (4 tests): single claim, precision with many claims, threshold logic

### Mock Strategy

```python
@pytest.fixture
def mock_judge():
    judge = MagicMock(spec=LLMJudge)
    judge.extract_claims = AsyncMock(return_value=ClaimsResult(claims=[...]))
    judge.verify_claim = AsyncMock(side_effect=[...])
    return judge
```

---

## 7. Files Changed

| File | Change | Description |
|------|--------|-------------|
| `src/ragaliq/judges/base.py` | Modified | Added `ClaimsResult`, `ClaimVerdict` models and abstract methods |
| `src/ragaliq/judges/claude.py` | Modified | Implemented `extract_claims()`, `verify_claim()` using YAML templates |
| `src/ragaliq/judges/__init__.py` | Modified | Exported new models |
| `src/ragaliq/evaluators/faithfulness.py` | **Created** | Main evaluator implementation |
| `src/ragaliq/evaluators/__init__.py` | Modified | Exported `FaithfulnessEvaluator` |
| `tests/unit/test_faithfulness_evaluator.py` | **Created** | 21 unit tests |
| `tests/unit/test_claude_judge.py` | Modified | Added tests for new judge methods |
| `tests/unit/test_judges.py` | Modified | Updated mock implementations |

---

## 8. API Surface

### New Models

```python
from ragaliq.judges import ClaimsResult, ClaimVerdict

# Extract claims
claims_result = await judge.extract_claims("Paris is the capital of France.")
# ClaimsResult(claims=["Paris is the capital of France"], tokens_used=50)

# Verify claim
verdict = await judge.verify_claim(
    claim="Paris is the capital of France",
    context=["France is a country. Its capital is Paris."]
)
# ClaimVerdict(verdict="SUPPORTED", evidence="Context states this")
```

### FaithfulnessEvaluator Usage

```python
from ragaliq.evaluators import FaithfulnessEvaluator

evaluator = FaithfulnessEvaluator(threshold=0.8)
result = await evaluator.evaluate(test_case, judge)

print(f"Score: {result.score}")  # 0.75
print(f"Passed: {result.passed}")  # False (0.75 < 0.8)

# Inspect individual claims
for claim in result.raw_response["claims"]:
    print(f"{claim['verdict']}: {claim['claim']}")
```

---

## 9. Edge Cases Handled

| Edge Case | Behavior |
|-----------|----------|
| Empty response | Returns score 1.0 (vacuously faithful) |
| Empty context | `verify_claim` returns `NOT_ENOUGH_INFO` |
| Whitespace-only response | Returns score 1.0 (no claims) |
| Invalid verdict from LLM | `JudgeResponseError` raised |
| Lowercase verdict from LLM | Normalized to uppercase |

---

## 10. Future Considerations

1. **Performance**: For responses with many claims (100+), consider parallel verification or batching
2. **Caching**: Extract claims could be cached if same response evaluated multiple times
3. **Partial Support**: Future enhancement could track `PARTIALLY_SUPPORTED` verdict
4. **Scoring Variants**: Could offer weighted scoring (contradicted = -1, not_enough_info = 0, supported = 1)

---

## 11. Quality Gates

```bash
make test       # 149 passed, 1 skipped
make lint       # All checks passed
make typecheck  # All checks passed
```
