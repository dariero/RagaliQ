# ADR-006: FaithfulnessEvaluator with Claim-Level Decomposition

**Status:** Implemented
**Date:** 2026-02-05
**Issue:** #6 — Task 5: FaithfulnessEvaluator

## Context

RagaliQ needs an evaluator that measures whether an LLM response is **faithful** to the provided context — meaning every claim is grounded in the context with no hallucinations or fabricated information.

### Acceptance Criteria (from Issue #6)

| Scenario | Expected Score |
|----------|---------------|
| All claims supported | 1.0 |
| Half supported | 0.5 |
| No claims (empty) | 1.0 (vacuously faithful) |
| All unsupported | 0.0 |

### Problem Statement

How should faithfulness be assessed? We need a mechanism that:
- Produces granular 0.0-1.0 scores (not binary pass/fail)
- Provides actionable debugging information (which claims failed?)
- Integrates cleanly with the existing `LLMJudge` abstraction
- Follows the evaluator pattern established by the base `Evaluator` class

## Proposed Solution

### Claim-Based Decomposition

```
FaithfulnessEvaluator.evaluate(test_case, judge)
    │
    ├─── 1. judge.extract_claims(response) → ClaimsResult
    │         └─── Returns: ["claim 1", "claim 2", ...]
    │
    ├─── 2. For each claim:
    │         judge.verify_claim(claim, context) → ClaimVerdict
    │         └─── Returns: {verdict: SUPPORTED|CONTRADICTED|NOT_ENOUGH_INFO, evidence: "..."}
    │
    └─── 3. Calculate: score = supported_count / total_claims
              └─── Returns: EvaluationResult with claim-level metadata
```

### Why Claim-Level Decomposition

| Criterion | Claim-Level Approach | Holistic Alternative |
|-----------|---------------------|----------------------|
| **Debuggability** | See exactly which claims failed | Black-box score only |
| **Actionability** | Users can fix specific sentences | Must rewrite entire response |
| **Granularity** | Fine-grained 0.0-1.0 scoring | Often binary-feeling |
| **Metadata** | Detailed claim breakdown in `raw_response` | None |
| **Testability** | Each step independently testable | Single integration test |

### New Judge Methods Required

This evaluator requires two new abstract methods in `LLMJudge`:

1. **`extract_claims(response: str) -> ClaimsResult`**
   Decomposes a response into atomic verifiable claims.

2. **`verify_claim(claim: str, context: list[str]) -> ClaimVerdict`**
   Checks if a single claim is supported, contradicted, or unverifiable by the context.

### New Data Models

```python
class ClaimsResult(BaseModel):
    claims: list[str]
    tokens_used: int

class ClaimVerdict(BaseModel):
    verdict: Literal["SUPPORTED", "CONTRADICTED", "NOT_ENOUGH_INFO"]
    evidence: str
```

## Principles Applied

1. **Single Responsibility (SOLID):** Evaluator orchestrates the workflow; Judge encapsulates LLM calls.
2. **Dependency Inversion (SOLID):** Evaluator depends on `LLMJudge` abstraction, not `ClaudeJudge` concrete class.
3. **Strategy Pattern:** `LLMJudge` acts as the strategy; evaluators work with any judge implementation.
4. **Dependency Injection:** Judge is injected via method parameter, not constructed internally.
5. **Template Method:** Base `Evaluator` provides `is_passing()` helper; subclasses implement `evaluate()`.
6. **Value Objects (Pydantic):** All data models are immutable frozen Pydantic models.

## Alternatives Considered

### Alternative A: Holistic Evaluation (Rejected)

Use the existing `judge.evaluate_faithfulness()` method that returns a single holistic score.

**Why rejected:**
- No claim-level metadata (`raw_response["claims"]` was required by issue spec)
- Users can't understand why they got 0.6 vs 0.8
- Violates issue specification requiring claim extraction algorithm
- Less testable — can only verify final score
- Not actionable for debugging

### Alternative B: Evaluator Calls LLM Directly (Rejected)

Have the evaluator make direct API calls using prompt templates, bypassing the judge abstraction.

**Why rejected:**
- Violates **Dependency Inversion Principle** — evaluator depends on concrete Claude, not abstraction
- Untestable without mocking Anthropic client
- Breaks architecture — judge exists to encapsulate LLM calls
- Duplicate API key management
- Can't swap LLM providers (tightly coupled to Claude)

## Implementation Details

### Files Created/Modified

| File | Change | Description |
|------|--------|-------------|
| `src/ragaliq/judges/base.py` | Modified | Added `ClaimsResult`, `ClaimVerdict` models and abstract methods |
| `src/ragaliq/judges/claude.py` | Modified | Implemented `extract_claims()`, `verify_claim()` using YAML templates |
| `src/ragaliq/judges/__init__.py` | Modified | Exported new models |
| `src/ragaliq/judges/prompts/extract_claims.yaml` | Created | Prompt template for claim extraction |
| `src/ragaliq/judges/prompts/verify_claim.yaml` | Created | Prompt template for claim verification |
| `src/ragaliq/evaluators/faithfulness.py` | **Created** | Main evaluator implementation |
| `src/ragaliq/evaluators/__init__.py` | Modified | Exported `FaithfulnessEvaluator` |
| `tests/unit/test_faithfulness_evaluator.py` | **Created** | 21 unit tests |
| `tests/unit/test_claude_judge.py` | Modified | Added tests for new judge methods |

### Test Coverage

**21 unit tests** organized by concern:
1. Attributes (5 tests): name, description, threshold, custom threshold, repr
2. Acceptance Criteria (4 tests): all supported, half supported, no claims, all unsupported
3. Verdict Handling (3 tests): contradicted, not_enough_info, mixed verdicts
4. Metadata (2 tests): claim details, summary stats
5. Result Structure (3 tests): EvaluationResult type, evaluator name, reasoning
6. Edge Cases (4 tests): single claim, precision with many claims, threshold logic

### Edge Cases Handled

| Edge Case | Behavior |
|-----------|----------|
| Empty response | Returns score 1.0 (vacuously faithful, no claims to verify) |
| Empty context | `verify_claim` returns `NOT_ENOUGH_INFO` |
| Whitespace-only response | Returns score 1.0 (no claims extracted) |
| Invalid verdict from LLM | `JudgeResponseError` raised |
| Lowercase verdict from LLM | Normalized to uppercase |

## Future Considerations

1. **Performance**: For responses with many claims (100+), consider parallel verification or batching
2. **Caching**: Claim extraction could be cached if same response is evaluated multiple times
3. **Partial Support**: Future enhancement could add `PARTIALLY_SUPPORTED` verdict
4. **Scoring Variants**: Could offer weighted scoring (contradicted = -1, not_enough_info = 0, supported = 1)
5. **Confidence Scores**: Add per-claim confidence to `ClaimVerdict` for more nuanced assessment
