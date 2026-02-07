# ADR-007: RelevanceEvaluator as Thin Adapter over Judge

**Status:** Implemented
**Date:** 2026-02-06
**Issue:** #7 — Task 7: RelevanceEvaluator

## Context

RagaliQ needs an evaluator that measures whether an LLM response **answers the user's query**. This is the second evaluator in RagaliQ's pipeline, following FaithfulnessEvaluator (Task 6/ADR-006).

### Acceptance Criteria (from Issue #7)

| Criterion | Requirement |
|-----------|-------------|
| Score passthrough | Score from judge passes directly to result (already 0-1 normalized) |
| Reasoning included | Judge's reasoning string appears in `EvaluationResult.reasoning` |
| Quality gates pass | `hatch run lint && hatch run typecheck && hatch run test` |

### Key Constraint

The issue explicitly states: *"Simplest evaluator - wraps judge method."* Unlike FaithfulnessEvaluator (which orchestrates claim extraction + verification), RelevanceEvaluator delegates the entire assessment to `judge.evaluate_relevance()`.

## Proposed Solution

### Thin Adapter Architecture

```
RelevanceEvaluator.evaluate(test_case, judge)
    │
    └─── judge.evaluate_relevance(query, response) → JudgeResult
              │
              └─── Return EvaluationResult(
                       score = judge_result.score,
                       reasoning = judge_result.reasoning,
                       passed = is_passing(score),
                       raw_response = {score, reasoning, tokens_used}
                   )
```

### Data Flow

```
RAGTestCase ──► RelevanceEvaluator ──► LLMJudge.evaluate_relevance() ──► JudgeResult
                        │                                                      │
                        │              ◄────── score, reasoning ───────────────┘
                        │
                        └──► EvaluationResult(
                                evaluator_name="relevance",
                                score=judge_score,
                                passed=is_passing(score),
                                reasoning=judge_reasoning,
                                raw_response={score, reasoning, tokens_used}
                             )
```

### Why This Approach

The evaluator is intentionally thin because:

1. **The judge already handles the complexity** — `evaluate_relevance()` in ClaudeJudge sends a specialized prompt, parses the JSON response, clamps the score, and handles retries. Duplicating any of this in the evaluator would violate DRY.

2. **Interface conformance is the value** — The runner expects `Evaluator.evaluate()` → `EvaluationResult`. The judge returns `JudgeResult`. This evaluator bridges those two interfaces, adding threshold-based pass/fail semantics.

3. **Preserves raw judge output** — Storing `tokens_used` and full `JudgeResult` data in `raw_response` enables debugging and cost tracking without leaking judge internals through the evaluator API.

## Principles Applied

### 1. Adapter Pattern

RelevanceEvaluator adapts between two incompatible interfaces:

| `LLMJudge.evaluate_relevance()` returns | `Evaluator.evaluate()` must return |
|----------------------------------------|-----------------------------------|
| `JudgeResult(score, reasoning, tokens_used)` | `EvaluationResult(evaluator_name, score, passed, reasoning, raw_response)` |

The evaluator adds `evaluator_name`, computes `passed` via `is_passing()`, and packs the judge's full response into `raw_response`.

### 2. Single Responsibility Principle (SOLID)

- **Evaluator**: Defines *what* to evaluate (query-response relevance) and *whether* it passes (threshold comparison)
- **Judge**: Defines *how* to evaluate (LLM prompt, parsing, scoring)

The evaluator does NOT contain scoring logic, prompt templates, or API calls.

### 3. Dependency Inversion Principle (SOLID)

The evaluator depends on `LLMJudge` abstraction, not `ClaudeJudge` concrete class. This means:
- Tests use `MagicMock(spec=LLMJudge)` — no API calls needed
- Future OpenAI/local judge implementations work without evaluator changes

### 4. Open/Closed Principle (SOLID)

The Evaluator base class is open for extension (new evaluators) but closed for modification. RelevanceEvaluator extends it without changing the abstract interface.

### 5. Liskov Substitution Principle (SOLID)

RelevanceEvaluator is substitutable anywhere an `Evaluator` is expected — same `evaluate()` signature, same return type, same behavioral contract.

## Alternatives Considered

### Alternative A: Multi-Step Relevance with Aspect Decomposition (Rejected)

Decompose the query into sub-questions, evaluate each against the response, aggregate scores.

```python
# Would look like:
aspects = await judge.extract_query_aspects(query)
for aspect in aspects:
    score = await judge.score_aspect_coverage(aspect, response)
final_score = aggregate(scores)
```

**Why rejected:**
- The issue spec explicitly says "wraps judge method" — this adds unnecessary complexity
- Would require new `LLMJudge` abstract methods not yet defined
- Over-engineering for the current phase; can be added as a separate evaluator later (e.g., `DetailedRelevanceEvaluator`)
- More LLM calls = higher cost and latency for marginal benefit at this stage

### Alternative B: Reuse `evaluate_faithfulness()` with Query as Context (Rejected)

Treat the query as a single-item context list and reuse the faithfulness pipeline.

```python
result = await judge.evaluate_faithfulness(response, context=[query])
```

**Why rejected:**
- **Semantic mismatch**: Faithfulness measures grounding in *documents*, not *query answering*
- Would produce misleading scores — a response can be faithful to context but irrelevant to the query
- Conflates two orthogonal quality dimensions
- The judge already has a dedicated `evaluate_relevance()` method with a specialized prompt designed for query-response matching

## Implementation Details

### Files Created/Modified

| File | Change | Description |
|------|--------|-------------|
| `src/ragaliq/evaluators/relevance.py` | **Created** | RelevanceEvaluator implementation (thin adapter) |
| `src/ragaliq/evaluators/__init__.py` | Modified | Exported `RelevanceEvaluator` |
| `tests/unit/test_relevance_evaluator.py` | **Created** | 18 unit tests |

### Test Coverage

**18 unit tests** organized by concern:
1. **Attributes** (5 tests): name, description, threshold, custom threshold, repr
2. **Score Passthrough** (3 tests): High/medium/low scores pass through correctly
3. **Threshold Logic** (3 tests): Pass/fail based on configurable threshold
4. **Raw Response** (3 tests): Judge metadata preserved in raw_response
5. **Result Structure** (2 tests): Returns correct type, evaluator_name set
6. **Judge Interaction** (2 tests): Calls correct judge method, doesn't call others

### What We're NOT Testing

- Judge's actual scoring logic (that's `test_claude_judge.py`)
- Prompt quality (integration/E2E concern)
- API retries and error handling (judge's responsibility)

The tests focus on the adapter's responsibility: correctly translating `JudgeResult` → `EvaluationResult`.

## Future Considerations

1. **Weighted relevance**: Future evaluators could assess partial relevance (e.g., response addresses 3 of 5 query aspects)
2. **Context-aware relevance**: A variant could factor in whether the *context* was relevant, not just the response
3. **Comparative evaluation**: Rank multiple responses by relevance to the same query
4. **Query decomposition**: Break complex queries into sub-questions and assess coverage individually
