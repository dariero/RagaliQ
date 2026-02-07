# ADR-008: HallucinationEvaluator Implementation

**Status:** Accepted
**Date:** 2026-02-07
**Issue:** #8 — Task 7: HallucinationEvaluator

## Context

RagaliQ needs a HallucinationEvaluator to detect made-up facts in RAG responses that aren't grounded in the provided context. This is conceptually adjacent to the existing FaithfulnessEvaluator but serves a distinct quality dimension:

- **Faithfulness** answers: "Is the response grounded in context?" (positive framing)
- **Hallucination** answers: "Did the model fabricate information?" (negative framing)

The issue specifies: extract claims, verify against context, flag claims where `supported=false` OR `confidence < 0.8`, and compute `score = 1.0 - (hallucinated / total)`.

### The Confidence Question

The existing `ClaimVerdict` model returns a three-way `verdict` (SUPPORTED / CONTRADICTED / NOT_ENOUGH_INFO) and `evidence`, but **no confidence score**. The issue mentions `confidence < 0.8` as a per-claim criterion. Two interpretations exist:

1. **Per-claim LLM confidence** — requires adding a `confidence: float` field to `ClaimVerdict`, updating the prompt template, and modifying `ClaudeJudge.verify_claim()`. This is a cross-cutting interface change affecting the judge layer.
2. **Evaluator-level threshold** — the "0.8 confidence" refers to the evaluator's pass/fail threshold (the `threshold` attribute), which is already 0.8 (stricter than faithfulness's 0.7).

We choose interpretation **2** for this iteration because:
- The CLAUDE.md constraint says "NEVER change existing architectural patterns without explicit approval"
- Adding `confidence` to `ClaimVerdict` modifies the judge contract that FaithfulnessEvaluator also depends on
- The three-way verdict already captures the essential semantic: SUPPORTED claims are grounded, everything else is a hallucination
- Per-claim confidence can be added as a backwards-compatible enhancement in a future issue

## Proposed Solution

### Claim-Based Evaluation (same decomposition as FaithfulnessEvaluator)

```
1. Extract atomic claims from response      → judge.extract_claims()
2. Verify each claim against context         → judge.verify_claim()
3. Classify: CONTRADICTED or NOT_ENOUGH_INFO → hallucinated
4. Score = 1.0 - (hallucinated_count / total_claims)
5. Store hallucinated_claims list in raw_response metadata
```

### Key Structural Differences from FaithfulnessEvaluator

| Dimension | FaithfulnessEvaluator | HallucinationEvaluator |
|-----------|----------------------|----------------------|
| Default threshold | 0.7 | **0.8** (stricter) |
| Score formula | `supported / total` | `1.0 - (hallucinated / total)` |
| Metadata focus | `supported_claims` count | `hallucinated_claims` list + count |
| Reasoning framing | "X of Y claims supported" | "Found X hallucinated claims" |
| Empty claims | 1.0 (vacuously faithful) | 1.0 (no hallucinations) |

While the score formulas are mathematically equivalent (both yield the same numeric value), the **metadata and framing** differ significantly. The hallucination evaluator explicitly identifies and lists which claims are fabricated, making it actionable for debugging — users can see exactly which statements their RAG pipeline invented.

### Module Structure

- `src/ragaliq/evaluators/hallucination.py` — evaluator implementation
- `tests/unit/test_hallucination_evaluator.py` — comprehensive test suite
- Update `src/ragaliq/evaluators/__init__.py` — package export

## Principles Applied

1. **Open/Closed Principle (SOLID):** New evaluator extends the `Evaluator` ABC without modifying existing code. The judge interface remains untouched.
2. **Dependency Inversion (SOLID):** HallucinationEvaluator depends on the `LLMJudge` abstraction, not on `ClaudeJudge` directly. Judge is injected via method parameter.
3. **Template Method Pattern:** The base `Evaluator` class provides `is_passing()` and `__repr__()`; the subclass implements `evaluate()`.
4. **Single Responsibility:** This evaluator does one thing — detect hallucinated claims. It doesn't try to assess relevance or faithfulness.
5. **DRY (pragmatic):** We reuse the existing `judge.extract_claims()` and `judge.verify_claim()` rather than creating parallel judge methods. The evaluator-level interpretation of verdicts is what differentiates it.

## Alternatives Considered

### Alternative A: Add `evaluate_hallucination()` to LLMJudge (Rejected)

A dedicated judge method (like `evaluate_relevance()`) that returns a single holistic score.

**Why rejected:**
- Breaks the claim-level decomposition that makes hallucination evaluations debuggable and actionable
- Adds a new abstract method that all judge implementations must support
- Loses granularity — users can't see *which* claims are hallucinated
- The issue explicitly asks for claim-level detection with `hallucinated_claims` in metadata

### Alternative B: Extend ClaimVerdict with `confidence: float` (Deferred)

Add a confidence score to each verdict so the evaluator can flag "SUPPORTED but low confidence" claims.

**Why deferred (not rejected):**
- Requires modifying the judge interface contract (`ClaimVerdict` model)
- Requires updating the `verify_claim` prompt template to request confidence
- All existing judge implementations and tests would need updates
- The three-way verdict already captures the core semantic for this iteration
- Can be added as a backwards-compatible enhancement (optional field with default 1.0)
- Tracked as a potential follow-up: "Add per-claim confidence scoring to judge interface"
