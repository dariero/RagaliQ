# new-evaluator

## Purpose
Scaffold a new LLM response evaluator for RagaliQ. Evaluators assess RAG responses against metrics like faithfulness, relevance, hallucination, or custom domain-specific criteria.

## Usage
Invoke when:
- Adding a new built-in evaluator (e.g., toxicity, coherence, completeness)
- Creating custom domain-specific evaluators (e.g., legal-compliance, medical-accuracy)
- Implementing metrics from research papers (e.g., RAGAS, DeepEval patterns)

## Automated Steps

1. **Analyze existing evaluator patterns**
   - Read `src/ragaliq/evaluators/` for established patterns
   - Review `src/ragaliq/core/evaluator.py` for base class interface
   - Check registry at `src/ragaliq/evaluators/registry.py`

2. **Generate evaluator implementation**
   ```
   src/ragaliq/evaluators/{name}.py
   ```
   - Inherit from `Evaluator` base class
   - Implement async `evaluate(test_case, judge)` method
   - Define scoring algorithm (0.0-1.0 scale)
   - Store detailed metadata for debugging

3. **Register evaluator**
   - Add `@register_evaluator("{name}")` decorator
   - Update `src/ragaliq/evaluators/__init__.py` exports

4. **Create comprehensive tests**
   ```
   tests/unit/test_{name}_evaluator.py
   ```
   - Mock judge responses for deterministic testing
   - Test edge cases: empty input, perfect score, complete failure
   - Test score boundary conditions

5. **Add integration test**
   ```
   tests/integration/test_{name}_evaluator.py
   ```
   - Real API test with `@pytest.mark.skipif(not ANTHROPIC_API_KEY)`

6. **Update documentation**
   - Add to evaluator table in README.md
   - Document scoring rubric and interpretation

## Domain Expertise Applied

### LLM Evaluation Best Practices
- **Score calibration**: Ensure 0.7 threshold is meaningful for this metric
- **Reasoning transparency**: Always include human-readable explanation
- **Claim-level granularity**: Break down evaluation into verifiable units
- **Confidence tracking**: Report confidence alongside scores

### Common Evaluator Patterns
```python
class {Name}Evaluator(Evaluator):
    name: str = "{name}"
    description: str = "{Description of what this measures}"

    async def evaluate(
        self,
        test_case: RAGTestCase,
        judge: LLMJudge
    ) -> EvaluationResult:
        # 1. Extract evaluation units (claims, facts, etc.)
        # 2. Score each unit via judge
        # 3. Aggregate scores with weighting
        # 4. Generate reasoning summary
        return EvaluationResult(
            score=aggregated_score,
            reasoning=explanation,
            metadata={"details": unit_scores}
        )
```

### Pitfalls to Avoid
- Don't hardcode thresholds - make configurable
- Don't ignore empty/edge cases - handle gracefully
- Don't lose claim-level detail - store in metadata
- Don't block on single slow judge call - use asyncio.gather

## Interactive Prompts

**Ask for:**
- Evaluator name (snake_case): `{name}`
- What does this evaluator measure?
- Scoring algorithm (claim-based, continuous, binary)?
- Required RAGTestCase fields (context, expected_facts, etc.)?
- Default threshold (0.7 recommended)?

**Suggest:**
- Similar existing evaluators to reference
- Appropriate judge methods to use
- Metadata structure for debugging

**Validate:**
- Score interpretation makes sense (higher = better)
- Threshold is calibrated for this metric
- Edge cases are handled

## Success Criteria
- [ ] Evaluator class created with full type hints
- [ ] Registered in evaluator registry
- [ ] Unit tests with >80% coverage
- [ ] Integration test (skipped without API key)
- [ ] `make test && make typecheck` passes
- [ ] Docstrings explain scoring algorithm
- [ ] README.md updated with new evaluator
