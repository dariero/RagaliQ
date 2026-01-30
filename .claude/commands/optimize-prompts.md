# optimize-prompts

Improve evaluator prompt accuracy and consistency.

## When to Use

- Evaluator scores are inconsistent across runs
- Judge responses are malformed/unparseable
- Need to calibrate scoring thresholds

## Workflow

### 1. Create Evaluation Dataset

```python
# tests/fixtures/prompt_eval_{prompt_name}.json
[
    {
        "context": "...",
        "claim": "...",
        "expected_supported": true,
        "expected_confidence_min": 0.8
    }
]
```

Include edge cases: obvious true, obvious false, ambiguous.

### 2. Measure Baseline

```python
async def measure_accuracy(prompt_file: str, test_cases: list) -> dict:
    judge = ClaudeJudge()
    correct = 0
    for tc in test_cases:
        result = await judge.verify_claim(tc["claim"], [tc["context"]])
        if result.supported == tc["expected_supported"]:
            correct += 1
    return {"accuracy": correct / len(test_cases)}
```

### 3. Apply Changes to Prompts

Location: `src/ragaliq/judges/prompts/*.yaml`

Focus on:
- Structured output format (JSON schema)
- Few-shot examples for edge cases
- Explicit scoring anchors

### 4. A/B Test

```python
results_before = await measure_accuracy("v1.yaml", test_cases)
results_after = await measure_accuracy("v2.yaml", test_cases)
# Compare accuracy, check for regressions
```

### 5. Validate

```bash
hatch run test tests/integration/
```

Ensure JSON parsing still works after prompt changes.

## Success Criteria

- [ ] Evaluation dataset created
- [ ] Baseline measured
- [ ] Accuracy improved or maintained
- [ ] No parsing regressions
