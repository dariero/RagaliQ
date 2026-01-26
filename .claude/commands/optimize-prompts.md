# optimize-prompts

## Purpose
Optimize evaluator prompts for accuracy, consistency, and speed. Apply prompt engineering best practices to improve LLM judge reliability.

## Usage
Invoke when:
- Evaluator scores are inconsistent across runs
- Judge responses are malformed or unparseable
- Evaluation is too slow (optimize for fewer tokens)
- Adding few-shot examples to improve accuracy
- Implementing chain-of-thought for complex evaluations

## Automated Steps

1. **Analyze current prompts**
   - Review prompts in `src/ragaliq/judges/prompts/`
   - Identify failure modes (parsing errors, inconsistent scores)
   - Measure baseline accuracy and latency

2. **Apply optimization techniques**
   - Add/improve few-shot examples
   - Implement structured output formats
   - Add chain-of-thought where beneficial
   - Optimize token usage

3. **Test optimizations**
   - Create evaluation dataset with known answers
   - Compare before/after accuracy
   - Measure latency changes

4. **Update prompt templates**
   - Apply changes to YAML files
   - Document optimization rationale
   - Version prompts for A/B testing

5. **Validate with integration tests**
   - Run against real API
   - Verify parsing still works
   - Check score calibration

## Domain Expertise Applied

### Prompt Engineering Techniques

**1. Structured Output Format**
```yaml
# Before: Unstructured
output_format: |
  Respond with whether the claim is supported and your confidence.

# After: JSON schema
output_format: |
  Respond with a JSON object exactly matching this schema:
  {
    "supported": boolean,
    "confidence": number between 0.0 and 1.0,
    "reasoning": "brief explanation"
  }

  Example:
  {"supported": true, "confidence": 0.95, "reasoning": "Claim directly stated in context"}
```

**2. Few-Shot Examples**
```yaml
# src/ragaliq/judges/prompts/verify_claim.yaml
examples:
  - context: "Paris is the capital of France. It has a population of 2.1 million."
    claim: "Paris is the capital of France"
    output: |
      {"supported": true, "confidence": 0.99, "reasoning": "Exact match in context"}

  - context: "Paris is the capital of France. It has a population of 2.1 million."
    claim: "Paris has 10 million people"
    output: |
      {"supported": false, "confidence": 0.95, "reasoning": "Context says 2.1 million, not 10 million"}

  - context: "Paris is a major European city known for the Eiffel Tower."
    claim: "The Eiffel Tower is 330 meters tall"
    output: |
      {"supported": false, "confidence": 0.90, "reasoning": "Height not mentioned in context"}
```

**3. Chain-of-Thought for Complex Evaluation**
```yaml
# For relevance scoring
system_prompt: |
  You are evaluating if a response answers the user's query.

  Think through this step-by-step:
  1. What is the user actually asking?
  2. What information does the response provide?
  3. Does the response address the core question?
  4. Are there any parts of the question left unanswered?

  After your analysis, provide a score from 0.0 to 1.0:
  - 0.0-0.3: Response is off-topic or doesn't address the question
  - 0.4-0.6: Response partially addresses the question
  - 0.7-0.9: Response answers the question with minor gaps
  - 1.0: Response fully and directly answers the question

user_template: |
  Query: {query}

  Response: {response}

  Analyze step-by-step, then provide your final score as JSON:
  {{"analysis": "your step-by-step thinking", "score": 0.X, "reasoning": "brief summary"}}
```

**4. Token Optimization**
```yaml
# Before: Verbose
system_prompt: |
  You are an AI assistant that helps evaluate whether claims made in
  a response are supported by the provided context. Your job is to
  carefully analyze each claim and determine if there is sufficient
  evidence in the context to support it...
  [500+ tokens]

# After: Concise
system_prompt: |
  Verify if the claim is supported by the context.
  Output JSON: {"supported": bool, "confidence": 0-1, "reasoning": "brief"}

  Rules:
  - supported=true only if context explicitly or strongly implies the claim
  - confidence reflects certainty (0.9+ for explicit matches)
  - reasoning in <10 words
```

**5. Calibration Anchors**
```yaml
# Add explicit scoring anchors
system_prompt: |
  Score relevance from 0.0 to 1.0 using these anchors:

  1.0 - "What is the capital of France?" → "The capital of France is Paris."
  0.8 - "What is the capital of France?" → "Paris is France's capital city, known for the Eiffel Tower."
  0.5 - "What is the capital of France?" → "France is a European country with many cities."
  0.2 - "What is the capital of France?" → "The Eiffel Tower is a famous landmark."
  0.0 - "What is the capital of France?" → "Python is a programming language."
```

### Optimization Workflow

**1. Create Evaluation Dataset**
```python
# tests/fixtures/prompt_eval_dataset.json
{
  "verify_claim": [
    {
      "context": "...",
      "claim": "...",
      "expected_supported": true,
      "expected_confidence_min": 0.8
    },
    ...
  ]
}
```

**2. Measure Baseline**
```python
async def evaluate_prompt_accuracy(prompt_name: str, test_cases: list) -> dict:
    judge = ClaudeJudge()
    correct = 0
    total = len(test_cases)

    for tc in test_cases:
        result = await judge.verify_claim(tc["claim"], [tc["context"]])
        if result.supported == tc["expected_supported"]:
            correct += 1

    return {"accuracy": correct / total, "total": total}
```

**3. A/B Test Prompts**
```python
async def compare_prompts(prompt_a: str, prompt_b: str, test_cases: list):
    results_a = await evaluate_with_prompt(prompt_a, test_cases)
    results_b = await evaluate_with_prompt(prompt_b, test_cases)

    print(f"Prompt A accuracy: {results_a['accuracy']:.2%}")
    print(f"Prompt B accuracy: {results_b['accuracy']:.2%}")
```

### Pitfalls to Avoid
- Don't over-optimize for one test case - use diverse examples
- Don't make prompts so long they hit token limits
- Don't remove examples that handle edge cases
- Don't forget to test JSON parsing after changes

## Interactive Prompts

**Ask for:**
- Which prompt/evaluator to optimize?
- What issues are you seeing (inconsistency, parsing, speed)?
- Do you have ground truth data for testing?
- Target latency/cost constraints?

**Suggest:**
- Specific techniques for the issue
- Example formats that work well
- Testing approach

**Validate:**
- JSON output is parseable
- Scores are calibrated correctly
- Latency is acceptable

## Success Criteria
- [ ] Prompts updated in YAML files
- [ ] Few-shot examples added/improved
- [ ] Output format is consistently parseable
- [ ] Accuracy improved or maintained
- [ ] Token usage documented
- [ ] Integration tests pass
- [ ] `make test` passes
- [ ] Optimization rationale documented
