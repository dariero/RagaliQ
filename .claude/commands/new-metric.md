# new-metric

## Purpose
Implement a new evaluation metric for RagaliQ. Metrics are the mathematical/statistical calculations that don't require LLM calls - traditional NLP metrics like BLEU, ROUGE, or custom domain-specific calculations.

## Usage
Invoke when:
- Adding traditional NLP metrics (BLEU, ROUGE, BERTScore)
- Implementing lexical overlap metrics (Jaccard, cosine similarity)
- Creating custom scoring functions (keyword match, format validation)
- Building composite metrics from multiple signals

## Automated Steps

1. **Analyze metric requirements**
   - Determine inputs needed (response, reference, context?)
   - Research standard implementations if applicable
   - Check if third-party libraries exist (rouge-score, sacrebleu)

2. **Create metric module**
   ```
   src/ragaliq/metrics/{name}.py
   ```
   - Pure function or class-based implementation
   - No LLM calls - deterministic calculation
   - Normalized output (0.0-1.0 scale preferred)

3. **Create evaluator wrapper** (optional)
   ```
   src/ragaliq/evaluators/{name}_evaluator.py
   ```
   - Wraps metric for use in RagaliQ pipeline
   - Handles RAGTestCase unpacking

4. **Add comprehensive tests**
   ```
   tests/unit/test_{name}_metric.py
   ```
   - Test with known inputs/outputs
   - Test edge cases (empty strings, identical strings)
   - Test score bounds

5. **Add benchmarks** (for expensive metrics)
   ```
   tests/benchmarks/bench_{name}.py
   ```

## Domain Expertise Applied

### Common Metric Implementations

**1. N-gram Overlap (BLEU-style)**
```python
def bleu_score(
    candidate: str,
    references: list[str],
    max_n: int = 4,
    weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
) -> float:
    """Calculate BLEU score for candidate against references."""
    from collections import Counter

    def ngrams(text: str, n: int) -> Counter:
        tokens = text.lower().split()
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

    # Calculate precision for each n-gram level
    precisions = []
    for n in range(1, max_n + 1):
        candidate_ngrams = ngrams(candidate, n)
        max_counts = Counter()
        for ref in references:
            ref_ngrams = ngrams(ref, n)
            for ng in candidate_ngrams:
                max_counts[ng] = max(max_counts[ng], ref_ngrams[ng])

        clipped = sum(min(candidate_ngrams[ng], max_counts[ng]) for ng in candidate_ngrams)
        total = sum(candidate_ngrams.values())
        precisions.append(clipped / total if total > 0 else 0)

    # Geometric mean with weights
    import math
    score = math.exp(sum(w * math.log(p + 1e-10) for w, p in zip(weights, precisions)))
    return min(1.0, score)
```

**2. Longest Common Subsequence (ROUGE-L)**
```python
def rouge_l_score(candidate: str, reference: str) -> float:
    """Calculate ROUGE-L F1 score."""
    def lcs_length(x: list[str], y: list[str]) -> int:
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()
    lcs = lcs_length(cand_tokens, ref_tokens)

    precision = lcs / len(cand_tokens) if cand_tokens else 0
    recall = lcs / len(ref_tokens) if ref_tokens else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

**3. Semantic Similarity (with sentence-transformers)**
```python
def semantic_similarity(
    text1: str,
    text2: str,
    model_name: str = "all-MiniLM-L6-v2"
) -> float:
    """Calculate cosine similarity between text embeddings."""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer(model_name)
    embeddings = model.encode([text1, text2])

    similarity = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    return float((similarity + 1) / 2)  # Normalize to 0-1
```

### Metric Design Principles
- **Determinism**: Same inputs always produce same output
- **Normalization**: Output should be 0.0-1.0 or clearly documented
- **Efficiency**: Cache expensive computations (embeddings)
- **Interpretability**: Score should have clear meaning

### Pitfalls to Avoid
- Don't forget edge cases (empty strings, single tokens)
- Don't assume tokenization method - be explicit
- Don't ignore numerical precision issues
- Don't mix metric with evaluator logic

## Interactive Prompts

**Ask for:**
- Metric name and type (n-gram, embedding, custom)
- Input requirements (candidate, reference, context?)
- Output scale (0-1 or raw score?)
- Any reference implementation to follow?
- Dependencies needed?

**Suggest:**
- Existing libraries if available
- Normalization approach
- Caching strategy for expensive ops

**Validate:**
- Score interpretation is clear
- Edge cases handled
- Performance is acceptable

## Success Criteria
- [ ] Metric function/class implemented
- [ ] Type hints and docstrings complete
- [ ] Unit tests with known values
- [ ] Edge cases covered (empty, identical, very long)
- [ ] Performance benchmarked if expensive
- [ ] `make test && make typecheck` passes
- [ ] Documented in README if user-facing
