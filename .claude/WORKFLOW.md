# RagaliQ Development Workflow

The complete reference for building RagaliQ. Two commands to ship code, three patterns to extend it.

## Philosophy

RagaliQ follows three principles:

1. **LLM-as-Judge over hardcoded rules** -- Human language is too nuanced for regex. An LLM evaluates LLM output the same way a senior engineer reviews a junior's work: holistically, with context.
2. **Async-first** -- Every LLM call is I/O-bound (1-10s latency). Async enables parallel claim verification and non-blocking test runners. Synchronous threading was considered but rejected due to GIL limitations and inferior error propagation.
3. **Evaluator-per-metric** -- Each quality dimension (faithfulness, relevance, toxicity) is a separate class. This follows the Single Responsibility Principle and allows users to compose only the evaluators they need.

## The Two-Command Workflow

```
/start-work <issue>   -->   implement   -->   /ship
```

These are the only two commands. Everything else is inline guidance below.

## Project Constants

```
OWNER:         dariero
PROJECT_ID:    PVT_kwHODR8J4s4BNe_Y
PROJECT_NUM:   2
STATUS_FIELD:  PVTSSF_lAHODR8J4s4BNe_Yzg8dwP8

Board statuses:
  Todo:   98236657
  Doing:  47fc9ee4
  Done:   caff0873
```

**Board URL:** https://github.com/users/dariero/projects/2/views/1

**Branch naming:** `<prefix>/<issue>-<description>`

| Title Prefix | Branch Prefix |
|--------------|---------------|
| `[FEAT]` | `feat/` |
| `[FIX]` | `fix/` |
| `[REFACTOR]` | `refactor/` |
| `[DOCS]` | `docs/` |
| (none) | `feat/` |

**Commit format:** `[TYPE #issue] Description`

**Quality gates:** `hatch run lint && hatch run typecheck && hatch run test`

---

## Implementation Patterns

The following are not commands. They are reference patterns for extending RagaliQ.

---

### Pattern: Creating an Evaluator

**When:** Adding a new quality metric (toxicity, relevance, context precision, etc.)

**Files to create:**

```
src/ragaliq/evaluators/{name}.py
tests/unit/evaluators/test_{name}.py
tests/integration/evaluators/test_{name}.py
```

**Template:**

```python
from ragaliq.core.evaluator import Evaluator, EvaluationResult
from ragaliq.core.test_case import RAGTestCase
from ragaliq.judges.base import LLMJudge


class {Name}Evaluator(Evaluator):
    """
    One-line description.

    Score interpretation:
        1.0 = [what perfect means]
        0.0 = [what failure means]
    """

    name: str = "{name}"
    description: str = "..."

    async def evaluate(
        self,
        test_case: RAGTestCase,
        judge: LLMJudge,
    ) -> EvaluationResult:
        if not test_case.response:
            return EvaluationResult(
                evaluator_name=self.name,
                score=0.0,
                passed=False,
                reasoning="Empty response",
            )

        # Implementation: extract units, score via judge, aggregate

        return EvaluationResult(
            evaluator_name=self.name,
            score=...,          # 0.0-1.0
            passed=...,         # score >= threshold
            reasoning=...,      # Human-readable explanation
            raw_response=...,   # Debugging details
        )
```

**Checklist:**
- [ ] Async `evaluate()` with correct signature
- [ ] Score normalized 0.0-1.0
- [ ] Empty input handled
- [ ] Export added to `evaluators/__init__.py`
- [ ] Unit tests mock the judge
- [ ] `hatch run test && hatch run typecheck` passes

**Design rationale:** Scores use 0.0-1.0 floats (not integer 1-5 scales) because normalized floats enable flexible thresholds, mathematical aggregation across evaluators, and cross-metric comparison.

---

### Pattern: Creating a Judge

**When:** Adding a new LLM backend (OpenAI, Gemini, Mistral, Ollama, etc.)

**Files to create:**

```
src/ragaliq/judges/{provider}.py
tests/unit/judges/test_{provider}.py
tests/integration/judges/test_{provider}.py
```

**Template:**

```python
import os

from tenacity import retry, stop_after_attempt, wait_exponential

from ragaliq.judges.base import LLMJudge, JudgeConfig


class {Provider}Judge(LLMJudge):
    """Judge using {Provider} API. Requires {PROVIDER}_API_KEY."""

    def __init__(
        self,
        api_key: str | None = None,
        config: JudgeConfig | None = None,
    ):
        super().__init__(config=config)
        self.api_key = api_key or os.getenv("{PROVIDER}_API_KEY")
        if not self.api_key:
            raise ValueError("{PROVIDER}_API_KEY not found.")
        self._usage = {"prompt_tokens": 0, "completion_tokens": 0}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _call_llm(self, system: str, user: str) -> str:
        # API call + track self._usage
        ...

    # Implement all abstract methods from LLMJudge
    async def extract_claims(self, response: str) -> ClaimsResult: ...
    async def verify_claim(self, claim: str, context: list[str]) -> ClaimVerification: ...
```

**Checklist:**
- [ ] All API calls async
- [ ] Retry with exponential backoff (tenacity)
- [ ] Token usage tracked in `_usage`
- [ ] Missing API key raises clear error
- [ ] All `LLMJudge` abstract methods implemented
- [ ] Export added to `judges/__init__.py`
- [ ] `hatch run test && hatch run typecheck` passes

**Design rationale:** Class-based (not function-based) because judges carry state: token counters, configuration, and client instances. Async + retry handles the three realities of LLM APIs: high latency, rate limits, and transient failures.

---

### Pattern: Optimizing Prompts

**When:** Evaluator scores are inconsistent, judge responses are malformed, or thresholds need calibration.

**Workflow:**

1. **Create eval dataset** at `tests/fixtures/prompt_eval_{name}.json` with 15+ cases spanning obvious-true, obvious-false, and ambiguous categories.
2. **Measure baseline** accuracy against the dataset.
3. **Modify prompts** in `src/ragaliq/judges/prompts/*.yaml` (focus: JSON schema, few-shot examples, scoring anchors).
4. **A/B test** old vs. new. Ship only if: accuracy improves AND no new regressions.
5. **Validate** integration tests still parse JSON correctly.

**Decision criteria:**
- Accuracy up, zero new failures --> ship
- Accuracy up >5%, one new failure --> review, likely ship
- Accuracy down --> reject
- Multiple new failures --> investigate
