# RagaliQ - Implementation Plan

## Vision

RagaliQ (RAG + Quality) - a Python library that brings software testing discipline to LLM/RAG systems. QA engineers should be able to write LLM quality tests as easily as they write unit tests.

## Target Users

- AI/ML engineers deploying RAG systems to production
- QA engineers transitioning to AI Quality Engineering
- DevOps teams setting up CI/CD for AI projects

---

## Architecture Overview

### Core Components

#### 1. Test Case Model (`core/test_case.py`)

```python
class RAGTestCase(BaseModel):
    id: str
    name: str
    query: str                    # User question
    context: list[str]            # Retrieved documents
    response: str                 # LLM response to evaluate
    expected_answer: str | None   # Optional ground truth
    expected_facts: list[str] | None
    tags: list[str]
```

#### 2. Evaluator Interface (`core/evaluator.py`)

```python
class Evaluator(ABC):
    name: str
    description: str
    threshold: float = 0.7

    @abstractmethod
    async def evaluate(self, test_case: RAGTestCase, judge: LLMJudge) -> EvaluationResult:
        pass
```

#### 3. LLM Judge (`judges/base.py`)

```python
class LLMJudge(ABC):
    @abstractmethod
    async def extract_claims(self, response: str) -> ClaimsResult:
        pass

    @abstractmethod
    async def verify_claim(self, claim: str, context: list[str]) -> ClaimVerdict:
        pass

    @abstractmethod
    async def evaluate_relevance(self, query: str, response: str) -> JudgeResult:
        pass

    @abstractmethod
    async def evaluate_faithfulness(self, response: str, context: list[str]) -> JudgeResult:
        pass
```

### Evaluators to Implement

| Evaluator | Description | Priority |
|-----------|-------------|----------|
| `FaithfulnessEvaluator` | Is response grounded in context only? | P0 |
| `RelevanceEvaluator` | Does response answer the question? | P0 |
| `HallucinationEvaluator` | Does response contain made-up facts? | P0 |
| `ContextPrecisionEvaluator` | Are retrieved docs relevant? | P1 |
| `ContextRecallEvaluator` | Do retrieved docs cover the answer? | P1 |
| `ToxicityEvaluator` | Safety and appropriateness | P2 |

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)

#### Week 1: Project Structure & Core Models

- [ ] Initialize project with `pyproject.toml`
- [ ] Set up development environment (ruff, mypy, pytest)
- [ ] Implement `RAGTestCase` and `RAGTestResult` models
- [ ] Implement `Evaluator` base class
- [ ] Implement `EvaluationResult` model
- [ ] Write unit tests for all models

**Acceptance Criteria:**
- `pip install -e .` works
- All models validate correctly
- 100% test coverage on core models

#### Week 2: Claude Judge Integration

- [ ] Implement `LLMJudge` abstract base
- [ ] Implement `ClaudeJudge` with Anthropic SDK
- [ ] Create prompt templates in YAML
- [ ] Add retry logic with tenacity
- [ ] Add token usage tracking
- [ ] Integration tests with Claude API

**Acceptance Criteria:**
- Can call Claude to extract claims from text
- Can call Claude to verify claim against context
- Graceful handling of API errors

---

### Phase 2: Evaluators (Weeks 3-4)

#### Week 3: Core Evaluators

- [ ] `FaithfulnessEvaluator` - claim extraction + verification
- [ ] `RelevanceEvaluator` - query-response alignment
- [ ] `HallucinationEvaluator` - detect unsupported claims

**Acceptance Criteria:**
- Each evaluator returns score 0-1
- Each evaluator provides reasoning
- Unit tests with mocked judge

#### Week 4: RAG-Specific Evaluators

- [ ] `ContextPrecisionEvaluator` - retrieval quality
- [ ] `ContextRecallEvaluator` - retrieval completeness
- [ ] Custom evaluator framework
- [ ] Evaluator registry

**Acceptance Criteria:**
- 5+ working evaluators
- Users can create custom evaluators
- Integration tests with real API

---

### Phase 3: Usability (Weeks 5-6)

#### Week 5: CLI & Datasets

- [ ] Typer CLI with `run`, `generate`, `report` commands
- [ ] Dataset loader (JSON, YAML, CSV)
- [ ] Dataset schema validation
- [ ] Basic dataset generator from documents
- [ ] Progress bars and colored output

**Acceptance Criteria:**
- `ragaliq run tests.json` works end-to-end
- `ragaliq generate docs/` creates test cases

#### Week 6: Pytest Integration

- [ ] Pytest plugin registration
- [ ] `rag_tester` fixture
- [ ] `assert_rag_quality()` helper
- [ ] Pytest markers for RAG tests
- [ ] Example test files

**Acceptance Criteria:**
- `pytest` discovers and runs RAG tests
- Clear failure messages with scores

---

### Phase 4: Reports & Polish (Weeks 7-8)

#### Week 7: Reporting

- [ ] Console reporter with Rich tables
- [ ] HTML report with Jinja2
- [ ] JSON export for CI/CD
- [ ] Trend tracking (compare runs)
- [ ] GitHub Actions workflow example

**Acceptance Criteria:**
- Beautiful terminal output
- Shareable HTML reports
- CI/CD integration documented

#### Week 8: Documentation & Release

- [ ] Comprehensive README
- [ ] API documentation
- [ ] Tutorial: "Testing Your First RAG System"
- [ ] Example projects
- [ ] PyPI publication

**Acceptance Criteria:**
- New user can start in < 5 minutes
- Published on PyPI
- GitHub repo polished

---

## File Structure

```
ragaliq/
├── src/
│   └── ragaliq/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── test_case.py
│       │   ├── evaluator.py
│       │   ├── result.py
│       │   └── runner.py
│       ├── evaluators/
│       │   ├── __init__.py
│       │   ├── faithfulness.py
│       │   ├── relevance.py
│       │   ├── hallucination.py
│       │   ├── context_precision.py
│       │   ├── context_recall.py
│       │   └── custom.py
│       ├── judges/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── claude.py
│       │   ├── openai.py
│       │   └── prompts/
│       │       ├── extract_claims.yaml
│       │       ├── verify_claim.yaml
│       │       └── relevance.yaml
│       ├── datasets/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   ├── generator.py
│       │   └── schemas.py
│       ├── reports/
│       │   ├── __init__.py
│       │   ├── console.py
│       │   ├── html.py
│       │   ├── json_export.py
│       │   └── templates/
│       │       └── report.html.j2
│       ├── integrations/
│       │   ├── __init__.py
│       │   ├── pytest_plugin.py
│       │   └── github_actions.py
│       └── cli/
│           ├── __init__.py
│           └── main.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_models.py
│   │   ├── test_evaluators.py
│   │   └── test_judges.py
│   ├── integration/
│   │   └── test_full_pipeline.py
│   └── fixtures/
│       └── sample_dataset.json
├── examples/
│   ├── basic_usage.py
│   ├── pytest_example/
│   │   ├── conftest.py
│   │   └── test_my_rag.py
│   └── ci_cd_example/
│       └── .github/workflows/rag-tests.yml
├── docs/
│   ├── PROJECT_PLAN.md
│   ├── ARCHITECTURE.md
│   └── TUTORIAL.md
├── CLAUDE.md
├── pyproject.toml
├── README.md
├── Makefile
└── .gitignore
```

---

## Dependencies and the provider seam

`pyproject.toml` is the single source of truth for versions. This section
deliberately does **not** restate them: a copy here drifted to seven wrong
floors and a dependency that never existed (`httpx`), which is exactly the
duplicated-fact problem the consistency checks in `scripts/verify_release.py`
now guard against elsewhere.

What is worth describing here is the shape the dependencies serve.

### The transport seam

Provider access is isolated behind one structural type, so swapping or adding
a provider does not touch judging logic. See
[`ADR-000-judge-transport-protocol.md`](https://github.com/dariero/RagaliQ/blob/main/.decisions/ADR-000-judge-transport-protocol.md).

- **`JudgeTransport`** (`judges/transport.py`) is a `Protocol`, not a base
  class. Anything with a matching `send()` satisfies it — no registration, no
  inheritance.
- **`ClaudeTransport`** implements it by wrapping `AsyncAnthropic`. It owns
  provider HTTP calls, retries and token counting, and nothing else.
- **`BaseJudge`** depends on the protocol rather than any concrete transport.
  Prompt building, response parsing and score clamping live there, so they are
  written once and shared by every provider.

Because the dependency points at a structural type, adding a second provider
is additive: an `OpenAITransport` with a matching `send()` requires no change
to `BaseJudge`. That is what preserves cross-provider optionality — not a
declared extra. Tracked in #114.

### Direct runtime dependencies, by role

| Package | Role |
|---|---|
| `anthropic` | the only provider SDK currently wired, used solely by `ClaudeTransport` |
| `pydantic` | every data structure; strict validation at the boundaries |
| `tenacity` | retry policy around transport calls |
| `typer` | CLI |
| `rich` | terminal rendering |
| `jinja2` | HTML report templates |
| `pyyaml` | dataset and prompt loading |

`httpx` is **not** a RagaliQ dependency. It appears in `pylock.toml` only as
anthropic's transitive pull, and is imported nowhere in the project.

---

## Success Metrics

1. **Usability**: New user can run first test in < 5 minutes
2. **Accuracy**: Evaluators correlate with human judgment > 80%
3. **Performance**: < 5 seconds per test case evaluation
4. **Adoption**: 100+ GitHub stars in first month (stretch goal)
