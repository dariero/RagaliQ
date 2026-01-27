# RagaliQ: Sequential Commands for Claude

> Copy-paste each command block to Claude sequentially. Each task builds on the previous one.

---

## Phase 1: Foundation (Judge Integration)

### Task 1: LLMJudge Abstract Base

```
Implement the LLMJudge abstract base class in src/ragaliq/judges/base.py

Create an ABC with these async methods:
- extract_claims(response: str) -> list[str]: Extract factual claims from LLM response
- verify_claim(claim: str, context: list[str]) -> ClaimVerification: Check if claim is supported by context
- score_relevance(query: str, response: str) -> float: Rate how well response answers query (0-1)
- score_coherence(response: str) -> float: Rate response coherence/fluency (0-1)

Also create these Pydantic models in the same file:
- ClaimVerification(supported: bool, confidence: float, reasoning: str)
- JudgeConfig(model: str, temperature: float = 0.0, max_tokens: int = 1024, timeout: float = 30.0)

Export in src/ragaliq/judges/__init__.py. Follow existing patterns from src/ragaliq/core/evaluator.py. Add unit tests in tests/unit/test_judges.py.

After completion run: make test && make typecheck
```

---

### Task 2: Prompt Templates

```
Create YAML prompt templates in src/ragaliq/judges/prompts/

Files to create:
1. extract_claims.yaml - System prompt for extracting factual claims from text. Output should be JSON array of strings. Include few-shot examples.

2. verify_claim.yaml - System prompt for verifying a single claim against context. Output should be JSON with {supported: bool, confidence: float, reasoning: str}. Include examples of supported/unsupported claims.

3. score_relevance.yaml - System prompt for scoring query-response alignment. Output should be JSON with {score: float, reasoning: str}. Include scoring rubric (0.0=completely off-topic, 0.5=partially relevant, 1.0=fully answers).

Each YAML should have: system_prompt, user_template (with {placeholders}), output_format, examples. Create a prompts.py loader in src/ragaliq/judges/prompts.py that reads these YAMLs using PyYAML and provides get_prompt(name: str) -> PromptTemplate function.

After completion run: make test
```

---

### Task 3: ClaudeJudge Implementation

```
Implement ClaudeJudge in src/ragaliq/judges/claude.py

Requirements:
- Inherits from LLMJudge base class
- Uses Anthropic SDK (anthropic.AsyncAnthropic)
- Loads prompts from YAML templates via the prompts loader
- Implements all abstract methods using Claude API calls
- Uses tenacity for retry logic: @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), retry=retry_if_exception_type(anthropic.APIError))
- Tracks token usage: add usage_stats property returning dict with prompt_tokens, completion_tokens, total_tokens
- Default model: claude-opus-4-5-20251101
- Parse Claude responses as JSON, handle malformed responses gracefully

Add integration test in tests/integration/test_claude_judge.py that:
- Uses @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"))
- Tests extract_claims with a simple response
- Tests verify_claim with supported and unsupported claims
- Tests score_relevance with relevant and irrelevant responses

After completion run: make test && make typecheck
```

---

### Task 4: Wire Judge into Runner

```
Update src/ragaliq/core/runner.py to use ClaudeJudge

Changes:
1. Replace _init_judge() NotImplementedError with actual initialization:
   - Accept judge parameter in __init__ (optional, defaults to ClaudeJudge)
   - Accept judge_config: JudgeConfig parameter
   - Lazy initialization on first evaluate() call

2. Add api_key parameter to RagaliQ.__init__ that's passed to ClaudeJudge

3. Update evaluate_async() to pass the judge to evaluators

4. Add a simple integration test in tests/integration/test_runner.py that creates RagaliQ with a mock judge and verifies the flow works.

After completion run: make test && make typecheck
```

---

## Phase 2: Evaluators

### Task 5: FaithfulnessEvaluator

```
Implement FaithfulnessEvaluator in src/ragaliq/evaluators/faithfulness.py

Algorithm:
1. Use judge.extract_claims(test_case.response) to get all claims
2. For each claim, call judge.verify_claim(claim, test_case.context)
3. Score = (number of supported claims) / (total claims)
4. If no claims extracted, return score 1.0 (vacuously faithful)

Class requirements:
- Inherits from Evaluator base class (from src/ragaliq/core/evaluator.py)
- name = "faithfulness"
- description = "Measures if response is grounded only in provided context"
- Store individual claim results in EvaluationResult.metadata["claims"]
- Reasoning should explain which claims failed verification

Add unit tests in tests/unit/test_evaluators.py using a mock judge. Test cases:
- All claims supported -> score 1.0
- Half claims supported -> score 0.5
- No claims -> score 1.0
- All claims unsupported -> score 0.0

Export in src/ragaliq/evaluators/__init__.py

After completion run: make test && make typecheck
```

---

### Task 6: RelevanceEvaluator

```
Implement RelevanceEvaluator in src/ragaliq/evaluators/relevance.py

Algorithm:
1. Call judge.score_relevance(test_case.query, test_case.response)
2. Return the score directly (already 0-1)
3. Include judge's reasoning in EvaluationResult.reasoning

Class requirements:
- Inherits from Evaluator base class
- name = "relevance"
- description = "Measures if response answers the user's query"

Add unit tests in tests/unit/test_evaluators.py with mock judge. Test edge cases:
- Highly relevant response -> score ~0.9-1.0
- Partial answer -> score ~0.5
- Completely off-topic -> score ~0.0

Export in src/ragaliq/evaluators/__init__.py

After completion run: make test && make typecheck
```

---

### Task 7: HallucinationEvaluator

```
Implement HallucinationEvaluator in src/ragaliq/evaluators/hallucination.py

Algorithm (inverse of faithfulness with stricter logic):
1. Extract claims from response using judge.extract_claims()
2. For each claim, verify against context using judge.verify_claim()
3. Hallucination score = (unsupported claims) / (total claims)
4. Final score = 1.0 - hallucination_score (higher is better = less hallucination)
5. Flag any claim with confidence < 0.8 as potentially hallucinated

Class requirements:
- Inherits from Evaluator base class
- name = "hallucination"
- description = "Detects made-up facts not supported by context"
- Store hallucinated_claims list in metadata["hallucinated_claims"]

Add unit tests in tests/unit/test_evaluators.py with mock judge.

Export in src/ragaliq/evaluators/__init__.py

After completion run: make test && make typecheck
```

---

### Task 8: ContextPrecisionEvaluator

```
Implement ContextPrecisionEvaluator in src/ragaliq/evaluators/context_precision.py

Algorithm:
1. For each document in test_case.context, score its relevance to test_case.query using judge.score_relevance(query, doc)
2. Calculate weighted precision: higher-ranked docs should be more relevant
3. Score = sum(relevance_i * (1/rank_i)) / sum(1/rank_i) for i in 1..n

Class requirements:
- Inherits from Evaluator base class
- name = "context_precision"
- description = "Measures if retrieved documents are relevant to the query"
- Store per-document scores in metadata["doc_scores"]

Add unit tests covering:
- All docs highly relevant -> score ~1.0
- First doc relevant, rest not -> high score (good ranking)
- First doc irrelevant, last relevant -> low score (bad ranking)

Export in src/ragaliq/evaluators/__init__.py

After completion run: make test && make typecheck
```

---

### Task 9: ContextRecallEvaluator

```
Implement ContextRecallEvaluator in src/ragaliq/evaluators/context_recall.py

Requirements:
- Only usable when test_case.expected_facts is provided
- For each expected fact, check if it's covered by any context document using judge.verify_claim(fact, context)
- Score = (covered facts) / (total expected facts)
- If expected_facts is None or empty, raise ValueError with message: "ContextRecallEvaluator requires expected_facts to be set on the test case"

Class requirements:
- Inherits from Evaluator base class
- name = "context_recall"
- description = "Measures if retrieved context covers all necessary information"
- Store which facts were covered/missed in metadata["fact_coverage"]

Add unit tests with various expected_facts scenarios.

Export in src/ragaliq/evaluators/__init__.py

After completion run: make test && make typecheck
```

---

### Task 10: Evaluator Registry

```
Create evaluator registry in src/ragaliq/evaluators/registry.py

Implement:
1. EVALUATOR_REGISTRY: dict[str, type[Evaluator]] - maps names to classes
2. register_evaluator(name: str) -> Callable decorator for registering custom evaluators
3. get_evaluator(name: str) -> type[Evaluator]: Get evaluator class by name, raise KeyError if not found
4. list_evaluators() -> list[str]: Return list of available evaluator names

Auto-register built-in evaluators:
- faithfulness -> FaithfulnessEvaluator
- relevance -> RelevanceEvaluator
- hallucination -> HallucinationEvaluator
- context_precision -> ContextPrecisionEvaluator
- context_recall -> ContextRecallEvaluator

Update src/ragaliq/evaluators/__init__.py to export:
- All evaluator classes
- register_evaluator, get_evaluator, list_evaluators functions

Update src/ragaliq/core/runner.py _init_evaluators() to use the registry instead of raising NotImplementedError.

After completion run: make test && make typecheck
```

---

## Phase 3: Usability

### Task 11: Dataset Loader

```
Implement dataset loader in src/ragaliq/datasets/loader.py

Create DatasetLoader class with:
- load(path: str | Path) -> list[RAGTestCase]: Auto-detect format from extension (.json, .yaml, .yml, .csv)
- load_json(path: Path) -> list[RAGTestCase]
- load_yaml(path: Path) -> list[RAGTestCase]
- load_csv(path: Path) -> list[RAGTestCase]: context column should be JSON array string or pipe-separated values

Create dataset schema in src/ragaliq/datasets/schemas.py:
- DatasetSchema(version: str = "1.0", test_cases: list[RAGTestCase], metadata: dict = {})
- Validate using Pydantic

Handle errors gracefully:
- FileNotFoundError: "Dataset file not found: {path}"
- ValidationError: Include specific field that failed
- UnsupportedFormatError: "Unsupported dataset format: {ext}. Supported: json, yaml, csv"

Create sample fixtures:
- tests/fixtures/sample_dataset.json (3 test cases)
- tests/fixtures/sample_dataset.yaml (same 3 test cases)
- tests/fixtures/sample_dataset.csv (same 3 test cases)

Add unit tests in tests/unit/test_datasets.py

Export DatasetLoader in src/ragaliq/datasets/__init__.py

After completion run: make test && make typecheck
```

---

### Task 12: CLI Foundation

```
Implement CLI in src/ragaliq/cli/main.py using Typer

Commands to implement:

1. ragaliq run <dataset_path> [OPTIONS]
   --evaluators, -e: Comma-separated evaluator names (default: "faithfulness,relevance,hallucination")
   --threshold, -t: Pass threshold float (default: 0.7)
   --output, -o: Output format "console", "json", "html" (default: "console")
   --output-file: File path for json/html output
   --api-key: Anthropic API key (falls back to ANTHROPIC_API_KEY env var)

2. ragaliq list-evaluators
   Print table of available evaluators with name and description columns using Rich

3. ragaliq validate <dataset_path>
   Validate dataset schema without running evaluation, print success or validation errors

4. ragaliq version
   Print package version

Use Rich for:
- Progress bar during evaluation (track per test case)
- Colored pass/fail output (green=pass, red=fail)
- Tables for results

Create Typer app and register in pyproject.toml:
[project.scripts]
ragaliq = "ragaliq.cli.main:app"

Add CLI tests in tests/unit/test_cli.py using typer.testing.CliRunner

After completion run: make test && pip install -e . && ragaliq --help
```

---

### Task 13: Dataset Generator

```
Implement dataset generator in src/ragaliq/datasets/generator.py

Create TestCaseGenerator class:
- __init__(judge: LLMJudge)
- generate_from_documents(documents: list[str], n: int = 10) -> list[RAGTestCase]
- generate_from_document(doc: str, n: int = 3) -> list[RAGTestCase]

Algorithm for each test case:
1. Use judge to generate a question that the document can answer
2. Select the source document as context (can add 1-2 related docs if available)
3. Use judge to generate a response based on context
4. Extract expected_facts from the generated response

Create prompt templates:
- src/ragaliq/judges/prompts/generate_question.yaml
- src/ragaliq/judges/prompts/generate_response.yaml

Add CLI command to src/ragaliq/cli/main.py:
ragaliq generate <docs_path> [OPTIONS]
  --count, -n: Number of test cases to generate (default: 10)
  --output, -o: Output file path (default: stdout as JSON)
  --format, -f: Output format "json" or "yaml" (default: "json")

docs_path can be a directory (reads all .txt, .md files) or single file.

Export TestCaseGenerator in src/ragaliq/datasets/__init__.py

After completion run: make test && make typecheck
```

---

### Task 14: Pytest Plugin

```
Implement pytest plugin in src/ragaliq/integrations/pytest_plugin.py

Create:

1. Fixture: rag_tester
@pytest.fixture
def rag_tester(request) -> RagaliQ:
    # Read config from pytest.ini [ragaliq] section or pyproject.toml [tool.ragaliq]
    # Initialize RagaliQ with ClaudeJudge
    # Return configured instance

2. Helper function:
def assert_rag_quality(
    tester: RagaliQ,
    test_case: RAGTestCase,
    evaluators: list[str] | None = None,
    threshold: float = 0.7
) -> RAGTestResult:
    # Run evaluation
    # If any evaluator fails threshold, raise AssertionError with detailed report
    # Return result on success

3. Pytest markers:
- pytest.mark.rag_test: Mark test as RAG test (for filtering with -m rag_test)
- pytest.mark.rag_slow: Mark as slow/requires API (for skipping with -m "not rag_slow")

Register plugin in pyproject.toml:
[project.entry-points.pytest11]
ragaliq = "ragaliq.integrations.pytest_plugin"

Create examples/pytest_example/:
- conftest.py: Import and configure rag_tester
- test_my_rag.py: 2-3 example RAG tests using the fixture

After completion run: make test && pytest examples/pytest_example/ --collect-only
```

---

## Phase 4: Reports & Polish

### Task 15: Console Reporter

```
Implement console reporter in src/ragaliq/reports/console.py

Create ConsoleReporter class:
- __init__(verbose: bool = False)
- report(results: list[RAGTestResult]) -> None: Print formatted report to stdout

Output format using Rich:

1. Header panel:
   "RagaliQ Evaluation Report"
   Timestamp, number of tests, evaluators used

2. Per-test-case (if verbose or failed):
   Panel with test name/ID
   Query text (truncated to 100 chars if longer)
   Table: Evaluator | Score | Threshold | Status
   Overall: PASS or FAIL

3. Summary table:
   Columns: Evaluator | Avg Score | Pass Rate
   Row per evaluator
   Final row: Overall | X.XX | XX%

4. Final status line:
   "X/Y tests passed" in green if all pass, red if any fail

Integrate with CLI: update `ragaliq run` to use ConsoleReporter when --output=console

Export ConsoleReporter in src/ragaliq/reports/__init__.py

After completion run: make test
```

---

### Task 16: JSON and HTML Reports

```
Implement report exporters in src/ragaliq/reports/

1. src/ragaliq/reports/json_export.py
Create JSONReporter class:
- export(results: list[RAGTestResult], output_path: Path) -> None
- Schema: {
    "version": "1.0",
    "timestamp": "ISO8601",
    "summary": {"total": N, "passed": N, "failed": N, "pass_rate": 0.XX, "avg_scores": {"evaluator": score}},
    "results": [serialized RAGTestResult objects]
  }
- Pretty-print with indent=2

2. src/ragaliq/reports/html.py
Create HTMLReporter class:
- export(results: list[RAGTestResult], output_path: Path) -> None
- Use Jinja2 template

3. src/ragaliq/reports/templates/report.html.j2
Create self-contained HTML template with:
- Embedded CSS (no external dependencies)
- Summary cards at top (total, passed, failed, pass rate)
- Results table with expandable rows for details
- Color coding (green/red for pass/fail)
- Embedded Chart.js for score visualization (optional, via CDN is OK)

Update CLI `ragaliq run` to use appropriate reporter based on --output flag.

Export JSONReporter, HTMLReporter in src/ragaliq/reports/__init__.py

After completion run: make test && ragaliq run tests/fixtures/sample_dataset.json --output html --output-file /tmp/report.html && open /tmp/report.html
```

---

### Task 17: GitHub Actions Integration

```
Create GitHub Actions helpers in src/ragaliq/integrations/github_actions.py

Implement:
1. is_github_actions() -> bool: Check if GITHUB_ACTIONS env var is set
2. set_output(name: str, value: str) -> None: Write to GITHUB_OUTPUT file
3. create_step_summary(results: list[RAGTestResult]) -> str: Markdown summary for GITHUB_STEP_SUMMARY
4. create_annotations(results: list[RAGTestResult]) -> None: Print ::error:: and ::warning:: for failed tests

Update CLI to auto-detect GitHub Actions:
- If is_github_actions(), write step summary automatically
- Disable Rich colors/formatting in CI (check CI env var too)

Create examples/ci_cd_example/.github/workflows/rag-tests.yml:
name: RAG Quality Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.14'
      - run: pip install ragaliq
      - run: ragaliq run tests/rag_tests.json --output json --output-file results.json
      - run: ragaliq run tests/rag_tests.json --output html --output-file report.html
      - uses: actions/upload-artifact@v4
        with:
          name: rag-report
          path: report.html
        if: always()
    env:
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

Export helpers in src/ragaliq/integrations/__init__.py

After completion run: make test
```

---

### Task 18: Documentation and Examples

```
Create documentation and examples:

1. docs/TUTORIAL.md - "Testing Your First RAG System"
Sections:
- Prerequisites (Python 3.14+, Anthropic API key)
- Installation (pip install ragaliq)
- Quick Start (3 commands to first test)
- Creating Test Cases (manual JSON structure)
- Running Evaluations (CLI usage)
- Understanding Results (score interpretation)
- Writing Pytest Tests (code examples)
- CI/CD Integration (link to workflow example)

2. examples/basic_usage.py
Complete runnable script:
- Load sample dataset
- Initialize RagaliQ
- Run evaluation
- Print results
Include comments explaining each step

3. Update README.md:
- Add badges placeholder comments for PyPI, tests, coverage
- Rewrite Quick Start to be 3 simple steps
- Add "Available Evaluators" section with table
- Add "CLI Reference" section
- Add "Pytest Integration" section with code example
- Add "CI/CD" section linking to examples

4. Audit all public classes in src/ragaliq/:
- Ensure Google-style docstrings on all public classes and methods
- Add module-level docstrings to each __init__.py

After completion: Review docs manually for clarity
```

---

### Task 19: Final Integration and Release

```
Final integration and release preparation:

1. Create tests/integration/test_full_pipeline.py
End-to-end test that:
- Loads tests/fixtures/sample_dataset.json
- Initializes RagaliQ with real or mock judge
- Runs all evaluators
- Generates console, JSON, and HTML reports
- Verifies outputs are valid
Use @pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY")) for real API tests

2. Update src/ragaliq/__init__.py with clean public API:
from ragaliq.core import RagaliQ, RAGTestCase, RAGTestResult, Evaluator, EvaluationResult
from ragaliq.evaluators import (
    FaithfulnessEvaluator, RelevanceEvaluator, HallucinationEvaluator,
    ContextPrecisionEvaluator, ContextRecallEvaluator,
    get_evaluator, list_evaluators, register_evaluator
)
from ragaliq.judges import ClaudeJudge, LLMJudge
from ragaliq.datasets import DatasetLoader, TestCaseGenerator
__version__ = "0.1.0"
__all__ = [list of all exports]

3. Create CHANGELOG.md:
# Changelog
## [0.1.0] - YYYY-MM-DD
### Added
- Initial release
- Core evaluators: faithfulness, relevance, hallucination, context_precision, context_recall
- Claude judge integration
- CLI with run, generate, validate commands
- Pytest plugin with rag_tester fixture
- Console, JSON, HTML reporters
- GitHub Actions integration

4. Verify pyproject.toml has correct:
- version = "0.1.0"
- All dependencies
- All entry points (CLI, pytest plugin)
- Classifiers for PyPI

5. Run full verification:
make clean && make install && make test && make lint && make typecheck && make build

After completion run: twine check dist/* && pip install dist/*.whl && ragaliq --help
```

---

## Quick Reference: Task Dependencies

```
1 (Judge Base) ─┬─► 2 (Prompts) ─► 3 (ClaudeJudge) ─► 4 (Wire Runner)
                │
                └─► 5-9 (Evaluators) ─► 10 (Registry)
                                            │
                    11 (Dataset Loader) ────┼─► 12 (CLI) ─► 13 (Generator)
                                            │
                                            └─► 14 (Pytest Plugin)
                                                    │
                    15 (Console Report) ◄───────────┤
                    16 (JSON/HTML Reports) ◄────────┤
                    17 (GitHub Actions) ◄───────────┘
                                            │
                    18 (Docs) ──────────────┴─► 19 (Release)
```

**Start with Task 1. Each task is designed to be completable in one session.**
