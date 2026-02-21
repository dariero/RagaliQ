"""
RagaliQ Basic Usage Examples
=============================

Demonstrates the core RagaliQ API patterns:
  1. Single evaluation (Python API)
  2. Batch evaluation
  3. Custom evaluators and thresholds
  4. Dataset loading from JSON
  5. All three report formats (console, JSON, HTML)
  6. TraceCollector for observability
  7. TestCaseGenerator

Prerequisites:
    pip install ragaliq
    export ANTHROPIC_API_KEY=sk-ant-...

Run the full script:
    python examples/basic_usage.py

Run a specific section:
    python examples/basic_usage.py --section single
    python examples/basic_usage.py --section batch
    python examples/basic_usage.py --section reports
    python examples/basic_usage.py --section generate
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from ragaliq import RagaliQ, RAGTestCase


# ---------------------------------------------------------------------------
# Sample test cases reused across sections
# ---------------------------------------------------------------------------

FAITHFUL_CASE = RAGTestCase(
    id="ex-faithful-1",
    name="Capital of France",
    query="What is the capital of France?",
    context=[
        "France is a country in Western Europe.",
        "The capital city of France is Paris, home to the Eiffel Tower.",
    ],
    response="The capital of France is Paris.",
)

RELEVANT_CASE = RAGTestCase(
    id="ex-relevant-1",
    name="Machine learning definition",
    query="What is machine learning?",
    context=["Machine learning is a subset of AI that enables systems to learn from data."],
    response="Machine learning is an AI technique where systems improve through data exposure.",
)

HALLUCINATION_CASE = RAGTestCase(
    id="ex-hallucination-1",
    name="Python version (with added detail)",
    query="When was Python released?",
    context=["Python was first released in 1991 by Guido van Rossum."],
    # Response adds 'version 3' — not in context → potential hallucination
    response="Python version 3 was released in 1991 by Guido van Rossum.",
)

MULTI_DOC_CASE = RAGTestCase(
    id="ex-multi-1",
    name="Async programming benefits",
    query="What are the benefits of async programming?",
    context=[
        "Async programming allows handling many tasks concurrently without blocking.",
        "In Python, asyncio enables writing non-blocking I/O-bound code.",
        "Async code improves throughput for network-bound applications.",
    ],
    response="Async programming improves concurrency and throughput for I/O-bound tasks.",
)

RECALL_CASE = RAGTestCase(
    id="ex-recall-1",
    name="Python facts",
    query="Tell me about Python's origins.",
    context=["Python was created by Guido van Rossum and released in 1991."],
    response="Python was created by Guido van Rossum in 1991.",
    expected_facts=["created by Guido van Rossum", "released in 1991"],
)


# ---------------------------------------------------------------------------
# Section 1: Single Evaluation
# ---------------------------------------------------------------------------

def section_single() -> None:
    """Single evaluation — the simplest usage pattern."""
    print("\n" + "=" * 60)
    print("SECTION 1: Single Evaluation")
    print("=" * 60)

    # RagaliQ defaults: judge=claude, evaluators=[faithfulness, relevance], threshold=0.7
    tester = RagaliQ(judge="claude")

    print(f"\nRunner: {tester!r}")
    print(f"Evaluating: {FAITHFUL_CASE.name!r}...")

    result = tester.evaluate(FAITHFUL_CASE)

    print(f"\nStatus:          {result.status}")
    print(f"Passed:          {result.passed}")
    print(f"Execution time:  {result.execution_time_ms}ms")
    print(f"Tokens used:     {result.judge_tokens_used}")
    print("\nScores:")
    for metric, score in result.scores.items():
        mark = "✓" if score >= 0.7 else "✗"
        print(f"  {mark} {metric}: {score:.2f}")

    print("\nFaithfulness reasoning:")
    print(f"  {result.details['faithfulness']['reasoning'][:200]}...")


# ---------------------------------------------------------------------------
# Section 2: Batch Evaluation
# ---------------------------------------------------------------------------

def section_batch() -> None:
    """Batch evaluation — efficient parallel processing."""
    print("\n" + "=" * 60)
    print("SECTION 2: Batch Evaluation")
    print("=" * 60)

    tester = RagaliQ(
        judge="claude",
        evaluators=["faithfulness", "relevance", "hallucination"],
        default_threshold=0.75,
        max_concurrency=3,
    )

    test_cases = [FAITHFUL_CASE, RELEVANT_CASE, HALLUCINATION_CASE, MULTI_DOC_CASE]
    print(f"\nEvaluating {len(test_cases)} test cases (concurrency=3)...")

    results = tester.evaluate_batch(test_cases)

    passed = sum(1 for r in results if r.passed)
    total_tokens = sum(r.judge_tokens_used for r in results)

    print(f"\nResults: {passed}/{len(results)} passed")
    print(f"Total tokens: {total_tokens}")
    print()

    for r in results:
        status_mark = "✓" if r.passed else "✗"
        scores_str = " | ".join(f"{k}: {v:.2f}" for k, v in r.scores.items())
        print(f"  {status_mark} {r.test_case.name}: [{scores_str}]")

    # Show failure details
    failures = [r for r in results if not r.passed]
    if failures:
        print("\nFailure details:")
        for r in failures:
            for metric, detail in r.details.items():
                if not detail["passed"]:
                    print(f"  [{r.test_case.name}] {metric}: {detail['reasoning'][:150]}...")


# ---------------------------------------------------------------------------
# Section 3: Custom Evaluators
# ---------------------------------------------------------------------------

def section_custom_evaluator() -> None:
    """Custom evaluators — extending the built-in set."""
    print("\n" + "=" * 60)
    print("SECTION 3: Custom Evaluator")
    print("=" * 60)

    from ragaliq.core.evaluator import Evaluator, EvaluationResult
    from ragaliq.evaluators import register_evaluator
    from ragaliq.judges.base import LLMJudge

    @register_evaluator("conciseness")
    class ConcisenessEvaluator(Evaluator):
        """Evaluates whether the response is appropriately concise."""

        name = "conciseness"
        description = "Measures whether the response is brief and to the point"
        threshold = 0.6

        async def evaluate(
            self, test_case: RAGTestCase, judge: LLMJudge
        ) -> EvaluationResult:
            """Evaluate conciseness by scoring relevance of a summary question."""
            query = f"Is this a concise and direct answer to: {test_case.query}"
            result = await judge.evaluate_relevance(
                query=query,
                response=test_case.response,
            )
            return EvaluationResult(
                evaluator_name=self.name,
                score=result.score,
                passed=self.is_passing(result.score),
                reasoning=result.reasoning,
                tokens_used=result.tokens_used,
                raw_response={"score": result.score},
            )

    # Use the custom evaluator
    tester = RagaliQ(
        judge="claude",
        evaluators=["faithfulness", "conciseness"],
        default_threshold=0.65,
    )

    print(f"\nEvaluating with custom 'conciseness' evaluator...")
    result = tester.evaluate(FAITHFUL_CASE)

    print(f"\nStatus: {result.status}")
    for metric, score in result.scores.items():
        mark = "✓" if score >= 0.65 else "✗"
        print(f"  {mark} {metric}: {score:.2f}")


# ---------------------------------------------------------------------------
# Section 4: Context Recall (requires expected_facts)
# ---------------------------------------------------------------------------

def section_context_recall() -> None:
    """Context recall — evaluates retrieval completeness."""
    print("\n" + "=" * 60)
    print("SECTION 4: Context Recall")
    print("=" * 60)

    tester = RagaliQ(
        judge="claude",
        evaluators=["context_recall"],
    )

    print(f"\nEvaluating: {RECALL_CASE.name!r}")
    print(f"Expected facts: {RECALL_CASE.expected_facts}")

    result = tester.evaluate(RECALL_CASE)

    print(f"\nContext recall score: {result.scores['context_recall']:.2f}")
    print(f"Passed: {result.passed}")

    raw = result.details["context_recall"]["raw"]
    if "fact_coverage" in raw:
        print("\nFact coverage:")
        for fact_result in raw["fact_coverage"]:
            verdict = fact_result.get("verdict", "?")
            fact = fact_result.get("fact", fact_result.get("claim", ""))
            mark = "✓" if verdict == "SUPPORTED" else "✗"
            print(f"  {mark} [{verdict}] {fact}")


# ---------------------------------------------------------------------------
# Section 5: Dataset Loading
# ---------------------------------------------------------------------------

def section_dataset() -> None:
    """Dataset loading from JSON file."""
    print("\n" + "=" * 60)
    print("SECTION 5: Dataset Loading")
    print("=" * 60)

    from ragaliq.datasets import DatasetLoader

    # Create a temporary dataset file
    dataset_data = {
        "version": "1.0",
        "metadata": {"source": "basic_usage_example"},
        "test_cases": [
            {
                "id": "ds-1",
                "name": "Capital query",
                "query": "What is the capital of France?",
                "context": ["The capital city of France is Paris."],
                "response": "The capital of France is Paris.",
                "tags": ["geography"],
            },
            {
                "id": "ds-2",
                "name": "ML definition",
                "query": "What is machine learning?",
                "context": ["Machine learning is a subset of AI that learns from data."],
                "response": "Machine learning is an AI technique for learning from data.",
                "tags": ["ai"],
            },
        ],
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(dataset_data, f, indent=2)
        dataset_path = Path(f.name)

    try:
        print(f"\nLoading dataset from: {dataset_path}")
        dataset = DatasetLoader.load(dataset_path)
        print(f"Loaded {len(dataset.test_cases)} test cases")
        print(f"Version: {dataset.version}")
        print(f"Metadata: {dataset.metadata}")

        for tc in dataset.test_cases:
            print(f"  - [{tc.id}] {tc.name} (tags: {tc.tags})")

        # Evaluate the loaded dataset
        tester = RagaliQ(judge="claude")
        print("\nEvaluating dataset...")
        results = tester.evaluate_batch(dataset.test_cases)

        passed = sum(1 for r in results if r.passed)
        print(f"Results: {passed}/{len(results)} passed")

    finally:
        dataset_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Section 6: Reports
# ---------------------------------------------------------------------------

def section_reports() -> None:
    """All three report formats — console, JSON, HTML."""
    print("\n" + "=" * 60)
    print("SECTION 6: Reports")
    print("=" * 60)

    from ragaliq.reports import ConsoleReporter, HTMLReporter, JSONReporter

    tester = RagaliQ(judge="claude", evaluators=["faithfulness", "relevance"])
    test_cases = [FAITHFUL_CASE, RELEVANT_CASE, HALLUCINATION_CASE]
    results = tester.evaluate_batch(test_cases)

    # --- Console Report ---
    print("\n--- Console Report ---")
    ConsoleReporter(threshold=0.7).report(results)

    # --- JSON Report ---
    print("\n--- JSON Report (summary only) ---")
    from ragaliq.reports import JSONReporter

    json_str = JSONReporter(threshold=0.7).export(results)
    doc = json.loads(json_str)
    summary = doc["summary"]
    print(f"Total:     {summary['total']}")
    print(f"Passed:    {summary['passed']}")
    print(f"Failed:    {summary['failed']}")
    print(f"Pass rate: {summary['pass_rate']:.1%}")
    for ev_name, stats in summary["evaluators"].items():
        print(f"  {ev_name}: passed={stats['passed']}, avg={stats['avg_score']:.2f}")

    # Write to file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        f.write(json_str)
        json_path = Path(f.name)
    print(f"\nJSON report written to: {json_path}")

    # --- HTML Report ---
    html_str = HTMLReporter(threshold=0.7).export(results)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        f.write(html_str)
        html_path = Path(f.name)
    print(f"HTML report written to: {html_path}")

    # Cleanup
    json_path.unlink(missing_ok=True)
    html_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Section 7: Observability (TraceCollector)
# ---------------------------------------------------------------------------

def section_observability() -> None:
    """TraceCollector — per-call timing and token tracking."""
    print("\n" + "=" * 60)
    print("SECTION 7: Observability (TraceCollector)")
    print("=" * 60)

    from ragaliq.judges import ClaudeJudge
    from ragaliq.judges.trace import TraceCollector

    collector = TraceCollector()
    judge = ClaudeJudge(trace_collector=collector)

    tester = RagaliQ(
        judge=judge,
        evaluators=["faithfulness", "relevance"],
    )

    print("\nEvaluating two test cases...")
    tester.evaluate_batch([FAITHFUL_CASE, RELEVANT_CASE])

    print(f"\nTrace summary:")
    print(f"  Total API calls:    {len(collector.traces)}")
    print(f"  Total tokens:       {collector.total_tokens}")
    print(f"  Input tokens:       {collector.total_input_tokens}")
    print(f"  Output tokens:      {collector.total_output_tokens}")
    print(f"  Total latency:      {collector.total_latency_ms}ms")
    print(f"  Estimated cost:     ${collector.total_cost_estimate:.4f}")
    print(f"  Successes:          {collector.success_count}")
    print(f"  Failures:           {collector.failure_count}")

    # Per-operation breakdown
    ops: dict[str, int] = {}
    for trace in collector.traces:
        ops[trace.operation] = ops.get(trace.operation, 0) + 1

    print("\nCalls by operation:")
    for op, count in sorted(ops.items()):
        print(f"  {op}: {count}")


# ---------------------------------------------------------------------------
# Section 8: Test Case Generation
# ---------------------------------------------------------------------------

def section_generate() -> None:
    """TestCaseGenerator — generate test cases from documents."""
    print("\n" + "=" * 60)
    print("SECTION 8: Test Case Generation")
    print("=" * 60)

    from ragaliq import TestCaseGenerator
    from ragaliq.judges import ClaudeJudge

    documents = [
        "Python is a high-level, interpreted programming language created by Guido van "
        "Rossum and first released in 1991. Python emphasizes code readability and simplicity.",
        "Python supports multiple programming paradigms, including structured, object-oriented, "
        "and functional programming. It uses dynamic typing and garbage collection.",
        "The Python Package Index (PyPI) hosts hundreds of thousands of packages that extend "
        "Python's functionality. Notable packages include NumPy, Pandas, Flask, and Django.",
    ]

    print(f"\nGenerating 3 test cases from {len(documents)} documents...")

    judge = ClaudeJudge()
    generator = TestCaseGenerator()
    test_cases = asyncio.run(
        generator.generate_from_documents(documents=documents, n=3, judge=judge)
    )

    print(f"\nGenerated {len(test_cases)} test cases:")
    for tc in test_cases:
        print(f"  [{tc.id[:8]}...] {tc.name}")
        print(f"    Q: {tc.query}")
        print(f"    A: {tc.response[:80]}...")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

SECTIONS = {
    "single": section_single,
    "batch": section_batch,
    "custom": section_custom_evaluator,
    "recall": section_context_recall,
    "dataset": section_dataset,
    "reports": section_reports,
    "observability": section_observability,
    "generate": section_generate,
}


def main() -> None:
    """Run one or all example sections."""
    import argparse

    parser = argparse.ArgumentParser(description="RagaliQ basic usage examples")
    parser.add_argument(
        "--section",
        choices=list(SECTIONS.keys()),
        default=None,
        help="Run a specific section (default: run all)",
    )
    args = parser.parse_args()

    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print("Set it with: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    if args.section:
        SECTIONS[args.section]()
    else:
        for name, fn in SECTIONS.items():
            try:
                fn()
            except Exception as exc:
                print(f"\n[{name}] ERROR: {exc}")
                raise

    print("\nDone.")


if __name__ == "__main__":
    main()
