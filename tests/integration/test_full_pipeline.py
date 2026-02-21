"""Full end-to-end integration tests covering the complete user workflow.

These tests exercise the outer integration layer that test_pipeline.py does not:
    Dataset file → DatasetLoader → RagaliQ runner (all 5 evaluators)
                 → Reporters (Console, HTML, JSON)

No network calls — the FakeTransport returns canned JSON and everything
else runs as production code.  This catches integration issues across
subsystem boundaries: dataset parsing → evaluator routing → reporter
serialization.
"""

import json
from pathlib import Path

import pytest
from rich.console import Console

from ragaliq.core.runner import RagaliQ
from ragaliq.core.test_case import EvalStatus, RAGTestCase, RAGTestResult
from ragaliq.datasets.loader import DatasetLoader
from ragaliq.judges.base_judge import BaseJudge
from ragaliq.judges.trace import TraceCollector
from ragaliq.judges.transport import TransportResponse
from ragaliq.reports.console import ConsoleReporter
from ragaliq.reports.html import HTMLReporter
from ragaliq.reports.json_export import JSONReporter

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# FakeTransport — handles all 5 evaluator operation types
# ---------------------------------------------------------------------------


class FullPipelineTransport:
    """Transport that returns canned JSON for all evaluator operations.

    Extends the operation detection to handle context_precision
    (evaluate_relevance per document) and context_recall (verify_claim
    per expected fact), in addition to faithfulness, hallucination,
    and relevance operations.
    """

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self.call_count = 0
        self.calls: list[dict[str, str]] = []
        self._responses = responses or {}

    async def send(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "fake-model",
        temperature: float = 0.0,  # noqa: ARG002
        max_tokens: int = 1024,  # noqa: ARG002
    ) -> TransportResponse:
        self.call_count += 1
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        text = self._resolve_response(system_prompt, user_prompt)
        return TransportResponse(
            text=text,
            input_tokens=50,
            output_tokens=30,
            model=model,
        )

    def _resolve_response(self, system_prompt: str, user_prompt: str) -> str:
        """Pick the right canned response based on prompt keywords."""
        combined = (system_prompt + user_prompt).lower()

        # Check explicit overrides first
        for key, response in self._responses.items():
            if key.lower() in combined:
                return response

        # Faithfulness / Hallucination claim pipeline
        if "extract" in combined and "claim" in combined:
            return json.dumps({"claims": ["Claim A", "Claim B"]})
        if "verify" in combined and "claim" in combined:
            return json.dumps({"verdict": "SUPPORTED", "evidence": "Confirmed by context."})

        # Relevance (used by both relevance evaluator and context_precision)
        if "relevance" in combined or "relevant" in combined:
            return json.dumps({"score": 0.85, "reasoning": "Relevant to the query."})

        # Faithfulness direct (fallback if prompt says "faithfulness")
        if "faithfulness" in combined:
            return json.dumps({"score": 0.9, "reasoning": "Mostly faithful."})

        return json.dumps({"score": 0.75, "reasoning": "Default response."})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def transport():
    return FullPipelineTransport()


@pytest.fixture
def judge(transport):
    return BaseJudge(transport=transport)


@pytest.fixture
def test_case_with_facts():
    """Test case with expected_facts for context_recall evaluator."""
    return RAGTestCase(
        id="e2e-001",
        name="Python list methods",
        query="How do I add an item to a list in Python?",
        context=[
            "Python lists have several methods: append() adds to end, insert() adds at position.",
            "The append() method takes a single argument and adds it to the end of the list.",
        ],
        response="Use the append() method to add items to a Python list.",
        expected_facts=["append() adds items to list", "append() adds to the end"],
    )


@pytest.fixture
def multiple_test_cases():
    """Multiple test cases for batch evaluation."""
    return [
        RAGTestCase(
            id="batch-001",
            name="Capital of France",
            query="What is the capital of France?",
            context=["The capital of France is Paris."],
            response="The capital of France is Paris.",
            expected_facts=["Paris is the capital of France"],
        ),
        RAGTestCase(
            id="batch-002",
            name="RAG components",
            query="What are the main components of a RAG system?",
            context=[
                "A RAG system has a retriever, document store, and generator.",
                "The retriever uses vector embeddings.",
            ],
            response="A RAG system has three main components: retriever, store, and generator.",
            expected_facts=["RAG has three components", "retriever fetches documents"],
        ),
    ]


# ---------------------------------------------------------------------------
# Test: Dataset loading → Runner → All 5 evaluators
# ---------------------------------------------------------------------------


class TestDatasetToEvaluation:
    """Load a real dataset file and run all 5 evaluators through the full pipeline."""

    @pytest.mark.asyncio
    async def test_load_fixture_and_evaluate_all_evaluators(self, judge):
        """Dataset file → DatasetLoader → RagaliQ(all 5 evaluators) → results."""
        dataset = DatasetLoader.load(FIXTURES_DIR / "sample_dataset.json")
        assert len(dataset.test_cases) >= 1

        runner = RagaliQ(
            judge=judge,
            evaluators=[
                "faithfulness",
                "relevance",
                "hallucination",
                "context_precision",
                "context_recall",
            ],
        )

        # Use the first test case (has expected_facts)
        tc = dataset.test_cases[0]
        result = await runner.evaluate_async(tc)

        assert isinstance(result, RAGTestResult)
        assert result.status in {EvalStatus.PASSED, EvalStatus.FAILED}
        assert "faithfulness" in result.scores
        assert "relevance" in result.scores
        assert "hallucination" in result.scores
        assert "context_precision" in result.scores
        assert "context_recall" in result.scores
        assert all(0.0 <= s <= 1.0 for s in result.scores.values())
        assert result.execution_time_ms >= 0
        assert result.judge_tokens_used > 0

    @pytest.mark.asyncio
    async def test_batch_evaluation_from_dataset(self, judge):
        """All test cases from dataset evaluated in batch."""
        dataset = DatasetLoader.load(FIXTURES_DIR / "sample_dataset.json")

        runner = RagaliQ(
            judge=judge,
            evaluators=["faithfulness", "relevance"],
        )

        results = await runner.evaluate_batch_async(dataset.test_cases)

        assert len(results) == len(dataset.test_cases)
        assert all(isinstance(r, RAGTestResult) for r in results)
        assert all(r.status in {EvalStatus.PASSED, EvalStatus.FAILED} for r in results)

    @pytest.mark.asyncio
    async def test_all_five_evaluators_produce_details(self, judge, test_case_with_facts):
        """Each evaluator produces reasoning in the details dict."""
        runner = RagaliQ(
            judge=judge,
            evaluators=[
                "faithfulness",
                "relevance",
                "hallucination",
                "context_precision",
                "context_recall",
            ],
        )

        result = await runner.evaluate_async(test_case_with_facts)

        for evaluator_name in [
            "faithfulness",
            "relevance",
            "hallucination",
            "context_precision",
            "context_recall",
        ]:
            assert evaluator_name in result.details
            detail = result.details[evaluator_name]
            assert "reasoning" in detail
            assert "passed" in detail
            assert "raw" in detail


# ---------------------------------------------------------------------------
# Test: Evaluation → Reporters
# ---------------------------------------------------------------------------


class TestEvaluationToReporters:
    """Verify that reporter output is valid after full pipeline evaluation."""

    @pytest.fixture
    async def evaluation_results(self, judge, multiple_test_cases):
        """Run the full pipeline and return results for reporter tests."""
        runner = RagaliQ(
            judge=judge,
            evaluators=["faithfulness", "relevance", "hallucination"],
        )
        return await runner.evaluate_batch_async(multiple_test_cases)

    @pytest.mark.asyncio
    async def test_json_reporter_produces_valid_json(self, evaluation_results):
        """JSON reporter serializes full pipeline results correctly."""
        reporter = JSONReporter(threshold=0.7)
        json_str = reporter.export(evaluation_results)

        doc = json.loads(json_str)
        assert "summary" in doc
        assert "results" in doc
        assert doc["summary"]["total"] == 2
        assert doc["summary"]["passed"] + doc["summary"]["failed"] == 2
        assert "evaluators" in doc["summary"]
        assert "faithfulness" in doc["summary"]["evaluators"]
        assert "relevance" in doc["summary"]["evaluators"]
        assert "hallucination" in doc["summary"]["evaluators"]

        for result in doc["results"]:
            assert "scores" in result
            assert "details" in result
            assert "test_case" in result

    @pytest.mark.asyncio
    async def test_html_reporter_produces_valid_html(self, evaluation_results):
        """HTML reporter renders full pipeline results without errors."""
        reporter = HTMLReporter(threshold=0.7)
        html_str = reporter.export(evaluation_results)

        assert "<html" in html_str.lower()
        assert "faithfulness" in html_str.lower()
        assert "relevance" in html_str.lower()
        assert len(html_str) > 500  # Sanity: non-trivial HTML

    @pytest.mark.asyncio
    async def test_console_reporter_renders_without_error(self, evaluation_results):
        """Console reporter renders full pipeline results to a captured console."""
        console = Console(record=True, width=120)
        reporter = ConsoleReporter(console=console, threshold=0.7)

        reporter.report(evaluation_results)

        text = console.export_text()
        assert "Evaluation Results" in text
        assert "Summary" in text
        assert len(text) > 100


# ---------------------------------------------------------------------------
# Test: Full workflow — Dataset → Evaluation → JSON report round-trip
# ---------------------------------------------------------------------------


class TestFullWorkflowRoundTrip:
    """End-to-end workflow: dataset file → evaluate → report → verify."""

    @pytest.mark.asyncio
    async def test_dataset_evaluate_report_roundtrip(self, judge):
        """The complete user workflow as a single test."""
        # Step 1: Load dataset
        dataset = DatasetLoader.load(FIXTURES_DIR / "sample_dataset.json")

        # Step 2: Evaluate
        runner = RagaliQ(
            judge=judge,
            evaluators=["faithfulness", "relevance"],
        )
        results = await runner.evaluate_batch_async(dataset.test_cases)

        # Step 3: Generate JSON report
        reporter = JSONReporter(threshold=0.7)
        json_str = reporter.export(results)

        # Step 4: Verify the report is a valid, self-consistent document
        doc = json.loads(json_str)
        assert doc["summary"]["total"] == len(dataset.test_cases)
        assert len(doc["results"]) == len(dataset.test_cases)

        # Verify test case IDs match the input dataset
        result_ids = {r["id"] for r in doc["results"]}
        input_ids = {tc.id for tc in dataset.test_cases}
        assert result_ids == input_ids

    def test_sync_api_evaluate(self, transport, test_case_with_facts):
        """Sync evaluate() wrapper works end-to-end.

        This test is intentionally synchronous — it verifies that the sync
        wrapper (which calls asyncio.run() internally) works correctly when
        called from outside an event loop.
        """
        judge = BaseJudge(transport=transport)
        runner = RagaliQ(
            judge=judge,
            evaluators=["faithfulness", "relevance"],
        )

        result = runner.evaluate(test_case_with_facts)

        assert isinstance(result, RAGTestResult)
        assert result.status in {EvalStatus.PASSED, EvalStatus.FAILED}
        assert "faithfulness" in result.scores
        assert "relevance" in result.scores


# ---------------------------------------------------------------------------
# Test: Trace collection across full pipeline
# ---------------------------------------------------------------------------


class TestTraceCollectionFullPipeline:
    """Verify trace collection works across all 5 evaluators."""

    @pytest.mark.asyncio
    async def test_traces_across_all_evaluators(self, transport, test_case_with_facts):
        """TraceCollector captures operations from all 5 evaluators."""
        collector = TraceCollector()
        judge = BaseJudge(transport=transport, trace_collector=collector)
        runner = RagaliQ(
            judge=judge,
            evaluators=[
                "faithfulness",
                "relevance",
                "hallucination",
                "context_precision",
                "context_recall",
            ],
        )

        await runner.evaluate_async(test_case_with_facts)

        # All traces should be successful
        assert all(t.success for t in collector.traces)

        # Verify we have traces from multiple operation types
        operations = {t.operation for t in collector.traces}
        assert "extract_claims" in operations
        assert "verify_claim" in operations
        assert "evaluate_relevance" in operations

        # Total tokens should be accumulated
        total_tokens = sum(t.input_tokens + t.output_tokens for t in collector.traces)
        assert total_tokens > 0


# ---------------------------------------------------------------------------
# Test: Error handling across full pipeline
# ---------------------------------------------------------------------------


class TestErrorHandlingFullPipeline:
    """Verify error envelopes work across the full workflow."""

    @pytest.mark.asyncio
    async def test_transport_failure_produces_error_envelope(self):
        """Transport exception → error envelope in batch results, not crash."""

        class FailingTransport:
            async def send(self, **kwargs):  # noqa: ARG002
                raise ConnectionError("Network unavailable")

        judge = BaseJudge(transport=FailingTransport())
        runner = RagaliQ(
            judge=judge,
            evaluators=["faithfulness"],
            fail_fast=False,
        )

        test_cases = [
            RAGTestCase(
                id="err-001",
                name="Error case",
                query="test",
                context=["ctx"],
                response="resp",
            ),
        ]

        results = await runner.evaluate_batch_async(test_cases)

        assert len(results) == 1
        assert results[0].status == EvalStatus.ERROR

        # Reporter should still handle error results gracefully
        reporter = JSONReporter(threshold=0.7)
        json_str = reporter.export(results)
        doc = json.loads(json_str)
        assert doc["summary"]["errored"] == 1
