"""Unit tests for JSONReporter."""

from __future__ import annotations

import json

from ragaliq.core.test_case import EvalStatus, RAGTestCase, RAGTestResult
from ragaliq.reports.json_export import JSONReporter


def _make_tc(name: str = "Test", id: str = "t1") -> RAGTestCase:
    return RAGTestCase(
        id=id,
        name=name,
        query="What is X?",
        context=["X is Y."],
        response="X is Y.",
    )


def _make_result(
    name: str = "Test",
    id: str = "t1",
    status: EvalStatus = EvalStatus.PASSED,
    scores: dict | None = None,
    details: dict | None = None,
    execution_time_ms: int = 100,
    judge_tokens_used: int = 50,
) -> RAGTestResult:
    return RAGTestResult(
        test_case=_make_tc(name=name, id=id),
        status=status,
        scores=scores if scores is not None else {},
        details=details if details is not None else {},
        execution_time_ms=execution_time_ms,
        judge_tokens_used=judge_tokens_used,
    )


def _export(results: list[RAGTestResult], threshold: float = 0.7) -> dict:
    """Export results and parse the JSON back to a dict."""
    raw = JSONReporter(threshold=threshold).export(results)
    return json.loads(raw)


class TestJSONStructure:
    """Test top-level JSON document structure."""

    def test_returns_valid_json(self) -> None:
        """export() returns a string that parses as valid JSON."""
        result = _make_result(scores={"faithfulness": 0.9})
        raw = JSONReporter().export([result])
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_has_generated_at_field(self) -> None:
        """Document includes a generated_at ISO timestamp."""
        doc = _export([_make_result()])
        assert "generated_at" in doc
        assert "T" in doc["generated_at"]  # ISO format check

    def test_has_threshold_field(self) -> None:
        """Document includes the configured threshold value."""
        doc = _export([_make_result()], threshold=0.85)
        assert doc["threshold"] == 0.85

    def test_has_summary_key(self) -> None:
        """Document has a summary key."""
        doc = _export([_make_result()])
        assert "summary" in doc

    def test_has_results_key(self) -> None:
        """Document has a results key."""
        doc = _export([_make_result()])
        assert "results" in doc

    def test_pretty_printed(self) -> None:
        """JSON output is indented (pretty-printed)."""
        raw = JSONReporter().export([_make_result()])
        assert "\n" in raw
        assert "  " in raw


class TestSummarySection:
    """Test the summary section of the JSON document."""

    def test_summary_total(self) -> None:
        """summary.total matches the number of results."""
        r1 = _make_result(id="t1")
        r2 = _make_result(id="t2")
        doc = _export([r1, r2])
        assert doc["summary"]["total"] == 2

    def test_summary_passed_count(self) -> None:
        """summary.passed counts results with PASSED status."""
        r1 = _make_result(id="t1", status=EvalStatus.PASSED)
        r2 = _make_result(id="t2", status=EvalStatus.FAILED)
        doc = _export([r1, r2])
        assert doc["summary"]["passed"] == 1

    def test_summary_failed_count(self) -> None:
        """summary.failed counts non-passing results."""
        r1 = _make_result(id="t1", status=EvalStatus.PASSED)
        r2 = _make_result(id="t2", status=EvalStatus.FAILED)
        doc = _export([r1, r2])
        assert doc["summary"]["failed"] == 1

    def test_summary_errored_count(self) -> None:
        """summary.errored counts results with ERROR status."""
        r1 = _make_result(id="t1", status=EvalStatus.PASSED)
        r2 = _make_result(id="t2", status=EvalStatus.ERROR)
        doc = _export([r1, r2])
        assert doc["summary"]["errored"] == 1

    def test_summary_pass_rate(self) -> None:
        """summary.pass_rate is passed/total."""
        r1 = _make_result(id="t1", status=EvalStatus.PASSED)
        r2 = _make_result(id="t2", status=EvalStatus.FAILED)
        doc = _export([r1, r2])
        assert doc["summary"]["pass_rate"] == 0.5

    def test_summary_pass_rate_empty(self) -> None:
        """summary.pass_rate is 0.0 for empty results."""
        doc = _export([])
        assert doc["summary"]["pass_rate"] == 0.0

    def test_summary_total_tokens(self) -> None:
        """summary.total_tokens sums judge_tokens_used across results."""
        r1 = _make_result(id="t1", judge_tokens_used=100)
        r2 = _make_result(id="t2", judge_tokens_used=250)
        doc = _export([r1, r2])
        assert doc["summary"]["total_tokens"] == 350

    def test_summary_total_execution_ms(self) -> None:
        """summary.total_execution_ms sums execution_time_ms across results."""
        r1 = _make_result(id="t1", execution_time_ms=500)
        r2 = _make_result(id="t2", execution_time_ms=800)
        doc = _export([r1, r2])
        assert doc["summary"]["total_execution_ms"] == 1300

    def test_summary_evaluators_keys(self) -> None:
        """summary.evaluators contains a key per evaluator."""
        result = _make_result(scores={"faithfulness": 0.9, "relevance": 0.8})
        doc = _export([result])
        ev = doc["summary"]["evaluators"]
        assert "faithfulness" in ev
        assert "relevance" in ev

    def test_summary_evaluator_passed_count(self) -> None:
        """Per-evaluator passed count respects the threshold."""
        r1 = _make_result(id="t1", scores={"faithfulness": 0.9})
        r2 = _make_result(id="t2", scores={"faithfulness": 0.5})
        doc = _export([r1, r2], threshold=0.7)
        ev = doc["summary"]["evaluators"]["faithfulness"]
        assert ev["passed"] == 1
        assert ev["failed"] == 1

    def test_summary_evaluator_avg_score(self) -> None:
        """Per-evaluator avg_score is the mean of all scores."""
        r1 = _make_result(id="t1", scores={"faithfulness": 0.9})
        r2 = _make_result(id="t2", scores={"faithfulness": 0.7})
        doc = _export([r1, r2])
        avg = doc["summary"]["evaluators"]["faithfulness"]["avg_score"]
        assert abs(avg - 0.8) < 0.001


class TestResultsSection:
    """Test individual result entries in the results array."""

    def test_results_length_matches_input(self) -> None:
        """results array has one entry per input result."""
        r1 = _make_result(id="t1")
        r2 = _make_result(id="t2")
        doc = _export([r1, r2])
        assert len(doc["results"]) == 2

    def test_result_id_field(self) -> None:
        """Each result entry has the test case id."""
        result = _make_result(id="tc-42")
        doc = _export([result])
        assert doc["results"][0]["id"] == "tc-42"

    def test_result_name_field(self) -> None:
        """Each result entry has the test case name."""
        result = _make_result(name="Capital query")
        doc = _export([result])
        assert doc["results"][0]["name"] == "Capital query"

    def test_result_status_is_string(self) -> None:
        """Status field is serialized as a plain string."""
        result = _make_result(status=EvalStatus.PASSED)
        doc = _export([result])
        assert doc["results"][0]["status"] == "passed"

    def test_result_passed_field(self) -> None:
        """passed field is a boolean."""
        r_pass = _make_result(id="t1", status=EvalStatus.PASSED)
        r_fail = _make_result(id="t2", status=EvalStatus.FAILED)
        doc = _export([r_pass, r_fail])
        assert doc["results"][0]["passed"] is True
        assert doc["results"][1]["passed"] is False

    def test_result_scores_dict(self) -> None:
        """scores dict is preserved as-is."""
        result = _make_result(scores={"faithfulness": 0.92, "relevance": 0.85})
        doc = _export([result])
        assert doc["results"][0]["scores"]["faithfulness"] == 0.92
        assert doc["results"][0]["scores"]["relevance"] == 0.85

    def test_result_execution_time_ms(self) -> None:
        """execution_time_ms field is included."""
        result = _make_result(execution_time_ms=1234)
        doc = _export([result])
        assert doc["results"][0]["execution_time_ms"] == 1234

    def test_result_judge_tokens_used(self) -> None:
        """judge_tokens_used field is included."""
        result = _make_result(judge_tokens_used=777)
        doc = _export([result])
        assert doc["results"][0]["judge_tokens_used"] == 777

    def test_result_test_case_nested(self) -> None:
        """Each result entry contains a nested test_case object."""
        result = _make_result(name="My test")
        doc = _export([result])
        tc = doc["results"][0]["test_case"]
        assert tc["name"] == "My test"
        assert "query" in tc
        assert "context" in tc
        assert "response" in tc

    def test_result_test_case_context_is_list(self) -> None:
        """test_case.context is serialized as a list."""
        result = _make_result()
        doc = _export([result])
        assert isinstance(doc["results"][0]["test_case"]["context"], list)


class TestEdgeCases:
    """Test edge cases and empty states."""

    def test_empty_results(self) -> None:
        """Empty results list produces valid JSON with empty results array."""
        doc = _export([])
        assert doc["results"] == []
        assert doc["summary"]["total"] == 0

    def test_no_evaluators(self) -> None:
        """Result with no scores produces empty evaluators dict in summary."""
        result = _make_result(scores={})
        doc = _export([result])
        assert doc["summary"]["evaluators"] == {}

    def test_threshold_used_in_evaluator_stats(self) -> None:
        """Custom threshold is applied when computing per-evaluator pass/fail."""
        result = _make_result(scores={"faithfulness": 0.65})
        doc_low = _export([result], threshold=0.6)  # 0.65 >= 0.6 → passes
        doc_high = _export([result], threshold=0.7)  # 0.65 < 0.7 → fails
        assert doc_low["summary"]["evaluators"]["faithfulness"]["passed"] == 1
        assert doc_high["summary"]["evaluators"]["faithfulness"]["passed"] == 0
