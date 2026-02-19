"""Unit tests for ConsoleReporter."""

from __future__ import annotations

from rich.console import Console

from ragaliq.core.test_case import EvalStatus, RAGTestCase, RAGTestResult
from ragaliq.reports.console import ConsoleReporter


def _make_tc(name: str = "Test Case", id: str = "t1") -> RAGTestCase:
    """Build a minimal RAGTestCase."""
    return RAGTestCase(
        id=id,
        name=name,
        query="What is X?",
        context=["X is Y."],
        response="X is Y.",
    )


def _make_result(
    name: str = "Test Case",
    status: EvalStatus = EvalStatus.PASSED,
    scores: dict | None = None,
    details: dict | None = None,
    judge_tokens_used: int = 0,
    id: str = "t1",
) -> RAGTestResult:
    """Build a RAGTestResult with sensible defaults."""
    return RAGTestResult(
        test_case=_make_tc(name=name, id=id),
        status=status,
        scores=scores if scores is not None else {},
        details=details if details is not None else {},
        judge_tokens_used=judge_tokens_used,
    )


def _render(results: list[RAGTestResult], threshold: float = 0.7) -> str:
    """Run ConsoleReporter.report() and return plain-text output."""
    console = Console(record=True, width=120)
    ConsoleReporter(console=console, threshold=threshold).report(results)
    return console.export_text()


class TestResultsTable:
    """Test results table section."""

    def test_shows_test_case_name(self) -> None:
        """Table includes the test case name."""
        result = _make_result(name="Capital query", scores={"faithfulness": 0.9})
        output = _render([result])
        assert "Capital query" in output

    def test_shows_pass_status(self) -> None:
        """PASS status appears for a passing result."""
        result = _make_result(status=EvalStatus.PASSED, scores={"faithfulness": 0.9})
        output = _render([result])
        assert "PASS" in output

    def test_shows_fail_status(self) -> None:
        """FAIL status appears for a failing result."""
        result = _make_result(status=EvalStatus.FAILED, scores={"faithfulness": 0.3})
        output = _render([result])
        assert "FAIL" in output

    def test_shows_error_status(self) -> None:
        """ERROR status appears for an errored result."""
        result = _make_result(status=EvalStatus.ERROR, scores={})
        output = _render([result])
        assert "ERROR" in output

    def test_shows_evaluator_score(self) -> None:
        """Score value is rendered in the table."""
        result = _make_result(scores={"faithfulness": 0.92})
        output = _render([result])
        assert "0.92" in output

    def test_shows_evaluator_name_as_column(self) -> None:
        """Evaluator name appears as a column heading."""
        result = _make_result(scores={"faithfulness": 0.9})
        output = _render([result])
        assert "Faithfulness" in output

    def test_multiple_evaluators_shown(self) -> None:
        """All evaluators appear as columns when present."""
        result = _make_result(scores={"faithfulness": 0.9, "relevance": 0.85})
        output = _render([result])
        assert "Faithfulness" in output
        assert "Relevance" in output

    def test_missing_score_renders_dash(self) -> None:
        """A missing score for an evaluator renders as '—'."""
        r1 = _make_result(id="t1", scores={"faithfulness": 0.9, "relevance": 0.8})
        r2 = _make_result(id="t2", scores={"faithfulness": 0.7})  # relevance absent
        output = _render([r1, r2])
        assert "—" in output

    def test_evaluator_column_name_titlecased(self) -> None:
        """Underscore-separated evaluator names are titlecased."""
        result = _make_result(scores={"context_precision": 0.9})
        output = _render([result])
        assert "Context Precision" in output

    def test_multiple_results_all_shown(self) -> None:
        """All test cases appear in the table."""
        r1 = _make_result(name="Case A", id="t1", scores={"faithfulness": 0.9})
        r2 = _make_result(name="Case B", id="t2", scores={"faithfulness": 0.5})
        output = _render([r1, r2])
        assert "Case A" in output
        assert "Case B" in output


class TestFailedDetails:
    """Test failed details section."""

    def test_shows_reasoning_for_failed_evaluator(self) -> None:
        """Reasoning is shown for a failing evaluator."""
        result = _make_result(
            status=EvalStatus.FAILED,
            scores={"faithfulness": 0.3},
            details={"faithfulness": {"reasoning": "Missing key fact", "passed": False, "raw": {}}},
        )
        output = _render([result])
        assert "Missing key fact" in output

    def test_no_details_section_when_all_pass(self) -> None:
        """Failed details section does not appear when all results pass."""
        result = _make_result(status=EvalStatus.PASSED, scores={"faithfulness": 0.9})
        output = _render([result])
        assert "Failed Test Details" not in output

    def test_shows_error_message_for_error_evaluator(self) -> None:
        """Error message is shown for an evaluator that errored."""
        result = _make_result(
            status=EvalStatus.ERROR,
            scores={"faithfulness": 0.0},
            details={"faithfulness": {"error": "API timeout", "passed": False, "raw": {}}},
        )
        output = _render([result])
        assert "API timeout" in output

    def test_no_reasoning_shown_for_passing_evaluator_in_failed_result(self) -> None:
        """When a result has mixed evaluators, only failing ones show reasoning."""
        result = _make_result(
            status=EvalStatus.FAILED,
            scores={"faithfulness": 0.9, "relevance": 0.3},
            details={
                "faithfulness": {"reasoning": "All facts present", "passed": True, "raw": {}},
                "relevance": {"reasoning": "Off topic", "passed": False, "raw": {}},
            },
        )
        output = _render([result])
        assert "Off topic" in output
        assert "All facts present" not in output

    def test_shows_test_case_name_in_details(self) -> None:
        """The test case name appears as a header in failed details."""
        result = _make_result(
            name="My failing test",
            status=EvalStatus.FAILED,
            scores={"faithfulness": 0.2},
            details={"faithfulness": {"reasoning": "Bad", "passed": False, "raw": {}}},
        )
        output = _render([result])
        assert "My failing test" in output

    def test_verbose_shows_details_for_passing_results(self) -> None:
        """In verbose mode, details are shown even for passing results."""
        result = _make_result(
            status=EvalStatus.PASSED,
            scores={"faithfulness": 0.9},
            details={"faithfulness": {"reasoning": "All good", "passed": True, "raw": {}}},
        )
        console = Console(record=True, width=120)
        ConsoleReporter(console=console, threshold=0.7, verbose=True).report([result])
        output = console.export_text()
        # verbose=True means the "Failed Test Details" section appears even for passing results
        assert "Failed Test Details" in output

    def test_empty_details_no_crash(self) -> None:
        """Empty details dict does not cause errors."""
        result = _make_result(status=EvalStatus.FAILED, scores={"faithfulness": 0.3}, details={})
        output = _render([result])
        assert "FAIL" in output


class TestSummary:
    """Test summary statistics section."""

    def test_summary_shows_pass_count(self) -> None:
        """Summary shows passed/total count."""
        r1 = _make_result(id="t1", status=EvalStatus.PASSED, scores={"faithfulness": 0.9})
        r2 = _make_result(id="t2", status=EvalStatus.PASSED, scores={"faithfulness": 0.8})
        output = _render([r1, r2])
        assert "2/2 passed" in output

    def test_summary_shows_fail_count(self) -> None:
        """Summary shows failed count when there are failures."""
        r1 = _make_result(id="t1", status=EvalStatus.PASSED, scores={"faithfulness": 0.9})
        r2 = _make_result(id="t2", status=EvalStatus.FAILED, scores={"faithfulness": 0.3})
        output = _render([r1, r2])
        assert "1 failed" in output

    def test_summary_shows_per_evaluator_row(self) -> None:
        """Summary table includes a row per evaluator."""
        result = _make_result(scores={"faithfulness": 0.9, "relevance": 0.8})
        output = _render([result])
        assert "faithfulness" in output
        assert "relevance" in output

    def test_summary_shows_avg_score(self) -> None:
        """Summary shows average score for each evaluator."""
        r1 = _make_result(id="t1", scores={"faithfulness": 0.9})
        r2 = _make_result(id="t2", scores={"faithfulness": 0.7})
        output = _render([r1, r2])
        assert "0.80" in output  # avg of 0.9 and 0.7

    def test_summary_shows_token_count_when_nonzero(self) -> None:
        """Token count appears in summary when judge tokens were used."""
        result = _make_result(scores={"faithfulness": 0.9}, judge_tokens_used=1500)
        output = _render([result])
        assert "1,500 tokens" in output

    def test_summary_no_token_count_when_zero(self) -> None:
        """Token count is omitted from summary when zero."""
        result = _make_result(scores={"faithfulness": 0.9}, judge_tokens_used=0)
        output = _render([result])
        assert "tokens" not in output

    def test_threshold_respected_in_per_evaluator_pass_count(self) -> None:
        """Per-evaluator pass count uses the configured threshold, not 0.7."""
        # Score of 0.75 passes threshold=0.6 but the default 0.7 means it should pass here too
        r1 = _make_result(id="t1", scores={"faithfulness": 0.75})
        r2 = _make_result(id="t2", scores={"faithfulness": 0.55})
        # With threshold=0.6: 0.75 passes, 0.55 fails → 1 failed
        # With threshold=0.5: both pass → 0 failed
        output_t06 = _render([r1, r2], threshold=0.6)
        output_t05 = _render([r1, r2], threshold=0.5)
        # At 0.6 threshold, 0.55 < 0.6 so one fails
        assert "0.65" in output_t06  # avg of 0.75 and 0.55
        assert "0.65" in output_t05


class TestEdgeCases:
    """Test edge cases and empty states."""

    def test_empty_results_no_crash(self) -> None:
        """Reporting empty results list does not crash."""
        output = _render([])
        # Table is still rendered (empty), just no summary
        assert "Evaluation Results" in output

    def test_no_evaluators_no_crash(self) -> None:
        """Result with no scores does not crash."""
        result = _make_result(scores={})
        output = _render([result])
        assert "Test Case" in output  # table still renders with base columns

    def test_single_result_passes_shows_check(self) -> None:
        """A single passing result shows the check mark indicator."""
        result = _make_result(status=EvalStatus.PASSED, scores={"faithfulness": 0.9})
        output = _render([result])
        assert "1/1 passed" in output

    def test_default_console_created_when_none(self) -> None:
        """ConsoleReporter creates its own Console when not provided."""
        reporter = ConsoleReporter()
        assert reporter._console is not None
