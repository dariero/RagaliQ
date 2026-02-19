"""Unit tests for HTMLReporter."""

from __future__ import annotations

from ragaliq.core.test_case import EvalStatus, RAGTestCase, RAGTestResult
from ragaliq.reports.html import HTMLReporter


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
    judge_tokens_used: int = 0,
) -> RAGTestResult:
    return RAGTestResult(
        test_case=_make_tc(name=name, id=id),
        status=status,
        scores=scores if scores is not None else {},
        details=details if details is not None else {},
        execution_time_ms=execution_time_ms,
        judge_tokens_used=judge_tokens_used,
    )


def _export(results: list[RAGTestResult], threshold: float = 0.7) -> str:
    return HTMLReporter(threshold=threshold).export(results)


class TestHTMLStructure:
    """Test that output is a well-formed HTML document."""

    def test_returns_string(self) -> None:
        """export() returns a string."""
        result = _make_result()
        html = _export([result])
        assert isinstance(html, str)

    def test_has_doctype(self) -> None:
        """Output starts with a DOCTYPE declaration."""
        html = _export([_make_result()])
        assert "<!DOCTYPE html>" in html

    def test_has_html_tag(self) -> None:
        """Output contains opening and closing html tags."""
        html = _export([_make_result()])
        assert "<html" in html
        assert "</html>" in html

    def test_has_head_and_body(self) -> None:
        """Output contains head and body sections."""
        html = _export([_make_result()])
        assert "<head>" in html
        assert "<body>" in html

    def test_has_embedded_style(self) -> None:
        """Output contains embedded CSS (no external stylesheet links)."""
        html = _export([_make_result()])
        assert "<style>" in html
        # Must NOT reference external stylesheets
        assert 'rel="stylesheet"' not in html

    def test_no_script_tags(self) -> None:
        """Output contains no JavaScript (truly self-contained)."""
        html = _export([_make_result()])
        assert "<script" not in html

    def test_has_title(self) -> None:
        """Output has a page title."""
        html = _export([_make_result()])
        assert "<title>" in html
        assert "RagaliQ" in html


class TestHTMLContent:
    """Test that report content appears in the output."""

    def test_shows_test_case_name(self) -> None:
        """Test case name appears in the HTML output."""
        result = _make_result(name="Capital query")
        html = _export([result])
        assert "Capital query" in html

    def test_shows_passed_status_badge(self) -> None:
        """PASSED status is represented in the output."""
        result = _make_result(status=EvalStatus.PASSED)
        html = _export([result])
        assert "passed" in html.lower()

    def test_shows_failed_status_badge(self) -> None:
        """FAILED status is represented in the output."""
        result = _make_result(status=EvalStatus.FAILED, scores={"faithfulness": 0.3})
        html = _export([result])
        assert "failed" in html.lower()

    def test_shows_error_status_badge(self) -> None:
        """ERROR status is represented in the output."""
        result = _make_result(status=EvalStatus.ERROR)
        html = _export([result])
        assert "error" in html.lower()

    def test_shows_evaluator_score(self) -> None:
        """Evaluator score value appears in the HTML."""
        result = _make_result(scores={"faithfulness": 0.92})
        html = _export([result])
        assert "0.92" in html

    def test_shows_evaluator_column_name(self) -> None:
        """Evaluator name appears as a column header."""
        result = _make_result(scores={"faithfulness": 0.9})
        html = _export([result])
        assert "faithfulness" in html.lower() or "Faithfulness" in html

    def test_shows_all_test_case_names(self) -> None:
        """All test case names appear when multiple results are present."""
        r1 = _make_result(name="Case Alpha", id="t1", scores={"faithfulness": 0.9})
        r2 = _make_result(name="Case Beta", id="t2", scores={"faithfulness": 0.5})
        html = _export([r1, r2])
        assert "Case Alpha" in html
        assert "Case Beta" in html

    def test_shows_summary_totals(self) -> None:
        """Summary card shows total number of test cases."""
        r1 = _make_result(id="t1")
        r2 = _make_result(id="t2")
        html = _export([r1, r2])
        assert "2" in html  # total count appears somewhere

    def test_shows_pass_rate(self) -> None:
        """Pass rate percentage appears in the summary cards."""
        r1 = _make_result(id="t1", status=EvalStatus.PASSED)
        r2 = _make_result(id="t2", status=EvalStatus.FAILED)
        html = _export([r1, r2])
        assert "50%" in html

    def test_shows_token_count(self) -> None:
        """Token count appears in the summary when nonzero."""
        result = _make_result(judge_tokens_used=1500)
        html = _export([result])
        assert "1,500" in html or "1500" in html


class TestHTMLFailureDetails:
    """Test expandable failure detail sections."""

    def test_shows_details_element_for_failed(self) -> None:
        """HTML5 details element is present for failed results with details."""
        result = _make_result(
            status=EvalStatus.FAILED,
            scores={"faithfulness": 0.3},
            details={"faithfulness": {"reasoning": "Missing key fact", "passed": False, "raw": {}}},
        )
        html = _export([result])
        assert "<details>" in html

    def test_shows_reasoning_text(self) -> None:
        """Reasoning text appears in the HTML for failing evaluators."""
        result = _make_result(
            status=EvalStatus.FAILED,
            scores={"faithfulness": 0.3},
            details={"faithfulness": {"reasoning": "Missing key fact", "passed": False, "raw": {}}},
        )
        html = _export([result])
        assert "Missing key fact" in html

    def test_shows_error_text(self) -> None:
        """Error message appears in the HTML for errored evaluators."""
        result = _make_result(
            status=EvalStatus.ERROR,
            scores={"faithfulness": 0.0},
            details={"faithfulness": {"error": "API timeout", "passed": False, "raw": {}}},
        )
        html = _export([result])
        assert "API timeout" in html

    def test_no_details_element_for_passing_results(self) -> None:
        """No details element appears when all results pass."""
        result = _make_result(
            status=EvalStatus.PASSED,
            scores={"faithfulness": 0.9},
            details={"faithfulness": {"reasoning": "All good", "passed": True, "raw": {}}},
        )
        html = _export([result])
        assert "<details>" not in html


class TestHTMLEdgeCases:
    """Test edge cases."""

    def test_empty_results_no_crash(self) -> None:
        """Empty results list renders without error."""
        html = _export([])
        assert "<!DOCTYPE html>" in html

    def test_no_evaluators_no_crash(self) -> None:
        """Result with no scores renders without error."""
        result = _make_result(scores={})
        html = _export([result])
        assert "Test" in html

    def test_threshold_respected_in_template(self) -> None:
        """Custom threshold affects score pass/fail coloring class."""
        result = _make_result(scores={"faithfulness": 0.65})
        html_low = _export([result], threshold=0.6)  # 0.65 passes
        html_high = _export([result], threshold=0.7)  # 0.65 fails
        # score-pass class for low threshold, score-fail for high
        assert "score-pass" in html_low
        assert "score-fail" in html_high

    def test_special_chars_escaped(self) -> None:
        """HTML special characters in test case names are escaped."""
        result = _make_result(name="Test <query> & more")
        html = _export([result])
        # Jinja2 autoescape should encode < > &
        assert "<query>" not in html
        assert "&lt;" in html or "Test" in html  # at minimum, raw < not injected

    def test_default_reporter_has_threshold(self) -> None:
        """HTMLReporter created without args has default threshold."""
        reporter = HTMLReporter()
        assert reporter._threshold == 0.7
