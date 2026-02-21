"""Unit tests for GitHub Actions integration helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from ragaliq.core.test_case import EvalStatus, RAGTestCase, RAGTestResult
from ragaliq.integrations.github_actions import (
    create_annotations,
    emit_ci_summary,
    format_summary_markdown,
    is_ci,
    is_github_actions,
    set_output,
    write_step_summary,
)


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
) -> RAGTestResult:
    return RAGTestResult(
        test_case=_make_tc(name=name, id=id),
        status=status,
        scores=scores if scores is not None else {"faithfulness": 0.9},
        details={},
    )


# ---------------------------------------------------------------------------
# CI Detection
# ---------------------------------------------------------------------------


class TestIsCI:
    """Test is_ci() environment detection."""

    def test_true_when_ci_is_true(self) -> None:
        """Returns True when CI=true."""
        with patch.dict(os.environ, {"CI": "true"}):
            assert is_ci() is True

    def test_true_when_ci_is_one(self) -> None:
        """Returns True when CI=1."""
        with patch.dict(os.environ, {"CI": "1"}):
            assert is_ci() is True

    def test_true_when_ci_is_yes(self) -> None:
        """Returns True when CI=yes."""
        with patch.dict(os.environ, {"CI": "yes"}):
            assert is_ci() is True

    def test_true_case_insensitive(self) -> None:
        """Returns True when CI=TRUE (uppercase)."""
        with patch.dict(os.environ, {"CI": "TRUE"}):
            assert is_ci() is True

    def test_false_when_unset(self) -> None:
        """Returns False when CI is not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert is_ci() is False

    def test_false_when_empty(self) -> None:
        """Returns False when CI is empty string."""
        with patch.dict(os.environ, {"CI": ""}):
            assert is_ci() is False

    def test_false_when_false(self) -> None:
        """Returns False when CI=false."""
        with patch.dict(os.environ, {"CI": "false"}):
            assert is_ci() is False


class TestIsGitHubActions:
    """Test is_github_actions() environment detection."""

    def test_true_when_github_actions_is_true(self) -> None:
        """Returns True when GITHUB_ACTIONS=true."""
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}):
            assert is_github_actions() is True

    def test_true_case_insensitive(self) -> None:
        """Returns True when GITHUB_ACTIONS=True (mixed case)."""
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "True"}):
            assert is_github_actions() is True

    def test_false_when_unset(self) -> None:
        """Returns False when GITHUB_ACTIONS is not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert is_github_actions() is False

    def test_false_when_empty(self) -> None:
        """Returns False when GITHUB_ACTIONS is empty."""
        with patch.dict(os.environ, {"GITHUB_ACTIONS": ""}):
            assert is_github_actions() is False


# ---------------------------------------------------------------------------
# set_output
# ---------------------------------------------------------------------------


class TestSetOutput:
    """Test set_output() writing to GITHUB_OUTPUT file."""

    def test_writes_name_value_pair(self) -> None:
        """Writes name=value line to the GITHUB_OUTPUT file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name
        try:
            with patch.dict(os.environ, {"GITHUB_OUTPUT": path}):
                set_output("result", "pass")
            content = Path(path).read_text(encoding="utf-8")
            assert "result=pass\n" in content
        finally:
            os.unlink(path)

    def test_appends_multiple_outputs(self) -> None:
        """Multiple calls append to the same file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name
        try:
            with patch.dict(os.environ, {"GITHUB_OUTPUT": path}):
                set_output("a", "1")
                set_output("b", "2")
            content = Path(path).read_text(encoding="utf-8")
            assert "a=1\n" in content
            assert "b=2\n" in content
        finally:
            os.unlink(path)

    def test_noop_when_env_unset(self) -> None:
        """Does nothing when GITHUB_OUTPUT is not set."""
        with patch.dict(os.environ, {}, clear=True):
            set_output("key", "value")  # should not raise


# ---------------------------------------------------------------------------
# write_step_summary
# ---------------------------------------------------------------------------


class TestWriteStepSummary:
    """Test write_step_summary() writing to GITHUB_STEP_SUMMARY file."""

    def test_writes_markdown_content(self) -> None:
        """Writes markdown to the GITHUB_STEP_SUMMARY file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = f.name
        try:
            with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": path}):
                write_step_summary("## Results\n\nAll passed!")
            content = Path(path).read_text(encoding="utf-8")
            assert "## Results" in content
            assert "All passed!" in content
        finally:
            os.unlink(path)

    def test_appends_trailing_newline(self) -> None:
        """Ensures content ends with a newline."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = f.name
        try:
            with patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": path}):
                write_step_summary("no trailing newline")
            content = Path(path).read_text(encoding="utf-8")
            assert content.endswith("\n")
        finally:
            os.unlink(path)

    def test_noop_when_env_unset(self) -> None:
        """Does nothing when GITHUB_STEP_SUMMARY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            write_step_summary("anything")  # should not raise


# ---------------------------------------------------------------------------
# create_annotations
# ---------------------------------------------------------------------------


class TestCreateAnnotations:
    """Test create_annotations() workflow command output."""

    def test_no_output_for_passing_results(self, capsys: object) -> None:
        """No annotations emitted when all results pass."""
        results = [_make_result(status=EvalStatus.PASSED, scores={"faithfulness": 0.9})]
        create_annotations(results, threshold=0.7)
        out = capsys.readouterr().out  # type: ignore[union-attr]
        assert "::error::" not in out

    def test_emits_error_for_failing_result(self, capsys: object) -> None:
        """Emits ::error:: annotation for a failing result."""
        results = [
            _make_result(
                name="bad-test",
                status=EvalStatus.FAILED,
                scores={"faithfulness": 0.3},
            )
        ]
        create_annotations(results, threshold=0.7)
        out = capsys.readouterr().out  # type: ignore[union-attr]
        assert "::error::" in out
        assert "bad-test" in out

    def test_includes_failing_scores_in_message(self, capsys: object) -> None:
        """Annotation message includes the failing metric scores."""
        results = [
            _make_result(
                status=EvalStatus.FAILED,
                scores={"faithfulness": 0.3, "relevance": 0.4},
            )
        ]
        create_annotations(results, threshold=0.7)
        out = capsys.readouterr().out  # type: ignore[union-attr]
        assert "faithfulness=0.30" in out
        assert "relevance=0.40" in out

    def test_skips_passing_in_mixed_batch(self, capsys: object) -> None:
        """Only emits annotations for failing results in a mixed batch."""
        results = [
            _make_result(id="t1", status=EvalStatus.PASSED, scores={"faithfulness": 0.9}),
            _make_result(
                id="t2",
                name="failed-one",
                status=EvalStatus.FAILED,
                scores={"faithfulness": 0.2},
            ),
        ]
        create_annotations(results, threshold=0.7)
        out = capsys.readouterr().out  # type: ignore[union-attr]
        lines = [ln for ln in out.strip().splitlines() if "::error::" in ln]
        assert len(lines) == 1
        assert "failed-one" in lines[0]


# ---------------------------------------------------------------------------
# format_summary_markdown
# ---------------------------------------------------------------------------


class TestFormatSummaryMarkdown:
    """Test format_summary_markdown() output."""

    def test_includes_header(self) -> None:
        """Summary includes the RagaliQ header."""
        md = format_summary_markdown([_make_result()])
        assert "## RagaliQ Evaluation Results" in md

    def test_includes_pass_count(self) -> None:
        """Summary includes pass/total count."""
        results = [
            _make_result(id="t1", status=EvalStatus.PASSED),
            _make_result(id="t2", status=EvalStatus.FAILED, scores={"faithfulness": 0.3}),
        ]
        md = format_summary_markdown(results)
        assert "1/2" in md

    def test_includes_threshold(self) -> None:
        """Summary includes the configured threshold."""
        md = format_summary_markdown([_make_result()], threshold=0.85)
        assert "0.85" in md

    def test_includes_table_header(self) -> None:
        """Summary includes a Markdown table with evaluator columns."""
        md = format_summary_markdown([_make_result(scores={"faithfulness": 0.9})])
        assert "| Test Case |" in md
        assert "faithfulness" in md

    def test_includes_test_case_name_in_table(self) -> None:
        """Table rows include the test case name."""
        md = format_summary_markdown([_make_result(name="My test")])
        assert "My test" in md

    def test_empty_results(self) -> None:
        """Empty results produce a header-only summary."""
        md = format_summary_markdown([])
        assert "0/0" in md
        assert "| Test Case |" not in md


# ---------------------------------------------------------------------------
# emit_ci_summary (integration of sub-helpers)
# ---------------------------------------------------------------------------


class TestEmitCISummary:
    """Test emit_ci_summary() high-level helper."""

    def test_writes_summary_and_outputs(self) -> None:
        """Writes step summary + GITHUB_OUTPUT values."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as sf:
            summary_path = sf.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as of:
            output_path = of.name
        try:
            env = {
                "GITHUB_STEP_SUMMARY": summary_path,
                "GITHUB_OUTPUT": output_path,
            }
            with patch.dict(os.environ, env):
                results = [
                    _make_result(id="t1", status=EvalStatus.PASSED, scores={"faithfulness": 0.9}),
                    _make_result(
                        id="t2",
                        status=EvalStatus.FAILED,
                        scores={"faithfulness": 0.3},
                    ),
                ]
                emit_ci_summary(results, threshold=0.7)

            summary = Path(summary_path).read_text(encoding="utf-8")
            assert "RagaliQ" in summary
            assert "1/2" in summary

            outputs = Path(output_path).read_text(encoding="utf-8")
            assert "total=2" in outputs
            assert "passed=1" in outputs
            assert "failed=1" in outputs
        finally:
            os.unlink(summary_path)
            os.unlink(output_path)

    def test_noop_outside_github_actions(self) -> None:
        """Does not raise when GITHUB_STEP_SUMMARY and GITHUB_OUTPUT are unset."""
        with patch.dict(os.environ, {}, clear=True):
            emit_ci_summary([_make_result()])  # should not raise
