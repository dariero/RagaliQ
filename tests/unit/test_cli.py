"""Unit tests for RagaliQ CLI entry point."""

from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

import ragaliq
from ragaliq.cli.main import app

runner = CliRunner()


class TestCLIVersion:
    """Test CLI version flag."""

    def test_version_flag_long(self):
        """--version flag shows version and exits."""
        result = runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert f"RagaliQ version {ragaliq.__version__}" in result.output

    def test_version_flag_short(self):
        """-V flag shows version and exits."""
        result = runner.invoke(app, ["-V"])

        assert result.exit_code == 0
        assert f"RagaliQ version {ragaliq.__version__}" in result.output


class TestCLINoArgs:
    """Test CLI with no arguments."""

    def test_no_args_shows_help(self):
        """Running with no args shows help message."""
        result = runner.invoke(app, [])

        assert "Usage:" in result.output or "Commands:" in result.output


class TestVersionCommand:
    """Test version subcommand."""

    def test_version_command_shows_version(self):
        """version command outputs the current version."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert ragaliq.__version__ in result.output


class TestListEvaluatorsCommand:
    """Test list-evaluators command."""

    def test_exits_zero(self):
        """list-evaluators exits successfully."""
        result = runner.invoke(app, ["list-evaluators"])

        assert result.exit_code == 0

    def test_shows_built_in_evaluator_faithfulness(self):
        """list-evaluators includes faithfulness in output."""
        result = runner.invoke(app, ["list-evaluators"])

        assert "faithfulness" in result.output

    def test_shows_built_in_evaluator_relevance(self):
        """list-evaluators includes relevance in output."""
        result = runner.invoke(app, ["list-evaluators"])

        assert "relevance" in result.output

    def test_shows_description_heading(self):
        """list-evaluators shows a Description column heading."""
        result = runner.invoke(app, ["list-evaluators"])

        assert "Description" in result.output


class TestValidateCommand:
    """Test validate command."""

    def _mock_dataset(self, n: int = 2) -> MagicMock:
        """Return a minimal mock DatasetSchema with n test cases."""
        dataset = MagicMock()
        dataset.test_cases = [MagicMock() for _ in range(n)]
        return dataset

    def test_valid_dataset_exits_zero(self):
        """validate exits 0 for a loadable dataset."""
        with patch("ragaliq.datasets.DatasetLoader.load", return_value=self._mock_dataset(2)):
            result = runner.invoke(app, ["validate", "dataset.json"])

        assert result.exit_code == 0

    def test_valid_dataset_shows_count(self):
        """validate reports the number of test cases found."""
        with patch("ragaliq.datasets.DatasetLoader.load", return_value=self._mock_dataset(3)):
            result = runner.invoke(app, ["validate", "dataset.json"])

        assert "3 test cases" in result.output

    def test_singular_when_one_test_case(self):
        """validate uses singular 'test case' when there is exactly one."""
        with patch("ragaliq.datasets.DatasetLoader.load", return_value=self._mock_dataset(1)):
            result = runner.invoke(app, ["validate", "dataset.json"])

        assert "1 test case" in result.output
        assert "1 test cases" not in result.output

    def test_invalid_dataset_exits_one(self):
        """validate exits 1 when the dataset fails to load."""
        from ragaliq.datasets import DatasetLoadError

        with patch(
            "ragaliq.datasets.DatasetLoader.load",
            side_effect=DatasetLoadError("unsupported format"),
        ):
            result = runner.invoke(app, ["validate", "bad.json"])

        assert result.exit_code == 1

    def test_invalid_dataset_shows_error_message(self):
        """validate shows an error message when loading fails."""
        from ragaliq.datasets import DatasetLoadError

        with patch(
            "ragaliq.datasets.DatasetLoader.load",
            side_effect=DatasetLoadError("unsupported format"),
        ):
            result = runner.invoke(app, ["validate", "bad.json"])

        assert "Validation failed" in result.output

    def test_missing_path_argument_exits_nonzero(self):
        """validate requires a dataset path argument."""
        result = runner.invoke(app, ["validate"])

        assert result.exit_code != 0


class TestRunCommand:
    """Test run command."""

    def _mock_passing_result(self, name: str = "test-1") -> MagicMock:
        """Build a mock RAGTestResult that passes."""
        from ragaliq.core.test_case import EvalStatus

        tc = MagicMock()
        tc.name = name

        result = MagicMock()
        result.passed = True
        result.status = EvalStatus.PASSED
        result.scores = {"faithfulness": 0.9, "relevance": 0.85}
        result.details = {}
        result.judge_tokens_used = 0
        result.test_case = tc
        return result

    def _mock_failing_result(self, name: str = "test-2") -> MagicMock:
        """Build a mock RAGTestResult that fails."""
        from ragaliq.core.test_case import EvalStatus

        tc = MagicMock()
        tc.name = name

        result = MagicMock()
        result.passed = False
        result.status = EvalStatus.FAILED
        result.scores = {"faithfulness": 0.3, "relevance": 0.4}
        result.details = {}
        result.judge_tokens_used = 0
        result.test_case = tc
        return result

    def test_all_passing_exits_zero(self):
        """run exits 0 when every test case passes."""
        mock_dataset = MagicMock()
        mock_dataset.test_cases = [MagicMock()]

        with (
            patch("ragaliq.datasets.DatasetLoader.load", return_value=mock_dataset),
            patch("ragaliq.RagaliQ") as mock_cls,
        ):
            mock_cls.return_value.evaluate_batch_async = AsyncMock(
                return_value=[self._mock_passing_result()]
            )
            result = runner.invoke(app, ["run", "dataset.json"])

        assert result.exit_code == 0

    def test_any_failure_exits_one(self):
        """run exits 1 when at least one test case fails."""
        mock_dataset = MagicMock()
        mock_dataset.test_cases = [MagicMock()]

        with (
            patch("ragaliq.datasets.DatasetLoader.load", return_value=mock_dataset),
            patch("ragaliq.RagaliQ") as mock_cls,
        ):
            mock_cls.return_value.evaluate_batch_async = AsyncMock(
                return_value=[self._mock_failing_result()]
            )
            result = runner.invoke(app, ["run", "dataset.json"])

        assert result.exit_code == 1

    def test_dataset_load_error_exits_one(self):
        """run exits 1 when the dataset fails to load."""
        from ragaliq.datasets import DatasetLoadError

        with patch(
            "ragaliq.datasets.DatasetLoader.load",
            side_effect=DatasetLoadError("file not found"),
        ):
            result = runner.invoke(app, ["run", "missing.json"])

        assert result.exit_code == 1

    def test_dataset_load_error_shows_message(self):
        """run shows an error message when dataset loading fails."""
        from ragaliq.datasets import DatasetLoadError

        with patch(
            "ragaliq.datasets.DatasetLoader.load",
            side_effect=DatasetLoadError("file not found"),
        ):
            result = runner.invoke(app, ["run", "missing.json"])

        assert "Error" in result.output

    def test_evaluator_option_forwarded_to_runner(self):
        """--evaluator option is forwarded to RagaliQ constructor."""
        mock_result = self._mock_passing_result()
        mock_result.scores = {"relevance": 0.9}
        mock_dataset = MagicMock()
        mock_dataset.test_cases = [MagicMock()]

        with (
            patch("ragaliq.datasets.DatasetLoader.load", return_value=mock_dataset),
            patch("ragaliq.RagaliQ") as mock_cls,
        ):
            mock_cls.return_value.evaluate_batch_async = AsyncMock(return_value=[mock_result])
            runner.invoke(app, ["run", "dataset.json", "--evaluator", "relevance"])

        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["evaluators"] == ["relevance"]

    def test_threshold_option_forwarded_to_runner(self):
        """--threshold option is forwarded to RagaliQ constructor."""
        mock_dataset = MagicMock()
        mock_dataset.test_cases = [MagicMock()]

        with (
            patch("ragaliq.datasets.DatasetLoader.load", return_value=mock_dataset),
            patch("ragaliq.RagaliQ") as mock_cls,
        ):
            mock_cls.return_value.evaluate_batch_async = AsyncMock(
                return_value=[self._mock_passing_result()]
            )
            runner.invoke(app, ["run", "dataset.json", "--threshold", "0.9"])

        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["default_threshold"] == 0.9

    def test_summary_shows_pass_count(self):
        """run outputs a summary line with the pass/total count."""
        mock_dataset = MagicMock()
        mock_dataset.test_cases = [MagicMock()]

        with (
            patch("ragaliq.datasets.DatasetLoader.load", return_value=mock_dataset),
            patch("ragaliq.RagaliQ") as mock_cls,
        ):
            mock_cls.return_value.evaluate_batch_async = AsyncMock(
                return_value=[self._mock_passing_result()]
            )
            result = runner.invoke(app, ["run", "dataset.json"])

        assert "1/1 passed" in result.output

    def test_missing_path_argument_exits_nonzero(self):
        """run requires a dataset path argument."""
        result = runner.invoke(app, ["run"])

        assert result.exit_code != 0

    def _real_passing_result(self, name: str = "test-1") -> object:
        """Build a real RAGTestResult for reporters that need serializable data."""
        from ragaliq.core.test_case import EvalStatus, RAGTestCase, RAGTestResult

        tc = RAGTestCase(id="t1", name=name, query="Q?", context=["C."], response="A.")
        return RAGTestResult(
            test_case=tc,
            status=EvalStatus.PASSED,
            scores={"faithfulness": 0.9},
            details={},
            judge_tokens_used=0,
        )

    def test_unknown_output_format_exits_one(self):
        """--output with an unknown format exits 1 before loading dataset."""
        result = runner.invoke(app, ["run", "dataset.json", "--output", "xml"])

        assert result.exit_code == 1
        assert "unknown output format" in result.output.lower() or "xml" in result.output

    def test_output_json_writes_file(self):
        """--output json writes a report.json file."""
        import json
        import tempfile
        from pathlib import Path

        mock_dataset = MagicMock()
        mock_dataset.test_cases = [MagicMock()]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "out.json"
            with (
                patch("ragaliq.datasets.DatasetLoader.load", return_value=mock_dataset),
                patch("ragaliq.RagaliQ") as mock_cls,
            ):
                mock_cls.return_value.evaluate_batch_async = AsyncMock(
                    return_value=[self._real_passing_result()]
                )
                result = runner.invoke(
                    app,
                    ["run", "dataset.json", "--output", "json", "--output-file", str(output_path)],
                )

            assert result.exit_code == 0
            assert output_path.exists()
            doc = json.loads(output_path.read_text())
            assert "results" in doc
            assert "summary" in doc

    def test_output_html_writes_file(self):
        """--output html writes an HTML report file."""
        import tempfile
        from pathlib import Path

        mock_dataset = MagicMock()
        mock_dataset.test_cases = [MagicMock()]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "out.html"
            with (
                patch("ragaliq.datasets.DatasetLoader.load", return_value=mock_dataset),
                patch("ragaliq.RagaliQ") as mock_cls,
            ):
                mock_cls.return_value.evaluate_batch_async = AsyncMock(
                    return_value=[self._real_passing_result()]
                )
                result = runner.invoke(
                    app,
                    ["run", "dataset.json", "--output", "html", "--output-file", str(output_path)],
                )

            assert result.exit_code == 0
            assert output_path.exists()
            content = output_path.read_text()
            assert "<!DOCTYPE html>" in content

    def test_output_json_shows_file_path_in_output(self):
        """--output json echoes the file path written."""
        import tempfile
        from pathlib import Path

        mock_dataset = MagicMock()
        mock_dataset.test_cases = [MagicMock()]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            with (
                patch("ragaliq.datasets.DatasetLoader.load", return_value=mock_dataset),
                patch("ragaliq.RagaliQ") as mock_cls,
            ):
                mock_cls.return_value.evaluate_batch_async = AsyncMock(
                    return_value=[self._real_passing_result()]
                )
                result = runner.invoke(
                    app,
                    ["run", "dataset.json", "--output", "json", "--output-file", str(output_path)],
                )

            assert "report.json" in result.output or str(output_path) in result.output

    def test_output_console_is_default(self):
        """Without --output flag, console output is used (table shown)."""
        mock_dataset = MagicMock()
        mock_dataset.test_cases = [MagicMock()]

        with (
            patch("ragaliq.datasets.DatasetLoader.load", return_value=mock_dataset),
            patch("ragaliq.RagaliQ") as mock_cls,
        ):
            mock_cls.return_value.evaluate_batch_async = AsyncMock(
                return_value=[self._mock_passing_result()]
            )
            result = runner.invoke(app, ["run", "dataset.json"])

        # Console reporter renders a summary line
        assert "passed" in result.output
