"""Unit tests for RagaliQ CLI entry point."""

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
        assert f"RagaliQ version {ragaliq.__version__}" in result.stdout

    def test_version_flag_short(self):
        """-V flag shows version and exits."""
        result = runner.invoke(app, ["-V"])

        assert result.exit_code == 0
        assert f"RagaliQ version {ragaliq.__version__}" in result.stdout


class TestCLINoArgs:
    """Test CLI with no arguments."""

    def test_no_args_shows_help(self):
        """Running with no args shows help message."""
        result = runner.invoke(app, [])

        # Typer shows help when no_args_is_help=True
        assert "Usage:" in result.stdout or "Commands:" in result.stdout


class TestEvaluateCommand:
    """Test evaluate command placeholder."""

    def test_evaluate_command_exits_with_error(self):
        """evaluate command prints placeholder and exits with code 1."""
        result = runner.invoke(app, ["evaluate"])

        assert result.exit_code == 1
        assert "not yet implemented" in result.stdout

    def test_evaluate_command_shows_library_usage(self):
        """evaluate command suggests using RagaliQ as a library."""
        result = runner.invoke(app, ["evaluate"])

        assert "library" in result.stdout.lower()
