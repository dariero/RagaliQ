"""Unit tests for TestCaseGenerator."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragaliq.datasets.generator import TestCaseGenerator, _derive_name
from ragaliq.judges.base import GeneratedAnswerResult, GeneratedQuestionsResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_judge(questions: list[str], answers: list[str]) -> MagicMock:
    """Build a mock judge that returns fixed questions then fixed answers."""
    judge = MagicMock()
    judge.generate_questions = AsyncMock(
        return_value=GeneratedQuestionsResult(questions=questions, tokens_used=100)
    )
    judge.generate_answer = AsyncMock(
        side_effect=[GeneratedAnswerResult(answer=a, tokens_used=50) for a in answers]
    )
    return judge


# ---------------------------------------------------------------------------
# _derive_name
# ---------------------------------------------------------------------------


class TestDeriveName:
    """Tests for the _derive_name helper."""

    def test_strips_trailing_question_mark(self) -> None:
        name = _derive_name("What is Python?", 1)
        assert not name.endswith("?")

    def test_includes_index_prefix(self) -> None:
        name = _derive_name("What is Python?", 3)
        assert name.startswith("3.")

    def test_short_question_kept_in_full(self) -> None:
        name = _derive_name("What is Python?", 1)
        assert "What is Python" in name

    def test_long_question_truncated(self) -> None:
        long_q = "What is the exact mechanism by which Python resolves attribute lookup in the MRO chain for multiple inheritance hierarchies?"
        name = _derive_name(long_q, 2)
        # Name should be well under a generous upper bound
        assert len(name) < 80

    def test_long_question_ends_with_ellipsis(self) -> None:
        long_q = "A" * 70
        name = _derive_name(long_q, 1)
        assert name.endswith("...")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestTestCaseGeneratorValidation:
    """Tests for generate_from_documents input validation."""

    @pytest.mark.asyncio
    async def test_empty_documents_raises_value_error(self) -> None:
        generator = TestCaseGenerator()
        mock_judge = MagicMock()

        with pytest.raises(ValueError, match="documents must not be empty"):
            await generator.generate_from_documents(documents=[], n=3, judge=mock_judge)

    @pytest.mark.asyncio
    async def test_zero_n_raises_value_error(self) -> None:
        generator = TestCaseGenerator()
        mock_judge = MagicMock()

        with pytest.raises(ValueError, match="n must be at least 1"):
            await generator.generate_from_documents(documents=["doc"], n=0, judge=mock_judge)

    @pytest.mark.asyncio
    async def test_negative_n_raises_value_error(self) -> None:
        generator = TestCaseGenerator()
        mock_judge = MagicMock()

        with pytest.raises(ValueError, match="n must be at least 1"):
            await generator.generate_from_documents(documents=["doc"], n=-5, judge=mock_judge)


# ---------------------------------------------------------------------------
# Core generation behaviour
# ---------------------------------------------------------------------------


class TestTestCaseGeneratorGeneration:
    """Tests for the core generate_from_documents logic."""

    @pytest.mark.asyncio
    async def test_returns_correct_count(self) -> None:
        judge = _make_judge(["Q1?", "Q2?", "Q3?"], ["A1", "A2", "A3"])
        result = await TestCaseGenerator().generate_from_documents(
            documents=["doc"], n=3, judge=judge
        )
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_returns_ragtest_case_instances(self) -> None:
        from ragaliq.core.test_case import RAGTestCase

        judge = _make_judge(["Q?"], ["A."])
        result = await TestCaseGenerator().generate_from_documents(
            documents=["doc"], n=1, judge=judge
        )
        assert all(isinstance(tc, RAGTestCase) for tc in result)

    @pytest.mark.asyncio
    async def test_query_matches_generated_question(self) -> None:
        judge = _make_judge(["What is Python?"], ["Python is a language."])
        result = await TestCaseGenerator().generate_from_documents(
            documents=["Python is a language."], n=1, judge=judge
        )
        assert result[0].query == "What is Python?"

    @pytest.mark.asyncio
    async def test_response_matches_generated_answer(self) -> None:
        judge = _make_judge(["What is Python?"], ["Python is a language."])
        result = await TestCaseGenerator().generate_from_documents(
            documents=["Python is a language."], n=1, judge=judge
        )
        assert result[0].response == "Python is a language."

    @pytest.mark.asyncio
    async def test_context_contains_all_documents(self) -> None:
        docs = ["Doc 1 content", "Doc 2 content"]
        judge = _make_judge(["Q?"], ["A."])
        result = await TestCaseGenerator().generate_from_documents(documents=docs, n=1, judge=judge)
        assert result[0].context == docs

    @pytest.mark.asyncio
    async def test_tagged_as_generated(self) -> None:
        judge = _make_judge(["Q?"], ["A."])
        result = await TestCaseGenerator().generate_from_documents(
            documents=["doc"], n=1, judge=judge
        )
        assert "generated" in result[0].tags

    @pytest.mark.asyncio
    async def test_all_ids_are_unique(self) -> None:
        judge = _make_judge(["Q1?", "Q2?", "Q3?"], ["A1", "A2", "A3"])
        result = await TestCaseGenerator().generate_from_documents(
            documents=["doc"], n=3, judge=judge
        )
        ids = [tc.id for tc in result]
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_judge_generate_questions_called_with_correct_args(self) -> None:
        judge = _make_judge(["Q1?", "Q2?"], ["A1", "A2"])
        await TestCaseGenerator().generate_from_documents(documents=["doc"], n=2, judge=judge)
        judge.generate_questions.assert_called_once_with(["doc"], 2)

    @pytest.mark.asyncio
    async def test_generate_answer_called_once_per_question(self) -> None:
        judge = _make_judge(["Q1?", "Q2?", "Q3?"], ["A1", "A2", "A3"])
        await TestCaseGenerator().generate_from_documents(documents=["doc"], n=3, judge=judge)
        assert judge.generate_answer.call_count == 3

    @pytest.mark.asyncio
    async def test_trims_excess_questions_from_llm(self) -> None:
        """If LLM returns more questions than n, only the first n are used."""
        # 5 questions returned, but only 3 answers needed
        judge = _make_judge(
            ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"],
            ["A1", "A2", "A3"],
        )
        result = await TestCaseGenerator().generate_from_documents(
            documents=["doc"], n=3, judge=judge
        )
        assert len(result) == 3
        assert judge.generate_answer.call_count == 3


# ---------------------------------------------------------------------------
# Synchronous wrapper
# ---------------------------------------------------------------------------


class TestTestCaseGeneratorSync:
    """Tests for the synchronous generate_from_documents_sync wrapper."""

    def test_sync_returns_same_as_async(self) -> None:
        judge = _make_judge(["What is X?"], ["X is Y."])
        result = TestCaseGenerator().generate_from_documents_sync(
            documents=["X is Y."], n=1, judge=judge
        )
        assert len(result) == 1
        assert result[0].query == "What is X?"
        assert result[0].response == "X is Y."

    def test_sync_validation_propagates(self) -> None:
        judge = MagicMock()
        with pytest.raises(ValueError, match="documents must not be empty"):
            TestCaseGenerator().generate_from_documents_sync(documents=[], n=1, judge=judge)


# ---------------------------------------------------------------------------
# CLI generate command
# ---------------------------------------------------------------------------


class TestCLIGenerateCommand:
    """Tests for the ragaliq generate CLI command."""

    def _real_test_case(self):  # type: ignore[return]
        """Return a real RAGTestCase for CLI serialization tests."""
        from ragaliq.core.test_case import RAGTestCase

        return RAGTestCase(
            id="tc-1",
            name="1. What is X",
            query="What is X?",
            context=["X is Y."],
            response="X is Y.",
            tags=["generated"],
        )

    def test_generate_exits_zero_on_success(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from ragaliq.cli.main import app

        # Create a real .txt file so _load_documents succeeds
        doc_file = tmp_path / "doc.txt"
        doc_file.write_text("Python is a programming language.")

        output_file = tmp_path / "out.json"
        real_tc = self._real_test_case()

        with (
            patch("ragaliq.judges.ClaudeJudge") as mock_judge_cls,
            patch("ragaliq.datasets.generator.TestCaseGenerator") as mock_gen_cls,
        ):
            mock_judge_cls.return_value = MagicMock()
            mock_gen_cls.return_value.generate_from_documents = AsyncMock(return_value=[real_tc])
            runner = CliRunner()
            result = runner.invoke(
                app,
                ["generate", str(doc_file), "-n", "1", "-o", str(output_file)],
            )

        assert result.exit_code == 0

    def test_generate_exits_one_for_missing_docs(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from ragaliq.cli.main import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["generate", str(tmp_path / "nonexistent.txt"), "-n", "1"],
        )

        assert result.exit_code == 1

    def test_generate_exits_one_for_empty_txt_file(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from ragaliq.cli.main import app

        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["generate", str(empty_file), "-n", "1"],
        )

        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# _load_documents helper
# ---------------------------------------------------------------------------


class TestLoadDocuments:
    """Tests for the _load_documents CLI helper."""

    def test_txt_file_returns_single_document(self, tmp_path: Path) -> None:
        from ragaliq.cli.main import _load_documents

        doc = tmp_path / "doc.txt"
        doc.write_text("Hello world")
        result = _load_documents(doc)
        assert result == ["Hello world"]

    def test_empty_txt_file_returns_empty_list(self, tmp_path: Path) -> None:
        from ragaliq.cli.main import _load_documents

        doc = tmp_path / "empty.txt"
        doc.write_text("   ")
        result = _load_documents(doc)
        assert result == []

    def test_directory_loads_all_txt_files(self, tmp_path: Path) -> None:
        from ragaliq.cli.main import _load_documents

        (tmp_path / "a.txt").write_text("Doc A")
        (tmp_path / "b.txt").write_text("Doc B")
        result = _load_documents(tmp_path)
        assert sorted(result) == ["Doc A", "Doc B"]

    def test_directory_ignores_non_txt_files(self, tmp_path: Path) -> None:
        from ragaliq.cli.main import _load_documents

        (tmp_path / "a.txt").write_text("Doc A")
        (tmp_path / "notes.md").write_text("Markdown note")
        result = _load_documents(tmp_path)
        assert result == ["Doc A"]

    def test_json_list_of_strings(self, tmp_path: Path) -> None:
        import json

        from ragaliq.cli.main import _load_documents

        doc = tmp_path / "docs.json"
        doc.write_text(json.dumps(["Doc One", "Doc Two"]))
        result = _load_documents(doc)
        assert result == ["Doc One", "Doc Two"]

    def test_yaml_list_of_strings(self, tmp_path: Path) -> None:
        from ragaliq.cli.main import _load_documents

        doc = tmp_path / "docs.yaml"
        doc.write_text("- Doc One\n- Doc Two\n")
        result = _load_documents(doc)
        assert result == ["Doc One", "Doc Two"]

    def test_nonexistent_path_returns_empty_list(self, tmp_path: Path) -> None:
        from ragaliq.cli.main import _load_documents

        result = _load_documents(tmp_path / "does_not_exist.txt")
        assert result == []

    def test_unsupported_extension_returns_empty_list(self, tmp_path: Path) -> None:
        from ragaliq.cli.main import _load_documents

        doc = tmp_path / "docs.csv"
        doc.write_text("a,b,c")
        result = _load_documents(doc)
        assert result == []
