"""Unit tests for RagaliQ runner instantiation, configuration, and wiring."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragaliq.core.runner import RagaliQ
from ragaliq.judges.base import JudgeConfig, LLMJudge
from ragaliq.judges.claude import ClaudeJudge


class TestRagaliQInstantiation:
    """Test that RagaliQ can be instantiated correctly."""

    def test_instantiate_default(self):
        """RagaliQ() instantiates without error using defaults."""
        runner = RagaliQ()

        assert runner.judge_type == "claude"
        assert runner._judge is None  # Lazy initialization
        assert runner.evaluator_names == ["faithfulness", "relevance"]
        assert runner.default_threshold == 0.7

    def test_instantiate_with_judge_type(self):
        """RagaliQ can be instantiated with specific judge type."""
        runner = RagaliQ(judge="claude")

        assert runner.judge_type == "claude"
        assert runner._judge is None

    def test_instantiate_with_config(self):
        """RagaliQ can be instantiated with custom judge config."""
        config = JudgeConfig(model="claude-sonnet-4-20250514", temperature=0.1)
        runner = RagaliQ(judge_config=config)

        assert runner._judge_config == config
        assert runner._judge is None

    def test_instantiate_with_api_key(self):
        """RagaliQ can be instantiated with explicit API key."""
        runner = RagaliQ(api_key="test-api-key")

        assert runner._api_key == "test-api-key"
        assert runner._judge is None


class TestCustomJudgeInjection:
    """Test that pre-configured judges can be injected."""

    def test_inject_custom_judge(self):
        """Pre-configured LLMJudge instance can be injected."""
        # Create a mock judge that inherits from LLMJudge
        mock_judge = MagicMock(spec=LLMJudge)

        runner = RagaliQ(judge=mock_judge)

        assert runner._judge is mock_judge
        assert runner.judge_type is None  # No string type when instance provided

    def test_inject_claude_judge_instance(self):
        """Pre-configured ClaudeJudge can be injected."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            judge = ClaudeJudge()
            runner = RagaliQ(judge=judge)

            assert runner._judge is judge
            assert runner.judge_type is None


class TestLazyInitialization:
    """Test that judge is lazily initialized."""

    def test_judge_not_initialized_on_construction(self):
        """Judge should not be initialized when RagaliQ is created."""
        runner = RagaliQ()

        assert runner._judge is None

    def test_judge_initialized_on_first_init_call(self):
        """Judge is initialized when _init_judge() is called."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            runner = RagaliQ()
            assert runner._judge is None

            runner._init_judge()

            assert runner._judge is not None
            assert isinstance(runner._judge, ClaudeJudge)

    def test_judge_init_is_idempotent(self):
        """Multiple _init_judge() calls don't recreate judge."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            runner = RagaliQ()
            runner._init_judge()
            first_judge = runner._judge

            runner._init_judge()

            assert runner._judge is first_judge  # Same instance

    def test_lazy_init_with_custom_config(self):
        """Judge is initialized with provided config."""
        config = JudgeConfig(model="claude-sonnet-4-20250514", temperature=0.5)

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            runner = RagaliQ(judge_config=config)
            runner._init_judge()

            assert runner._judge is not None
            assert runner._judge.config.temperature == 0.5

    def test_lazy_init_with_api_key(self):
        """Judge is initialized with provided API key."""
        # Remove any existing env var to ensure we're using the explicit key
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)

        with patch.dict(os.environ, env, clear=True):
            runner = RagaliQ(api_key="explicit-api-key")
            runner._init_judge()

            assert runner._judge is not None


class TestJudgeTypeHandling:
    """Test handling of different judge types."""

    def test_openai_judge_not_implemented(self):
        """OpenAI judge raises NotImplementedError."""
        runner = RagaliQ(judge="openai")

        with pytest.raises(NotImplementedError, match="OpenAI judge not yet implemented"):
            runner._init_judge()

    def test_invalid_judge_type_raises(self):
        """Invalid judge type string raises ValueError."""
        runner = RagaliQ()
        runner.judge_type = "invalid"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="Unknown judge type"):
            runner._init_judge()


class TestJudgePassedToEvaluators:
    """Test that judge is passed correctly to evaluators."""

    @pytest.mark.asyncio
    async def test_evaluate_async_passes_judge(self, sample_test_case):
        """Judge is passed to evaluators during evaluation."""
        # Create a mock judge
        mock_judge = MagicMock(spec=LLMJudge)

        # Create a mock evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.name = "test_evaluator"
        mock_evaluator.evaluate = AsyncMock(
            return_value=MagicMock(
                score=0.9,
                reasoning="Test reasoning",
                passed=True,
                raw_response={},
                tokens_used=100,
            )
        )

        runner = RagaliQ(judge=mock_judge)
        runner._evaluators = [mock_evaluator]

        await runner.evaluate_async(sample_test_case)

        # Verify evaluator.evaluate was called with test_case and judge
        mock_evaluator.evaluate.assert_called_once_with(sample_test_case, mock_judge)


class TestEvaluatorInitialization:
    """Test that evaluators are initialized correctly from registry."""

    def test_init_evaluators_with_default_names(self):
        """Default evaluator names are resolved correctly."""
        runner = RagaliQ()
        assert runner._evaluators == []

        runner._init_evaluators()

        assert len(runner._evaluators) == 2
        assert runner._evaluators[0].name == "faithfulness"
        assert runner._evaluators[1].name == "relevance"

    def test_init_evaluators_with_custom_names(self):
        """Custom evaluator names are resolved correctly."""
        runner = RagaliQ(evaluators=["hallucination", "faithfulness"])

        runner._init_evaluators()

        assert len(runner._evaluators) == 2
        assert runner._evaluators[0].name == "hallucination"
        assert runner._evaluators[1].name == "faithfulness"

    def test_init_evaluators_unknown_raises(self):
        """Unknown evaluator name raises ValueError with available options."""
        runner = RagaliQ(evaluators=["nonexistent"])

        with pytest.raises(ValueError, match="Unknown evaluator: 'nonexistent'"):
            runner._init_evaluators()
        with pytest.raises(ValueError, match="Available evaluators:"):
            runner._init_evaluators()

    def test_init_evaluators_applies_threshold(self):
        """Default threshold is applied to all evaluators."""
        runner = RagaliQ(default_threshold=0.9)

        runner._init_evaluators()

        for evaluator in runner._evaluators:
            assert evaluator.threshold == 0.9

    def test_init_evaluators_is_idempotent(self):
        """Multiple _init_evaluators() calls don't recreate evaluators."""
        runner = RagaliQ()

        runner._init_evaluators()
        first_evaluators = runner._evaluators

        runner._init_evaluators()

        assert runner._evaluators is first_evaluators

    def test_init_evaluators_all_three_loadable(self):
        """All three evaluators can be loaded simultaneously."""
        runner = RagaliQ(evaluators=["faithfulness", "relevance", "hallucination"])

        runner._init_evaluators()

        assert len(runner._evaluators) == 3
        assert runner._evaluators[0].name == "faithfulness"
        assert runner._evaluators[1].name == "relevance"
        assert runner._evaluators[2].name == "hallucination"


class TestBoundedConcurrency:
    """Test that batch evaluation respects concurrency limits."""

    @pytest.mark.asyncio
    async def test_default_concurrency_limit(self):
        """Default max_concurrency is 5."""
        runner = RagaliQ()
        assert runner.max_concurrency == 5

    @pytest.mark.asyncio
    async def test_custom_concurrency_limit(self):
        """Custom max_concurrency can be set via __init__."""
        runner = RagaliQ(max_concurrency=10)
        assert runner.max_concurrency == 10

    @pytest.mark.asyncio
    async def test_batch_concurrency_override(self, sample_test_case):
        """Batch evaluation accepts concurrency override."""
        mock_judge = MagicMock(spec=LLMJudge)
        mock_evaluator = MagicMock()
        mock_evaluator.name = "test"
        mock_evaluator.evaluate = AsyncMock(
            return_value=MagicMock(
                score=0.9, reasoning="", passed=True, raw_response={}, tokens_used=50
            )
        )

        runner = RagaliQ(judge=mock_judge, max_concurrency=5)
        runner._evaluators = [mock_evaluator]

        # Should not raise, just verify it accepts the parameter
        await runner.evaluate_batch_async([sample_test_case], max_concurrency=2)
