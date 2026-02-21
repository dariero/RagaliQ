"""Unit tests for RagaliQ runner instantiation, configuration, and wiring."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragaliq.core.runner import RagaliQ
from ragaliq.core.test_case import EvalStatus
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
        config = JudgeConfig(model="claude-sonnet-4-6", temperature=0.1)
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
        config = JudgeConfig(model="claude-sonnet-4-6", temperature=0.5)

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

    @pytest.mark.asyncio
    async def test_max_judge_concurrency_limits_parallel_calls(self):
        """Test that max_judge_concurrency actually limits concurrent judge API calls."""
        import asyncio

        from ragaliq.judges.base import JudgeConfig
        from ragaliq.judges.base_judge import BaseJudge
        from ragaliq.judges.transport import TransportResponse

        # Track concurrent call count
        concurrent_calls = 0
        max_concurrent = 0

        async def mock_send(*_args, **_kwargs):
            nonlocal concurrent_calls, max_concurrent
            concurrent_calls += 1
            max_concurrent = max(max_concurrent, concurrent_calls)

            # Simulate slow API call
            await asyncio.sleep(0.05)

            concurrent_calls -= 1
            return TransportResponse(
                text='{"score": 0.9, "reasoning": "test"}',
                input_tokens=10,
                output_tokens=10,
                model="test-model",
            )

        # Create mock transport
        mock_transport = MagicMock()
        mock_transport.send = mock_send

        # Create judge with concurrency limit of 3
        judge = BaseJudge(transport=mock_transport, config=JudgeConfig(), max_concurrency=3)

        # Create 10 concurrent calls that would normally run in parallel
        tasks = [judge._call_llm("system", f"user {i}", operation=f"op_{i}") for i in range(10)]

        await asyncio.gather(*tasks)

        # Max concurrent should be 3 (the limit), not 10
        assert max_concurrent == 3, f"Expected max 3 concurrent, got {max_concurrent}"


class TestErrorEnvelopes:
    """Test that evaluator failures are gracefully handled with error envelopes."""

    @pytest.mark.asyncio
    async def test_single_evaluator_failure_returns_error_result(self, sample_test_case):
        """Single evaluator throwing exception returns EvaluationResult with error field."""
        from ragaliq.core.test_case import EvalStatus

        mock_judge = MagicMock(spec=LLMJudge)

        # Create evaluator that raises an exception
        failing_evaluator = MagicMock()
        failing_evaluator.name = "failing"
        failing_evaluator.evaluate = AsyncMock(side_effect=ValueError("Test failure"))

        runner = RagaliQ(judge=mock_judge)
        runner._evaluators = [failing_evaluator]

        result = await runner.evaluate_async(sample_test_case)

        # Should return ERROR status, not raise
        assert result.status == EvalStatus.ERROR
        assert "failing" in result.scores
        assert result.scores["failing"] == 0.0
        assert "error" in result.details["failing"]
        assert "ValueError" in result.details["failing"]["error"]

    @pytest.mark.asyncio
    async def test_partial_failure_preserves_successful_scores(self, sample_test_case):
        """First evaluator succeeds, second throws; first score should be preserved."""
        from ragaliq.core.test_case import EvalStatus

        mock_judge = MagicMock(spec=LLMJudge)

        # Success evaluator
        success_evaluator = MagicMock()
        success_evaluator.name = "success"
        success_evaluator.evaluate = AsyncMock(
            return_value=MagicMock(
                score=0.85,
                reasoning="Good",
                passed=True,
                raw_response={},
                tokens_used=100,
                error=None,
            )
        )

        # Failing evaluator
        failing_evaluator = MagicMock()
        failing_evaluator.name = "failing"
        failing_evaluator.evaluate = AsyncMock(side_effect=RuntimeError("Judge API timeout"))

        runner = RagaliQ(judge=mock_judge)
        runner._evaluators = [success_evaluator, failing_evaluator]

        result = await runner.evaluate_async(sample_test_case)

        # Should have ERROR status but preserve successful score
        assert result.status == EvalStatus.ERROR
        assert result.scores["success"] == 0.85
        assert result.details["success"]["passed"] is True
        assert result.scores["failing"] == 0.0
        assert "error" in result.details["failing"]
        assert "RuntimeError" in result.details["failing"]["error"]

    @pytest.mark.asyncio
    async def test_batch_single_failure_doesnt_crash_batch(self, sample_test_case):
        """One test case failing shouldn't crash the entire batch."""
        from ragaliq.core.test_case import EvalStatus, RAGTestCase

        mock_judge = MagicMock(spec=LLMJudge)
        mock_evaluator = MagicMock()
        mock_evaluator.name = "test"

        # First call succeeds, second fails, third succeeds
        mock_evaluator.evaluate = AsyncMock(
            side_effect=[
                MagicMock(
                    score=0.9,
                    reasoning="",
                    passed=True,
                    raw_response={},
                    tokens_used=50,
                    error=None,
                ),
                RuntimeError("Middle test case failed"),
                MagicMock(
                    score=0.8,
                    reasoning="",
                    passed=True,
                    raw_response={},
                    tokens_used=50,
                    error=None,
                ),
            ]
        )

        runner = RagaliQ(judge=mock_judge)
        runner._evaluators = [mock_evaluator]

        test_cases = [
            sample_test_case,
            RAGTestCase(
                id="tc2",
                name="Test 2",
                query="Query 2",
                context=["Context 2"],
                response="Response 2",
            ),
            RAGTestCase(
                id="tc3",
                name="Test 3",
                query="Query 3",
                context=["Context 3"],
                response="Response 3",
            ),
        ]

        results = await runner.evaluate_batch_async(test_cases)

        # All 3 results should be returned
        assert len(results) == 3
        assert results[0].status == EvalStatus.PASSED
        assert results[1].status == EvalStatus.ERROR
        assert results[2].status == EvalStatus.PASSED

    @pytest.mark.asyncio
    async def test_error_status_distinct_from_failed(self, sample_test_case):
        """ERROR status is distinct from FAILED (low score)."""
        from ragaliq.core.test_case import EvalStatus

        mock_judge = MagicMock(spec=LLMJudge)

        # Evaluator returns low score (not an error)
        low_score_evaluator = MagicMock()
        low_score_evaluator.name = "low"
        low_score_evaluator.evaluate = AsyncMock(
            return_value=MagicMock(
                score=0.3,
                reasoning="Low quality",
                passed=False,
                raw_response={},
                tokens_used=50,
                error=None,
            )
        )

        runner = RagaliQ(judge=mock_judge)
        runner._evaluators = [low_score_evaluator]

        result = await runner.evaluate_async(sample_test_case)

        # Should be FAILED, not ERROR
        assert result.status == EvalStatus.FAILED
        assert "error" not in result.details["low"]

    @pytest.mark.asyncio
    async def test_fail_fast_true_propagates_exceptions(self, sample_test_case):
        """When fail_fast=True, evaluator exceptions propagate immediately."""
        mock_judge = MagicMock(spec=LLMJudge)

        failing_evaluator = MagicMock()
        failing_evaluator.name = "failing"
        failing_evaluator.evaluate = AsyncMock(side_effect=ValueError("Fast fail test"))

        runner = RagaliQ(judge=mock_judge, fail_fast=True)
        runner._evaluators = [failing_evaluator]

        # Should raise, not return error envelope
        with pytest.raises(ValueError, match="Fast fail test"):
            await runner.evaluate_async(sample_test_case)

    @pytest.mark.asyncio
    async def test_fail_fast_false_converts_to_error_envelope(self, sample_test_case):
        """When fail_fast=False (default), exceptions convert to error envelopes."""
        from ragaliq.core.test_case import EvalStatus

        mock_judge = MagicMock(spec=LLMJudge)

        failing_evaluator = MagicMock()
        failing_evaluator.name = "failing"
        failing_evaluator.evaluate = AsyncMock(side_effect=ValueError("Envelope test"))

        runner = RagaliQ(judge=mock_judge, fail_fast=False)
        runner._evaluators = [failing_evaluator]

        # Should return error envelope, not raise
        result = await runner.evaluate_async(sample_test_case)
        assert result.status == EvalStatus.ERROR
        assert "error" in result.details["failing"]
        assert "ValueError" in result.details["failing"]["error"]

    @pytest.mark.asyncio
    async def test_fail_fast_applies_to_batch_mode(self, sample_test_case):
        """fail_fast also works in batch mode."""

        mock_judge = MagicMock(spec=LLMJudge)

        failing_evaluator = MagicMock()
        failing_evaluator.name = "failing"
        failing_evaluator.evaluate = AsyncMock(side_effect=RuntimeError("Batch fail fast"))

        runner = RagaliQ(judge=mock_judge, fail_fast=True)
        runner._evaluators = [failing_evaluator]

        test_cases = [sample_test_case, sample_test_case]

        # Should raise on first failure, not complete batch
        with pytest.raises(RuntimeError, match="Batch fail fast"):
            await runner.evaluate_batch_async(test_cases)


class TestAsyncInitSafety:
    """Test that concurrent async initialization is safe from race conditions.

    Uses threading.Lock instead of asyncio.Lock to ensure initialization
    works correctly even when the same runner is used across multiple
    event loops (e.g., repeated sync evaluate() calls).
    """

    @pytest.mark.asyncio
    async def test_concurrent_init_creates_single_judge(self, sample_test_case):
        """Multiple concurrent evaluate_async calls create only one judge instance."""
        import asyncio

        # Track judge instantiation count
        judge_creation_count = 0
        created_judge = None

        def mock_judge_factory(*_args, **_kwargs):
            nonlocal judge_creation_count, created_judge
            judge_creation_count += 1
            if created_judge is None:
                created_judge = MagicMock(spec=LLMJudge)
            return created_judge

        # Create mock evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.name = "test"
        mock_evaluator.evaluate = AsyncMock(
            return_value=MagicMock(
                score=0.9, reasoning="", passed=True, raw_response={}, tokens_used=50
            )
        )

        def mock_get_evaluator(_name):
            def factory(**_kwargs):
                return mock_evaluator

            return factory

        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}),
            patch("ragaliq.judges.claude.ClaudeJudge", side_effect=mock_judge_factory),
            patch("ragaliq.evaluators.get_evaluator", side_effect=mock_get_evaluator),
        ):
            runner = RagaliQ(judge="claude")

            # Launch 10 concurrent evaluations
            tasks = [runner.evaluate_async(sample_test_case) for _ in range(10)]
            await asyncio.gather(*tasks)

            # Judge should be created exactly once despite 10 concurrent calls
            assert judge_creation_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_init_creates_single_evaluator_list(self, sample_test_case):
        """Multiple concurrent evaluate_async calls create evaluators only once."""
        import asyncio

        # Track evaluator instantiation count
        evaluator_creation_count = 0

        def mock_evaluator_factory(*_args, **_kwargs):
            nonlocal evaluator_creation_count
            evaluator_creation_count += 1
            mock_ev = MagicMock()
            mock_ev.name = f"test_{evaluator_creation_count}"
            mock_ev.evaluate = AsyncMock(
                return_value=MagicMock(
                    score=0.9, reasoning="", passed=True, raw_response={}, tokens_used=50
                )
            )
            return mock_ev

        mock_judge = MagicMock(spec=LLMJudge)

        with patch("ragaliq.evaluators.get_evaluator", return_value=mock_evaluator_factory):
            runner = RagaliQ(judge=mock_judge, evaluators=["faithfulness"])

            # Launch 10 concurrent evaluations
            tasks = [runner.evaluate_async(sample_test_case) for _ in range(10)]
            await asyncio.gather(*tasks)

            # Evaluators should be created exactly once (1 evaluator name × 1 init call)
            assert evaluator_creation_count == 1

    def test_repeated_sync_calls_across_event_loops(self, sample_test_case):
        """Repeated sync evaluate() calls work correctly across different event loops.

        This tests the fix for loop-bound asyncio.Lock issues. Using threading.Lock
        instead ensures the same runner instance can be used across multiple
        asyncio.run() calls (each creating a new event loop).
        """
        # Track judge instantiation count
        judge_creation_count = 0

        def mock_judge_factory(*_args, **_kwargs):
            nonlocal judge_creation_count
            judge_creation_count += 1
            return MagicMock(spec=LLMJudge)

        # Create mock evaluator with proper result structure
        from ragaliq.core.evaluator import EvaluationResult

        mock_result = EvaluationResult(
            evaluator_name="test",
            score=0.9,
            passed=True,
            reasoning="Mock test",
            raw_response={},
            tokens_used=50,
        )

        mock_evaluator = MagicMock()
        mock_evaluator.name = "test"
        mock_evaluator.evaluate = AsyncMock(return_value=mock_result)

        def mock_get_evaluator(_name):
            def factory(**_kwargs):
                return mock_evaluator

            return factory

        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}),
            patch("ragaliq.judges.claude.ClaudeJudge", side_effect=mock_judge_factory),
            patch("ragaliq.evaluators.get_evaluator", side_effect=mock_get_evaluator),
        ):
            runner = RagaliQ(judge="claude")

            # Call evaluate() three times (sync API, each creates new event loop)
            result1 = runner.evaluate(sample_test_case)
            result2 = runner.evaluate(sample_test_case)
            result3 = runner.evaluate(sample_test_case)

            # All should succeed
            assert result1.status == EvalStatus.PASSED
            assert result2.status == EvalStatus.PASSED
            assert result3.status == EvalStatus.PASSED

            # Judge should be created exactly once despite 3 calls across 3 event loops
            assert judge_creation_count == 1
