"""Main test runner for RagaliQ."""

import asyncio
import logging
import threading
from typing import Any, Literal

from ragaliq.core.evaluator import EvaluationResult, Evaluator
from ragaliq.core.test_case import EvalStatus, RAGTestCase, RAGTestResult
from ragaliq.judges.base import JudgeConfig, LLMJudge

logger = logging.getLogger(__name__)


class RagaliQ:
    """
    Main entry point for RAG testing.

    Orchestrates test execution, evaluator management, and result collection.

    Example:
        >>> tester = RagaliQ(judge="claude")
        >>> result = tester.evaluate(test_case)
        >>> print(result.scores)

        # With custom configuration
        >>> from ragaliq.judges import JudgeConfig
        >>> config = JudgeConfig(model="claude-sonnet-4-20250514", temperature=0.0)
        >>> tester = RagaliQ(judge_config=config, api_key="sk-ant-...")

        # With pre-configured judge
        >>> from ragaliq.judges import ClaudeJudge
        >>> judge = ClaudeJudge(api_key="sk-ant-...")
        >>> tester = RagaliQ(judge=judge)
    """

    def __init__(
        self,
        judge: Literal["claude", "openai"] | LLMJudge = "claude",
        evaluators: list[str] | None = None,
        default_threshold: float = 0.7,
        judge_config: JudgeConfig | None = None,
        api_key: str | None = None,
        max_concurrency: int = 5,
    ) -> None:
        """
        Initialize RagaliQ.

        Args:
            judge: Which LLM to use as judge ("claude" or "openai"), or
                a pre-configured LLMJudge instance.
            evaluators: List of evaluator names to use. Defaults to all.
            default_threshold: Default passing threshold for evaluators.
            judge_config: Optional configuration for the judge (model, temperature, etc.).
            api_key: Optional API key for the judge. Falls back to environment variable.
            max_concurrency: Maximum number of concurrent evaluations in batch mode.
        """
        self.evaluator_names = evaluators or ["faithfulness", "relevance"]
        self.default_threshold = default_threshold
        self.max_concurrency = max_concurrency
        self._judge_config = judge_config
        self._api_key = api_key

        # Handle pre-configured judge instance vs judge type string
        if isinstance(judge, LLMJudge):
            self._judge: LLMJudge | None = judge
            self.judge_type: Literal["claude", "openai"] | None = None
        else:
            self._judge = None
            self.judge_type = judge

        self._evaluators: list[Evaluator] = []
        self._init_lock = threading.Lock()

    def __repr__(self) -> str:
        return f"RagaliQ(judge_type={self.judge_type!r}, evaluators={self.evaluator_names!r})"

    def _init_judge(self) -> None:
        """Lazily initialize the LLM judge."""
        if self._judge is not None:
            return

        from ragaliq.judges.claude import ClaudeJudge

        if self.judge_type == "claude":
            self._judge = ClaudeJudge(
                config=self._judge_config,
                api_key=self._api_key,
            )
        elif self.judge_type == "openai":
            # OpenAI judge not yet implemented
            raise NotImplementedError("OpenAI judge not yet implemented")
        else:
            raise ValueError(f"Unknown judge type: {self.judge_type}")

    def _init_evaluators(self) -> None:
        """Initialize evaluators based on configuration."""
        if self._evaluators:
            return

        from ragaliq.evaluators import get_evaluator

        for name in self.evaluator_names:
            evaluator_class = get_evaluator(name)
            self._evaluators.append(evaluator_class(threshold=self.default_threshold))

    async def _ensure_initialized(self) -> None:
        """
        Ensure judge and evaluators are initialized exactly once.

        Uses a threading lock to prevent race conditions when multiple
        concurrent calls (or repeated sync calls across different event loops)
        attempt initialization. Threading lock is used instead of asyncio.Lock
        to avoid loop-binding issues when the same runner is used across
        multiple asyncio.run() calls.
        """
        with self._init_lock:
            self._init_judge()
            self._init_evaluators()

    async def evaluate_async(self, test_case: RAGTestCase) -> RAGTestResult:
        """
        Evaluate a single test case asynchronously.

        Args:
            test_case: The test case to evaluate.

        Returns:
            RAGTestResult with all metric scores.
        """
        import time

        start_time = time.perf_counter()

        await self._ensure_initialized()

        if self._judge is None:
            raise RuntimeError("Judge must be initialized before evaluation")

        scores: dict[str, float] = {}
        details: dict[str, dict[str, Any]] = {}
        total_tokens = 0
        all_passed = True
        has_error = False

        # Run all evaluators in parallel
        async def _run_evaluator(evaluator: Evaluator) -> tuple[str, EvaluationResult]:
            """Run a single evaluator, returning (evaluator_name, result)."""
            try:
                result = await evaluator.evaluate(test_case, self._judge)
                return evaluator.name, result
            except Exception as exc:
                logger.exception("Evaluator '%s' failed", evaluator.name)
                error_result = EvaluationResult(
                    evaluator_name=evaluator.name,
                    score=0.0,
                    passed=False,
                    reasoning=f"Evaluation failed: {exc}",
                    error=f"{type(exc).__name__}: {exc}",
                    raw_response={"exception": str(exc)},
                    tokens_used=0,
                )
                return evaluator.name, error_result

        eval_tasks = [_run_evaluator(ev) for ev in self._evaluators]
        results = await asyncio.gather(*eval_tasks)

        # Collect results
        for evaluator_name, result in results:
            scores[evaluator_name] = result.score
            details[evaluator_name] = {
                "reasoning": result.reasoning,
                "passed": result.passed,
                "raw": result.raw_response,
            }
            if result.error:
                details[evaluator_name]["error"] = result.error
                has_error = True

            total_tokens += result.tokens_used
            if not result.passed:
                all_passed = False

        execution_time = int((time.perf_counter() - start_time) * 1000)

        # Determine status: ERROR trumps FAILED/PASSED
        if has_error:
            status = EvalStatus.ERROR
        elif all_passed:
            status = EvalStatus.PASSED
        else:
            status = EvalStatus.FAILED

        return RAGTestResult(
            test_case=test_case,
            status=status,
            scores=scores,
            details=details,
            execution_time_ms=execution_time,
            judge_tokens_used=total_tokens,
        )

    def evaluate(self, test_case: RAGTestCase) -> RAGTestResult:
        """
        Evaluate a single test case synchronously.

        Args:
            test_case: The test case to evaluate.

        Returns:
            RAGTestResult with all metric scores.
        """
        import asyncio

        return asyncio.run(self.evaluate_async(test_case))

    async def evaluate_batch_async(
        self, test_cases: list[RAGTestCase], max_concurrency: int | None = None
    ) -> list[RAGTestResult]:
        """
        Evaluate multiple test cases asynchronously.

        Args:
            test_cases: List of test cases to evaluate.
            max_concurrency: Optional override for maximum concurrent evaluations.

        Returns:
            List of RAGTestResults in the same order.
        """
        import asyncio

        concurrency = max_concurrency if max_concurrency is not None else self.max_concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_evaluate(tc: RAGTestCase) -> RAGTestResult:
            async with semaphore:
                try:
                    return await self.evaluate_async(tc)
                except Exception as exc:
                    logger.exception("Batch evaluation failed for test case '%s'", tc.id)
                    # Return error result instead of propagating exception
                    return RAGTestResult(
                        test_case=tc,
                        status=EvalStatus.ERROR,
                        scores={},
                        details={"error": f"{type(exc).__name__}: {exc}"},
                        execution_time_ms=0,
                        judge_tokens_used=0,
                    )

        tasks = [bounded_evaluate(tc) for tc in test_cases]
        return await asyncio.gather(*tasks)

    def evaluate_batch(self, test_cases: list[RAGTestCase]) -> list[RAGTestResult]:
        """
        Evaluate multiple test cases synchronously.

        Args:
            test_cases: List of test cases to evaluate.

        Returns:
            List of RAGTestResults in the same order.
        """
        import asyncio

        return asyncio.run(self.evaluate_batch_async(test_cases))
