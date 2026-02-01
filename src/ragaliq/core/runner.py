"""Main test runner for RagaliQ."""

from typing import Any, Literal

from ragaliq.core.evaluator import EvaluationResult, Evaluator
from ragaliq.core.test_case import EvalStatus, RAGTestCase, RAGTestResult
from ragaliq.judges.base import JudgeConfig, LLMJudge


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
        >>> tester = RagaliQ(judge_config=config, api_key="sk-...")

        # With pre-configured judge
        >>> from ragaliq.judges import ClaudeJudge
        >>> judge = ClaudeJudge(api_key="sk-...")
        >>> tester = RagaliQ(judge=judge)
    """

    def __init__(
        self,
        judge: Literal["claude", "openai"] | LLMJudge = "claude",
        evaluators: list[str] | None = None,
        default_threshold: float = 0.7,
        judge_config: JudgeConfig | None = None,
        api_key: str | None = None,
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
        """
        self.evaluator_names = evaluators or ["faithfulness", "relevance"]
        self.default_threshold = default_threshold
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

        # TODO: Implement evaluator registry and initialization
        # from ragtestkit.evaluators import get_evaluator
        #
        # for name in self.evaluator_names:
        #     self._evaluators.append(get_evaluator(name, self.default_threshold))
        raise NotImplementedError("Evaluator initialization not yet implemented")

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

        self._init_judge()
        self._init_evaluators()

        assert self._judge is not None, "Judge must be initialized"

        scores: dict[str, float] = {}
        details: dict[str, dict[str, Any]] = {}
        total_tokens = 0
        all_passed = True

        for evaluator in self._evaluators:
            result: EvaluationResult = await evaluator.evaluate(test_case, self._judge)
            scores[evaluator.name] = result.score
            details[evaluator.name] = {
                "reasoning": result.reasoning,
                "passed": result.passed,
                "raw": result.raw_response,
            }
            if not result.passed:
                all_passed = False

        execution_time = int((time.perf_counter() - start_time) * 1000)

        return RAGTestResult(
            test_case=test_case,
            status=EvalStatus.PASSED if all_passed else EvalStatus.FAILED,
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

    async def evaluate_batch_async(self, test_cases: list[RAGTestCase]) -> list[RAGTestResult]:
        """
        Evaluate multiple test cases asynchronously.

        Args:
            test_cases: List of test cases to evaluate.

        Returns:
            List of RAGTestResults in the same order.
        """
        import asyncio

        tasks = [self.evaluate_async(tc) for tc in test_cases]
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
