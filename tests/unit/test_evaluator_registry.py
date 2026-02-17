"""Unit tests for evaluator registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ragaliq.core.evaluator import EvaluationResult, Evaluator
from ragaliq.evaluators.registry import (
    get_evaluator,
    list_evaluators,
    register_evaluator,
    register_evaluator_class,
)

if TYPE_CHECKING:
    from ragaliq.core.test_case import RAGTestCase
    from ragaliq.judges.base import LLMJudge


@pytest.fixture
def clean_registry():
    """
    Snapshot and restore _REGISTRY around tests that register custom evaluators.

    This ensures test isolation when mutating the global registry.
    """
    import ragaliq.evaluators.registry as reg

    original = reg._REGISTRY.copy()
    yield
    reg._REGISTRY.clear()
    reg._REGISTRY.update(original)


class TestRegisterEvaluatorDecorator:
    """Tests for the @register_evaluator decorator."""

    def test_registers_evaluator_class(self, clean_registry) -> None:  # noqa: ARG002
        """Decorator should register the class in the global registry."""

        @register_evaluator("test_metric")
        class TestEvaluator(Evaluator):
            name = "test_metric"
            description = "Test evaluator"

            async def evaluate(self, _test_case: RAGTestCase, _judge: LLMJudge) -> EvaluationResult:
                return EvaluationResult(
                    evaluator_name=self.name,
                    score=1.0,
                    passed=True,
                    reasoning="Test",
                )

        assert get_evaluator("test_metric") is TestEvaluator

    def test_returns_class_unchanged(self, clean_registry) -> None:  # noqa: ARG002
        """Decorator should return the class unchanged (type-preserving)."""

        @register_evaluator("test_metric")
        class TestEvaluator(Evaluator):
            name = "test_metric"
            description = "Test evaluator"

            async def evaluate(self, _test_case: RAGTestCase, _judge: LLMJudge) -> EvaluationResult:
                return EvaluationResult(
                    evaluator_name=self.name,
                    score=1.0,
                    passed=True,
                    reasoning="Test",
                )

        # Decorator should return the class itself
        assert TestEvaluator.name == "test_metric"
        assert TestEvaluator.description == "Test evaluator"

    def test_rejects_non_evaluator_class(self, clean_registry) -> None:  # noqa: ARG002
        """Decorator should reject classes that don't subclass Evaluator."""
        with pytest.raises(ValueError, match="must be a subclass of Evaluator"):

            @register_evaluator("not_an_evaluator")
            class NotAnEvaluator:
                pass

    def test_rejects_empty_name(self, clean_registry) -> None:  # noqa: ARG002
        """Decorator should reject empty evaluator names."""
        with pytest.raises(ValueError, match="name cannot be empty"):

            @register_evaluator("")
            class TestEvaluator(Evaluator):
                name = "test"
                description = "Test"

                async def evaluate(
                    self, _test_case: RAGTestCase, _judge: LLMJudge
                ) -> EvaluationResult:
                    return EvaluationResult(
                        evaluator_name=self.name,
                        score=1.0,
                        passed=True,
                        reasoning="Test",
                    )

    def test_rejects_duplicate_name(self, clean_registry) -> None:  # noqa: ARG002
        """Decorator should reject duplicate evaluator names."""

        @register_evaluator("duplicate")
        class FirstEvaluator(Evaluator):
            name = "duplicate"
            description = "First"

            async def evaluate(self, _test_case: RAGTestCase, _judge: LLMJudge) -> EvaluationResult:
                return EvaluationResult(
                    evaluator_name=self.name,
                    score=1.0,
                    passed=True,
                    reasoning="Test",
                )

        # Attempting to register with the same name should fail
        with pytest.raises(ValueError, match="already registered"):

            @register_evaluator("duplicate")
            class SecondEvaluator(Evaluator):
                name = "duplicate"
                description = "Second"

                async def evaluate(
                    self, _test_case: RAGTestCase, _judge: LLMJudge
                ) -> EvaluationResult:
                    return EvaluationResult(
                        evaluator_name=self.name,
                        score=1.0,
                        passed=True,
                        reasoning="Test",
                    )


class TestRegisterEvaluatorClass:
    """Tests for the programmatic register_evaluator_class() function."""

    def test_registers_evaluator_class(self, clean_registry) -> None:  # noqa: ARG002
        """Should register the class programmatically."""

        class TestEvaluator(Evaluator):
            name = "test_metric"
            description = "Test evaluator"

            async def evaluate(self, _test_case: RAGTestCase, _judge: LLMJudge) -> EvaluationResult:
                return EvaluationResult(
                    evaluator_name=self.name,
                    score=1.0,
                    passed=True,
                    reasoning="Test",
                )

        register_evaluator_class("test_metric", TestEvaluator)

        assert get_evaluator("test_metric") is TestEvaluator

    def test_rejects_non_evaluator_class(self, clean_registry) -> None:  # noqa: ARG002
        """Should reject classes that don't subclass Evaluator."""

        class NotAnEvaluator:
            pass

        with pytest.raises(ValueError, match="must be a subclass of Evaluator"):
            register_evaluator_class("not_an_evaluator", NotAnEvaluator)  # type: ignore[arg-type]

    def test_rejects_empty_name(self, clean_registry) -> None:  # noqa: ARG002
        """Should reject empty evaluator names."""

        class TestEvaluator(Evaluator):
            name = "test"
            description = "Test"

            async def evaluate(self, _test_case: RAGTestCase, _judge: LLMJudge) -> EvaluationResult:
                return EvaluationResult(
                    evaluator_name=self.name,
                    score=1.0,
                    passed=True,
                    reasoning="Test",
                )

        with pytest.raises(ValueError, match="name cannot be empty"):
            register_evaluator_class("", TestEvaluator)

        with pytest.raises(ValueError, match="name cannot be empty"):
            register_evaluator_class("   ", TestEvaluator)

    def test_rejects_duplicate_name(self, clean_registry) -> None:  # noqa: ARG002
        """Should reject duplicate evaluator names."""

        class FirstEvaluator(Evaluator):
            name = "duplicate"
            description = "First"

            async def evaluate(self, _test_case: RAGTestCase, _judge: LLMJudge) -> EvaluationResult:
                return EvaluationResult(
                    evaluator_name=self.name,
                    score=1.0,
                    passed=True,
                    reasoning="Test",
                )

        class SecondEvaluator(Evaluator):
            name = "duplicate"
            description = "Second"

            async def evaluate(self, _test_case: RAGTestCase, _judge: LLMJudge) -> EvaluationResult:
                return EvaluationResult(
                    evaluator_name=self.name,
                    score=1.0,
                    passed=True,
                    reasoning="Test",
                )

        register_evaluator_class("duplicate", FirstEvaluator)

        with pytest.raises(ValueError, match="already registered"):
            register_evaluator_class("duplicate", SecondEvaluator)


class TestGetEvaluator:
    """Tests for the get_evaluator() lookup function."""

    def test_returns_registered_evaluator(self):
        """Should return the correct evaluator class for a registered name."""
        from ragaliq.evaluators import FaithfulnessEvaluator

        evaluator_class = get_evaluator("faithfulness")
        assert evaluator_class is FaithfulnessEvaluator

    def test_raises_for_unknown_evaluator(self):
        """Should raise ValueError for unknown evaluator names."""
        with pytest.raises(ValueError, match="Unknown evaluator: 'nonexistent'"):
            get_evaluator("nonexistent")

    def test_error_message_includes_available_evaluators(self):
        """Error message should list available evaluators."""
        with pytest.raises(ValueError, match="Available evaluators:"):
            get_evaluator("nonexistent")

        # Should include at least the built-in evaluators
        with pytest.raises(ValueError, match="faithfulness.*hallucination.*relevance"):
            get_evaluator("nonexistent")


class TestListEvaluators:
    """Tests for the list_evaluators() function."""

    def test_returns_sorted_list(self):
        """Should return a sorted list of evaluator names."""
        evaluators = list_evaluators()

        assert isinstance(evaluators, list)
        assert evaluators == sorted(evaluators)

    def test_contains_built_in_evaluators(self):
        """Should contain all built-in evaluators."""
        evaluators = list_evaluators()

        assert "faithfulness" in evaluators
        assert "relevance" in evaluators
        assert "hallucination" in evaluators

    def test_includes_custom_evaluators(self, clean_registry) -> None:  # noqa: ARG002
        """Should include custom evaluators after registration."""

        @register_evaluator("custom_metric")
        class CustomEvaluator(Evaluator):
            name = "custom_metric"
            description = "Custom"

            async def evaluate(self, _test_case: RAGTestCase, _judge: LLMJudge) -> EvaluationResult:
                return EvaluationResult(
                    evaluator_name=self.name,
                    score=1.0,
                    passed=True,
                    reasoning="Test",
                )

        evaluators = list_evaluators()
        assert "custom_metric" in evaluators


class TestBuiltInRegistration:
    """Tests that built-in evaluators are auto-registered at import time."""

    def test_faithfulness_registered(self):
        """FaithfulnessEvaluator should be registered as 'faithfulness'."""
        from ragaliq.evaluators import FaithfulnessEvaluator

        assert get_evaluator("faithfulness") is FaithfulnessEvaluator

    def test_relevance_registered(self):
        """RelevanceEvaluator should be registered as 'relevance'."""
        from ragaliq.evaluators import RelevanceEvaluator

        assert get_evaluator("relevance") is RelevanceEvaluator

    def test_hallucination_registered(self):
        """HallucinationEvaluator should be registered as 'hallucination'."""
        from ragaliq.evaluators import HallucinationEvaluator

        assert get_evaluator("hallucination") is HallucinationEvaluator

    def test_all_built_ins_retrievable(self):
        """All built-in evaluators should be retrievable via get_evaluator()."""
        from ragaliq.evaluators import (
            FaithfulnessEvaluator,
            HallucinationEvaluator,
            RelevanceEvaluator,
        )

        assert get_evaluator("faithfulness") is FaithfulnessEvaluator
        assert get_evaluator("relevance") is RelevanceEvaluator
        assert get_evaluator("hallucination") is HallucinationEvaluator


class TestCustomEvaluatorRegistration:
    """Tests for user-defined custom evaluator registration."""

    def test_custom_evaluator_via_decorator(self, clean_registry) -> None:  # noqa: ARG002
        """Users should be able to register custom evaluators using the decorator."""

        @register_evaluator("user_custom")
        class UserCustomEvaluator(Evaluator):
            name = "user_custom"
            description = "User's custom evaluator"

            async def evaluate(self, _test_case: RAGTestCase, _judge: LLMJudge) -> EvaluationResult:
                return EvaluationResult(
                    evaluator_name=self.name,
                    score=0.95,
                    passed=True,
                    reasoning="Custom logic applied",
                )

        # Should be retrievable
        evaluator_class = get_evaluator("user_custom")
        assert evaluator_class is UserCustomEvaluator

        # Should be in list
        assert "user_custom" in list_evaluators()

        # Should be instantiable and usable
        evaluator = evaluator_class(threshold=0.8)
        assert evaluator.name == "user_custom"
        assert evaluator.threshold == 0.8

    def test_custom_evaluator_via_programmatic_api(self, clean_registry) -> None:  # noqa: ARG002
        """Users should be able to register custom evaluators programmatically."""

        class UserCustomEvaluator(Evaluator):
            name = "programmatic_custom"
            description = "Programmatically registered"

            async def evaluate(self, _test_case: RAGTestCase, _judge: LLMJudge) -> EvaluationResult:
                return EvaluationResult(
                    evaluator_name=self.name,
                    score=0.85,
                    passed=True,
                    reasoning="Programmatic registration",
                )

        register_evaluator_class("programmatic_custom", UserCustomEvaluator)

        # Should be retrievable
        evaluator_class = get_evaluator("programmatic_custom")
        assert evaluator_class is UserCustomEvaluator

        # Should be in list
        assert "programmatic_custom" in list_evaluators()

    def test_custom_evaluator_with_custom_threshold(self, clean_registry) -> None:  # noqa: ARG002
        """Custom evaluators should support custom default thresholds."""

        @register_evaluator("strict_custom")
        class StrictCustomEvaluator(Evaluator):
            name = "strict_custom"
            description = "Strict evaluator"
            threshold = 0.95  # Custom default threshold

            async def evaluate(self, _test_case: RAGTestCase, _judge: LLMJudge) -> EvaluationResult:
                return EvaluationResult(
                    evaluator_name=self.name,
                    score=0.9,
                    passed=False,  # Would fail with 0.95 threshold
                    reasoning="Strict evaluation",
                )

        evaluator_class = get_evaluator("strict_custom")
        evaluator = evaluator_class()

        # Should use custom default threshold
        assert evaluator.threshold == 0.95
