"""Evaluators package for RagaliQ."""

from ragaliq.core.evaluator import Evaluator

# Import evaluator classes (triggers decorator registration at import time)
from ragaliq.evaluators.context_precision import ContextPrecisionEvaluator
from ragaliq.evaluators.context_recall import ContextRecallEvaluator
from ragaliq.evaluators.faithfulness import FaithfulnessEvaluator
from ragaliq.evaluators.hallucination import HallucinationEvaluator

# Import registry functions
from ragaliq.evaluators.registry import (
    get_evaluator,
    list_evaluators,
    register_evaluator,
    register_evaluator_class,
)
from ragaliq.evaluators.relevance import RelevanceEvaluator

__all__ = [
    "ContextPrecisionEvaluator",
    "ContextRecallEvaluator",
    "Evaluator",
    "FaithfulnessEvaluator",
    "HallucinationEvaluator",
    "RelevanceEvaluator",
    "get_evaluator",
    "list_evaluators",
    "register_evaluator",
    "register_evaluator_class",
]
