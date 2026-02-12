"""Evaluators package for RagaliQ."""

from ragaliq.core.evaluator import Evaluator
from ragaliq.evaluators.faithfulness import FaithfulnessEvaluator
from ragaliq.evaluators.hallucination import HallucinationEvaluator
from ragaliq.evaluators.relevance import RelevanceEvaluator

EVALUATOR_REGISTRY: dict[str, type[Evaluator]] = {
    "faithfulness": FaithfulnessEvaluator,
    "relevance": RelevanceEvaluator,
    "hallucination": HallucinationEvaluator,
}

__all__ = [
    "FaithfulnessEvaluator",
    "HallucinationEvaluator",
    "RelevanceEvaluator",
    "EVALUATOR_REGISTRY",
]
