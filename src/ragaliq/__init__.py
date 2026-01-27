"""RagaliQ - LLM Testing Framework for RAG Systems."""

from ragaliq.core.evaluator import EvaluationResult, Evaluator
from ragaliq.core.runner import RagaliQ
from ragaliq.core.test_case import RAGTestCase, RAGTestResult

__version__ = "0.1.0"

__all__ = [
    "RagaliQ",
    "RAGTestCase",
    "RAGTestResult",
    "Evaluator",
    "EvaluationResult",
]
