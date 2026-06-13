"""RagaliQ — LLM & RAG Evaluation Testing Framework."""

from ragaliq.core.evaluator import EvaluationResult, Evaluator
from ragaliq.core.runner import RagaliQ
from ragaliq.core.test_case import EvalStatus, RAGTestCase, RAGTestResult
from ragaliq.datasets.generator import TestCaseGenerator
from ragaliq.datasets.loader import DatasetLoader
from ragaliq.judges.base import JudgeConfig, LLMJudge
from ragaliq.judges.claude import ClaudeJudge
from ragaliq.reports.console import ConsoleReporter
from ragaliq.reports.html import HTMLReporter
from ragaliq.reports.json_export import JSONReporter

__version__ = "0.2.0"

__all__ = [
    "RagaliQ",
    "RAGTestCase",
    "RAGTestResult",
    "EvalStatus",
    "Evaluator",
    "EvaluationResult",
    "ClaudeJudge",
    "LLMJudge",
    "JudgeConfig",
    "DatasetLoader",
    "TestCaseGenerator",
    "ConsoleReporter",
    "HTMLReporter",
    "JSONReporter",
]
