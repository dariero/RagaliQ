"""LLM Judges package for RagaliQ."""

from ragaliq.judges.base import (
    ClaimsResult,
    ClaimVerdict,
    GeneratedAnswerResult,
    GeneratedQuestionsResult,
    JudgeAPIError,
    JudgeConfig,
    JudgeError,
    JudgeResponseError,
    JudgeResult,
    LLMJudge,
)
from ragaliq.judges.base_judge import BaseJudge
from ragaliq.judges.claude import ClaudeJudge
from ragaliq.judges.models import DEFAULT_JUDGE_MODEL, GOLD_STANDARD_JUDGE_MODEL
from ragaliq.judges.trace import JudgeTrace, TraceCollector
from ragaliq.judges.transport import ClaudeTransport, JudgeTransport, TransportResponse

__all__ = [
    "BaseJudge",
    "ClaimsResult",
    "ClaimVerdict",
    "ClaudeJudge",
    "ClaudeTransport",
    "DEFAULT_JUDGE_MODEL",
    "GeneratedAnswerResult",
    "GeneratedQuestionsResult",
    "GOLD_STANDARD_JUDGE_MODEL",
    "JudgeAPIError",
    "JudgeConfig",
    "JudgeError",
    "JudgeResponseError",
    "JudgeResult",
    "JudgeTrace",
    "JudgeTransport",
    "LLMJudge",
    "TraceCollector",
    "TransportResponse",
]
