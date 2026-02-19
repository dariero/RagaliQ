"""Integration tests for the full evaluation pipeline with a fake transport.

These tests exercise the chain:
    FakeTransport → BaseJudge → Evaluator → Runner → RAGTestResult

No network calls, no mocks on internals — the transport returns canned JSON
strings and everything else runs as production code. This catches integration
issues in JSON parsing, score clamping, Pydantic model construction, and
evaluator aggregation that unit tests (which mock at LLMJudge level) miss.
"""

import json

import pytest

from ragaliq.core.runner import RagaliQ
from ragaliq.core.test_case import EvalStatus, RAGTestCase, RAGTestResult
from ragaliq.evaluators.faithfulness import FaithfulnessEvaluator
from ragaliq.evaluators.hallucination import HallucinationEvaluator
from ragaliq.evaluators.relevance import RelevanceEvaluator
from ragaliq.judges.base_judge import BaseJudge
from ragaliq.judges.trace import TraceCollector
from ragaliq.judges.transport import TransportResponse

# ---------------------------------------------------------------------------
# FakeTransport — canned JSON responses keyed by operation
# ---------------------------------------------------------------------------


class FakeTransport:
    """Transport that returns canned JSON based on prompt content.

    Inspects the system_prompt / user_prompt to decide which operation is
    being requested, then returns the matching canned JSON.  This is the
    minimal implementation of the JudgeTransport protocol needed to drive
    the full BaseJudge → Evaluator chain.
    """

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self.call_count = 0
        self.calls: list[dict[str, str]] = []
        self._responses = responses or {}

    async def send(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str = "fake-model",
        temperature: float = 0.0,  # noqa: ARG002
        max_tokens: int = 1024,  # noqa: ARG002
    ) -> TransportResponse:
        self.call_count += 1
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})

        # Determine operation from prompt content
        text = self._resolve_response(system_prompt, user_prompt)

        return TransportResponse(
            text=text,
            input_tokens=50,
            output_tokens=30,
            model=model,
        )

    def _resolve_response(self, system_prompt: str, user_prompt: str) -> str:
        """Pick the right canned response based on prompt keywords."""
        combined = (system_prompt + user_prompt).lower()

        # Check explicit overrides first
        for key, response in self._responses.items():
            if key.lower() in combined:
                return response

        # Default heuristics based on BaseJudge prompt template names
        if "extract" in combined and "claim" in combined:
            return json.dumps({"claims": ["Claim A", "Claim B"]})
        if "verify" in combined and "claim" in combined:
            return json.dumps(
                {
                    "verdict": "SUPPORTED",
                    "evidence": "The context confirms this.",
                }
            )
        if "faithfulness" in combined:
            return json.dumps({"score": 0.9, "reasoning": "Mostly faithful."})
        if "relevance" in combined or "relevant" in combined:
            return json.dumps({"score": 0.85, "reasoning": "Relevant response."})
        if "question" in combined and "generate" in combined:
            return json.dumps({"questions": ["Q1?", "Q2?"]})
        if "answer" in combined and "generate" in combined:
            return json.dumps({"answer": "The answer is 42."})

        # Fallback
        return json.dumps({"score": 0.5, "reasoning": "Default response."})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_transport():
    return FakeTransport()


@pytest.fixture
def fake_judge(fake_transport):
    return BaseJudge(transport=fake_transport)


@pytest.fixture
def traced_judge(fake_transport):
    collector = TraceCollector()
    judge = BaseJudge(transport=fake_transport, trace_collector=collector)
    return judge, collector


@pytest.fixture
def test_case():
    return RAGTestCase(
        id="integration-001",
        name="Capital of France",
        query="What is the capital of France?",
        context=[
            "France is a country in Western Europe.",
            "The capital of France is Paris.",
        ],
        response="The capital of France is Paris.",
    )


# ---------------------------------------------------------------------------
# S1: Transport → BaseJudge integration tests
# ---------------------------------------------------------------------------


class TestTransportToJudge:
    """Verify FakeTransport → BaseJudge → parsed result chain."""

    @pytest.mark.asyncio
    async def test_evaluate_faithfulness_through_transport(self, fake_judge):
        """Full chain: transport JSON → _parse_json_response → JudgeResult."""
        result = await fake_judge.evaluate_faithfulness(
            response="Paris is the capital.",
            context=["The capital of France is Paris."],
        )
        assert result.score == 0.9
        assert result.reasoning == "Mostly faithful."
        assert result.tokens_used == 80  # 50 input + 30 output

    @pytest.mark.asyncio
    async def test_evaluate_relevance_through_transport(self, fake_judge):
        result = await fake_judge.evaluate_relevance(
            query="What is the capital?",
            response="The capital is Paris.",
        )
        assert result.score == 0.85
        assert result.reasoning == "Relevant response."

    @pytest.mark.asyncio
    async def test_extract_claims_through_transport(self, fake_judge):
        result = await fake_judge.extract_claims("Paris is in France. It has the Eiffel Tower.")
        assert result.claims == ["Claim A", "Claim B"]
        assert result.tokens_used == 80

    @pytest.mark.asyncio
    async def test_verify_claim_through_transport(self, fake_judge):
        result = await fake_judge.verify_claim(
            "Paris is in France",
            context=["France's capital is Paris."],
        )
        assert result.verdict == "SUPPORTED"
        assert result.evidence == "The context confirms this."

    @pytest.mark.asyncio
    async def test_score_clamping_above_1(self):
        """Transport returns score > 1.0, BaseJudge clamps to 1.0."""
        transport = FakeTransport(
            responses={"faithfulness": json.dumps({"score": 1.5, "reasoning": "Over"})}
        )
        judge = BaseJudge(transport=transport)
        result = await judge.evaluate_faithfulness(
            response="test",
            context=["test context"],
        )
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_score_clamping_below_0(self):
        """Transport returns score < 0.0, BaseJudge clamps to 0.0."""
        transport = FakeTransport(
            responses={"faithfulness": json.dumps({"score": -0.5, "reasoning": "Under"})}
        )
        judge = BaseJudge(transport=transport)
        result = await judge.evaluate_faithfulness(
            response="test",
            context=["test context"],
        )
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_trace_collector_receives_traces(self, traced_judge):
        """Verify TraceCollector captures calls through the full chain."""
        judge, collector = traced_judge
        await judge.evaluate_faithfulness(
            response="test",
            context=["context"],
        )
        assert len(collector.traces) == 1
        trace = collector.traces[0]
        assert trace.operation == "evaluate_faithfulness"
        assert trace.success is True
        assert trace.input_tokens == 50
        assert trace.output_tokens == 30

    @pytest.mark.asyncio
    async def test_markdown_wrapped_json(self):
        """Transport returns JSON wrapped in markdown — BaseJudge strips it."""
        wrapped = '```json\n{"score": 0.75, "reasoning": "Wrapped"}\n```'
        transport = FakeTransport(responses={"faithfulness": wrapped})
        judge = BaseJudge(transport=transport)
        result = await judge.evaluate_faithfulness(
            response="test",
            context=["ctx"],
        )
        assert result.score == 0.75
        assert result.reasoning == "Wrapped"


# ---------------------------------------------------------------------------
# S1 continued: Transport → BaseJudge → Evaluator chain
# ---------------------------------------------------------------------------


class TestTransportToEvaluator:
    """Verify the claim pipeline works through real BaseJudge, not mocked LLMJudge."""

    @pytest.mark.asyncio
    async def test_faithfulness_evaluator_with_fake_transport(
        self,
        fake_judge,
        test_case,
    ):
        """FaithfulnessEvaluator: extract_claims → verify_claim → score."""
        evaluator = FaithfulnessEvaluator()
        result = await evaluator.evaluate(test_case, fake_judge)

        # FakeTransport returns 2 claims, both SUPPORTED → score = 1.0
        assert result.score == 1.0
        assert result.passed is True
        assert result.evaluator_name == "faithfulness"
        assert "claims" in result.raw_response

    @pytest.mark.asyncio
    async def test_hallucination_evaluator_with_fake_transport(
        self,
        fake_judge,
        test_case,
    ):
        """HallucinationEvaluator: same pipeline, inverse scoring."""
        evaluator = HallucinationEvaluator()
        result = await evaluator.evaluate(test_case, fake_judge)

        # All SUPPORTED → 0 hallucinated → score = 1.0
        assert result.score == 1.0
        assert result.passed is True
        assert result.evaluator_name == "hallucination"

    @pytest.mark.asyncio
    async def test_relevance_evaluator_with_fake_transport(
        self,
        fake_judge,
        test_case,
    ):
        """RelevanceEvaluator: thin adapter over judge.evaluate_relevance()."""
        evaluator = RelevanceEvaluator()
        result = await evaluator.evaluate(test_case, fake_judge)

        assert result.score == 0.85
        assert result.passed is True
        assert result.evaluator_name == "relevance"

    @pytest.mark.asyncio
    async def test_faithfulness_with_mixed_verdicts(self, test_case):
        """One claim SUPPORTED, one CONTRADICTED → score = 0.5."""
        call_count = 0

        class AlternatingTransport:
            async def send(self, system_prompt, user_prompt, **kwargs):  # noqa: ARG002
                nonlocal call_count
                call_count += 1
                combined = (system_prompt + user_prompt).lower()

                if "extract" in combined and "claim" in combined:
                    text = json.dumps({"claims": ["Claim A", "Claim B"]})
                elif "verify" in combined:
                    # Alternate verdicts
                    if call_count % 2 == 0:
                        text = json.dumps({"verdict": "SUPPORTED", "evidence": "ok"})
                    else:
                        text = json.dumps({"verdict": "CONTRADICTED", "evidence": "nope"})
                else:
                    text = json.dumps({"score": 0.5, "reasoning": "default"})

                return TransportResponse(
                    text=text,
                    input_tokens=10,
                    output_tokens=10,
                    model="fake",
                )

        judge = BaseJudge(transport=AlternatingTransport())
        evaluator = FaithfulnessEvaluator()
        result = await evaluator.evaluate(test_case, judge)

        assert result.score == 0.5
        assert result.passed is False  # threshold=0.7

    @pytest.mark.asyncio
    async def test_transport_call_count(self, fake_transport, fake_judge, test_case):
        """Faithfulness makes 1 extract_claims + N verify_claim calls."""
        evaluator = FaithfulnessEvaluator()
        await evaluator.evaluate(test_case, fake_judge)

        # 1 extract + 2 verify (FakeTransport returns 2 claims)
        assert fake_transport.call_count == 3


# ---------------------------------------------------------------------------
# S4: Full E2E pipeline — RagaliQ runner with FakeTransport
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """Full pipeline: RagaliQ.evaluate_async → evaluators → BaseJudge → FakeTransport."""

    @pytest.mark.asyncio
    async def test_single_test_case_evaluation(self, test_case):
        """End-to-end: runner picks evaluators, runs them, builds RAGTestResult."""
        transport = FakeTransport()
        judge = BaseJudge(transport=transport)
        runner = RagaliQ(
            judge=judge,
            evaluators=["faithfulness", "relevance"],
        )

        result = await runner.evaluate_async(test_case)

        assert isinstance(result, RAGTestResult)
        assert result.status == EvalStatus.PASSED
        assert result.passed is True
        assert "faithfulness" in result.scores
        assert "relevance" in result.scores
        assert result.scores["faithfulness"] == 1.0  # All claims SUPPORTED
        assert result.scores["relevance"] == 0.85
        assert result.execution_time_ms >= 0
        assert result.judge_tokens_used > 0

    @pytest.mark.asyncio
    async def test_batch_evaluation(self, test_case):
        """Batch mode: multiple test cases through the full pipeline."""
        transport = FakeTransport()
        judge = BaseJudge(transport=transport)
        runner = RagaliQ(
            judge=judge,
            evaluators=["faithfulness"],
        )

        cases = [test_case, test_case]
        results = await runner.evaluate_batch_async(cases)

        assert len(results) == 2
        assert all(r.passed for r in results)
        assert all(r.status == EvalStatus.PASSED for r in results)

    @pytest.mark.asyncio
    async def test_failing_test_case(self):
        """Pipeline correctly marks a test case as FAILED when score < threshold."""
        transport = FakeTransport(
            responses={
                "relevant": json.dumps({"score": 0.3, "reasoning": "Off topic."}),
            }
        )
        judge = BaseJudge(transport=transport)
        runner = RagaliQ(
            judge=judge,
            evaluators=["relevance"],
            default_threshold=0.7,
        )

        test_case = RAGTestCase(
            id="fail-001",
            name="Irrelevant response",
            query="What is Python?",
            context=["Python is a programming language."],
            response="The weather is nice today.",
        )

        result = await runner.evaluate_async(test_case)

        assert result.status == EvalStatus.FAILED
        assert result.passed is False
        assert result.scores["relevance"] == 0.3

    @pytest.mark.asyncio
    async def test_trace_collection_through_pipeline(self, test_case):
        """TraceCollector accumulates traces from the full E2E pipeline."""
        collector = TraceCollector()
        transport = FakeTransport()
        judge = BaseJudge(transport=transport, trace_collector=collector)
        runner = RagaliQ(
            judge=judge,
            evaluators=["faithfulness", "relevance"],
        )

        await runner.evaluate_async(test_case)

        # faithfulness: 1 extract_claims + 2 verify_claim = 3
        # relevance: 1 evaluate_relevance = 1
        # total = 4 traces
        assert len(collector.traces) == 4
        operations = [t.operation for t in collector.traces]
        assert "extract_claims" in operations
        assert "verify_claim" in operations
        assert "evaluate_relevance" in operations
        assert all(t.success for t in collector.traces)

    @pytest.mark.asyncio
    async def test_error_envelope_on_transport_failure(self):
        """Transport exception → error envelope, not crash."""

        class FailingTransport:
            async def send(self, **kwargs):  # noqa: ARG002
                raise ConnectionError("Network down")

        judge = BaseJudge(transport=FailingTransport())
        runner = RagaliQ(
            judge=judge,
            evaluators=["relevance"],
            fail_fast=False,
        )
        test_case = RAGTestCase(
            id="err-001",
            name="Error case",
            query="test",
            context=["ctx"],
            response="resp",
        )

        result = await runner.evaluate_async(test_case)

        assert result.status == EvalStatus.ERROR
        assert "relevance" in result.details
        assert "error" in result.details["relevance"]
