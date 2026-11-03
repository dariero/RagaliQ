"""A judge used across event loops (#179).

The sync API (`RagaliQ.evaluate()`, `evaluate_batch()`, `assert_rag_quality()`,
the CLI) runs each call in a new event loop with asyncio.run(), and the plugin
runs each async test in its own loop, while a judge outlives both. The tests
with a server run real SDK clients against a local HTTP server on 127.0.0.1;
nothing leaves the machine.
"""

import asyncio
import gc
import json
import threading
import warnings
import weakref
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ragaliq import RagaliQ, RAGTestCase
from ragaliq.integrations.pytest_plugin import assert_rag_quality
from ragaliq.judges import ClaudeJudge, ClaudeTransport, JudgeConfig, TransportResponse
from ragaliq.judges.base_judge import BaseJudge

_VERDICT = json.dumps({"score": 0.9, "reasoning": "ok"})
_BODY = json.dumps(
    {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [{"type": "text", "text": _VERDICT}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
).encode()


class _MessagesServer:
    """Answers every POST like the Messages API and counts the requests it reads."""

    def __init__(self) -> None:
        self.requests = 0
        lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            # Keep-alive, so the SDK client pools its connections as against the API.
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("content-length", 0)))
                with lock:
                    server.requests += 1
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(_BODY)))
                self.end_headers()
                self.wfile.write(_BODY)

            def log_message(self, *args: object) -> None:
                pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._httpd.server_address[1]}"
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def messages_server(monkeypatch: pytest.MonkeyPatch) -> Iterator[_MessagesServer]:
    server = _MessagesServer()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", server.url)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    yield server
    server.close()


def _case(i: int) -> RAGTestCase:
    return RAGTestCase(
        id=f"t{i}",
        name=f"case {i}",
        query="What is the capital of France?",
        context=["France's capital city is Paris."],
        response="The capital of France is Paris.",
    )


def _resource_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [str(w.message) for w in caught if issubclass(w.category, ResourceWarning)]


def test_sync_calls_on_one_runner_send_each_request_once(
    messages_server: _MessagesServer,
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runner = RagaliQ(evaluators=["relevance"])
        results = [runner.evaluate(_case(i)) for i in range(3)]
        del runner
        gc.collect()

    assert [r.passed for r in results] == [True, True, True]
    # One client shared by three loops sent 5 requests for these 3: each later
    # call's first request went out on a connection pooled in a closed loop.
    assert messages_server.requests == 3
    assert _resource_warnings(caught) == []


def test_assert_rag_quality_without_a_judge_closes_its_client(
    messages_server: _MessagesServer,
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert_rag_quality(_case(0), evaluators=["relevance"])
        gc.collect()

    assert messages_server.requests == 1
    assert _resource_warnings(caught) == []


def test_plugin_session_judge_sends_each_request_once(
    pytester: pytest.Pytester, messages_server: _MessagesServer
) -> None:
    pytester.makepyfile(
        """
        import pytest

        from ragaliq import RagaliQ, RAGTestCase

        CASE = RAGTestCase(
            id="t",
            name="case",
            query="What is the capital of France?",
            context=["France's capital city is Paris."],
            response="The capital of France is Paris.",
        )


        def _runner(judge):
            return RagaliQ(judge=judge, evaluators=["relevance"])


        def test_sync_1(ragaliq_judge):
            assert _runner(ragaliq_judge).evaluate(CASE).passed


        def test_sync_2(ragaliq_judge):
            assert _runner(ragaliq_judge).evaluate(CASE).passed


        @pytest.mark.asyncio
        async def test_async_1(ragaliq_judge):
            assert (await _runner(ragaliq_judge).evaluate_async(CASE)).passed


        @pytest.mark.asyncio
        async def test_async_2(ragaliq_judge):
            assert (await _runner(ragaliq_judge).evaluate_async(CASE)).passed
        """
    )

    result = pytester.runpytest("-p", "ragaliq")

    result.assert_outcomes(passed=4)
    # Each test runs in its own event loop and uses the session-scoped judge.
    assert messages_server.requests == 4


def test_judge_keeps_no_finished_event_loop_alive(messages_server: _MessagesServer) -> None:
    runner = RagaliQ(judge=ClaudeJudge(max_concurrency=1), evaluators=["relevance"])
    loops: list[weakref.ref[asyncio.AbstractEventLoop]] = []

    async def batch_of_three() -> list[bool]:
        loops.append(weakref.ref(asyncio.get_running_loop()))
        return [r.passed for r in await runner.evaluate_batch_async([_case(i) for i in range(3)])]

    # In each loop two of the three judge calls wait for the semaphore, so both the
    # client and the semaphore are bound to that loop.
    for _ in range(4):
        assert asyncio.run(batch_of_three()) == [True, True, True]
    gc.collect()

    # The semaphore of the newest closed loop is dropped when the next loop binds.
    assert [ref() is None for ref in loops[:-1]] == [True, True, True]
    assert messages_server.requests == 12


class _SlowTransport:
    async def send(self, **kwargs: object) -> TransportResponse:
        await asyncio.sleep(0.01)
        return TransportResponse(
            text=_VERDICT, input_tokens=1, output_tokens=1, model=str(kwargs["model"])
        )


def test_concurrency_limit_works_in_each_sync_call() -> None:
    judge = BaseJudge(transport=_SlowTransport(), config=JudgeConfig(), max_concurrency=1)

    async def three_at_once() -> list[float]:
        results = await asyncio.gather(
            *(judge.evaluate_relevance(query="q?", response="r") for _ in range(3))
        )
        return [r.score for r in results]

    # Two of the three calls wait for the semaphore in each loop. A single
    # semaphore bound to the first loop raised in the second.
    assert asyncio.run(three_at_once()) == [0.9, 0.9, 0.9]
    assert asyncio.run(three_at_once()) == [0.9, 0.9, 0.9]


def test_negative_max_concurrency_raises_when_the_judge_is_built() -> None:
    # Each loop's semaphore is made on its first use, so the value is checked here.
    with pytest.raises(ValueError, match="max_concurrency"):
        BaseJudge(transport=_SlowTransport(), config=JudgeConfig(), max_concurrency=-1)


@pytest.mark.asyncio
async def test_send_after_aclose_raises(messages_server: _MessagesServer) -> None:
    transport = ClaudeTransport(api_key="test-key")
    await transport.aclose()

    with pytest.raises(RuntimeError, match="ClaudeTransport is closed"):
        await transport.send("system", "user")
    assert messages_server.requests == 0
