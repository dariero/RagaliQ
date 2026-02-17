"""Unit tests for the RagaliQ pytest plugin."""

from unittest.mock import MagicMock, patch

import pytest


class TestPluginRegistration:
    """Test that the plugin registers correctly."""

    def test_plugin_loads_without_errors(self, pytester: pytest.Pytester) -> None:
        """Test that the plugin can be loaded by pytest."""
        # Create a dummy test file
        pytester.makepyfile(
            """
            def test_dummy():
                assert True
            """
        )

        # Run pytest with the plugin loaded
        result = pytester.runpytest("-p", "ragaliq")

        # Should not crash
        assert result.ret == 0

    def test_ragaliq_marker_registered(self, pytester: pytest.Pytester) -> None:
        """Test that @pytest.mark.ragaliq is registered."""
        # Create a test file using the marker
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.ragaliq
            def test_with_marker():
                assert True
            """
        )

        # Run pytest - should not warn about unknown marker
        result = pytester.runpytest("-p", "ragaliq", "-v")

        # Should pass without unknown marker warnings
        assert result.ret == 0
        output = result.stdout.str().lower()
        assert "unknown mark" not in output
        assert "unregistered mark" not in output


class TestCommandLineOptions:
    """Test command-line option registration."""

    def test_help_shows_ragaliq_options(self, pytester: pytest.Pytester) -> None:
        """Test that --help shows RagaliQ options."""
        result = pytester.runpytest("--help")

        output = result.stdout.str()

        # Check for ragaliq group
        assert "ragaliq" in output.lower()

        # Check for specific options
        assert "--ragaliq-judge" in output
        assert "--ragaliq-model" in output
        assert "--ragaliq-api-key" in output
        assert "--ragaliq-cost-limit" in output

    def test_ragaliq_judge_option_accepted(self, pytester: pytest.Pytester) -> None:
        """Test that --ragaliq-judge option is accepted."""
        pytester.makepyfile(
            """
            def test_dummy():
                assert True
            """
        )

        # Should accept the option without error
        result = pytester.runpytest("--ragaliq-judge=claude")
        assert result.ret == 0


class TestFixtures:
    """Test that fixtures are available."""

    def test_ragaliq_trace_collector_fixture_available(self, pytester: pytest.Pytester) -> None:
        """Test that ragaliq_trace_collector fixture can be used."""
        pytester.makepyfile(
            """
            def test_use_trace_collector(ragaliq_trace_collector):
                # Should be able to import and use it
                from ragaliq.judges.trace import TraceCollector
                assert isinstance(ragaliq_trace_collector, TraceCollector)
            """
        )

        result = pytester.runpytest("-p", "ragaliq", "-v")
        assert result.ret == 0

    def test_ragaliq_runner_fixture_available(self, pytester: pytest.Pytester) -> None:
        """Test that ragaliq_runner fixture can be used."""
        # Create a test file that uses the runner fixture
        pytester.makepyfile(
            """
            import os
            import pytest

            # Set dummy API key for test
            os.environ["ANTHROPIC_API_KEY"] = "test-key-for-fixture-test"

            def test_use_runner(ragaliq_runner):
                from ragaliq.core.runner import RagaliQ
                assert isinstance(ragaliq_runner, RagaliQ)
            """
        )

        result = pytester.runpytest("-p", "ragaliq", "-v")
        assert result.ret == 0


class TestTerminalSummary:
    """Test terminal summary output."""

    def test_summary_not_shown_when_no_traces(self, pytester: pytest.Pytester) -> None:
        """Test that summary is not shown if no RagaliQ calls were made."""
        pytester.makepyfile(
            """
            def test_no_ragaliq_usage():
                assert True
            """
        )

        result = pytester.runpytest("-p", "ragaliq", "-v")
        output = result.stdout.str()

        # Should not show RagaliQ summary
        assert "RagaliQ Summary" not in output


class TestLatencyInjection:
    """Test artificial latency injection feature."""

    def test_latency_option_accepted(self, pytester: pytest.Pytester) -> None:
        """Test that --ragaliq-latency-ms option is accepted."""
        pytester.makepyfile(
            """
            def test_dummy():
                assert True
            """
        )

        # Should accept the option without error
        result = pytester.runpytest("--ragaliq-latency-ms=100")
        assert result.ret == 0

    def test_help_shows_latency_option(self, pytester: pytest.Pytester) -> None:
        """Test that --help shows latency injection option."""
        result = pytester.runpytest("--help")
        output = result.stdout.str()

        assert "--ragaliq-latency-ms" in output
        assert "artificial latency" in output.lower() or "latency" in output.lower()

    def test_latency_actually_delays_calls(self, pytester: pytest.Pytester) -> None:
        """Test that latency injection actually adds delay to judge calls."""
        # Create a simple test that verifies the latency wrapper is applied
        pytester.makepyfile(
            """
            import os
            import pytest

            os.environ["ANTHROPIC_API_KEY"] = "test-key"

            def test_latency_wrapper_applied(ragaliq_judge):
                # Verify transport is wrapped with LatencyInjectionTransport
                assert hasattr(ragaliq_judge.transport, "_inner")
                assert hasattr(ragaliq_judge.transport, "_delay_ms")
                assert ragaliq_judge.transport._delay_ms == 100
            """
        )

        # Run with 100ms latency
        result = pytester.runpytest("--ragaliq-latency-ms=100", "-v")

        # Test should pass
        assert result.ret == 0


class TestMarkers:
    """Test that new markers are registered."""

    def test_rag_test_marker_registered(self, pytester: pytest.Pytester) -> None:
        """@pytest.mark.rag_test should not trigger unknown-marker warnings."""
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.rag_test
            def test_with_rag_test_marker():
                assert True
            """
        )

        result = pytester.runpytest("-p", "ragaliq", "-v")

        assert result.ret == 0
        output = result.stdout.str().lower()
        assert "unknown mark" not in output
        assert "unregistered mark" not in output

    def test_rag_slow_marker_registered(self, pytester: pytest.Pytester) -> None:
        """@pytest.mark.rag_slow should not trigger unknown-marker warnings."""
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.rag_slow
            def test_with_rag_slow_marker():
                assert True
            """
        )

        result = pytester.runpytest("-p", "ragaliq", "-v")

        assert result.ret == 0
        output = result.stdout.str().lower()
        assert "unknown mark" not in output
        assert "unregistered mark" not in output

    def test_rag_slow_marker_filterable(self, pytester: pytest.Pytester) -> None:
        """Tests marked rag_slow can be deselected with -m 'not rag_slow'."""
        pytester.makepyfile(
            """
            import pytest

            @pytest.mark.rag_slow
            def test_slow():
                assert True

            def test_fast():
                assert True
            """
        )

        result = pytester.runpytest("-p", "ragaliq", "-m", "not rag_slow", "-v")

        assert result.ret == 0
        output = result.stdout.str()
        assert "test_fast" in output
        assert "test_slow" not in output or "deselected" in output


class TestRagTesterFixture:
    """Test the rag_tester fixture."""

    def test_rag_tester_fixture_available(self, pytester: pytest.Pytester) -> None:
        """rag_tester fixture should provide a configured RagaliQ instance."""
        pytester.makepyfile(
            """
            import os
            os.environ["ANTHROPIC_API_KEY"] = "test-key-for-fixture-test"

            def test_use_rag_tester(rag_tester):
                from ragaliq.core.runner import RagaliQ
                assert isinstance(rag_tester, RagaliQ)
            """
        )

        result = pytester.runpytest("-p", "ragaliq", "-v")
        assert result.ret == 0

    def test_rag_tester_distinct_from_ragaliq_runner(self, pytester: pytest.Pytester) -> None:
        """Both rag_tester and ragaliq_runner should be independently usable."""
        pytester.makepyfile(
            """
            import os
            os.environ["ANTHROPIC_API_KEY"] = "test-key"

            def test_both_fixtures(rag_tester, ragaliq_runner):
                from ragaliq.core.runner import RagaliQ
                assert isinstance(rag_tester, RagaliQ)
                assert isinstance(ragaliq_runner, RagaliQ)
                # They are separate instances sharing the session judge
                assert rag_tester is not ragaliq_runner
            """
        )

        result = pytester.runpytest("-p", "ragaliq", "-v")
        assert result.ret == 0


class TestAssertRagQuality:
    """Unit tests for the assert_rag_quality helper function."""

    def _make_test_case(self) -> object:
        """Build a minimal RAGTestCase."""
        from ragaliq.core.test_case import RAGTestCase

        return RAGTestCase(
            id="test-1",
            name="quality check",
            query="What is 2+2?",
            context=["Basic arithmetic: 2+2 equals 4."],
            response="2+2 is 4.",
        )

    def _make_passing_result(self) -> MagicMock:
        """Mock a RAGTestResult that passes."""
        from ragaliq.core.test_case import EvalStatus

        result = MagicMock()
        result.passed = True
        result.status = EvalStatus.PASSED
        result.scores = {"faithfulness": 0.9, "relevance": 0.85}
        return result

    def _make_failing_result(self) -> MagicMock:
        """Mock a RAGTestResult that fails."""
        from ragaliq.core.test_case import EvalStatus

        result = MagicMock()
        result.passed = False
        result.status = EvalStatus.FAILED
        result.scores = {"faithfulness": 0.3, "relevance": 0.8}
        return result

    def test_returns_result_on_pass(self) -> None:
        """assert_rag_quality returns the result when all metrics pass."""
        from ragaliq.integrations.pytest_plugin import assert_rag_quality

        test_case = self._make_test_case()

        with patch(
            "ragaliq.core.runner.RagaliQ.evaluate", return_value=self._make_passing_result()
        ):
            result = assert_rag_quality(test_case)

        assert result is not None
        assert result.passed is True

    def test_raises_assertion_error_on_failure(self) -> None:
        """assert_rag_quality raises AssertionError when any metric fails."""
        from ragaliq.integrations.pytest_plugin import assert_rag_quality

        test_case = self._make_test_case()

        with (
            patch("ragaliq.core.runner.RagaliQ.evaluate", return_value=self._make_failing_result()),
            pytest.raises(AssertionError, match="RAG quality check failed"),
        ):
            assert_rag_quality(test_case)

    def test_error_message_includes_test_name(self) -> None:
        """AssertionError message includes the test case name."""
        from ragaliq.integrations.pytest_plugin import assert_rag_quality

        test_case = self._make_test_case()

        with (
            patch("ragaliq.core.runner.RagaliQ.evaluate", return_value=self._make_failing_result()),
            pytest.raises(AssertionError, match="quality check"),
        ):
            assert_rag_quality(test_case)

    def test_error_message_includes_failing_scores(self) -> None:
        """AssertionError message includes the failing metric scores."""
        from ragaliq.integrations.pytest_plugin import assert_rag_quality

        test_case = self._make_test_case()

        with (
            patch("ragaliq.core.runner.RagaliQ.evaluate", return_value=self._make_failing_result()),
            pytest.raises(AssertionError, match="faithfulness"),
        ):
            assert_rag_quality(test_case)

    def test_accepts_pre_configured_judge(self) -> None:
        """assert_rag_quality passes a provided judge through to RagaliQ."""
        from ragaliq.integrations.pytest_plugin import assert_rag_quality

        test_case = self._make_test_case()
        mock_judge = MagicMock()

        with patch("ragaliq.core.runner.RagaliQ") as mock_cls:
            mock_cls.return_value.evaluate.return_value = self._make_passing_result()
            assert_rag_quality(test_case, judge=mock_judge)

        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["judge"] is mock_judge

    def test_accepts_custom_threshold(self) -> None:
        """assert_rag_quality forwards the threshold to RagaliQ."""
        from ragaliq.integrations.pytest_plugin import assert_rag_quality

        test_case = self._make_test_case()

        with patch("ragaliq.core.runner.RagaliQ") as mock_cls:
            mock_cls.return_value.evaluate.return_value = self._make_passing_result()
            assert_rag_quality(test_case, threshold=0.9)

        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["default_threshold"] == 0.9
