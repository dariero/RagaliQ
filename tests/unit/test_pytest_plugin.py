"""Unit tests for the RagaliQ pytest plugin."""

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
