from unittest.mock import Mock, patch

import pytest

from tetherai.budget import BudgetTracker
from tetherai.exceptions import BudgetExceededError, TetherError
from tetherai.interceptor import LLMInterceptor
from tetherai.pricing import PricingRegistry
from tetherai.token_counter import TokenCounter
from tetherai.trace import TraceCollector


class MockResponse:
    def __init__(self, content="test", prompt_tokens=10, completion_tokens=5):
        self.content = content
        self.choices = [Mock(message=Mock(content=content))]
        self.usage = Mock(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class TestLLMInterceptor:
    def test_activate_patches_litellm(self):
        try:
            import litellm
        except ImportError:
            pytest.skip("litellm not installed")

        tracker = BudgetTracker(run_id="test", max_usd=10.0)
        counter = TokenCounter(backend="tiktoken")
        pricing = PricingRegistry()
        collector = TraceCollector()
        collector.start_trace("test")

        interceptor = LLMInterceptor(tracker, counter, pricing, collector)

        original = litellm.completion
        interceptor.activate()
        assert litellm.completion is not original

        interceptor.deactivate()

    def test_deactivate_restores_litellm(self):
        try:
            import litellm
        except ImportError:
            pytest.skip("litellm not installed")

        tracker = BudgetTracker(run_id="test", max_usd=10.0)
        counter = TokenCounter(backend="tiktoken")
        pricing = PricingRegistry()
        collector = TraceCollector()
        collector.start_trace("test")

        original = litellm.completion

        interceptor = LLMInterceptor(tracker, counter, pricing, collector)
        interceptor.activate()
        interceptor.deactivate()

        assert litellm.completion is original

    def test_context_manager_cleanup(self):
        try:
            import litellm
        except ImportError:
            pytest.skip("litellm not installed")

        tracker = BudgetTracker(run_id="test", max_usd=10.0)
        counter = TokenCounter(backend="tiktoken")
        pricing = PricingRegistry()
        collector = TraceCollector()
        collector.start_trace("test")

        original = litellm.completion

        with LLMInterceptor(tracker, counter, pricing, collector):
            pass

        assert litellm.completion is original

    def test_pre_check_runs_before_network_call(self):
        try:
            import litellm
        except ImportError:
            pytest.skip("litellm not installed")

        tracker = BudgetTracker(run_id="test", max_usd=0.0001)
        counter = TokenCounter(backend="tiktoken")
        pricing = PricingRegistry()
        collector = TraceCollector()
        collector.start_trace("test")

        mock_completion = Mock(return_value=MockResponse())

        with patch.object(litellm, "completion", mock_completion):
            interceptor = LLMInterceptor(tracker, counter, pricing, collector)
            interceptor.activate()

            with pytest.raises(BudgetExceededError):
                litellm.completion(model="gpt-4o", messages=[{"role": "user", "content": "test"}])

            mock_completion.assert_not_called()

    def test_actual_usage_recorded_after_call(self):
        try:
            import litellm
        except ImportError:
            pytest.skip("litellm not installed")

        tracker = BudgetTracker(run_id="test", max_usd=10.0)
        counter = TokenCounter(backend="tiktoken")
        pricing = PricingRegistry()
        collector = TraceCollector()
        collector.start_trace("test")

        mock_response = MockResponse(prompt_tokens=100, completion_tokens=50)

        with patch.object(litellm, "completion", return_value=mock_response):
            interceptor = LLMInterceptor(tracker, counter, pricing, collector)
            interceptor.activate()

            litellm.completion(
                model="gpt-4o", messages=[{"role": "user", "content": "test message"}]
            )

        assert tracker.spent_usd > 0

    def test_response_object_unchanged(self):
        try:
            import litellm
        except ImportError:
            pytest.skip("litellm not installed")

        tracker = BudgetTracker(run_id="test", max_usd=10.0)
        counter = TokenCounter(backend="tiktoken")
        pricing = PricingRegistry()
        collector = TraceCollector()
        collector.start_trace("test")

        mock_response = MockResponse(content="Hello!")

        with patch.object(litellm, "completion", return_value=mock_response):
            interceptor = LLMInterceptor(tracker, counter, pricing, collector)
            interceptor.activate()

            result = litellm.completion(
                model="gpt-4o", messages=[{"role": "user", "content": "test"}]
            )

        assert result.choices[0].message.content == "Hello!"

    def test_trace_span_emitted(self):
        try:
            import litellm
        except ImportError:
            pytest.skip("litellm not installed")

        tracker = BudgetTracker(run_id="test", max_usd=10.0)
        counter = TokenCounter(backend="tiktoken")
        pricing = PricingRegistry()
        collector = TraceCollector()
        collector.start_trace("test")

        mock_response = MockResponse()

        with patch.object(litellm, "completion", return_value=mock_response):
            interceptor = LLMInterceptor(tracker, counter, pricing, collector)
            interceptor.activate()

            litellm.completion(model="gpt-4o", messages=[{"role": "user", "content": "test"}])

        trace = collector.get_current_trace()
        assert trace is not None
        assert len(trace.spans) == 1
        assert trace.spans[0].model == "gpt-4o"

    def test_no_litellm_manual_tracking(self):
        tracker = BudgetTracker(run_id="test", max_usd=10.0)
        counter = TokenCounter(backend="tiktoken")
        pricing = PricingRegistry()
        collector = TraceCollector()
        collector.start_trace("test")

        interceptor = LLMInterceptor(tracker, counter, pricing, collector)
        interceptor.track_call("gpt-4o", 100, 50)

        assert tracker.spent_usd > 0

    def test_exception_from_llm_still_recorded(self):
        try:
            import litellm
        except ImportError:
            pytest.skip("litellm not installed")

        tracker = BudgetTracker(run_id="test", max_usd=10.0)
        counter = TokenCounter(backend="tiktoken")
        pricing = PricingRegistry()
        collector = TraceCollector()
        collector.start_trace("test")

        with patch.object(litellm, "completion", side_effect=Exception("Rate limited")):
            interceptor = LLMInterceptor(tracker, counter, pricing, collector)
            interceptor.activate()

            with pytest.raises(Exception):  # noqa: B017
                litellm.completion(model="gpt-4o", messages=[{"role": "user", "content": "test"}])

        trace = collector.get_current_trace()
        assert trace.spans[-1].status == "error"


class TestLLMInterceptorErrors:
    def test_double_activate_raises(self):
        try:
            import litellm  # noqa: F401
        except ImportError:
            pytest.skip("litellm not installed")

        tracker = BudgetTracker(run_id="test", max_usd=10.0)
        counter = TokenCounter(backend="tiktoken")
        pricing = PricingRegistry()
        collector = TraceCollector()
        collector.start_trace("test")

        interceptor = LLMInterceptor(tracker, counter, pricing, collector)
        interceptor.activate()

        with pytest.raises(TetherError, match="already active"):
            interceptor.activate()

        interceptor.deactivate()
