import pytest
import threading
import asyncio
from unittest.mock import Mock, patch

from tetherai.circuit_breaker import enforce_budget
from tetherai.budget import BudgetTracker
from tetherai.token_counter import TokenCounter
from tetherai.pricing import PricingRegistry
from tetherai.trace import TraceCollector
from tetherai.interceptor import LLMInterceptor
from tetherai.exceptions import BudgetExceededError, TurnLimitError


def make_mock_llm_call(cost: float = 0.01):
    from unittest.mock import Mock
    response = Mock()
    response.choices = [Mock(message=Mock(content="test response"))]
    response.usage = Mock(prompt_tokens=10, completion_tokens=5)
    return response


class TestEnforceBudgetDecorator:
    def test_basic_decorator_syntax(self):
        @enforce_budget(max_usd=2.0)
        def my_function():
            return "hello"

        assert my_function() == "hello"

    def test_function_under_budget_returns_normally(self):
        @enforce_budget(max_usd=10.0)
        def cheap_function():
            return "result"

        result = cheap_function()
        assert result == "result"

    def test_decorator_preserves_function_metadata(self):
        @enforce_budget(max_usd=2.0)
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_decorator_works_on_methods(self):
        class MyClass:
            @enforce_budget(max_usd=2.0)
            def my_method(self):
                return "method result"

        obj = MyClass()
        assert obj.my_method() == "method result"

    def test_decorator_works_on_async_functions(self):
        @enforce_budget(max_usd=2.0)
        async def async_function():
            await asyncio.sleep(0.001)
            return "async result"

        result = asyncio.run(async_function())
        assert result == "async result"

    def test_interceptor_cleaned_up_on_exception(self):
        @enforce_budget(max_usd=2.0)
        def function_that_raises():
            raise ValueError("Some other error")

        with pytest.raises(ValueError):
            function_that_raises()


class TestEnforceBudgetConcurrent:
    def test_concurrent_decorated_functions_isolated(self):
        results = []

        def make_func(budget):
            @enforce_budget(max_usd=budget)
            def func():
                return budget
            return func

        funcs = [make_func(i) for i in [1.0, 2.0, 3.0]]

        def run_func(f):
            results.append(f())

        threads = [threading.Thread(target=run_func, args=(f,)) for f in funcs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) == [1.0, 2.0, 3.0]
