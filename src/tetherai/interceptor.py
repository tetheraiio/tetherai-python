import time
from collections.abc import Callable
from typing import Any

from tetherai.budget import BudgetTracker
from tetherai.exceptions import BudgetExceededError, TetherError
from tetherai.pricing import PricingRegistry
from tetherai.token_counter import TokenCounter
from tetherai.trace import Span, TraceCollector


class LLMInterceptor:
    def __init__(
        self,
        budget_tracker: BudgetTracker,
        token_counter: TokenCounter,
        pricing: PricingRegistry,
        trace_collector: TraceCollector,
    ):
        self.budget_tracker = budget_tracker
        self.token_counter = token_counter
        self.pricing = pricing
        self.trace_collector = trace_collector

        self._original_completion: Callable[..., Any] | None = None
        self._original_acompletion: Callable[..., Any] | None = None
        self._active = False

    def activate(self) -> None:
        if self._active:
            raise TetherError("Interceptor is already active")

        try:
            import litellm
        except ImportError:
            return

        self._original_completion = litellm.completion
        self._original_acompletion = litellm.acompletion

        litellm.completion = self._patched_completion
        litellm.acompletion = self._patched_acompletion
        self._active = True

    def deactivate(self) -> None:
        if not self._active:
            return

        try:
            import litellm
        except ImportError:
            self._active = False
            return

        if self._original_completion:
            litellm.completion = self._original_completion
        if self._original_acompletion:
            litellm.acompletion = self._original_acompletion

        self._active = False

    def __enter__(self) -> "LLMInterceptor":
        self.activate()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.deactivate()

    def _patched_completion(self, *args: Any, **kwargs: Any) -> Any:
        return self._intercept_call(self._original_completion, *args, **kwargs)

    async def _patched_acompletion(self, *args: Any, **kwargs: Any) -> Any:
        return await self._intercept_call_async(self._original_acompletion, *args, **kwargs)

    def _intercept_call(
        self, original_fn: Callable[..., Any] | None, *args: Any, **kwargs: Any
    ) -> Any:
        if original_fn is None:
            raise TetherError("Interceptor not properly activated")
        model = kwargs.get("model", args[0] if args else "unknown")
        messages = kwargs.get("messages", [])

        start_time = time.time()

        try:
            input_tokens = self.token_counter.count_messages(messages, model)
        except Exception:
            input_tokens = 0

        try:
            estimated_input_cost = (
                self.pricing.get_input_cost(model) * input_tokens
            )
        except Exception:
            estimated_input_cost = 0

        try:
            self.budget_tracker.pre_check(estimated_input_cost)
        except BudgetExceededError:
            raise

        span = Span(
            run_id=self.budget_tracker.run_id,
            span_type="llm_call",
            model=model,
            input_tokens=input_tokens,
            input_preview=messages[0].get("content", "")[:200] if messages else None,
        )
        self.trace_collector.add_span(span)

        try:
            response = original_fn(*args, **kwargs)
        except Exception:
            span.status = "error"
            span.duration_ms = (time.time() - start_time) * 1000
            raise

        duration_ms = (time.time() - start_time) * 1000

        try:
            usage = getattr(response, "usage", None)
            if usage:
                output_tokens = getattr(usage, "completion_tokens", 0)
                actual_input_tokens = getattr(usage, "prompt_tokens", input_tokens)
            else:
                output_tokens = 0
                actual_input_tokens = input_tokens

            cost_usd = self.pricing.estimate_call_cost(
                model, actual_input_tokens, output_tokens
            )

            span.output_tokens = output_tokens
            span.input_tokens = actual_input_tokens
            span.cost_usd = cost_usd
            span.duration_ms = duration_ms
            span.status = "ok"

            try:
                content = response.choices[0].message.content if response.choices else ""
                span.output_preview = content[:200] if content else None
            except Exception:
                pass

            self.budget_tracker.record_call(
                actual_input_tokens,
                output_tokens,
                model,
                cost_usd,
                duration_ms,
            )

        except Exception:
            pass

        return response

    async def _intercept_call_async(
        self, original_fn: Callable[..., Any] | None, *args: Any, **kwargs: Any
    ) -> Any:
        if original_fn is None:
            raise TetherError("Interceptor not properly activated")
        model = kwargs.get("model", args[0] if args else "unknown")
        messages = kwargs.get("messages", [])

        start_time = time.time()

        try:
            input_tokens = self.token_counter.count_messages(messages, model)
        except Exception:
            input_tokens = 0

        try:
            estimated_input_cost = (
                self.pricing.get_input_cost(model) * input_tokens
            )
        except Exception:
            estimated_input_cost = 0

        try:
            self.budget_tracker.pre_check(estimated_input_cost)
        except BudgetExceededError:
            raise

        span = Span(
            run_id=self.budget_tracker.run_id,
            span_type="llm_call",
            model=model,
            input_tokens=input_tokens,
            input_preview=messages[0].get("content", "")[:200] if messages else None,
        )
        self.trace_collector.add_span(span)

        try:
            response = await original_fn(*args, **kwargs)
        except Exception:
            span.status = "error"
            span.duration_ms = (time.time() - start_time) * 1000
            raise

        duration_ms = (time.time() - start_time) * 1000

        try:
            usage = getattr(response, "usage", None)
            if usage:
                output_tokens = getattr(usage, "completion_tokens", 0)
                actual_input_tokens = getattr(usage, "prompt_tokens", input_tokens)
            else:
                output_tokens = 0
                actual_input_tokens = input_tokens

            cost_usd = self.pricing.estimate_call_cost(
                model, actual_input_tokens, output_tokens
            )

            span.output_tokens = output_tokens
            span.input_tokens = actual_input_tokens
            span.cost_usd = cost_usd
            span.duration_ms = duration_ms
            span.status = "ok"

            try:
                content = response.choices[0].message.content if response.choices else ""
                span.output_preview = content[:200] if content else None
            except Exception:
                pass

            self.budget_tracker.record_call(
                actual_input_tokens,
                output_tokens,
                model,
                cost_usd,
                duration_ms,
            )

        except Exception:
            pass

        return response

    def track_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        cost_usd = self.pricing.estimate_call_cost(model, input_tokens, output_tokens)
        self.budget_tracker.pre_check(
            self.pricing.get_input_cost(model) * input_tokens
        )
        self.budget_tracker.record_call(
            input_tokens,
            output_tokens,
            model,
            cost_usd,
            0.0,
        )

        span = Span(
            run_id=self.budget_tracker.run_id,
            span_type="llm_call",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            status="ok",
        )
        self.trace_collector.add_span(span)
