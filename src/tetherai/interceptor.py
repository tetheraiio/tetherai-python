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

        self._originals: dict[str, Callable[..., Any]] = {}
        self._active = False

    def activate(self) -> None:
        if self._active:
            raise TetherError("Interceptor is already active")

        self._patch_litellm()
        self._patch_openai()
        self._patch_crewai()

        self._active = True

    def _patch_litellm(self) -> None:
        try:
            import litellm
        except ImportError:
            return

        methods_to_patch = [
            "completion",
            "acompletion",
            "chat.completions.create",
            "completion_with_functions",
            "acompletion_with_functions",
        ]

        for method in methods_to_patch:
            parts = method.split(".")
            obj = litellm
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    break
            else:
                self._originals[f"litellm.{method}"] = obj

                if method == "chat.completions.create":
                    key = f"litellm.{method}"
                    litellm.chat.completions.create = self._make_patcher(key, obj)
                elif method == "completion_with_functions":
                    key = f"litellm.{method}"
                    litellm.completion_with_functions = self._make_patcher(key, obj)
                elif method == "acompletion_with_functions":
                    key = f"litellm.{method}"
                    litellm.acompletion_with_functions = self._make_patcher(key, obj)
                else:
                    key = f"litellm.{method}"
                    setattr(litellm, method, self._make_patcher(key, obj))

    def _patch_openai(self) -> None:
        try:
            import openai
        except ImportError:
            return

        # Patch sync OpenAI client
        if (
            hasattr(openai, "OpenAI")
            and hasattr(openai.OpenAI, "chat")
            and hasattr(openai.OpenAI.chat, "completions")
            and hasattr(openai.OpenAI.chat.completions, "create")
        ):
            original = openai.OpenAI.chat.completions.create
            self._originals["openai.chat.completions.create"] = original
            openai.OpenAI.chat.completions.create = self._make_patcher(
                "openai.chat.completions.create", original
            )

        # Patch async OpenAI client
        if (
            hasattr(openai, "AsyncOpenAI")
            and hasattr(openai.AsyncOpenAI, "chat")
            and hasattr(openai.AsyncOpenAI.chat, "completions")
            and hasattr(openai.AsyncOpenAI.chat.completions, "create")
        ):
            original = openai.AsyncOpenAI.chat.completions.create
            self._originals["openai.async.chat.completions.create"] = original
            openai.AsyncOpenAI.chat.completions.create = self._make_async_patcher(
                "openai.async.chat.completions.create", original
            )

    def _patch_crewai(self) -> None:
        try:
            from crewai.llms.providers.openai.completion import OpenAICompletion
        except ImportError:
            return

        interceptor = self

        if hasattr(OpenAICompletion, "_call_completions"):
            original_call = OpenAICompletion._call_completions
            self._originals["crewai._call_completions"] = original_call

            def patched(self: Any, *args: Any, **kwargs: Any) -> Any:
                return interceptor._intercept_crewai_call(original_call, self, *args, **kwargs)

            OpenAICompletion._call_completions = patched

        if hasattr(OpenAICompletion, "_acall_completions"):
            original_acall = OpenAICompletion._acall_completions
            self._originals["crewai._acall_completions"] = original_acall

            async def patched(self: Any, *args: Any, **kwargs: Any) -> Any:
                return await interceptor._intercept_crewai_call_async(
                    original_acall, self, *args, **kwargs
                )

            OpenAICompletion._acall_completions = patched

    def _intercept_crewai_call(
        self, original: Callable[..., Any], self_obj: Any, *args: Any, **kwargs: Any
    ) -> Any:
        import sys

        model = kwargs.get("model", getattr(self_obj, "model", None) or "gpt-4o-mini")
        messages = kwargs.get("messages", args[0] if args else [])

        start_time = time.time()

        try:
            input_tokens = self.token_counter.count_messages(messages, model)
        except Exception:
            input_tokens = 0

        estimated_output_tokens = input_tokens * 4

        try:
            estimated_input_cost = self.pricing.get_input_cost(model) * input_tokens / 1000
            estimated_output_cost = (
                self.pricing.get_output_cost(model) * estimated_output_tokens / 1000
            )
            estimated_total_cost = estimated_input_cost + estimated_output_cost
        except Exception:
            estimated_total_cost = 0

        try:
            self.budget_tracker.pre_check(estimated_total_cost, model)
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
            response = original(self_obj, *args, **kwargs)
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
                usage_summary = getattr(self_obj, "get_token_usage_summary", None)
                if usage_summary:
                    token_usage = usage_summary()
                    total_usage = (
                        token_usage
                        if isinstance(token_usage, dict)
                        else token_usage.model_dump()
                        if hasattr(token_usage, "model_dump")
                        else {}
                    )
                    actual_input_tokens = total_usage.get("prompt_tokens", input_tokens)
                    output_tokens = total_usage.get("completion_tokens", 0)
                else:
                    output_tokens = 0
                    actual_input_tokens = input_tokens

            cost_usd = self.pricing.estimate_call_cost(model, actual_input_tokens, output_tokens)

            span.output_tokens = output_tokens
            span.input_tokens = actual_input_tokens
            span.cost_usd = cost_usd
            span.duration_ms = duration_ms
            span.status = "ok"

            self.budget_tracker.record_call(
                actual_input_tokens,
                output_tokens,
                model,
                cost_usd,
                duration_ms,
            )
        except BudgetExceededError:
            raise
        except Exception:
            pass

        return response

    async def _intercept_crewai_call_async(
        self, original: Callable[..., Any], self_obj: Any, *args: Any, **kwargs: Any
    ) -> Any:
        model = kwargs.get("model", getattr(self_obj, "model", None) or "gpt-4o-mini")
        messages = kwargs.get("messages", args[0] if args else [])

        start_time = time.time()

        try:
            input_tokens = self.token_counter.count_messages(messages, model)
        except Exception:
            input_tokens = 0

        estimated_output_tokens = input_tokens * 4

        try:
            estimated_input_cost = self.pricing.get_input_cost(model) * input_tokens / 1000
            estimated_output_cost = (
                self.pricing.get_output_cost(model) * estimated_output_tokens / 1000
            )
            estimated_total_cost = estimated_input_cost + estimated_output_cost
        except Exception:
            estimated_total_cost = 0

        try:
            self.budget_tracker.pre_check(estimated_total_cost, model)
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
            response = await original(self_obj, *args, **kwargs)
        except Exception:
            span.status = "error"
            span.duration_ms = (time.time() - start_time) * 1000
            raise

        duration_ms = (time.time() - start_time) * 1000

        try:
            output_tokens = getattr(response, "usage", None)
            if output_tokens:
                output_tokens = getattr(output_tokens, "completion_tokens", 0)
                actual_input_tokens = getattr(output_tokens, "prompt_tokens", input_tokens)
            else:
                output_tokens = 0
                actual_input_tokens = input_tokens

            cost_usd = self.pricing.estimate_call_cost(model, actual_input_tokens, output_tokens)

            span.output_tokens = output_tokens
            span.input_tokens = actual_input_tokens
            span.cost_usd = cost_usd
            span.duration_ms = duration_ms
            span.status = "ok"

            self.budget_tracker.record_call(
                actual_input_tokens,
                output_tokens,
                model,
                cost_usd,
                duration_ms,
            )
        except BudgetExceededError:
            raise
        except Exception:
            pass

        return response

    def _make_patcher(self, method: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def patched(*args: Any, **kwargs: Any) -> Any:
            return self._intercept_call(original, *args, **kwargs)

        return patched

    def _make_async_patcher(self, method: str, original: Callable[..., Any]) -> Callable[..., Any]:
        async def patched(*args: Any, **kwargs: Any) -> Any:
            return await self._intercept_call_async(original, *args, **kwargs)

        return patched

    def deactivate(self) -> None:
        if not self._active:
            return

        try:
            import litellm
        except ImportError:
            litellm = None

        for method, original in self._originals.items():
            if method.startswith("litellm."):
                name = method[8:]
                if name == "chat.completions.create" and litellm:
                    litellm.chat.completions.create = original
                elif name == "completion_with_functions" and litellm:
                    litellm.completion_with_functions = original
                elif name == "acompletion_with_functions" and litellm:
                    litellm.acompletion_with_functions = original
                elif litellm:
                    setattr(litellm, name, original)
            elif method == "openai.chat.completions.create":
                try:
                    import openai

                    openai.OpenAI.chat.completions.create = original
                except ImportError:
                    pass
            elif method == "openai.async.chat.completions.create":
                try:
                    import openai

                    openai.AsyncOpenAI.chat.completions.create = original
                except ImportError:
                    pass

        self._originals.clear()
        self._active = False

    def __enter__(self) -> "LLMInterceptor":
        self.activate()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.deactivate()

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

        estimated_output_tokens = input_tokens * 4

        try:
            estimated_input_cost = self.pricing.get_input_cost(model) * input_tokens / 1000
            estimated_output_cost = (
                self.pricing.get_output_cost(model) * estimated_output_tokens / 1000
            )
            estimated_total_cost = estimated_input_cost + estimated_output_cost
        except Exception:
            estimated_total_cost = 0

        try:
            self.budget_tracker.pre_check(estimated_total_cost, model)
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

            cost_usd = self.pricing.estimate_call_cost(model, actual_input_tokens, output_tokens)

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

        estimated_output_tokens = input_tokens * 4

        try:
            estimated_input_cost = self.pricing.get_input_cost(model) * input_tokens / 1000
            estimated_output_cost = (
                self.pricing.get_output_cost(model) * estimated_output_tokens / 1000
            )
            estimated_total_cost = estimated_input_cost + estimated_output_cost
        except Exception:
            estimated_total_cost = 0

        try:
            self.budget_tracker.pre_check(estimated_total_cost, model)
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

            cost_usd = self.pricing.estimate_call_cost(model, actual_input_tokens, output_tokens)

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
        self.budget_tracker.pre_check(cost_usd, model)
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
