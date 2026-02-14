import asyncio
import functools
import uuid
from collections.abc import Callable
from typing import Any, TypeVar

from tetherai.budget import BudgetTracker
from tetherai.config import TetherConfig
from tetherai.exceptions import BudgetExceededError
from tetherai.exporter import get_exporter
from tetherai.interceptor import LLMInterceptor
from tetherai.pricing import PricingRegistry
from tetherai.token_counter import TokenCounter
from tetherai.trace import TraceCollector

F = TypeVar("F", bound=Callable[..., Any])


def enforce_budget(
    max_usd: float,
    max_turns: int | None = None,
    on_exceed: str = "raise",
    trace_export: str | None = None,
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return _run_with_budget(
                func, max_usd, max_turns, on_exceed, trace_export, *args, **kwargs
            )

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            return await _run_with_budget_async(
                func, max_usd, max_turns, on_exceed, trace_export, *args, **kwargs
            )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return wrapper  # type: ignore[return-value]

    return decorator


def _run_with_budget(
    func: Callable[..., Any],
    max_usd: float,
    max_turns: int | None,
    on_exceed: str,
    trace_export: str | None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    config = TetherConfig()

    if trace_export is None:
        trace_export = config.trace_export

    budget_tracker = BudgetTracker(run_id=run_id, max_usd=max_usd, max_turns=max_turns)
    token_counter = TokenCounter()
    pricing = PricingRegistry()
    trace_collector = TraceCollector()

    trace_collector.start_trace(run_id, budget_tracker.get_summary())

    interceptor = LLMInterceptor(
        budget_tracker=budget_tracker,
        token_counter=token_counter,
        pricing=pricing,
        trace_collector=trace_collector,
    )

    try:
        interceptor.activate()
        result = func(*args, **kwargs)
        return result
    except BudgetExceededError:
        if on_exceed == "return_none":
            return None
        raise
    finally:
        interceptor.deactivate()
        trace = trace_collector.end_trace()
        if trace and trace_export != "none":
            exporter = get_exporter(trace_export, config.trace_export_path)
            exporter.export(trace)


async def _run_with_budget_async(
    func: Callable[..., Any],
    max_usd: float,
    max_turns: int | None,
    on_exceed: str,
    trace_export: str | None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    config = TetherConfig()

    if trace_export is None:
        trace_export = config.trace_export

    budget_tracker = BudgetTracker(run_id=run_id, max_usd=max_usd, max_turns=max_turns)
    token_counter = TokenCounter()
    pricing = PricingRegistry()
    trace_collector = TraceCollector()

    trace_collector.start_trace(run_id, budget_tracker.get_summary())

    interceptor = LLMInterceptor(
        budget_tracker=budget_tracker,
        token_counter=token_counter,
        pricing=pricing,
        trace_collector=trace_collector,
    )

    try:
        interceptor.activate()
        result = await func(*args, **kwargs)
        return result
    except BudgetExceededError:
        if on_exceed == "return_none":
            return None
        raise
    finally:
        interceptor.deactivate()
        trace = trace_collector.end_trace()
        if trace and trace_export != "none":
            exporter = get_exporter(trace_export, config.trace_export_path)
            exporter.export(trace)
