import pytest
from datetime import datetime

from tetherai.trace import Span, Trace, TraceCollector, generate_id


class TestSpan:
    def test_span_creation_with_required_fields(self):
        span = Span(run_id="test-123")
        assert span.run_id == "test-123"
        assert span.span_id is not None

    def test_span_id_is_unique(self):
        ids = set()
        for _ in range(1000):
            ids.add(generate_id())
        assert len(ids) == 1000

    def test_span_parent_child_relationship(self):
        parent = Span(run_id="test", span_id="parent-123")
        child = Span(run_id="test", span_id="child-456", parent_span_id="parent-123")
        assert child.parent_span_id == parent.span_id

    def test_input_preview_truncated(self):
        long_input = "a" * 10000
        span = Span(run_id="test", input_preview=long_input)
        assert len(span.input_preview) == 203
        assert span.input_preview.endswith("...")

    def test_output_preview_truncated(self):
        long_output = "b" * 10000
        span = Span(run_id="test", output_preview=long_output)
        assert len(span.output_preview) == 203
        assert span.output_preview.endswith("...")

    def test_span_to_dict_serializable(self):
        span = Span(
            run_id="test-123",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        )
        d = span.to_dict()
        assert isinstance(d, dict)
        assert d["run_id"] == "test-123"
        assert d["model"] == "gpt-4o"


class TestTrace:
    def test_trace_contains_ordered_spans(self):
        trace = Trace(run_id="test-123")
        span1 = Span(run_id="test-123", span_type="agent_step")
        span2 = Span(run_id="test-123", span_type="llm_call")

        import time
        time.sleep(0.01)
        span2.timestamp = datetime.now()

        trace.add_span(span1)
        trace.add_span(span2)

        assert trace.spans[0].timestamp <= trace.spans[1].timestamp

    def test_trace_total_cost(self):
        trace = Trace(run_id="test-123")
        trace.add_span(Span(run_id="test", cost_usd=0.01))
        trace.add_span(Span(run_id="test", cost_usd=0.02))
        trace.add_span(Span(run_id="test", cost_usd=0.03))

        assert trace.total_cost == 0.06

    def test_trace_to_dict_serializable(self):
        trace = Trace(run_id="test-123")
        trace.add_span(Span(run_id="test-123", cost_usd=0.01))

        d = trace.to_dict()
        assert isinstance(d, dict)
        assert "run_id" in d
        assert "spans" in d

    def test_budget_summary_included(self):
        summary = {"budget_usd": 2.0, "spent_usd": 1.5}
        trace = Trace(run_id="test-123", budget_summary=summary)

        d = trace.to_dict()
        assert d["budget_summary"]["budget_usd"] == 2.0
        assert d["budget_summary"]["spent_usd"] == 1.5


class TestTraceCollector:
    def test_trace_collector_starts_and_ends_trace(self):
        collector = TraceCollector()
        trace = collector.start_trace("run-123")
        assert trace.run_id == "run-123"

        ended = collector.end_trace()
        assert ended is not None
        assert ended.run_id == "run-123"
        assert ended.end_time is not None

    def test_trace_collector_add_span(self):
        collector = TraceCollector()
        collector.start_trace("run-123")
        collector.add_span(Span(run_id="run-123", model="gpt-4o"))

        trace = collector.get_current_trace()
        assert len(trace.spans) == 1

    def test_trace_collector_no_current_trace(self):
        collector = TraceCollector()
        assert collector.get_current_trace() is None
