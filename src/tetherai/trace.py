from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

MAX_PREVIEW_LENGTH = 200


def generate_id() -> str:
    import uuid

    return uuid.uuid4().hex[:16]


@dataclass
class Span:
    span_id: str = field(default_factory=generate_id)
    parent_span_id: str | None = None
    run_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    span_type: str = "llm_call"
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    status: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)
    input_preview: str | None = None
    output_preview: str | None = None

    def __post_init__(self) -> None:
        if self.input_preview and len(self.input_preview) > MAX_PREVIEW_LENGTH:
            self.input_preview = self.input_preview[:MAX_PREVIEW_LENGTH] + "..."

        if self.output_preview and len(self.output_preview) > MAX_PREVIEW_LENGTH:
            self.output_preview = self.output_preview[:MAX_PREVIEW_LENGTH] + "..."

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "span_type": self.span_type,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "status": self.status,
            "metadata": self.metadata,
            "input_preview": self.input_preview,
            "output_preview": self.output_preview,
        }


@dataclass
class Trace:
    run_id: str
    spans: list[Span] = field(default_factory=list)
    budget_summary: dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    def add_span(self, span: Span) -> None:
        self.spans.append(span)

    @property
    def total_cost(self) -> float:
        return sum(span.cost_usd or 0 for span in self.spans)

    @property
    def total_input_tokens(self) -> int:
        return sum(span.input_tokens or 0 for span in self.spans)

    @property
    def total_output_tokens(self) -> int:
        return sum(span.output_tokens or 0 for span in self.spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "spans": [span.to_dict() for span in self.spans],
            "budget_summary": self.budget_summary,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_cost": self.total_cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }


class TraceCollector:
    def __init__(self) -> None:
        self._current_trace: Trace | None = None

    def start_trace(self, run_id: str, budget_summary: dict[str, Any] | None = None) -> Trace:
        self._current_trace = Trace(
            run_id=run_id,
            budget_summary=budget_summary or {},
        )
        return self._current_trace

    def end_trace(self) -> Trace | None:
        if self._current_trace:
            self._current_trace.end_time = datetime.now()
            trace = self._current_trace
            self._current_trace = None
            return trace
        return None

    def add_span(self, span: Span) -> None:
        if self._current_trace:
            self._current_trace.add_span(span)

    def get_current_trace(self) -> Trace | None:
        return self._current_trace
