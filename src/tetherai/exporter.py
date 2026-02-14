import json
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

from tetherai.trace import Trace


@runtime_checkable
class TraceExporter(Protocol):
    def export(self, trace: Trace) -> None: ...


class ConsoleExporter:
    def export(self, trace: Trace) -> None:
        print(f"=== TetherAI Trace: {trace.run_id} ===", file=sys.stderr)
        print(f"Total Cost: ${trace.total_cost:.4f}", file=sys.stderr)
        print(f"Input Tokens: {trace.total_input_tokens}", file=sys.stderr)
        print(f"Output Tokens: {trace.total_output_tokens}", file=sys.stderr)
        print(f"Spans: {len(trace.spans)}", file=sys.stderr)
        print(file=sys.stderr)

        for i, span in enumerate(trace.spans):
            print(f"  [{i + 1}] {span.span_type}: {span.model or 'N/A'}", file=sys.stderr)
            if span.cost_usd is not None:
                print(f"      Cost: ${span.cost_usd:.6f}", file=sys.stderr)
            if span.input_tokens:
                print(f"      Input: {span.input_tokens} tokens", file=sys.stderr)
            if span.output_tokens:
                print(f"      Output: {span.output_tokens} tokens", file=sys.stderr)
            print(file=sys.stderr)


class JSONFileExporter:
    def __init__(self, output_dir: str = "./tetherai_traces/"):
        self.output_dir = Path(output_dir)

    def export(self, trace: Trace) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{trace.run_id}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(trace.to_dict(), f, indent=2)


class NoopExporter:
    def export(self, trace: Trace) -> None:
        pass


def get_exporter(exporter_type: str, output_dir: str = "./tetherai_traces/") -> TraceExporter:
    if exporter_type == "console":
        return ConsoleExporter()
    elif exporter_type == "json":
        return JSONFileExporter(output_dir=output_dir)
    elif exporter_type == "none" or exporter_type == "noop":
        return NoopExporter()
    else:
        raise ValueError(f"Unknown exporter type: {exporter_type}")
