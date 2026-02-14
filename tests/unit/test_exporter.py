import json
import os
import pytest
import tempfile
from pathlib import Path

from tetherai.exporter import (
    ConsoleExporter,
    JSONFileExporter,
    NoopExporter,
    get_exporter,
)
from tetherai.trace import Trace, Span


class TestConsoleExporter:
    def test_console_exporter_writes_to_stderr(self, capsys):
        trace = Trace(run_id="test-123")
        trace.add_span(
            Span(
                run_id="test-123",
                model="gpt-4o",
                cost_usd=0.01,
                input_tokens=100,
                output_tokens=50,
            )
        )

        exporter = ConsoleExporter()
        exporter.export(trace)

        captured = capsys.readouterr()
        assert "test-123" in captured.err
        assert "gpt-4o" in captured.err


class TestJSONFileExporter:
    def test_json_exporter_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = JSONFileExporter(output_dir=tmpdir)

            trace = Trace(run_id="run-456")
            trace.add_span(Span(run_id="run-456", model="gpt-4o", cost_usd=0.01))

            exporter.export(trace)

            filepath = Path(tmpdir) / "run-456.json"
            assert filepath.exists()

    def test_json_exporter_output_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = JSONFileExporter(output_dir=tmpdir)

            trace = Trace(run_id="run-789")
            trace.add_span(Span(run_id="run-789", model="gpt-4o", cost_usd=0.01))

            exporter.export(trace)

            filepath = Path(tmpdir) / "run-789.json"
            with open(filepath) as f:
                data = json.load(f)
            assert "run_id" in data
            assert len(data["spans"]) == 1

    def test_json_exporter_contains_all_spans(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = JSONFileExporter(output_dir=tmpdir)

            trace = Trace(run_id="run-full")
            trace.add_span(Span(run_id="run-full", model="gpt-4o", cost_usd=0.01))
            trace.add_span(Span(run_id="run-full", model="gpt-4o-mini", cost_usd=0.001))

            exporter.export(trace)

            filepath = Path(tmpdir) / "run-full.json"
            with open(filepath) as f:
                data = json.load(f)
            assert len(data["spans"]) == 2

    def test_json_file_path_created_if_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "nested", "traces")
            exporter = JSONFileExporter(output_dir=nested_path)

            trace = Trace(run_id="run-nested")
            exporter.export(trace)

            assert Path(nested_path, "run-nested.json").exists()


class TestNoopExporter:
    def test_noop_exporter_does_nothing(self):
        exporter = NoopExporter()
        trace = Trace(run_id="test")
        exporter.export(trace)


class TestExporterHandlesEmptyTrace:
    def test_exporter_handles_empty_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = JSONFileExporter(output_dir=tmpdir)
            trace = Trace(run_id="empty-run")
            exporter.export(trace)

            filepath = Path(tmpdir) / "empty-run.json"
            with open(filepath) as f:
                data = json.load(f)
            assert data["run_id"] == "empty-run"
            assert data["spans"] == []


class TestGetExporter:
    def test_get_console_exporter(self):
        exporter = get_exporter("console")
        assert isinstance(exporter, ConsoleExporter)

    def test_get_json_exporter(self):
        exporter = get_exporter("json", output_dir="/tmp/test")
        assert isinstance(exporter, JSONFileExporter)

    def test_get_noop_exporter(self):
        exporter = get_exporter("none")
        assert isinstance(exporter, NoopExporter)

    def test_invalid_exporter_raises(self):
        with pytest.raises(ValueError, match="Unknown exporter"):
            get_exporter("invalid")
