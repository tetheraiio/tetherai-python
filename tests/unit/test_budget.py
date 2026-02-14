import pytest
import threading

from tetherai.budget import BudgetTracker, CallRecord
from tetherai.exceptions import BudgetExceededError, TurnLimitError


class TestBudgetTracker:
    def test_fresh_tracker_has_zero_spend(self):
        tracker = BudgetTracker(run_id="test-123", max_usd=10.0)
        assert tracker.spent_usd == 0
        assert tracker.remaining_usd == 10.0

    def test_record_call_accumulates(self):
        tracker = BudgetTracker(run_id="test-123", max_usd=10.0)
        tracker.record_call(100, 50, "gpt-4o", 0.50, 100.0)
        tracker.record_call(100, 50, "gpt-4o", 0.50, 100.0)
        tracker.record_call(100, 50, "gpt-4o", 0.50, 100.0)

        assert tracker.spent_usd == 1.50

    def test_pre_check_passes_under_budget(self):
        tracker = BudgetTracker(run_id="test-123", max_usd=2.0)
        tracker.record_call(100, 50, "gpt-4o", 0.10, 100.0)
        assert tracker.spent_usd == 0.10
        tracker.pre_check(0.05)

    def test_pre_check_blocks_over_budget(self):
        tracker = BudgetTracker(run_id="test-123", max_usd=2.0)
        tracker.record_call(100, 50, "gpt-4o", 1.95, 100.0)
        assert tracker.spent_usd == 1.95

        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.pre_check(0.10)
        assert exc_info.value.run_id == "test-123"

    def test_pre_check_blocks_exact_boundary(self):
        tracker = BudgetTracker(run_id="test-123", max_usd=2.0)
        tracker.record_call(100, 50, "gpt-4o", 2.0, 100.0)
        assert tracker.spent_usd == 2.0

        with pytest.raises(BudgetExceededError):
            tracker.pre_check(0.001)

    def test_turn_limit_enforced(self):
        tracker = BudgetTracker(run_id="test-123", max_usd=100.0, max_turns=3)
        tracker.record_call(100, 50, "gpt-4o", 0.10, 100.0)
        tracker.record_call(100, 50, "gpt-4o", 0.10, 100.0)
        tracker.record_call(100, 50, "gpt-4o", 0.10, 100.0)

        assert tracker.turn_count == 3

        with pytest.raises(TurnLimitError) as exc_info:
            tracker.record_call(100, 50, "gpt-4o", 0.10, 100.0)
        assert exc_info.value.max_turns == 3
        assert exc_info.value.current_turn == 4

    def test_turn_limit_none_means_unlimited(self):
        tracker = BudgetTracker(run_id="test-123", max_usd=100.0, max_turns=None)
        for _ in range(100):
            tracker.record_call(100, 50, "gpt-4o", 0.10, 100.0)
        assert tracker.turn_count == 100

    def test_thread_safety_concurrent_records(self):
        tracker = BudgetTracker(run_id="test-123", max_usd=100.0)

        def make_calls():
            for _ in range(100):
                tracker.record_call(10, 5, "gpt-4o", 0.01, 10.0)

        threads = [threading.Thread(target=make_calls) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert tracker.spent_usd == pytest.approx(10.0, rel=0.01)

    def test_get_summary_structure(self):
        tracker = BudgetTracker(run_id="test-123", max_usd=10.0)
        tracker.record_call(100, 50, "gpt-4o", 0.50, 100.0)

        summary = tracker.get_summary()
        assert summary["run_id"] == "test-123"
        assert summary["budget_usd"] == 10.0
        assert summary["spent_usd"] == 0.50
        assert summary["remaining_usd"] == 9.5
        assert summary["turn_count"] == 1
        assert len(summary["calls"]) == 1

    def test_negative_cost_rejected(self):
        tracker = BudgetTracker(run_id="test-123", max_usd=10.0)
        with pytest.raises(ValueError, match="non-negative"):
            tracker.record_call(100, 50, "gpt-4o", -0.10, 100.0)

    def test_run_id_is_set(self):
        tracker = BudgetTracker(run_id="my-run-456", max_usd=10.0)
        assert tracker.run_id == "my-run-456"

    def test_remaining_usd_never_negative(self):
        tracker = BudgetTracker(run_id="test-123", max_usd=1.0)
        tracker.record_call(10000, 5000, "gpt-4o", 10.0, 100.0)

        assert tracker.remaining_usd == 0.0

    def test_pre_check_includes_already_spent(self):
        tracker = BudgetTracker(run_id="test-123", max_usd=2.0)
        tracker.record_call(100, 50, "gpt-4o", 1.0, 100.0)
        assert tracker.spent_usd == 1.0
        assert tracker.remaining_usd == 1.0

        with pytest.raises(BudgetExceededError):
            tracker.pre_check(1.5)


class TestBudgetTrackerProperties:
    def test_is_exceeded_false_initially(self):
        tracker = BudgetTracker(run_id="test", max_usd=10.0)
        assert tracker.is_exceeded is False

    def test_is_exceeded_true_when_at_limit(self):
        tracker = BudgetTracker(run_id="test", max_usd=1.0)
        tracker.record_call(100, 50, "gpt-4o", 1.0, 100.0)
        assert tracker.is_exceeded is True
