import pickle
import pytest

from tetherai.exceptions import (
    BudgetExceededError,
    TurnLimitError,
    TokenCountError,
    TetherError,
)


class TestExceptions:
    def test_budget_exceeded_is_tether_error(self):
        err = BudgetExceededError(
            message="Budget exceeded",
            run_id="test-123",
            budget_usd=2.0,
            spent_usd=2.5,
            last_model="gpt-4o",
        )
        assert isinstance(err, TetherError)

    def test_budget_exceeded_carries_metadata(self):
        err = BudgetExceededError(
            message="Budget exceeded",
            run_id="abc123",
            budget_usd=2.0,
            spent_usd=2.34,
            last_model="gpt-4o",
            trace_url="https://tetherai.com/traces/abc123",
        )
        assert err.run_id == "abc123"
        assert err.budget_usd == 2.0
        assert err.spent_usd == 2.34
        assert err.last_model == "gpt-4o"
        assert err.trace_url == "https://tetherai.com/traces/abc123"

    def test_budget_exceeded_str_repr(self):
        err = BudgetExceededError(
            message="Budget exceeded",
            run_id="abc123",
            budget_usd=2.0,
            spent_usd=2.34,
            last_model="gpt-4o",
        )
        assert "2.34" in str(err)
        assert "2.00" in str(err)
        assert "abc123" in str(err)

    def test_turn_limit_is_tether_error(self):
        err = TurnLimitError(
            message="Turn limit exceeded",
            run_id="test-123",
            max_turns=5,
            current_turn=6,
        )
        assert isinstance(err, TetherError)

    def test_turn_limit_carries_metadata(self):
        err = TurnLimitError(
            message="Turn limit exceeded",
            run_id="xyz789",
            max_turns=10,
            current_turn=15,
        )
        assert err.run_id == "xyz789"
        assert err.max_turns == 10
        assert err.current_turn == 15

    def test_exceptions_are_picklable(self):
        err1 = BudgetExceededError(
            message="Budget exceeded",
            run_id="abc123",
            budget_usd=2.0,
            spent_usd=2.34,
            last_model="gpt-4o",
        )
        err2 = TurnLimitError(
            message="Turn limit exceeded",
            run_id="xyz789",
            max_turns=10,
            current_turn=15,
        )
        err3 = TokenCountError(
            message="Token counting failed",
            model="unknown-model",
        )

        pickled1 = pickle.dumps(err1)
        pickled2 = pickle.dumps(err2)
        pickled3 = pickle.dumps(err3)

        unpickled1 = pickle.loads(pickled1)
        unpickled2 = pickle.loads(pickled2)
        unpickled3 = pickle.loads(pickled3)

        assert unpickled1.run_id == "abc123"
        assert unpickled1.spent_usd == 2.34
        assert unpickled2.run_id == "xyz789"
        assert unpickled2.max_turns == 10
        assert unpickled3.model == "unknown-model"

    def test_token_count_error_carries_model(self):
        err = TokenCountError(
            message="Unknown encoding",
            model="some-model",
        )
        assert err.model == "some-model"
