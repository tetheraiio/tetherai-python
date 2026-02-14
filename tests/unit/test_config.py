from unittest.mock import patch

import pytest

from tetherai.config import TetherConfig, load_config


class TestConfig:
    def test_defaults_are_sane(self):
        config = TetherConfig()
        assert config.default_budget_usd == 10.0
        assert config.default_max_turns == 50
        assert config.token_counter_backend == "auto"
        assert config.pricing_source == "bundled"
        assert config.log_level == "WARNING"
        assert config.trace_export == "console"
        assert config.trace_export_path == "./tetherai_traces/"

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("TETHERAI_DEFAULT_BUDGET_USD", "5.0")
        config = TetherConfig.from_env()
        assert config.default_budget_usd == 5.0

    def test_kwargs_beat_env_vars(self, monkeypatch):
        monkeypatch.setenv("TETHERAI_DEFAULT_BUDGET_USD", "5.0")
        config = TetherConfig(default_budget_usd=8.0)
        assert config.default_budget_usd == 8.0

    def test_config_is_frozen(self):
        from dataclasses import FrozenInstanceError

        config = TetherConfig()
        with pytest.raises(FrozenInstanceError):
            config.default_budget_usd = 5.0

    def test_invalid_token_counter_backend(self):
        with pytest.raises(ValueError, match="Invalid token_counter_backend"):
            TetherConfig(token_counter_backend="magic")

    def test_negative_budget_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            TetherConfig(default_budget_usd=-1)

    def test_negative_max_turns_rejected(self):
        with pytest.raises(ValueError):
            TetherConfig(default_max_turns=-1)

    def test_invalid_pricing_source(self):
        with pytest.raises(ValueError, match="Invalid pricing_source"):
            TetherConfig(pricing_source="invalid")

    def test_invalid_trace_export(self):
        with pytest.raises(ValueError, match="Invalid trace_export"):
            TetherConfig(trace_export="invalid")


class TestConfigAutoBackend:
    def test_auto_backend_resolves_to_tiktoken_when_no_litellm(self):
        with patch("tetherai.config.TetherConfig._resolve_backend") as mock:
            mock.side_effect = lambda x: "tiktoken" if x == "auto" else x
            config = TetherConfig(token_counter_backend="auto")
            assert config.resolve_backend() == "tiktoken"

    def test_kwargs_override_collector_url(self):
        config = TetherConfig(collector_url="http://localhost:8080")
        assert config.collector_url == "http://localhost:8080"


class TestLoadConfig:
    def test_load_config_uses_kwargs(self):
        config = load_config(default_budget_usd=15.0)
        assert config.default_budget_usd == 15.0

    def test_load_config_merges_with_env(self, monkeypatch):
        monkeypatch.setenv("TETHERAI_LOG_LEVEL", "DEBUG")
        config = load_config()
        assert config.log_level == "DEBUG"

    def test_load_config_env_var_takes_precedence_over_default(self, monkeypatch):
        monkeypatch.setenv("TETHERAI_DEFAULT_BUDGET_USD", "7.5")
        config = load_config()
        assert config.default_budget_usd == 7.5
