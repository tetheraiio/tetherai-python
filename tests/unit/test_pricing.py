import pytest

from tetherai.exceptions import UnknownModelError
from tetherai.pricing import BUNDLED_PRICING, PricingRegistry, get_pricing_registry


class TestPricingRegistry:
    def test_bundled_registry_has_major_models(self):
        registry = PricingRegistry()
        assert registry.get_input_cost("gpt-4o") > 0
        assert registry.get_input_cost("gpt-4o-mini") > 0
        assert registry.get_input_cost("claude-3-5-sonnet-20241022") > 0
        assert registry.get_input_cost("claude-3-haiku-20240307") > 0

    def test_input_cheaper_than_output(self):
        registry = PricingRegistry()
        for model in BUNDLED_PRICING:
            input_cost = registry.get_input_cost(model)
            output_cost = registry.get_output_cost(model)
            assert input_cost <= output_cost, f"{model}: {input_cost} > {output_cost}"

    def test_estimate_call_cost_math(self):
        registry = PricingRegistry()
        cost = registry.estimate_call_cost("gpt-4o", 1000, 500)
        expected = (0.0025 * 1000 / 1000) + (0.01 * 500 / 1000)
        assert abs(cost - expected) < 0.0001

    def test_unknown_model_raises(self):
        registry = PricingRegistry()
        with pytest.raises(UnknownModelError) as exc_info:
            registry.get_input_cost("my-custom-model")
        assert exc_info.value.model == "my-custom-model"

    def test_model_alias_resolution(self):
        registry = PricingRegistry()
        assert registry.resolve_model_alias("gpt4o") == "gpt-4o"
        assert registry.resolve_model_alias("claude-sonnet") == "claude-3-5-sonnet-20241022"

    def test_zero_tokens_returns_zero_cost(self):
        registry = PricingRegistry()
        cost = registry.estimate_call_cost("gpt-4o", 0, 0)
        assert cost == 0.0

    def test_custom_model_registration(self):
        registry = PricingRegistry()
        registry.register_custom_model("my-fine-tune", 0.001, 0.002)
        assert registry.get_input_cost("my-fine-tune") == 0.001
        assert registry.get_output_cost("my-fine-tune") == 0.002


class TestPricingRegistryLitellm:
    def test_litellm_backend_uses_litellm(self):
        registry = PricingRegistry(source="litellm")
        try:
            import litellm  # noqa: F401

            cost = registry.get_input_cost("gpt-4o")
            assert cost > 0
        except ImportError:
            pytest.skip("litellm not installed")

    def test_bundled_backend_does_not_require_litellm(self):
        registry = PricingRegistry(source="bundled")
        cost = registry.get_input_cost("gpt-4o")
        assert cost > 0


class TestGetPricingRegistry:
    def test_get_pricing_registry_returns_registry(self):
        registry = get_pricing_registry()
        assert isinstance(registry, PricingRegistry)

    def test_get_pricing_registry_with_source(self):
        registry = get_pricing_registry(source="litellm")
        assert registry._source == "litellm"
