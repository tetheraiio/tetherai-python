from tetherai.exceptions import UnknownModelError

BUNDLED_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4.1": (0.003, 0.012),
    "gpt-4.1-mini": (0.0008, 0.0032),
    "gpt-4.1-nano": (0.0002, 0.0008),
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5-turbo": (0.0005, 0.002),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-5-sonnet": (0.003, 0.015),
    "claude-3-opus-20240229": (0.015, 0.075),
    "claude-3-opus": (0.015, 0.075),
    "claude-3-sonnet-20240229": (0.003, 0.015),
    "claude-3-sonnet": (0.003, 0.015),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
    "claude-3-haiku": (0.00025, 0.00125),
    "gemini-1.5-pro": (0.00125, 0.005),
    "gemini-1.5-flash": (0.000075, 0.0003),
    "gemini-1.5-flash-8b": (0.0000375, 0.00015),
    "llama-3-70b": (0.0008, 0.0008),
    "llama-3-8b": (0.0002, 0.0002),
    "mixtral-8x7b": (0.00024, 0.00024),
    "mistral-small": (0.001, 0.003),
    "mistral-medium": (0.0024, 0.0072),
    "mistral-large": (0.004, 0.012),
}

MODEL_ALIASES: dict[str, str] = {
    "gpt4o": "gpt-4o",
    "gpt-4o": "gpt-4o",
    "gpt4o-mini": "gpt-4o-mini",
    "gpt-4-turbo": "gpt-4-turbo",
    "gpt4": "gpt-4",
    "gpt-4": "gpt-4",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "claude-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-opus": "claude-3-opus-20240229",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-sonnet-20240229": "claude-3-sonnet-20240229",
    "claude-3-sonnet-20240229": "claude-3-sonnet-20240229",
    "claude-haiku": "claude-3-haiku-20240307",
    "claude-3-haiku": "claude-3-haiku-20240307",
}


class PricingRegistry:
    def __init__(self, source: str = "bundled"):
        self._source = source
        self._custom_models: dict[str, tuple[float, float]] = {}
        self._bundled = BUNDLED_PRICING.copy()

    def get_input_cost(self, model: str) -> float:
        resolved = self.resolve_model_alias(model)
        if resolved in self._custom_models:
            return self._custom_models[resolved][0]
        if resolved in self._bundled:
            return self._bundled[resolved][0]
        if self._source == "litellm":
            return self._get_litellm_cost(model, "input")
        raise UnknownModelError(f"Unknown model: {model}", model)

    def get_output_cost(self, model: str) -> float:
        resolved = self.resolve_model_alias(model)
        if resolved in self._custom_models:
            return self._custom_models[resolved][1]
        if resolved in self._bundled:
            return self._bundled[resolved][1]
        if self._source == "litellm":
            return self._get_litellm_cost(model, "output")
        raise UnknownModelError(f"Unknown model: {model}", model)

    def estimate_call_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        input_cost = self.get_input_cost(model) * input_tokens / 1000
        output_cost = self.get_output_cost(model) * output_tokens / 1000
        return input_cost + output_cost

    def resolve_model_alias(self, model: str) -> str:
        normalized = model.lower().strip()
        return MODEL_ALIASES.get(normalized, model)

    def register_custom_model(self, model: str, input_cost: float, output_cost: float) -> None:
        self._custom_models[model] = (input_cost, output_cost)

    def _get_litellm_cost(self, model: str, direction: str) -> float:
        try:
            import litellm
        except ImportError:
            raise UnknownModelError(
                f"Unknown model: {model} (litellm not installed)", model
            ) from None
        cost = litellm.cost_per_token(model, direction)  # type: ignore[arg-type,attr-defined]
        if isinstance(cost, tuple):
            return cost[0] if direction == "input" else cost[1]
        return cost


def get_pricing_registry(source: str = "bundled") -> PricingRegistry:
    return PricingRegistry(source=source)
