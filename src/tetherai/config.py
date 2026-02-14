import os
from dataclasses import dataclass
from typing import Any, Literal, cast

TokenCounterBackend = Literal["tiktoken", "litellm", "auto"]
PricingSource = Literal["bundled", "litellm"]
TraceExport = Literal["console", "json", "none", "otlp"]


@dataclass(frozen=True)
class TetherConfig:
    collector_url: str | None = None
    default_budget_usd: float = 10.0
    default_max_turns: int = 50
    token_counter_backend: TokenCounterBackend = "auto"
    pricing_source: PricingSource = "bundled"
    log_level: str = "WARNING"
    trace_export: TraceExport = "console"
    trace_export_path: str = "./tetherai_traces/"

    def __post_init__(self) -> None:
        if self.default_budget_usd < 0:
            raise ValueError("default_budget_usd must be non-negative")

        if self.default_max_turns is not None and self.default_max_turns < 0:
            raise ValueError("default_max_turns must be non-negative")

        valid_backends = ("tiktoken", "litellm", "auto")
        if self.token_counter_backend not in valid_backends:
            raise ValueError(
                f"Invalid token_counter_backend: {self.token_counter_backend}. "
                f"Must be one of {valid_backends}"
            )

        valid_pricing = ("bundled", "litellm")
        if self.pricing_source not in valid_pricing:
            raise ValueError(
                f"Invalid pricing_source: {self.pricing_source}. "
                f"Must be one of {valid_pricing}"
            )

        valid_export = ("console", "json", "none", "otlp")
        if self.trace_export not in valid_export:
            raise ValueError(
                f"Invalid trace_export: {self.trace_export}. "
                f"Must be one of {valid_export}"
            )

    @classmethod
    def from_env(cls) -> "TetherConfig":
        return cls(
            collector_url=os.getenv("TETHERAI_COLLECTOR_URL"),
            default_budget_usd=float(
                os.getenv("TETHERAI_DEFAULT_BUDGET_USD", "10.0")
            ),
            default_max_turns=int(
                os.getenv("TETHERAI_DEFAULT_MAX_TURNS", "50")
            ),
            token_counter_backend=cls._resolve_backend(
                os.getenv("TETHERAI_TOKEN_COUNTER_BACKEND", "auto")
            ),
            pricing_source=cast(
                PricingSource, os.getenv("TETHERAI_PRICING_SOURCE", "bundled") or "bundled"
            ),
            log_level=os.getenv("TETHERAI_LOG_LEVEL", "WARNING"),
            trace_export=cast(
                TraceExport, os.getenv("TETHERAI_TRACE_EXPORT", "console") or "console"
            ),
            trace_export_path=os.getenv(
                "TETHERAI_TRACE_EXPORT_PATH", "./tetherai_traces/"
            ),
        )

    @staticmethod
    def _resolve_backend(backend: str) -> TokenCounterBackend:
        if backend == "auto":
            try:
                import litellm
                return "litellm"
            except ImportError:
                return "tiktoken"
        return backend  # type: ignore[return-value]

    def resolve_backend(self) -> TokenCounterBackend:
        return self._resolve_backend(self.token_counter_backend)


def load_config(**kwargs: Any) -> TetherConfig:
    env_config = TetherConfig.from_env()

    config_dict = {}
    for field_name in TetherConfig.__dataclass_fields__:
        if field_name in kwargs:
            config_dict[field_name] = kwargs[field_name]
        else:
            config_dict[field_name] = getattr(env_config, field_name)

    return TetherConfig(**config_dict)
