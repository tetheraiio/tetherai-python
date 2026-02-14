import logging
from typing import Any

from tetherai.exceptions import TokenCountError

logger = logging.getLogger(__name__)

TOKENIZER_CACHE: dict[str, Any] = {}

CHATML_FORMATTING = {
    "system": {"prefix": "<|im_start|>system\n", "suffix": "<|im_end|>\n"},
    "user": {"prefix": "<|im_start|>user\n", "suffix": "<|im_end|>\n"},
    "assistant": {"prefix": "<|im_start|>assistant\n", "suffix": "<|im_end|>\n"},
    "tool": {"prefix": "<|im_start|>tool\n", "suffix": "<|im_end|>\n"},
}


def _get_tiktoken_encoder(encoding_name: str = "cl100k_base") -> Any:
    if encoding_name in TOKENIZER_CACHE:
        return TOKENIZER_CACHE[encoding_name]

    try:
        import tiktoken

        encoder = tiktoken.get_encoding(encoding_name)
        TOKENIZER_CACHE[encoding_name] = encoder
        return encoder
    except Exception as e:
        raise TokenCountError(f"Failed to load tiktoken: {e}") from e


def _get_litellm_tokenizer(model: str) -> Any:
    try:
        import litellm

        return litellm.token_counter  # type: ignore[attr-defined]
    except ImportError as e:
        raise TokenCountError("litellm not installed", model) from e


class TokenCounter:
    def __init__(self, backend: str = "auto"):
        self._backend = backend
        self._tiktoken_encoder = None
        self._litellm_tokenizer = None

        if backend == "auto":
            try:
                import litellm  # noqa: F401

                self._backend = "litellm"
            except ImportError:
                self._backend = "tiktoken"

        if self._backend == "tiktoken":
            self._tiktoken_encoder = _get_tiktoken_encoder()

    def count_tokens(self, text: str, model: str = "gpt-4o") -> int:
        if not text:
            return 0

        if self._backend == "tiktoken":
            return self._count_with_tiktoken(text, model)
        elif self._backend == "litellm":
            return self._count_with_litellm(text, model)
        else:
            raise TokenCountError(f"Unknown backend: {self._backend}")

    def count_messages(self, messages: list[dict[str, str]], model: str = "gpt-4o") -> int:
        if not messages:
            return 0

        if self._backend == "tiktoken":
            return self._count_messages_with_tiktoken(messages, model)
        elif self._backend == "litellm":
            return self._count_messages_with_litellm(messages, model)
        else:
            raise TokenCountError(f"Unknown backend: {self._backend}")

    def _count_with_tiktoken(self, text: str, model: str) -> int:
        if model.startswith("claude-"):
            logger.warning(
                f"Using tiktoken for Claude model {model}. "
                f"Token counts may be inaccurate (up to 12% error)."
            )

        encoder = self._tiktoken_encoder
        if encoder is None:
            encoder = _get_tiktoken_encoder()

        return len(encoder.encode(text))

    def _count_with_litellm(self, text: str, model: str) -> int:
        if self._litellm_tokenizer is None:
            self._litellm_tokenizer = _get_litellm_tokenizer(model)

        if model.startswith("claude-"):
            try:
                return self._litellm_tokenizer(model=model, text=text)  # type: ignore[no-any-return,misc]
            except Exception:
                logger.warning(
                    f"litellm token_counter failed for {model}, falling back to tiktoken"
                )
                return self._count_with_tiktoken(text, model)

        return self._litellm_tokenizer(model=model, text=text)  # type: ignore[no-any-return,misc]

    def _count_messages_with_tiktoken(self, messages: list[dict[str, str]], model: str) -> int:
        encoder = self._tiktoken_encoder
        if encoder is None:
            encoder = _get_tiktoken_encoder()

        if model.startswith("claude-"):
            logger.warning(
                f"Using tiktoken for Claude model {model}. Token counts may be inaccurate."
            )

        total_tokens = 0
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            formatting = CHATML_FORMATTING.get(role, CHATML_FORMATTING["user"])
            formatted = f"{formatting['prefix']}{content}{formatting['suffix']}"
            total_tokens += len(encoder.encode(formatted))

        total_tokens += 3
        return total_tokens

    def _count_messages_with_litellm(self, messages: list[dict[str, str]], model: str) -> int:
        if self._litellm_tokenizer is None:
            self._litellm_tokenizer = _get_litellm_tokenizer(model)

        try:
            return self._litellm_tokenizer(model=model, messages=messages)  # type: ignore[no-any-return,misc]
        except Exception:
            logger.warning(f"litellm token_counter failed for {model}, falling back to tiktoken")
            return self._count_messages_with_tiktoken(messages, model)


def count_tokens(text: str, model: str = "gpt-4o", backend: str = "auto") -> int:
    counter = TokenCounter(backend=backend)
    return counter.count_tokens(text, model)


def count_messages(
    messages: list[dict[str, str]], model: str = "gpt-4o", backend: str = "auto"
) -> int:
    counter = TokenCounter(backend=backend)
    return counter.count_messages(messages, model)
