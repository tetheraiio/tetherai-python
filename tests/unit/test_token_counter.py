import time

import pytest

from tetherai.token_counter import TokenCounter, count_messages, count_tokens


class TestTokenCounter:
    def test_empty_string_returns_zero(self):
        counter = TokenCounter(backend="tiktoken")
        assert counter.count_tokens("") == 0

    def test_hello_world_openai(self):
        counter = TokenCounter(backend="tiktoken")
        count = counter.count_tokens("Hello, world!", model="gpt-4o")
        assert count > 0
        assert count == 4

    def test_unicode_and_emoji(self):
        counter = TokenCounter(backend="tiktoken")
        japanese = "こんにちは世界"
        count = counter.count_tokens(japanese, model="gpt-4o")
        assert count > 0

        emoji = "Hello 👋🌍"
        count_emoji = counter.count_tokens(emoji, model="gpt-4o")
        assert count_emoji > 0

    def test_large_input_performance(self):
        counter = TokenCounter(backend="tiktoken")
        large_text = "a" * 10000

        start = time.time()
        count = counter.count_tokens(large_text, model="gpt-4o")
        elapsed = time.time() - start

        assert elapsed < 5.0
        assert count > 0

    def test_messages_include_framing_overhead(self):
        counter = TokenCounter(backend="tiktoken")
        messages = [{"role": "user", "content": "hi"}]
        message_count = counter.count_messages(messages, model="gpt-4o")
        text_count = counter.count_tokens("hi", model="gpt-4o")
        assert message_count > text_count

    def test_tiktoken_fallback_warns_for_claude(self, caplog):
        counter = TokenCounter(backend="tiktoken")
        with caplog.at_level("WARNING"):
            counter.count_tokens("hello", model="claude-3-sonnet")
        assert any("Claude" in record.message for record in caplog.records)

    def test_unknown_model_falls_back_gracefully(self):
        counter = TokenCounter(backend="tiktoken")
        count = counter.count_tokens("test", model="unknown-model-xyz")
        assert count > 0

    def test_system_message_counted(self):
        counter = TokenCounter(backend="tiktoken")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        count = counter.count_messages(messages, model="gpt-4o")
        assert count > 0

    def test_tool_definitions_counted(self):
        counter = TokenCounter(backend="tiktoken")
        messages = [
            {
                "role": "user",
                "content": "What's the weather?",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather info",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            }
        ]
        count = counter.count_messages(messages, model="gpt-4o")
        assert count > 0


class TestTokenCounterLitellm:
    def test_litellm_backend_uses_correct_tokenizer(self):
        try:
            import litellm  # noqa: F401
        except ImportError:
            pytest.skip("litellm not installed")

        counter = TokenCounter(backend="litellm")
        count = counter.count_tokens("Hello, world!", model="gpt-4o")
        assert count > 0


class TestTokenCounterFunctions:
    def test_count_tokens_function(self):
        count = count_tokens("Hello, world!", model="gpt-4o")
        assert count > 0

    def test_count_messages_function(self):
        messages = [{"role": "user", "content": "Hi"}]
        count = count_messages(messages, model="gpt-4o")
        assert count > 0
