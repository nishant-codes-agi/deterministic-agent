"""Tests for OpenRouter LLM provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import LLMConfig
from src.llm.base import LLMMessage, LLMResponse
from src.llm.openrouter_provider import MODEL_PRICING, OpenRouterProvider


def make_config(**overrides) -> LLMConfig:
    """Create a test LLMConfig."""
    defaults = {
        "provider": "openrouter",
        "api_key": "test-key",
        "base_url": "https://openrouter.ai/api/v1",
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    defaults.update(overrides)
    return LLMConfig(**defaults)


def make_mock_response(content: str = "hello", model: str = "test-model",
                       prompt_tokens: int = 10, completion_tokens: int = 5):
    """Create a mock OpenAI response object."""
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = content
    mock_resp.model = model
    mock_resp.usage = MagicMock()
    mock_resp.usage.prompt_tokens = prompt_tokens
    mock_resp.usage.completion_tokens = completion_tokens
    return mock_resp


@pytest.fixture
def provider():
    config = make_config()
    return OpenRouterProvider(config)


class TestMessageFormatting:
    async def test_openrouter_message_formatting(self, provider):
        """Verify messages are converted to OpenAI format correctly."""
        messages = [
            LLMMessage(role="system", content="You are helpful"),
            LLMMessage(role="user", content="Hello"),
        ]
        mock_resp = make_mock_response()

        with patch.object(
            provider._client.chat.completions, "create",
            new_callable=AsyncMock, return_value=mock_resp,
        ) as mock_create:
            await provider.complete(messages)

            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["messages"] == [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]


class TestModelRouting:
    async def test_per_call_model_routing(self, provider):
        """Call complete() with different model strings, verify routing."""
        messages = [LLMMessage(role="user", content="test")]
        mock_resp = make_mock_response()

        for model in ["google/gemini-2.5-flash", "anthropic/claude-sonnet-4", "openai/gpt-4o-mini"]:
            with patch.object(
                provider._client.chat.completions, "create",
                new_callable=AsyncMock, return_value=mock_resp,
            ) as mock_create:
                await provider.complete(messages, model=model)
                call_kwargs = mock_create.call_args[1]
                assert call_kwargs["model"] == model


class TestCostCalculation:
    def test_cost_calculation_gemini(self, provider):
        """Gemini Flash, 4K in 2K out -> $0.0062."""
        cost = provider.estimate_cost("google/gemini-2.5-flash", 4000, 2000)
        expected = (4000 / 1_000_000) * 0.30 + (2000 / 1_000_000) * 2.50
        assert abs(cost - expected) < 1e-9
        assert abs(cost - 0.0062) < 1e-6

    def test_cost_calculation_sonnet(self, provider):
        """Claude Sonnet 4, 4K in 4K out -> $0.072."""
        cost = provider.estimate_cost("anthropic/claude-sonnet-4", 4000, 4000)
        expected = (4000 / 1_000_000) * 3.00 + (4000 / 1_000_000) * 15.00
        assert abs(cost - expected) < 1e-9
        assert abs(cost - 0.072) < 1e-6

    def test_cost_calculation_gpt4o_mini(self, provider):
        """GPT-4o-mini, 2K in 500 out -> $0.0006."""
        cost = provider.estimate_cost("openai/gpt-4o-mini", 2000, 500)
        expected = (2000 / 1_000_000) * 0.15 + (500 / 1_000_000) * 0.60
        assert abs(cost - expected) < 1e-9
        assert abs(cost - 0.0006) < 1e-6

    def test_cost_calculation_unknown_model(self, provider):
        """Unknown model uses conservative estimate."""
        cost = provider.estimate_cost("unknown/model-xyz", 1000, 1000)
        # Default rates: $1/$5 per 1M
        expected = (1000 / 1_000_000) * 1.00 + (1000 / 1_000_000) * 5.00
        assert abs(cost - expected) < 1e-9

    def test_cost_versioned_model_string(self, provider):
        """Versioned model string like 'anthropic/claude-sonnet-4:2025-05-14' works."""
        cost = provider.estimate_cost("anthropic/claude-sonnet-4:2025-05-14", 1000, 1000)
        expected = (1000 / 1_000_000) * 3.00 + (1000 / 1_000_000) * 15.00
        assert abs(cost - expected) < 1e-9


class TestRetryAndFallback:
    async def test_rate_limit_retry(self, provider):
        """Mock RateLimitError on first call, success on second."""
        messages = [LLMMessage(role="user", content="test")]
        mock_resp = make_mock_response()

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("429 rate limit exceeded")
            return mock_resp

        with patch.object(
            provider._client.chat.completions, "create",
            new_callable=AsyncMock, side_effect=side_effect,
        ):
            result = await provider.complete(messages)
            assert result.content == "hello"
            assert call_count == 2

    async def test_fallback_on_api_error(self, provider):
        """Mock APIError on primary model, verify fallback model is tried."""
        messages = [LLMMessage(role="user", content="test")]
        mock_resp = make_mock_response(model="deepseek/deepseek-chat-v3.2")

        call_count = 0

        async def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                err = type("APIError", (Exception,), {"status_code": 500})()
                raise err
            return mock_resp

        with patch.object(
            provider._client.chat.completions, "create",
            new_callable=AsyncMock, side_effect=side_effect,
        ):
            result = await provider.complete(messages, model="google/gemini-2.5-flash")
            assert call_count == 2


class TestJSONParsing:
    async def test_json_response_parsing(self, provider):
        """Mock response with JSON content, verify structured output is parsed."""
        messages = [LLMMessage(role="user", content="test")]
        json_content = '{"status": "ok", "data": [1, 2, 3]}'
        mock_resp = make_mock_response(content=json_content)

        with patch.object(
            provider._client.chat.completions, "create",
            new_callable=AsyncMock, return_value=mock_resp,
        ):
            from pydantic import BaseModel

            class DummyFormat(BaseModel):
                pass

            result = await provider.complete(messages, response_format=DummyFormat)
            assert result.structured == {"status": "ok", "data": [1, 2, 3]}

    async def test_json_extraction_from_markdown_fences(self, provider):
        """Response wrapped in ```json ```, verify extraction."""
        messages = [LLMMessage(role="user", content="test")]
        fenced_content = 'Here is the response:\n```json\n{"result": "success"}\n```'
        mock_resp = make_mock_response(content=fenced_content)

        with patch.object(
            provider._client.chat.completions, "create",
            new_callable=AsyncMock, return_value=mock_resp,
        ):
            from pydantic import BaseModel

            class DummyFormat(BaseModel):
                pass

            result = await provider.complete(messages, response_format=DummyFormat)
            assert result.structured == {"result": "success"}
