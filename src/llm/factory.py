"""LLM provider factory."""

from __future__ import annotations

from src.config import LLMConfig
from src.llm.base import LLMProvider
from src.llm.openrouter_provider import OpenRouterProvider


def create_llm_provider(config: LLMConfig) -> LLMProvider:
    """Create an LLM provider based on configuration.

    Default and recommended: OpenRouter (one key, all models).
    Also supports direct provider SDKs for users who prefer them.
    """
    match config.provider:
        case "openrouter":
            return OpenRouterProvider(config)
        case "openai":
            # Direct OpenAI -- uses same class, just different base_url
            config_copy = config.model_copy(update={"base_url": "https://api.openai.com/v1"})
            return OpenRouterProvider(config_copy)
        case "anthropic":
            # For direct Anthropic, we'd need a separate provider class since
            # their API format differs. For MVP, recommend OpenRouter instead.
            raise ValueError(
                "Direct Anthropic SDK not implemented. Use provider=openrouter "
                "with an OpenRouter API key to access Claude models."
            )
        case _:
            raise ValueError(f"Unknown LLM provider: {config.provider}")
