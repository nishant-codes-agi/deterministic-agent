"""LLM provider abstract base class and data models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class LLMMessage(BaseModel):
    """A single message in an LLM conversation."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant"] = Field(description="Message role")
    content: str = Field(description="Message content")


class LLMResponse(BaseModel):
    """Response from an LLM provider."""

    content: str = Field(description="Response content text")
    structured: Optional[dict] = Field(default=None, description="Parsed structured output")
    tokens_in: int = Field(description="Input tokens used")
    tokens_out: int = Field(description="Output tokens used")
    latency_ms: int = Field(description="Response latency in milliseconds")
    model: str = Field(description="Actual model used")
    cost_usd: float = Field(description="Estimated cost in USD")


class LLMProvider(ABC):
    """Abstract base for LLM providers.

    Supports per-call model routing: the same provider instance can send requests
    to different models. This enables our multi-model strategy where planning
    uses Gemini, coding uses Sonnet, and evaluation uses GPT-4o-mini -- all
    through one OpenRouter API key.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        response_format: Optional[type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send messages to the LLM and get a response.

        Args:
            messages: Conversation messages to send.
            model: Model to use for this specific call. If None, uses the
                   provider's default model. This enables per-phase routing.
            response_format: If provided, instruct the model to return JSON
                           matching this Pydantic schema.
            temperature: Override temperature for this call.
            max_tokens: Maximum output tokens.
        """
        ...

    @abstractmethod
    def estimate_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Estimate cost in USD for given token counts on a specific model.

        Args:
            model: Model identifier.
            tokens_in: Number of input tokens.
            tokens_out: Number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name string."""
        ...
