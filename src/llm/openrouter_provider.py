"""OpenRouter LLM provider implementation."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Optional

from openai import AsyncOpenAI
from pydantic import BaseModel

from src.config import LLMConfig
from src.llm.base import LLMMessage, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

# Pricing table: model prefix -> (input_cost_per_1M, output_cost_per_1M)
# OpenRouter passes through provider pricing — these match official rates.
# Keys are the model IDs returned by OpenRouter in response.model (may differ
# from what you send in the request — OpenRouter often returns versioned strings).
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Gemini (planning — best structured reasoning value)
    "google/gemini-2.5-flash": (0.30, 2.50),
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
    # Claude (coding + recovery — highest first-pass code success)
    # Both the logical alias and the versioned ID OpenRouter returns
    "anthropic/claude-sonnet-4": (3.00, 15.00),
    "anthropic/claude-4-sonnet-20250522": (3.00, 15.00),
    "anthropic/claude-sonnet-4-5": (3.00, 15.00),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "anthropic/claude-haiku-4-5-20250514": (1.00, 5.00),
    # GPT (evaluation — cheapest reliable JSON classifier)
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    # DeepSeek (fallback — 95% cheaper than frontier models)
    "deepseek/deepseek-v3.2": (0.28, 0.42),
    "deepseek/deepseek-chat-v3-0324": (0.28, 0.28),
}


class OpenRouterProvider(LLMProvider):
    """Unified LLM provider via OpenRouter.

    Uses the OpenAI SDK with a custom base_url. Supports per-call model routing:
    the same instance can call Gemini for planning, Sonnet for coding, and
    GPT-4o-mini for evaluation -- all with one API key.

    Why one class instead of one-per-provider:
    - OpenRouter normalizes the API surface -- all models speak OpenAI format
    - Eliminates the provider factory pattern entirely
    - Adding a new model = adding one line to MODEL_PRICING
    - The `model` parameter on complete() handles routing, not class hierarchy
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            default_headers={
                "HTTP-Referer": "https://github.com/deterministic-agent",
                "X-Title": "Deterministic Agent",
            },
        )
        self._default_model = config.model_planning

    async def complete(
        self,
        messages: list[LLMMessage],
        model: Optional[str] = None,
        response_format: Optional[type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send completion request to any model via OpenRouter.

        The `model` parameter is how per-phase routing works:
        - _plan() passes model=config.model_planning (Gemini)
        - _code() passes model=config.model_coding (Sonnet)
        - _evaluate() passes model=config.model_evaluation (GPT-4o-mini)
        """
        resolved_model = model or self._default_model
        temp = temperature if temperature is not None else self._config.temperature

        kwargs: dict = {
            "model": resolved_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temp,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = {"type": "json_object"}

        start = time.perf_counter()

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                # Retry with exponential backoff (3 attempts)
                for attempt in range(3):
                    wait = 2 ** attempt
                    logger.warning(f"Rate limited by {resolved_model}, retry in {wait}s")
                    await asyncio.sleep(wait)
                    try:
                        response = await self._client.chat.completions.create(**kwargs)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
            elif hasattr(e, "status_code") or "api" in str(type(e).__name__).lower():
                logger.error(f"API error from {resolved_model}: {e}")
                # If the primary model fails, try the fallback model
                if resolved_model != self._config.model_fallback:
                    logger.warning(f"Falling back to {self._config.model_fallback}")
                    kwargs["model"] = self._config.model_fallback
                    response = await self._client.chat.completions.create(**kwargs)
                else:
                    raise
            else:
                raise

        latency_ms = int((time.perf_counter() - start) * 1000)

        content = response.choices[0].message.content or ""
        # Claude extended-thinking models sometimes prepend <thinking>...</thinking>
        # before the JSON even when json_object mode is requested. Strip it.
        content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL).strip()
        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        actual_model = response.model or resolved_model

        cost = self.estimate_cost(actual_model, tokens_in, tokens_out)

        # Parse structured output if requested
        structured = None
        if response_format and content:
            try:
                structured = json.loads(content)
            except json.JSONDecodeError:
                structured = self._extract_json(content)

        logger.info(
            "LLM call completed",
            extra={
                "model": actual_model,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": latency_ms,
                "cost_usd": f"{cost:.6f}",
            },
        )

        return LLMResponse(
            content=content,
            structured=structured,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            model=actual_model,
            cost_usd=cost,
        )

    def estimate_cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        """Estimate cost using the pricing table."""
        # Strip version suffix (e.g. "anthropic/claude-sonnet-4:2025-05-14" -> "anthropic/claude-sonnet-4")
        base_model = model.split(":")[0] if ":" in model else model
        rates = MODEL_PRICING.get(base_model)

        if not rates:
            # Try prefix matching: "anthropic/claude-4-sonnet-*" -> claude-sonnet-4 rates
            for key, key_rates in MODEL_PRICING.items():
                # e.g. key="anthropic/claude-sonnet-4", base_model="anthropic/claude-4-sonnet-20250522"
                provider = key.split("/")[0] if "/" in key else ""
                b_provider = base_model.split("/")[0] if "/" in base_model else ""
                if provider == b_provider:
                    key_name = key.split("/", 1)[1].replace("-", "").replace(".", "").lower()
                    b_name = base_model.split("/", 1)[1].replace("-", "").replace(".", "").lower()
                    # Strip trailing date-like suffixes (8 digits)
                    b_name = re.sub(r"\d{8}$", "", b_name)
                    if key_name in b_name or b_name in key_name:
                        rates = key_rates
                        break

        if not rates:
            logger.warning(f"No pricing data for model {model}, using estimate")
            rates = (1.00, 5.00)

        input_cost = (tokens_in / 1_000_000) * rates[0]
        output_cost = (tokens_out / 1_000_000) * rates[1]
        return input_cost + output_cost

    @property
    def provider_name(self) -> str:
        """Return the provider name string."""
        return "openrouter"

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """Extract JSON from markdown fences or mixed text."""
        # Try ```json ... ``` pattern first
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Walk brace depth to find outermost JSON object (handles large nested JSON)
        brace_depth = 0
        start_idx = None
        for i, char in enumerate(text):
            if char == "{":
                if brace_depth == 0:
                    start_idx = i
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                if brace_depth == 0 and start_idx is not None:
                    candidate = text[start_idx: i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        start_idx = None
                        continue
        return None
