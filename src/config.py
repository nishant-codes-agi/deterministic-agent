"""Pydantic Settings configuration, reads .env file."""

from __future__ import annotations

import enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings


class AgentPhase(str, enum.Enum):
    """Phases of agent execution for model routing."""
    PLANNING = "planning"
    CODING = "coding"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    RECOVERING = "recovering"


class LLMConfig(BaseSettings):
    """LLM provider configuration with per-phase model routing."""

    model_config = {"env_prefix": "LLM_"}

    provider: str = Field(default="openrouter", description="LLM provider name")
    api_key: SecretStr = Field(default=SecretStr(""), description="API key for LLM provider")
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for LLM API",
    )
    temperature: float = Field(default=0.0, description="Default temperature")
    max_tokens: int = Field(default=4096, description="Default max tokens")

    # Per-phase model routing
    model_planning: str = Field(
        default="google/gemini-2.5-flash",
        description="Model for planning phase",
    )
    model_coding: str = Field(
        default="anthropic/claude-sonnet-4",
        description="Model for coding phase",
    )
    model_evaluation: str = Field(
        default="openai/gpt-4o-mini",
        description="Model for evaluation phase",
    )
    model_recovery: str = Field(
        default="anthropic/claude-sonnet-4",
        description="Model for recovery phase",
    )
    model_fallback: str = Field(
        default="deepseek/deepseek-chat-v3.2",
        description="Fallback model on API errors",
    )
    model_override: Optional[str] = Field(
        default=None,
        description="Override all per-phase models with this single model",
    )

    def get_model_for_phase(self, phase: AgentPhase) -> str:
        """Get the configured model for a given agent phase.

        Checks override first, then returns the phase-specific model.
        """
        if self.model_override:
            return self.model_override
        phase_model_map = {
            AgentPhase.PLANNING: self.model_planning,
            AgentPhase.CODING: self.model_coding,
            AgentPhase.EVALUATING: self.model_evaluation,
            AgentPhase.RECOVERING: self.model_recovery,
            AgentPhase.EXECUTING: self.model_coding,  # No LLM call, but default to coding
        }
        return phase_model_map[phase]


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    model_config = {"env_prefix": "DATABASE_"}

    url: str = Field(
        default="postgresql+asyncpg://agent:agent@db:5432/deterministic_agent",
        alias="DATABASE_URL",
        description="Async database URL",
    )
    echo: bool = Field(default=False, description="Echo SQL statements")
    pool_size: int = Field(default=5, description="Connection pool size")


class RedisConfig(BaseSettings):
    """Redis configuration."""

    model_config = {"env_prefix": "REDIS_"}

    url: str = Field(
        default="redis://redis:6379/0",
        alias="REDIS_URL",
        description="Redis connection URL",
    )
    ttl_default: int = Field(default=3600, description="Default TTL in seconds")
    max_memory_mb: int = Field(default=100, description="Max memory for in-memory cache")


class AgentConfig(BaseSettings):
    """Agent behavior configuration."""

    model_config = {"env_prefix": "AGENT_"}

    max_iterations: int = Field(default=10, description="Max agent loop iterations")
    execution_timeout: int = Field(default=120, description="Sandbox execution timeout seconds")
    max_cost_usd: float = Field(default=2.00, description="Max cost per run in USD")


class Settings(BaseSettings):
    """Root settings composing all config sections."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    runs_dir: Path = Field(default=Path("./runs"), description="Run artifacts directory")
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings singleton."""
    return Settings()
