# Deterministic Agent

A system that wraps a coding agent with a **decision-tracing and path-locking layer**, enabling reproducibility, exploration, and comparison of the multiple valid solution paths an AI agent can take when solving a programming task.

## The Problem

When an LLM-powered coding agent tackles a moderately complex task, the problem isn't that it fails -- it's that it succeeds *differently* every time. The system captures the structured decision trace of each run, allows exact replay via path locking, and enables users to fork at any decision point to explore counterfactual paths.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/nishant-codes-agi/deterministic-agent.git
cd deterministic-agent

# 2. Copy the example env and fill in your API key
cp .env.example .env
# Edit .env and set LLM_API_KEY to your OpenRouter key

# 3. Start services
docker-compose up -d

# 4. Run the agent
deterministic-agent run "Build a stock anomaly detection pipeline"
```

## Environment Variables

All configuration is done via environment variables, loaded from a `.env` file at the project root. Copy `.env.example` to `.env` and fill in your values.

### LLM Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_PROVIDER` | No | `openrouter` | LLM provider backend. OpenRouter is recommended as it provides access to all models with a single API key. Options: `openrouter`, `openai`, `anthropic`. |
| `LLM_API_KEY` | **Yes** | -- | Your API key. For OpenRouter, get one at [openrouter.ai/keys](https://openrouter.ai/keys). This is the only secret required to run the system. |
| `LLM_BASE_URL` | No | `https://openrouter.ai/api/v1` | Base URL for the LLM API. Change this if using a direct provider or local model server. |
| `LLM_TEMPERATURE` | No | `0.0` | Default sampling temperature. Set to 0 for maximum determinism. The system does NOT rely on temperature for reproducibility -- it uses decision capture and prompt injection. |
| `LLM_MAX_TOKENS` | No | `4096` | Default maximum output tokens per LLM call. |

### Per-Phase Model Routing

The agent uses different models for different phases to optimize cost and quality. All models are accessed through the same OpenRouter API key.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_MODEL_PLANNING` | No | `google/gemini-2.5-flash` | Model for the **planning** phase. Needs structured reasoning ability. Gemini Flash is cheap and good at JSON output. |
| `LLM_MODEL_CODING` | No | `anthropic/claude-sonnet-4` | Model for the **coding** phase. Needs highest code quality. Claude Sonnet 4 produces the best first-pass code. |
| `LLM_MODEL_EVALUATION` | No | `openai/gpt-4o-mini` | Model for the **evaluation** phase. Just needs to classify success/failure as JSON. GPT-4o-mini is cheapest for this. |
| `LLM_MODEL_RECOVERY` | No | `anthropic/claude-sonnet-4` | Model for the **recovery** phase (fixing errors). Needs code comprehension to diagnose and fix issues. |
| `LLM_MODEL_FALLBACK` | No | `deepseek/deepseek-chat-v3.2` | Fallback model used when the primary model returns an API error. |
| `LLM_MODEL_OVERRIDE` | No | -- | If set, **overrides all per-phase models** with this single model. Useful for testing or cost control. Example: `anthropic/claude-sonnet-4`. |

### Database

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | No | `postgresql+asyncpg://agent:agent@db:5432/deterministic_agent` | Async PostgreSQL connection URL. The `db` hostname resolves within docker-compose. For local development outside Docker, use `localhost`. |

### Redis

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REDIS_URL` | No | `redis://redis:6379/0` | Redis connection URL for caching. Redis is optional -- if unavailable, the system gracefully degrades to in-memory + filesystem caching only. |

### Agent Behavior

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AGENT_MAX_ITERATIONS` | No | `10` | Maximum number of code-execute-evaluate loops before the agent gives up. Each iteration may involve coding, executing, evaluating, and optionally recovering. |
| `AGENT_EXECUTION_TIMEOUT` | No | `120` | Timeout in seconds for each sandbox code execution. Prevents runaway processes. |
| `AGENT_MAX_COST_USD` | No | `2.00` | Maximum total LLM cost per run in USD. If exceeded, the run is finalized with `PARTIAL` status and a `CostLimitExceededError`. |

### Paths

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RUNS_DIR` | No | `./runs` | Directory where run artifacts are stored. Each run creates a subdirectory containing the decision trace (JSON), generated code (workspace/), produced artifacts, and logs. |

### API Server

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_HOST` | No | `0.0.0.0` | Host to bind the FastAPI server to. |
| `API_PORT` | No | `8000` | Port to bind the FastAPI server to. |

### Sample `.env` file

```bash
# === REQUIRED ===
LLM_API_KEY=sk-or-v1-your-openrouter-key-here

# === LLM (all optional, sensible defaults) ===
LLM_PROVIDER=openrouter
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=4096

# Per-phase model routing (cost-optimized defaults)
LLM_MODEL_PLANNING=google/gemini-2.5-flash
LLM_MODEL_CODING=anthropic/claude-sonnet-4
LLM_MODEL_EVALUATION=openai/gpt-4o-mini
LLM_MODEL_RECOVERY=anthropic/claude-sonnet-4
LLM_MODEL_FALLBACK=deepseek/deepseek-chat-v3.2

# Uncomment to use a single model for all phases:
# LLM_MODEL_OVERRIDE=anthropic/claude-sonnet-4

# === Infrastructure (defaults work with docker-compose) ===
DATABASE_URL=postgresql+asyncpg://agent:agent@db:5432/deterministic_agent
REDIS_URL=redis://redis:6379/0

# === Agent behavior ===
AGENT_MAX_ITERATIONS=10
AGENT_EXECUTION_TIMEOUT=120
AGENT_MAX_COST_USD=2.00

# === Paths ===
RUNS_DIR=./runs

# === API server ===
API_HOST=0.0.0.0
API_PORT=8000
```

## Architecture

```
Interface Layer        CLI  .  FastAPI (stretch)  .  WS Stream
Orchestration Layer    RunManager  .  PathLocker  .  ForkEngine
Agent Layer            Planner  .  Coder  .  Executor  .  Evaluator
Decision Capture       DecisionTracer  .  DecisionPointRegistry
Infrastructure         LLM Provider  .  Sandbox  .  TraceStore  .  EventBus
```

## License

MIT
