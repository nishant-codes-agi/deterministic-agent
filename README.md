# Deterministic Agent

A system that wraps a coding agent with a **decision-tracing and path-locking layer**, enabling reproducibility, exploration, and comparison of the multiple valid solution paths an AI agent can take when solving a programming task.

The core insight: when an LLM-powered coding agent tackles a moderately complex task, the problem isn't that it fails -- it's that it succeeds *differently* every time. This system captures the structured decision trace of each run, allows exact replay via path locking, and enables users to fork at any decision point to explore counterfactual paths.

The reference task (stock anomaly detection pipeline) is intentionally chosen to surface ~7+ genuine branching points before a single line of code is written.

## Table of Contents

- [Quick Start](#quick-start)
- [Local Development](#local-development)
- [Architecture](#architecture)
- [LLM Strategy: Multi-Model Routing via OpenRouter](#llm-strategy-multi-model-routing-via-openrouter)
- [Path Locking Strategy](#path-locking-strategy)
- [Decision Point Structure](#decision-point-structure)
- [Prompts Used](#prompts-used)
- [Variance Analysis](#variance-analysis)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)
- [Tradeoffs](#tradeoffs)
- [AI Tool Usage](#ai-tool-usage)
- [What I'd Do With More Time](#what-id-do-with-more-time)

## Quick Start

### Prerequisites

- Docker and Docker Compose
- An OpenRouter API key (one key for all models) -- get one at [openrouter.ai/keys](https://openrouter.ai/keys)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/nishant-codes-agi/deterministic-agent.git
cd deterministic-agent

# 2. Copy the example env and add your API key
cp .env.example .env
# Edit .env and set LLM_API_KEY to your OpenRouter key

# 3. Start all services (app, PostgreSQL, Redis)
docker-compose up -d

# 4. Run database migrations
docker-compose exec app alembic upgrade head

# 5. Run the agent
deterministic-agent run "Fetch 2 years of daily AAPL stock data, detect anomalous trading days using statistical methods, and print a summary of the top 5 anomalies"

# 6. View your runs
deterministic-agent list

# 7. Inspect a trace
deterministic-agent trace <run-id>

# 8. Replay with locked decisions
deterministic-agent replay <run-id>

# 9. Fork from a decision point
deterministic-agent fork <run-id> -d dp-003 -c z_score --compare
```

> **Cost note:** First run costs ~$0.05-0.15 using our multi-model strategy. All 5 recorded runs cost ~$0.66 total.

## Local Development

For development outside Docker, you can run the components directly.

### Prerequisites

- Python 3.12+
- PostgreSQL 16+
- Redis 7+ (optional -- gracefully degrades without it)

### Setup

```bash
# 1. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies (including dev)
pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
# Edit .env:
#   - Set LLM_API_KEY to your OpenRouter key
#   - Change DATABASE_URL to use localhost:
#     DATABASE_URL=postgresql+asyncpg://agent:agent@localhost:5432/deterministic_agent
#   - Change REDIS_URL to use localhost:
#     REDIS_URL=redis://localhost:6379/0

# 4. Start PostgreSQL and Redis (if using Docker for infra only)
docker-compose up -d db redis

# 5. Run database migrations
alembic upgrade head

# 6. Start the API server
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# 7. Use the CLI
deterministic-agent run "Your task here"
```

### Database Migrations

The project uses Alembic for database schema management:

```bash
# Apply all pending migrations
alembic upgrade head

# Check current migration status
alembic current

# Create a new migration (after modifying models)
alembic revision --autogenerate -m "description of changes"

# Rollback one migration
alembic downgrade -1

# Inside Docker
docker-compose exec app alembic upgrade head
```

The migration configuration is in `alembic.ini` and migration scripts live in `alembic/versions/`.

## Architecture

```
+-----------------------------------------------------------+
|  Interface Layer                                          |
|  CLI  .  FastAPI  .  WebSocket Stream                     |
+-----------------------------------------------------------+
|  Orchestration Layer                                      |
|  RunManager  .  PathLocker  .  ForkEngine                 |
+-----------------------------------------------------------+
|  Agent Layer                                              |
|  Planner  .  Coder  .  Executor  .  Evaluator             |
+-----------------------------------------------------------+
|  Decision Capture Layer                                   |
|  DecisionTracer  .  DecisionPointRegistry                 |
+-----------------------------------------------------------+
|  Infrastructure Layer                                     |
|  LLM Provider  .  Sandbox  .  TraceStore  .  EventBus     |
+-----------------------------------------------------------+
```

**Interface Layer** -- Entry points: Click CLI (primary), FastAPI REST API, WebSocket streaming.

**Orchestration Layer** -- Coordinates runs, manages path locking injection, handles fork logic. This is the "brain" that decides whether the agent is running freely or following a locked path.

**Agent Layer** -- The coding agent state machine: Plan -> Code -> Execute -> Evaluate -> (Recover if needed). Each phase can produce decision points.

**Decision Capture Layer** -- Intercepts agent decisions and records them as structured Pydantic models. The `PersistentTracer` writes incrementally to JSONL (crash recovery) and finalizes to JSON + DB.

**Infrastructure Layer** -- LLM abstraction (OpenRouter), subprocess sandbox for safe execution, tiered cache (memory -> Redis -> filesystem), persistent trace storage, and an event bus for real-time observability.

### Key Components

| Component | File | Responsibility |
|-----------|------|---------------|
| `AgentRunner` | `src/agent/runner.py` | State machine orchestrating Plan->Code->Execute->Evaluate->Recover |
| `PromptInjectionLocker` | `src/locking/locker.py` | Three-layer path locking via prompt injection + verification |
| `ForkEngine` | `src/forking/engine.py` | Fork-and-explore: lock upstream, override fork point, free downstream |
| `PathAnalyzer` | `src/analysis/path_analyzer.py` | Cross-run decision trees, Shannon entropy variance, outcome correlations |
| `RunComparator` | `src/analysis/comparator.py` | Side-by-side comparison with variance scoring |
| `PersistentTracer` | `src/tracing/tracer.py` | Incremental JSONL persistence + finalization to JSON/DB |
| `TraceStore` | `src/tracing/store.py` | Load/save traces with filesystem + DB + cache layers |
| `Container` | `src/dependencies.py` | Dependency injection: wires LLM, sandbox, cache, DB |
| `RedisEventPublisher` | `src/streaming/publisher.py` | Publishes agent events to Redis pub/sub for streaming |

## LLM Strategy: Multi-Model Routing via OpenRouter

### Why OpenRouter

OpenRouter provides a single API key and a single bill for 300+ models from every major provider (OpenAI, Anthropic, Google, Meta, DeepSeek). The API is OpenAI-SDK compatible -- switching models means changing a string. This eliminates the need for multiple provider accounts, multiple SDKs, and multiple billing dashboards.

### Per-Phase Model Routing

Different agent phases have fundamentally different requirements. Using a single model for all phases either overpays for simple tasks or underperforms on hard ones. Our strategy assigns the best model for each job:

| Phase | Model | Why This Model | Cost (per 1M tokens) |
|-------|-------|----------------|---------------------|
| **Planning** | `google/gemini-2.5-flash` | Native JSON schema enforcement eliminates parsing retries. Built-in thinking mode for better alternative enumeration. | $0.30 in / $2.50 out |
| **Coding** | `anthropic/claude-sonnet-4` | Highest first-pass success rate on multi-file Python. A model that gets code right on pass 1 saves 3+ retry cycles x 3K tokens each. | $3.00 in / $15.00 out |
| **Evaluation** | `openai/gpt-4o-mini` | This is a classification task ("did exit_code == 0?"), not a reasoning task. GPT-4o-mini's JSON mode is the most reliable at the lowest cost. | $0.15 in / $0.60 out |
| **Recovery** | `anthropic/claude-sonnet-4` | Misdiagnosed errors cascade into wasted cycles. Code comprehension quality justifies the cost. | $3.00 in / $15.00 out |
| **Fallback** | `deepseek/deepseek-chat-v3.2` | Automatic fallback when any primary model returns an API error. Cheap and reliable. | $0.28 in / $0.42 out |

### Cost Comparison

| Strategy | Cost Per Run | Notes |
|----------|-------------|-------|
| Single model (Claude Sonnet 4 for all) | ~$0.34 | Overpays for evaluation and planning |
| Multi-model routing | ~$0.10 | 70% cost reduction, same or better quality |

The ~78% of cost goes to coding (where quality matters most), while evaluation costs <1% of the run.

### Configuration

All routing is controlled via environment variables in `.env`:

```bash
LLM_MODEL_PLANNING=google/gemini-2.5-flash
LLM_MODEL_CODING=anthropic/claude-sonnet-4
LLM_MODEL_EVALUATION=openai/gpt-4o-mini
LLM_MODEL_RECOVERY=anthropic/claude-sonnet-4
LLM_MODEL_FALLBACK=deepseek/deepseek-chat-v3.2

# Override everything to a single model:
# LLM_MODEL_OVERRIDE=anthropic/claude-sonnet-4
```

## Path Locking Strategy

Path locking enables deterministic replay by constraining the LLM to follow previous decisions. The system uses a three-layer approach:

### Layer 1: Prompt Injection

Locked decision values are injected directly into the planning prompt:

```
LOCKED DECISIONS (you MUST follow these exactly):
For "Which stock ticker(s) to analyze?", you MUST choose AAPL.
For "Which anomaly detection method?", you MUST choose IQR.
```

**Why prompt injection over post-hoc filtering:** The LLM must *reason with* the constraint for coherent downstream code. If we let the LLM choose freely and then replace its choice afterward, the generated code would be internally inconsistent (e.g., code that references z-score variables but the decision says "IQR").

### Layer 2: Output Verification

After the LLM responds, the system verifies that each locked decision actually matches the required value. Synonym-aware normalization handles equivalent values (e.g., "IQR" == "interquartile range").

### Layer 3: Divergence Detection

If a locked decision can't be applied (e.g., an API changed and the locked choice is no longer valid), the system raises a `PathLockDivergenceError` rather than silently producing incorrect results.

### Locking Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| **Full lock** | All decisions locked to source run values | Exact replay |
| **Partial lock** | Lock decisions 1-N, free the rest | Controlled experiment |
| **Fork lock** | Lock upstream, override fork point, free downstream | What-if exploration |

### Cross-Run Question Matching

When replaying across different models or runs, decision questions may be phrased differently. The `QuestionMatcher` handles this with three strategies:

1. **Exact match** -- same category + normalized question text
2. **Fuzzy match** -- Jaccard similarity on word tokens (threshold: 0.6)
3. **Sequence fallback** -- match by position in the decision sequence

### Known Limitations

- LLMs may occasionally ignore weak prompts; verification catches this
- Cross-model replay has phrasing differences (mitigated by fuzzy matching)
- Very different models may produce structurally incompatible decision sequences

## Decision Point Structure

Every decision the agent makes is captured as a structured `DecisionPoint`:

```python
class DecisionPoint(BaseModel):
    id: str                          # Unique ID, e.g. "dp-aa6142c5"
    sequence_number: int             # Ordinal position in the run
    category: DecisionCategory       # data_selection, algorithm_selection, etc.
    phase: AgentPhase                # planning, coding, recovering, etc.
    question: str                    # "Which stock ticker(s) to analyze?"
    alternatives: list[Alternative]  # At least 2 options considered
    chosen: str                      # The selected value
    reasoning: str                   # Why this choice was made
    confidence: float                # 0.0-1.0
    locked: bool                     # True if forced by path locking
    cost_usd: float                  # Cost of the LLM call
    tokens_in: int                   # Input tokens used
    tokens_out: int                  # Output tokens used
```

### Decision Categories

| Category | Description | Example |
|----------|-------------|---------|
| `data_selection` | Which data sources and subsets to use | "Which stock ticker?" -> AAPL |
| `algorithm_selection` | Which analytical method to apply | "Which anomaly method?" -> IQR |
| `architecture` | Code structure and patterns | "Single file or modular?" -> class-based |
| `library_selection` | Which libraries to use | "Which data library?" -> pandas |
| `error_recovery` | How to handle failures | "Remove Flask or fix import?" -> remove |
| `parameter_tuning` | Numeric configuration choices | "IQR multiplier?" -> 1.5 |

### Variance Tiers

Decisions are classified by their impact on output variance across runs:

| Tier | Score Range | Description | Strategy |
|------|------------|-------------|----------|
| **HIGH** | 7.5-10 | Fundamentally changes output (algorithm, data source) | Always trace, never auto-pin |
| **MEDIUM** | 5-7 | Changes presentation (output format, visualization) | Trace, consider pinning |
| **LOW** | 2-3.5 | Cosmetic code differences (variable names, structure) | Optional tracing |
| **NOOP** | 1-1.5 | Zero output impact (import order, comment style) | Auto-pin to save tokens |

### Real Example (from recorded run)

```json
{
  "id": "dp-aa6142c5",
  "sequence_number": 0,
  "category": "data_selection",
  "phase": "planning",
  "question": "Which stock ticker(s) to analyze?",
  "alternatives": [
    {
      "value": "AAPL",
      "reasoning": "The task explicitly mentions AAPL, so it's the most direct interpretation."
    },
    {
      "value": "MSFT",
      "reasoning": "Another large-cap tech stock with reliable data."
    },
    {
      "value": "GOOG",
      "reasoning": "Another large-cap tech stock with reliable data."
    }
  ],
  "chosen": "AAPL",
  "reasoning": "The task explicitly specifies 'AAPL stock data', so adhering to the prompt is the primary goal.",
  "confidence": 1.0,
  "locked": false
}
```

## Prompts Used

All prompts live in `src/agent/prompts.py` -- the single source of truth for every string sent to the LLM.

### 1. System Prompt

**Target:** All models (prepended to every LLM call)
**Purpose:** Establishes the agent's role and decision-making requirements

The system prompt instructs the agent to:
- Analyze tasks and make explicit decisions at every branching point
- Consider at least 2 alternatives for each decision
- Output structured JSON for decisions
- Write complete, production-ready Python code
- Categorize decisions using the defined taxonomy

### 2. Planning Prompt (`planning_prompt`)

**Target:** Gemini 2.5 Flash (optimized for structured JSON output)
**Purpose:** Produces a structured plan with explicit decision points

The planning prompt:
- Takes the task description and any locked decisions
- Injects locked decision values directly (prompt injection for path locking)
- Requires JSON output with `decisions[]` and `implementation_plan`
- Each decision must have question, category, alternatives (2+), chosen, reasoning, confidence

**Why structured this way:** Gemini's native JSON enforcement means no parsing retries. The explicit alternatives requirement forces the model into deliberate reasoning mode rather than jumping to a default.

### 3. Coding Prompt (`coding_prompt`)

**Target:** Claude Sonnet 4 (highest first-pass code quality)
**Purpose:** Generates complete, runnable Python code

The coding prompt:
- Provides the task, plan, and decisions summary
- Includes previous error context on retry iterations
- Requires complete files (not pseudocode), a `requirements.txt`, and `main.py` entry point
- Outputs JSON with `files{}`, `requirements[]`, `entry_point`

**Why structured this way:** Claude produces the best multi-file Python on first pass. The explicit "not pseudocode" instruction and file-dict format prevent partial implementations.

### 4. Evaluation Prompt (`evaluation_prompt`)

**Target:** GPT-4o-mini (cheapest reliable JSON classifier)
**Purpose:** Assesses execution results and determines next action

The evaluation prompt:
- Takes task, stdout, stderr, exit code, and file list
- Outputs a simple JSON classification: `status`, `assessment`, `recovery_strategy`, `next_action`
- Next actions: `done` (success), `fix` (fixable error), `retry` (transient), `abort` (impossible)

**Why structured this way:** This is pure classification, not reasoning. GPT-4o-mini's JSON mode gives the most reliable structured output at the lowest cost. Output is truncated to 3000 chars to stay within token limits.

### 5. Recovery Prompt (`recovery_prompt`)

**Target:** Claude Sonnet 4 (best at diagnosing code errors)
**Purpose:** Fixes failing code based on error analysis

The recovery prompt:
- Includes full code context, error details, assessment, and suggested strategy
- Can produce new decisions (category: `error_recovery`) in addition to fixed code
- Returns corrected files in the same format as the coding prompt

**Why structured this way:** Misdiagnosed errors cascade into wasted cycles. Including full code context (not just the error) lets the model understand the broader architecture before making fixes.

### 6. Lock Decision Prompt (`lock_decision_prompt`)

**Target:** Injected into planning prompt
**Purpose:** Forces a specific decision value during path-locked replay

```
For "Which anomaly detection method?", you MUST choose IQR.
Do not consider other options. Set locked=true in your response.
```

## Variance Analysis

### Summary of Recorded Runs

We recorded 5 successful runs on the same task with no path locking. Each run used multi-model routing (Gemini for planning, Sonnet for coding, GPT-4o-mini for evaluation):

| Run ID | LLM Calls | Tokens | Cost | Recovery Needed? |
|--------|-----------|--------|------|-----------------|
| `run-46b79cde` | 9 | 28,383 | $0.24 | Yes (3 recovery cycles) |
| `run-489eef3a` | 3 | 7,011 | $0.05 | No |
| `run-5b59d454` | 6 | 16,003 | $0.15 | Yes (1 recovery cycle) |
| `run-6e8c6eb9` | 3 | 6,446 | $0.05 | No |
| `run-70758e7c` | 6 | 20,175 | $0.17 | Yes (1 recovery cycle) |
| **Total** | **27** | **78,018** | **$0.66** | |

### What Varied

Across the 5 runs, these decisions showed the highest variance (measured by Shannon entropy):

- **Algorithm selection** (anomaly detection method): IQR, Z-score, standard deviation -- HIGH variance
- **Architecture** (program structure): class-based vs. functional vs. single-function -- MEDIUM variance
- **Error recovery** (when it occurred): remove Flask, fix imports, simplify code -- HIGH variance

### What Didn't Vary

- **Data selection** (stock ticker): Always AAPL when explicitly specified -- NOOP, can be auto-pinned
- **Library selection** (data library): Always yfinance + pandas -- LOW variance

### Cost by Phase

The cost breakdown confirms the multi-model routing strategy:

- **Coding** (~78% of cost): Where quality matters most -- Claude Sonnet 4
- **Planning** (~15% of cost): Structured reasoning -- Gemini 2.5 Flash
- **Recovery** (~6% of cost): When needed -- Claude Sonnet 4
- **Evaluation** (<1% of cost): Pure classification -- GPT-4o-mini

This validates that paying premium rates for coding quality while using cheap models for classification is the right tradeoff.

## API Reference

The FastAPI server runs on port 8000 and provides:

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with service status |
| `POST` | `/api/v1/runs` | Start a new agent run (background) |
| `GET` | `/api/v1/runs` | List runs (pagination, status filter) |
| `GET` | `/api/v1/runs/{id}` | Get run metadata |
| `GET` | `/api/v1/runs/{id}/trace` | Get full decision trace |
| `DELETE` | `/api/v1/runs/{id}` | Delete a run |
| `POST` | `/api/v1/runs/{id}/replay` | Replay with locked decisions |
| `POST` | `/api/v1/runs/{id}/fork` | Fork from a decision point |
| `POST` | `/api/v1/compare` | Compare two runs |
| `GET` | `/api/v1/analysis` | Cross-run path analysis |
| `POST` | `/api/v1/estimate` | Estimate cost without running |

### WebSocket Streaming

```
WS /api/v1/runs/{run_id}/stream
```

Streams real-time `AgentEvent` JSON messages during a run (requires Redis).

### Interactive Docs

Visit `http://localhost:8000/docs` for the Swagger UI, or `http://localhost:8000/redoc` for ReDoc.

### Example

```bash
# Start a run
curl -X POST http://localhost:8000/api/v1/runs \
  -H "Content-Type: application/json" \
  -d '{"task": "Analyze stock anomalies"}'

# List runs
curl http://localhost:8000/api/v1/runs

# Compare two runs
curl -X POST http://localhost:8000/api/v1/compare \
  -H "Content-Type: application/json" \
  -d '{"run_a_id": "run-abc", "run_b_id": "run-def"}'
```

## Running Tests

```bash
# Run all tests (inside Docker)
docker-compose exec app pytest tests/ -v

# Run all tests (local)
pytest tests/ -v

# Run specific test suites
pytest tests/unit/test_api.py -v              # API endpoints
pytest tests/unit/test_streaming.py -v        # WebSocket + Redis streaming
pytest tests/unit/test_fork_engine.py -v      # Fork & explore
pytest tests/unit/test_path_analyzer.py -v    # Cross-run analysis
pytest tests/unit/test_locker.py -v           # Path locking
pytest tests/unit/test_models.py -v           # Pydantic models
pytest tests/unit/test_trace_store.py -v      # Trace persistence

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

Current test count: **224 tests** covering decision tracing, path locking correctness, fork semantics, cross-run analysis, API endpoints, streaming, caching, sandbox execution, LLM provider integration, and model parsing.

## Tradeoffs

### Prompt Injection vs. Post-Hoc Filtering vs. Constrained Decoding

We chose **prompt injection** for path locking. Post-hoc filtering (let the LLM choose freely, then replace its answer) would produce internally inconsistent code -- the LLM needs to reason *with* the constraint. Constrained decoding (logit bias) would require provider-specific APIs and doesn't work through OpenRouter. Prompt injection is provider-agnostic and produces coherent downstream reasoning.

### Multi-Model Routing vs. Single Model

Multi-model adds routing complexity but delivers **~70% cost reduction** with same or better quality. The key insight: evaluation is classification (use cheap model), coding needs quality (use best model), planning needs structure (use structured-output model). A single model either overpays for simple tasks or underperforms on hard ones.

### OpenRouter Dependency vs. Direct Provider SDKs

OpenRouter adds a hop of latency (~50-100ms per call) but provides: one API key, one bill, one SDK (OpenAI-compatible), and instant model switching. Direct SDKs would require 4 provider accounts, 4 API keys, and provider-specific code paths. The latency tradeoff is negligible relative to LLM inference time (1-10s).

### Semantic-Level Decisions vs. Fine-Grained vs. Coarse-Grained

We capture decisions at the **semantic level** ("which algorithm?" not "which variable name?"). Fine-grained would capture too many NOOP decisions, inflating trace size without adding replay value. Coarse-grained ("which approach?") would lose the branching points that actually drive variance.

### Filesystem vs. Database for Trace Storage

Traces are stored on the **filesystem** (JSON files) with optional database persistence. Filesystem is zero-config, inspectable with standard tools, and works without infrastructure. The database layer adds structured queries and relational integrity but isn't required for core functionality.

### Subprocess vs. Docker for Sandbox

We use **subprocess** isolation. Docker sandboxing would be more secure but adds startup latency (~2s per execution) and Docker-in-Docker complexity. Subprocess with timeouts is sufficient for the reference task and keeps the development loop fast.

### Temperature 0 is NOT Determinism

Setting `temperature=0.0` does **not** make LLM output deterministic -- it only makes it *more likely* to be similar. Different batching, different hardware, and different model versions can all produce different outputs at temperature 0. True reproducibility requires **decision capture + path locking**, which is exactly what this system provides.

## AI Tool Usage

### What AI Tools Were Used

- **Claude Code (CLI)** -- Used for implementation assistance across all phases
- **Claude (chat)** -- Used for system design discussions and architecture review

### What Was Human-Designed

The following design decisions were made by the human and communicated to AI tools:

- **Multi-model routing strategy** -- The per-phase model selection (Gemini for planning, Sonnet for coding, GPT-4o-mini for evaluation) and the cost-optimization rationale
- **Path locking approach** -- The three-layer strategy (prompt injection -> verification -> divergence detection) and why prompt injection over alternatives
- **Variance tier classification** -- The HIGH/MEDIUM/LOW/NOOP taxonomy and the insight that NOOP decisions can be auto-pinned to save tokens
- **Fork semantics** -- Lock upstream, override at fork point, free downstream
- **Architecture** -- The layered architecture, DI container pattern, protocol-based abstractions
- **System Design Document** -- The full SDD with requirements, component design, and interface specifications

### What Was AI-Assisted

- **Implementation** -- Code generation for each phase, following the SDD and build plan
- **Test writing** -- Unit tests for each component
- **Bug fixes** -- Fixing Python 3.9 compatibility issues, mock behavior in tests, import chain errors
- **README drafting** -- This document was drafted with AI assistance based on human-specified requirements

### What Had to Be Refactored or Fixed

- **Fork semantics** -- Initial CLI implementation locked ALL decisions with one override; refactored to use `lock_through_sequence` for proper upstream-lock/downstream-free behavior
- **Python 3.9 compatibility** -- `match` statements in existing code caused SyntaxError on the development machine; fixed by deferring imports in the API module
- **Mock test behavior** -- MockCodingAgent had a `for...else` bug that caused fork tests to fall through; fixed control flow
- **Variance test assertions** -- Shannon entropy calculations showed stock ticker had equal entropy to algorithm selection; fixed tests to assert on entropy value rather than specific question names

## What I'd Do With More Time

- **Checkpoint caching** -- Cache LLM responses by (prompt_hash, model, temperature) for ~90% token savings on replays
- **Adaptive model routing** -- Start with cheap model, escalate on failure: DeepSeek -> Gemini -> Sonnet cascade
- **Cross-provider replay** -- Semantic normalization layer for replaying a Gemini-planned run with Claude planning
- **Visual decision tree UI** -- React frontend rendering the cross-run decision tree with interactive fork-and-explore
- **A/B testing framework** -- Automated fork-and-compare for systematic exploration of decision alternatives
- **Cost prediction model** -- Train on historical runs to predict cost before execution
- **Parallel fork execution** -- Run multiple forks concurrently and compare results

## Environment Variables

All configuration is done via environment variables. Copy `.env.example` to `.env` and set your OpenRouter API key.

### LLM Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openrouter` | LLM provider. Options: `openrouter`, `openai`, `anthropic` |
| `LLM_API_KEY` | -- | **Required.** Your OpenRouter API key |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Base URL for the LLM API |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature |
| `LLM_MAX_TOKENS` | `4096` | Max output tokens per LLM call |

### Per-Phase Model Routing

| Variable | Default | Phase |
|----------|---------|-------|
| `LLM_MODEL_PLANNING` | `google/gemini-2.5-flash` | Planning |
| `LLM_MODEL_CODING` | `anthropic/claude-sonnet-4` | Coding |
| `LLM_MODEL_EVALUATION` | `openai/gpt-4o-mini` | Evaluation |
| `LLM_MODEL_RECOVERY` | `anthropic/claude-sonnet-4` | Recovery |
| `LLM_MODEL_FALLBACK` | `deepseek/deepseek-chat-v3.2` | Fallback on API error |
| `LLM_MODEL_OVERRIDE` | -- | Override all phases with one model |

### Infrastructure

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://agent:agent@db:5432/deterministic_agent` | PostgreSQL connection |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection (optional) |
| `RUNS_DIR` | `./runs` | Run artifacts directory |
| `AGENT_MAX_ITERATIONS` | `10` | Max code-execute-evaluate loops |
| `AGENT_EXECUTION_TIMEOUT` | `120` | Sandbox timeout (seconds) |
| `AGENT_MAX_COST_USD` | `2.00` | Max cost per run (USD) |
| `API_HOST` | `0.0.0.0` | API server host |
| `API_PORT` | `8000` | API server port |

## License

MIT
