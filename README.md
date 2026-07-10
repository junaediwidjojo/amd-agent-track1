# AMD Hackathon Track 1 — General Purpose AI Agent

Production-quality submission for the AMD Developer Hackathon. A token-efficient, rule-routed AI agent that processes diverse natural language tasks via Fireworks AI.

## Scoring strategy

1. **Accuracy gate** — pass LLM-judge evaluation with specialized handlers per task type
2. **Token efficiency** — minimal prompts, no conversation history, no chain-of-thought, rule-based routing (zero LLM tokens for classification)

## Architecture

```
Task → Rule-based classifier → Specialized handler → Prompt builder → Fireworks API → Post-processor → results.json
```

## Project structure

```
app/
  main.py              # CLI entry (run, benchmark)
  config.py            # Environment-based settings
  agent.py             # Orchestrator
  router.py            # Keyword/heuristic classifier
  fireworks/
    client.py          # API client with retry, fallback, cache
    models.py          # Pydantic models
  handlers/            # One handler per task category
  prompts/             # Minimal system prompts (token-optimized)
  utils/               # I/O, logging, JSON cleanup
tests/                 # pytest suite
Dockerfile             # Multi-stage, linux/amd64
docker-compose.yml     # Local container testing
```

## Task categories

| Category | Router triggers |
|----------|----------------|
| Summarization | summarize, condense, one sentence |
| NER | extract entities, named entity |
| Sentiment | sentiment, classify review |
| Debugging | bug, fix, debug |
| Code generation | write a function, implement |
| Math | calculate, how many, percent |
| Logic | puzzle, who owns, constraints |
| Factual | default fallback |

## Quick start

### 1. Install dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt  # includes pytest/ruff for local dev; the Docker image only installs requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your Fireworks credentials
```

Required variables (injected by hackathon harness at evaluation):

| Variable | Description |
|----------|-------------|
| `FIREWORKS_API_KEY` | API key (provided by harness) |
| `FIREWORKS_BASE_URL` | Base URL for all API calls |
| `ALLOWED_MODELS` | Comma-separated model IDs |

### 3. Run locally (native)

```bash
chmod +x scripts/run_native.sh
./scripts/run_native.sh
```

### 4. Run locally (Docker)

```bash
chmod +x scripts/run_local.sh
./scripts/run_local.sh
```

### 5. Run tests

```bash
pytest -v
ruff check app tests
ruff format app tests
```

## CLI

```bash
# Production mode (reads /input/tasks.json, writes /output/results.json)
python -m app.main run

# Custom paths
python -m app.main run --input ./input/tasks.json --output ./output/results.json

# Benchmark with token/latency metrics
python -m app.main benchmark --input ./input/tasks.json --output ./output/benchmark.json
```

## Docker build and push

Apple Silicon must target `linux/amd64`:

```bash
docker buildx build \
  --platform linux/amd64 \
  --tag ghcr.io/<your-user>/amd-agent:latest \
  --push .
```

## Features

- Rule-based router (no LLM classification tokens)
- 8 specialized handlers with minimal prompts
- Fireworks client with retry, exponential backoff, model fallback
- Prompt cache disabled by default (competition compliance)
- Token counter, latency metrics, cost estimator
- Structured JSON logging
- Graceful per-task error handling (batch never crashes)
- Multi-stage Docker image with bundled 3B Q4 local model (~3 GB compressed)
- Full pytest suite

## Competition constraints

- Exit code 0 on success
- Max runtime: 10 minutes
- Container ready within 60 seconds
- Per-task response under 30 seconds
- Image: linux/amd64, < 10 GB compressed
- No hardcoded answers, no response caching, no hardcoded model IDs
- 10-minute global runtime budget enforced
- All Fireworks calls through `FIREWORKS_BASE_URL`

## Practice tasks

See `input/tasks.json` for the 8 illustrative tasks from the participant guide. Generate additional paraphrased prompts per category to validate generalization before submitting.
