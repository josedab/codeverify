# CodeVerify Worker

Celery-based analysis workers for CodeVerify.

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![Celery](https://img.shields.io/badge/Celery-5.3+-green)](https://docs.celeryq.dev/)

## Overview

The worker service handles:

- Analysis job execution from the Redis queue
- Multi-stage analysis pipeline orchestration
- AI agent coordination
- Z3 formal verification
- Result storage and notification

## Architecture

```mermaid
flowchart TB
    subgraph Queue["📬 Job Queue"]
        Redis[(Redis)]
    end

    subgraph Worker["⚙️ Celery Worker"]
        direction TB
        
        subgraph Tasks["Celery Tasks"]
            AnalyzePR[analyze_pr]
            VerifyCode[verify_code]
            Notify[notify_complete]
        end

        subgraph Pipeline["Analysis Pipeline"]
            direction TB
            P1[1. Fetch PR Data]
            P2[2. Parse Code]
            P3[3. Parallel Analysis]
            P4[4. Synthesize]
            P5[5. Store Results]
            P6[6. Notify]
            
            P1 --> P2 --> P3 --> P4 --> P5 --> P6
        end
        
        subgraph Engines["Analysis Engines"]
            Semantic[Semantic Agent<br/>GPT-4/Claude]
            Security[Security Agent<br/>OWASP/CWE]
            Z3[Z3 Verifier<br/>Formal Proofs]
        end
    end

    subgraph External["🔗 External"]
        GitHub[GitHub API]
        OpenAI[OpenAI API]
        Anthropic[Anthropic API]
        DB[(PostgreSQL)]
    end

    Redis --> Tasks
    Tasks --> Pipeline
    P3 --> Engines
    Engines --> P4
    Pipeline --> GitHub
    Engines --> OpenAI
    Engines --> Anthropic
    P5 --> DB
```

## Quick Start

### Prerequisites

- Python 3.11+
- Redis 7+ (for job queue)
- PostgreSQL 14+ (for results)
- OpenAI/Anthropic API key (for AI analysis)

### Development Setup

```bash
# Install dependencies
pip install -e "apps/worker[dev]"
pip install -e "packages/core"
pip install -e "packages/verifier"
pip install -e "packages/ai-agents"

# Set environment variables
cp .env.example .env

# Start worker
celery -A codeverify_worker.main worker --loglevel=info
```

### Using Docker

```bash
docker compose up worker
```

## Project Structure

```
apps/worker/
├── src/
│   └── codeverify_worker/
│       ├── main.py           # Celery app entry point
│       ├── config.py         # Configuration
│       ├── tasks/            # Celery task definitions
│       │   ├── analyze.py    # Main analysis task
│       │   ├── verify.py     # Z3 verification task
│       │   └── notify.py     # Notification task
│       ├── pipeline/         # Analysis pipeline stages
│       │   ├── parser.py     # Code parsing
│       │   ├── semantic.py   # Semantic analysis
│       │   ├── security.py   # Security analysis
│       │   ├── formal.py     # Formal verification
│       │   └── synthesis.py  # Result synthesis
│       └── utils/
├── tests/
└── pyproject.toml
```

## Analysis Pipeline

The pipeline runs in stages with parallel execution where possible:

```mermaid
sequenceDiagram
    participant Q as Redis Queue
    participant W as Worker
    participant GH as GitHub API
    participant AI as AI Agents
    participant Z3 as Z3 Verifier
    participant DB as PostgreSQL

    Q->>W: Dequeue job
    W->>GH: Fetch PR diff
    GH-->>W: Code changes
    W->>W: Parse with tree-sitter
    
    par Parallel Analysis
        W->>AI: Semantic analysis
        AI-->>W: Intent & contracts
    and
        W->>AI: Security analysis
        AI-->>W: Vulnerabilities
    and
        W->>Z3: Formal verification
        Z3-->>W: Proofs & counterexamples
    end
    
    W->>W: Synthesize findings
    W->>DB: Store results
    W->>GH: Post PR comment
    W->>GH: Update check status
```

## Celery Tasks

### `analyze_pr`

Main PR analysis task:

```python
from codeverify_worker.tasks import analyze_pr

# Queue analysis job
result = analyze_pr.delay(
    repository_id="repo_123",
    pr_number=42,
    head_sha="abc123",
    base_sha="def456",
)

# Wait for result
analysis = result.get(timeout=300)
```

### `verify_code`

Run Z3 verification:

```python
from codeverify_worker.tasks import verify_code

result = verify_code.delay(
    code=code,
    language="python",
    checks=["null_safety", "division_by_zero"],
)
```

### `notify_complete`

Send notifications:

```python
from codeverify_worker.tasks import notify_complete

notify_complete.delay(
    analysis_id="analysis_123",
    channels=["slack", "pr_comment"],
)
```

## Configuration

### Environment Variables

```bash
# Redis (Celery broker)
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/codeverify

# GitHub API
GITHUB_APP_ID=12345
GITHUB_APP_PRIVATE_KEY=-----BEGIN RSA PRIVATE KEY-----...

# AI Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Verification
VERIFICATION_TIMEOUT=30
MAX_FILE_SIZE_KB=500

# Concurrency
CELERY_WORKER_CONCURRENCY=4
```

### Celery Configuration

```python
# codeverify_worker/config.py
from celery import Celery

app = Celery('codeverify')

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    task_time_limit=600,  # 10 minutes max
    task_soft_time_limit=540,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)
```

## Monitoring

### Flower Dashboard

```bash
# Install flower
pip install flower

# Start monitoring dashboard
celery -A codeverify_worker.main flower --port=5555
```

Access at http://localhost:5555

### Prometheus Metrics

The worker exports metrics at `/metrics`:

- `codeverify_tasks_total` - Total tasks processed
- `codeverify_task_duration_seconds` - Task duration histogram
- `codeverify_findings_total` - Findings by severity

## Testing

```bash
# Run tests
pytest apps/worker/tests -v

# Run with coverage
pytest apps/worker/tests --cov=codeverify_worker

# Test specific task
pytest apps/worker/tests/test_analyze.py -v
```

## Scaling

### Horizontal Scaling

```bash
# Start multiple workers
celery -A codeverify_worker.main worker --concurrency=4 -n worker1@%h
celery -A codeverify_worker.main worker --concurrency=4 -n worker2@%h
```

### Queue Prioritization

```python
# High priority queue for paid customers
analyze_pr.apply_async(
    args=[repo_id, pr_number],
    queue='high_priority',
)

# Background queue for full scans
scan_repository.apply_async(
    args=[repo_id],
    queue='background',
)
```

## Development

### Adding a New Analysis Stage

1. Create stage in `src/codeverify_worker/pipeline/`
2. Implement `analyze()` method returning findings
3. Register in pipeline configuration
4. Add tests in `tests/`

### Code Style

```bash
black apps/worker
ruff check apps/worker
mypy apps/worker/src
```
