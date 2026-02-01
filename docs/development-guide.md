# CodeVerify Development Guide

This guide covers everything you need to know to contribute to CodeVerify.

## Prerequisites

- **Python 3.11+** - Core backend services
- **Node.js 20+** - GitHub App and web dashboard
- **Docker & Docker Compose** - Local development environment
- **Git** - Version control

## Repository Structure

```
codeverify/
├── apps/
│   ├── api/              # FastAPI REST API
│   ├── worker/           # Celery analysis workers
│   ├── web/              # Next.js dashboard
│   └── github-app/       # GitHub webhook handler
├── packages/
│   ├── core/             # Shared models, config, rules
│   ├── verifier/         # Z3 formal verification
│   ├── ai-agents/        # LLM-powered agents
│   ├── cli/              # Command-line interface
│   ├── z3-mcp/           # Z3 MCP server
│   └── vscode-extension/ # VS Code extension
├── docker/               # Dockerfile configurations
├── docs/                 # Documentation
├── scripts/              # Development utilities
└── tests/                # Integration tests
```

## Quick Setup

### 1. Clone and Configure

```bash
git clone https://github.com/codeverify/codeverify.git
cd codeverify

# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# Required: DATABASE_URL, REDIS_URL, OPENAI_API_KEY
```

### 2. Start Infrastructure

```bash
# Start PostgreSQL and Redis
docker compose up -d postgres redis

# Verify services are running
docker compose ps
```

### 3. Install Dependencies

```bash
# Python packages (using virtual environment)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install all packages in development mode
pip install -e "packages/core[dev]" \
            -e "packages/verifier[dev]" \
            -e "packages/ai-agents[dev]" \
            -e "packages/cli" \
            -e "apps/api[dev]" \
            -e "apps/worker[dev]"

# Node.js packages
cd apps/web && npm install && cd ../..
cd apps/github-app && npm install && cd ../..
```

### 4. Initialize Database

```bash
cd apps/api
alembic upgrade head
cd ../..
```

### 5. Validate Setup

```bash
python scripts/validate_env.py
```

## Running Services

### Development Mode (Individual Services)

```bash
# Terminal 1: API Server
cd apps/api
uvicorn codeverify_api.main:app --reload --port 8000

# Terminal 2: Celery Worker
cd apps/worker
celery -A codeverify_worker.main worker --loglevel=info

# Terminal 3: Web Dashboard
cd apps/web
npm run dev

# Terminal 4: GitHub App
cd apps/github-app
npm run dev
```

### Docker Compose (All Services)

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f api worker

# Stop all services
docker compose down
```

### Service URLs

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | REST API |
| API Docs | http://localhost:8000/docs | OpenAPI/Swagger |
| Dashboard | http://localhost:3000 | Web UI |
| GitHub App | http://localhost:3001 | Webhook handler |

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow the coding standards below and make your changes.

### 3. Run Tests

```bash
# All tests
./scripts/test.sh

# Specific package
pytest packages/verifier/tests -v

# With coverage
pytest --cov=codeverify --cov-report=html
```

### 4. Run Linters

```bash
# Python
ruff check .
ruff format .
mypy packages/

# TypeScript
cd apps/web && npm run lint
cd apps/github-app && npm run lint
```

### 5. Submit PR

```bash
git add .
git commit -m "feat: description of your feature"
git push origin feature/your-feature-name
```

## Coding Standards

### Python Style

```python
"""Module docstring describing the purpose."""

from typing import Any
from dataclasses import dataclass

from codeverify_core.models import Finding


@dataclass
class MyClass:
    """Class docstring.
    
    Attributes:
        name: The name of the thing.
        value: The numeric value.
    """
    
    name: str
    value: int
    
    def process(self, data: dict[str, Any]) -> list[Finding]:
        """Process data and return findings.
        
        Args:
            data: Input data dictionary.
            
        Returns:
            List of findings from analysis.
            
        Raises:
            ValueError: If data is invalid.
        """
        if not data:
            raise ValueError("Data cannot be empty")
        
        # Implementation
        return []
```

### TypeScript Style

```typescript
/**
 * Process analysis results and format for display.
 * @param results - Raw analysis results from API
 * @returns Formatted results for UI rendering
 */
export function formatResults(results: AnalysisResult[]): FormattedResult[] {
  return results.map((result) => ({
    id: result.id,
    title: result.finding.title,
    severity: formatSeverity(result.finding.severity),
  }));
}
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add trust score calculation
fix: resolve null pointer in Z3 verifier
docs: update API reference
refactor: extract common agent logic
test: add tests for security agent
chore: update dependencies
```

## Testing

### Unit Tests

```python
# packages/verifier/tests/test_z3_verifier.py
import pytest
from codeverify_verifier import Z3Verifier


class TestZ3Verifier:
    """Tests for Z3Verifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.verifier = Z3Verifier()
    
    def test_division_by_zero_detection(self):
        """Should detect division by zero vulnerability."""
        code = "def divide(a, b): return a / b"
        
        result = self.verifier.verify_function(
            code=code,
            function_name="divide",
            checks=["division_by_zero"],
        )
        
        assert result.status == "vulnerable"
        assert result.counterexample["b"] == 0
    
    def test_safe_division(self):
        """Should verify safe division with guard."""
        code = """
def divide(a, b):
    if b == 0:
        return 0
    return a / b
"""
        
        result = self.verifier.verify_function(
            code=code,
            function_name="divide",
            checks=["division_by_zero"],
        )
        
        assert result.status == "safe"
```

### Integration Tests

```python
# tests/integration/test_analysis_pipeline.py
import pytest
from httpx import AsyncClient


@pytest.mark.integration
async def test_full_analysis_pipeline(client: AsyncClient):
    """Test complete analysis flow."""
    # Trigger analysis
    response = await client.post(
        "/api/v1/analyses",
        json={"repo": "test/repo", "ref": "main"},
    )
    assert response.status_code == 201
    analysis_id = response.json()["id"]
    
    # Wait for completion
    # ... (poll for status)
    
    # Verify results
    response = await client.get(f"/api/v1/analyses/{analysis_id}")
    assert response.json()["status"] == "completed"
```

### Mocking LLM Calls

```python
# Use CODEVERIFY_MOCK_LLM=true for tests
import os
os.environ["CODEVERIFY_MOCK_LLM"] = "true"

# Or use pytest fixture
@pytest.fixture
def mock_llm(monkeypatch):
    """Mock LLM responses for testing."""
    monkeypatch.setenv("CODEVERIFY_MOCK_LLM", "true")
```

## Adding New Features

### Adding a Verification Check

1. **Implement the check** in `packages/verifier/src/codeverify_verifier/z3_verifier.py`:

```python
def check_new_property(self, code: str, **kwargs) -> VerificationResult:
    """Check for new property violation.
    
    Args:
        code: Source code to verify.
        
    Returns:
        Verification result with findings.
    """
    # Parse code to AST
    parsed = self.parser.parse(code)
    
    # Generate Z3 constraints
    constraints = self._generate_constraints(parsed)
    
    # Check satisfiability
    solver = Solver()
    solver.add(constraints)
    
    if solver.check() == sat:
        return VerificationResult(
            status="vulnerable",
            counterexample=self._extract_model(solver.model()),
        )
    
    return VerificationResult(status="safe")
```

2. **Add tests** in `packages/verifier/tests/test_z3_verifier.py`

3. **Update config schema** in `packages/core/src/codeverify_core/config.py`

4. **Document** in `docs/verification.md`

### Adding a New AI Agent

1. **Create agent** in `packages/ai-agents/src/codeverify_agents/`:

```python
# my_agent.py
from .base import BaseAgent, AgentConfig, AgentResult


class MyAgent(BaseAgent):
    """Agent for specialized analysis."""
    
    async def analyze(self, code: str, **kwargs) -> AgentResult:
        prompt = self._build_prompt(code, kwargs)
        response = await self._call_llm(prompt)
        findings = self._parse_response(response)
        
        return AgentResult(
            success=True,
            data={"findings": findings},
        )
```

2. **Export** in `packages/ai-agents/src/codeverify_agents/__init__.py`

3. **Add tests** in `packages/ai-agents/tests/`

4. **Integrate** into the analysis pipeline

### Adding an API Endpoint

1. **Create router** in `apps/api/src/codeverify_api/routers/`:

```python
# my_feature.py
from fastapi import APIRouter, Depends
from ..auth.dependencies import get_current_user
from ..services.my_service import MyService

router = APIRouter(prefix="/my-feature", tags=["My Feature"])


@router.post("/")
async def create_something(
    data: CreateRequest,
    user: User = Depends(get_current_user),
    service: MyService = Depends(),
):
    """Create a new something."""
    return await service.create(data, user)
```

2. **Register router** in `apps/api/src/codeverify_api/main.py`

3. **Add tests** in `apps/api/tests/`

4. **Document** in `docs/api/PUBLIC_API.md`

## Database Migrations

### Creating a Migration

```bash
cd apps/api
alembic revision --autogenerate -m "Add new_table"
```

### Review and Edit

Edit the generated migration file in `apps/api/alembic/versions/`.

### Apply Migration

```bash
alembic upgrade head
```

### Rollback

```bash
alembic downgrade -1  # One step back
alembic downgrade base  # Reset to beginning
```

## Debugging

### API Debugging

```bash
# Enable debug mode
uvicorn codeverify_api.main:app --reload --log-level debug

# Or set environment variable
DEBUG=true uvicorn codeverify_api.main:app --reload
```

### Worker Debugging

```bash
# Verbose logging
celery -A codeverify_worker.main worker --loglevel=debug

# Single task execution
celery -A codeverify_worker.main call codeverify_worker.tasks.analyze_pr --args='["owner/repo", 42]'
```

### Z3 Debugging

```python
from codeverify_verifier.debugger import VerificationDebugger

debugger = VerificationDebugger()
session = debugger.create_session(code, "my_function")

# Step through verification
while session.has_next():
    step = session.step()
    print(f"Step: {step.description}")
    print(f"Constraints: {step.constraints}")
    print(f"Status: {step.status}")
```

### VS Code Debugging

1. Install Python extension
2. Use provided `.vscode/launch.json`:

```json
{
  "configurations": [
    {
      "name": "API",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": ["codeverify_api.main:app", "--reload"],
      "cwd": "${workspaceFolder}/apps/api"
    },
    {
      "name": "Worker",
      "type": "python",
      "request": "launch",
      "module": "celery",
      "args": ["-A", "codeverify_worker.main", "worker", "--loglevel=debug"],
      "cwd": "${workspaceFolder}/apps/worker"
    }
  ]
}
```

## Performance Profiling

### API Profiling

```python
# Enable profiling middleware
from pyinstrument import Profiler

@app.middleware("http")
async def profile_request(request: Request, call_next):
    profiler = Profiler()
    profiler.start()
    response = await call_next(request)
    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))
    return response
```

### Memory Profiling

```bash
# Install memory profiler
pip install memory_profiler

# Run with profiling
mprof run python -m codeverify_cli analyze src/

# Generate report
mprof plot
```

## Troubleshooting

### Common Issues

#### "Z3 not found"

```bash
pip install z3-solver
```

#### "Redis connection refused"

```bash
docker compose up -d redis
```

#### "Database migration failed"

```bash
# Reset database
docker compose down -v
docker compose up -d postgres
cd apps/api && alembic upgrade head
```

#### "LLM API rate limited"

- Check your API key quota
- Implement retry with exponential backoff
- Use `CODEVERIFY_MOCK_LLM=true` for testing

### Getting Help

- **Documentation:** `/docs` directory
- **Issues:** GitHub Issues
- **Discussions:** GitHub Discussions
- **Slack:** `#codeverify` channel

## Release Process

### Version Bumping

```bash
# Bump version
./scripts/bump-version.sh patch  # or minor, major
```

### Creating a Release

1. Update `CHANGELOG.md`
2. Create version tag: `git tag v0.3.0`
3. Push tag: `git push origin v0.3.0`
4. GitHub Actions handles the rest

### Package Publishing

- **PyPI:** Automatic on tag push
- **npm:** Automatic on tag push
- **Docker Hub:** Automatic on tag push
