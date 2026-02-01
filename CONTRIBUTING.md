# Contributing to CodeVerify

Thank you for your interest in contributing to CodeVerify! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker & Docker Compose
- Git

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/codeverify.git
   cd codeverify
   ```

2. **Set up the environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   
   docker compose up -d postgres redis
   ```

3. **Install dependencies**
   ```bash
   # Python packages
   pip install -e "packages/core[dev]" \
               -e "packages/verifier[dev]" \
               -e "packages/ai-agents[dev]" \
               -e "apps/api[dev]" \
               -e "apps/worker[dev]"
   
   # Node.js packages
   cd apps/web && npm install && cd ../..
   cd apps/github-app && npm install && cd ../..
   ```

4. **Run migrations**
   ```bash
   cd apps/api && alembic upgrade head && cd ../..
   ```

5. **Validate setup**
   ```bash
   python scripts/validate_env.py
   ```

## 📋 Development Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/updates

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(verifier): add integer overflow detection
fix(api): handle null repository in webhook
docs(readme): update installation instructions
```

### Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Run tests and linting
4. Submit a PR with a clear description
5. Address review feedback
6. Squash and merge when approved

## 🧪 Testing

### Running Tests

```bash
# All tests
./scripts/test.sh

# Specific package
pytest packages/verifier/tests -v

# With coverage
pytest --cov=codeverify --cov-report=html
```

### Writing Tests

- Place tests in `tests/` directory within each package/app
- Use pytest fixtures for shared setup
- Mock external services (GitHub API, LLM APIs)
- Aim for >80% coverage on new code

### Test Structure

```python
"""Tests for feature_name."""
import pytest

class TestFeatureName:
    """Tests for FeatureName class."""
    
    @pytest.fixture
    def setup(self):
        """Common test setup."""
        return SomeFixture()
    
    def test_happy_path(self, setup):
        """Test normal operation."""
        result = do_something()
        assert result == expected
    
    def test_edge_case(self, setup):
        """Test edge case handling."""
        with pytest.raises(ExpectedException):
            do_something_invalid()
```

## 📝 Code Style

### Python

- Follow [PEP 8](https://pep8.org/)
- Use type hints for all functions
- Use `ruff` for linting: `ruff check .`
- Use `black` for formatting: `black .`

```python
from typing import Optional

def process_analysis(
    analysis_id: str,
    config: Optional[dict] = None,
) -> AnalysisResult:
    """Process an analysis job.
    
    Args:
        analysis_id: Unique identifier for the analysis
        config: Optional configuration overrides
        
    Returns:
        The analysis result with findings
        
    Raises:
        AnalysisError: If processing fails
    """
    ...
```

### TypeScript

- Use TypeScript strict mode
- Use ESLint + Prettier
- Prefer functional components for React

```typescript
interface AnalysisProps {
  analysisId: string;
  onComplete?: (result: AnalysisResult) => void;
}

export function AnalysisView({ analysisId, onComplete }: AnalysisProps) {
  // ...
}
```

## 📁 Project Structure

```
codeverify/
├── apps/
│   ├── api/           # FastAPI backend
│   ├── worker/        # Celery workers
│   ├── web/           # Next.js dashboard
│   └── github-app/    # GitHub webhook handler
├── packages/
│   ├── core/          # Shared utilities
│   ├── verifier/      # Z3 verification engine
│   └── ai-agents/     # LLM agents
├── docs/              # Documentation
├── scripts/           # Development scripts
└── tests/             # Integration tests
```

## 🔧 Adding Features

### New Verification Check

1. Add the check to `packages/verifier/src/codeverify_verifier/z3_verifier.py`
2. Add tests in `packages/verifier/tests/`
3. Update config schema in `packages/core/src/codeverify_core/config.py`
4. Document in `docs/verification.md`

### New AI Agent

1. Create agent file in `packages/ai-agents/src/codeverify_agents/`
2. Inherit from `BaseAgent`
3. Implement `analyze()` method
4. Add tests
5. Register in analysis pipeline

### New API Endpoint

1. Create router in `apps/api/src/codeverify_api/routers/`
2. Add to `main.py`
3. Add tests
4. Update API documentation

## 🐛 Reporting Issues

### Bug Reports

Include:
- Description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages

### Feature Requests

Include:
- Description of the feature
- Use case / motivation
- Proposed implementation (if any)
- Alternatives considered

## 🔐 Security

If you discover a security vulnerability:

1. **Do NOT** open a public issue
2. Email security@codeverify.dev with details
3. Allow time for us to respond and fix
4. We'll credit you in our security advisories

## 📚 Documentation

- Update docs when changing user-facing behavior
- Use clear, concise language
- Include code examples where helpful
- Keep the README up to date

## 💬 Communication

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, show & tell

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to CodeVerify! 🎉
