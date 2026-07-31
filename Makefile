# Prefer python3.12 or python3.11 over bare python3
PYTHON := $(shell command -v python3.12 2>/dev/null || command -v python3.11 2>/dev/null || echo python3)

.PHONY: setup dev dev-api test test-core test-fast lint format docker-up docker-down validate clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Full development environment setup
	@echo "🚀 Setting up CodeVerify development environment..."
	@$(PYTHON) -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null \
		|| { echo "❌ Python 3.11+ required. Found: $$($(PYTHON) --version 2>&1). Install with: brew install python@3.12"; exit 1; }
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -e "packages/core[dev]" \
	                                   -e "packages/verifier[dev]" \
	                                   -e "packages/ai-agents[dev]" \
	                                   -e "packages/lsp-server[dev]" \
	                                   -e "packages/cli[dev]" \
	                                   -e "packages/sdk-python[pytest]" \
	                                   -e "packages/z3-mcp" \
	                                   -e "apps/api[dev]" \
	                                   -e "apps/worker[dev]"
	cd apps/github-app && npm install
	cd apps/web && npm install
	. .venv/bin/activate && pip install pre-commit && pre-commit install
	@test -f .env || (cp .env.minimal .env && echo "⚠️  Created .env from minimal template — for AI features, see .env.example")
	@echo "✅ Setup complete! Run 'source .venv/bin/activate' then 'make dev'"

dev: docker-up migrate ## Start all services for local development
	@echo "Starting services (use Ctrl+C to stop)..."
	@echo "  API:        http://localhost:8000"
	@echo "  Dashboard:  http://localhost:3000"
	@echo "  GitHub App: http://localhost:3001"
	@trap 'kill 0' EXIT; \
	. .venv/bin/activate && uvicorn codeverify_api.main:app --reload --port 8000 & \
	. .venv/bin/activate && celery -A codeverify_worker.main worker --loglevel=info & \
	cd apps/web && npm run dev & \
	cd apps/github-app && npm run dev & \
	wait

dev-api: docker-up ## Start just the API server (no web/GitHub app)
	. .venv/bin/activate && uvicorn codeverify_api.main:app --reload --port 8000

test: ## Run all tests (requires running infrastructure)
	. .venv/bin/activate && pytest packages/ apps/ tests/ -v --tb=short

test-core: ## Run core package tests only (no infrastructure needed)
	. .venv/bin/activate && pytest packages/core/ -v --tb=short

test-fast: ## Run all package tests (no app/infrastructure tests)
	. .venv/bin/activate && pytest packages/ -v --tb=short

test-coverage: ## Run tests with coverage report
	. .venv/bin/activate && pytest packages/ apps/ tests/ --cov --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

lint: ## Run linters (ruff + mypy)
	. .venv/bin/activate && ruff check .
	. .venv/bin/activate && ruff format --check .
	. .venv/bin/activate && mypy --strict packages/core/src packages/verifier/src

format: ## Auto-format code
	. .venv/bin/activate && ruff check --fix .
	. .venv/bin/activate && ruff format .

docker-up: ## Start infrastructure (PostgreSQL + Redis)
	docker compose up -d postgres redis
	@echo "⏳ Waiting for PostgreSQL..."
	@for i in 1 2 3 4 5 6 7 8 9 10; do \
		docker compose exec -T postgres pg_isready -U codeverify >/dev/null 2>&1 && break; \
		sleep 1; \
	done
	@docker compose exec -T postgres pg_isready -U codeverify >/dev/null 2>&1 \
		&& echo "✅ PostgreSQL ready" \
		|| echo "⚠️  PostgreSQL not ready yet — check: docker compose logs postgres"
	@docker compose exec -T redis redis-cli ping >/dev/null 2>&1 \
		&& echo "✅ Redis ready" \
		|| echo "⚠️  Redis not ready yet — check: docker compose logs redis"

docker-down: ## Stop all Docker services
	docker compose down

docker-all: ## Start all services via Docker Compose
	docker compose up -d

validate: ## Validate environment configuration
	. .venv/bin/activate && python scripts/validate_env.py

migrate: ## Run database migrations
	. .venv/bin/activate && cd apps/api && alembic upgrade head

clean: ## Remove build artifacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml
	@echo "✅ Cleaned"
