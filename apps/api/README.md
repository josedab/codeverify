# CodeVerify API Service

FastAPI backend service for CodeVerify.

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com/)

## Overview

The API service provides:

- REST API for the web dashboard
- Authentication via GitHub OAuth and JWT
- Analysis results retrieval and management
- Organization and repository configuration
- Webhooks and API key management

## Architecture

```mermaid
flowchart TB
    subgraph Clients["🖥️ Clients"]
        Web[Web Dashboard]
        CLI[CLI Tool]
        External[External Apps]
    end

    subgraph API["⚡ FastAPI Service"]
        direction TB
        Auth[Auth Middleware<br/>JWT/OAuth]
        Rate[Rate Limiter]
        
        subgraph Routers["Routers"]
            AuthR[/auth]
            AnalysesR[/analyses]
            ReposR[/repositories]
            OrgsR[/organizations]
            StatsR[/stats]
            WebhooksR[/webhooks]
        end
        
        subgraph Services["Services"]
            AnalysisSvc[Analysis Service]
            AuthSvc[Auth Service]
            StatsSvc[Stats Service]
        end
    end

    subgraph Data["💾 Data Layer"]
        PG[(PostgreSQL)]
        Redis[(Redis Cache)]
    end

    Clients --> Auth
    Auth --> Rate
    Rate --> Routers
    Routers --> Services
    Services --> PG
    Services --> Redis
```

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Redis 7+

### Development Setup

```bash
# Install dependencies
pip install -e "apps/api[dev]"

# Set environment variables
cp .env.example .env
# Edit .env with your settings

# Run database migrations
cd apps/api && alembic upgrade head

# Start the server
uvicorn codeverify_api.main:app --reload --port 8000
```

### Using Docker

```bash
docker compose up api
```

## Project Structure

```
apps/api/
├── alembic/           # Database migrations
│   └── versions/      # Migration scripts
├── src/
│   └── codeverify_api/
│       ├── main.py        # FastAPI app entry point
│       ├── config.py      # Configuration
│       ├── database.py    # SQLAlchemy setup
│       ├── models/        # SQLAlchemy ORM models
│       ├── routers/       # API route handlers
│       │   ├── analyses.py
│       │   ├── auth.py
│       │   ├── organizations.py
│       │   ├── repositories.py
│       │   ├── stats.py
│       │   └── webhooks.py
│       ├── schemas/       # Pydantic schemas
│       ├── services/      # Business logic
│       └── middleware/    # Auth, rate limiting
├── tests/
└── pyproject.toml
```

## API Endpoints

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/auth/github` | Start GitHub OAuth flow |
| POST | `/auth/token` | Exchange code for token |
| POST | `/auth/refresh` | Refresh access token |
| POST | `/auth/logout` | Invalidate token |

### Analyses

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/analyses` | List analyses |
| GET | `/analyses/{id}` | Get analysis details |
| GET | `/analyses/{id}/findings` | Get analysis findings |
| POST | `/analyses/{id}/retry` | Retry failed analysis |
| POST | `/analyses/{id}/cancel` | Cancel running analysis |

### Organizations & Repositories

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/organizations` | List organizations |
| GET | `/organizations/{id}` | Get organization |
| GET | `/repositories` | List repositories |
| PATCH | `/repositories/{id}` | Update repository config |

### Statistics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/stats/dashboard` | Dashboard statistics |
| GET | `/stats/trends` | Historical trends |

See [API Reference](../../docs/api-reference.md) for complete documentation.

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/codeverify

# Redis
REDIS_URL=redis://localhost:6379/0

# Authentication
JWT_SECRET=your-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# GitHub OAuth
GITHUB_CLIENT_ID=your-client-id
GITHUB_CLIENT_SECRET=your-client-secret

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# CORS
CORS_ORIGINS=http://localhost:3000,https://dashboard.codeverify.io
```

## Database

### Running Migrations

```bash
# Apply all migrations
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "description"

# Rollback one migration
alembic downgrade -1
```

### Models

- `Organization` - GitHub organizations
- `Repository` - Repositories with config
- `Analysis` - PR analysis records
- `Finding` - Individual findings
- `User` - Authenticated users
- `ApiKey` - API keys for programmatic access
- `Webhook` - Webhook subscriptions

## Testing

```bash
# Run all tests
pytest apps/api/tests -v

# Run with coverage
pytest apps/api/tests --cov=codeverify_api

# Run specific test file
pytest apps/api/tests/test_analyses.py -v
```

## Development

### Adding a New Endpoint

1. Create router in `src/codeverify_api/routers/`
2. Define Pydantic schemas in `src/codeverify_api/schemas/`
3. Implement service logic in `src/codeverify_api/services/`
4. Register router in `main.py`
5. Add tests in `tests/`
6. Update API documentation

### Code Style

```bash
# Format code
black apps/api

# Lint
ruff check apps/api

# Type check
mypy apps/api/src
```

## Deployment

See [Deployment Guide](../../deploy/README.md) for production deployment instructions.
