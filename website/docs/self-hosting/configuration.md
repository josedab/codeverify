---
sidebar_position: 4
---

# Self-Hosting Configuration

Complete configuration reference for self-hosted deployments.

## Environment Variables

### Core Settings

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SECRET_KEY` | Yes | - | Application secret for encryption |
| `DATABASE_URL` | Yes | - | PostgreSQL connection string |
| `REDIS_URL` | Yes | - | Redis connection string |
| `API_URL` | No | `http://localhost:8000` | Internal API URL |
| `PUBLIC_URL` | No | `http://localhost:3000` | Public-facing URL |

### Database

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | Yes | - | Full connection string |
| `DB_POOL_SIZE` | No | `10` | Connection pool size |
| `DB_MAX_OVERFLOW` | No | `20` | Max overflow connections |
| `DB_POOL_TIMEOUT` | No | `30` | Pool timeout in seconds |

Format:
```
postgresql://user:password@host:5432/database
```

### Redis

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REDIS_URL` | Yes | - | Full connection string |
| `REDIS_MAX_CONNECTIONS` | No | `50` | Max connections |

Format:
```
redis://:password@host:6379/0
```

### AI Providers

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | No | - | OpenAI API key |
| `OPENAI_ORG_ID` | No | - | OpenAI organization ID |
| `ANTHROPIC_API_KEY` | No | - | Anthropic API key |
| `AI_PROVIDER` | No | `openai` | Default AI provider |
| `AI_MODEL` | No | `gpt-4` | Default AI model |

### GitHub Integration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GITHUB_APP_ID` | No | - | GitHub App ID |
| `GITHUB_APP_PRIVATE_KEY` | No | - | Private key (PEM format) |
| `GITHUB_WEBHOOK_SECRET` | No | - | Webhook secret |
| `GITHUB_CLIENT_ID` | No | - | OAuth client ID |
| `GITHUB_CLIENT_SECRET` | No | - | OAuth client secret |

### GitLab Integration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GITLAB_URL` | No | `https://gitlab.com` | GitLab instance URL |
| `GITLAB_APP_ID` | No | - | Application ID |
| `GITLAB_APP_SECRET` | No | - | Application secret |

### Authentication

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AUTH_SECRET` | Yes | - | JWT signing secret |
| `AUTH_ALGORITHM` | No | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE` | No | `3600` | Token expiry (seconds) |
| `REFRESH_TOKEN_EXPIRE` | No | `604800` | Refresh expiry |

### Worker Settings

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CELERY_CONCURRENCY` | No | `4` | Worker concurrency |
| `CELERY_PREFETCH` | No | `1` | Task prefetch multiplier |
| `ANALYSIS_TIMEOUT` | No | `300` | Max analysis time (seconds) |
| `Z3_TIMEOUT` | No | `30` | Z3 solver timeout |

### Logging

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LOG_LEVEL` | No | `INFO` | Log level |
| `LOG_FORMAT` | No | `text` | Log format (text/json) |
| `SENTRY_DSN` | No | - | Sentry error tracking |

## Configuration File

Create `config.yaml` for complex configuration:

```yaml
# config.yaml
server:
  host: 0.0.0.0
  port: 8000
  workers: 4
  
database:
  url: ${DATABASE_URL}
  pool_size: 10
  max_overflow: 20
  echo: false

redis:
  url: ${REDIS_URL}
  max_connections: 50

ai:
  default_provider: openai
  providers:
    openai:
      api_key: ${OPENAI_API_KEY}
      model: gpt-4
      max_tokens: 4096
    anthropic:
      api_key: ${ANTHROPIC_API_KEY}
      model: claude-3-opus

verification:
  timeout: 300
  z3_timeout: 30
  max_workers: 4
  checks:
    - null_safety
    - array_bounds
    - integer_overflow
    - division_by_zero

auth:
  secret: ${AUTH_SECRET}
  access_token_expire: 3600
  refresh_token_expire: 604800
  
github:
  app_id: ${GITHUB_APP_ID}
  private_key: ${GITHUB_APP_PRIVATE_KEY}
  webhook_secret: ${GITHUB_WEBHOOK_SECRET}

logging:
  level: INFO
  format: json
```

Mount in Docker:

```yaml
services:
  api:
    volumes:
      - ./config.yaml:/app/config.yaml
    environment:
      - CONFIG_FILE=/app/config.yaml
```

## Database Configuration

### Connection String

```
postgresql://[user]:[password]@[host]:[port]/[database]?[options]
```

Options:
- `sslmode=require` — Require SSL
- `connect_timeout=10` — Connection timeout

Example:
```
postgresql://codeverify:secret@db.example.com:5432/codeverify?sslmode=require
```

### SSL Configuration

```yaml
database:
  url: postgresql://...
  ssl:
    mode: require
    ca: /path/to/ca.pem
```

### Read Replicas

```yaml
database:
  primary:
    url: postgresql://primary-host/codeverify
  replicas:
    - url: postgresql://replica-1/codeverify
    - url: postgresql://replica-2/codeverify
```

## Redis Configuration

### Sentinel (High Availability)

```yaml
redis:
  sentinel:
    master: mymaster
    sentinels:
      - host: sentinel-1
        port: 26379
      - host: sentinel-2
        port: 26379
      - host: sentinel-3
        port: 26379
    password: ${REDIS_PASSWORD}
```

### Cluster

```yaml
redis:
  cluster:
    nodes:
      - host: redis-1
        port: 6379
      - host: redis-2
        port: 6379
      - host: redis-3
        port: 6379
```

## Storage Configuration

### Local Storage

```yaml
storage:
  type: local
  path: /data/codeverify
```

### S3

```yaml
storage:
  type: s3
  bucket: codeverify-storage
  region: us-east-1
  access_key: ${AWS_ACCESS_KEY_ID}
  secret_key: ${AWS_SECRET_ACCESS_KEY}
```

### Google Cloud Storage

```yaml
storage:
  type: gcs
  bucket: codeverify-storage
  credentials: /path/to/service-account.json
```

## Email Configuration

```yaml
email:
  enabled: true
  smtp:
    host: smtp.example.com
    port: 587
    username: ${SMTP_USER}
    password: ${SMTP_PASSWORD}
    tls: true
  from:
    name: CodeVerify
    email: noreply@yourcompany.com
```

## Rate Limiting

```yaml
rate_limiting:
  enabled: true
  
  # Default limits
  default:
    requests_per_minute: 60
    requests_per_hour: 1000
  
  # Per-endpoint limits
  endpoints:
    /api/v1/analyses:
      requests_per_minute: 10
    /api/v1/findings:
      requests_per_minute: 100
```

## Security Settings

```yaml
security:
  # Allowed origins for CORS
  cors_origins:
    - https://codeverify.yourcompany.com
    - https://dashboard.yourcompany.com
  
  # Content Security Policy
  csp:
    default_src: ["'self'"]
    script_src: ["'self'"]
    
  # Rate limiting on auth endpoints
  auth_rate_limit:
    login_attempts: 5
    lockout_duration: 900
```

## Feature Flags

```yaml
features:
  ai_analysis: true
  team_learning: true
  proof_carrying_prs: false
  semantic_diff: true
```

## Monitoring Configuration

```yaml
monitoring:
  # Prometheus metrics
  prometheus:
    enabled: true
    path: /metrics
    
  # Health checks
  health:
    enabled: true
    path: /health
    
  # Sentry error tracking
  sentry:
    dsn: ${SENTRY_DSN}
    environment: production
    traces_sample_rate: 0.1
```

## Environment-Specific Files

```
config/
├── base.yaml          # Shared settings
├── development.yaml   # Dev overrides
├── staging.yaml       # Staging overrides
└── production.yaml    # Production overrides
```

Set active environment:
```bash
export CODEVERIFY_ENV=production
```

## Validating Configuration

```bash
# Validate config
codeverify config validate

# Show effective config
codeverify config show

# Test database connection
codeverify config test-db

# Test Redis connection
codeverify config test-redis
```

## Next Steps

- [Docker Deployment](/docs/self-hosting/docker) — Docker setup
- [Kubernetes Deployment](/docs/self-hosting/kubernetes) — Kubernetes setup
- [Troubleshooting](/docs/resources/troubleshooting) — Common issues
