---
sidebar_position: 2
---

# Docker Deployment

Deploy CodeVerify using Docker Compose.

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 8 GB RAM minimum
- 50 GB disk space

## Quick Start

### 1. Clone and Configure

```bash
git clone https://github.com/codeverify/codeverify.git
cd codeverify

# Copy example environment
cp .env.example .env
```

### 2. Edit Configuration

```bash
# .env
# Required settings
POSTGRES_PASSWORD=secure-password-here
REDIS_PASSWORD=secure-redis-password
SECRET_KEY=generate-a-random-key

# Optional: AI providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional: GitHub App
GITHUB_APP_ID=12345
GITHUB_APP_PRIVATE_KEY_PATH=/app/github-app.pem
```

### 3. Start Services

```bash
docker compose up -d
```

### 4. Check Status

```bash
docker compose ps
```

Expected output:
```
NAME                    STATUS          PORTS
codeverify-api-1        Up (healthy)    0.0.0.0:8000->8000/tcp
codeverify-dashboard-1  Up (healthy)    0.0.0.0:3000->3000/tcp
codeverify-worker-1     Up (healthy)    
codeverify-postgres-1   Up (healthy)    5432/tcp
codeverify-redis-1      Up (healthy)    6379/tcp
```

### 5. Access Dashboard

Open http://localhost:3000 and create your admin account.

## Docker Compose File

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    image: codeverify/api:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/codeverify
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - SECRET_KEY=${SECRET_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  dashboard:
    image: codeverify/dashboard:latest
    ports:
      - "3000:3000"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api

  worker:
    image: codeverify/worker:latest
    environment:
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/codeverify
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - redis
      - postgres
    deploy:
      replicas: 2

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=codeverify
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
  redis_data:
```

## Scaling Workers

Increase worker count for faster analysis:

```bash
docker compose up -d --scale worker=4
```

## Configuration Options

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `POSTGRES_PASSWORD` | Yes | Database password |
| `REDIS_PASSWORD` | Yes | Redis password |
| `SECRET_KEY` | Yes | Application secret |
| `OPENAI_API_KEY` | No | OpenAI API key |
| `ANTHROPIC_API_KEY` | No | Anthropic API key |
| `GITHUB_APP_ID` | No | GitHub App ID |
| `GITHUB_APP_PRIVATE_KEY` | No | GitHub App private key |

### Persistent Storage

Data is stored in Docker volumes:
- `postgres_data` — Database
- `redis_data` — Queue data

Backup volumes:
```bash
docker run --rm -v codeverify_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres-backup.tar.gz /data
```

## HTTPS Setup

### With Nginx Reverse Proxy

```yaml
# docker-compose.override.yml
services:
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs
    depends_on:
      - api
      - dashboard
```

```nginx
# nginx.conf
events { }

http {
    server {
        listen 443 ssl;
        server_name codeverify.yourcompany.com;
        
        ssl_certificate /etc/nginx/certs/fullchain.pem;
        ssl_certificate_key /etc/nginx/certs/privkey.pem;
        
        location / {
            proxy_pass http://dashboard:3000;
        }
        
        location /api {
            proxy_pass http://api:8000;
        }
    }
}
```

### With Traefik

```yaml
# docker-compose.override.yml
services:
  traefik:
    image: traefik:v2.10
    command:
      - "--providers.docker"
      - "--entrypoints.websecure.address=:443"
      - "--certificatesresolvers.le.acme.email=admin@yourcompany.com"
      - "--certificatesresolvers.le.acme.storage=/letsencrypt/acme.json"
      - "--certificatesresolvers.le.acme.httpchallenge.entrypoint=web"
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - letsencrypt:/letsencrypt

  dashboard:
    labels:
      - "traefik.http.routers.dashboard.rule=Host(`codeverify.yourcompany.com`)"
      - "traefik.http.routers.dashboard.tls.certresolver=le"
```

## Upgrading

### Check for Updates

```bash
docker compose pull
```

### Apply Updates

```bash
docker compose down
docker compose up -d
```

### Database Migrations

Migrations run automatically on startup. For manual control:

```bash
docker compose exec api alembic upgrade head
```

## Monitoring

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f worker
```

### Health Checks

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "services": {
    "database": "connected",
    "redis": "connected",
    "workers": 2
  }
}
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker compose logs api

# Verify database connection
docker compose exec api python -c "from app.db import engine; print(engine.connect())"
```

### Workers Not Processing

```bash
# Check worker status
docker compose exec worker celery -A app.worker inspect active

# Restart workers
docker compose restart worker
```

### Out of Memory

Increase memory limits:

```yaml
services:
  worker:
    deploy:
      resources:
        limits:
          memory: 4G
```

## Next Steps

- [Configuration](/docs/self-hosting/configuration) — Full configuration options
- [Kubernetes](/docs/self-hosting/kubernetes) — Production deployment
- [Troubleshooting](/docs/resources/troubleshooting) — Common issues
