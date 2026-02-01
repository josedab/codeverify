---
sidebar_position: 1
---

# Self-Hosting Overview

Run CodeVerify on your own infrastructure for complete control and data privacy.

## Why Self-Host?

- **Data sovereignty** — Code never leaves your network
- **Compliance** — Meet regulatory requirements (SOC 2, HIPAA, etc.)
- **Performance** — Reduce latency for large codebases
- **Customization** — Full control over configuration
- **Air-gapped** — Run without internet connectivity

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CodeVerify Stack                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   API       │  │  Dashboard  │  │  GitHub App │          │
│  │  (FastAPI)  │  │  (Next.js)  │  │  (Node.js)  │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         │                │                │                  │
│         └────────────────┴────────────────┘                  │
│                          │                                   │
│                    ┌─────┴─────┐                             │
│                    │   Redis   │                             │
│                    │  (Queue)  │                             │
│                    └─────┬─────┘                             │
│                          │                                   │
│  ┌───────────────────────┴───────────────────────┐          │
│  │              Worker Pool (Celery)              │          │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐       │          │
│  │  │Worker 1 │  │Worker 2 │  │Worker N │       │          │
│  │  │  (Z3)   │  │  (Z3)   │  │  (Z3)   │       │          │
│  │  └─────────┘  └─────────┘  └─────────┘       │          │
│  └───────────────────────────────────────────────┘          │
│                          │                                   │
│                    ┌─────┴─────┐                             │
│                    │ PostgreSQL│                             │
│                    │(Database) │                             │
│                    └───────────┘                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **API** | REST API for analyses | FastAPI (Python) |
| **Dashboard** | Web UI | Next.js (TypeScript) |
| **Worker** | Verification engine | Celery + Z3 |
| **GitHub App** | GitHub integration | Node.js |
| **Queue** | Job queue | Redis |
| **Database** | Persistent storage | PostgreSQL |

## Deployment Options

### Docker Compose (Development/Small Teams)

Quick setup for evaluation or small teams:

```bash
docker compose up -d
```

See [Docker Deployment](/docs/self-hosting/docker) for details.

### Kubernetes (Production)

Scalable deployment for production:

```bash
helm install codeverify codeverify/codeverify
```

See [Kubernetes Deployment](/docs/self-hosting/kubernetes) for details.

## Requirements

### Minimum (Single Server)

- **CPU**: 4 cores
- **RAM**: 8 GB
- **Storage**: 50 GB SSD
- **OS**: Linux (Ubuntu 20.04+, Debian 11+)

### Recommended (Production)

- **API**: 2 replicas, 2 CPU, 4 GB RAM each
- **Workers**: 4+ replicas, 4 CPU, 8 GB RAM each
- **Database**: PostgreSQL 14+, 100 GB storage
- **Redis**: 2 GB RAM

### High Availability

- Load balancer (nginx, HAProxy, or cloud LB)
- PostgreSQL with replication
- Redis Sentinel or Cluster
- Multiple worker nodes

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/codeverify/codeverify.git
cd codeverify
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 3. Start Services

```bash
docker compose up -d
```

### 4. Access Dashboard

Open http://localhost:3000 and create an admin account.

## Configuration

Key configuration areas:

- [General Settings](/docs/self-hosting/configuration#general)
- [Database](/docs/self-hosting/configuration#database)
- [Authentication](/docs/self-hosting/configuration#authentication)
- [AI Providers](/docs/self-hosting/configuration#ai)
- [GitHub/GitLab Integration](/docs/self-hosting/configuration#git)

## License

Self-hosting requires an Enterprise license for production use. Contact sales@codeverify.dev.

Development and evaluation use is free with the open-source license.

## Support

| Plan | Support Level |
|------|--------------|
| Community | GitHub Issues |
| Pro | Email support |
| Enterprise | Dedicated support, SLA |

## Next Steps

- [Docker Deployment](/docs/self-hosting/docker) — Quick start with Docker
- [Kubernetes Deployment](/docs/self-hosting/kubernetes) — Production deployment
- [Configuration](/docs/self-hosting/configuration) — Full configuration reference
