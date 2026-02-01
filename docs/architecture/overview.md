# CodeVerify Architecture Overview

## System Architecture

CodeVerify is a multi-service application that combines AI-powered code analysis with formal verification to provide comprehensive code review capabilities.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CodeVerify Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐                                                            │
│  │   GitHub    │                                                            │
│  │  Webhooks   │────────────────────────────────────┐                       │
│  └─────────────┘                                    │                       │
│                                                     ▼                       │
│  ┌─────────────┐     ┌─────────────────────────────────────────────────┐   │
│  │   GitHub    │     │              GitHub App Service                  │   │
│  │    App      │◀───▶│         (Node.js / Express)                     │   │
│  └─────────────┘     │    - Webhook handling                           │   │
│                      │    - Job queuing                                 │   │
│                      │    - PR status updates                          │   │
│                      └──────────────────────┬──────────────────────────┘   │
│                                             │                               │
│                                             ▼                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                           Redis Queue                                │   │
│  │                    (Job queue + Result cache)                       │   │
│  └──────────────────────┬──────────────────────────────────────────────┘   │
│                         │                                                   │
│                         ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       Analysis Worker                                │   │
│  │                     (Python / Celery)                               │   │
│  │  ┌───────────────────────────────────────────────────────────────┐ │   │
│  │  │                    Analysis Pipeline                           │ │   │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │ │   │
│  │  │  │  Semantic   │  │   Formal    │  │  Security   │           │ │   │
│  │  │  │   Agent     │  │  Verifier   │  │   Agent     │           │ │   │
│  │  │  │   (LLM)     │  │    (Z3)     │  │   (LLM)     │           │ │   │
│  │  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘           │ │   │
│  │  │         │                │                │                   │ │   │
│  │  │         └────────────────┼────────────────┘                   │ │   │
│  │  │                          │                                     │ │   │
│  │  │                 ┌────────▼────────┐                           │ │   │
│  │  │                 │   Synthesis     │                           │ │   │
│  │  │                 │     Agent       │                           │ │   │
│  │  │                 └─────────────────┘                           │ │   │
│  │  └───────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         API Service                                  │   │
│  │                    (Python / FastAPI)                               │   │
│  │    - REST API for dashboard                                         │   │
│  │    - Authentication / Authorization                                 │   │
│  │    - Analysis results retrieval                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Web Dashboard                                │   │
│  │                      (Next.js / React)                              │   │
│  │    - Team analytics                                                 │   │
│  │    - Configuration management                                       │   │
│  │    - Finding details                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Data Layer                                    │   │
│  │  ┌──────────┐  ┌──────────┐                                        │   │
│  │  │ Postgres │  │  Redis   │                                        │   │
│  │  │  (Data)  │  │ (Cache)  │                                        │   │
│  │  └──────────┘  └──────────┘                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. GitHub App Service (`apps/github-app`)

**Technology:** Node.js, Express, Octokit

**Responsibilities:**
- Receive and validate GitHub webhooks
- Queue analysis jobs to Redis
- Post PR comments and check statuses
- Handle app installation events

### 2. API Service (`apps/api`)

**Technology:** Python, FastAPI, SQLAlchemy

**Responsibilities:**
- REST API for the web dashboard
- Authentication via GitHub OAuth
- Analysis results retrieval
- Organization/repository management

### 3. Analysis Worker (`apps/worker`)

**Technology:** Python, Celery, Z3

**Responsibilities:**
- Execute multi-stage analysis pipeline
- Coordinate AI agents
- Run formal verification
- Store results in database

### 4. Web Dashboard (`apps/web`)

**Technology:** Next.js, React, Tailwind CSS

**Responsibilities:**
- Team analytics dashboard
- Analysis result viewing
- Configuration management
- User authentication

## Packages

### `packages/core`

Shared data models and utilities used across all Python services.

### `packages/verifier`

Z3 SMT solver integration for formal verification:
- Integer overflow checking
- Array bounds verification
- Null dereference detection
- Division by zero checking

### `packages/ai-agents`

LLM-powered analysis agents:
- **Semantic Agent:** Understands code intent and extracts contracts
- **Security Agent:** Finds vulnerabilities and security issues
- **Synthesis Agent:** Consolidates results and generates fixes

### `packages/z3-mcp`

Model Context Protocol server for Z3, allowing AI agents to use the SMT solver as a tool.

## Data Flow

1. **PR Created/Updated**
   - GitHub sends webhook to GitHub App service
   - Webhook handler validates and extracts PR data
   - Analysis job queued to Redis

2. **Analysis Execution**
   - Worker picks up job from queue
   - Fetches PR diff from GitHub API
   - Runs multi-stage analysis pipeline:
     - Parse code to AST
     - Semantic analysis (LLM)
     - Formal verification (Z3)
     - Security analysis (LLM)
     - Synthesis and fix generation

3. **Results Delivery**
   - Findings stored in PostgreSQL
   - PR comment posted via GitHub API
   - Check status updated
   - Dashboard updated via API

## Security Considerations

- **Code Privacy:** Source code is processed in memory and not persisted
- **Authentication:** GitHub OAuth for users, JWT for API
- **Webhook Security:** HMAC signature verification
- **Data Encryption:** TLS for transit, AES-256 for rest
