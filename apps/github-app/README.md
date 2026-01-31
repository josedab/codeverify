# CodeVerify GitHub App

GitHub webhook handler and PR integration for CodeVerify.

## Overview

The GitHub App service handles:

- Webhook events from GitHub (PR opened, synchronized, etc.)
- Queuing analysis jobs to Redis
- Posting PR comments with results
- Updating check statuses
- App installation management

## Quick Start

### Prerequisites

- Node.js 20+
- Redis 7+
- GitHub App credentials

### Development Setup

```bash
cd apps/github-app

# Install dependencies
npm install

# Set environment variables
cp .env.example .env
# Edit .env with your GitHub App credentials

# Start development server
npm run dev
```

The server runs on http://localhost:3001

### Using Docker

```bash
docker compose up github-app
```

## Project Structure

```
apps/github-app/
├── src/
│   ├── index.ts           # Entry point
│   ├── app.ts             # Express app setup
│   ├── config.ts          # Configuration
│   ├── webhooks/          # Webhook handlers
│   │   ├── index.ts       # Webhook router
│   │   ├── pullRequest.ts # PR events
│   │   ├── installation.ts # App installation
│   │   └── checkRun.ts    # Check run events
│   ├── services/
│   │   ├── github.ts      # GitHub API client
│   │   ├── queue.ts       # Redis queue
│   │   └── comments.ts    # PR comment builder
│   └── utils/
├── dist/                  # Compiled output
└── package.json
```

## GitHub App Setup

### 1. Create GitHub App

Go to **GitHub Settings → Developer settings → GitHub Apps → New GitHub App**

**Settings:**
- **Name:** CodeVerify
- **Homepage URL:** https://codeverify.io
- **Webhook URL:** https://your-domain/webhooks/github
- **Webhook secret:** Generate a secure secret

**Permissions:**
- **Repository:**
  - Contents: Read
  - Pull requests: Read & Write
  - Checks: Read & Write
  - Metadata: Read
- **Organization:**
  - Members: Read

**Events:**
- Pull request
- Check run
- Installation

### 2. Generate Private Key

After creating the app, generate and download a private key.

### 3. Configure Environment

```bash
GITHUB_APP_ID=123456
GITHUB_APP_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\n..."
GITHUB_WEBHOOK_SECRET=your-webhook-secret
GITHUB_CLIENT_ID=Iv1.abc123
GITHUB_CLIENT_SECRET=secret123
```

## Webhook Handlers

### Pull Request Events

```typescript
// src/webhooks/pullRequest.ts
export async function handlePullRequest(payload: PullRequestEvent) {
  const { action, pull_request, repository } = payload;
  
  if (action === 'opened' || action === 'synchronize') {
    // Queue analysis job
    await queue.add('analyze', {
      repository: repository.full_name,
      prNumber: pull_request.number,
      headSha: pull_request.head.sha,
      baseSha: pull_request.base.sha,
    });
    
    // Set pending check
    await github.createCheckRun({
      owner: repository.owner.login,
      repo: repository.name,
      name: 'CodeVerify',
      head_sha: pull_request.head.sha,
      status: 'in_progress',
    });
  }
}
```

### Installation Events

```typescript
// src/webhooks/installation.ts
export async function handleInstallation(payload: InstallationEvent) {
  const { action, installation, repositories } = payload;
  
  if (action === 'created') {
    // Record new installation
    await db.installations.create({
      installationId: installation.id,
      accountId: installation.account.id,
      accountType: installation.account.type,
      repositories: repositories?.map(r => r.id) ?? [],
    });
  }
}
```

## Services

### GitHub API Client

```typescript
import { github } from './services/github';

// Get authenticated client for installation
const octokit = await github.getInstallationClient(installationId);

// Get PR files
const files = await octokit.pulls.listFiles({
  owner: 'org',
  repo: 'repo',
  pull_number: 123,
});

// Create check run
await octokit.checks.create({
  owner: 'org',
  repo: 'repo',
  name: 'CodeVerify',
  head_sha: 'abc123',
  status: 'completed',
  conclusion: 'success',
});
```

### Queue Service

```typescript
import { queue } from './services/queue';

// Add job
await queue.add('analyze', {
  repository: 'org/repo',
  prNumber: 123,
});

// Listen for results
queue.on('analysis.complete', async (result) => {
  await postPRComment(result);
  await updateCheckRun(result);
});
```

### Comment Builder

```typescript
import { buildComment } from './services/comments';

const comment = buildComment({
  summary: {
    total: 5,
    critical: 0,
    high: 1,
    medium: 2,
    low: 2,
  },
  findings: [...],
  passed: true,
});

// Posts formatted markdown comment
await octokit.issues.createComment({
  owner: 'org',
  repo: 'repo',
  issue_number: 123,
  body: comment,
});
```

## Configuration

### Environment Variables

```bash
# GitHub App
GITHUB_APP_ID=123456
GITHUB_APP_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----..."
GITHUB_WEBHOOK_SECRET=secret

# Redis
REDIS_URL=redis://localhost:6379/0

# API
API_URL=http://localhost:8000

# Server
PORT=3001
NODE_ENV=development
```

## Webhook Verification

All webhooks are verified using HMAC signatures:

```typescript
import crypto from 'crypto';

function verifyWebhook(payload: string, signature: string): boolean {
  const expected = `sha256=${crypto
    .createHmac('sha256', process.env.GITHUB_WEBHOOK_SECRET!)
    .update(payload)
    .digest('hex')}`;
    
  return crypto.timingSafeEqual(
    Buffer.from(signature),
    Buffer.from(expected)
  );
}
```

## Testing

```bash
# Run tests
npm test

# Run with coverage
npm run test:coverage

# Test webhook locally with smee.io
npx smee -u https://smee.io/your-channel -t http://localhost:3001/webhooks/github
```

## Build & Deploy

```bash
# Build TypeScript
npm run build

# Start production
npm start
```

### Docker

```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY dist/ ./dist/
EXPOSE 3001
CMD ["node", "dist/index.js"]
```

## Monitoring

### Health Check

```bash
curl http://localhost:3001/health
# {"status":"ok","version":"0.3.0"}
```

### Metrics

Prometheus metrics at `/metrics`:
- `github_webhooks_total` - Webhooks received
- `github_api_calls_total` - API calls made
- `analysis_queue_size` - Pending analyses
