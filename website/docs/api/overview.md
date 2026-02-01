---
sidebar_position: 1
---

# API Overview

CodeVerify provides a REST API for programmatic access to verification results.

## Base URL

```
Production: https://api.codeverify.dev/v1
Self-hosted: https://your-instance.com/api/v1
```

## Authentication

All requests require an API key:

```bash
curl -H "Authorization: Bearer cv_your_api_key" \
  https://api.codeverify.dev/v1/analyses
```

See [Authentication](/docs/api/authentication) for details.

## Quick Start

### Trigger an Analysis

```bash
curl -X POST https://api.codeverify.dev/v1/analyses \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "repository": "owner/repo",
    "ref": "main"
  }'
```

Response:

```json
{
  "id": "an_abc123",
  "status": "queued",
  "repository": "owner/repo",
  "ref": "main",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Check Analysis Status

```bash
curl https://api.codeverify.dev/v1/analyses/an_abc123 \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY"
```

### Get Findings

```bash
curl https://api.codeverify.dev/v1/analyses/an_abc123/findings \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY"
```

## Resources

| Resource | Description |
|----------|-------------|
| [Analyses](/docs/api/analyses) | Trigger and manage analyses |
| [Findings](/docs/api/findings) | Access verification findings |
| [Webhooks](/docs/api/webhooks) | Receive real-time notifications |

## Response Format

All responses are JSON with consistent structure:

```json
{
  "data": { ... },
  "meta": {
    "request_id": "req_xyz789",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Pagination

List endpoints support pagination:

```bash
curl "https://api.codeverify.dev/v1/analyses?limit=20&offset=40" \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY"
```

### Errors

```json
{
  "error": {
    "code": "invalid_request",
    "message": "Repository not found"
  }
}
```

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `unauthorized` | 401 | Invalid API key |
| `forbidden` | 403 | Insufficient permissions |
| `not_found` | 404 | Resource not found |
| `rate_limited` | 429 | Too many requests |

## Rate Limits

| Plan | Requests/minute | Analyses/day |
|------|-----------------|--------------|
| Free | 60 | 100 |
| Pro | 300 | 1,000 |
| Enterprise | 1,000 | Unlimited |

## SDKs

### Python

```python
from codeverify import Client

client = Client(api_key="cv_your_api_key")
analysis = client.analyses.create(repository="owner/repo", ref="main")
analysis.wait()

for finding in analysis.findings:
    print(f"{finding.severity}: {finding.message}")
```

### JavaScript

```typescript
import { CodeVerify } from '@codeverify/sdk';

const client = new CodeVerify({ apiKey: 'cv_your_api_key' });
const analysis = await client.analyses.create({ repository: 'owner/repo', ref: 'main' });
await analysis.waitForCompletion();
```

## OpenAPI Spec

```bash
curl -O https://api.codeverify.dev/v1/openapi.json
```
