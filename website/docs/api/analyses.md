---
sidebar_position: 3
---

# Analyses API

Trigger and manage code analyses.

## Create Analysis

Trigger a new analysis for a repository.

```http
POST /v1/analyses
```

### Request

```bash
curl -X POST https://api.codeverify.dev/v1/analyses \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "repository": "owner/repo",
    "ref": "main",
    "options": {
      "checks": ["null_safety", "array_bounds"],
      "ai_enabled": true
    }
  }'
```

### Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `repository` | string | Yes | Repository in `owner/repo` format |
| `ref` | string | Yes | Branch, tag, or commit SHA |
| `options.checks` | array | No | Specific checks to run |
| `options.ai_enabled` | boolean | No | Enable AI analysis (default: true) |
| `options.paths` | array | No | Limit to specific paths |
| `callback_url` | string | No | URL to POST results when complete |

### Response

```json
{
  "id": "an_abc123",
  "status": "queued",
  "repository": "owner/repo",
  "ref": "main",
  "created_at": "2024-01-15T10:30:00Z",
  "options": {
    "checks": ["null_safety", "array_bounds"],
    "ai_enabled": true
  }
}
```

## Get Analysis

Retrieve analysis details and results.

```http
GET /v1/analyses/{id}
```

### Request

```bash
curl https://api.codeverify.dev/v1/analyses/an_abc123 \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY"
```

### Response

```json
{
  "id": "an_abc123",
  "status": "completed",
  "repository": "owner/repo",
  "ref": "main",
  "commit_sha": "abc123def456",
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:30:05Z",
  "completed_at": "2024-01-15T10:32:15Z",
  "duration_seconds": 130,
  "result": {
    "trust_score": 85,
    "status": "pass",
    "findings_count": 3,
    "findings_by_severity": {
      "critical": 0,
      "high": 1,
      "medium": 2,
      "low": 0
    },
    "files_analyzed": 45,
    "lines_analyzed": 12500
  }
}
```

### Status Values

| Status | Description |
|--------|-------------|
| `queued` | Waiting to start |
| `running` | Analysis in progress |
| `completed` | Finished successfully |
| `failed` | Analysis failed |
| `cancelled` | Cancelled by user |

## List Analyses

Get analyses for a repository.

```http
GET /v1/analyses
```

### Request

```bash
curl "https://api.codeverify.dev/v1/analyses?repository=owner/repo&limit=10" \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY"
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `repository` | string | Filter by repository |
| `status` | string | Filter by status |
| `ref` | string | Filter by branch/ref |
| `limit` | integer | Results per page (max 100) |
| `offset` | integer | Pagination offset |

### Response

```json
{
  "data": [
    {
      "id": "an_abc123",
      "status": "completed",
      "repository": "owner/repo",
      "ref": "main",
      "created_at": "2024-01-15T10:30:00Z",
      "result": {
        "trust_score": 85,
        "findings_count": 3
      }
    }
  ],
  "pagination": {
    "total": 156,
    "limit": 10,
    "offset": 0,
    "has_more": true
  }
}
```

## Cancel Analysis

Cancel a running or queued analysis.

```http
POST /v1/analyses/{id}/cancel
```

### Request

```bash
curl -X POST https://api.codeverify.dev/v1/analyses/an_abc123/cancel \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY"
```

### Response

```json
{
  "id": "an_abc123",
  "status": "cancelled",
  "cancelled_at": "2024-01-15T10:31:00Z"
}
```

## Rerun Analysis

Rerun a completed analysis.

```http
POST /v1/analyses/{id}/rerun
```

### Request

```bash
curl -X POST https://api.codeverify.dev/v1/analyses/an_abc123/rerun \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY"
```

### Response

Returns a new analysis object with status `queued`.

## Get Analysis Logs

Retrieve analysis execution logs.

```http
GET /v1/analyses/{id}/logs
```

### Request

```bash
curl https://api.codeverify.dev/v1/analyses/an_abc123/logs \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY"
```

### Response

```json
{
  "logs": [
    {
      "timestamp": "2024-01-15T10:30:05Z",
      "level": "info",
      "message": "Starting analysis"
    },
    {
      "timestamp": "2024-01-15T10:30:06Z",
      "level": "info",
      "message": "Analyzing 45 files"
    },
    {
      "timestamp": "2024-01-15T10:32:15Z",
      "level": "info",
      "message": "Analysis complete: 3 findings"
    }
  ]
}
```

## Polling for Completion

### Simple Polling

```python
import time
import requests

def wait_for_analysis(analysis_id, api_key, timeout=300):
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"https://api.codeverify.dev/v1/analyses/{analysis_id}"
    
    start = time.time()
    while time.time() - start < timeout:
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if data["status"] in ["completed", "failed", "cancelled"]:
            return data
        
        time.sleep(5)  # Poll every 5 seconds
    
    raise TimeoutError("Analysis timed out")
```

### Using Webhooks (Recommended)

```bash
curl -X POST https://api.codeverify.dev/v1/analyses \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "repository": "owner/repo",
    "ref": "main",
    "callback_url": "https://your-server.com/webhook"
  }'
```

Your server receives a POST when analysis completes:

```json
{
  "event": "analysis.completed",
  "analysis": {
    "id": "an_abc123",
    "status": "completed",
    "result": { ... }
  }
}
```

## SDK Examples

### Python

```python
from codeverify import Client

client = Client(api_key="cv_your_api_key")

# Create and wait
analysis = client.analyses.create(
    repository="owner/repo",
    ref="main"
)
analysis.wait(timeout=300)

print(f"Trust Score: {analysis.result.trust_score}")
print(f"Findings: {analysis.result.findings_count}")
```

### JavaScript

```typescript
const client = new CodeVerify({ apiKey: 'cv_your_api_key' });

const analysis = await client.analyses.create({
  repository: 'owner/repo',
  ref: 'main'
});

await analysis.waitForCompletion({ timeout: 300 });

console.log(`Trust Score: ${analysis.result.trustScore}`);
```

## Next Steps

- [Findings API](/docs/api/findings) — Access detailed findings
- [Webhooks](/docs/api/webhooks) — Real-time notifications
- [Authentication](/docs/api/authentication) — API key management
