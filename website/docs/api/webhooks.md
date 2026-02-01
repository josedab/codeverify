---
sidebar_position: 5
---

# Webhooks

Receive real-time notifications when events occur.

## Overview

Webhooks notify your server when:
- Analysis completes
- New findings are discovered
- Finding status changes
- Trust score changes significantly

## Creating a Webhook

```http
POST /v1/webhooks
```

### Request

```bash
curl -X POST https://api.codeverify.dev/v1/webhooks \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-server.com/webhooks/codeverify",
    "events": ["analysis.completed", "finding.created"],
    "secret": "your-webhook-secret"
  }'
```

### Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | string | Yes | HTTPS endpoint to receive webhooks |
| `events` | array | Yes | Events to subscribe to |
| `secret` | string | Yes | Secret for signature verification |
| `repositories` | array | No | Limit to specific repositories |
| `active` | boolean | No | Enable/disable (default: true) |

### Response

```json
{
  "id": "wh_abc123",
  "url": "https://your-server.com/webhooks/codeverify",
  "events": ["analysis.completed", "finding.created"],
  "repositories": [],
  "active": true,
  "created_at": "2024-01-15T10:30:00Z"
}
```

## Events

### analysis.completed

Triggered when an analysis finishes.

```json
{
  "event": "analysis.completed",
  "timestamp": "2024-01-15T10:32:15Z",
  "data": {
    "analysis": {
      "id": "an_abc123",
      "repository": "owner/repo",
      "ref": "main",
      "status": "completed",
      "result": {
        "trust_score": 85,
        "findings_count": 3,
        "findings_by_severity": {
          "critical": 0,
          "high": 1,
          "medium": 2,
          "low": 0
        }
      }
    }
  }
}
```

### analysis.failed

Triggered when an analysis fails.

```json
{
  "event": "analysis.failed",
  "timestamp": "2024-01-15T10:32:15Z",
  "data": {
    "analysis": {
      "id": "an_abc123",
      "repository": "owner/repo",
      "status": "failed",
      "error": {
        "code": "timeout",
        "message": "Analysis timed out after 300 seconds"
      }
    }
  }
}
```

### finding.created

Triggered when a new finding is discovered.

```json
{
  "event": "finding.created",
  "timestamp": "2024-01-15T10:32:15Z",
  "data": {
    "finding": {
      "id": "f_xyz789",
      "analysis_id": "an_abc123",
      "repository": "owner/repo",
      "severity": "high",
      "category": "null_safety",
      "message": "Potential null dereference",
      "file": "src/api/handlers.py",
      "line": 45
    }
  }
}
```

### finding.resolved

Triggered when a finding is resolved.

```json
{
  "event": "finding.resolved",
  "timestamp": "2024-01-15T11:00:00Z",
  "data": {
    "finding": {
      "id": "f_xyz789",
      "status": "resolved",
      "resolution": {
        "type": "fixed",
        "resolved_by": "user@example.com"
      }
    }
  }
}
```

### trust_score.changed

Triggered when trust score changes significantly.

```json
{
  "event": "trust_score.changed",
  "timestamp": "2024-01-15T10:32:15Z",
  "data": {
    "repository": "owner/repo",
    "ref": "main",
    "previous_score": 92,
    "current_score": 78,
    "change": -14
  }
}
```

## Webhook Delivery

### Request Format

CodeVerify sends POST requests with:

```http
POST /your-endpoint
Content-Type: application/json
X-CodeVerify-Signature: sha256=abc123...
X-CodeVerify-Event: analysis.completed
X-CodeVerify-Delivery: d_xyz789

{
  "event": "analysis.completed",
  "timestamp": "2024-01-15T10:32:15Z",
  "data": { ... }
}
```

### Headers

| Header | Description |
|--------|-------------|
| `X-CodeVerify-Signature` | HMAC signature for verification |
| `X-CodeVerify-Event` | Event type |
| `X-CodeVerify-Delivery` | Unique delivery ID |

## Signature Verification

Verify webhook authenticity using the signature:

### Python

```python
import hmac
import hashlib

def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    expected = 'sha256=' + hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)

# In your webhook handler
@app.post("/webhooks/codeverify")
def handle_webhook(request):
    signature = request.headers.get('X-CodeVerify-Signature')
    if not verify_signature(request.body, signature, WEBHOOK_SECRET):
        return Response(status=401)
    
    event = request.json()
    # Process event...
```

### JavaScript

```javascript
const crypto = require('crypto');

function verifySignature(payload, signature, secret) {
  const expected = 'sha256=' + crypto
    .createHmac('sha256', secret)
    .update(payload)
    .digest('hex');
  return crypto.timingSafeEqual(
    Buffer.from(expected),
    Buffer.from(signature)
  );
}

// Express handler
app.post('/webhooks/codeverify', (req, res) => {
  const signature = req.headers['x-codeverify-signature'];
  if (!verifySignature(req.rawBody, signature, WEBHOOK_SECRET)) {
    return res.status(401).send('Invalid signature');
  }
  
  const event = req.body;
  // Process event...
});
```

## Managing Webhooks

### List Webhooks

```bash
curl https://api.codeverify.dev/v1/webhooks \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY"
```

### Update Webhook

```bash
curl -X PATCH https://api.codeverify.dev/v1/webhooks/wh_abc123 \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "events": ["analysis.completed"],
    "active": false
  }'
```

### Delete Webhook

```bash
curl -X DELETE https://api.codeverify.dev/v1/webhooks/wh_abc123 \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY"
```

## Testing Webhooks

### Send Test Event

```bash
curl -X POST https://api.codeverify.dev/v1/webhooks/wh_abc123/test \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY"
```

### View Recent Deliveries

```bash
curl https://api.codeverify.dev/v1/webhooks/wh_abc123/deliveries \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY"
```

```json
{
  "data": [
    {
      "id": "d_xyz789",
      "event": "analysis.completed",
      "status": "success",
      "response_code": 200,
      "duration_ms": 145,
      "delivered_at": "2024-01-15T10:32:16Z"
    },
    {
      "id": "d_abc123",
      "event": "analysis.completed",
      "status": "failed",
      "response_code": 500,
      "error": "Connection timeout",
      "delivered_at": "2024-01-15T09:30:00Z",
      "retries": 3
    }
  ]
}
```

### Redeliver

```bash
curl -X POST https://api.codeverify.dev/v1/webhooks/deliveries/d_abc123/redeliver \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY"
```

## Retry Policy

Failed deliveries are retried automatically:

| Attempt | Delay |
|---------|-------|
| 1 | Immediate |
| 2 | 1 minute |
| 3 | 5 minutes |
| 4 | 30 minutes |
| 5 | 2 hours |

After 5 failed attempts, the webhook is marked as failing and disabled after 3 days.

## Best Practices

1. **Respond quickly** — Return 200 within 10 seconds
2. **Process async** — Queue events for background processing
3. **Verify signatures** — Always check the HMAC signature
4. **Handle duplicates** — Use delivery ID for idempotency
5. **Monitor failures** — Set up alerts for failing webhooks

## Next Steps

- [API Overview](/docs/api/overview) — API basics
- [Analyses API](/docs/api/analyses) — Trigger analyses
- [Notifications](/docs/integrations/slack-teams) — Slack/Teams integration
