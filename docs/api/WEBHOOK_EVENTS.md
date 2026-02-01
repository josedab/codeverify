# CodeVerify Webhook Events

This document describes all webhook events that CodeVerify can send to your configured endpoints.

## Event Structure

All webhook payloads follow this structure:

```json
{
  "event": "event.name",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "delivery_id": "del_abc123xyz",
  "data": {
    // Event-specific data
  }
}
```

## Headers

Each webhook request includes these headers:

| Header | Description |
|--------|-------------|
| `Content-Type` | `application/json` |
| `X-CodeVerify-Event` | The event type (e.g., `analysis.completed`) |
| `X-CodeVerify-Delivery` | Unique delivery ID |
| `X-CodeVerify-Signature` | HMAC-SHA256 signature (if secret configured) |

---

## Analysis Events

### `analysis.started`

Triggered when an analysis begins.

```json
{
  "event": "analysis.started",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "data": {
    "analysis_id": "analysis_xyz789",
    "repository": {
      "owner": "acme",
      "name": "webapp",
      "full_name": "acme/webapp"
    },
    "ref": "refs/pull/42/head",
    "commit_sha": "a1b2c3d4e5f6",
    "pr_number": 42,
    "triggered_by": "pull_request"
  }
}
```

### `analysis.completed`

Triggered when an analysis finishes successfully.

```json
{
  "event": "analysis.completed",
  "timestamp": "2024-01-15T10:35:00.000Z",
  "data": {
    "analysis_id": "analysis_xyz789",
    "repository": {
      "owner": "acme",
      "name": "webapp",
      "full_name": "acme/webapp"
    },
    "ref": "refs/pull/42/head",
    "commit_sha": "a1b2c3d4e5f6",
    "pr_number": 42,
    "duration_seconds": 45,
    "result": {
      "pass": false,
      "findings_count": 3,
      "by_severity": {
        "critical": 1,
        "high": 1,
        "medium": 1,
        "low": 0
      }
    },
    "check_run_url": "https://github.com/acme/webapp/runs/12345"
  }
}
```

### `analysis.failed`

Triggered when an analysis encounters an error.

```json
{
  "event": "analysis.failed",
  "timestamp": "2024-01-15T10:35:00.000Z",
  "data": {
    "analysis_id": "analysis_xyz789",
    "repository": {
      "owner": "acme",
      "name": "webapp",
      "full_name": "acme/webapp"
    },
    "error": {
      "code": "timeout",
      "message": "Analysis timed out after 300 seconds",
      "retryable": true
    }
  }
}
```

---

## Finding Events

### `finding.created`

Triggered for each new finding discovered during analysis.

```json
{
  "event": "finding.created",
  "timestamp": "2024-01-15T10:35:00.000Z",
  "data": {
    "finding_id": "finding_abc123",
    "analysis_id": "analysis_xyz789",
    "repository": {
      "owner": "acme",
      "name": "webapp",
      "full_name": "acme/webapp"
    },
    "finding": {
      "category": "security",
      "severity": "critical",
      "title": "SQL Injection Vulnerability",
      "description": "User input is directly interpolated into SQL query without sanitization",
      "file_path": "src/database/queries.py",
      "line_start": 42,
      "line_end": 45,
      "code_snippet": "query = f\"SELECT * FROM users WHERE id = {user_id}\"",
      "confidence": 0.95,
      "verification_type": "formal",
      "fix_suggestion": "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
      "references": [
        "https://owasp.org/www-community/attacks/SQL_Injection"
      ]
    },
    "pr_number": 42
  }
}
```

---

## Scan Events

### `scan.started`

Triggered when a codebase-wide scan begins.

```json
{
  "event": "scan.started",
  "timestamp": "2024-01-15T00:00:00.000Z",
  "data": {
    "scan_id": "scan_def456",
    "repository": {
      "owner": "acme",
      "name": "webapp",
      "full_name": "acme/webapp"
    },
    "branch": "main",
    "triggered_by": "schedule",
    "config": {
      "include_patterns": ["**/*.py", "**/*.js"],
      "exclude_patterns": ["**/test/**", "**/vendor/**"]
    }
  }
}
```

### `scan.completed`

Triggered when a codebase scan finishes.

```json
{
  "event": "scan.completed",
  "timestamp": "2024-01-15T00:15:00.000Z",
  "data": {
    "scan_id": "scan_def456",
    "repository": {
      "owner": "acme",
      "name": "webapp",
      "full_name": "acme/webapp"
    },
    "branch": "main",
    "duration_seconds": 900,
    "files_scanned": 1247,
    "result": {
      "total_findings": 15,
      "new_findings": 3,
      "fixed_findings": 2,
      "by_severity": {
        "critical": 0,
        "high": 2,
        "medium": 8,
        "low": 5
      },
      "by_category": {
        "security": 3,
        "logic_error": 5,
        "performance": 4,
        "code_quality": 3
      }
    },
    "trends": {
      "findings_change": -2,
      "critical_change": 0,
      "coverage_change": 0.02
    }
  }
}
```

---

## Security Events

### `security.critical`

Triggered immediately when a critical security vulnerability is found.

```json
{
  "event": "security.critical",
  "timestamp": "2024-01-15T10:35:00.000Z",
  "data": {
    "analysis_id": "analysis_xyz789",
    "repository": {
      "owner": "acme",
      "name": "webapp",
      "full_name": "acme/webapp"
    },
    "finding": {
      "id": "finding_sec001",
      "category": "security",
      "severity": "critical",
      "title": "Remote Code Execution via eval()",
      "description": "User-controlled input is passed to eval(), allowing arbitrary code execution",
      "file_path": "src/api/handlers.py",
      "line_start": 127,
      "cwe_id": "CWE-94",
      "cvss_score": 9.8
    },
    "alert": {
      "priority": "immediate",
      "recommendation": "Block deployment and fix immediately"
    }
  }
}
```

---

## Test Event

### `test`

Sent when you test a webhook configuration.

```json
{
  "event": "test",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "data": {
    "message": "This is a test webhook delivery from CodeVerify",
    "webhook_id": "wh_abc123"
  }
}
```

---

## Signature Verification

If you configured a webhook secret, verify the signature to ensure the request came from CodeVerify.

### Python

```python
import hmac
import hashlib

def verify_webhook(payload: bytes, signature: str, secret: str) -> bool:
    """Verify CodeVerify webhook signature."""
    if not signature.startswith('sha256='):
        return False
    
    expected = 'sha256=' + hmac.new(
        secret.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(expected, signature)

# Usage in Flask
@app.route('/webhook', methods=['POST'])
def webhook():
    signature = request.headers.get('X-CodeVerify-Signature', '')
    
    if not verify_webhook(request.data, signature, WEBHOOK_SECRET):
        return 'Invalid signature', 401
    
    event = request.json
    # Process event...
    return 'OK', 200
```

### Node.js

```javascript
const crypto = require('crypto');

function verifyWebhook(payload, signature, secret) {
  if (!signature.startsWith('sha256=')) {
    return false;
  }
  
  const expected = 'sha256=' + crypto
    .createHmac('sha256', secret)
    .update(payload)
    .digest('hex');
  
  return crypto.timingSafeEqual(
    Buffer.from(expected),
    Buffer.from(signature)
  );
}

// Usage in Express
app.post('/webhook', express.raw({type: 'application/json'}), (req, res) => {
  const signature = req.headers['x-codeverify-signature'] || '';
  
  if (!verifyWebhook(req.body, signature, process.env.WEBHOOK_SECRET)) {
    return res.status(401).send('Invalid signature');
  }
  
  const event = JSON.parse(req.body);
  // Process event...
  res.status(200).send('OK');
});
```

### Go

```go
package main

import (
    "crypto/hmac"
    "crypto/sha256"
    "encoding/hex"
    "strings"
)

func verifyWebhook(payload []byte, signature, secret string) bool {
    if !strings.HasPrefix(signature, "sha256=") {
        return false
    }
    
    mac := hmac.New(sha256.New, []byte(secret))
    mac.Write(payload)
    expected := "sha256=" + hex.EncodeToString(mac.Sum(nil))
    
    return hmac.Equal([]byte(expected), []byte(signature))
}
```

---

## Best Practices

1. **Always verify signatures** - Never process webhooks without signature verification in production.

2. **Respond quickly** - Return a 2xx response within 10 seconds. Process events asynchronously if needed.

3. **Handle retries** - CodeVerify retries failed deliveries (5xx responses) up to 3 times with exponential backoff.

4. **Use idempotency** - Use the `delivery_id` to deduplicate events in case of retries.

5. **Subscribe selectively** - Only subscribe to events you actually need to reduce load.

---

## Webhook Management

### Create Webhook

```bash
curl -X POST https://api.codeverify.io/api/webhooks \
  -H "X-API-Key: cv_your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-server.com/webhook",
    "events": ["analysis.completed", "security.critical"],
    "secret": "your_secret_here"
  }'
```

### List Webhooks

```bash
curl https://api.codeverify.io/api/webhooks \
  -H "X-API-Key: cv_your_key"
```

### Delete Webhook

```bash
curl -X DELETE https://api.codeverify.io/api/webhooks/{webhook_id} \
  -H "X-API-Key: cv_your_key"
```

### View Recent Deliveries

```bash
curl https://api.codeverify.io/api/webhooks/{webhook_id}/deliveries \
  -H "X-API-Key: cv_your_key"
```
