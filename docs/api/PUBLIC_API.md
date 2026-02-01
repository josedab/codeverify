# CodeVerify Public API Documentation

## Overview

The CodeVerify API provides programmatic access to AI-powered code review with formal verification capabilities. This document covers authentication, available endpoints, and usage examples.

## Base URL

```
Production: https://api.codeverify.io
Development: http://localhost:8000
```

## Authentication

### API Keys

All API requests require authentication using an API key. Include your key in the request headers:

```bash
# Using X-API-Key header
curl -H "X-API-Key: cv_your_api_key_here" https://api.codeverify.io/api/v1/analyses

# Using Authorization header
curl -H "Authorization: Bearer cv_your_api_key_here" https://api.codeverify.io/api/v1/analyses
```

### Scopes

API keys can have the following scopes:
- `read` - Read-only access to analyses, findings, and statistics
- `write` - Create analyses, manage rules, trigger scans
- `admin` - Full access including API key management

### Creating API Keys

```bash
POST /api/keys
Content-Type: application/json

{
  "name": "CI/CD Integration",
  "description": "Key for GitHub Actions",
  "scopes": ["read", "write"],
  "expires_in_days": 365
}
```

Response:
```json
{
  "id": "key_abc123",
  "name": "CI/CD Integration",
  "key": "cv_xK9mN2pQ...",  // Only shown once!
  "key_prefix": "cv_xK9mN2",
  "scopes": ["read", "write"],
  "created_at": "2024-01-15T10:30:00Z",
  "expires_at": "2025-01-15T10:30:00Z"
}
```

⚠️ **Important**: The full API key is only returned once at creation. Store it securely!

---

## Rate Limits

| Plan | Requests/Minute | Requests/Hour |
|------|-----------------|---------------|
| Free | 60 | 500 |
| Pro | 300 | 5,000 |
| Enterprise | 1,000 | 50,000 |

Rate limit information is included in response headers:
```
X-RateLimit-Limit: 300
X-RateLimit-Remaining: 298
X-RateLimit-Reset: 1705312200
```

---

## Endpoints

### Analyses

#### Trigger Analysis

```bash
POST /api/v1/analyses
```

Request:
```json
{
  "repo": "owner/repo",
  "ref": "main",
  "pr_number": 42  // optional
}
```

Response:
```json
{
  "id": "analysis_xyz789",
  "status": "queued",
  "repo": "owner/repo",
  "ref": "main",
  "pr_number": 42,
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### Get Analysis

```bash
GET /api/v1/analyses/{analysis_id}
```

Response:
```json
{
  "id": "analysis_xyz789",
  "status": "completed",
  "repo": "owner/repo",
  "findings": [
    {
      "id": "finding_001",
      "category": "security",
      "severity": "high",
      "title": "SQL Injection Vulnerability",
      "description": "User input used directly in SQL query",
      "file_path": "src/db.py",
      "line_start": 42,
      "line_end": 42,
      "confidence": 0.95,
      "fix_suggestion": "Use parameterized queries"
    }
  ],
  "summary": {
    "total_findings": 1,
    "by_severity": {"critical": 0, "high": 1, "medium": 0, "low": 0},
    "pass": false
  }
}
```

#### List Analyses

```bash
GET /api/v1/analyses?repo=owner/repo&status=completed&limit=20
```

---

### Trust Score

#### Analyze Code Trust Score

```bash
POST /api/v1/trust-score/analyze
```

Request:
```json
{
  "code": "def process(data):\n    return data * 2",
  "language": "python",
  "context": {
    "file_path": "src/processor.py",
    "author": "john.doe"
  }
}
```

Response:
```json
{
  "score": 78,
  "risk_level": "low",
  "ai_probability": 15,
  "factors": {
    "complexity_score": 0.85,
    "pattern_score": 0.75,
    "historical_score": 0.80,
    "verification_score": 0.70,
    "quality_score": 0.82
  },
  "recommendations": [
    "Add type hints for better verification",
    "Consider adding input validation"
  ],
  "confidence": 0.88
}
```

---

### Custom Rules

#### Create Rule

```bash
POST /api/v1/rules
```

Request:
```json
{
  "id": "no-print",
  "name": "No Print Statements",
  "description": "Disallow print statements in production code",
  "type": "pattern",
  "pattern": "print\\s*\\(",
  "severity": "warning",
  "message": "Use logging instead of print statements"
}
```

#### Test Rule

```bash
POST /api/v1/rules/test
```

Request:
```json
{
  "rule": {
    "type": "pattern",
    "pattern": "print\\s*\\(",
    "severity": "warning",
    "message": "No print"
  },
  "code": "def hello():\n    print('hello')\n    print('world')"
}
```

Response:
```json
{
  "violations": [
    {"line": 2, "column": 4, "message": "No print"},
    {"line": 3, "column": 4, "message": "No print"}
  ],
  "total": 2
}
```

---

### Codebase Scanning

#### Trigger Full Scan

```bash
POST /api/v1/scans
```

Request:
```json
{
  "repository": "owner/repo",
  "branch": "main",
  "config": {
    "include_patterns": ["**/*.py", "**/*.js"],
    "exclude_patterns": ["**/test/**"],
    "severity_threshold": "medium"
  }
}
```

#### Schedule Recurring Scan

```bash
POST /api/v1/scans/schedule
```

Request:
```json
{
  "repository": "owner/repo",
  "cron": "0 0 * * *",
  "branch": "main",
  "notify_on": ["failure", "new_critical"]
}
```

---

### Notifications

#### Configure Slack

```bash
POST /api/v1/notifications/slack
```

Request:
```json
{
  "webhook_url": "https://hooks.slack.com/services/xxx/yyy/zzz",
  "channel": "#code-reviews",
  "events": ["analysis.completed", "security.critical"],
  "format": "detailed"
}
```

#### Configure Microsoft Teams

```bash
POST /api/v1/notifications/teams
```

Request:
```json
{
  "webhook_url": "https://outlook.office.com/webhook/...",
  "events": ["analysis.completed"]
}
```

---

### Webhooks

#### Create Webhook

```bash
POST /api/webhooks
```

Request:
```json
{
  "url": "https://your-server.com/codeverify-webhook",
  "events": ["analysis.completed", "analysis.failed", "finding.created"],
  "secret": "your_webhook_secret"
}
```

#### Available Events

| Event | Description |
|-------|-------------|
| `analysis.started` | Analysis has begun |
| `analysis.completed` | Analysis finished successfully |
| `analysis.failed` | Analysis encountered an error |
| `finding.created` | New finding discovered |
| `scan.started` | Codebase scan started |
| `scan.completed` | Codebase scan finished |
| `security.critical` | Critical security issue found |

#### Webhook Payload

```json
{
  "event": "analysis.completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "analysis_id": "analysis_xyz789",
    "repo": "owner/repo",
    "findings_count": 3,
    "pass": false
  }
}
```

#### Verifying Webhook Signatures

Webhooks are signed using HMAC-SHA256. Verify using:

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
```

The signature is sent in the `X-CodeVerify-Signature` header.

---

### Verification Debugger

#### Trace Verification

```bash
POST /api/v1/debugger/trace
```

Request:
```json
{
  "code": "def divide(a, b):\n    assert b != 0\n    return a / b",
  "language": "python"
}
```

Response:
```json
{
  "result": "verified",
  "steps": [
    {
      "step_number": 1,
      "title": "Parse function signature",
      "status": "passed",
      "description": "Extracted parameters: a, b"
    },
    {
      "step_number": 2,
      "title": "Check assertion: b != 0",
      "status": "passed",
      "constraint": "b != 0",
      "description": "Precondition verified"
    },
    {
      "step_number": 3,
      "title": "Verify division safety",
      "status": "passed",
      "description": "Division is safe given b != 0"
    }
  ]
}
```

---

### Diff Summarizer

#### Generate PR Summary

```bash
POST /api/v1/diff/summarize
```

Request:
```json
{
  "diff": "diff --git a/src/auth.py...",
  "context": {
    "pr_number": 42,
    "base_branch": "main",
    "files_changed": 3
  }
}
```

Response:
```json
{
  "title": "Improve authentication flow",
  "description": "This PR refactors the authentication module to support OAuth2...",
  "changes": [
    {"type": "added", "description": "OAuth2 provider support"},
    {"type": "modified", "description": "Token refresh logic"},
    {"type": "security", "description": "Added CSRF protection"}
  ],
  "risk_assessment": {
    "level": "medium",
    "reasons": ["Modifies authentication logic", "Changes session handling"]
  },
  "suggested_reviewers": ["@security-team"]
}
```

---

## Error Handling

All errors follow a consistent format:

```json
{
  "error": {
    "code": "invalid_request",
    "message": "The request body is missing required field 'code'",
    "details": {
      "field": "code",
      "constraint": "required"
    }
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `missing_api_key` | 401 | No API key provided |
| `invalid_api_key` | 401 | API key is invalid or expired |
| `insufficient_scope` | 403 | API key lacks required scope |
| `rate_limit_exceeded` | 429 | Too many requests |
| `invalid_request` | 400 | Request validation failed |
| `not_found` | 404 | Resource not found |
| `internal_error` | 500 | Server error |

---

## SDKs

Official SDKs are available for:
- Python: `pip install codeverify`
- JavaScript/TypeScript: `npm install @codeverify/sdk`
- Go: `go get github.com/codeverify/go-sdk`

### Python Example

```python
from codeverify import CodeVerifyClient

client = CodeVerifyClient(api_key="cv_your_key")

# Analyze code
result = client.trust_score.analyze(
    code="def process(x): return x * 2",
    language="python"
)
print(f"Trust Score: {result.score}")

# Trigger analysis
analysis = client.analyses.create(
    repo="owner/repo",
    ref="main"
)
print(f"Analysis ID: {analysis.id}")
```

---

## Support

- Documentation: https://docs.codeverify.io
- API Status: https://status.codeverify.io
- Support Email: api-support@codeverify.io
- GitHub Issues: https://github.com/codeverify/codeverify/issues
