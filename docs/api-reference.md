# CodeVerify API Reference

## Overview

The CodeVerify API provides programmatic access to analyses, findings, and configuration.

**Base URL:** `https://api.codeverify.io/v1`

**Authentication:** Bearer token in `Authorization` header

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.codeverify.io/v1/analyses
```

## Authentication

### Get API Key

Generate API keys from the dashboard: **Settings > API Keys > Generate New Key**

### Token Expiration

API keys don't expire but can be revoked. For short-lived tokens, use OAuth.

### OAuth Flow

```
GET /auth/github?redirect_uri=YOUR_CALLBACK_URL
```

Exchange the authorization code for an access token:

```bash
POST /auth/token
Content-Type: application/json

{
  "grant_type": "authorization_code",
  "code": "AUTH_CODE",
  "redirect_uri": "YOUR_CALLBACK_URL"
}
```

---

## Organizations

### List Organizations

```
GET /organizations
```

**Response:**
```json
{
  "data": [
    {
      "id": "org_abc123",
      "github_id": 12345,
      "name": "Acme Corp",
      "slug": "acme-corp",
      "avatar_url": "https://...",
      "plan": "team",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### Get Organization

```
GET /organizations/{org_id}
```

---

## Repositories

### List Repositories

```
GET /repositories?organization_id={org_id}
```

**Query Parameters:**
- `organization_id` (optional): Filter by organization
- `enabled` (optional): Filter by enabled status

**Response:**
```json
{
  "data": [
    {
      "id": "repo_xyz789",
      "github_id": 67890,
      "name": "api-service",
      "full_name": "acme/api-service",
      "default_branch": "main",
      "enabled": true,
      "config": {},
      "organization_id": "org_abc123",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### Get Repository

```
GET /repositories/{repo_id}
```

### Update Repository

```
PATCH /repositories/{repo_id}
Content-Type: application/json

{
  "enabled": true,
  "config": {
    "thresholds": {
      "critical": 0,
      "high": 0
    }
  }
}
```

---

## Analyses

### List Analyses

```
GET /analyses?repository_id={repo_id}&status={status}
```

**Query Parameters:**
- `repository_id` (optional): Filter by repository
- `status` (optional): Filter by status (`pending`, `running`, `completed`, `failed`)
- `limit` (optional): Number of results (default: 20, max: 100)
- `offset` (optional): Pagination offset

**Response:**
```json
{
  "data": [
    {
      "id": "analysis_123",
      "repository_id": "repo_xyz789",
      "pr_number": 42,
      "pr_title": "Add user auth",
      "head_sha": "abc123",
      "base_sha": "def456",
      "status": "completed",
      "conclusion": "passed",
      "started_at": "2024-01-15T10:30:00Z",
      "completed_at": "2024-01-15T10:32:34Z",
      "summary": {
        "total_issues": 2,
        "critical": 0,
        "high": 0,
        "medium": 1,
        "low": 1,
        "pass": true
      },
      "created_at": "2024-01-15T10:29:00Z"
    }
  ],
  "pagination": {
    "total": 156,
    "limit": 20,
    "offset": 0
  }
}
```

### Get Analysis

```
GET /analyses/{analysis_id}
```

### Get Analysis Findings

```
GET /analyses/{analysis_id}/findings
```

**Response:**
```json
{
  "data": [
    {
      "id": "finding_f1",
      "analysis_id": "analysis_123",
      "category": "security",
      "severity": "medium",
      "title": "SQL Injection Risk",
      "description": "User input used directly in query",
      "file_path": "src/db/users.py",
      "line_start": 42,
      "line_end": 42,
      "confidence": 0.87,
      "verification_type": "ai",
      "fix_suggestion": "Use parameterized queries",
      "metadata": {
        "owasp_category": "A03:2021",
        "cwe": "CWE-89"
      },
      "created_at": "2024-01-15T10:32:00Z"
    }
  ]
}
```

### Retry Analysis

```
POST /analyses/{analysis_id}/retry
```

Returns a new analysis object.

### Cancel Analysis

```
POST /analyses/{analysis_id}/cancel
```

---

## Statistics

### Dashboard Stats

```
GET /stats/dashboard?organization_id={org_id}
```

**Response:**
```json
{
  "total_analyses": 1234,
  "passed": 1150,
  "failed": 84,
  "total_findings": 4521,
  "by_severity": {
    "critical": 12,
    "high": 89,
    "medium": 1234,
    "low": 3186
  },
  "recent_activity": [
    {
      "id": "analysis_123",
      "repository_name": "api-service",
      "pr_number": 42,
      "conclusion": "passed",
      "completed_at": "2024-01-15T10:32:34Z"
    }
  ],
  "trends": {
    "pass_rate_7d": 0.93,
    "pass_rate_30d": 0.91,
    "avg_issues_per_pr": 1.8
  }
}
```

---

## Webhooks

Configure webhooks to receive real-time notifications.

### Create Webhook

```
POST /webhooks
Content-Type: application/json

{
  "url": "https://your-server.com/webhook",
  "events": ["analysis.completed", "analysis.failed"],
  "secret": "your-signing-secret"
}
```

### Webhook Events

- `analysis.started` - Analysis began
- `analysis.completed` - Analysis finished successfully
- `analysis.failed` - Analysis encountered an error
- `finding.created` - New finding detected

### Webhook Payload

```json
{
  "event": "analysis.completed",
  "timestamp": "2024-01-15T10:32:34Z",
  "data": {
    "analysis_id": "analysis_123",
    "repository": "acme/api-service",
    "pr_number": 42,
    "conclusion": "passed",
    "summary": {
      "total_issues": 2
    }
  }
}
```

### Verifying Webhooks

Webhooks include an `X-CodeVerify-Signature` header:

```python
import hmac
import hashlib

def verify_signature(payload, signature, secret):
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```

---

## Rate Limits

| Tier | Requests/min | Analyses/day |
|------|-------------|--------------|
| Free | 60 | 50 |
| Team | 300 | 500 |
| Enterprise | 1000 | Unlimited |

Rate limit headers:
- `X-RateLimit-Limit`: Max requests per window
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Unix timestamp when limit resets

---

## Errors

### Error Response Format

```json
{
  "error": {
    "code": "validation_error",
    "message": "Invalid repository ID",
    "details": {
      "field": "repository_id",
      "reason": "must be a valid UUID"
    }
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `unauthorized` | 401 | Invalid or missing API key |
| `forbidden` | 403 | Insufficient permissions |
| `not_found` | 404 | Resource not found |
| `validation_error` | 422 | Invalid request parameters |
| `rate_limited` | 429 | Too many requests |
| `internal_error` | 500 | Server error |

---

## SDKs

### Python

```bash
pip install codeverify
```

```python
from codeverify import CodeVerify

client = CodeVerify(api_key="YOUR_API_KEY")

# List analyses
analyses = client.analyses.list(repository_id="repo_xyz")

# Get findings
findings = client.analyses.get_findings("analysis_123")
```

### JavaScript/TypeScript

```bash
npm install @codeverify/sdk
```

```typescript
import { CodeVerify } from '@codeverify/sdk';

const client = new CodeVerify({ apiKey: 'YOUR_API_KEY' });

// List analyses
const analyses = await client.analyses.list({
  repositoryId: 'repo_xyz'
});

// Get findings
const findings = await client.analyses.getFindings('analysis_123');
```

---

## Export & Compliance

### Export Analyses to CSV

```
GET /export/analyses/csv?organization_id={org_id}&start_date={date}&end_date={date}
```

Downloads a CSV file containing analysis summaries.

### Export Findings to CSV

```
GET /export/findings/csv?organization_id={org_id}&severities=critical,high
```

**Query Parameters:**
- `severities` (optional): Comma-separated list (critical, high, medium, low)
- `categories` (optional): Comma-separated list (security, verification, semantic)
- `start_date` (optional): ISO 8601 date
- `end_date` (optional): ISO 8601 date

### Generate Compliance Report (PDF)

```
GET /export/report/pdf?organization_id={org_id}
```

Generates a comprehensive compliance report suitable for audits.

### Export Summary

```
GET /export/summary?organization_id={org_id}
```

Returns metadata about what will be exported:

```json
{
  "total_analyses": 1234,
  "total_findings": 5678,
  "severity_breakdown": {
    "critical": 12,
    "high": 89,
    "medium": 1234,
    "low": 4343
  },
  "export_formats": ["csv", "pdf"]
}
```

---

## Feedback & False Positives

### Submit Feedback

```
POST /feedback
Content-Type: application/json

{
  "finding_id": "finding_abc123",
  "feedback_type": "false_positive",
  "comment": "This is intentional test code",
  "finding_title": "SQL Injection Risk",
  "finding_category": "security"
}
```

**Feedback Types:**
- `false_positive` - Finding is incorrect
- `helpful` - Finding was useful
- `not_helpful` - Finding was not useful
- `incorrect_severity` - Severity should be different

### Dismiss Finding

```
POST /feedback/dismiss/{finding_id}
Content-Type: application/json

{
  "reason": "Test code, not production",
  "learn_pattern": true,
  "finding_title": "SQL Injection Risk",
  "finding_category": "security"
}
```

Setting `learn_pattern: true` helps improve future analysis accuracy.

### Get Learned Patterns

```
GET /feedback/patterns
```

Returns patterns learned from false positive reports:

```json
{
  "total_patterns": 42,
  "patterns": [
    {
      "pattern_hash": "abc123",
      "finding_title": "SQL Injection Risk",
      "finding_category": "security",
      "occurrence_count": 15,
      "confidence_adjustment": 0.3
    }
  ]
}
```

### Check Confidence Adjustment

```
GET /feedback/adjustment/{finding_title}?category=security
```

Returns confidence adjustment for a potential finding.

---

## SSO / SAML

Enterprise customers can configure SAML SSO.

### Configure SAML

```
POST /sso/config/{organization_id}
Content-Type: application/json

{
  "idp_entity_id": "https://idp.example.com/saml",
  "idp_sso_url": "https://idp.example.com/sso",
  "idp_certificate": "-----BEGIN CERTIFICATE-----\n...",
  "enforce_sso": true,
  "domains": ["example.com"]
}
```

### Get SP Metadata

```
GET /sso/metadata/{organization_id}
```

Returns XML metadata for IdP configuration.

### Check Domain SSO Requirement

```
GET /sso/domains?email=user@example.com
```

```json
{
  "sso_required": true,
  "sso_available": true,
  "organization_id": "org_abc123",
  "login_url": "/sso/login/org_abc123"
}
```

### Initiate SSO Login

```
GET /sso/login/{organization_id}?redirect_uri=/dashboard
```

Redirects to IdP for authentication.

---

## Usage & Billing

### Get Current Usage

```
GET /usage/current
```

```json
{
  "period_start": "2024-01-01T00:00:00Z",
  "period_end": "2024-01-31T23:59:59Z",
  "analyses_used": 342,
  "analyses_limit": 500,
  "llm_tokens_used": 1250000,
  "z3_verifications": 8500
}
```

### Get Usage History

```
GET /usage/history?months=6
```

Returns monthly usage breakdown.

---

## Changelog

### v1.3.0 (2026-03)
- Added export endpoints for compliance (CSV/PDF)
- Added SAML SSO support
- Enhanced feedback loop for false positives
- Added Go and Java language support

### v1.2.0 (2026-02)
- Added webhook support
- New `/stats/dashboard` endpoint

### v1.1.0 (2026-01)
- Added retry and cancel endpoints
- Improved error responses

### v1.0.0 (2025-12)
- Initial release
