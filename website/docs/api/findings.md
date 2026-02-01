---
sidebar_position: 4
---

# Findings API

Access and manage verification findings.

## List Findings

Get findings for an analysis.

```http
GET /v1/analyses/{analysis_id}/findings
```

### Request

```bash
curl https://api.codeverify.dev/v1/analyses/an_abc123/findings \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY"
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `severity` | string | Filter: critical, high, medium, low |
| `category` | string | Filter: null_safety, array_bounds, etc. |
| `file` | string | Filter by file path |
| `status` | string | Filter: open, resolved, suppressed |
| `limit` | integer | Results per page (max 100) |
| `offset` | integer | Pagination offset |

### Response

```json
{
  "data": [
    {
      "id": "f_xyz789",
      "analysis_id": "an_abc123",
      "severity": "high",
      "category": "null_safety",
      "message": "Potential null dereference",
      "file": "src/api/handlers.py",
      "line": 45,
      "column": 12,
      "code_snippet": "return user.name",
      "status": "open",
      "created_at": "2024-01-15T10:32:00Z",
      "details": {
        "variable": "user",
        "type": "User | None",
        "counterexample": {
          "user": null
        }
      },
      "suggestion": {
        "description": "Add null check before accessing 'name'",
        "code": "if user is not None:\n    return user.name\nreturn None"
      }
    }
  ],
  "pagination": {
    "total": 3,
    "limit": 50,
    "offset": 0,
    "has_more": false
  }
}
```

## Get Finding

Get details for a specific finding.

```http
GET /v1/findings/{id}
```

### Request

```bash
curl https://api.codeverify.dev/v1/findings/f_xyz789 \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY"
```

### Response

```json
{
  "id": "f_xyz789",
  "analysis_id": "an_abc123",
  "repository": "owner/repo",
  "ref": "main",
  "commit_sha": "abc123def456",
  "severity": "high",
  "category": "null_safety",
  "message": "Potential null dereference",
  "description": "The variable 'user' may be None when accessing property 'name'. This can cause an AttributeError at runtime.",
  "file": "src/api/handlers.py",
  "line": 45,
  "column": 12,
  "end_line": 45,
  "end_column": 21,
  "code_snippet": "return user.name",
  "context": {
    "before": [
      "def get_user_name(user_id: str) -> str:",
      "    user = find_user(user_id)"
    ],
    "after": [
      "",
      "def process_user(user: User) -> None:"
    ]
  },
  "verification": {
    "type": "formal",
    "engine": "z3",
    "proof": {
      "query": "∃ user: user = None",
      "result": "sat",
      "counterexample": {
        "user": null
      }
    }
  },
  "suggestion": {
    "description": "Add null check before accessing 'name'",
    "code": "if user is not None:\n    return user.name\nreturn None",
    "confidence": 0.95
  },
  "references": [
    {
      "title": "Null Safety Documentation",
      "url": "https://codeverify.dev/docs/verification/null-safety"
    }
  ],
  "status": "open",
  "created_at": "2024-01-15T10:32:00Z"
}
```

## Update Finding Status

Update the status of a finding.

```http
PATCH /v1/findings/{id}
```

### Request

```bash
curl -X PATCH https://api.codeverify.dev/v1/findings/f_xyz789 \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "status": "resolved",
    "resolution": {
      "type": "fixed",
      "commit_sha": "def456abc789",
      "note": "Added null check"
    }
  }'
```

### Status Values

| Status | Description |
|--------|-------------|
| `open` | Not yet addressed |
| `resolved` | Fixed in code |
| `suppressed` | Intentionally ignored |
| `false_positive` | Incorrectly flagged |

### Resolution Types

| Type | Description |
|------|-------------|
| `fixed` | Code was corrected |
| `wont_fix` | Accepted risk |
| `false_positive` | Not a real issue |
| `duplicate` | Same as another finding |

### Response

```json
{
  "id": "f_xyz789",
  "status": "resolved",
  "resolution": {
    "type": "fixed",
    "commit_sha": "def456abc789",
    "note": "Added null check",
    "resolved_by": "user@example.com",
    "resolved_at": "2024-01-15T11:00:00Z"
  }
}
```

## Suppress Finding

Suppress a finding with a reason.

```http
POST /v1/findings/{id}/suppress
```

### Request

```bash
curl -X POST https://api.codeverify.dev/v1/findings/f_xyz789/suppress \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "reason": "false_positive",
    "note": "External validation ensures this is never null",
    "expires_at": "2024-06-15T00:00:00Z"
  }'
```

### Response

```json
{
  "id": "f_xyz789",
  "status": "suppressed",
  "suppression": {
    "reason": "false_positive",
    "note": "External validation ensures this is never null",
    "suppressed_by": "user@example.com",
    "suppressed_at": "2024-01-15T11:00:00Z",
    "expires_at": "2024-06-15T00:00:00Z"
  }
}
```

## Bulk Update

Update multiple findings at once.

```http
POST /v1/findings/bulk
```

### Request

```bash
curl -X POST https://api.codeverify.dev/v1/findings/bulk \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "finding_ids": ["f_xyz789", "f_abc123", "f_def456"],
    "action": "suppress",
    "data": {
      "reason": "false_positive",
      "note": "Bulk suppression for legacy code"
    }
  }'
```

## Search Findings

Search findings across all analyses.

```http
GET /v1/findings/search
```

### Request

```bash
curl "https://api.codeverify.dev/v1/findings/search?repository=owner/repo&severity=critical,high&status=open" \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY"
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `repository` | string | Filter by repository |
| `severity` | string | Comma-separated severities |
| `category` | string | Comma-separated categories |
| `status` | string | Comma-separated statuses |
| `file` | string | Filter by file pattern |
| `message` | string | Search in message text |
| `since` | datetime | Findings after this date |
| `limit` | integer | Results per page |

## Findings Statistics

Get aggregated statistics.

```http
GET /v1/findings/stats
```

### Request

```bash
curl "https://api.codeverify.dev/v1/findings/stats?repository=owner/repo&period=30d" \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY"
```

### Response

```json
{
  "period": "30d",
  "total": 156,
  "by_severity": {
    "critical": 2,
    "high": 15,
    "medium": 89,
    "low": 50
  },
  "by_category": {
    "null_safety": 45,
    "array_bounds": 32,
    "division_by_zero": 12,
    "security": 67
  },
  "by_status": {
    "open": 45,
    "resolved": 98,
    "suppressed": 13
  },
  "trend": {
    "new_findings": 23,
    "resolved_findings": 45,
    "net_change": -22
  }
}
```

## Export Findings

Export findings in various formats.

```http
GET /v1/analyses/{id}/findings/export
```

### SARIF Format

```bash
curl "https://api.codeverify.dev/v1/analyses/an_abc123/findings/export?format=sarif" \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY" \
  -o findings.sarif
```

### CSV Format

```bash
curl "https://api.codeverify.dev/v1/analyses/an_abc123/findings/export?format=csv" \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY" \
  -o findings.csv
```

### JSON Format

```bash
curl "https://api.codeverify.dev/v1/analyses/an_abc123/findings/export?format=json" \
  -H "Authorization: Bearer $CODEVERIFY_API_KEY" \
  -o findings.json
```

## Next Steps

- [Analyses API](/docs/api/analyses) — Trigger analyses
- [Webhooks](/docs/api/webhooks) — Real-time notifications
- [Understanding Findings](/docs/concepts/findings) — Findings explained
