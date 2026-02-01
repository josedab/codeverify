---
sidebar_position: 2
---

# Authentication

Secure your API requests with API keys.

## Getting an API Key

### Via Dashboard

1. Go to [app.codeverify.dev/settings/api](https://app.codeverify.dev/settings/api)
2. Click **Create API Key**
3. Name your key (e.g., "CI Pipeline")
4. Select permissions
5. Copy the key (shown only once)

### Via CLI

```bash
codeverify auth create-key --name "CI Pipeline" --scope analyses:write
```

## Using API Keys

### Header Authentication (Recommended)

```bash
curl -H "Authorization: Bearer cv_your_api_key" \
  https://api.codeverify.dev/v1/analyses
```

### Query Parameter

```bash
curl "https://api.codeverify.dev/v1/analyses?api_key=cv_your_api_key"
```

:::warning
Avoid query parameters in logs and browser history. Use headers when possible.
:::

## Key Format

API keys follow the format: `cv_[environment]_[random]`

- `cv_live_abc123...` — Production key
- `cv_test_xyz789...` — Test/sandbox key

## Permissions (Scopes)

| Scope | Description |
|-------|-------------|
| `analyses:read` | View analysis results |
| `analyses:write` | Trigger new analyses |
| `findings:read` | View findings |
| `findings:write` | Update finding status |
| `webhooks:read` | View webhooks |
| `webhooks:write` | Create/delete webhooks |
| `admin` | Full access |

### Create Scoped Key

```bash
codeverify auth create-key \
  --name "Read-only Dashboard" \
  --scope analyses:read,findings:read
```

## Key Management

### List Keys

```bash
codeverify auth list-keys
```

```
ID              Name            Scopes              Created         Last Used
────────────────────────────────────────────────────────────────────────────────
key_abc123      CI Pipeline     analyses:write      2024-01-15      2 hours ago
key_def456      Dashboard       analyses:read       2024-01-10      5 minutes ago
```

### Revoke Key

```bash
codeverify auth revoke-key key_abc123
```

### Rotate Key

```bash
codeverify auth rotate-key key_abc123
```

This creates a new key and revokes the old one.

## Environment Variables

Store keys in environment variables:

```bash
# .env (never commit this file)
CODEVERIFY_API_KEY=cv_live_abc123...
```

```bash
# Shell
export CODEVERIFY_API_KEY=cv_live_abc123...
```

### CI/CD Secrets

| Platform | Where to Store |
|----------|----------------|
| GitHub Actions | Repository Secrets |
| GitLab CI | CI/CD Variables (masked) |
| CircleCI | Environment Variables |
| Jenkins | Credentials |

Example GitHub Actions:

```yaml
- name: Run CodeVerify
  env:
    CODEVERIFY_API_KEY: ${{ secrets.CODEVERIFY_API_KEY }}
  run: codeverify analyze .
```

## Organization Keys

For organization-wide access:

```bash
codeverify auth create-key \
  --org acme-corp \
  --name "Org CI" \
  --scope analyses:write
```

Org keys can access all repositories in the organization.

## GitHub App Authentication

For GitHub integration, you can also use GitHub App tokens:

```yaml
# .codeverify.yml
auth:
  github_app: true
```

The GitHub App handles authentication automatically when installed.

## Security Best Practices

1. **Use minimal scopes** — Only request permissions you need
2. **Rotate regularly** — Rotate keys every 90 days
3. **Use separate keys** — Different keys for different environments
4. **Never commit keys** — Use environment variables or secrets
5. **Monitor usage** — Check last-used timestamps
6. **Revoke immediately** — If a key is compromised

## Troubleshooting

### 401 Unauthorized

```json
{
  "error": {
    "code": "unauthorized",
    "message": "Invalid API key"
  }
}
```

Causes:
- Key is incorrect or malformed
- Key has been revoked
- Key is for wrong environment (test vs live)

### 403 Forbidden

```json
{
  "error": {
    "code": "forbidden",
    "message": "Insufficient permissions"
  }
}
```

Causes:
- Key doesn't have required scope
- Key doesn't have access to this repository

## Next Steps

- [API Overview](/docs/api/overview) — API basics
- [Analyses](/docs/api/analyses) — Trigger analyses
- [Webhooks](/docs/api/webhooks) — Real-time notifications
