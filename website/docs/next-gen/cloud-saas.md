---
sidebar_position: 2
---

# Zero-Config Cloud SaaS

Multi-tenant SaaS platform with OAuth, quotas, and tiered access control.

## Overview

The Cloud SaaS module provides a complete multi-tenant infrastructure for running CodeVerify as a hosted service. It handles tenant isolation, usage quotas, OAuth authentication, and tiered feature gating.

## Tenant Tiers

| Tier       | Analyses/Month | Max Files | Formal Verification | Priority |
|------------|:--------------:|:---------:|:-------------------:|:--------:|
| Free       | 100            | 50        | ❌                  | Low      |
| Pro        | 5,000          | 500       | ✅                  | Medium   |
| Enterprise | Unlimited      | Unlimited | ✅                  | High     |

## Quick Start

```python
from codeverify_core.cloud_saas import TenantManager, OAuthManager

# Initialize tenant management
manager = TenantManager()

# Create a new tenant
tenant = manager.create_tenant(
    name="Acme Corp",
    tier="pro",
    admin_email="admin@acme.com",
)

# Check quotas before analysis
if manager.check_quota(tenant.id):
    # Run analysis...
    manager.record_usage(tenant.id, operation="analysis")
```

## OAuth Integration

```python
oauth = OAuthManager(
    github_client_id="your-client-id",
    github_client_secret="your-secret",
)

# Generate authorization URL
auth_url = oauth.get_auth_url(provider="github", state="random-state")

# Exchange code for token
token = await oauth.exchange_code(provider="github", code="auth-code")
```

## Redis-Backed Persistence

For production deployments, use the Redis backend for tenant data:

```python
from codeverify_core.redis_backends import RedisTenantStore

store = RedisTenantStore(redis_url="redis://localhost:6379/2")

# Save tenant config
store.save_tenant(tenant.id, tenant.to_dict())

# Track usage
store.record_usage(tenant.id, {"operation": "analysis", "files": 42})
```

## Configuration

```yaml
# .codeverify.yml
saas:
  enabled: true
  default_tier: free
  
  quotas:
    free:
      max_analyses: 100
      max_files_per_analysis: 50
    pro:
      max_analyses: 5000
      max_files_per_analysis: 500
```

:::note
Self-hosted deployments can use the TenantManager for internal team isolation without needing the full OAuth flow.
:::
