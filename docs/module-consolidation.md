# Module Consolidation Guide

This document maps deprecated modules to their canonical replacements.
All deprecated modules emit `DeprecationWarning` on import and will be
removed in v2.0.0.

## Migration Table

| Deprecated Module | → Use Instead | Migration Notes |
|-------------------|---------------|-----------------|
| `saas_platform` | `hosted_saas` | `SaaSPlatformService` → `HostedSaaSService` |
| `cloud_saas` | `hosted_saas` | `CloudSaaSManager` → `HostedSaaSService` |
| `multi_tenancy` | `hosted_saas` | `TenantManager` → `HostedSaaSService.create_tenant()` |
| `proof_marketplace` | `proof_artifact_marketplace` | `ProofMarketplace` → `ProofArtifactMarketplaceService` |
| `proof_marketplace_v2` | `proof_artifact_marketplace` | Same as above, v2 merged |
| `budget_marketplace` | `proof_artifact_marketplace` | Budget features in `ProofArtifactMarketplaceService` |
| `compliance_as_code` | `compliance_engine` | `ComplianceFramework` → `ComplianceAsCodeService` |
| `compliance_framework` | `compliance_engine` | `EnterpriseComplianceFramework` → `ComplianceAsCodeService` |
| `compliance_reports` | `compliance_engine` | Reports via `ComplianceAsCodeService.run_framework_audit()` |
| `autofix_loop` | `autofix_verified_patches` | `AutoFixPipeline` → `AutofixVerifiedService` |
| `autofix_validation` | `autofix_verified_patches` | `FixValidator` → `AutofixVerifiedService.generate_fix()` |

## How to Migrate

### Before (deprecated)
```python
from codeverify_core.saas_platform import SaaSPlatformService
svc = SaaSPlatformService()
```

### After (canonical)
```python
from codeverify_core.hosted_saas import HostedSaaSService
svc = HostedSaaSService()
```

## Timeline

- **v1.6.0**: Deprecated modules emit `DeprecationWarning`
- **v2.0.0**: Deprecated modules will be removed
