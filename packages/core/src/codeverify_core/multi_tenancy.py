"""Multi-Tenancy Module - SaaS tenant management and isolation.

Provides tenant lifecycle management, usage tracking, resource limits,
and data isolation for CodeVerify's multi-tenant SaaS offering.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class TenantTier(str, Enum):
    """Subscription tier for a tenant."""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class TenantConfig:
    """Configuration and metadata for a single tenant."""
    id: str
    name: str
    slug: str
    tier: TenantTier

    # Resource limits
    max_repos: int
    max_analyses_per_month: int
    max_users: int

    # Feature configuration
    features: list[str] = field(default_factory=list)
    custom_models: list[str] = field(default_factory=list)

    # Capability flags
    sso_enabled: bool = False
    audit_logs_enabled: bool = False

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TenantLimits:
    """Tier-specific default resource limits."""
    max_repos: int
    max_analyses_per_month: int
    max_users: int
    sso_enabled: bool
    audit_logs_enabled: bool
    default_features: list[str] = field(default_factory=list)

    @classmethod
    def for_tier(cls, tier: TenantTier) -> TenantLimits:
        """Return default limits for a subscription tier."""
        if tier == TenantTier.FREE:
            return cls(
                max_repos=3,
                max_analyses_per_month=100,
                max_users=5,
                sso_enabled=False,
                audit_logs_enabled=False,
                default_features=["basic_scanning", "pattern_matching"],
            )
        if tier == TenantTier.PRO:
            return cls(
                max_repos=25,
                max_analyses_per_month=2000,
                max_users=50,
                sso_enabled=False,
                audit_logs_enabled=True,
                default_features=[
                    "basic_scanning", "pattern_matching", "ai_analysis",
                    "custom_rules", "api_access", "priority_support",
                ],
            )
        if tier == TenantTier.ENTERPRISE:
            return cls(
                max_repos=999999,
                max_analyses_per_month=50000,
                max_users=999999,
                sso_enabled=True,
                audit_logs_enabled=True,
                default_features=[
                    "basic_scanning", "pattern_matching", "ai_analysis",
                    "custom_rules", "api_access", "priority_support",
                    "formal_verification", "custom_models",
                    "dedicated_support", "sla_guarantee",
                ],
            )
        raise ValueError(f"Unknown tenant tier: {tier}")


@dataclass
class _TenantUsage:
    """Internal mutable usage counters for a single tenant."""
    analyses_used: int = 0
    repos_count: int = 0
    users_count: int = 0
    last_reset: datetime = field(default_factory=datetime.utcnow)


class UsageTracker:
    """Tracks per-tenant resource consumption against plan limits."""

    _RESOURCE_LIMIT_MAP: dict[str, str] = {
        "analyses": "max_analyses_per_month",
        "repos": "max_repos",
        "users": "max_users",
    }

    _RESOURCE_USAGE_MAP: dict[str, str] = {
        "analyses": "analyses_used",
        "repos": "repos_count",
        "users": "users_count",
    }

    def __init__(self) -> None:
        """Initialise an empty usage tracker."""
        self._usage: dict[str, _TenantUsage] = {}
        self._tenants: dict[str, TenantConfig] = {}

    def register_tenant(self, tenant: TenantConfig) -> None:
        """Register a tenant so its limits are available for checks."""
        self._tenants[tenant.id] = tenant
        if tenant.id not in self._usage:
            self._usage[tenant.id] = _TenantUsage()

    def unregister_tenant(self, tenant_id: str) -> None:
        """Remove a tenant from the tracker."""
        self._tenants.pop(tenant_id, None)
        self._usage.pop(tenant_id, None)

    def check_limit(self, tenant_id: str, resource: str) -> tuple[bool, int]:
        """Check whether a tenant may consume more of a resource.

        Returns (allowed, remaining) where allowed is True when the
        tenant still has capacity for the given resource.
        """
        if resource not in self._RESOURCE_LIMIT_MAP:
            raise ValueError(
                f"Unknown resource '{resource}'. "
                f"Must be one of {list(self._RESOURCE_LIMIT_MAP)}"
            )
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            raise KeyError(f"Tenant '{tenant_id}' is not registered")

        usage = self._usage[tenant_id]
        limit_value: int = getattr(tenant, self._RESOURCE_LIMIT_MAP[resource])
        current_value: int = getattr(usage, self._RESOURCE_USAGE_MAP[resource])
        remaining = max(0, limit_value - current_value)
        allowed = remaining > 0

        logger.debug(
            "Limit check performed",
            tenant_id=tenant_id,
            resource=resource,
            current=current_value,
            limit=limit_value,
            allowed=allowed,
        )
        return allowed, remaining

    def record_usage(self, tenant_id: str, resource: str, count: int = 1) -> None:
        """Record consumption of a resource for a tenant."""
        if resource not in self._RESOURCE_USAGE_MAP:
            raise ValueError(
                f"Unknown resource '{resource}'. "
                f"Must be one of {list(self._RESOURCE_USAGE_MAP)}"
            )
        if tenant_id not in self._usage:
            raise KeyError(f"Tenant '{tenant_id}' is not registered")

        usage = self._usage[tenant_id]
        usage_attr = self._RESOURCE_USAGE_MAP[resource]
        current = getattr(usage, usage_attr)
        setattr(usage, usage_attr, current + count)

        logger.info(
            "Usage recorded",
            tenant_id=tenant_id,
            resource=resource,
            count=count,
            new_total=current + count,
        )

    def get_usage_report(self, tenant_id: str) -> dict[str, Any]:
        """Build a usage summary with limits and utilisation percentages."""
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            raise KeyError(f"Tenant '{tenant_id}' is not registered")

        usage = self._usage[tenant_id]
        report: dict[str, Any] = {
            "tenant_id": tenant_id,
            "tenant_name": tenant.name,
            "tier": tenant.tier.value,
            "period_start": usage.last_reset.isoformat(),
            "resources": {},
        }

        for resource, usage_attr in self._RESOURCE_USAGE_MAP.items():
            limit_attr = self._RESOURCE_LIMIT_MAP[resource]
            current = getattr(usage, usage_attr)
            limit = getattr(tenant, limit_attr)
            utilisation = (current / limit * 100) if limit > 0 else 0.0
            report["resources"][resource] = {
                "used": current,
                "limit": limit,
                "remaining": max(0, limit - current),
                "utilisation_percent": round(utilisation, 1),
            }

        return report

    def reset_monthly_counters(self, tenant_id: str) -> None:
        """Reset the monthly analysis counter at the start of a billing period."""
        usage = self._usage.get(tenant_id)
        if usage is not None:
            usage.analyses_used = 0
            usage.last_reset = datetime.utcnow()
            logger.info("Monthly counters reset", tenant_id=tenant_id)


class TenantManager:
    """Manages the full lifecycle of tenants.

    Provides CRUD operations over an in-memory tenant store and
    wires up the associated UsageTracker automatically.
    """

    def __init__(self, usage_tracker: UsageTracker | None = None) -> None:
        """Initialise the tenant manager."""
        self._tenants: dict[str, TenantConfig] = {}
        self._usage_tracker = usage_tracker or UsageTracker()

    @property
    def usage_tracker(self) -> UsageTracker:
        """Return the usage tracker bound to this manager."""
        return self._usage_tracker

    def create_tenant(
        self,
        name: str,
        slug: str,
        tier: TenantTier = TenantTier.FREE,
    ) -> TenantConfig:
        """Create a new tenant with tier-appropriate defaults."""
        for existing in self._tenants.values():
            if existing.slug == slug:
                raise ValueError(f"Tenant slug '{slug}' is already in use")

        limits = TenantLimits.for_tier(tier)
        tenant_id = str(uuid.uuid4())

        tenant = TenantConfig(
            id=tenant_id,
            name=name,
            slug=slug,
            tier=tier,
            max_repos=limits.max_repos,
            max_analyses_per_month=limits.max_analyses_per_month,
            max_users=limits.max_users,
            features=list(limits.default_features),
            custom_models=[],
            sso_enabled=limits.sso_enabled,
            audit_logs_enabled=limits.audit_logs_enabled,
        )

        self._tenants[tenant_id] = tenant
        self._usage_tracker.register_tenant(tenant)

        logger.info(
            "Tenant created",
            tenant_id=tenant_id,
            name=name,
            slug=slug,
            tier=tier.value,
        )
        return tenant

    def get_tenant(self, tenant_id: str) -> TenantConfig | None:
        """Retrieve a tenant by its identifier."""
        return self._tenants.get(tenant_id)

    def update_tier(self, tenant_id: str, new_tier: TenantTier) -> TenantConfig:
        """Change a tenant's subscription tier and adjust limits accordingly."""
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            raise KeyError(f"Tenant '{tenant_id}' not found")

        old_tier = tenant.tier
        limits = TenantLimits.for_tier(new_tier)

        tenant.tier = new_tier
        tenant.max_repos = limits.max_repos
        tenant.max_analyses_per_month = limits.max_analyses_per_month
        tenant.max_users = limits.max_users
        tenant.features = list(limits.default_features)
        tenant.sso_enabled = limits.sso_enabled
        tenant.audit_logs_enabled = limits.audit_logs_enabled
        tenant.updated_at = datetime.utcnow()

        # Re-register so the usage tracker sees updated limits.
        self._usage_tracker.register_tenant(tenant)

        logger.info(
            "Tenant tier updated",
            tenant_id=tenant_id,
            old_tier=old_tier.value,
            new_tier=new_tier.value,
        )
        return tenant

    def list_tenants(self) -> list[TenantConfig]:
        """Return all tenants ordered by creation date."""
        return sorted(self._tenants.values(), key=lambda t: t.created_at)

    def delete_tenant(self, tenant_id: str) -> None:
        """Permanently remove a tenant and its usage records."""
        if tenant_id not in self._tenants:
            raise KeyError(f"Tenant '{tenant_id}' not found")

        tenant = self._tenants.pop(tenant_id)
        self._usage_tracker.unregister_tenant(tenant_id)

        logger.info(
            "Tenant deleted",
            tenant_id=tenant_id,
            name=tenant.name,
            slug=tenant.slug,
        )


class TenantIsolation:
    """Data isolation primitives for multi-tenant deployments.

    Generates namespaced keys for cache/database isolation, validates
    cross-tenant access attempts, and maps tenants to encryption key
    identifiers for data-at-rest protection.
    """

    def __init__(
        self,
        namespace_prefix: str = "cv",
        manager: TenantManager | None = None,
    ) -> None:
        """Initialise the isolation helper."""
        self._namespace_prefix = namespace_prefix
        self._manager = manager

    def get_namespace(self, tenant_id: str) -> str:
        """Derive a namespace for Redis/DB isolation (``{prefix}:tenant:{id}``)."""
        namespace = f"{self._namespace_prefix}:tenant:{tenant_id}"
        logger.debug("Namespace resolved", tenant_id=tenant_id, namespace=namespace)
        return namespace

    def validate_access(self, tenant_id: str, resource_tenant_id: str) -> bool:
        """Check whether a tenant may access another tenant's resource.

        Returns True only when both ids match and, if a TenantManager
        was provided, the tenant exists.
        """
        if tenant_id != resource_tenant_id:
            logger.warning(
                "Cross-tenant access denied",
                requesting_tenant=tenant_id,
                resource_tenant=resource_tenant_id,
            )
            return False

        if self._manager is not None:
            if self._manager.get_tenant(tenant_id) is None:
                logger.warning("Access denied for unknown tenant", tenant_id=tenant_id)
                return False

        return True

    def get_encryption_key_id(self, tenant_id: str) -> str:
        """Derive a deterministic encryption key id (``cmk-{hash}``) for KMS lookup."""
        digest = hashlib.sha256(
            f"codeverify:encryption:{tenant_id}".encode()
        ).hexdigest()[:16]
        key_id = f"cmk-{digest}"
        logger.debug("Encryption key id resolved", tenant_id=tenant_id, key_id=key_id)
        return key_id
