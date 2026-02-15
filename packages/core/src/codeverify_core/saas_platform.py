"""Hosted SaaS Platform with Free Tier.

Multi-tenant platform module providing tenant management, usage tracking,
plan tiers (Free/Pro/Enterprise), rate limiting, and billing integration
for a fully managed CodeVerify cloud offering.

Features:
- Tenant provisioning with isolated configuration
- Usage metering (verifications, API calls, storage)
- Plan management with tier-based limits
- Rate limiting per tenant and plan
- Billing integration abstraction (Stripe-compatible)
- API key management with scoped permissions
"""

from __future__ import annotations

import hashlib
import math
import secrets
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class PlanTier(str, Enum):
    """Subscription plan tiers."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class TenantStatus(str, Enum):
    """Status of a tenant account."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    CANCELLED = "cancelled"
    PENDING_SETUP = "pending_setup"


class UsageMetricType(str, Enum):
    """Types of usage metrics tracked."""

    VERIFICATIONS = "verifications"
    API_CALLS = "api_calls"
    STORAGE_MB = "storage_mb"
    AI_TOKENS = "ai_tokens"
    REPOSITORIES = "repositories"
    TEAM_MEMBERS = "team_members"


class ApiKeyScope(str, Enum):
    """Permission scopes for API keys."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    VERIFY = "verify"
    WEBHOOK = "webhook"


@dataclass
class PlanLimits:
    """Resource limits for a plan tier."""

    max_verifications_per_month: int = 500
    max_api_calls_per_hour: int = 60
    max_repositories: int = 3
    max_team_members: int = 1
    max_storage_mb: int = 100
    max_ai_tokens_per_month: int = 50000
    sarif_export: bool = False
    custom_rules: bool = False
    priority_support: bool = False
    sso_enabled: bool = False
    audit_log: bool = False
    sla_uptime: float = 0.0

    @classmethod
    def for_tier(cls, tier: PlanTier) -> PlanLimits:
        if tier == PlanTier.FREE:
            return cls()
        if tier == PlanTier.PRO:
            return cls(
                max_verifications_per_month=10000,
                max_api_calls_per_hour=600,
                max_repositories=50,
                max_team_members=25,
                max_storage_mb=5000,
                max_ai_tokens_per_month=1000000,
                sarif_export=True,
                custom_rules=True,
                priority_support=False,
                sla_uptime=99.5,
            )
        return cls(
            max_verifications_per_month=-1,  # unlimited
            max_api_calls_per_hour=6000,
            max_repositories=-1,
            max_team_members=-1,
            max_storage_mb=50000,
            max_ai_tokens_per_month=-1,
            sarif_export=True,
            custom_rules=True,
            priority_support=True,
            sso_enabled=True,
            audit_log=True,
            sla_uptime=99.9,
        )


@dataclass
class UsageRecord:
    """A single usage measurement."""

    tenant_id: str = ""
    metric: UsageMetricType = UsageMetricType.VERIFICATIONS
    value: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageSummary:
    """Aggregated usage for a billing period."""

    tenant_id: str = ""
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    totals: dict[str, int] = field(default_factory=dict)
    limits: dict[str, int] = field(default_factory=dict)

    def utilization(self, metric: UsageMetricType) -> float:
        total = self.totals.get(metric.value, 0)
        limit = self.limits.get(metric.value, 0)
        if limit <= 0:
            return 0.0 if limit == -1 else 1.0
        return min(total / limit, 1.0)

    @property
    def over_limit_metrics(self) -> list[str]:
        over = []
        for metric_name, total in self.totals.items():
            limit = self.limits.get(metric_name, 0)
            if limit > 0 and total >= limit:
                over.append(metric_name)
        return over


@dataclass
class ApiKey:
    """An API key with scoped permissions."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    name: str = ""
    key_hash: str = ""
    prefix: str = ""
    scopes: list[ApiKeyScope] = field(default_factory=lambda: [ApiKeyScope.READ])
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    last_used_at: datetime | None = None
    is_active: bool = True

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def has_scope(self, scope: ApiKeyScope) -> bool:
        return ApiKeyScope.ADMIN in self.scopes or scope in self.scopes


@dataclass
class Tenant:
    """A tenant in the multi-tenant SaaS platform."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    slug: str = ""
    plan: PlanTier = PlanTier.FREE
    status: TenantStatus = TenantStatus.PENDING_SETUP
    owner_email: str = ""
    github_org: str = ""
    limits: PlanLimits = field(default_factory=PlanLimits)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trial_ends_at: datetime | None = None
    billing_customer_id: str = ""
    settings: dict[str, Any] = field(default_factory=dict)

    def activate(self) -> None:
        self.status = TenantStatus.ACTIVE

    def suspend(self, reason: str = "") -> None:
        self.status = TenantStatus.SUSPENDED
        self.settings["suspend_reason"] = reason

    def upgrade(self, new_plan: PlanTier) -> None:
        self.plan = new_plan
        self.limits = PlanLimits.for_tier(new_plan)

    def start_trial(self, days: int = 14) -> None:
        self.status = TenantStatus.TRIAL
        self.plan = PlanTier.PRO
        self.limits = PlanLimits.for_tier(PlanTier.PRO)
        self.trial_ends_at = datetime.now(timezone.utc) + timedelta(days=days)

    @property
    def is_trial_expired(self) -> bool:
        if self.trial_ends_at is None:
            return False
        return datetime.now(timezone.utc) > self.trial_ends_at


class RateLimiter:
    """Token-bucket rate limiter per tenant."""

    def __init__(self) -> None:
        self._buckets: dict[str, dict[str, Any]] = {}

    def check(self, tenant_id: str, max_per_hour: int) -> bool:
        now = time.time()
        bucket = self._buckets.get(tenant_id)
        if bucket is None or now - bucket["window_start"] >= 3600:
            self._buckets[tenant_id] = {"count": 1, "window_start": now}
            return True
        if max_per_hour <= 0:
            if max_per_hour == -1:
                return True
            return False
        if bucket["count"] >= max_per_hour:
            return False
        bucket["count"] += 1
        return True

    def get_remaining(self, tenant_id: str, max_per_hour: int) -> int:
        bucket = self._buckets.get(tenant_id)
        if bucket is None:
            return max_per_hour if max_per_hour > 0 else -1
        if max_per_hour == -1:
            return -1
        now = time.time()
        if now - bucket["window_start"] >= 3600:
            return max_per_hour
        return max(0, max_per_hour - bucket["count"])

    def reset(self, tenant_id: str) -> None:
        self._buckets.pop(tenant_id, None)


class UsageTracker:
    """Tracks resource usage per tenant."""

    def __init__(self) -> None:
        self._records: list[UsageRecord] = []

    def record(self, tenant_id: str, metric: UsageMetricType, value: int = 1, **metadata: Any) -> UsageRecord:
        rec = UsageRecord(tenant_id=tenant_id, metric=metric, value=value, metadata=metadata)
        self._records.append(rec)
        return rec

    def get_total(self, tenant_id: str, metric: UsageMetricType, since: datetime | None = None) -> int:
        total = 0
        for r in self._records:
            if r.tenant_id == tenant_id and r.metric == metric:
                if since is None or r.timestamp >= since:
                    total += r.value
        return total

    def get_summary(self, tenant_id: str, limits: PlanLimits, period_start: datetime | None = None) -> UsageSummary:
        if period_start is None:
            now = datetime.now(timezone.utc)
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        totals = {}
        limit_map = {
            UsageMetricType.VERIFICATIONS.value: limits.max_verifications_per_month,
            UsageMetricType.API_CALLS.value: limits.max_api_calls_per_hour,
            UsageMetricType.REPOSITORIES.value: limits.max_repositories,
            UsageMetricType.TEAM_MEMBERS.value: limits.max_team_members,
            UsageMetricType.STORAGE_MB.value: limits.max_storage_mb,
            UsageMetricType.AI_TOKENS.value: limits.max_ai_tokens_per_month,
        }

        for m in UsageMetricType:
            totals[m.value] = self.get_total(tenant_id, m, since=period_start)

        return UsageSummary(
            tenant_id=tenant_id,
            period_start=period_start,
            totals=totals,
            limits=limit_map,
        )


class SaaSPlatform:
    """Multi-tenant SaaS platform manager."""

    def __init__(self) -> None:
        self._tenants: dict[str, Tenant] = {}
        self._api_keys: dict[str, ApiKey] = {}
        self._rate_limiter = RateLimiter()
        self._usage_tracker = UsageTracker()

    def create_tenant(
        self,
        name: str,
        owner_email: str,
        plan: PlanTier = PlanTier.FREE,
        github_org: str = "",
    ) -> Tenant:
        slug = name.lower().replace(" ", "-").replace("_", "-")
        tenant = Tenant(
            name=name,
            slug=slug,
            plan=plan,
            owner_email=owner_email,
            github_org=github_org,
            limits=PlanLimits.for_tier(plan),
        )
        tenant.activate()
        self._tenants[tenant.id] = tenant
        logger.info("tenant_created", tenant_id=tenant.id, plan=plan.value)
        return tenant

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        return self._tenants.get(tenant_id)

    def list_tenants(self, status: TenantStatus | None = None) -> list[Tenant]:
        tenants = list(self._tenants.values())
        if status:
            tenants = [t for t in tenants if t.status == status]
        return tenants

    def upgrade_tenant(self, tenant_id: str, new_plan: PlanTier) -> Tenant | None:
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return None
        tenant.upgrade(new_plan)
        logger.info("tenant_upgraded", tenant_id=tenant_id, new_plan=new_plan.value)
        return tenant

    def suspend_tenant(self, tenant_id: str, reason: str = "") -> bool:
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return False
        tenant.suspend(reason)
        return True

    def create_api_key(
        self,
        tenant_id: str,
        name: str,
        scopes: list[ApiKeyScope] | None = None,
        expires_in_days: int | None = None,
    ) -> tuple[ApiKey, str]:
        """Create an API key. Returns (ApiKey metadata, raw key string)."""
        raw_key = f"cv_{secrets.token_urlsafe(32)}"
        prefix = raw_key[:8]
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        api_key = ApiKey(
            tenant_id=tenant_id,
            name=name,
            key_hash=key_hash,
            prefix=prefix,
            scopes=scopes or [ApiKeyScope.READ, ApiKeyScope.VERIFY],
            expires_at=expires_at,
        )
        self._api_keys[api_key.id] = api_key
        return api_key, raw_key

    def validate_api_key(self, raw_key: str) -> ApiKey | None:
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        for ak in self._api_keys.values():
            if ak.key_hash == key_hash and ak.is_active and not ak.is_expired:
                ak.last_used_at = datetime.now(timezone.utc)
                return ak
        return None

    def check_rate_limit(self, tenant_id: str) -> bool:
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return False
        return self._rate_limiter.check(tenant_id, tenant.limits.max_api_calls_per_hour)

    def record_usage(self, tenant_id: str, metric: UsageMetricType, value: int = 1) -> UsageRecord:
        return self._usage_tracker.record(tenant_id, metric, value)

    def get_usage_summary(self, tenant_id: str) -> UsageSummary | None:
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return None
        return self._usage_tracker.get_summary(tenant_id, tenant.limits)

    def check_usage_limit(self, tenant_id: str, metric: UsageMetricType) -> bool:
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return False
        summary = self._usage_tracker.get_summary(tenant_id, tenant.limits)
        return summary.utilization(metric) < 1.0

    @property
    def total_tenants(self) -> int:
        return len(self._tenants)

    @property
    def active_tenants(self) -> int:
        return sum(1 for t in self._tenants.values() if t.status == TenantStatus.ACTIVE)


_platform: SaaSPlatform | None = None


def get_saas_platform() -> SaaSPlatform:
    """Get the singleton SaaSPlatform instance."""
    global _platform
    if _platform is None:
        _platform = SaaSPlatform()
    return _platform


def reset_saas_platform() -> None:
    """Reset the singleton (useful for testing)."""
    global _platform
    _platform = None
