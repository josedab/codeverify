"""Cloud SaaS Multi-Tenant Support.

Provides tenant isolation, GitHub OAuth flow, and multi-tenant
configuration for CodeVerify's hosted SaaS offering.

.. deprecated::
    This module is superseded by ``codeverify_core.hosted_saas``.
    It remains importable for backward compatibility but will be
    removed in a future release.
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "codeverify_core.cloud_saas is deprecated. Use codeverify_core.hosted_saas instead.",
    DeprecationWarning,
    stacklevel=2,
)


import hashlib
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class AuthProvider(str, Enum):
    """Supported OAuth providers."""

    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"
    GOOGLE = "google"


class TenantTier(str, Enum):
    """Tenant subscription tier."""

    FREE = "free"
    TEAM = "team"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


class TenantStatus(str, Enum):
    """Tenant lifecycle status."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    DEACTIVATED = "deactivated"


@dataclass
class TenantLimits:
    """Usage limits per tier."""

    analyses_per_month: int = 100
    repos_limit: int = 3
    team_members: int = 1
    retention_days: int = 30
    api_rate_limit_per_minute: int = 30
    max_file_size_kb: int = 500

    @staticmethod
    def for_tier(tier: TenantTier) -> TenantLimits:
        limits = {
            TenantTier.FREE: TenantLimits(100, 3, 1, 30, 30, 500),
            TenantTier.TEAM: TenantLimits(5000, 25, 10, 90, 120, 2000),
            TenantTier.BUSINESS: TenantLimits(50000, 100, 50, 365, 600, 5000),
            TenantTier.ENTERPRISE: TenantLimits(999999, 9999, 9999, 730, 3000, 10000),
        }
        return limits.get(tier, TenantLimits())


@dataclass
class Tenant:
    """A tenant (organization) in the SaaS platform."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    slug: str = ""
    tier: TenantTier = TenantTier.FREE
    status: TenantStatus = TenantStatus.TRIAL
    owner_email: str = ""
    auth_provider: AuthProvider = AuthProvider.GITHUB
    external_id: str = ""
    limits: TenantLimits = field(default_factory=TenantLimits)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    trial_ends_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_within_limits(self, analyses_used: int) -> bool:
        return analyses_used < self.limits.analyses_per_month

    def is_trial_expired(self) -> bool:
        if self.trial_ends_at is None:
            return False
        return datetime.now(UTC) > self.trial_ends_at


@dataclass
class OAuthToken:
    """OAuth token for a user session."""

    access_token: str
    refresh_token: str | None = None
    provider: AuthProvider = AuthProvider.GITHUB
    expires_at: datetime | None = None
    scopes: list[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(UTC) > self.expires_at


@dataclass
class TenantUser:
    """A user within a tenant."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str = ""
    email: str = ""
    display_name: str = ""
    role: str = "member"  # admin, member, viewer
    auth_provider: AuthProvider = AuthProvider.GITHUB
    external_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class UsageRecord:
    """Track usage for a tenant."""

    tenant_id: str = ""
    month: str = ""  # YYYY-MM
    analyses_count: int = 0
    api_calls: int = 0
    storage_bytes: int = 0


class TenantManager:
    """Manages tenant lifecycle and isolation."""

    def __init__(self) -> None:
        self._tenants: dict[str, Tenant] = {}
        self._users: dict[str, list[TenantUser]] = {}
        self._usage: dict[str, UsageRecord] = {}

    def create_tenant(
        self,
        name: str,
        owner_email: str,
        tier: TenantTier = TenantTier.FREE,
        auth_provider: AuthProvider = AuthProvider.GITHUB,
        external_id: str = "",
        trial_days: int = 14,
    ) -> Tenant:
        """Create a new tenant with trial period."""
        slug = name.lower().replace(" ", "-")
        tenant = Tenant(
            name=name,
            slug=slug,
            tier=tier,
            status=TenantStatus.TRIAL,
            owner_email=owner_email,
            auth_provider=auth_provider,
            external_id=external_id,
            limits=TenantLimits.for_tier(tier),
            trial_ends_at=datetime.now(UTC) + timedelta(days=trial_days),
        )
        self._tenants[tenant.id] = tenant
        logger.info("tenant_created", tenant_id=tenant.id, name=name, tier=tier.value)
        return tenant

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        return self._tenants.get(tenant_id)

    def upgrade_tier(self, tenant_id: str, new_tier: TenantTier) -> Tenant | None:
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return None
        tenant.tier = new_tier
        tenant.limits = TenantLimits.for_tier(new_tier)
        tenant.status = TenantStatus.ACTIVE
        tenant.trial_ends_at = None
        logger.info("tenant_upgraded", tenant_id=tenant_id, tier=new_tier.value)
        return tenant

    def suspend_tenant(self, tenant_id: str, reason: str = "") -> bool:
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return False
        tenant.status = TenantStatus.SUSPENDED
        tenant.metadata["suspend_reason"] = reason
        logger.warning("tenant_suspended", tenant_id=tenant_id, reason=reason)
        return True

    def add_user(
        self, tenant_id: str, email: str, display_name: str, role: str = "member"
    ) -> TenantUser | None:
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return None
        users = self._users.setdefault(tenant_id, [])
        if len(users) >= tenant.limits.team_members:
            logger.warning("tenant_user_limit", tenant_id=tenant_id)
            return None
        user = TenantUser(tenant_id=tenant_id, email=email, display_name=display_name, role=role)
        users.append(user)
        return user

    def get_users(self, tenant_id: str) -> list[TenantUser]:
        return list(self._users.get(tenant_id, []))

    def record_usage(self, tenant_id: str, analyses: int = 1) -> UsageRecord:
        month = datetime.now(UTC).strftime("%Y-%m")
        key = f"{tenant_id}:{month}"
        usage = self._usage.get(key)
        if usage is None:
            usage = UsageRecord(tenant_id=tenant_id, month=month)
            self._usage[key] = usage
        usage.analyses_count += analyses
        return usage

    def check_quota(self, tenant_id: str) -> bool:
        """Return True if tenant is within usage quota."""
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return False
        month = datetime.now(UTC).strftime("%Y-%m")
        key = f"{tenant_id}:{month}"
        usage = self._usage.get(key)
        used = usage.analyses_count if usage else 0
        return tenant.is_within_limits(used)

    def list_tenants(self, status: TenantStatus | None = None) -> list[Tenant]:
        tenants = list(self._tenants.values())
        if status:
            tenants = [t for t in tenants if t.status == status]
        return tenants


class OAuthManager:
    """Handles OAuth flow for GitHub/GitLab/Bitbucket."""

    def __init__(
        self, client_id: str = "", client_secret: str = "", redirect_uri: str = ""
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self._pending_states: dict[str, dict[str, Any]] = {}

    def generate_auth_url(
        self, provider: AuthProvider, scopes: list[str] | None = None
    ) -> dict[str, str]:
        """Generate OAuth authorization URL with state parameter."""
        state = secrets.token_urlsafe(32)
        scopes = scopes or ["repo", "read:user", "user:email"]

        urls = {
            AuthProvider.GITHUB: "https://github.com/login/oauth/authorize",
            AuthProvider.GITLAB: "https://gitlab.com/oauth/authorize",
            AuthProvider.BITBUCKET: "https://bitbucket.org/site/oauth2/authorize",
            AuthProvider.GOOGLE: "https://accounts.google.com/o/oauth2/v2/auth",
        }
        base_url = urls.get(provider, urls[AuthProvider.GITHUB])
        scope_str = " ".join(scopes)

        self._pending_states[state] = {"provider": provider, "scopes": scopes}

        auth_url = f"{base_url}?client_id={self.client_id}&redirect_uri={self.redirect_uri}&scope={scope_str}&state={state}"
        return {"url": auth_url, "state": state}

    def validate_callback(self, state: str, code: str) -> OAuthToken | None:
        """Validate OAuth callback and exchange code for token."""
        pending = self._pending_states.pop(state, None)
        if pending is None:
            return None
        # In production, this would exchange the code with the provider
        token = OAuthToken(
            access_token=hashlib.sha256(code.encode()).hexdigest(),
            provider=pending["provider"],
            scopes=pending["scopes"],
            expires_at=datetime.now(UTC) + timedelta(hours=8),
        )
        logger.info("oauth_token_issued", provider=pending["provider"].value)
        return token


# Singletons
_tenant_manager: TenantManager | None = None


def get_tenant_manager() -> TenantManager:
    global _tenant_manager
    if _tenant_manager is None:
        _tenant_manager = TenantManager()
    return _tenant_manager


def reset_tenant_manager() -> None:
    global _tenant_manager
    _tenant_manager = None
