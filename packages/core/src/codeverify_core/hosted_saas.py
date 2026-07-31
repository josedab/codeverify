"""Multi-Tenant Hosted SaaS Platform.

Managed SaaS service with Stripe billing integration, tenant isolation,
usage metering, and plan management (Free/Pro/Enterprise).

Features:
- Multi-tenant isolation with per-tenant configuration
- Stripe-compatible billing with usage-based pricing
- Plan management (Free/Pro/Enterprise) with feature gates
- Usage metering and quota enforcement
- Tenant lifecycle management (provision, suspend, delete)
- Invoice generation and payment tracking
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class SaaSPlan(str, Enum):
    """Available SaaS plans."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class TenantStatus(str, Enum):
    """Tenant lifecycle status."""

    PROVISIONING = "provisioning"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEACTIVATED = "deactivated"
    DELETED = "deleted"


class BillingCycle(str, Enum):
    """Billing cycle options."""

    MONTHLY = "monthly"
    ANNUAL = "annual"


class PaymentStatus(str, Enum):
    """Payment status."""

    PENDING = "pending"
    PAID = "paid"
    FAILED = "failed"
    REFUNDED = "refunded"


class FeatureFlag(str, Enum):
    """Feature flags gated by plan."""

    FORMAL_VERIFICATION = "formal_verification"
    AI_ANALYSIS = "ai_analysis"
    CUSTOM_RULES = "custom_rules"
    SARIF_EXPORT = "sarif_export"
    API_ACCESS = "api_access"
    WEBHOOKS = "webhooks"
    SSO = "sso"
    AUDIT_LOG = "audit_log"
    PRIORITY_SUPPORT = "priority_support"
    CUSTOM_MODELS = "custom_models"


@dataclass
class PlanConfig:
    """Configuration for a SaaS plan."""

    plan: SaaSPlan = SaaSPlan.FREE
    price_monthly_cents: int = 0
    price_annual_cents: int = 0
    max_repos: int = 3
    max_users: int = 1
    verifications_per_month: int = 100
    ai_analyses_per_month: int = 50
    storage_gb: float = 1.0
    features: list[FeatureFlag] = field(default_factory=list)

    @classmethod
    def for_plan(cls, plan: SaaSPlan) -> PlanConfig:
        configs = {
            SaaSPlan.FREE: cls(
                plan=SaaSPlan.FREE,
                price_monthly_cents=0,
                price_annual_cents=0,
                max_repos=3,
                max_users=1,
                verifications_per_month=100,
                ai_analyses_per_month=50,
                storage_gb=1.0,
                features=[
                    FeatureFlag.FORMAL_VERIFICATION,
                    FeatureFlag.AI_ANALYSIS,
                ],
            ),
            SaaSPlan.PRO: cls(
                plan=SaaSPlan.PRO,
                price_monthly_cents=4900,
                price_annual_cents=47000,
                max_repos=50,
                max_users=25,
                verifications_per_month=10000,
                ai_analyses_per_month=5000,
                storage_gb=50.0,
                features=[
                    FeatureFlag.FORMAL_VERIFICATION,
                    FeatureFlag.AI_ANALYSIS,
                    FeatureFlag.CUSTOM_RULES,
                    FeatureFlag.SARIF_EXPORT,
                    FeatureFlag.API_ACCESS,
                    FeatureFlag.WEBHOOKS,
                ],
            ),
            SaaSPlan.ENTERPRISE: cls(
                plan=SaaSPlan.ENTERPRISE,
                price_monthly_cents=19900,
                price_annual_cents=190000,
                max_repos=-1,
                max_users=-1,
                verifications_per_month=-1,
                ai_analyses_per_month=-1,
                storage_gb=500.0,
                features=list(FeatureFlag),
            ),
        }
        return configs.get(plan, cls())


@dataclass
class Tenant:
    """A SaaS tenant (organization)."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    slug: str = ""
    owner_email: str = ""
    plan: SaaSPlan = SaaSPlan.FREE
    status: TenantStatus = TenantStatus.PROVISIONING
    stripe_customer_id: str = ""
    stripe_subscription_id: str = ""
    billing_cycle: BillingCycle = BillingCycle.MONTHLY
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    settings: dict[str, Any] = field(default_factory=dict)
    user_count: int = 1
    repo_count: int = 0


@dataclass
class UsageMeter:
    """Usage tracking for a tenant in a billing period."""

    tenant_id: str = ""
    period_start: datetime = field(default_factory=lambda: datetime.now(UTC))
    period_end: datetime = field(default_factory=lambda: datetime.now(UTC) + timedelta(days=30))
    verifications: int = 0
    ai_analyses: int = 0
    storage_used_gb: float = 0.0
    api_calls: int = 0

    def increment(self, metric: str, count: int = 1) -> None:
        if hasattr(self, metric):
            setattr(self, metric, getattr(self, metric) + count)


@dataclass
class Invoice:
    """A billing invoice."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    tenant_id: str = ""
    amount_cents: int = 0
    currency: str = "usd"
    status: PaymentStatus = PaymentStatus.PENDING
    period_start: datetime = field(default_factory=lambda: datetime.now(UTC))
    period_end: datetime = field(default_factory=lambda: datetime.now(UTC) + timedelta(days=30))
    line_items: list[dict[str, Any]] = field(default_factory=list)
    stripe_invoice_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    paid_at: datetime | None = None


class TenantProvisioner:
    """Handles tenant provisioning and lifecycle."""

    def provision(
        self,
        name: str,
        owner_email: str,
        plan: SaaSPlan = SaaSPlan.FREE,
    ) -> Tenant:
        """Provision a new tenant."""
        slug = name.lower().replace(" ", "-").replace("_", "-")
        tenant = Tenant(
            name=name,
            slug=slug,
            owner_email=owner_email,
            plan=plan,
            status=TenantStatus.ACTIVE,
        )
        logger.info("tenant_provisioned", tenant_id=tenant.id, name=name, plan=plan.value)
        return tenant

    def suspend(self, tenant: Tenant, reason: str = "") -> Tenant:
        tenant.status = TenantStatus.SUSPENDED
        tenant.settings["suspend_reason"] = reason
        return tenant

    def reactivate(self, tenant: Tenant) -> Tenant:
        tenant.status = TenantStatus.ACTIVE
        tenant.settings.pop("suspend_reason", None)
        return tenant

    def deactivate(self, tenant: Tenant) -> Tenant:
        tenant.status = TenantStatus.DEACTIVATED
        return tenant


class BillingEngine:
    """Handles billing, invoicing, and plan management."""

    def __init__(self) -> None:
        self._invoices: dict[str, list[Invoice]] = defaultdict(list)

    def change_plan(
        self,
        tenant: Tenant,
        new_plan: SaaSPlan,
        billing_cycle: BillingCycle = BillingCycle.MONTHLY,
    ) -> Tenant:
        """Change a tenant's plan."""
        old_plan = tenant.plan
        tenant.plan = new_plan
        tenant.billing_cycle = billing_cycle
        logger.info(
            "plan_changed",
            tenant_id=tenant.id,
            old_plan=old_plan.value,
            new_plan=new_plan.value,
        )
        return tenant

    def generate_invoice(self, tenant: Tenant, usage: UsageMeter) -> Invoice:
        """Generate an invoice for a billing period."""
        plan_config = PlanConfig.for_plan(tenant.plan)
        base_price = (
            plan_config.price_annual_cents // 12
            if tenant.billing_cycle == BillingCycle.ANNUAL
            else plan_config.price_monthly_cents
        )

        line_items: list[dict[str, Any]] = [
            {"description": f"{tenant.plan.value} plan", "amount": base_price}
        ]

        # Overage charges for Pro plan
        if tenant.plan == SaaSPlan.PRO:
            ver_limit = plan_config.verifications_per_month
            if ver_limit > 0 and usage.verifications > ver_limit:
                overage = usage.verifications - ver_limit
                overage_cost = overage * 1  # $0.01 per overage verification
                line_items.append(
                    {
                        "description": f"Verification overage ({overage} extra)",
                        "amount": overage_cost,
                    }
                )

        total = sum(int(item["amount"]) for item in line_items)

        invoice = Invoice(
            tenant_id=tenant.id,
            amount_cents=total,
            line_items=line_items,
            period_start=usage.period_start,
            period_end=usage.period_end,
        )
        self._invoices[tenant.id].append(invoice)
        return invoice

    def record_payment(self, invoice_id: str, tenant_id: str) -> bool:
        """Record a payment for an invoice."""
        for inv in self._invoices.get(tenant_id, []):
            if inv.id == invoice_id:
                inv.status = PaymentStatus.PAID
                inv.paid_at = datetime.now(UTC)
                return True
        return False

    def get_invoices(self, tenant_id: str) -> list[Invoice]:
        return self._invoices.get(tenant_id, [])


class FeatureGate:
    """Checks if a feature is available for a tenant's plan."""

    def is_enabled(self, tenant: Tenant, feature: FeatureFlag) -> bool:
        config = PlanConfig.for_plan(tenant.plan)
        return feature in config.features

    def check_quota(self, tenant: Tenant, usage: UsageMeter, metric: str) -> tuple[bool, int]:
        """Check if quota is available. Returns (allowed, remaining)."""
        config = PlanConfig.for_plan(tenant.plan)
        limit_map = {
            "verifications": config.verifications_per_month,
            "ai_analyses": config.ai_analyses_per_month,
        }
        limit = limit_map.get(metric, 0)
        if limit == -1:
            return True, -1
        current = getattr(usage, metric, 0)
        remaining = max(0, limit - current)
        return current < limit, remaining


class HostedSaaSService:
    """Main service for the hosted SaaS platform."""

    def __init__(self) -> None:
        self._provisioner = TenantProvisioner()
        self._billing = BillingEngine()
        self._feature_gate = FeatureGate()
        self._tenants: dict[str, Tenant] = {}
        self._usage: dict[str, UsageMeter] = {}

    @property
    def billing(self) -> BillingEngine:
        return self._billing

    @property
    def feature_gate(self) -> FeatureGate:
        return self._feature_gate

    def create_tenant(
        self,
        name: str,
        owner_email: str,
        plan: SaaSPlan = SaaSPlan.FREE,
    ) -> Tenant:
        """Create and provision a new tenant."""
        tenant = self._provisioner.provision(name, owner_email, plan)
        self._tenants[tenant.id] = tenant
        self._usage[tenant.id] = UsageMeter(tenant_id=tenant.id)
        return tenant

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        return self._tenants.get(tenant_id)

    def suspend_tenant(self, tenant_id: str, reason: str = "") -> bool:
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False
        self._provisioner.suspend(tenant, reason)
        return True

    def reactivate_tenant(self, tenant_id: str) -> bool:
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False
        self._provisioner.reactivate(tenant)
        return True

    def change_plan(
        self,
        tenant_id: str,
        new_plan: SaaSPlan,
        billing_cycle: BillingCycle = BillingCycle.MONTHLY,
    ) -> bool:
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False
        self._billing.change_plan(tenant, new_plan, billing_cycle)
        return True

    def record_usage(self, tenant_id: str, metric: str, count: int = 1) -> bool:
        """Record usage and check quota."""
        tenant = self._tenants.get(tenant_id)
        usage = self._usage.get(tenant_id)
        if not tenant or not usage:
            return False
        allowed, _ = self._feature_gate.check_quota(tenant, usage, metric)
        if allowed:
            usage.increment(metric, count)
            return True
        return False

    def get_usage(self, tenant_id: str) -> UsageMeter | None:
        return self._usage.get(tenant_id)

    def generate_invoice(self, tenant_id: str) -> Invoice | None:
        tenant = self._tenants.get(tenant_id)
        usage = self._usage.get(tenant_id)
        if not tenant or not usage:
            return None
        return self._billing.generate_invoice(tenant, usage)

    def is_feature_enabled(self, tenant_id: str, feature: FeatureFlag) -> bool:
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            return False
        return self._feature_gate.is_enabled(tenant, feature)

    def list_tenants(self, status: TenantStatus | None = None) -> list[Tenant]:
        tenants = list(self._tenants.values())
        if status:
            tenants = [t for t in tenants if t.status == status]
        return tenants

    def get_platform_stats(self) -> dict[str, Any]:
        tenants = list(self._tenants.values())
        active = [t for t in tenants if t.status == TenantStatus.ACTIVE]
        plan_dist: dict[str, int] = defaultdict(int)
        for t in active:
            plan_dist[t.plan.value] += 1

        return {
            "total_tenants": len(tenants),
            "active_tenants": len(active),
            "plan_distribution": dict(plan_dist),
            "total_users": sum(t.user_count for t in active),
            "total_repos": sum(t.repo_count for t in active),
        }


# ─── Singleton Access ──────────────────────────────────────────────────


_hosted_saas_instance: HostedSaaSService | None = None


def get_hosted_saas_service() -> HostedSaaSService:
    """Get or create the singleton HostedSaaSService."""
    global _hosted_saas_instance
    if _hosted_saas_instance is None:
        _hosted_saas_instance = HostedSaaSService()
    return _hosted_saas_instance


def reset_hosted_saas_service() -> None:
    """Reset the singleton (for testing)."""
    global _hosted_saas_instance
    _hosted_saas_instance = None
