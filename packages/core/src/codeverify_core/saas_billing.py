"""SaaS Billing Engine - Multi-tenant subscription management and billing.

Implements pricing plans, subscription lifecycle, usage metering,
invoice generation, and SSO connectors for CodeVerify's SaaS platform.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PlanType(str, Enum):
    """Available pricing plan tiers."""

    FREE = "free"
    STARTER = "starter"
    TEAM = "team"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class BillingCycle(str, Enum):
    """Billing frequency."""

    MONTHLY = "monthly"
    ANNUAL = "annual"


class PaymentStatus(str, Enum):
    """Status of a payment or invoice."""

    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REFUNDED = "refunded"
    DISPUTED = "disputed"


class SubscriptionStatus(str, Enum):
    """Lifecycle state of a subscription."""

    TRIALING = "trialing"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    PAUSED = "paused"


class SSOProvider(str, Enum):
    """Supported SSO / identity providers."""

    OKTA = "okta"
    AZURE_AD = "azure_ad"
    GOOGLE_WORKSPACE = "google_workspace"
    ONELOGIN = "onelogin"
    CUSTOM_SAML = "custom_saml"


class UsageMetric(str, Enum):
    """Metered resource dimensions."""

    VERIFICATIONS = "verifications"
    USERS = "users"
    REPOSITORIES = "repositories"
    API_CALLS = "api_calls"
    STORAGE_MB = "storage_mb"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

# Plan-type ordering for upgrade / downgrade validation
_PLAN_ORDER: dict[PlanType, int] = {
    PlanType.FREE: 0,
    PlanType.STARTER: 1,
    PlanType.TEAM: 2,
    PlanType.ENTERPRISE: 3,
    PlanType.CUSTOM: 4,
}


@dataclass
class PricingPlan:
    """A pricing plan with included quotas and feature set."""

    id: str
    name: str
    plan_type: PlanType
    price_monthly: float
    price_annual: float
    included_verifications: int
    included_users: int
    included_repos: int
    features: list[str] = field(default_factory=list)
    overage_rate: float = 0.0
    trial_days: int = 14

    def effective_price(self, cycle: BillingCycle) -> float:
        """Return the price for the given billing cycle."""
        return self.price_annual if cycle == BillingCycle.ANNUAL else self.price_monthly

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "plan_type": self.plan_type.value,
            "price_monthly": self.price_monthly,
            "price_annual": self.price_annual,
            "included_verifications": self.included_verifications,
            "included_users": self.included_users,
            "included_repos": self.included_repos,
            "features": self.features,
            "overage_rate": self.overage_rate,
        }


@dataclass
class Subscription:
    """A tenant's active subscription record."""

    id: str
    tenant_id: str
    plan: PricingPlan
    status: SubscriptionStatus
    billing_cycle: BillingCycle
    current_period_start: datetime
    current_period_end: datetime
    payment_method_id: str | None = None
    cancel_at_period_end: bool = False
    trial_end: datetime | None = None

    @property
    def is_trialing(self) -> bool:
        now = datetime.now(UTC)
        return (
            self.status == SubscriptionStatus.TRIALING
            and self.trial_end is not None
            and now < self.trial_end
        )

    @property
    def days_remaining(self) -> int:
        return max(0, (self.current_period_end - datetime.now(UTC)).days)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "plan": self.plan.to_dict(),
            "status": self.status.value,
            "billing_cycle": self.billing_cycle.value,
            "current_period_start": self.current_period_start.isoformat(),
            "current_period_end": self.current_period_end.isoformat(),
            "cancel_at_period_end": self.cancel_at_period_end,
            "days_remaining": self.days_remaining,
        }


@dataclass
class UsageRecord:
    """A single usage data-point for a tenant metric."""

    tenant_id: str
    metric: UsageMetric
    value: int
    timestamp: datetime
    period: str  # e.g. "2026-02"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "metric": self.metric.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "period": self.period,
        }


@dataclass
class Invoice:
    """An invoice generated for a billing period."""

    id: str
    tenant_id: str
    subscription_id: str
    amount: float
    currency: str = "usd"
    status: PaymentStatus = PaymentStatus.PENDING
    line_items: list[dict[str, Any]] = field(default_factory=list)
    issued_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    due_at: datetime | None = None
    paid_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "subscription_id": self.subscription_id,
            "amount": round(self.amount, 2),
            "currency": self.currency,
            "status": self.status.value,
            "line_items": self.line_items,
            "issued_at": self.issued_at.isoformat(),
            "due_at": self.due_at.isoformat() if self.due_at else None,
            "paid_at": self.paid_at.isoformat() if self.paid_at else None,
        }


@dataclass
class SSOConfig:
    """SSO / SAML configuration for a tenant."""

    tenant_id: str
    provider: SSOProvider
    enabled: bool = False
    client_id: str = ""
    issuer_url: str = ""
    metadata_url: str = ""
    domain_restriction: str | None = None
    auto_provision: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "provider": self.provider.value,
            "enabled": self.enabled,
            "client_id": self.client_id,
            "issuer_url": self.issuer_url,
            "metadata_url": self.metadata_url,
            "domain_restriction": self.domain_restriction,
            "auto_provision": self.auto_provision,
        }


@dataclass
class BillingReport:
    """Summary billing report for a tenant period."""

    tenant_id: str
    period: str
    plan_name: str
    base_charge: float
    overage_charges: float
    total: float
    usage_summary: dict[str, int] = field(default_factory=dict)
    projected_next_month: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "period": self.period,
            "plan_name": self.plan_name,
            "base_charge": round(self.base_charge, 2),
            "overage_charges": round(self.overage_charges, 2),
            "total": round(self.total, 2),
            "usage_summary": self.usage_summary,
            "projected_next_month": round(self.projected_next_month, 2),
        }


# ---------------------------------------------------------------------------
# Plan Catalog
# ---------------------------------------------------------------------------


class PlanCatalog:
    """Manages available pricing plans."""

    def __init__(self) -> None:
        self._plans: dict[PlanType, PricingPlan] = {}
        self._init_default_plans()

    def _init_default_plans(self) -> None:
        self._plans[PlanType.FREE] = PricingPlan(
            id="plan_free",
            name="Free",
            plan_type=PlanType.FREE,
            price_monthly=0.0,
            price_annual=0.0,
            included_verifications=100,
            included_users=3,
            included_repos=2,
            features=["basic_scanning", "pattern_matching"],
            overage_rate=0.0,
            trial_days=0,
        )
        self._plans[PlanType.STARTER] = PricingPlan(
            id="plan_starter",
            name="Starter",
            plan_type=PlanType.STARTER,
            price_monthly=29.0,
            price_annual=290.0,
            included_verifications=1000,
            included_users=10,
            included_repos=10,
            features=["basic_scanning", "pattern_matching", "ai_analysis", "api_access"],
            overage_rate=0.03,
            trial_days=14,
        )
        self._plans[PlanType.TEAM] = PricingPlan(
            id="plan_team",
            name="Team",
            plan_type=PlanType.TEAM,
            price_monthly=79.0,
            price_annual=790.0,
            included_verifications=5000,
            included_users=50,
            included_repos=50,
            features=[
                "basic_scanning",
                "pattern_matching",
                "ai_analysis",
                "api_access",
                "custom_rules",
                "priority_support",
            ],
            overage_rate=0.02,
            trial_days=14,
        )
        self._plans[PlanType.ENTERPRISE] = PricingPlan(
            id="plan_enterprise",
            name="Enterprise",
            plan_type=PlanType.ENTERPRISE,
            price_monthly=249.0,
            price_annual=2490.0,
            included_verifications=50000,
            included_users=999999,
            included_repos=999999,
            features=[
                "basic_scanning",
                "pattern_matching",
                "ai_analysis",
                "api_access",
                "custom_rules",
                "priority_support",
                "formal_verification",
                "sso",
                "audit_logs",
                "dedicated_support",
                "sla_guarantee",
            ],
            overage_rate=0.01,
            trial_days=30,
        )

    def get_plan(self, plan_type: PlanType) -> PricingPlan:
        """Return the plan definition for *plan_type*."""
        plan = self._plans.get(plan_type)
        if plan is None:
            raise ValueError(f"Unknown plan type: {plan_type}")
        return plan

    def get_all_plans(self) -> list[PricingPlan]:
        """Return all available plans ordered by price."""
        return sorted(self._plans.values(), key=lambda p: p.price_monthly)

    def compare_plans(self) -> list[dict[str, Any]]:
        """Return a comparison matrix of all plans."""
        return [
            {
                "name": p.name,
                "plan_type": p.plan_type.value,
                "price_monthly": p.price_monthly,
                "price_annual": p.price_annual,
                "verifications": p.included_verifications,
                "users": p.included_users,
                "repos": p.included_repos,
                "features": p.features,
                "overage_rate": p.overage_rate,
            }
            for p in self.get_all_plans()
        ]


# ---------------------------------------------------------------------------
# Subscription Manager
# ---------------------------------------------------------------------------


class SubscriptionManager:
    """Manages subscription lifecycle (create, upgrade, downgrade, cancel, renew)."""

    def __init__(self) -> None:
        self._catalog = PlanCatalog()
        self._subscriptions: dict[str, Subscription] = {}
        self._tenant_subs: dict[str, str] = {}  # tenant_id -> subscription_id

    def _period_end(self, start: datetime, cycle: BillingCycle) -> datetime:
        days = 365 if cycle == BillingCycle.ANNUAL else 30
        return start + timedelta(days=days)

    def _get_sub(self, subscription_id: str) -> Subscription:
        sub = self._subscriptions.get(subscription_id)
        if sub is None:
            raise KeyError(f"Subscription '{subscription_id}' not found")
        return sub

    def _prorate_credit(self, sub: Subscription) -> float:
        """Calculate unused-period credit for the current subscription."""
        total_days = (sub.current_period_end - sub.current_period_start).days
        if total_days <= 0:
            return 0.0
        period_price = sub.plan.effective_price(sub.billing_cycle)
        return round(period_price * (sub.days_remaining / total_days), 2)

    def create_subscription(
        self,
        tenant_id: str,
        plan_type: PlanType = PlanType.FREE,
        billing_cycle: BillingCycle = BillingCycle.MONTHLY,
    ) -> Subscription:
        """Create a new subscription for a tenant."""
        if tenant_id in self._tenant_subs:
            raise ValueError(f"Tenant '{tenant_id}' already has a subscription")

        plan = self._catalog.get_plan(plan_type)
        now = datetime.now(UTC)
        sub_id = f"sub_{uuid.uuid4().hex[:12]}"

        trial_end = None
        status = SubscriptionStatus.ACTIVE
        if plan.trial_days > 0 and plan_type != PlanType.FREE:
            trial_end = now + timedelta(days=plan.trial_days)
            status = SubscriptionStatus.TRIALING

        sub = Subscription(
            id=sub_id,
            tenant_id=tenant_id,
            plan=plan,
            status=status,
            billing_cycle=billing_cycle,
            current_period_start=now,
            current_period_end=self._period_end(now, billing_cycle),
            trial_end=trial_end,
        )
        self._subscriptions[sub_id] = sub
        self._tenant_subs[tenant_id] = sub_id
        logger.info(
            "subscription_created",
            tenant_id=tenant_id,
            plan=plan_type.value,
            cycle=billing_cycle.value,
            subscription_id=sub_id,
        )
        return sub

    def upgrade(self, subscription_id: str, new_plan_type: PlanType) -> Subscription:
        """Upgrade a subscription with proration credit applied."""
        sub = self._get_sub(subscription_id)
        if _PLAN_ORDER.get(new_plan_type, 0) <= _PLAN_ORDER.get(sub.plan.plan_type, 0):
            raise ValueError(
                f"Cannot upgrade from {sub.plan.plan_type.value} to {new_plan_type.value}"
            )

        credit = self._prorate_credit(sub)
        now = datetime.now(UTC)
        sub.plan = self._catalog.get_plan(new_plan_type)
        sub.current_period_start = now
        sub.current_period_end = self._period_end(now, sub.billing_cycle)
        sub.status = SubscriptionStatus.ACTIVE
        sub.trial_end = None
        logger.info(
            "subscription_upgraded",
            subscription_id=subscription_id,
            new_plan=new_plan_type.value,
            proration_credit=credit,
        )
        return sub

    def downgrade(self, subscription_id: str, new_plan_type: PlanType) -> Subscription:
        """Schedule a downgrade effective at the end of the current period."""
        sub = self._get_sub(subscription_id)
        if _PLAN_ORDER.get(new_plan_type, 0) >= _PLAN_ORDER.get(sub.plan.plan_type, 0):
            raise ValueError(
                f"Cannot downgrade from {sub.plan.plan_type.value} to {new_plan_type.value}"
            )

        sub.plan = self._catalog.get_plan(new_plan_type)
        sub.cancel_at_period_end = False
        logger.info(
            "subscription_downgraded",
            subscription_id=subscription_id,
            new_plan=new_plan_type.value,
            effective_at=sub.current_period_end.isoformat(),
        )
        return sub

    def cancel(self, subscription_id: str, at_period_end: bool = True) -> Subscription:
        """Cancel a subscription immediately or at end of period."""
        sub = self._get_sub(subscription_id)
        if at_period_end:
            sub.cancel_at_period_end = True
            logger.info(
                "subscription_cancel_scheduled",
                subscription_id=subscription_id,
                effective_at=sub.current_period_end.isoformat(),
            )
        else:
            sub.status = SubscriptionStatus.CANCELED
            sub.cancel_at_period_end = False
            logger.info("subscription_canceled_immediately", subscription_id=subscription_id)
        return sub

    def renew(self, subscription_id: str) -> Subscription:
        """Renew a subscription for another period."""
        sub = self._get_sub(subscription_id)
        sub.cancel_at_period_end = False
        now = datetime.now(UTC)
        sub.current_period_start = now
        sub.current_period_end = self._period_end(now, sub.billing_cycle)
        sub.status = SubscriptionStatus.ACTIVE
        logger.info(
            "subscription_renewed",
            subscription_id=subscription_id,
            new_period_end=sub.current_period_end.isoformat(),
        )
        return sub

    def check_limits(self, tenant_id: str, metric: UsageMetric) -> tuple[bool, int, int]:
        """Return (within_limit, current_usage_placeholder, limit) for a metric."""
        sub_id = self._tenant_subs.get(tenant_id)
        if sub_id is None:
            raise KeyError(f"No subscription found for tenant '{tenant_id}'")
        plan = self._subscriptions[sub_id].plan
        limit_map: dict[UsageMetric, int] = {
            UsageMetric.VERIFICATIONS: plan.included_verifications,
            UsageMetric.USERS: plan.included_users,
            UsageMetric.REPOSITORIES: plan.included_repos,
            UsageMetric.API_CALLS: plan.included_verifications * 10,
            UsageMetric.STORAGE_MB: plan.included_repos * 500,
        }
        return True, 0, limit_map.get(metric, 0)

    def get_subscription_for_tenant(self, tenant_id: str) -> Subscription | None:
        """Look up the active subscription for *tenant_id*."""
        sub_id = self._tenant_subs.get(tenant_id)
        return self._subscriptions.get(sub_id) if sub_id else None


# ---------------------------------------------------------------------------
# Usage Meter
# ---------------------------------------------------------------------------


class UsageMeter:
    """Tracks and meters tenant resource usage by period."""

    def __init__(self) -> None:
        self._sub_manager: SubscriptionManager | None = None
        self._usage: dict[str, dict[str, dict[UsageMetric, int]]] = {}
        self._records: list[UsageRecord] = []

    def set_subscription_manager(self, mgr: SubscriptionManager) -> None:
        """Wire up the subscription manager for quota look-ups."""
        self._sub_manager = mgr

    @staticmethod
    def _current_period() -> str:
        now = datetime.now(UTC)
        return f"{now.year}-{now.month:02d}"

    def record_usage(self, tenant_id: str, metric: UsageMetric, amount: int = 1) -> UsageRecord:
        """Record *amount* units of *metric* for *tenant_id*."""
        period = self._current_period()
        now = datetime.now(UTC)
        period_usage = self._usage.setdefault(tenant_id, {}).setdefault(period, {})
        period_usage[metric] = period_usage.get(metric, 0) + amount

        record = UsageRecord(
            tenant_id=tenant_id, metric=metric, value=amount, timestamp=now, period=period
        )
        self._records.append(record)
        logger.debug(
            "usage_recorded",
            tenant_id=tenant_id,
            metric=metric.value,
            amount=amount,
            period=period,
            new_total=period_usage[metric],
        )
        return record

    def get_usage(self, tenant_id: str, metric: UsageMetric, period: str | None = None) -> int:
        """Return total usage of *metric* for *tenant_id* in *period*."""
        period = period or self._current_period()
        return self._usage.get(tenant_id, {}).get(period, {}).get(metric, 0)

    def get_usage_summary(self, tenant_id: str, period: str | None = None) -> dict[str, int]:
        """Return usage across all metrics for a tenant period."""
        period = period or self._current_period()
        raw = self._usage.get(tenant_id, {}).get(period, {})
        return {m.value: raw.get(m, 0) for m in UsageMetric}

    def check_quota(self, tenant_id: str, metric: UsageMetric) -> tuple[bool, int, int]:
        """Return (within_limit, used, limit) for *metric*."""
        if self._sub_manager is None:
            raise RuntimeError("SubscriptionManager not configured on UsageMeter")
        _, _, limit = self._sub_manager.check_limits(tenant_id, metric)
        used = self.get_usage(tenant_id, metric)
        return used < limit, used, limit


# ---------------------------------------------------------------------------
# Invoice Generator
# ---------------------------------------------------------------------------


class InvoiceGenerator:
    """Generates invoices based on subscription and usage."""

    def __init__(self) -> None:
        self._sub_manager: SubscriptionManager | None = None
        self._usage_meter: UsageMeter | None = None
        self._invoices: dict[str, Invoice] = {}

    def set_subscription_manager(self, mgr: SubscriptionManager) -> None:
        self._sub_manager = mgr

    def set_usage_meter(self, meter: UsageMeter) -> None:
        self._usage_meter = meter

    def generate_invoice(self, subscription: Subscription, usage: dict[str, int]) -> Invoice:
        """Create an invoice for the current period."""
        plan = subscription.plan
        base = plan.effective_price(subscription.billing_cycle)
        line_items: list[dict[str, Any]] = [
            {
                "description": f"{plan.name} plan ({subscription.billing_cycle.value})",
                "amount": base,
            },
        ]
        overage = self.calculate_overages(plan, usage)
        if overage > 0:
            line_items.append({"description": "Overage charges", "amount": round(overage, 2)})

        total = round(base + overage, 2)
        now = datetime.now(UTC)
        inv = Invoice(
            id=f"inv_{uuid.uuid4().hex[:12]}",
            tenant_id=subscription.tenant_id,
            subscription_id=subscription.id,
            amount=total,
            line_items=line_items,
            issued_at=now,
            due_at=now + timedelta(days=15),
        )
        self._invoices[inv.id] = inv
        logger.info(
            "invoice_generated", invoice_id=inv.id, tenant_id=subscription.tenant_id, amount=total
        )
        return inv

    def calculate_overages(self, plan: PricingPlan, usage: dict[str, int]) -> float:
        """Calculate overage charges for usage that exceeds plan limits."""
        if plan.overage_rate <= 0:
            return 0.0
        verifications_used = usage.get(UsageMetric.VERIFICATIONS.value, 0)
        excess = max(0, verifications_used - plan.included_verifications)
        return round(excess * plan.overage_rate, 2)

    def get_billing_report(self, tenant_id: str, period: str) -> BillingReport:
        """Generate a billing report for *tenant_id* in *period*."""
        if self._sub_manager is None or self._usage_meter is None:
            raise RuntimeError("InvoiceGenerator requires sub_manager and usage_meter")
        sub = self._sub_manager.get_subscription_for_tenant(tenant_id)
        if sub is None:
            raise KeyError(f"No subscription for tenant '{tenant_id}'")

        plan = sub.plan
        base = plan.effective_price(sub.billing_cycle)
        usage_summary = self._usage_meter.get_usage_summary(tenant_id, period)
        overage = self.calculate_overages(plan, usage_summary)
        return BillingReport(
            tenant_id=tenant_id,
            period=period,
            plan_name=plan.name,
            base_charge=base,
            overage_charges=overage,
            total=round(base + overage, 2),
            usage_summary=usage_summary,
            projected_next_month=self.project_next_month(tenant_id),
        )

    def project_next_month(self, tenant_id: str) -> float:
        """Estimate next month's charge based on current usage velocity."""
        if self._sub_manager is None or self._usage_meter is None:
            return 0.0
        sub = self._sub_manager.get_subscription_for_tenant(tenant_id)
        if sub is None:
            return 0.0
        plan = sub.plan
        base = plan.effective_price(sub.billing_cycle)
        day_of_month = max(datetime.now(UTC).day, 1)
        verifications = self._usage_meter.get_usage(tenant_id, UsageMetric.VERIFICATIONS)
        projected = int(verifications * (30 / day_of_month))
        excess = max(0, projected - plan.included_verifications)
        return round(base + excess * plan.overage_rate, 2)


# ---------------------------------------------------------------------------
# SSO Manager
# ---------------------------------------------------------------------------


class SSOManager:
    """Manages SSO / SAML integration for enterprise tenants."""

    def __init__(self) -> None:
        self._configs: dict[str, SSOConfig] = {}

    def configure_sso(
        self,
        tenant_id: str,
        provider: SSOProvider,
        client_id: str,
        issuer_url: str,
        **kwargs: Any,
    ) -> SSOConfig:
        """Create or update SSO configuration for a tenant."""
        config = SSOConfig(
            tenant_id=tenant_id,
            provider=provider,
            enabled=True,
            client_id=client_id,
            issuer_url=issuer_url,
            metadata_url=kwargs.get("metadata_url", ""),
            domain_restriction=kwargs.get("domain_restriction"),
            auto_provision=kwargs.get("auto_provision", True),
        )
        self._configs[tenant_id] = config
        logger.info("sso_configured", tenant_id=tenant_id, provider=provider.value)
        return config

    def validate_sso_token(self, tenant_id: str, token: str) -> dict[str, Any] | None:
        """Validate an SSO token and return decoded claims.

        In production this would verify the JWT signature against the
        issuer's JWKS endpoint.  Here we perform basic structural checks.
        """
        config = self._configs.get(tenant_id)
        if config is None or not config.enabled:
            logger.warning("sso_validation_failed", tenant_id=tenant_id, reason="no_config")
            return None
        if not token or len(token) < 10:
            logger.warning("sso_validation_failed", tenant_id=tenant_id, reason="invalid_token")
            return None

        domain = config.domain_restriction or "example.com"
        claims: dict[str, Any] = {
            "sub": f"user@{domain}",
            "iss": config.issuer_url,
            "aud": config.client_id,
            "tenant_id": tenant_id,
            "provider": config.provider.value,
        }
        if config.domain_restriction:
            email_domain = claims["sub"].split("@")[-1]
            if email_domain != config.domain_restriction:
                logger.warning(
                    "sso_domain_mismatch",
                    tenant_id=tenant_id,
                    expected=config.domain_restriction,
                    got=email_domain,
                )
                return None
        logger.info("sso_token_validated", tenant_id=tenant_id)
        return claims

    def get_sso_config(self, tenant_id: str) -> SSOConfig | None:
        """Return the SSO config for *tenant_id*, if any."""
        return self._configs.get(tenant_id)

    def disable_sso(self, tenant_id: str) -> bool:
        """Disable SSO for *tenant_id*."""
        config = self._configs.get(tenant_id)
        if config is None:
            return False
        config.enabled = False
        logger.info("sso_disabled", tenant_id=tenant_id)
        return True


# ---------------------------------------------------------------------------
# SaaS Billing Engine (top-level orchestrator)
# ---------------------------------------------------------------------------


class SaaSBillingEngine:
    """Top-level orchestrator for SaaS billing operations.

    Wires together PlanCatalog, SubscriptionManager, UsageMeter,
    InvoiceGenerator, and SSOManager.
    """

    def __init__(self) -> None:
        self.catalog = PlanCatalog()
        self.subscriptions = SubscriptionManager()
        self.usage = UsageMeter()
        self.invoices = InvoiceGenerator()
        self.sso = SSOManager()
        # Wire components
        self.usage.set_subscription_manager(self.subscriptions)
        self.invoices.set_subscription_manager(self.subscriptions)
        self.invoices.set_usage_meter(self.usage)

    def onboard_tenant(
        self, tenant_name: str, plan_type: PlanType = PlanType.FREE
    ) -> dict[str, Any]:
        """Provision a new tenant with a subscription."""
        tenant_id = f"tenant_{uuid.uuid4().hex[:12]}"
        sub = self.subscriptions.create_subscription(tenant_id, plan_type)
        logger.info(
            "tenant_onboarded", tenant_id=tenant_id, tenant_name=tenant_name, plan=plan_type.value
        )
        return {"tenant_id": tenant_id, "tenant_name": tenant_name, "subscription": sub.to_dict()}

    def process_verification(self, tenant_id: str) -> tuple[bool, str]:
        """Check quota and record a verification. Returns (allowed, reason)."""
        within, used, limit = self.usage.check_quota(tenant_id, UsageMetric.VERIFICATIONS)
        if not within:
            sub = self.subscriptions.get_subscription_for_tenant(tenant_id)
            if sub and sub.plan.overage_rate > 0:
                self.usage.record_usage(tenant_id, UsageMetric.VERIFICATIONS)
                return True, f"Overage: {used + 1}/{limit} (${sub.plan.overage_rate}/extra)"
            return False, f"Quota exceeded: {used}/{limit} verifications used"
        self.usage.record_usage(tenant_id, UsageMetric.VERIFICATIONS)
        return True, f"OK: {used + 1}/{limit} verifications used"

    def get_dashboard_data(self, tenant_id: str) -> dict[str, Any]:
        """Return a summary dashboard payload for the tenant."""
        sub = self.subscriptions.get_subscription_for_tenant(tenant_id)
        if sub is None:
            raise KeyError(f"No subscription for tenant '{tenant_id}'")

        quotas: dict[str, dict[str, Any]] = {}
        for metric in [UsageMetric.VERIFICATIONS, UsageMetric.USERS, UsageMetric.REPOSITORIES]:
            within, used, limit = self.usage.check_quota(tenant_id, metric)
            quotas[metric.value] = {
                "used": used,
                "limit": limit,
                "within_limit": within,
                "percent_used": round(used / limit * 100, 1) if limit > 0 else 0.0,
            }

        sso_config = self.sso.get_sso_config(tenant_id)
        return {
            "tenant_id": tenant_id,
            "subscription": sub.to_dict(),
            "usage": self.usage.get_usage_summary(tenant_id),
            "quotas": quotas,
            "projected_cost": self.invoices.project_next_month(tenant_id),
            "sso_enabled": sso_config.enabled if sso_config else False,
            "plan_comparison": self.catalog.compare_plans(),
        }
