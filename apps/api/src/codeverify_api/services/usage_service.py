"""Usage tracking and billing service for CodeVerify."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4
from enum import Enum

from pydantic import BaseModel, Field


class PlanTier(str, Enum):
    """Subscription plan tiers."""
    FREE = "free"
    TEAM = "team"
    ENTERPRISE = "enterprise"


class PlanLimits(BaseModel):
    """Limits for each plan tier."""
    
    analyses_per_month: int
    repositories: int
    users: int
    retention_days: int
    
    @classmethod
    def for_tier(cls, tier: PlanTier) -> "PlanLimits":
        """Get limits for a tier."""
        limits = {
            PlanTier.FREE: cls(
                analyses_per_month=50,
                repositories=5,
                users=3,
                retention_days=7,
            ),
            PlanTier.TEAM: cls(
                analyses_per_month=500,
                repositories=50,
                users=25,
                retention_days=90,
            ),
            PlanTier.ENTERPRISE: cls(
                analyses_per_month=999999,  # Unlimited
                repositories=999999,
                users=999999,
                retention_days=365,
            ),
        }
        return limits[tier]


class UsageRecord(BaseModel):
    """Record of resource usage."""
    
    id: UUID = Field(default_factory=uuid4)
    organization_id: UUID
    resource_type: str  # analysis, finding, api_call, storage
    quantity: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class UsageSummary(BaseModel):
    """Summary of usage for a billing period."""
    
    organization_id: UUID
    period_start: datetime
    period_end: datetime
    analyses_count: int = 0
    findings_count: int = 0
    api_calls_count: int = 0
    storage_bytes: int = 0
    repositories_count: int = 0
    users_count: int = 0


class BillingInfo(BaseModel):
    """Billing information for an organization."""
    
    organization_id: UUID
    tier: PlanTier = PlanTier.FREE
    billing_email: str | None = None
    stripe_customer_id: str | None = None
    current_period_start: datetime = Field(default_factory=datetime.utcnow)
    current_period_end: datetime = Field(
        default_factory=lambda: datetime.utcnow() + timedelta(days=30)
    )


class UsageService:
    """Service for tracking usage and managing billing."""
    
    def __init__(self):
        self._usage_records: list[UsageRecord] = []
        self._billing_info: dict[UUID, BillingInfo] = {}
    
    def record_usage(
        self,
        organization_id: UUID,
        resource_type: str,
        quantity: int = 1,
        metadata: dict[str, Any] | None = None
    ) -> UsageRecord:
        """Record a usage event."""
        record = UsageRecord(
            organization_id=organization_id,
            resource_type=resource_type,
            quantity=quantity,
            metadata=metadata or {},
        )
        self._usage_records.append(record)
        return record
    
    def get_usage_summary(
        self,
        organization_id: UUID,
        period_start: datetime | None = None,
        period_end: datetime | None = None
    ) -> UsageSummary:
        """Get usage summary for an organization."""
        # Default to current month
        now = datetime.utcnow()
        if period_start is None:
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if period_end is None:
            period_end = now
        
        # Filter records for this org and period
        records = [
            r for r in self._usage_records
            if r.organization_id == organization_id
            and period_start <= r.timestamp <= period_end
        ]
        
        # Aggregate by type
        summary = UsageSummary(
            organization_id=organization_id,
            period_start=period_start,
            period_end=period_end,
        )
        
        for record in records:
            if record.resource_type == "analysis":
                summary.analyses_count += record.quantity
            elif record.resource_type == "finding":
                summary.findings_count += record.quantity
            elif record.resource_type == "api_call":
                summary.api_calls_count += record.quantity
            elif record.resource_type == "storage":
                summary.storage_bytes += record.quantity
        
        return summary
    
    def check_limits(
        self,
        organization_id: UUID,
        resource_type: str
    ) -> tuple[bool, str]:
        """Check if organization is within usage limits."""
        billing = self._billing_info.get(organization_id, BillingInfo(
            organization_id=organization_id
        ))
        
        limits = PlanLimits.for_tier(billing.tier)
        summary = self.get_usage_summary(
            organization_id,
            billing.current_period_start,
            billing.current_period_end
        )
        
        if resource_type == "analysis":
            if summary.analyses_count >= limits.analyses_per_month:
                return False, f"Analysis limit reached ({limits.analyses_per_month}/month)"
        elif resource_type == "repository":
            if summary.repositories_count >= limits.repositories:
                return False, f"Repository limit reached ({limits.repositories})"
        
        return True, "OK"
    
    def get_billing_info(self, organization_id: UUID) -> BillingInfo:
        """Get billing info for an organization."""
        return self._billing_info.get(
            organization_id,
            BillingInfo(organization_id=organization_id)
        )
    
    def update_billing_tier(
        self,
        organization_id: UUID,
        tier: PlanTier
    ) -> BillingInfo:
        """Update organization's billing tier."""
        billing = self.get_billing_info(organization_id)
        billing.tier = tier
        self._billing_info[organization_id] = billing
        return billing


# Global instance
usage_service = UsageService()


# Helper functions
def record_analysis_usage(organization_id: UUID, analysis_id: UUID):
    """Record an analysis was run."""
    usage_service.record_usage(
        organization_id,
        "analysis",
        metadata={"analysis_id": str(analysis_id)}
    )


def record_finding_usage(organization_id: UUID, finding_count: int):
    """Record findings were created."""
    usage_service.record_usage(
        organization_id,
        "finding",
        quantity=finding_count
    )


def record_api_call(organization_id: UUID, endpoint: str):
    """Record an API call."""
    usage_service.record_usage(
        organization_id,
        "api_call",
        metadata={"endpoint": endpoint}
    )


def check_can_run_analysis(organization_id: UUID) -> tuple[bool, str]:
    """Check if organization can run another analysis."""
    return usage_service.check_limits(organization_id, "analysis")
