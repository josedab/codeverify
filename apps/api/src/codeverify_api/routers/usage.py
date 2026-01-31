"""Usage and billing router."""
from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from codeverify_api.auth.dependencies import get_current_user
from codeverify_api.db.database import get_db
from codeverify_api.db.models import User
from codeverify_api.services.usage_service import (
    usage_service,
    PlanTier,
    PlanLimits,
)
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/usage", tags=["usage"])


@router.get("/summary")
async def get_usage_summary(
    organization_id: UUID | None = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Get usage summary for the current billing period."""
    # Use first org if not specified
    org_id = organization_id or UUID("00000000-0000-0000-0000-000000000000")
    
    summary = usage_service.get_usage_summary(org_id)
    billing = usage_service.get_billing_info(org_id)
    limits = PlanLimits.for_tier(billing.tier)
    
    return {
        "organization_id": str(org_id),
        "period": {
            "start": billing.current_period_start.isoformat(),
            "end": billing.current_period_end.isoformat(),
        },
        "usage": {
            "analyses": {
                "used": summary.analyses_count,
                "limit": limits.analyses_per_month,
                "remaining": max(0, limits.analyses_per_month - summary.analyses_count),
            },
            "repositories": {
                "used": summary.repositories_count,
                "limit": limits.repositories,
            },
            "api_calls": summary.api_calls_count,
            "findings": summary.findings_count,
        },
        "plan": {
            "tier": billing.tier.value,
            "limits": {
                "analyses_per_month": limits.analyses_per_month,
                "repositories": limits.repositories,
                "users": limits.users,
                "retention_days": limits.retention_days,
            },
        },
    }


@router.get("/history")
async def get_usage_history(
    organization_id: UUID | None = Query(None),
    months: int = Query(6, ge=1, le=12),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Get usage history for past months."""
    org_id = organization_id or UUID("00000000-0000-0000-0000-000000000000")
    
    history = []
    now = datetime.utcnow()
    
    for i in range(months):
        # Calculate period for each month
        month = now.month - i
        year = now.year
        while month <= 0:
            month += 12
            year -= 1
        
        period_start = datetime(year, month, 1)
        if month == 12:
            period_end = datetime(year + 1, 1, 1)
        else:
            period_end = datetime(year, month + 1, 1)
        
        summary = usage_service.get_usage_summary(org_id, period_start, period_end)
        
        history.append({
            "period": f"{year}-{month:02d}",
            "analyses": summary.analyses_count,
            "findings": summary.findings_count,
            "api_calls": summary.api_calls_count,
        })
    
    return {
        "organization_id": str(org_id),
        "history": list(reversed(history)),
    }


@router.get("/billing")
async def get_billing_info(
    organization_id: UUID | None = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Get billing information."""
    org_id = organization_id or UUID("00000000-0000-0000-0000-000000000000")
    
    billing = usage_service.get_billing_info(org_id)
    
    return {
        "organization_id": str(org_id),
        "tier": billing.tier.value,
        "billing_email": billing.billing_email,
        "current_period": {
            "start": billing.current_period_start.isoformat(),
            "end": billing.current_period_end.isoformat(),
        },
        "plans": [
            {
                "tier": "free",
                "name": "Free",
                "price": 0,
                "features": [
                    "50 analyses/month",
                    "5 repositories",
                    "3 users",
                    "7 day retention",
                ],
            },
            {
                "tier": "team",
                "name": "Team",
                "price": 79,
                "price_unit": "per user/month",
                "features": [
                    "500 analyses/month",
                    "50 repositories",
                    "25 users",
                    "90 day retention",
                    "Priority support",
                ],
            },
            {
                "tier": "enterprise",
                "name": "Enterprise",
                "price": None,
                "price_unit": "custom",
                "features": [
                    "Unlimited analyses",
                    "Unlimited repositories",
                    "Unlimited users",
                    "1 year retention",
                    "SSO/SAML",
                    "Dedicated support",
                    "SLA guarantee",
                ],
            },
        ],
    }


@router.post("/upgrade")
async def request_upgrade(
    organization_id: UUID,
    target_tier: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Request a plan upgrade (placeholder for Stripe integration)."""
    try:
        tier = PlanTier(target_tier)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid tier")
    
    if tier == PlanTier.ENTERPRISE:
        return {
            "status": "contact_sales",
            "message": "Please contact sales@codeverify.io for Enterprise plans",
        }
    
    # In production, this would create a Stripe checkout session
    return {
        "status": "redirect",
        "checkout_url": f"https://checkout.codeverify.io/upgrade?org={organization_id}&tier={tier.value}",
        "message": "Redirecting to checkout...",
    }
