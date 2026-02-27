"""Stripe billing integration service for CodeVerify SaaS.

Provides abstractions for Stripe API operations: customer management,
subscription lifecycle, metered billing, and invoice handling.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SubscriptionStatus(str, Enum):
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELLED = "cancelled"
    TRIALING = "trialing"
    INCOMPLETE = "incomplete"


class BillingCustomer(BaseModel):
    """Represents a Stripe customer linked to a CodeVerify organization."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str
    stripe_customer_id: str | None = None
    email: str
    name: str
    tier: str = "free"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Subscription(BaseModel):
    """Represents a billing subscription."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str
    stripe_subscription_id: str | None = None
    tier: str = "free"
    status: SubscriptionStatus = SubscriptionStatus.ACTIVE
    current_period_start: datetime = Field(default_factory=datetime.utcnow)
    current_period_end: datetime = Field(
        default_factory=lambda: datetime.utcnow() + timedelta(days=30)
    )
    cancel_at_period_end: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class MeteredUsageRecord(BaseModel):
    """A metered usage record for pay-as-you-go billing."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str
    event_type: str
    quantity: int = 1
    unit_price_cents: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    reported_to_stripe: bool = False


# Pricing for pay-as-you-go (private repos on free tier)
PAYG_UNIT_PRICES: dict[str, int] = {
    "verification": 5,     # $0.05 per verification
    "analysis": 10,        # $0.10 per analysis
    "api_call": 1,         # $0.01 per API call
    "storage_mb": 2,       # $0.02 per MB/month
}


class BillingService:
    """Service encapsulating Stripe billing operations.

    In production, this would call the Stripe SDK.  The current implementation
    uses in-memory stores suitable for development and testing.
    """

    def __init__(self) -> None:
        self._customers: dict[str, BillingCustomer] = {}
        self._subscriptions: dict[str, Subscription] = {}
        self._metered_records: list[MeteredUsageRecord] = []

    # -- Customer management ------------------------------------------------

    def create_customer(
        self, organization_id: str, email: str, name: str
    ) -> BillingCustomer:
        """Create a billing customer (would call stripe.Customer.create)."""
        customer = BillingCustomer(
            organization_id=organization_id,
            stripe_customer_id=f"cus_{uuid.uuid4().hex[:14]}",
            email=email,
            name=name,
        )
        self._customers[organization_id] = customer
        return customer

    def get_customer(self, organization_id: str) -> BillingCustomer | None:
        return self._customers.get(organization_id)

    # -- Subscription management --------------------------------------------

    def create_subscription(
        self, organization_id: str, tier: str
    ) -> Subscription:
        """Create a subscription (would call stripe.Subscription.create)."""
        sub = Subscription(
            organization_id=organization_id,
            stripe_subscription_id=f"sub_{uuid.uuid4().hex[:14]}",
            tier=tier,
            status=SubscriptionStatus.ACTIVE,
        )
        self._subscriptions[organization_id] = sub
        return sub

    def get_subscription(self, organization_id: str) -> Subscription | None:
        return self._subscriptions.get(organization_id)

    def cancel_subscription(self, organization_id: str) -> Subscription | None:
        sub = self._subscriptions.get(organization_id)
        if sub:
            sub.cancel_at_period_end = True
        return sub

    def update_subscription_tier(
        self, organization_id: str, new_tier: str
    ) -> Subscription | None:
        """Upgrade/downgrade subscription tier."""
        sub = self._subscriptions.get(organization_id)
        if sub:
            sub.tier = new_tier
            sub.metadata["previous_tier_change"] = datetime.utcnow().isoformat()
        return sub

    # -- Metered billing ----------------------------------------------------

    def record_metered_usage(
        self,
        organization_id: str,
        event_type: str,
        quantity: int = 1,
    ) -> MeteredUsageRecord:
        """Record metered usage for pay-as-you-go billing."""
        unit_price = PAYG_UNIT_PRICES.get(event_type, 0)
        record = MeteredUsageRecord(
            organization_id=organization_id,
            event_type=event_type,
            quantity=quantity,
            unit_price_cents=unit_price * quantity,
        )
        self._metered_records.append(record)
        return record

    def get_outstanding_charges(self, organization_id: str) -> dict[str, Any]:
        """Calculate outstanding pay-as-you-go charges."""
        unreported = [
            r
            for r in self._metered_records
            if r.organization_id == organization_id and not r.reported_to_stripe
        ]
        total_cents = sum(r.unit_price_cents for r in unreported)

        by_type: dict[str, dict[str, int]] = {}
        for r in unreported:
            entry = by_type.setdefault(r.event_type, {"quantity": 0, "total_cents": 0})
            entry["quantity"] += r.quantity
            entry["total_cents"] += r.unit_price_cents

        return {
            "organization_id": organization_id,
            "total_cents": total_cents,
            "total_formatted": f"${total_cents / 100:.2f}",
            "breakdown": by_type,
            "record_count": len(unreported),
        }

    def report_usage_to_stripe(self, organization_id: str) -> int:
        """Mark all unreported records as reported (would call Stripe API)."""
        count = 0
        for r in self._metered_records:
            if r.organization_id == organization_id and not r.reported_to_stripe:
                r.reported_to_stripe = True
                count += 1
        return count


# Global service instance
billing_service = BillingService()
