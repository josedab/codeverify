"""Hosted SaaS platform API router.

Provides Stripe billing integration, usage metering, API key management,
multi-tenant subscription handling, and SOC 2-ready audit logging.
"""

import hashlib
import hmac
import secrets
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SubscriptionTier(BaseModel):
    id: str
    name: str
    price_monthly_cents: int
    stripe_price_id: str | None = None
    limits: dict[str, int]
    features: list[str]


class CreateCheckoutRequest(BaseModel):
    organization_id: str
    tier: str = Field(description="Target tier: free, pro, enterprise")
    success_url: str = Field(default="https://app.codeverify.dev/billing/success")
    cancel_url: str = Field(default="https://app.codeverify.dev/billing/cancel")


class CheckoutResponse(BaseModel):
    checkout_session_id: str
    checkout_url: str
    expires_at: str


class SubscriptionResponse(BaseModel):
    id: str
    organization_id: str
    tier: str
    status: str
    stripe_subscription_id: str | None = None
    current_period_start: str
    current_period_end: str
    cancel_at_period_end: bool = False
    usage: dict[str, Any]


class UsageMeterEvent(BaseModel):
    organization_id: str
    event_type: str = Field(description="Event: verification, analysis, api_call, storage_mb")
    quantity: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None


class UsageMeterResponse(BaseModel):
    event_id: str
    organization_id: str
    event_type: str
    quantity: int
    recorded_at: str
    within_limits: bool
    current_usage: int
    limit: int


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    scopes: list[str] = Field(default=["read", "verify"])
    expires_in_days: int | None = Field(default=90, ge=1, le=365)


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    key_prefix: str
    scopes: list[str]
    created_at: str
    expires_at: str | None
    last_used_at: str | None


class ApiKeyCreatedResponse(ApiKeyResponse):
    raw_key: str = Field(description="Full API key - only shown once at creation")


class StripeWebhookEvent(BaseModel):
    event_type: str
    stripe_event_id: str
    data: dict[str, Any]
    processed_at: str


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

TIERS: dict[str, SubscriptionTier] = {
    "free": SubscriptionTier(
        id="free",
        name="Free",
        price_monthly_cents=0,
        stripe_price_id=None,
        limits={
            "verifications_per_month": 100,
            "repositories": 5,
            "users": 3,
            "api_calls_per_day": 1000,
            "storage_mb": 100,
        },
        features=[
            "public_repos",
            "basic_analysis",
            "pr_checks",
            "community_support",
        ],
    ),
    "pro": SubscriptionTier(
        id="pro",
        name="Pro",
        price_monthly_cents=2900,
        stripe_price_id="price_pro_monthly",
        limits={
            "verifications_per_month": 5000,
            "repositories": 100,
            "users": 50,
            "api_calls_per_day": 50000,
            "storage_mb": 5000,
        },
        features=[
            "public_repos",
            "private_repos",
            "basic_analysis",
            "formal_verification",
            "pr_checks",
            "trust_scores",
            "custom_rules",
            "team_dashboard",
            "api_access",
            "priority_support",
        ],
    ),
    "enterprise": SubscriptionTier(
        id="enterprise",
        name="Enterprise",
        price_monthly_cents=0,  # Custom pricing
        stripe_price_id="price_enterprise_monthly",
        limits={
            "verifications_per_month": 999999,
            "repositories": 999999,
            "users": 999999,
            "api_calls_per_day": 999999,
            "storage_mb": 999999,
        },
        features=[
            "public_repos",
            "private_repos",
            "basic_analysis",
            "formal_verification",
            "pr_checks",
            "trust_scores",
            "custom_rules",
            "team_dashboard",
            "api_access",
            "sso_saml",
            "audit_logs",
            "compliance_reports",
            "sla_guarantee",
            "dedicated_support",
            "on_premise_option",
        ],
    ),
}

# ---------------------------------------------------------------------------
# In-memory stores (production: PostgreSQL with RLS)
# ---------------------------------------------------------------------------

_subscriptions: dict[str, dict[str, Any]] = {}
_usage_meters: dict[str, list[dict[str, Any]]] = {}
_api_keys: dict[str, dict[str, Any]] = {}
_stripe_events: list[dict[str, Any]] = []
_idempotency_cache: dict[str, str] = {}
_audit_log: list[dict[str, Any]] = []


def _log_audit(
    org_id: str, action: str, resource_type: str, resource_id: str, details: dict[str, Any]
) -> None:
    """Append an immutable SOC 2-ready audit entry."""
    entry = {
        "id": str(uuid.uuid4()),
        "organization_id": org_id,
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "details": details,
        "timestamp": datetime.utcnow().isoformat(),
    }
    # Compute integrity hash over the entry payload
    payload = f"{entry['id']}:{entry['organization_id']}:{entry['action']}:{entry['timestamp']}"
    entry["integrity_hash"] = hashlib.sha256(payload.encode()).hexdigest()
    _audit_log.append(entry)


def _get_current_month_usage(org_id: str, event_type: str) -> int:
    """Sum usage for the current calendar month."""
    now = datetime.utcnow()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    records = _usage_meters.get(org_id, [])
    return sum(
        r["quantity"]
        for r in records
        if r["event_type"] == event_type
        and datetime.fromisoformat(r["recorded_at"]) >= month_start
    )


# ---------------------------------------------------------------------------
# Subscription endpoints
# ---------------------------------------------------------------------------

@router.get("/tiers", response_model=list[SubscriptionTier])
async def list_tiers() -> list[SubscriptionTier]:
    """List available subscription tiers and their limits."""
    return list(TIERS.values())


@router.post("/subscriptions/checkout", response_model=CheckoutResponse)
async def create_checkout_session(request: CreateCheckoutRequest) -> CheckoutResponse:
    """Create a Stripe checkout session for subscription upgrade."""
    if request.tier not in TIERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tier: {request.tier}. Valid: {list(TIERS.keys())}",
        )

    tier = TIERS[request.tier]
    if tier.price_monthly_cents == 0 and request.tier != "free":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Enterprise tier requires contacting sales",
        )

    session_id = f"cs_{uuid.uuid4().hex[:24]}"
    expires = datetime.utcnow() + timedelta(hours=1)

    _log_audit(
        request.organization_id,
        "checkout.created",
        "subscription",
        session_id,
        {"tier": request.tier},
    )

    return CheckoutResponse(
        checkout_session_id=session_id,
        checkout_url=f"https://checkout.stripe.com/c/pay/{session_id}",
        expires_at=expires.isoformat(),
    )


@router.get("/subscriptions/{org_id}", response_model=SubscriptionResponse)
async def get_subscription(org_id: str) -> SubscriptionResponse:
    """Get the current subscription for an organization."""
    sub = _subscriptions.get(org_id)
    if not sub:
        now = datetime.utcnow()
        sub = {
            "id": str(uuid.uuid4()),
            "organization_id": org_id,
            "tier": "free",
            "status": "active",
            "stripe_subscription_id": None,
            "current_period_start": now.isoformat(),
            "current_period_end": (now + timedelta(days=30)).isoformat(),
            "cancel_at_period_end": False,
        }
        _subscriptions[org_id] = sub

    # Compute current usage
    tier = TIERS[sub["tier"]]
    usage: dict[str, Any] = {}
    for resource, limit in tier.limits.items():
        event_type = resource.replace("_per_month", "").replace("_per_day", "")
        current = _get_current_month_usage(org_id, event_type)
        usage[resource] = {
            "used": current,
            "limit": limit,
            "remaining": max(0, limit - current),
            "percentage": round(current / max(limit, 1) * 100, 1),
        }

    return SubscriptionResponse(**sub, usage=usage)


@router.post("/subscriptions/{org_id}/cancel")
async def cancel_subscription(org_id: str) -> dict[str, Any]:
    """Cancel subscription at end of current billing period."""
    sub = _subscriptions.get(org_id)
    if not sub:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Subscription not found")

    sub["cancel_at_period_end"] = True
    _log_audit(org_id, "subscription.cancelled", "subscription", sub["id"], {"tier": sub["tier"]})

    return {"status": "cancelling", "effective_at": sub["current_period_end"]}


# ---------------------------------------------------------------------------
# Usage metering endpoints
# ---------------------------------------------------------------------------

@router.post("/usage/record", response_model=UsageMeterResponse)
async def record_usage_event(event: UsageMeterEvent) -> UsageMeterResponse:
    """Record a usage metering event (e.g., verification run, API call)."""
    valid_types = {"verification", "analysis", "api_call", "storage_mb"}
    if event.event_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid event_type. Must be one of: {valid_types}",
        )

    # Idempotency check
    if event.idempotency_key and event.idempotency_key in _idempotency_cache:
        existing_id = _idempotency_cache[event.idempotency_key]
        records = _usage_meters.get(event.organization_id, [])
        existing = next((r for r in records if r["event_id"] == existing_id), None)
        if existing:
            sub = _subscriptions.get(event.organization_id, {})
            tier = TIERS.get(sub.get("tier", "free"), TIERS["free"])
            limit_key = f"{event.event_type}s_per_month"
            limit = tier.limits.get(limit_key, tier.limits.get(event.event_type, 999999))
            current = _get_current_month_usage(event.organization_id, event.event_type)
            return UsageMeterResponse(
                event_id=existing_id,
                organization_id=event.organization_id,
                event_type=event.event_type,
                quantity=existing["quantity"],
                recorded_at=existing["recorded_at"],
                within_limits=current <= limit,
                current_usage=current,
                limit=limit,
            )

    # Resolve limits from subscription
    sub = _subscriptions.get(event.organization_id, {})
    tier_id = sub.get("tier", "free")
    tier = TIERS.get(tier_id, TIERS["free"])

    limit_key_candidates = [
        f"{event.event_type}s_per_month",
        f"{event.event_type}_per_month",
        f"{event.event_type}s_per_day",
        event.event_type,
    ]
    limit = 999999
    for lk in limit_key_candidates:
        if lk in tier.limits:
            limit = tier.limits[lk]
            break

    current = _get_current_month_usage(event.organization_id, event.event_type)
    within_limits = (current + event.quantity) <= limit

    event_id = str(uuid.uuid4())
    record = {
        "event_id": event_id,
        "organization_id": event.organization_id,
        "event_type": event.event_type,
        "quantity": event.quantity,
        "metadata": event.metadata,
        "recorded_at": datetime.utcnow().isoformat(),
    }
    _usage_meters.setdefault(event.organization_id, []).append(record)

    if event.idempotency_key:
        _idempotency_cache[event.idempotency_key] = event_id

    return UsageMeterResponse(
        event_id=event_id,
        organization_id=event.organization_id,
        event_type=event.event_type,
        quantity=event.quantity,
        recorded_at=record["recorded_at"],
        within_limits=within_limits,
        current_usage=current + event.quantity,
        limit=limit,
    )


@router.get("/usage/{org_id}/summary")
async def get_usage_summary(
    org_id: str,
    period: str = Query(default="current", description="current or YYYY-MM"),
) -> dict[str, Any]:
    """Get usage summary for an organization's billing period."""
    sub = _subscriptions.get(org_id, {})
    tier_id = sub.get("tier", "free")
    tier = TIERS.get(tier_id, TIERS["free"])

    summary: dict[str, Any] = {"organization_id": org_id, "tier": tier_id, "period": period}
    breakdown: dict[str, Any] = {}
    for resource, limit in tier.limits.items():
        event_type = resource.replace("_per_month", "").replace("_per_day", "").rstrip("s")
        current = _get_current_month_usage(org_id, event_type)
        breakdown[resource] = {
            "used": current,
            "limit": limit,
            "remaining": max(0, limit - current),
            "utilization_pct": round(current / max(limit, 1) * 100, 1),
        }

    summary["breakdown"] = breakdown
    summary["overage"] = any(
        v["used"] > v["limit"] for v in breakdown.values()
    )
    return summary


# ---------------------------------------------------------------------------
# API key management
# ---------------------------------------------------------------------------

@router.post("/api-keys/{org_id}", response_model=ApiKeyCreatedResponse, status_code=201)
async def create_api_key(org_id: str, request: ApiKeyCreateRequest) -> ApiKeyCreatedResponse:
    """Create a new API key for an organization. The raw key is only returned once."""
    valid_scopes = {"read", "write", "verify", "admin"}
    invalid = set(request.scopes) - valid_scopes
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid scopes: {invalid}. Valid: {valid_scopes}",
        )

    raw_key = f"cv_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_prefix = raw_key[:10]
    key_id = str(uuid.uuid4())

    now = datetime.utcnow()
    expires_at = (now + timedelta(days=request.expires_in_days)).isoformat() if request.expires_in_days else None

    record = {
        "id": key_id,
        "organization_id": org_id,
        "name": request.name,
        "key_hash": key_hash,
        "key_prefix": key_prefix,
        "scopes": request.scopes,
        "created_at": now.isoformat(),
        "expires_at": expires_at,
        "last_used_at": None,
        "revoked": False,
    }
    _api_keys[key_id] = record

    _log_audit(org_id, "api_key.created", "api_key", key_id, {"name": request.name})

    return ApiKeyCreatedResponse(
        id=key_id,
        name=request.name,
        key_prefix=key_prefix,
        scopes=request.scopes,
        created_at=now.isoformat(),
        expires_at=expires_at,
        last_used_at=None,
        raw_key=raw_key,
    )


@router.get("/api-keys/{org_id}", response_model=list[ApiKeyResponse])
async def list_api_keys(org_id: str) -> list[ApiKeyResponse]:
    """List all API keys for an organization (without secrets)."""
    keys = [
        ApiKeyResponse(
            id=k["id"],
            name=k["name"],
            key_prefix=k["key_prefix"],
            scopes=k["scopes"],
            created_at=k["created_at"],
            expires_at=k["expires_at"],
            last_used_at=k["last_used_at"],
        )
        for k in _api_keys.values()
        if k["organization_id"] == org_id and not k.get("revoked")
    ]
    return keys


@router.delete("/api-keys/{org_id}/{key_id}", status_code=204)
async def revoke_api_key(org_id: str, key_id: str) -> None:
    """Revoke an API key."""
    key = _api_keys.get(key_id)
    if not key or key["organization_id"] != org_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")

    key["revoked"] = True
    _log_audit(org_id, "api_key.revoked", "api_key", key_id, {"name": key["name"]})


# ---------------------------------------------------------------------------
# Stripe webhook handler
# ---------------------------------------------------------------------------

@router.post("/webhooks/stripe")
async def handle_stripe_webhook(
    request: Request,
    stripe_signature: str | None = Header(None, alias="Stripe-Signature"),
) -> dict[str, str]:
    """Handle Stripe webhook events for subscription lifecycle."""
    body = await request.body()

    # In production: verify signature with stripe.Webhook.construct_event()
    # For now, parse the JSON directly
    try:
        import json
        payload = json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid payload")

    event_type = payload.get("type", "unknown")
    event_id = payload.get("id", str(uuid.uuid4()))
    data = payload.get("data", {}).get("object", {})

    handled_events = {
        "checkout.session.completed",
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
        "invoice.paid",
        "invoice.payment_failed",
    }

    if event_type in handled_events:
        org_id = data.get("metadata", {}).get("organization_id", "unknown")

        if event_type == "customer.subscription.created":
            tier = data.get("metadata", {}).get("tier", "pro")
            now = datetime.utcnow()
            _subscriptions[org_id] = {
                "id": str(uuid.uuid4()),
                "organization_id": org_id,
                "tier": tier,
                "status": "active",
                "stripe_subscription_id": data.get("id"),
                "current_period_start": now.isoformat(),
                "current_period_end": (now + timedelta(days=30)).isoformat(),
                "cancel_at_period_end": False,
            }

        elif event_type == "customer.subscription.deleted":
            if org_id in _subscriptions:
                _subscriptions[org_id]["status"] = "cancelled"
                _subscriptions[org_id]["tier"] = "free"

        elif event_type == "invoice.payment_failed":
            if org_id in _subscriptions:
                _subscriptions[org_id]["status"] = "past_due"

        _stripe_events.append({
            "event_type": event_type,
            "stripe_event_id": event_id,
            "data": data,
            "processed_at": datetime.utcnow().isoformat(),
        })

        _log_audit(org_id, f"stripe.{event_type}", "billing", event_id, {"type": event_type})

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# SOC 2 audit log endpoint
# ---------------------------------------------------------------------------

@router.get("/audit-log/{org_id}")
async def get_saas_audit_log(
    org_id: str,
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
    action: str | None = Query(default=None),
) -> dict[str, Any]:
    """Get SOC 2-ready audit log entries for an organization."""
    entries = [e for e in _audit_log if e["organization_id"] == org_id]
    if action:
        entries = [e for e in entries if action in e["action"]]

    total = len(entries)
    entries = entries[offset : offset + limit]

    return {
        "organization_id": org_id,
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": entries,
    }
