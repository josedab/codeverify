"""GraphQL Insights API endpoints.

Provides a GraphQL endpoint for querying verification results,
trust scores, findings, proofs, and trends. Includes API key
management and webhook subscription endpoints.
"""

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field

router = APIRouter()


# ─── Pydantic Models ──────────────────────────────────────────────────


class GraphQLRequest(BaseModel):
    """GraphQL query request."""

    query: str = Field(..., description="GraphQL query string")
    variables: dict[str, Any] | None = Field(default=None, description="Query variables")
    operation_name: str | None = Field(default=None, description="Operation name")


class ApiKeyCreateRequest(BaseModel):
    """Request to create an API key."""

    name: str = Field(..., description="Human-readable key name")
    scopes: list[str] = Field(default=["read"], description="Permissions: read, write, admin, webhooks")
    tier: str = Field(default="free", description="Rate limit tier: free, standard, premium, unlimited")


class ApiKeyResponse(BaseModel):
    """API key creation response (raw key shown once)."""

    id: str
    name: str
    key_prefix: str
    raw_key: str
    scopes: list[str]
    tier: str


class WebhookCreateRequest(BaseModel):
    """Request to create a webhook subscription."""

    url: str = Field(..., description="Webhook delivery URL")
    events: list[str] = Field(
        ...,
        description="Events to subscribe to: analysis.completed, analysis.failed, finding.created, trust_score.changed, scan.completed",
    )


class WebhookResponse(BaseModel):
    """Webhook subscription response."""

    id: str
    url: str
    events: list[str]
    secret: str
    is_active: bool


# ─── Endpoints ────────────────────────────────────────────────────────


@router.post("/graphql")
async def execute_graphql(
    request: GraphQLRequest,
    authorization: str = Header(..., description="Bearer <api_key>"),
) -> dict[str, Any]:
    """Execute a GraphQL query against verification data."""
    from codeverify_core.graphql_insights import GraphQLInsightsService

    raw_key = authorization.replace("Bearer ", "").strip()
    svc = GraphQLInsightsService()

    response = svc.execute_query(raw_key, request.query, request.variables)

    if response.has_errors:
        error_code = response.errors[0].get("extensions", {}).get("code", "UNKNOWN")
        status_code = {
            "UNAUTHENTICATED": 401,
            "RATE_LIMITED": 429,
            "QUERY_TOO_COMPLEX": 400,
            "FORBIDDEN": 403,
        }.get(error_code, 400)
        raise HTTPException(status_code=status_code, detail=response.errors)

    return {
        "data": response.data,
        "extensions": response.extensions,
    }


@router.get("/schema")
async def get_schema() -> dict[str, str]:
    """Get the GraphQL schema definition."""
    from codeverify_core.graphql_insights import GraphQLInsightsService

    return {"schema": GraphQLInsightsService.SCHEMA}


@router.post("/keys", status_code=status.HTTP_201_CREATED)
async def create_api_key(request: ApiKeyCreateRequest) -> ApiKeyResponse:
    """Create a new scoped API key."""
    from codeverify_core.graphql_insights import (
        ApiKeyScope,
        GraphQLInsightsService,
        RateLimitTier,
    )

    scope_map = {
        "read": ApiKeyScope.READ,
        "write": ApiKeyScope.WRITE,
        "admin": ApiKeyScope.ADMIN,
        "webhooks": ApiKeyScope.WEBHOOKS,
    }
    tier_map = {
        "free": RateLimitTier.FREE,
        "standard": RateLimitTier.STANDARD,
        "premium": RateLimitTier.PREMIUM,
        "unlimited": RateLimitTier.UNLIMITED,
    }

    scopes = [scope_map.get(s, ApiKeyScope.READ) for s in request.scopes]
    tier = tier_map.get(request.tier, RateLimitTier.FREE)

    svc = GraphQLInsightsService()
    key, raw = svc.create_api_key(request.name, "current_user", scopes, tier)

    return ApiKeyResponse(
        id=key.id,
        name=key.name,
        key_prefix=key.key_prefix,
        raw_key=raw,
        scopes=request.scopes,
        tier=request.tier,
    )


@router.delete("/keys/{key_id}")
async def revoke_api_key(key_id: str) -> dict[str, str]:
    """Revoke an API key."""
    from codeverify_core.graphql_insights import GraphQLInsightsService

    svc = GraphQLInsightsService()
    if not svc.revoke_api_key(key_id):
        raise HTTPException(status_code=404, detail="API key not found")
    return {"status": "revoked"}


@router.post("/webhooks", status_code=status.HTTP_201_CREATED)
async def create_webhook(request: WebhookCreateRequest) -> WebhookResponse:
    """Create a webhook subscription."""
    from codeverify_core.graphql_insights import GraphQLInsightsService, WebhookEvent

    event_map = {
        "analysis.completed": WebhookEvent.ANALYSIS_COMPLETED,
        "analysis.failed": WebhookEvent.ANALYSIS_FAILED,
        "finding.created": WebhookEvent.FINDING_CREATED,
        "trust_score.changed": WebhookEvent.TRUST_SCORE_CHANGED,
        "scan.completed": WebhookEvent.SCAN_COMPLETED,
    }

    events = [event_map[e] for e in request.events if e in event_map]
    if not events:
        raise HTTPException(status_code=400, detail="No valid events specified")

    svc = GraphQLInsightsService()
    webhook = svc.create_webhook("current_user", request.url, events)

    return WebhookResponse(
        id=webhook.id,
        url=webhook.url,
        events=request.events,
        secret=webhook.secret,
        is_active=webhook.is_active,
    )


@router.delete("/webhooks/{webhook_id}")
async def delete_webhook(webhook_id: str) -> dict[str, str]:
    """Delete a webhook subscription."""
    from codeverify_core.graphql_insights import GraphQLInsightsService

    svc = GraphQLInsightsService()
    if not svc.delete_webhook(webhook_id):
        raise HTTPException(status_code=404, detail="Webhook not found")
    return {"status": "deleted"}


@router.get("/stats")
async def get_insights_stats() -> dict[str, Any]:
    """Get insights API statistics."""
    from codeverify_core.graphql_insights import GraphQLInsightsService

    svc = GraphQLInsightsService()
    return svc.get_stats()
