"""Multi-tenancy management API router."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

router = APIRouter()


class CreateTenantRequest(BaseModel):
    name: str = Field(description="Tenant organization name")
    slug: str = Field(description="URL-safe tenant identifier")
    tier: str = Field(default="free", description="Subscription tier: free, pro, enterprise")


class TenantResponse(BaseModel):
    id: str
    name: str
    slug: str
    tier: str
    max_repos: int
    max_analyses_per_month: int
    max_users: int
    features: list[str]
    sso_enabled: bool


class UpdateTierRequest(BaseModel):
    tier: str = Field(description="New subscription tier")


class UsageResponse(BaseModel):
    tenant_id: str
    analyses_used: int
    analyses_limit: int
    repos_count: int
    repos_limit: int
    users_count: int
    users_limit: int
    usage_percentage: float


# In-memory store for demo (production uses database)
_tenants: dict[str, dict[str, Any]] = {}
_usage: dict[str, dict[str, int]] = {}

TIER_LIMITS = {
    "free": {"max_repos": 3, "max_analyses": 100, "max_users": 5, "sso": False},
    "pro": {"max_repos": 25, "max_analyses": 2000, "max_users": 50, "sso": False},
    "enterprise": {"max_repos": 999999, "max_analyses": 50000, "max_users": 999999, "sso": True},
}


@router.post("", response_model=TenantResponse, status_code=status.HTTP_201_CREATED)
async def create_tenant(request: CreateTenantRequest) -> TenantResponse:
    """Create a new tenant organization."""
    import uuid

    if request.tier not in TIER_LIMITS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tier: {request.tier}. Must be one of: {list(TIER_LIMITS.keys())}",
        )

    for t in _tenants.values():
        if t["slug"] == request.slug:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Tenant with slug '{request.slug}' already exists",
            )

    tenant_id = str(uuid.uuid4())
    limits = TIER_LIMITS[request.tier]
    features = ["basic_analysis", "pr_checks"]
    if request.tier in ("pro", "enterprise"):
        features.extend(["trust_score", "custom_rules", "team_dashboard", "api_access"])
    if request.tier == "enterprise":
        features.extend(["sso", "audit_logs", "formal_verification", "compliance", "priority_support"])

    tenant = {
        "id": tenant_id,
        "name": request.name,
        "slug": request.slug,
        "tier": request.tier,
        "max_repos": limits["max_repos"],
        "max_analyses_per_month": limits["max_analyses"],
        "max_users": limits["max_users"],
        "features": features,
        "sso_enabled": limits["sso"],
    }
    _tenants[tenant_id] = tenant
    _usage[tenant_id] = {"analyses": 0, "repos": 0, "users": 1}

    return TenantResponse(**tenant)


@router.get("", response_model=list[TenantResponse])
async def list_tenants(
    tier: str | None = Query(default=None, description="Filter by tier"),
) -> list[TenantResponse]:
    """List all tenants."""
    tenants = list(_tenants.values())
    if tier:
        tenants = [t for t in tenants if t["tier"] == tier]
    return [TenantResponse(**t) for t in tenants]


@router.get("/{tenant_id}", response_model=TenantResponse)
async def get_tenant(tenant_id: str) -> TenantResponse:
    """Get tenant details."""
    tenant = _tenants.get(tenant_id)
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")
    return TenantResponse(**tenant)


@router.put("/{tenant_id}/tier", response_model=TenantResponse)
async def update_tenant_tier(tenant_id: str, request: UpdateTierRequest) -> TenantResponse:
    """Update tenant subscription tier."""
    tenant = _tenants.get(tenant_id)
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")
    if request.tier not in TIER_LIMITS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tier: {request.tier}",
        )

    limits = TIER_LIMITS[request.tier]
    tenant["tier"] = request.tier
    tenant["max_repos"] = limits["max_repos"]
    tenant["max_analyses_per_month"] = limits["max_analyses"]
    tenant["max_users"] = limits["max_users"]
    tenant["sso_enabled"] = limits["sso"]

    features = ["basic_analysis", "pr_checks"]
    if request.tier in ("pro", "enterprise"):
        features.extend(["trust_score", "custom_rules", "team_dashboard", "api_access"])
    if request.tier == "enterprise":
        features.extend(["sso", "audit_logs", "formal_verification", "compliance", "priority_support"])
    tenant["features"] = features

    return TenantResponse(**tenant)


@router.get("/{tenant_id}/usage", response_model=UsageResponse)
async def get_tenant_usage(tenant_id: str) -> UsageResponse:
    """Get tenant resource usage."""
    tenant = _tenants.get(tenant_id)
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")

    usage = _usage.get(tenant_id, {"analyses": 0, "repos": 0, "users": 0})
    analyses_limit = tenant["max_analyses_per_month"]
    usage_pct = (usage["analyses"] / analyses_limit * 100) if analyses_limit > 0 else 0

    return UsageResponse(
        tenant_id=tenant_id,
        analyses_used=usage["analyses"],
        analyses_limit=analyses_limit,
        repos_count=usage["repos"],
        repos_limit=tenant["max_repos"],
        users_count=usage["users"],
        users_limit=tenant["max_users"],
        usage_percentage=round(usage_pct, 1),
    )


@router.post("/{tenant_id}/usage/record")
async def record_usage(
    tenant_id: str,
    resource: str = Query(description="Resource type: analyses, repos, users"),
    count: int = Query(default=1, ge=1, description="Usage count"),
) -> dict[str, Any]:
    """Record resource usage for a tenant."""
    tenant = _tenants.get(tenant_id)
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")

    if resource not in ("analyses", "repos", "users"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Resource must be: analyses, repos, or users",
        )

    usage = _usage.setdefault(tenant_id, {"analyses": 0, "repos": 0, "users": 0})

    limit_map = {
        "analyses": tenant["max_analyses_per_month"],
        "repos": tenant["max_repos"],
        "users": tenant["max_users"],
    }

    if usage[resource] + count > limit_map[resource]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Usage limit exceeded for {resource}. Current: {usage[resource]}, Limit: {limit_map[resource]}",
        )

    usage[resource] += count
    return {"recorded": True, "resource": resource, "current": usage[resource], "limit": limit_map[resource]}


@router.delete("/{tenant_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tenant(tenant_id: str) -> None:
    """Delete a tenant."""
    if tenant_id not in _tenants:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")
    del _tenants[tenant_id]
    _usage.pop(tenant_id, None)
