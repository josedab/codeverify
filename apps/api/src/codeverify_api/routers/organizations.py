"""Organizations API endpoints."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from codeverify_api.db.database import get_db
from codeverify_api.db.models import Organization, Repository, OrgMembership

router = APIRouter()


class OrganizationSettings(BaseModel):
    """Organization settings model."""
    default_enabled: bool | None = None
    default_verification_checks: list[str] | None = None
    default_ai_analysis: bool | None = None
    default_thresholds: dict[str, int] | None = None
    require_reviews: bool | None = None
    block_on_critical: bool | None = None
    notification_channels: list[str] | None = None


@router.get("")
async def list_organizations(
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """List all organizations the user has access to."""
    query = select(Organization).order_by(Organization.name).offset(offset).limit(limit)
    count_query = select(func.count(Organization.id))

    result = await db.execute(query)
    orgs = result.scalars().all()

    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    return {
        "organizations": [
            {
                "id": str(o.id),
                "github_id": o.github_id,
                "name": o.name,
                "login": o.login,
                "avatar_url": o.avatar_url,
                "settings": o.settings,
                "created_at": o.created_at.isoformat(),
                "updated_at": o.updated_at.isoformat(),
            }
            for o in orgs
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{org_id}")
async def get_organization(
    org_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get organization details."""
    org = await db.get(Organization, org_id)
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    # Get repository count
    repo_count_query = select(func.count(Repository.id)).where(Repository.org_id == org_id)
    repo_count_result = await db.execute(repo_count_query)
    repo_count = repo_count_result.scalar() or 0

    # Get member count
    member_count_query = select(func.count(OrgMembership.id)).where(OrgMembership.org_id == org_id)
    member_count_result = await db.execute(member_count_query)
    member_count = member_count_result.scalar() or 0

    return {
        "id": str(org.id),
        "github_id": org.github_id,
        "name": org.name,
        "login": org.login,
        "avatar_url": org.avatar_url,
        "settings": org.settings,
        "created_at": org.created_at.isoformat(),
        "updated_at": org.updated_at.isoformat(),
        "repository_count": repo_count,
        "member_count": member_count,
    }


@router.get("/{org_id}/settings")
async def get_organization_settings(
    org_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get organization settings."""
    org = await db.get(Organization, org_id)
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    # Return settings with defaults
    default_settings = {
        "default_enabled": True,
        "default_verification_checks": ["null_safety", "array_bounds", "integer_overflow", "division_by_zero"],
        "default_ai_analysis": True,
        "default_thresholds": {"critical": 0, "high": 0, "medium": 5, "low": 10},
        "require_reviews": False,
        "block_on_critical": True,
        "notification_channels": [],
    }

    return {
        "org_id": str(org_id),
        "settings": {**default_settings, **org.settings},
    }


@router.patch("/{org_id}/settings")
async def update_organization_settings(
    org_id: UUID,
    settings: OrganizationSettings,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Update organization settings."""
    org = await db.get(Organization, org_id)
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    # Update settings (merge with existing)
    updated_settings = dict(org.settings)
    settings_dict = settings.model_dump(exclude_none=True)
    updated_settings.update(settings_dict)
    org.settings = updated_settings

    await db.flush()
    await db.refresh(org)

    return {
        "org_id": str(org_id),
        "settings": org.settings,
    }


@router.get("/{org_id}/repositories")
async def get_organization_repositories(
    org_id: UUID,
    enabled: bool | None = None,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get repositories for an organization."""
    org = await db.get(Organization, org_id)
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    query = select(Repository).where(Repository.org_id == org_id)
    count_query = select(func.count(Repository.id)).where(Repository.org_id == org_id)

    if enabled is not None:
        query = query.where(Repository.enabled == enabled)
        count_query = count_query.where(Repository.enabled == enabled)

    query = query.order_by(Repository.name).offset(offset).limit(limit)

    result = await db.execute(query)
    repos = result.scalars().all()

    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    return {
        "repositories": [
            {
                "id": str(r.id),
                "github_id": r.github_id,
                "name": r.name,
                "full_name": r.full_name,
                "private": r.private,
                "enabled": r.enabled,
                "created_at": r.created_at.isoformat(),
            }
            for r in repos
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{org_id}/members")
async def get_organization_members(
    org_id: UUID,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get members of an organization."""
    org = await db.get(Organization, org_id)
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found")

    query = (
        select(OrgMembership)
        .where(OrgMembership.org_id == org_id)
        .options(selectinload(OrgMembership.user))
        .offset(offset)
        .limit(limit)
    )
    count_query = select(func.count(OrgMembership.id)).where(OrgMembership.org_id == org_id)

    result = await db.execute(query)
    memberships = result.scalars().all()

    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    return {
        "members": [
            {
                "id": str(m.user.id),
                "username": m.user.username,
                "email": m.user.email,
                "avatar_url": m.user.avatar_url,
                "role": m.role,
                "joined_at": m.created_at.isoformat(),
            }
            for m in memberships
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }
