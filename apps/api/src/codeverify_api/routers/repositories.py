"""Repositories API endpoints."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from codeverify_api.db.database import get_db
from codeverify_api.db.models import Repository, Analysis
from codeverify_api.db.repositories import RepositoryRepository

router = APIRouter()


class RepositorySettings(BaseModel):
    """Repository settings model."""
    enabled: bool | None = None
    auto_analyze: bool | None = None
    verification_checks: list[str] | None = None
    ai_analysis: bool | None = None
    thresholds: dict[str, int] | None = None
    exclude_patterns: list[str] | None = None


@router.get("")
async def list_repositories(
    org_id: UUID | None = None,
    enabled: bool | None = None,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """List repositories, optionally filtered by organization."""
    query = select(Repository)
    count_query = select(func.count(Repository.id))

    if org_id:
        query = query.where(Repository.org_id == org_id)
        count_query = count_query.where(Repository.org_id == org_id)
    if enabled is not None:
        query = query.where(Repository.enabled == enabled)
        count_query = count_query.where(Repository.enabled == enabled)

    query = query.order_by(Repository.created_at.desc()).offset(offset).limit(limit)

    result = await db.execute(query)
    repos = result.scalars().all()

    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    return {
        "repositories": [
            {
                "id": str(r.id),
                "github_id": r.github_id,
                "org_id": str(r.org_id) if r.org_id else None,
                "name": r.name,
                "full_name": r.full_name,
                "private": r.private,
                "default_branch": r.default_branch,
                "enabled": r.enabled,
                "settings": r.settings,
                "created_at": r.created_at.isoformat(),
                "updated_at": r.updated_at.isoformat(),
            }
            for r in repos
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{repo_id}")
async def get_repository(
    repo_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get repository details."""
    repo = await db.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repository not found")

    # Get analysis counts
    count_query = select(func.count(Analysis.id)).where(Analysis.repo_id == repo_id)
    count_result = await db.execute(count_query)
    analysis_count = count_result.scalar() or 0

    return {
        "id": str(repo.id),
        "github_id": repo.github_id,
        "org_id": str(repo.org_id) if repo.org_id else None,
        "name": repo.name,
        "full_name": repo.full_name,
        "private": repo.private,
        "default_branch": repo.default_branch,
        "enabled": repo.enabled,
        "settings": repo.settings,
        "created_at": repo.created_at.isoformat(),
        "updated_at": repo.updated_at.isoformat(),
        "analysis_count": analysis_count,
    }


@router.get("/{repo_id}/settings")
async def get_repository_settings(
    repo_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get repository-specific settings."""
    repo = await db.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repository not found")

    # Return settings with defaults
    default_settings = {
        "enabled": repo.enabled,
        "auto_analyze": True,
        "verification_checks": ["null_safety", "array_bounds", "integer_overflow", "division_by_zero"],
        "ai_analysis": True,
        "thresholds": {"critical": 0, "high": 0, "medium": 5, "low": 10},
        "exclude_patterns": ["node_modules/**", "venv/**", "*.min.js"],
    }

    return {
        "repo_id": str(repo_id),
        "settings": {**default_settings, **repo.settings},
    }


@router.patch("/{repo_id}/settings")
async def update_repository_settings(
    repo_id: UUID,
    settings: RepositorySettings,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Update repository settings."""
    repo = await db.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repository not found")

    # Update settings (merge with existing)
    updated_settings = dict(repo.settings)
    settings_dict = settings.model_dump(exclude_none=True)

    # Handle enabled flag separately
    if "enabled" in settings_dict:
        repo.enabled = settings_dict.pop("enabled")

    updated_settings.update(settings_dict)
    repo.settings = updated_settings

    await db.flush()
    await db.refresh(repo)

    return {
        "repo_id": str(repo_id),
        "settings": {
            "enabled": repo.enabled,
            **repo.settings,
        },
    }


@router.post("/{repo_id}/enable")
async def enable_repository(
    repo_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Enable CodeVerify for a repository."""
    repo = await db.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repository not found")

    repo.enabled = True
    await db.flush()
    await db.refresh(repo)

    return {"repo_id": str(repo_id), "enabled": True}


@router.post("/{repo_id}/disable")
async def disable_repository(
    repo_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Disable CodeVerify for a repository."""
    repo = await db.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repository not found")

    repo.enabled = False
    await db.flush()
    await db.refresh(repo)

    return {"repo_id": str(repo_id), "enabled": False}


@router.get("/{repo_id}/analyses")
async def get_repository_analyses(
    repo_id: UUID,
    status_filter: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get analyses for a specific repository."""
    repo = await db.get(Repository, repo_id)
    if not repo:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Repository not found")

    query = select(Analysis).where(Analysis.repo_id == repo_id)
    count_query = select(func.count(Analysis.id)).where(Analysis.repo_id == repo_id)

    if status_filter:
        query = query.where(Analysis.status == status_filter)
        count_query = count_query.where(Analysis.status == status_filter)

    query = query.order_by(Analysis.created_at.desc()).offset(offset).limit(limit)

    result = await db.execute(query)
    analyses = result.scalars().all()

    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    return {
        "analyses": [
            {
                "id": str(a.id),
                "pr_number": a.pr_number,
                "pr_title": a.pr_title,
                "head_sha": a.head_sha,
                "status": a.status,
                "started_at": a.started_at.isoformat() if a.started_at else None,
                "completed_at": a.completed_at.isoformat() if a.completed_at else None,
                "created_at": a.created_at.isoformat(),
            }
            for a in analyses
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }
