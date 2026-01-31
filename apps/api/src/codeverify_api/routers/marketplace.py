"""AI Agent Marketplace API router.

This module provides:
- Agent registry and discovery
- Publisher portal with submission workflow
- Usage analytics
- Agent search and recommendations
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from codeverify_api.auth.dependencies import get_current_user, get_current_user_optional
from codeverify_api.db.database import get_db
from codeverify_api.config import settings
from codeverify_api.db.models import User
from codeverify_core.agent_sdk import (
    AgentCapability,
    AgentCategory,
    AgentLanguage,
    AgentManifest,
    AgentPackage,
)

logger = structlog.get_logger()

router = APIRouter()


# Database models would be added to models.py
# For now, using in-memory storage for demonstration
_agents_registry: dict[str, dict[str, Any]] = {}
_agent_versions: dict[str, list[dict[str, Any]]] = {}
_agent_downloads: dict[str, int] = {}
_agent_ratings: dict[str, list[dict[str, Any]]] = {}


# Request/Response Models
class PublishAgentRequest(BaseModel):
    """Request to publish a new agent or version."""

    manifest: AgentManifest
    readme: str | None = None
    changelog: str | None = None
    is_public: bool = True


class AgentListItem(BaseModel):
    """Agent item in list responses."""

    id: str
    qualified_name: str
    display_name: str | None
    version: str
    description: str
    author: str
    category: AgentCategory
    capabilities: list[AgentCapability]
    languages: list[AgentLanguage]
    tags: list[str]
    icon: str | None
    downloads: int
    rating: float
    rating_count: int
    is_free: bool
    price_per_analysis: float | None
    created_at: datetime
    updated_at: datetime


class AgentDetailResponse(BaseModel):
    """Detailed agent information."""

    id: str
    qualified_name: str
    display_name: str | None
    version: str
    description: str
    author: str
    category: AgentCategory
    capabilities: list[AgentCapability]
    languages: list[AgentLanguage]
    tags: list[str]
    icon: str | None
    homepage: str | None
    repository: str | None
    license: str
    readme: str | None
    changelog: str | None
    downloads: int
    rating: float
    rating_count: int
    is_free: bool
    price_per_analysis: float | None
    is_verified: bool
    is_featured: bool
    versions: list[str]
    dependencies: list[str]
    requires_network: bool
    requires_filesystem: bool
    max_memory_mb: int
    max_cpu_seconds: int
    created_at: datetime
    updated_at: datetime


class AgentVersionResponse(BaseModel):
    """Agent version details."""

    version: str
    changelog: str | None
    published_at: datetime
    download_url: str
    checksum: str


class SearchAgentsRequest(BaseModel):
    """Agent search parameters."""

    query: str | None = None
    category: AgentCategory | None = None
    capabilities: list[AgentCapability] | None = None
    languages: list[AgentLanguage] | None = None
    tags: list[str] | None = None
    is_free: bool | None = None
    is_verified: bool | None = None
    author: str | None = None
    sort_by: str = "downloads"  # downloads, rating, updated, name
    sort_order: str = "desc"
    page: int = 1
    per_page: int = 20


class SearchAgentsResponse(BaseModel):
    """Agent search results."""

    agents: list[AgentListItem]
    total: int
    page: int
    per_page: int
    total_pages: int


class RatingRequest(BaseModel):
    """Request to rate an agent."""

    rating: int = Field(..., ge=1, le=5)
    review: str | None = None


class RatingResponse(BaseModel):
    """Rating response."""

    agent_id: str
    rating: int
    review: str | None
    user_id: str
    created_at: datetime


class UsageStatsResponse(BaseModel):
    """Agent usage statistics."""

    agent_id: str
    total_downloads: int
    total_analyses: int
    unique_users: int
    avg_execution_time_ms: float
    success_rate: float
    findings_per_analysis: float
    daily_stats: list[dict[str, Any]]


class PublisherProfile(BaseModel):
    """Publisher profile."""

    username: str
    display_name: str | None
    avatar_url: str | None
    bio: str | None
    website: str | None
    verified: bool
    agent_count: int
    total_downloads: int
    joined_at: datetime


# Helper functions
def _calculate_rating(agent_id: str) -> tuple[float, int]:
    """Calculate average rating and count for an agent."""
    ratings = _agent_ratings.get(agent_id, [])
    if not ratings:
        return 0.0, 0
    avg = sum(r["rating"] for r in ratings) / len(ratings)
    return round(avg, 1), len(ratings)


def _agent_to_list_item(agent: dict[str, Any]) -> AgentListItem:
    """Convert stored agent to list item."""
    rating, rating_count = _calculate_rating(agent["id"])
    return AgentListItem(
        id=agent["id"],
        qualified_name=agent["qualified_name"],
        display_name=agent.get("display_name"),
        version=agent["version"],
        description=agent["description"],
        author=agent["author"],
        category=AgentCategory(agent["category"]),
        capabilities=[AgentCapability(c) for c in agent["capabilities"]],
        languages=[AgentLanguage(l) for l in agent["languages"]],
        tags=agent.get("tags", []),
        icon=agent.get("icon"),
        downloads=_agent_downloads.get(agent["id"], 0),
        rating=rating,
        rating_count=rating_count,
        is_free=agent.get("is_free", True),
        price_per_analysis=agent.get("price_per_analysis"),
        created_at=agent["created_at"],
        updated_at=agent["updated_at"],
    )


# API Endpoints
@router.get("/agents", response_model=SearchAgentsResponse)
async def search_agents(
    query: str | None = None,
    category: AgentCategory | None = None,
    capability: AgentCapability | None = None,
    language: AgentLanguage | None = None,
    tag: str | None = None,
    is_free: bool | None = None,
    author: str | None = None,
    sort_by: str = Query(default="downloads", pattern="^(downloads|rating|updated|name)$"),
    sort_order: str = Query(default="desc", pattern="^(asc|desc)$"),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
) -> SearchAgentsResponse:
    """
    Search and browse marketplace agents.
    
    Filter by category, capabilities, languages, tags, price, and author.
    Sort by downloads, rating, update date, or name.
    """
    # Filter agents
    agents = list(_agents_registry.values())

    if query:
        query_lower = query.lower()
        agents = [
            a for a in agents
            if query_lower in a["qualified_name"].lower()
            or query_lower in a["description"].lower()
            or any(query_lower in t.lower() for t in a.get("tags", []))
        ]

    if category:
        agents = [a for a in agents if a["category"] == category.value]

    if capability:
        agents = [a for a in agents if capability.value in a["capabilities"]]

    if language:
        agents = [
            a for a in agents
            if language.value in a["languages"] or "all" in a["languages"]
        ]

    if tag:
        agents = [a for a in agents if tag.lower() in [t.lower() for t in a.get("tags", [])]]

    if is_free is not None:
        agents = [a for a in agents if a.get("is_free", True) == is_free]

    if author:
        agents = [a for a in agents if a["author"].lower() == author.lower()]

    # Sort
    def sort_key(a: dict[str, Any]) -> Any:
        if sort_by == "downloads":
            return _agent_downloads.get(a["id"], 0)
        elif sort_by == "rating":
            return _calculate_rating(a["id"])[0]
        elif sort_by == "updated":
            return a["updated_at"]
        else:
            return a["qualified_name"].lower()

    agents.sort(key=sort_key, reverse=(sort_order == "desc"))

    # Paginate
    total = len(agents)
    start = (page - 1) * per_page
    end = start + per_page
    page_agents = agents[start:end]

    return SearchAgentsResponse(
        agents=[_agent_to_list_item(a) for a in page_agents],
        total=total,
        page=page,
        per_page=per_page,
        total_pages=(total + per_page - 1) // per_page,
    )


@router.get("/agents/{author}/{name}", response_model=AgentDetailResponse)
async def get_agent(
    author: str,
    name: str,
    version: str | None = None,
) -> AgentDetailResponse:
    """Get detailed information about an agent."""
    qualified_name = f"{author}/{name}"

    if qualified_name not in _agents_registry:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = _agents_registry[qualified_name]
    rating, rating_count = _calculate_rating(agent["id"])
    versions = [v["version"] for v in _agent_versions.get(agent["id"], [])]

    return AgentDetailResponse(
        id=agent["id"],
        qualified_name=agent["qualified_name"],
        display_name=agent.get("display_name"),
        version=agent["version"],
        description=agent["description"],
        author=agent["author"],
        category=AgentCategory(agent["category"]),
        capabilities=[AgentCapability(c) for c in agent["capabilities"]],
        languages=[AgentLanguage(l) for l in agent["languages"]],
        tags=agent.get("tags", []),
        icon=agent.get("icon"),
        homepage=agent.get("homepage"),
        repository=agent.get("repository"),
        license=agent.get("license", "MIT"),
        readme=agent.get("readme"),
        changelog=agent.get("changelog"),
        downloads=_agent_downloads.get(agent["id"], 0),
        rating=rating,
        rating_count=rating_count,
        is_free=agent.get("is_free", True),
        price_per_analysis=agent.get("price_per_analysis"),
        is_verified=agent.get("is_verified", False),
        is_featured=agent.get("is_featured", False),
        versions=versions,
        dependencies=agent.get("dependencies", []),
        requires_network=agent.get("requires_network", False),
        requires_filesystem=agent.get("requires_filesystem", True),
        max_memory_mb=agent.get("max_memory_mb", 512),
        max_cpu_seconds=agent.get("max_cpu_seconds", 60),
        created_at=agent["created_at"],
        updated_at=agent["updated_at"],
    )


@router.post("/agents", response_model=AgentDetailResponse)
async def publish_agent(
    file: UploadFile = File(...),
    readme: str | None = None,
    changelog: str | None = None,
    is_public: bool = True,
    current_user: dict = Depends(get_current_user),
) -> AgentDetailResponse:
    """
    Publish a new agent or new version to the marketplace.
    
    Upload a .cvagent package file.
    """
    # Read package
    content = await file.read()

    try:
        manifest, files = AgentPackage.read_from_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid package: {e}")

    # Validate author matches current user
    # In production, use actual user authentication
    user_id = current_user.get("id", "unknown")
    user_name = current_user.get("username", manifest.author)

    # Check if updating existing agent
    qualified_name = f"{manifest.author}/{manifest.name}"
    is_update = qualified_name in _agents_registry

    if is_update:
        existing = _agents_registry[qualified_name]
        # Check ownership
        if existing.get("owner_id") != user_id:
            raise HTTPException(status_code=403, detail="You don't own this agent")

    # Store agent
    agent_id = str(manifest.id)
    now = datetime.utcnow()

    agent_data = {
        "id": agent_id,
        "qualified_name": qualified_name,
        "display_name": manifest.display_name,
        "name": manifest.name,
        "version": manifest.version,
        "description": manifest.description,
        "author": manifest.author,
        "owner_id": user_id,
        "category": manifest.category.value,
        "capabilities": [c.value for c in manifest.capabilities],
        "languages": [l.value for l in manifest.languages],
        "tags": manifest.tags,
        "icon": manifest.icon,
        "homepage": manifest.homepage,
        "repository": manifest.repository,
        "license": manifest.license,
        "dependencies": manifest.dependencies,
        "requires_network": manifest.requires_network,
        "requires_filesystem": manifest.requires_filesystem,
        "max_memory_mb": manifest.max_memory_mb,
        "max_cpu_seconds": manifest.max_cpu_seconds,
        "is_free": manifest.is_free,
        "price_per_analysis": manifest.price_per_analysis,
        "is_public": is_public,
        "readme": readme,
        "changelog": changelog,
        "package_checksum": hashlib.sha256(content).hexdigest(),
        "created_at": existing["created_at"] if is_update else now,
        "updated_at": now,
    }

    _agents_registry[qualified_name] = agent_data

    # Store version history
    if agent_id not in _agent_versions:
        _agent_versions[agent_id] = []

    _agent_versions[agent_id].append({
        "version": manifest.version,
        "changelog": changelog,
        "published_at": now,
        "checksum": agent_data["package_checksum"],
    })

    # Initialize download counter
    if agent_id not in _agent_downloads:
        _agent_downloads[agent_id] = 0

    logger.info(
        "Agent published",
        agent=qualified_name,
        version=manifest.version,
        publisher=user_name,
    )

    return await get_agent(manifest.author, manifest.name)


@router.get("/agents/{author}/{name}/download")
async def download_agent(
    author: str,
    name: str,
    version: str | None = None,
) -> dict[str, str]:
    """
    Get download URL for an agent package.
    
    Returns a signed URL for downloading the .cvagent file.
    """
    qualified_name = f"{author}/{name}"

    if qualified_name not in _agents_registry:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = _agents_registry[qualified_name]

    # Increment download counter
    _agent_downloads[agent["id"]] = _agent_downloads.get(agent["id"], 0) + 1

    # In production, return a signed S3 URL
    # For now, return a placeholder
    return {
        "download_url": f"https://codeverify.dev/api/v1/marketplace/packages/{author}/{name}/{version or agent['version']}.cvagent",
        "checksum": agent.get("package_checksum", ""),
        "version": version or agent["version"],
    }


@router.get("/agents/{author}/{name}/versions", response_model=list[AgentVersionResponse])
async def list_versions(
    author: str,
    name: str,
) -> list[AgentVersionResponse]:
    """List all versions of an agent."""
    qualified_name = f"{author}/{name}"

    if qualified_name not in _agents_registry:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = _agents_registry[qualified_name]
    versions = _agent_versions.get(agent["id"], [])

    return [
        AgentVersionResponse(
            version=v["version"],
            changelog=v.get("changelog"),
            published_at=v["published_at"],
            download_url=f"https://codeverify.dev/api/v1/marketplace/packages/{author}/{name}/{v['version']}.cvagent",
            checksum=v.get("checksum", ""),
        )
        for v in reversed(versions)
    ]


@router.delete("/agents/{author}/{name}")
async def delete_agent(
    author: str,
    name: str,
    current_user: dict = Depends(get_current_user),
) -> dict[str, str]:
    """Delete an agent from the marketplace."""
    qualified_name = f"{author}/{name}"

    if qualified_name not in _agents_registry:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = _agents_registry[qualified_name]
    user_id = current_user.get("id", "unknown")

    if agent.get("owner_id") != user_id:
        raise HTTPException(status_code=403, detail="You don't own this agent")

    # Remove agent
    del _agents_registry[qualified_name]
    _agent_versions.pop(agent["id"], None)
    _agent_downloads.pop(agent["id"], None)
    _agent_ratings.pop(agent["id"], None)

    return {"message": f"Agent {qualified_name} deleted"}


# Ratings and reviews
@router.post("/agents/{author}/{name}/ratings", response_model=RatingResponse)
async def rate_agent(
    author: str,
    name: str,
    request: RatingRequest,
    current_user: dict = Depends(get_current_user),
) -> RatingResponse:
    """Rate and optionally review an agent."""
    qualified_name = f"{author}/{name}"

    if qualified_name not in _agents_registry:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = _agents_registry[qualified_name]
    user_id = current_user.get("id", "unknown")

    # Check if user already rated
    agent_id = agent["id"]
    if agent_id not in _agent_ratings:
        _agent_ratings[agent_id] = []

    # Update or add rating
    existing_idx = next(
        (i for i, r in enumerate(_agent_ratings[agent_id]) if r["user_id"] == user_id),
        None,
    )

    rating_data = {
        "agent_id": agent_id,
        "user_id": user_id,
        "rating": request.rating,
        "review": request.review,
        "created_at": datetime.utcnow(),
    }

    if existing_idx is not None:
        _agent_ratings[agent_id][existing_idx] = rating_data
    else:
        _agent_ratings[agent_id].append(rating_data)

    return RatingResponse(**rating_data)


@router.get("/agents/{author}/{name}/ratings")
async def get_ratings(
    author: str,
    name: str,
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    """Get ratings and reviews for an agent."""
    qualified_name = f"{author}/{name}"

    if qualified_name not in _agents_registry:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = _agents_registry[qualified_name]
    ratings = _agent_ratings.get(agent["id"], [])

    # Paginate
    total = len(ratings)
    start = (page - 1) * per_page
    end = start + per_page
    page_ratings = ratings[start:end]

    avg_rating, count = _calculate_rating(agent["id"])

    return {
        "ratings": page_ratings,
        "average_rating": avg_rating,
        "total_ratings": count,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
        "distribution": {
            5: sum(1 for r in ratings if r["rating"] == 5),
            4: sum(1 for r in ratings if r["rating"] == 4),
            3: sum(1 for r in ratings if r["rating"] == 3),
            2: sum(1 for r in ratings if r["rating"] == 2),
            1: sum(1 for r in ratings if r["rating"] == 1),
        },
    }


# Usage analytics
@router.get("/agents/{author}/{name}/stats", response_model=UsageStatsResponse)
async def get_agent_stats(
    author: str,
    name: str,
    days: int = Query(default=30, ge=1, le=365),
    current_user: dict = Depends(get_current_user),
) -> UsageStatsResponse:
    """
    Get usage statistics for an agent.
    
    Only available to agent owner.
    """
    qualified_name = f"{author}/{name}"

    if qualified_name not in _agents_registry:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = _agents_registry[qualified_name]
    user_id = current_user.get("id", "unknown")

    if agent.get("owner_id") != user_id:
        raise HTTPException(status_code=403, detail="You don't own this agent")

    # In production, query actual analytics
    # For now, return placeholder data
    return UsageStatsResponse(
        agent_id=agent["id"],
        total_downloads=_agent_downloads.get(agent["id"], 0),
        total_analyses=0,
        unique_users=0,
        avg_execution_time_ms=0,
        success_rate=0,
        findings_per_analysis=0,
        daily_stats=[],
    )


# Featured and recommended
@router.get("/featured", response_model=list[AgentListItem])
async def get_featured_agents() -> list[AgentListItem]:
    """Get featured agents curated by CodeVerify team."""
    agents = [a for a in _agents_registry.values() if a.get("is_featured", False)]
    return [_agent_to_list_item(a) for a in agents[:10]]


@router.get("/popular", response_model=list[AgentListItem])
async def get_popular_agents(
    limit: int = Query(default=10, ge=1, le=50),
) -> list[AgentListItem]:
    """Get most popular agents by downloads."""
    agents = sorted(
        _agents_registry.values(),
        key=lambda a: _agent_downloads.get(a["id"], 0),
        reverse=True,
    )
    return [_agent_to_list_item(a) for a in agents[:limit]]


@router.get("/recent", response_model=list[AgentListItem])
async def get_recent_agents(
    limit: int = Query(default=10, ge=1, le=50),
) -> list[AgentListItem]:
    """Get most recently updated agents."""
    agents = sorted(
        _agents_registry.values(),
        key=lambda a: a["updated_at"],
        reverse=True,
    )
    return [_agent_to_list_item(a) for a in agents[:limit]]


@router.get("/categories", response_model=list[dict[str, Any]])
async def list_categories() -> list[dict[str, Any]]:
    """List all agent categories with counts."""
    category_counts: dict[str, int] = {}
    for agent in _agents_registry.values():
        cat = agent["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    return [
        {
            "category": cat.value,
            "display_name": cat.value.replace("-", " ").title(),
            "count": category_counts.get(cat.value, 0),
        }
        for cat in AgentCategory
    ]


# Publisher portal
@router.get("/publishers/{username}", response_model=PublisherProfile)
async def get_publisher_profile(username: str) -> PublisherProfile:
    """Get publisher profile and their agents."""
    # Find agents by this author
    user_agents = [a for a in _agents_registry.values() if a["author"] == username]

    if not user_agents:
        raise HTTPException(status_code=404, detail="Publisher not found")

    total_downloads = sum(_agent_downloads.get(a["id"], 0) for a in user_agents)

    return PublisherProfile(
        username=username,
        display_name=None,
        avatar_url=None,
        bio=None,
        website=None,
        verified=False,
        agent_count=len(user_agents),
        total_downloads=total_downloads,
        joined_at=min(a["created_at"] for a in user_agents),
    )


@router.get("/publishers/{username}/agents", response_model=list[AgentListItem])
async def list_publisher_agents(
    username: str,
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
) -> list[AgentListItem]:
    """List all agents by a publisher."""
    user_agents = [a for a in _agents_registry.values() if a["author"] == username]

    # Paginate
    start = (page - 1) * per_page
    end = start + per_page
    page_agents = user_agents[start:end]

    return [_agent_to_list_item(a) for a in page_agents]


# Agent templates
@router.get("/templates")
async def list_agent_templates() -> list[dict[str, Any]]:
    """List starter templates for agent development."""
    return [
        {
            "id": "basic",
            "name": "Basic Agent",
            "description": "Simple agent template with analyze method",
            "languages": ["python"],
            "download_url": "https://codeverify.dev/templates/basic-agent.zip",
        },
        {
            "id": "security",
            "name": "Security Scanner",
            "description": "Template for security vulnerability detection",
            "languages": ["python"],
            "download_url": "https://codeverify.dev/templates/security-agent.zip",
        },
        {
            "id": "quality",
            "name": "Code Quality",
            "description": "Template for code quality and style checks",
            "languages": ["python"],
            "download_url": "https://codeverify.dev/templates/quality-agent.zip",
        },
        {
            "id": "ai-detection",
            "name": "AI Code Detector",
            "description": "Template for detecting AI-generated code patterns",
            "languages": ["python"],
            "download_url": "https://codeverify.dev/templates/ai-detection-agent.zip",
        },
        {
            "id": "formal",
            "name": "Formal Verifier",
            "description": "Template using Z3 for formal verification",
            "languages": ["python"],
            "download_url": "https://codeverify.dev/templates/formal-agent.zip",
        },
    ]
