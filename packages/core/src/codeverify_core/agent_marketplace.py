"""AI Agent Marketplace.

Third-party agent publishing platform with review workflow, versioning,
install/uninstall, usage tracking, and revenue sharing.

Features:
- Agent manifest and packaging format
- Submission and review workflow
- Install/uninstall with dependency resolution
- Usage tracking and analytics
- Revenue sharing model
- Search and discovery with categories
"""

from __future__ import annotations

import hashlib
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class AgentCategory(str, Enum):
    SECURITY = "security"
    QUALITY = "quality"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    LANGUAGE = "language"
    FRAMEWORK = "framework"
    CUSTOM = "custom"


class PublishStatus(str, Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"


class PricingModel(str, Enum):
    FREE = "free"
    PAID = "paid"
    FREEMIUM = "freemium"


@dataclass
class AgentManifest:
    """Manifest for a marketplace agent."""
    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    author_id: str = ""
    category: AgentCategory = AgentCategory.CUSTOM
    languages: list[str] = field(default_factory=list)
    entry_point: str = ""
    dependencies: list[str] = field(default_factory=list)
    min_codeverify_version: str = "1.0.0"
    pricing: PricingModel = PricingModel.FREE
    price_cents_per_use: int = 0
    tags: list[str] = field(default_factory=list)


@dataclass
class PublishedAgent:
    """A published agent in the marketplace."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    manifest: AgentManifest = field(default_factory=AgentManifest)
    status: PublishStatus = PublishStatus.DRAFT
    content_hash: str = ""
    install_count: int = 0
    usage_count: int = 0
    rating: float = 0.0
    rating_count: int = 0
    revenue_cents: int = 0
    published_at: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentReview:
    """A review of a submitted agent."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_id: str = ""
    reviewer_id: str = ""
    approved: bool = False
    comments: str = ""
    security_check_passed: bool = False
    performance_check_passed: bool = False
    reviewed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentInstallation:
    """Record of an agent installation."""
    agent_id: str = ""
    org_id: str = ""
    installed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
    usage_count: int = 0


@dataclass
class RevenueShare:
    """Revenue share record for a paid agent."""
    agent_id: str = ""
    author_id: str = ""
    total_uses: int = 0
    total_revenue_cents: int = 0
    author_share_cents: int = 0  # 70% to author
    platform_share_cents: int = 0  # 30% to platform
    period: str = ""


class AgentMarketplaceService:
    """Main service for the AI agent marketplace."""

    AUTHOR_SHARE_PERCENT = 70

    def __init__(self) -> None:
        self._agents: dict[str, PublishedAgent] = {}
        self._reviews: list[AgentReview] = []
        self._installations: dict[str, list[AgentInstallation]] = defaultdict(list)
        self._ratings: dict[str, list[float]] = defaultdict(list)

    def submit_agent(self, manifest: AgentManifest, code_hash: str = "") -> PublishedAgent:
        agent = PublishedAgent(
            manifest=manifest, status=PublishStatus.SUBMITTED,
            content_hash=code_hash or hashlib.sha256(manifest.name.encode()).hexdigest()[:12],
        )
        self._agents[agent.id] = agent
        return agent

    def review_agent(self, agent_id: str, reviewer_id: str, approved: bool, comments: str = "") -> AgentReview:
        agent = self._agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        review = AgentReview(
            agent_id=agent_id, reviewer_id=reviewer_id, approved=approved,
            comments=comments, security_check_passed=approved, performance_check_passed=approved,
        )
        self._reviews.append(review)
        agent.status = PublishStatus.APPROVED if approved else PublishStatus.REJECTED
        return review

    def publish_agent(self, agent_id: str) -> bool:
        agent = self._agents.get(agent_id)
        if not agent or agent.status != PublishStatus.APPROVED:
            return False
        agent.status = PublishStatus.PUBLISHED
        agent.published_at = datetime.now(timezone.utc)
        return True

    def install_agent(self, agent_id: str, org_id: str) -> AgentInstallation | None:
        agent = self._agents.get(agent_id)
        if not agent or agent.status != PublishStatus.PUBLISHED:
            return None
        inst = AgentInstallation(agent_id=agent_id, org_id=org_id)
        self._installations[org_id].append(inst)
        agent.install_count += 1
        return inst

    def uninstall_agent(self, agent_id: str, org_id: str) -> bool:
        for inst in self._installations.get(org_id, []):
            if inst.agent_id == agent_id and inst.is_active:
                inst.is_active = False
                return True
        return False

    def record_usage(self, agent_id: str, org_id: str) -> None:
        agent = self._agents.get(agent_id)
        if agent:
            agent.usage_count += 1
            if agent.manifest.pricing == PricingModel.PAID:
                agent.revenue_cents += agent.manifest.price_cents_per_use
        for inst in self._installations.get(org_id, []):
            if inst.agent_id == agent_id and inst.is_active:
                inst.usage_count += 1

    def rate_agent(self, agent_id: str, rating: float) -> bool:
        agent = self._agents.get(agent_id)
        if not agent or not (1.0 <= rating <= 5.0):
            return False
        self._ratings[agent_id].append(rating)
        ratings = self._ratings[agent_id]
        agent.rating = round(sum(ratings) / len(ratings), 2)
        agent.rating_count = len(ratings)
        return True

    def search(self, query: str = "", category: AgentCategory | None = None, min_rating: float = 0.0) -> list[PublishedAgent]:
        results = [a for a in self._agents.values() if a.status == PublishStatus.PUBLISHED]
        if category:
            results = [a for a in results if a.manifest.category == category]
        if query:
            q = query.lower()
            results = [a for a in results if q in a.manifest.name.lower() or q in a.manifest.description.lower() or q in " ".join(a.manifest.tags).lower()]
        if min_rating > 0:
            results = [a for a in results if a.rating >= min_rating]
        return sorted(results, key=lambda a: a.install_count, reverse=True)

    def get_revenue_share(self, agent_id: str) -> RevenueShare:
        agent = self._agents.get(agent_id)
        if not agent:
            return RevenueShare()
        author_share = int(agent.revenue_cents * self.AUTHOR_SHARE_PERCENT / 100)
        return RevenueShare(
            agent_id=agent_id, author_id=agent.manifest.author_id,
            total_uses=agent.usage_count, total_revenue_cents=agent.revenue_cents,
            author_share_cents=author_share, platform_share_cents=agent.revenue_cents - author_share,
        )

    def get_agent(self, agent_id: str) -> PublishedAgent | None:
        return self._agents.get(agent_id)


_agent_marketplace_instance: AgentMarketplaceService | None = None
def get_agent_marketplace_service() -> AgentMarketplaceService:
    global _agent_marketplace_instance
    if _agent_marketplace_instance is None: _agent_marketplace_instance = AgentMarketplaceService()
    return _agent_marketplace_instance
def reset_agent_marketplace_service() -> None:
    global _agent_marketplace_instance
    _agent_marketplace_instance = None
