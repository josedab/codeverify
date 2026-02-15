"""Proof Marketplace — Community proof library with monetization and reputation.

Extends the proof repository with marketplace features: pricing tiers,
author reputation and leaderboards, licensing, and revenue sharing.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ProofLicense(str, Enum):
    """License types for shared proofs."""

    MIT = "MIT"
    APACHE_2 = "Apache-2.0"
    CC_BY = "CC-BY-4.0"
    CC_BY_SA = "CC-BY-SA-4.0"
    PROPRIETARY = "Proprietary"
    ENTERPRISE = "Enterprise"


class PricingTier(str, Enum):
    """Pricing tiers for marketplace proofs."""

    FREE = "free"
    BASIC = "basic"  # $5/proof
    PROFESSIONAL = "professional"  # $25/proof
    ENTERPRISE = "enterprise"  # Custom pricing


class BadgeType(str, Enum):
    """Reputation badges for proof authors."""

    NEWCOMER = "newcomer"
    CONTRIBUTOR = "contributor"
    EXPERT = "expert"
    MASTER = "master"
    LEGEND = "legend"
    SECURITY_SPECIALIST = "security_specialist"
    FORMAL_METHODS_GURU = "formal_methods_guru"


@dataclass
class AuthorProfile:
    """Profile for a proof marketplace author."""

    id: str
    username: str
    display_name: str
    proofs_published: int = 0
    proofs_sold: int = 0
    total_upvotes: int = 0
    total_downloads: int = 0
    reputation_score: float = 0.0
    badges: list[BadgeType] = field(default_factory=list)
    revenue_earned: float = 0.0
    revenue_share_percent: float = 70.0  # Author gets 70%
    joined_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def rank(self) -> BadgeType:
        """Calculate rank based on reputation score."""
        if self.reputation_score >= 10000:
            return BadgeType.LEGEND
        elif self.reputation_score >= 5000:
            return BadgeType.MASTER
        elif self.reputation_score >= 1000:
            return BadgeType.EXPERT
        elif self.reputation_score >= 100:
            return BadgeType.CONTRIBUTOR
        return BadgeType.NEWCOMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "display_name": self.display_name,
            "proofs_published": self.proofs_published,
            "total_upvotes": self.total_upvotes,
            "total_downloads": self.total_downloads,
            "reputation_score": round(self.reputation_score, 1),
            "rank": self.rank.value,
            "badges": [b.value for b in self.badges],
            "revenue_earned": round(self.revenue_earned, 2),
        }


@dataclass
class MarketplaceProof:
    """A proof listing in the marketplace."""

    id: str
    title: str
    description: str
    author_id: str
    category: str  # "null_safety", "bounds", "security", "concurrency"
    language: str
    proof_content: str  # Z3 assertions or specification
    example_code: str = ""
    license: ProofLicense = ProofLicense.MIT
    pricing_tier: PricingTier = PricingTier.FREE
    price: float = 0.0  # In USD
    upvotes: int = 0
    downvotes: int = 0
    downloads: int = 0
    verified: bool = False  # Staff-verified correctness
    tags: list[str] = field(default_factory=list)
    published_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0.0"

    @property
    def net_votes(self) -> int:
        return self.upvotes - self.downvotes

    @property
    def popularity_score(self) -> float:
        """Weighted score combining votes and downloads."""
        return (self.upvotes * 3) + self.downloads - (self.downvotes * 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "author_id": self.author_id,
            "category": self.category,
            "language": self.language,
            "license": self.license.value,
            "pricing_tier": self.pricing_tier.value,
            "price": self.price,
            "upvotes": self.upvotes,
            "downvotes": self.downvotes,
            "net_votes": self.net_votes,
            "downloads": self.downloads,
            "verified": self.verified,
            "tags": self.tags,
            "popularity_score": self.popularity_score,
            "version": self.version,
        }


@dataclass
class Purchase:
    """Record of a proof purchase."""

    id: str
    proof_id: str
    buyer_id: str
    author_id: str
    price: float
    author_revenue: float
    platform_fee: float
    purchased_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "proof_id": self.proof_id,
            "buyer_id": self.buyer_id,
            "price": self.price,
            "purchased_at": self.purchased_at.isoformat(),
        }


@dataclass
class LeaderboardEntry:
    """An entry on the reputation leaderboard."""

    rank: int
    author: AuthorProfile
    score: float
    highlight: str  # "Most proofs", "Top rated", etc.

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "username": self.author.username,
            "display_name": self.author.display_name,
            "score": round(self.score, 1),
            "badges": [b.value for b in self.author.badges],
            "highlight": self.highlight,
        }


class ProofMarketplace:
    """Community proof marketplace with monetization and reputation.

    Example:
        >>> marketplace = ProofMarketplace()
        >>> author = marketplace.register_author("alice", "Alice Smith")
        >>> proof = marketplace.publish_proof(
        ...     author_id=author.id,
        ...     title="Null Safety for Optional Parameters",
        ...     description="Z3 proof template for null safety checks",
        ...     category="null_safety",
        ...     language="python",
        ...     proof_content="(assert (not (= param None)))",
        ... )
        >>> marketplace.vote(proof.id, "bob-id", "up")
        >>> marketplace.purchase(proof.id, "bob-id")
    """

    def __init__(self, platform_fee_percent: float = 30.0) -> None:
        self._authors: dict[str, AuthorProfile] = {}
        self._proofs: dict[str, MarketplaceProof] = {}
        self._purchases: list[Purchase] = []
        self._user_purchases: dict[str, set[str]] = {}  # buyer_id -> set of proof_ids
        self._platform_fee_percent = platform_fee_percent

    def register_author(
        self, username: str, display_name: str
    ) -> AuthorProfile:
        """Register a new author on the marketplace."""
        author = AuthorProfile(
            id=str(uuid.uuid4()),
            username=username,
            display_name=display_name,
        )
        self._authors[author.id] = author
        return author

    def get_author(self, author_id: str) -> AuthorProfile | None:
        return self._authors.get(author_id)

    def publish_proof(
        self,
        author_id: str,
        title: str,
        description: str,
        category: str,
        language: str,
        proof_content: str,
        example_code: str = "",
        license: ProofLicense = ProofLicense.MIT,
        pricing_tier: PricingTier = PricingTier.FREE,
        price: float = 0.0,
        tags: list[str] | None = None,
    ) -> MarketplaceProof | None:
        """Publish a proof to the marketplace."""
        author = self._authors.get(author_id)
        if not author:
            logger.warning("Unknown author", author_id=author_id)
            return None

        proof = MarketplaceProof(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            author_id=author_id,
            category=category,
            language=language,
            proof_content=proof_content,
            example_code=example_code,
            license=license,
            pricing_tier=pricing_tier,
            price=price,
            tags=tags or [],
        )
        self._proofs[proof.id] = proof
        author.proofs_published += 1
        author.reputation_score += 10  # Publishing earns reputation

        logger.info("Proof published", proof_id=proof.id, author=author.username)
        return proof

    def vote(
        self, proof_id: str, voter_id: str, vote: str
    ) -> bool:
        """Vote on a proof (up/down)."""
        proof = self._proofs.get(proof_id)
        if not proof:
            return False

        author = self._authors.get(proof.author_id)
        if vote == "up":
            proof.upvotes += 1
            if author:
                author.total_upvotes += 1
                author.reputation_score += 2
        elif vote == "down":
            proof.downvotes += 1
            if author:
                author.reputation_score = max(0, author.reputation_score - 1)

        return True

    def download(self, proof_id: str, user_id: str) -> str | None:
        """Download a proof (checks purchase for paid proofs)."""
        proof = self._proofs.get(proof_id)
        if not proof:
            return None

        # Check if paid proof requires purchase
        if proof.price > 0:
            user_purchases = self._user_purchases.get(user_id, set())
            if proof_id not in user_purchases:
                logger.warning("Purchase required", proof_id=proof_id)
                return None

        proof.downloads += 1
        author = self._authors.get(proof.author_id)
        if author:
            author.total_downloads += 1
            author.reputation_score += 0.5

        return proof.proof_content

    def purchase(
        self, proof_id: str, buyer_id: str
    ) -> Purchase | None:
        """Purchase a paid proof."""
        proof = self._proofs.get(proof_id)
        if not proof or proof.price <= 0:
            return None

        # Check if already purchased
        if proof_id in self._user_purchases.get(buyer_id, set()):
            return None

        platform_fee = proof.price * (self._platform_fee_percent / 100)
        author_revenue = proof.price - platform_fee

        purchase = Purchase(
            id=str(uuid.uuid4()),
            proof_id=proof_id,
            buyer_id=buyer_id,
            author_id=proof.author_id,
            price=proof.price,
            author_revenue=author_revenue,
            platform_fee=platform_fee,
        )
        self._purchases.append(purchase)
        self._user_purchases.setdefault(buyer_id, set()).add(proof_id)

        # Update author revenue
        author = self._authors.get(proof.author_id)
        if author:
            author.proofs_sold += 1
            author.revenue_earned += author_revenue
            author.reputation_score += 5

        logger.info(
            "Proof purchased",
            proof_id=proof_id,
            buyer=buyer_id,
            price=proof.price,
        )
        return purchase

    def search(
        self,
        query: str = "",
        category: str | None = None,
        language: str | None = None,
        pricing_tier: PricingTier | None = None,
        sort_by: str = "popularity",
        limit: int = 20,
    ) -> list[MarketplaceProof]:
        """Search the marketplace for proofs."""
        results = list(self._proofs.values())

        if query:
            lower_query = query.lower()
            results = [
                p for p in results
                if lower_query in p.title.lower()
                or lower_query in p.description.lower()
                or any(lower_query in t.lower() for t in p.tags)
            ]

        if category:
            results = [p for p in results if p.category == category]
        if language:
            results = [p for p in results if p.language == language]
        if pricing_tier:
            results = [p for p in results if p.pricing_tier == pricing_tier]

        if sort_by == "popularity":
            results.sort(key=lambda p: p.popularity_score, reverse=True)
        elif sort_by == "newest":
            results.sort(key=lambda p: p.published_at, reverse=True)
        elif sort_by == "price_low":
            results.sort(key=lambda p: p.price)
        elif sort_by == "votes":
            results.sort(key=lambda p: p.net_votes, reverse=True)

        return results[:limit]

    def get_leaderboard(self, limit: int = 10) -> list[LeaderboardEntry]:
        """Get the reputation leaderboard."""
        authors = sorted(
            self._authors.values(),
            key=lambda a: a.reputation_score,
            reverse=True,
        )

        entries = []
        for rank, author in enumerate(authors[:limit], 1):
            if author.total_upvotes > author.total_downloads:
                highlight = "Top rated"
            elif author.proofs_published > 10:
                highlight = "Most proofs"
            elif author.revenue_earned > 100:
                highlight = "Top earner"
            else:
                highlight = "Active contributor"

            entries.append(
                LeaderboardEntry(
                    rank=rank,
                    author=author,
                    score=author.reputation_score,
                    highlight=highlight,
                )
            )
        return entries

    def get_marketplace_stats(self) -> dict[str, Any]:
        """Get overall marketplace statistics."""
        total_revenue = sum(p.price for p in self._purchases)
        return {
            "total_proofs": len(self._proofs),
            "total_authors": len(self._authors),
            "total_purchases": len(self._purchases),
            "total_revenue": round(total_revenue, 2),
            "total_downloads": sum(p.downloads for p in self._proofs.values()),
            "free_proofs": sum(
                1 for p in self._proofs.values() if p.pricing_tier == PricingTier.FREE
            ),
            "paid_proofs": sum(
                1 for p in self._proofs.values() if p.price > 0
            ),
            "verified_proofs": sum(
                1 for p in self._proofs.values() if p.verified
            ),
            "categories": list({p.category for p in self._proofs.values()}),
        }
