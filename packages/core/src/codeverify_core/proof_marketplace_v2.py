"""Proof Marketplace V2 — Enhanced community proof library with gamification.

Builds on the original proof marketplace with community features, quality
metrics, TF-IDF-like proof search, contributor gamification, and structured
review workflows.  Designed for large-scale proof reuse across organizations.
"""

from __future__ import annotations

import math
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


# =============================================================================
# Enums
# =============================================================================


class ProofCategory(str, Enum):
    NULL_SAFETY = "null_safety"
    BOUNDS_CHECK = "bounds_check"
    OVERFLOW = "overflow"
    TYPE_SAFETY = "type_safety"
    CONCURRENCY = "concurrency"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CORRECTNESS = "correctness"
    CUSTOM = "custom"


class QualityTier(str, Enum):
    UNREVIEWED = "unreviewed"
    COMMUNITY_REVIEWED = "community_reviewed"
    EXPERT_REVIEWED = "expert_reviewed"
    FORMALLY_VERIFIED = "formally_verified"


class ContributionType(str, Enum):
    PROOF_SUBMITTED = "proof_submitted"
    PROOF_REVIEWED = "proof_reviewed"
    BUG_REPORTED = "bug_reported"
    IMPROVEMENT_SUGGESTED = "improvement_suggested"
    DOCUMENTATION_ADDED = "documentation_added"


class SearchSortBy(str, Enum):
    RELEVANCE = "relevance"
    DOWNLOADS = "downloads"
    RATING = "rating"
    RECENT = "recent"
    QUALITY = "quality"


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class ProofMetadata:
    """Core metadata describing a published proof."""

    id: str
    name: str
    description: str
    category: ProofCategory
    author_id: str
    language: str
    framework: str | None
    tags: list[str]
    version: str = "1.0.0"
    quality_tier: QualityTier = QualityTier.UNREVIEWED
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "description": self.description,
            "category": self.category.value, "author_id": self.author_id,
            "language": self.language, "framework": self.framework,
            "tags": self.tags, "version": self.version,
            "quality_tier": self.quality_tier.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class ProofContent:
    """The actual proof content — Z3 expression, natural language, and tests."""

    proof_id: str
    z3_expression: str
    natural_language: str
    code_pattern: str
    test_cases: list[dict[str, Any]]
    applicable_languages: list[str]
    prerequisites: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof_id": self.proof_id, "z3_expression": self.z3_expression,
            "natural_language": self.natural_language,
            "code_pattern": self.code_pattern, "test_cases": self.test_cases,
            "applicable_languages": self.applicable_languages,
            "prerequisites": self.prerequisites,
        }


@dataclass
class ProofQualityMetrics:
    """Aggregated quality signals for a proof."""

    proof_id: str
    success_rate: float
    false_positive_rate: float
    reuse_count: int
    avg_rating: float
    review_count: int
    bug_reports: int
    last_verified: datetime | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof_id": self.proof_id,
            "success_rate": round(self.success_rate, 3),
            "false_positive_rate": round(self.false_positive_rate, 3),
            "reuse_count": self.reuse_count, "avg_rating": round(self.avg_rating, 2),
            "review_count": self.review_count, "bug_reports": self.bug_reports,
            "last_verified": self.last_verified.isoformat() if self.last_verified else None,
        }


@dataclass
class ProofReview:
    """A community review of a proof."""

    id: str
    proof_id: str
    reviewer_id: str
    rating: int
    comment: str
    quality_assessment: QualityTier
    issues_found: list[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    helpful_votes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "proof_id": self.proof_id,
            "reviewer_id": self.reviewer_id, "rating": self.rating,
            "comment": self.comment,
            "quality_assessment": self.quality_assessment.value,
            "issues_found": self.issues_found,
            "created_at": self.created_at.isoformat(),
            "helpful_votes": self.helpful_votes,
        }


@dataclass
class ContributorProfile:
    """Gamification profile for a marketplace contributor."""

    user_id: str
    display_name: str
    proofs_submitted: int = 0
    proofs_reviewed: int = 0
    reputation_points: int = 0
    badges: list[str] = field(default_factory=list)
    contributions: list[dict] = field(default_factory=list)
    rank: int = 0
    joined_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id, "display_name": self.display_name,
            "proofs_submitted": self.proofs_submitted,
            "proofs_reviewed": self.proofs_reviewed,
            "reputation_points": self.reputation_points,
            "badges": self.badges, "rank": self.rank,
            "joined_at": self.joined_at.isoformat() if self.joined_at else None,
        }


@dataclass
class SearchQuery:
    """Structured search query for the proof marketplace."""

    query: str
    category: ProofCategory | None = None
    language: str | None = None
    min_rating: float = 0.0
    quality_tier: QualityTier | None = None
    sort_by: SearchSortBy = SearchSortBy.RELEVANCE
    limit: int = 20
    offset: int = 0


@dataclass
class SearchResult:
    """Paginated search result with facets."""

    proofs: list[ProofMetadata]
    total_count: int
    facets: dict[str, dict[str, int]]
    query_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "proofs": [p.to_dict() for p in self.proofs],
            "total_count": self.total_count, "facets": self.facets,
            "query_time_ms": round(self.query_time_ms, 2),
        }


@dataclass
class LeaderboardEntry:
    """A single entry on the contributor leaderboard."""

    rank: int
    user_id: str
    display_name: str
    reputation_points: int
    proofs_count: int
    avg_rating: float
    badges: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank, "user_id": self.user_id,
            "display_name": self.display_name,
            "reputation_points": self.reputation_points,
            "proofs_count": self.proofs_count,
            "avg_rating": round(self.avg_rating, 2), "badges": self.badges,
        }


# =============================================================================
# Proof Storage
# =============================================================================


class ProofStorage:
    """In-memory storage backend for proof metadata and content."""

    def __init__(self) -> None:
        self._metadata: dict[str, ProofMetadata] = {}
        self._content: dict[str, ProofContent] = {}

    def store_proof(self, metadata: ProofMetadata, content: ProofContent) -> str:
        self._metadata[metadata.id] = metadata
        self._content[metadata.id] = content
        logger.info("Proof stored", proof_id=metadata.id, name=metadata.name)
        return metadata.id

    def get_proof(self, proof_id: str) -> tuple[ProofMetadata, ProofContent] | None:
        metadata = self._metadata.get(proof_id)
        content = self._content.get(proof_id)
        if metadata and content:
            return metadata, content
        return None

    def update_proof(self, proof_id: str, updates: dict) -> bool:
        metadata = self._metadata.get(proof_id)
        if not metadata:
            return False
        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        metadata.updated_at = datetime.now(UTC)
        return True

    def delete_proof(self, proof_id: str) -> bool:
        if proof_id not in self._metadata:
            return False
        del self._metadata[proof_id]
        self._content.pop(proof_id, None)
        logger.info("Proof deleted", proof_id=proof_id)
        return True

    def list_proofs(self, author_id: str | None = None, limit: int = 20) -> list[ProofMetadata]:
        proofs = list(self._metadata.values())
        if author_id:
            proofs = [p for p in proofs if p.author_id == author_id]
        proofs.sort(key=lambda p: p.created_at, reverse=True)
        return proofs[:limit]


# =============================================================================
# Proof Search Engine
# =============================================================================


class ProofSearchEngine:
    """TF-IDF-like search engine over the proof corpus.

    Tokenizes proof names, descriptions, and tags to build an inverted index.
    Supports text matching, category/language filters, and facet generation.
    """

    def __init__(self) -> None:
        self._storage: ProofStorage | None = None
        self._index: dict[str, set[str]] = defaultdict(set)
        self._doc_freq: dict[str, int] = defaultdict(int)
        self._total_docs: int = 0
        self._downloads: dict[str, int] = defaultdict(int)
        self._ratings: dict[str, list[int]] = defaultdict(list)

    def bind_storage(self, storage: ProofStorage) -> None:
        self._storage = storage

    # -- Indexing -------------------------------------------------------------

    def index_proof(self, metadata: ProofMetadata) -> None:
        terms = self._tokenize(metadata)
        for term in terms:
            if metadata.id not in self._index[term]:
                self._index[term].add(metadata.id)
                self._doc_freq[term] += 1
        self._total_docs += 1

    @staticmethod
    def _tokenize(metadata: ProofMetadata) -> set[str]:
        text = " ".join([
            metadata.name, metadata.description, metadata.language,
            metadata.framework or "", metadata.category.value,
            " ".join(metadata.tags),
        ])
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    # -- Search ---------------------------------------------------------------

    def search(self, query: SearchQuery) -> SearchResult:
        start = time.monotonic()
        if not self._storage:
            return SearchResult(proofs=[], total_count=0, facets={}, query_time_ms=0.0)

        all_proofs = list(self._storage._metadata.values())
        query_terms = set(re.findall(r"[a-z0-9]+", query.query.lower()))

        scored = [(p, self._compute_relevance(p, query_terms)) for p in all_proofs]
        filtered = self._apply_filters(scored, query)
        filtered = self._sort_results(filtered, query.sort_by)

        total_count = len(filtered)
        facets = self._generate_facets([p for p, _ in filtered])
        page = filtered[query.offset : query.offset + query.limit]

        return SearchResult(
            proofs=[p for p, _ in page], total_count=total_count,
            facets=facets, query_time_ms=(time.monotonic() - start) * 1000,
        )

    def find_similar(self, code_pattern: str, language: str) -> list[ProofMetadata]:
        if not self._storage:
            return []
        pattern_terms = set(re.findall(r"[a-z0-9]+", code_pattern.lower()))
        results: list[tuple[ProofMetadata, float]] = []
        for proof_id, content in self._storage._content.items():
            if language and language not in content.applicable_languages:
                continue
            content_terms = set(re.findall(r"[a-z0-9]+", content.code_pattern.lower()))
            overlap = len(pattern_terms & content_terms)
            union = len(pattern_terms | content_terms)
            if union > 0 and overlap > 0:
                metadata = self._storage._metadata.get(proof_id)
                if metadata:
                    results.append((metadata, overlap / union))
        results.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in results[:10]]

    def _compute_relevance(self, proof: ProofMetadata, query_terms: set[str]) -> float:
        if not query_terms:
            return 1.0
        proof_terms = self._tokenize(proof)
        score = 0.0
        for term in query_terms:
            if term in proof_terms:
                tf = 1.0
                df = self._doc_freq.get(term, 0)
                idf = math.log((max(self._total_docs, 1) + 1) / (df + 1)) + 1.0
                score += tf * idf
        # Boost for exact name match
        if " ".join(query_terms) in proof.name.lower():
            score *= 2.0
        # Boost for quality tier
        tier_boost = {
            QualityTier.UNREVIEWED: 1.0, QualityTier.COMMUNITY_REVIEWED: 1.1,
            QualityTier.EXPERT_REVIEWED: 1.25, QualityTier.FORMALLY_VERIFIED: 1.5,
        }
        score *= tier_boost.get(proof.quality_tier, 1.0)
        return score

    def _apply_filters(
        self, proofs: list[tuple[ProofMetadata, float]], query: SearchQuery,
    ) -> list[tuple[ProofMetadata, float]]:
        filtered: list[tuple[ProofMetadata, float]] = []
        for proof, score in proofs:
            if query.category and proof.category != query.category:
                continue
            if query.language and proof.language != query.language:
                continue
            if query.quality_tier and proof.quality_tier != query.quality_tier:
                continue
            if query.min_rating > 0:
                ratings = self._ratings.get(proof.id, [])
                avg = sum(ratings) / len(ratings) if ratings else 0.0
                if avg < query.min_rating:
                    continue
            if query.query and score <= 0:
                continue
            filtered.append((proof, score))
        return filtered

    def _generate_facets(self, proofs: list[ProofMetadata]) -> dict[str, dict[str, int]]:
        cats: dict[str, int] = defaultdict(int)
        langs: dict[str, int] = defaultdict(int)
        quals: dict[str, int] = defaultdict(int)
        for p in proofs:
            cats[p.category.value] += 1
            langs[p.language] += 1
            quals[p.quality_tier.value] += 1
        return {"category": dict(cats), "language": dict(langs), "quality_tier": dict(quals)}

    def _sort_results(
        self, proofs: list[tuple[ProofMetadata, float]], sort_by: SearchSortBy,
    ) -> list[tuple[ProofMetadata, float]]:
        if sort_by == SearchSortBy.RELEVANCE:
            proofs.sort(key=lambda x: x[1], reverse=True)
        elif sort_by == SearchSortBy.DOWNLOADS:
            proofs.sort(key=lambda x: self._downloads.get(x[0].id, 0), reverse=True)
        elif sort_by == SearchSortBy.RATING:
            def _avg_rating(pid: str) -> float:
                r = self._ratings.get(pid, [])
                return sum(r) / len(r) if r else 0.0
            proofs.sort(key=lambda x: _avg_rating(x[0].id), reverse=True)
        elif sort_by == SearchSortBy.RECENT:
            proofs.sort(key=lambda x: x[0].created_at, reverse=True)
        elif sort_by == SearchSortBy.QUALITY:
            _order = {QualityTier.FORMALLY_VERIFIED: 4, QualityTier.EXPERT_REVIEWED: 3,
                      QualityTier.COMMUNITY_REVIEWED: 2, QualityTier.UNREVIEWED: 1}
            proofs.sort(key=lambda x: _order.get(x[0].quality_tier, 0), reverse=True)
        return proofs

    def get_trending(self, days: int = 7, limit: int = 10) -> list[ProofMetadata]:
        if not self._storage:
            return []
        cutoff = datetime.now(UTC) - timedelta(days=days)
        recent = [p for p in self._storage._metadata.values() if p.created_at >= cutoff]
        recent.sort(key=lambda p: self._downloads.get(p.id, 0), reverse=True)
        return recent[:limit]

    def record_download(self, proof_id: str) -> None:
        self._downloads[proof_id] += 1

    def record_rating(self, proof_id: str, rating: int) -> None:
        self._ratings[proof_id].append(rating)


# =============================================================================
# Proof Quality Manager
# =============================================================================


class ProofQualityManager:
    """Manages reviews, quality metrics, and tier promotion for proofs."""

    def __init__(self) -> None:
        self._reviews: dict[str, list[ProofReview]] = defaultdict(list)
        self._issues: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._verification_timestamps: dict[str, datetime] = {}

    def submit_review(self, review: ProofReview) -> None:
        self._reviews[review.proof_id].append(review)
        logger.info("Review submitted", proof_id=review.proof_id, reviewer=review.reviewer_id)

    def get_quality_metrics(self, proof_id: str) -> ProofQualityMetrics:
        reviews = self._reviews.get(proof_id, [])
        issues = self._issues.get(proof_id, [])
        ratings = [r.rating for r in reviews]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0

        verified_count = sum(
            1 for r in reviews
            if r.quality_assessment in (QualityTier.EXPERT_REVIEWED, QualityTier.FORMALLY_VERIFIED)
        )
        success_rate = verified_count / len(reviews) if reviews else 0.0
        issue_reviews = sum(1 for r in reviews if r.issues_found)
        false_positive_rate = issue_reviews / len(reviews) if reviews else 0.0

        return ProofQualityMetrics(
            proof_id=proof_id, success_rate=success_rate,
            false_positive_rate=false_positive_rate, reuse_count=0,
            avg_rating=avg_rating, review_count=len(reviews),
            bug_reports=len(issues),
            last_verified=self._verification_timestamps.get(proof_id),
        )

    def update_quality_tier(self, proof_id: str) -> QualityTier:
        metrics = self.get_quality_metrics(proof_id)
        return self._calculate_tier(metrics)

    def flag_issue(self, proof_id: str, reporter_id: str, issue: str) -> None:
        self._issues[proof_id].append({
            "reporter_id": reporter_id, "issue": issue,
            "created_at": datetime.now(UTC).isoformat(),
        })
        logger.warning("Issue flagged", proof_id=proof_id, reporter=reporter_id)

    def _calculate_tier(self, metrics: ProofQualityMetrics) -> QualityTier:
        if (metrics.success_rate >= 0.95 and metrics.review_count >= 3
                and metrics.false_positive_rate < 0.05
                and metrics.last_verified is not None):
            return QualityTier.FORMALLY_VERIFIED
        if (metrics.avg_rating >= 4.0 and metrics.review_count >= 3
                and metrics.false_positive_rate < 0.1):
            return QualityTier.EXPERT_REVIEWED
        if metrics.review_count >= 1 and metrics.avg_rating >= 3.0:
            return QualityTier.COMMUNITY_REVIEWED
        return QualityTier.UNREVIEWED


# =============================================================================
# Gamification Engine
# =============================================================================


_POINT_VALUES: dict[ContributionType, int] = {
    ContributionType.PROOF_SUBMITTED: 25,
    ContributionType.PROOF_REVIEWED: 10,
    ContributionType.BUG_REPORTED: 15,
    ContributionType.IMPROVEMENT_SUGGESTED: 8,
    ContributionType.DOCUMENTATION_ADDED: 5,
}

_BADGE_DEFINITIONS: list[tuple[str, str, Any]] = [
    ("first_proof", "First Proof", lambda p: p.proofs_submitted >= 1),
    ("prolific_author", "Prolific Author", lambda p: p.proofs_submitted >= 10),
    ("centurion", "Centurion", lambda p: p.proofs_submitted >= 100),
    ("first_review", "First Review", lambda p: p.proofs_reviewed >= 1),
    ("trusted_reviewer", "Trusted Reviewer", lambda p: p.proofs_reviewed >= 25),
    ("rising_star", "Rising Star", lambda p: p.reputation_points >= 100),
    ("expert", "Expert", lambda p: p.reputation_points >= 1000),
    ("legend", "Legend", lambda p: p.reputation_points >= 5000),
]


class GamificationEngine:
    """Tracks contributor profiles, reputation points, badges, and leaderboard."""

    def __init__(self) -> None:
        self._profiles: dict[str, ContributorProfile] = {}

    def _ensure_profile(self, user_id: str) -> ContributorProfile:
        if user_id not in self._profiles:
            self._profiles[user_id] = ContributorProfile(
                user_id=user_id, display_name=user_id, joined_at=datetime.now(UTC),
            )
        return self._profiles[user_id]

    def award_contribution(
        self, user_id: str, contribution_type: ContributionType, details: dict | None = None,
    ) -> int:
        """Award reputation points for a contribution. Returns new total."""
        profile = self._ensure_profile(user_id)
        points = self._get_point_value(contribution_type)
        profile.reputation_points += points
        profile.contributions.append({
            "type": contribution_type.value, "points": points,
            "details": details or {}, "timestamp": datetime.now(UTC).isoformat(),
        })
        if contribution_type == ContributionType.PROOF_SUBMITTED:
            profile.proofs_submitted += 1
        elif contribution_type == ContributionType.PROOF_REVIEWED:
            profile.proofs_reviewed += 1
        profile.badges = self._calculate_badges(profile)
        logger.info(
            "Contribution awarded", user_id=user_id,
            type=contribution_type.value, points=points, total=profile.reputation_points,
        )
        return profile.reputation_points

    def get_profile(self, user_id: str) -> ContributorProfile:
        return self._ensure_profile(user_id)

    def get_leaderboard(self, top_n: int = 50) -> list[LeaderboardEntry]:
        sorted_profiles = sorted(
            self._profiles.values(), key=lambda p: p.reputation_points, reverse=True,
        )
        entries: list[LeaderboardEntry] = []
        for rank, profile in enumerate(sorted_profiles[:top_n], 1):
            profile.rank = rank
            review_contribs = [
                c for c in profile.contributions
                if c["type"] == ContributionType.PROOF_REVIEWED.value
            ]
            avg_rating = 0.0
            if review_contribs:
                ratings = [c["details"].get("rating", 0) for c in review_contribs]
                avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
            entries.append(LeaderboardEntry(
                rank=rank, user_id=profile.user_id,
                display_name=profile.display_name,
                reputation_points=profile.reputation_points,
                proofs_count=profile.proofs_submitted,
                avg_rating=avg_rating, badges=list(profile.badges),
            ))
        return entries

    def _calculate_badges(self, profile: ContributorProfile) -> list[str]:
        return [bid for bid, _, cond in _BADGE_DEFINITIONS if cond(profile)]

    def _get_point_value(self, contribution_type: ContributionType) -> int:
        return _POINT_VALUES.get(contribution_type, 0)


# =============================================================================
# Proof Marketplace V2
# =============================================================================


class ProofMarketplaceV2:
    """Enhanced community proof marketplace.

    Orchestrates storage, search, quality management, and gamification
    into a single high-level API.

    Example::

        mp = ProofMarketplaceV2()
        proof_id = mp.publish_proof(metadata, content, "alice")
        results = mp.search_proofs(SearchQuery(query="null safety"))
        mp.review_proof(proof_id, "bob", 5, "Excellent proof")
    """

    def __init__(self) -> None:
        self.storage = ProofStorage()
        self.search_engine = ProofSearchEngine()
        self.quality_manager = ProofQualityManager()
        self.gamification = GamificationEngine()
        self.search_engine.bind_storage(self.storage)
        self._imports: dict[str, set[str]] = defaultdict(set)

    def publish_proof(
        self, metadata: ProofMetadata, content: ProofContent, author_id: str,
    ) -> str:
        """Publish a proof: store, index, and award reputation."""
        metadata.author_id = author_id
        proof_id = self.storage.store_proof(metadata, content)
        self.search_engine.index_proof(metadata)
        self.gamification.award_contribution(
            author_id, ContributionType.PROOF_SUBMITTED, {"proof_id": proof_id},
        )
        logger.info("Proof published to marketplace", proof_id=proof_id, author=author_id)
        return proof_id

    def search_proofs(self, query: SearchQuery) -> SearchResult:
        return self.search_engine.search(query)

    def import_proof(self, proof_id: str, project_id: str) -> bool:
        """Import a proof into a project (track reuse)."""
        if not self.storage.get_proof(proof_id):
            logger.warning("Proof not found for import", proof_id=proof_id)
            return False
        self._imports[project_id].add(proof_id)
        self.search_engine.record_download(proof_id)
        logger.info("Proof imported", proof_id=proof_id, project_id=project_id)
        return True

    def review_proof(
        self, proof_id: str, reviewer_id: str, rating: int, comment: str,
    ) -> None:
        """Submit a review and refresh quality tier."""
        rating = max(1, min(rating, 5))
        review = ProofReview(
            id=str(uuid.uuid4()), proof_id=proof_id, reviewer_id=reviewer_id,
            rating=rating, comment=comment,
            quality_assessment=QualityTier.COMMUNITY_REVIEWED, issues_found=[],
        )
        self.quality_manager.submit_review(review)
        self.search_engine.record_rating(proof_id, rating)
        new_tier = self.quality_manager.update_quality_tier(proof_id)
        self.storage.update_proof(proof_id, {"quality_tier": new_tier})
        self.gamification.award_contribution(
            reviewer_id, ContributionType.PROOF_REVIEWED,
            {"proof_id": proof_id, "rating": rating},
        )

    def get_recommendations(
        self, project_languages: list[str], existing_proofs: list[str],
    ) -> list[ProofMetadata]:
        """Recommend proofs matching languages but not yet imported."""
        all_proofs = self.storage.list_proofs(author_id=None, limit=1000)
        existing_set = set(existing_proofs)
        candidates: list[tuple[ProofMetadata, float]] = []
        for proof in all_proofs:
            if proof.id in existing_set or proof.language not in project_languages:
                continue
            tier_score = {
                QualityTier.FORMALLY_VERIFIED: 4.0, QualityTier.EXPERT_REVIEWED: 3.0,
                QualityTier.COMMUNITY_REVIEWED: 2.0, QualityTier.UNREVIEWED: 1.0,
            }.get(proof.quality_tier, 1.0)
            downloads = self.search_engine._downloads.get(proof.id, 0)
            candidates.append((proof, tier_score + math.log(downloads + 1)))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in candidates[:20]]

    def get_community_stats(self) -> dict:
        """Return aggregate statistics about the marketplace."""
        all_proofs = self.storage.list_proofs(author_id=None, limit=100_000)
        total_downloads = sum(self.search_engine._downloads.get(p.id, 0) for p in all_proofs)
        total_reviews = sum(len(r) for r in self.quality_manager._reviews.values())
        cat_dist: dict[str, int] = defaultdict(int)
        qual_dist: dict[str, int] = defaultdict(int)
        lang_dist: dict[str, int] = defaultdict(int)
        for p in all_proofs:
            cat_dist[p.category.value] += 1
            qual_dist[p.quality_tier.value] += 1
            lang_dist[p.language] += 1
        return {
            "total_proofs": len(all_proofs), "total_downloads": total_downloads,
            "total_reviews": total_reviews,
            "total_contributors": len(self.gamification._profiles),
            "category_distribution": dict(cat_dist),
            "quality_distribution": dict(qual_dist),
            "language_distribution": dict(lang_dist),
        }
