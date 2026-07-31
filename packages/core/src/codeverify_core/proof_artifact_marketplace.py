"""Proof Artifact Marketplace.

Community marketplace for anonymized proof templates, verification rules,
and Z3 patterns with voting, search, and automatic proof reuse.

Features:
- Proof artifact submission with metadata and categorization
- Privacy-preserving anonymization pipeline
- Search by category, language, pattern type
- Community voting and quality scoring
- Automatic proof reuse via content matching
- Proof template generation from verification results
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ProofCategory(str, Enum):
    """Categories of proof artifacts."""

    NULL_SAFETY = "null_safety"
    BOUNDS_CHECK = "bounds_check"
    OVERFLOW = "overflow"
    DIVISION_ZERO = "division_zero"
    MEMORY_SAFETY = "memory_safety"
    CONCURRENCY = "concurrency"
    TYPE_SAFETY = "type_safety"
    SECURITY = "security"
    RESOURCE_MANAGEMENT = "resource_management"
    CUSTOM = "custom"


class ProofLanguage(str, Enum):
    """Languages for proof artifacts."""

    PYTHON = "python"
    TYPESCRIPT = "typescript"
    GO = "go"
    JAVA = "java"
    RUST = "rust"
    C = "c"
    CPP = "cpp"
    UNIVERSAL = "universal"


class ArtifactStatus(str, Enum):
    """Status of a proof artifact."""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"


@dataclass
class ProofArtifact:
    """A proof artifact in the marketplace."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    category: ProofCategory = ProofCategory.NULL_SAFETY
    language: ProofLanguage = ProofLanguage.UNIVERSAL
    z3_constraints: str = ""
    pattern_code: str = ""
    fix_template: str = ""
    tags: list[str] = field(default_factory=list)
    status: ArtifactStatus = ArtifactStatus.DRAFT
    author_id: str = ""
    is_anonymized: bool = False
    content_hash: str = ""
    upvotes: int = 0
    downvotes: int = 0
    download_count: int = 0
    reuse_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def quality_score(self) -> float:
        vote_score = self.upvotes - self.downvotes
        usage_score = self.download_count * 0.1 + self.reuse_count * 0.5
        return round(vote_score + usage_score, 2)

    def compute_hash(self) -> str:
        content = f"{self.z3_constraints}:{self.pattern_code}:{self.category.value}"
        self.content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self.content_hash


@dataclass
class ProofTemplate:
    """A reusable proof template generated from artifacts."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    artifact_id: str = ""
    name: str = ""
    parameters: list[str] = field(default_factory=list)
    z3_template: str = ""
    description: str = ""
    usage_example: str = ""


@dataclass
class SearchResult:
    """Search result from the marketplace."""

    artifact: ProofArtifact | None = None
    relevance_score: float = 0.0
    match_type: str = ""  # exact, category, tag, content


@dataclass
class MarketplaceStats:
    """Aggregate marketplace statistics."""

    total_artifacts: int = 0
    published_artifacts: int = 0
    total_downloads: int = 0
    total_reuses: int = 0
    category_distribution: dict[str, int] = field(default_factory=dict)
    language_distribution: dict[str, int] = field(default_factory=dict)
    top_contributors: list[dict[str, Any]] = field(default_factory=list)


class ProofAnonymizer:
    """Anonymizes proof artifacts to remove org-specific identifiers."""

    PATTERNS_TO_STRIP: list[tuple[str, str | Callable[[re.Match[str]], str]]] = [
        (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "<email>"),
        (r'https?://[^\s"\']+', "<url>"),
        (r'/(?:home|Users)/\w+/[^\s"\']*', "<path>"),
        (r"[A-Z][a-z]+(?:[A-Z][a-z]+)+", lambda m: f"Identifier_{hash(m.group()) % 1000}"),
        (r'(?:api[_-]?key|token|secret|password)\s*[=:]\s*["\'][^"\']+["\']', "<redacted>"),
    ]

    def anonymize(self, artifact: ProofArtifact) -> ProofArtifact:
        """Anonymize an artifact, stripping identifiable information."""
        artifact.z3_constraints = self._strip_identifiers(artifact.z3_constraints)
        artifact.pattern_code = self._strip_identifiers(artifact.pattern_code)
        artifact.fix_template = self._strip_identifiers(artifact.fix_template)
        artifact.author_id = hashlib.sha256(artifact.author_id.encode()).hexdigest()[:8]
        artifact.is_anonymized = True
        return artifact

    def _strip_identifiers(self, text: str) -> str:
        result = text
        for pattern, replacement in self.PATTERNS_TO_STRIP:
            if callable(replacement):
                result = re.sub(pattern, replacement, result)
            else:
                result = re.sub(pattern, replacement, result)
        return result


class ProofMatcher:
    """Matches incoming verification tasks to existing proofs."""

    def find_matches(
        self,
        code_pattern: str,
        category: ProofCategory,
        language: ProofLanguage,
        artifacts: list[ProofArtifact],
    ) -> list[SearchResult]:
        """Find matching proof artifacts for a verification task."""
        results: list[SearchResult] = []

        for artifact in artifacts:
            if artifact.status != ArtifactStatus.PUBLISHED:
                continue

            score = 0.0
            match_type = "none"

            if artifact.category == category:
                score += 0.5
                match_type = "category"

            if artifact.language in (language, ProofLanguage.UNIVERSAL):
                score += 0.2

            if code_pattern and artifact.pattern_code:
                overlap = self._compute_overlap(code_pattern, artifact.pattern_code)
                score += overlap * 0.3
                if overlap > 0.7:
                    match_type = "exact"

            if score > 0.3:
                results.append(
                    SearchResult(
                        artifact=artifact,
                        relevance_score=round(score, 3),
                        match_type=match_type,
                    )
                )

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results

    def _compute_overlap(self, a: str, b: str) -> float:
        tokens_a = set(a.split())
        tokens_b = set(b.split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)


class ProofArtifactMarketplaceService:
    """Main service for the proof artifact marketplace."""

    def __init__(self) -> None:
        self._artifacts: dict[str, ProofArtifact] = {}
        self._templates: dict[str, ProofTemplate] = {}
        self._anonymizer = ProofAnonymizer()
        self._matcher = ProofMatcher()
        self._votes: dict[str, dict[str, int]] = defaultdict(dict)  # artifact_id -> user_id -> vote

    def submit_artifact(
        self,
        title: str,
        description: str,
        category: ProofCategory,
        language: ProofLanguage,
        z3_constraints: str,
        pattern_code: str = "",
        fix_template: str = "",
        author_id: str = "",
        tags: list[str] | None = None,
        anonymize: bool = True,
    ) -> ProofArtifact:
        """Submit a new proof artifact."""
        artifact = ProofArtifact(
            title=title,
            description=description,
            category=category,
            language=language,
            z3_constraints=z3_constraints,
            pattern_code=pattern_code,
            fix_template=fix_template,
            author_id=author_id,
            tags=tags or [],
            status=ArtifactStatus.SUBMITTED,
        )
        artifact.compute_hash()

        if anonymize:
            artifact = self._anonymizer.anonymize(artifact)

        self._artifacts[artifact.id] = artifact
        logger.info("artifact_submitted", artifact_id=artifact.id, category=category.value)
        return artifact

    def publish_artifact(self, artifact_id: str) -> bool:
        """Publish an artifact (after review)."""
        artifact = self._artifacts.get(artifact_id)
        if not artifact:
            return False
        artifact.status = ArtifactStatus.PUBLISHED
        return True

    def deprecate_artifact(self, artifact_id: str) -> bool:
        artifact = self._artifacts.get(artifact_id)
        if not artifact:
            return False
        artifact.status = ArtifactStatus.DEPRECATED
        return True

    def vote(self, artifact_id: str, user_id: str, upvote: bool) -> bool:
        """Vote on an artifact. Returns False if already voted."""
        artifact = self._artifacts.get(artifact_id)
        if not artifact:
            return False

        existing = self._votes[artifact_id].get(user_id)
        if existing is not None:
            # Change vote
            if existing == 1:
                artifact.upvotes -= 1
            else:
                artifact.downvotes -= 1

        vote_val = 1 if upvote else -1
        self._votes[artifact_id][user_id] = vote_val
        if upvote:
            artifact.upvotes += 1
        else:
            artifact.downvotes += 1
        return True

    def download(self, artifact_id: str) -> ProofArtifact | None:
        """Download (retrieve) an artifact, incrementing download count."""
        artifact = self._artifacts.get(artifact_id)
        if artifact and artifact.status == ArtifactStatus.PUBLISHED:
            artifact.download_count += 1
            return artifact
        return None

    def search(
        self,
        query: str = "",
        category: ProofCategory | None = None,
        language: ProofLanguage | None = None,
        tags: list[str] | None = None,
        min_quality: float = 0.0,
    ) -> list[ProofArtifact]:
        """Search for artifacts."""
        results: list[ProofArtifact] = []
        for artifact in self._artifacts.values():
            if artifact.status != ArtifactStatus.PUBLISHED:
                continue
            if category and artifact.category != category:
                continue
            if language and artifact.language not in (language, ProofLanguage.UNIVERSAL):
                continue
            if tags and not any(t in artifact.tags for t in tags):
                continue
            if artifact.quality_score < min_quality:
                continue
            if query:
                q_lower = query.lower()
                searchable = (
                    f"{artifact.title} {artifact.description} {' '.join(artifact.tags)}".lower()
                )
                if q_lower not in searchable:
                    continue
            results.append(artifact)

        results.sort(key=lambda a: a.quality_score, reverse=True)
        return results

    def find_reusable_proof(
        self,
        code_pattern: str,
        category: ProofCategory,
        language: ProofLanguage,
    ) -> list[SearchResult]:
        """Find reusable proofs for automatic application."""
        published = [a for a in self._artifacts.values() if a.status == ArtifactStatus.PUBLISHED]
        matches = self._matcher.find_matches(code_pattern, category, language, published)

        for match in matches:
            if match.artifact:
                match.artifact.reuse_count += 1

        return matches

    def create_template(
        self,
        artifact_id: str,
        name: str,
        parameters: list[str],
        z3_template: str,
        description: str = "",
    ) -> ProofTemplate | None:
        """Create a reusable template from an artifact."""
        if artifact_id not in self._artifacts:
            return None
        template = ProofTemplate(
            artifact_id=artifact_id,
            name=name,
            parameters=parameters,
            z3_template=z3_template,
            description=description,
        )
        self._templates[template.id] = template
        return template

    def get_stats(self) -> MarketplaceStats:
        """Get marketplace statistics."""
        artifacts = list(self._artifacts.values())
        published = [a for a in artifacts if a.status == ArtifactStatus.PUBLISHED]

        cat_dist: dict[str, int] = defaultdict(int)
        lang_dist: dict[str, int] = defaultdict(int)
        author_downloads: dict[str, int] = defaultdict(int)

        for a in published:
            cat_dist[a.category.value] += 1
            lang_dist[a.language.value] += 1
            author_downloads[a.author_id] += a.download_count

        contributor_stats: list[dict[str, Any]] = [
            {"author_id": k, "downloads": v} for k, v in author_downloads.items()
        ]
        top_contributors = sorted(
            contributor_stats,
            key=lambda x: x["downloads"],
            reverse=True,
        )[:10]

        return MarketplaceStats(
            total_artifacts=len(artifacts),
            published_artifacts=len(published),
            total_downloads=sum(a.download_count for a in published),
            total_reuses=sum(a.reuse_count for a in published),
            category_distribution=dict(cat_dist),
            language_distribution=dict(lang_dist),
            top_contributors=top_contributors,
        )


# ─── Singleton Access ──────────────────────────────────────────────────


_proof_marketplace_instance: ProofArtifactMarketplaceService | None = None


def get_proof_marketplace_service() -> ProofArtifactMarketplaceService:
    """Get or create the singleton ProofArtifactMarketplaceService."""
    global _proof_marketplace_instance
    if _proof_marketplace_instance is None:
        _proof_marketplace_instance = ProofArtifactMarketplaceService()
    return _proof_marketplace_instance


def reset_proof_marketplace_service() -> None:
    """Reset the singleton (for testing)."""
    global _proof_marketplace_instance
    _proof_marketplace_instance = None
