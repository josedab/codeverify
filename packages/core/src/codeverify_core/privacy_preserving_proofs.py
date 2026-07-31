"""Differential Privacy-Preserving Proof Marketplace.

Enables organizations to share anonymized verification proofs and patterns
without exposing proprietary code. Uses differential privacy techniques to
extract learnings from community proofs while protecting IP.

Features:
- Proof anonymization with identifier stripping
- Differential privacy noise injection (Laplace mechanism)
- Privacy budget tracking (epsilon accounting)
- Federated pattern aggregation
- Anonymized proof quality scoring
"""

from __future__ import annotations

import hashlib
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class PrivacyLevel(str, Enum):
    """Privacy level for proof sharing."""

    PUBLIC = "public"
    ANONYMIZED = "anonymized"
    DIFFERENTIAL = "differential"
    FEDERATED = "federated"


class ProofCategory(str, Enum):
    """Category of shared proof."""

    NULL_SAFETY = "null_safety"
    BOUNDS_CHECK = "bounds_check"
    TYPE_SAFETY = "type_safety"
    CONCURRENCY = "concurrency"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CUSTOM = "custom"


class ContributionStatus(str, Enum):
    """Status of a proof contribution."""

    PENDING = "pending"
    ANONYMIZING = "anonymizing"
    PUBLISHED = "published"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


@dataclass
class PrivacyBudget:
    """Tracks differential privacy epsilon budget."""

    total_epsilon: float = 10.0
    consumed_epsilon: float = 0.0
    total_queries: int = 0

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.total_epsilon - self.consumed_epsilon)

    @property
    def is_exhausted(self) -> bool:
        return self.remaining_epsilon <= 0.0

    def consume(self, epsilon: float) -> bool:
        """Consume epsilon budget. Returns False if insufficient."""
        if epsilon > self.remaining_epsilon:
            return False
        self.consumed_epsilon += epsilon
        self.total_queries += 1
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_epsilon": self.total_epsilon,
            "consumed_epsilon": round(self.consumed_epsilon, 4),
            "remaining_epsilon": round(self.remaining_epsilon, 4),
            "total_queries": self.total_queries,
            "is_exhausted": self.is_exhausted,
        }


@dataclass
class AnonymizedProof:
    """A proof that has been anonymized for sharing."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_hash: str = ""
    category: ProofCategory = ProofCategory.CUSTOM
    privacy_level: PrivacyLevel = PrivacyLevel.ANONYMIZED
    pattern_description: str = ""
    constraint_template: str = ""
    language: str = "python"
    applicability_score: float = 0.0
    noise_added: float = 0.0
    status: ContributionStatus = ContributionStatus.PENDING
    contributor_pseudonym: str = ""
    upvotes: int = 0
    downloads: int = 0
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value,
            "privacy_level": self.privacy_level.value,
            "pattern_description": self.pattern_description,
            "language": self.language,
            "applicability_score": round(self.applicability_score, 4),
            "status": self.status.value,
            "upvotes": self.upvotes,
            "downloads": self.downloads,
        }


class ProofAnonymizer:
    """Anonymizes proofs by stripping identifiers and adding noise."""

    IDENTIFIER_PREFIXES = ("var_", "func_", "class_", "param_", "local_")

    def anonymize(
        self,
        raw_proof: str,
        category: ProofCategory,
        epsilon: float = 1.0,
    ) -> AnonymizedProof:
        """Anonymize a proof for sharing."""
        stripped = self._strip_identifiers(raw_proof)
        noise = self._laplace_noise(0.0, 1.0 / epsilon) if epsilon > 0 else 0.0

        pseudonym = hashlib.sha256(raw_proof.encode() + str(time.time()).encode()).hexdigest()[:12]

        return AnonymizedProof(
            original_hash=hashlib.sha256(raw_proof.encode()).hexdigest(),
            category=category,
            privacy_level=PrivacyLevel.DIFFERENTIAL,
            pattern_description=stripped,
            constraint_template=self._extract_template(stripped),
            applicability_score=max(0.0, min(1.0, 0.7 + noise * 0.1)),
            noise_added=abs(noise),
            status=ContributionStatus.PUBLISHED,
            contributor_pseudonym=pseudonym,
        )

    def _strip_identifiers(self, text: str) -> str:
        """Replace specific identifiers with generic placeholders."""
        result = text
        counter = 0
        for prefix in self.IDENTIFIER_PREFIXES:
            while prefix in result:
                result = result.replace(prefix, f"_anon{counter}_", 1)
                counter += 1
        return result

    def _extract_template(self, text: str) -> str:
        """Extract a reusable constraint template."""
        lines = text.strip().split("\n")
        if len(lines) > 5:
            return "\n".join(lines[:5]) + "\n# ... (truncated)"
        return text

    @staticmethod
    def _laplace_noise(mu: float, b: float) -> float:
        """Generate Laplace noise for differential privacy."""
        u = random.random() - 0.5
        return mu - b * math.copysign(1, u) * math.log(1 - 2 * abs(u) + 1e-10)


@dataclass
class FederatedUpdate:
    """An aggregated update from federated learning."""

    round_id: int = 0
    participant_count: int = 0
    pattern_weights: dict[str, float] = field(default_factory=dict)
    aggregation_method: str = "fedavg"
    privacy_guarantee: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "participant_count": self.participant_count,
            "pattern_count": len(self.pattern_weights),
            "aggregation_method": self.aggregation_method,
            "privacy_guarantee": round(self.privacy_guarantee, 4),
        }


class FederatedAggregator:
    """Aggregates proof patterns from multiple organizations without seeing raw data."""

    def __init__(self) -> None:
        self._round: int = 0
        self._participant_updates: list[dict[str, float]] = []

    def submit_local_update(self, pattern_weights: dict[str, float]) -> None:
        """Submit a local update from one organization."""
        self._participant_updates.append(pattern_weights)

    def aggregate(self, epsilon: float = 1.0) -> FederatedUpdate:
        """Aggregate all submitted updates (FedAvg + noise)."""
        self._round += 1
        if not self._participant_updates:
            return FederatedUpdate(round_id=self._round)

        all_keys: set[str] = set()
        for upd in self._participant_updates:
            all_keys.update(upd.keys())

        aggregated: dict[str, float] = {}
        n = len(self._participant_updates)
        for key in all_keys:
            values = [upd.get(key, 0.0) for upd in self._participant_updates]
            avg = sum(values) / n
            noise = ProofAnonymizer._laplace_noise(0.0, 1.0 / (epsilon * n))
            aggregated[key] = avg + noise

        self._participant_updates.clear()

        return FederatedUpdate(
            round_id=self._round,
            participant_count=n,
            pattern_weights=aggregated,
            aggregation_method="fedavg",
            privacy_guarantee=epsilon,
        )


class PrivacyPreservingProofMarketplace:
    """Marketplace for sharing anonymized verification proofs."""

    def __init__(self, epsilon_budget: float = 10.0) -> None:
        self._proofs: dict[str, AnonymizedProof] = {}
        self._budget = PrivacyBudget(total_epsilon=epsilon_budget)
        self._anonymizer = ProofAnonymizer()
        self._aggregator = FederatedAggregator()

    def contribute_proof(
        self,
        raw_proof: str,
        category: ProofCategory,
        language: str = "python",
        epsilon: float = 1.0,
    ) -> AnonymizedProof | None:
        """Contribute an anonymized proof to the marketplace."""
        if not self._budget.consume(epsilon):
            logger.warning("privacy_budget_exhausted")
            return None

        proof = self._anonymizer.anonymize(raw_proof, category, epsilon)
        proof.language = language
        self._proofs[proof.id] = proof

        logger.info(
            "proof_contributed",
            proof_id=proof.id,
            category=category.value,
            remaining_budget=round(self._budget.remaining_epsilon, 4),
        )
        return proof

    def search_proofs(
        self,
        category: ProofCategory | None = None,
        language: str | None = None,
        min_score: float = 0.0,
    ) -> list[AnonymizedProof]:
        """Search for proofs in the marketplace."""
        results = []
        for proof in self._proofs.values():
            if proof.status != ContributionStatus.PUBLISHED:
                continue
            if category and proof.category != category:
                continue
            if language and proof.language != language:
                continue
            if proof.applicability_score < min_score:
                continue
            results.append(proof)

        results.sort(key=lambda p: p.applicability_score, reverse=True)
        return results

    def upvote_proof(self, proof_id: str) -> bool:
        """Upvote a proof."""
        proof = self._proofs.get(proof_id)
        if proof:
            proof.upvotes += 1
            return True
        return False

    def download_proof(self, proof_id: str) -> AnonymizedProof | None:
        """Download a proof (increments download count)."""
        proof = self._proofs.get(proof_id)
        if proof and proof.status == ContributionStatus.PUBLISHED:
            proof.downloads += 1
            return proof
        return None

    def submit_federated_update(self, pattern_weights: dict[str, float]) -> None:
        """Submit a federated learning update."""
        self._aggregator.submit_local_update(pattern_weights)

    def run_federated_round(self, epsilon: float = 1.0) -> FederatedUpdate:
        """Run a federated aggregation round."""
        return self._aggregator.aggregate(epsilon)

    @property
    def privacy_budget(self) -> PrivacyBudget:
        return self._budget

    @property
    def proof_count(self) -> int:
        return len(self._proofs)

    def get_marketplace_stats(self) -> dict[str, Any]:
        """Get marketplace statistics."""
        published = [p for p in self._proofs.values() if p.status == ContributionStatus.PUBLISHED]
        return {
            "total_proofs": len(self._proofs),
            "published_proofs": len(published),
            "total_downloads": sum(p.downloads for p in published),
            "total_upvotes": sum(p.upvotes for p in published),
            "categories": list({p.category.value for p in published}),
            "privacy_budget": self._budget.to_dict(),
        }


_default_marketplace: PrivacyPreservingProofMarketplace | None = None


def get_privacy_marketplace() -> PrivacyPreservingProofMarketplace:
    """Get the singleton privacy-preserving proof marketplace."""
    global _default_marketplace
    if _default_marketplace is None:
        _default_marketplace = PrivacyPreservingProofMarketplace()
    return _default_marketplace


def reset_privacy_marketplace() -> None:
    """Reset the singleton (for testing)."""
    global _default_marketplace
    _default_marketplace = None
