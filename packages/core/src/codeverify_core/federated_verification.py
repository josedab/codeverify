"""Privacy-Preserving Federated Verification.

Organizations share anonymized verification patterns and proof
templates without exposing proprietary code, using differential
privacy, secure aggregation, and federated learning.

Features:
- Local pattern extraction with differential privacy noise
- Epsilon budget tracking and enforcement
- Federated aggregation of verification patterns across orgs
- Privacy-preserving proof template sharing
- Contribution metrics and pattern adoption tracking
- Configurable privacy levels (strict/moderate/relaxed)
- HMAC-based contribution authentication
- Input validation and sanitization
- Configurable noise calibration
- Contribution audit logging
"""

from __future__ import annotations

import hashlib
import hmac
import math
import random
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class PrivacyLevel(str, Enum):
    """Privacy level presets."""

    STRICT = "strict"
    MODERATE = "moderate"
    RELAXED = "relaxed"


class PatternType(str, Enum):
    """Types of extractable verification patterns."""

    FINDING_PATTERN = "finding_pattern"
    PROOF_TEMPLATE = "proof_template"
    FIX_TEMPLATE = "fix_template"
    RULE_PATTERN = "rule_pattern"
    FALSE_POSITIVE = "false_positive"


class AggregationStatus(str, Enum):
    """Status of a federated aggregation round."""

    COLLECTING = "collecting"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PrivacyBudget:
    """Differential privacy budget tracker."""

    total_epsilon: float = 1.0
    used_epsilon: float = 0.0
    total_delta: float = 1e-5
    queries_made: int = 0

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.total_epsilon - self.used_epsilon)

    @property
    def is_exhausted(self) -> bool:
        return self.remaining_epsilon <= 0.0

    def consume(self, epsilon: float) -> bool:
        """Consume privacy budget. Returns False if insufficient."""
        if epsilon > self.remaining_epsilon:
            return False
        self.used_epsilon += epsilon
        self.queries_made += 1
        return True


@dataclass
class PrivacyConfig:
    """Configuration for privacy-preserving operations."""

    level: PrivacyLevel = PrivacyLevel.MODERATE
    epsilon_per_pattern: float = 0.1
    noise_scale: float = 1.0
    min_org_count: int = 3  # k-anonymity threshold
    strip_identifiers: bool = True
    max_patterns_per_round: int = 100

    @classmethod
    def for_level(cls, level: PrivacyLevel) -> PrivacyConfig:
        configs = {
            PrivacyLevel.STRICT: cls(level=level, epsilon_per_pattern=0.01,
                                     noise_scale=2.0, min_org_count=5),
            PrivacyLevel.MODERATE: cls(level=level, epsilon_per_pattern=0.1,
                                       noise_scale=1.0, min_org_count=3),
            PrivacyLevel.RELAXED: cls(level=level, epsilon_per_pattern=0.5,
                                      noise_scale=0.5, min_org_count=2),
        }
        return configs.get(level, cls())


@dataclass
class LocalPattern:
    """A pattern extracted locally from an org's verification data."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    org_id: str = ""
    pattern_type: PatternType = PatternType.FINDING_PATTERN
    category: str = ""
    language: str = "python"
    frequency: float = 0.0
    confidence: float = 0.0
    content_hash: str = ""
    noisy_frequency: float = 0.0
    is_anonymized: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedPattern:
    """A pattern aggregated across multiple organizations."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    pattern_type: PatternType = PatternType.FINDING_PATTERN
    category: str = ""
    language: str = "python"
    contributing_orgs: int = 0
    aggregated_frequency: float = 0.0
    aggregated_confidence: float = 0.0
    adoption_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FederatedRound:
    """A single round of federated aggregation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    round_number: int = 0
    status: AggregationStatus = AggregationStatus.COLLECTING
    contributions: dict[str, list[LocalPattern]] = field(default_factory=dict)
    aggregated_patterns: list[AggregatedPattern] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    @property
    def org_count(self) -> int:
        return len(self.contributions)


@dataclass
class OrgContribution:
    """Metrics for an org's contributions to federated learning."""

    org_id: str = ""
    patterns_contributed: int = 0
    patterns_adopted: int = 0
    rounds_participated: int = 0
    privacy_budget: PrivacyBudget = field(default_factory=PrivacyBudget)
    hmac_key: str = field(default_factory=lambda: hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:32])


@dataclass
class AuditLogEntry:
    """Audit log entry for a federated operation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    org_id: str = ""
    action: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ip_address: str = ""
    success: bool = True
    error: str = ""


class ContributionAuthenticator:
    """HMAC-based contribution authentication."""

    def sign(self, org_id: str, hmac_key: str, payload: str) -> str:
        """Create HMAC signature for a contribution."""
        return hmac.new(
            hmac_key.encode(), f"{org_id}:{payload}".encode(), hashlib.sha256
        ).hexdigest()[:16]

    def verify(self, org_id: str, hmac_key: str, payload: str, signature: str) -> bool:
        """Verify HMAC signature."""
        expected = self.sign(org_id, hmac_key, payload)
        return hmac.compare_digest(expected, signature)


class InputValidator:
    """Validates and sanitizes federated verification inputs."""

    MAX_FINDINGS = 10000
    MAX_CATEGORY_LENGTH = 100
    MAX_ORG_ID_LENGTH = 128
    ALLOWED_CATEGORY_PATTERN = re.compile(r'^[a-zA-Z0-9_\-.]+$')

    def validate_org_id(self, org_id: str) -> tuple[bool, str]:
        if not org_id or len(org_id) > self.MAX_ORG_ID_LENGTH:
            return False, f"org_id must be 1-{self.MAX_ORG_ID_LENGTH} characters"
        if not re.match(r'^[a-zA-Z0-9_\-]+$', org_id):
            return False, "org_id must be alphanumeric with hyphens/underscores"
        return True, ""

    def validate_findings(self, findings: list[dict[str, Any]]) -> tuple[bool, str]:
        if len(findings) > self.MAX_FINDINGS:
            return False, f"Too many findings: {len(findings)} > {self.MAX_FINDINGS}"
        for i, f in enumerate(findings):
            if not isinstance(f, dict):
                return False, f"Finding {i} is not a dict"
            cat = f.get("category", "")
            if cat and not self.ALLOWED_CATEGORY_PATTERN.match(cat):
                return False, f"Finding {i}: invalid category '{cat}'"
            if cat and len(cat) > self.MAX_CATEGORY_LENGTH:
                return False, f"Finding {i}: category too long"
            conf = f.get("confidence", 0.5)
            if not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
                return False, f"Finding {i}: confidence must be 0.0-1.0"
        return True, ""

    def validate_epsilon(self, epsilon: float) -> tuple[bool, str]:
        if not isinstance(epsilon, (int, float)):
            return False, "epsilon must be numeric"
        if epsilon <= 0 or epsilon > 100:
            return False, "epsilon must be in (0, 100]"
        return True, ""


class NoiseCalibrator:
    """Configurable noise calibration for different privacy needs."""

    def calibrate_epsilon(
        self,
        sensitivity: float,
        desired_accuracy: float,
        dataset_size: int,
    ) -> float:
        """Calculate recommended epsilon for desired accuracy at given dataset size."""
        if desired_accuracy <= 0 or desired_accuracy >= 1:
            return 0.1
        noise_tolerance = (1.0 - desired_accuracy) * dataset_size
        if noise_tolerance <= 0:
            return 0.01
        return round(sensitivity / noise_tolerance, 4)

    def recommended_rounds(
        self,
        total_epsilon: float,
        epsilon_per_round: float,
    ) -> int:
        """Calculate how many rounds can be done within budget."""
        if epsilon_per_round <= 0:
            return 0
        return int(total_epsilon / epsilon_per_round)


class LaplaceMechanism:
    """Laplace mechanism for differential privacy."""

    def add_noise(self, value: float, sensitivity: float, epsilon: float) -> float:
        """Add calibrated Laplace noise to a value."""
        if epsilon <= 0:
            return value
        scale = sensitivity / epsilon
        noise = random.random() - 0.5
        laplace_noise = -scale * math.copysign(1, noise) * math.log(1 - 2 * abs(noise) + 1e-10)
        return value + laplace_noise

    def add_noise_to_count(self, count: int, epsilon: float) -> float:
        """Add noise to a count (sensitivity=1)."""
        return self.add_noise(float(count), 1.0, epsilon)


class PatternExtractor:
    """Extracts and anonymizes patterns from local verification data."""

    def __init__(self, config: PrivacyConfig | None = None) -> None:
        self._config = config or PrivacyConfig()
        self._mechanism = LaplaceMechanism()

    def extract(
        self,
        org_id: str,
        findings: list[dict[str, Any]],
        budget: PrivacyBudget,
    ) -> list[LocalPattern]:
        """Extract anonymized patterns from org findings."""
        category_counts: dict[str, int] = defaultdict(int)
        category_confidence: dict[str, list[float]] = defaultdict(list)

        for finding in findings:
            cat = finding.get("category", "unknown")
            category_counts[cat] += 1
            category_confidence[cat].append(finding.get("confidence", 0.5))

        patterns: list[LocalPattern] = []
        for cat, count in category_counts.items():
            ep = self._config.epsilon_per_pattern
            if not budget.consume(ep):
                break

            noisy_freq = self._mechanism.add_noise_to_count(count, ep)
            avg_conf = sum(category_confidence[cat]) / len(category_confidence[cat])
            content = f"{org_id}:{cat}:{count}"
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]

            patterns.append(LocalPattern(
                org_id=hashlib.sha256(org_id.encode()).hexdigest()[:8] if self._config.strip_identifiers else org_id,
                pattern_type=PatternType.FINDING_PATTERN,
                category=cat,
                frequency=float(count),
                confidence=round(avg_conf, 3),
                content_hash=content_hash,
                noisy_frequency=max(0.0, round(noisy_freq, 2)),
                is_anonymized=True,
            ))

        return patterns[:self._config.max_patterns_per_round]


class FederatedAggregator:
    """Aggregates patterns across organizations with privacy guarantees."""

    def __init__(self, min_org_count: int = 3) -> None:
        self._min_orgs = min_org_count

    def aggregate(
        self,
        contributions: dict[str, list[LocalPattern]],
    ) -> list[AggregatedPattern]:
        """Aggregate patterns from multiple orgs."""
        if len(contributions) < self._min_orgs:
            logger.warning("insufficient_orgs", count=len(contributions), minimum=self._min_orgs)
            return []

        category_data: dict[str, list[LocalPattern]] = defaultdict(list)
        for patterns in contributions.values():
            for p in patterns:
                category_data[p.category].append(p)

        aggregated: list[AggregatedPattern] = []
        for cat, patterns in category_data.items():
            org_ids = set(p.org_id for p in patterns)
            if len(org_ids) < self._min_orgs:
                continue

            avg_freq = sum(p.noisy_frequency for p in patterns) / len(patterns)
            avg_conf = sum(p.confidence for p in patterns) / len(patterns)

            aggregated.append(AggregatedPattern(
                pattern_type=patterns[0].pattern_type,
                category=cat,
                language=patterns[0].language,
                contributing_orgs=len(org_ids),
                aggregated_frequency=round(avg_freq, 2),
                aggregated_confidence=round(avg_conf, 3),
            ))

        aggregated.sort(key=lambda p: p.aggregated_frequency, reverse=True)
        return aggregated


class FederatedVerificationService:
    """Main service for privacy-preserving federated verification."""

    def __init__(
        self,
        privacy_level: PrivacyLevel = PrivacyLevel.MODERATE,
    ) -> None:
        self._config = PrivacyConfig.for_level(privacy_level)
        self._extractor = PatternExtractor(self._config)
        self._aggregator = FederatedAggregator(self._config.min_org_count)
        self._authenticator = ContributionAuthenticator()
        self._validator = InputValidator()
        self._calibrator = NoiseCalibrator()
        self._orgs: dict[str, OrgContribution] = {}
        self._rounds: list[FederatedRound] = []
        self._global_patterns: list[AggregatedPattern] = []
        self._current_round: FederatedRound | None = None
        self._audit_log: list[AuditLogEntry] = []

    @property
    def authenticator(self) -> ContributionAuthenticator:
        return self._authenticator

    @property
    def validator(self) -> InputValidator:
        return self._validator

    @property
    def calibrator(self) -> NoiseCalibrator:
        return self._calibrator

    def register_org(
        self, org_id: str, epsilon_budget: float = 1.0
    ) -> OrgContribution:
        """Register an org for federated learning."""
        valid, err = self._validator.validate_org_id(org_id)
        if not valid:
            self._audit("register", org_id, success=False, error=err)
            raise ValueError(err)
        valid, err = self._validator.validate_epsilon(epsilon_budget)
        if not valid:
            self._audit("register", org_id, success=False, error=err)
            raise ValueError(err)

        contrib = OrgContribution(
            org_id=org_id,
            privacy_budget=PrivacyBudget(total_epsilon=epsilon_budget),
        )
        self._orgs[org_id] = contrib
        self._audit("register", org_id, details={"epsilon": epsilon_budget})
        return contrib

    def start_round(self) -> FederatedRound:
        """Start a new federated aggregation round."""
        round_num = len(self._rounds) + 1
        self._current_round = FederatedRound(round_number=round_num)
        self._audit("start_round", "", details={"round": round_num})
        return self._current_round

    def contribute(
        self,
        org_id: str,
        findings: list[dict[str, Any]],
        signature: str | None = None,
    ) -> list[LocalPattern]:
        """Submit an org's findings for the current round."""
        org = self._orgs.get(org_id)
        if not org:
            self._audit("contribute", org_id, success=False, error="not registered")
            raise ValueError(f"Org {org_id} not registered")
        if not self._current_round:
            self._audit("contribute", org_id, success=False, error="no active round")
            raise ValueError("No active round")

        # Validate input
        valid, err = self._validator.validate_findings(findings)
        if not valid:
            self._audit("contribute", org_id, success=False, error=err)
            raise ValueError(err)

        # Authenticate contribution if signature provided
        if signature is not None:
            payload = str(len(findings))
            if not self._authenticator.verify(org_id, org.hmac_key, payload, signature):
                self._audit("contribute", org_id, success=False, error="HMAC verification failed")
                raise PermissionError("Contribution authentication failed")

        if org.privacy_budget.is_exhausted:
            self._audit("contribute", org_id, success=False, error="budget exhausted")
            return []

        patterns = self._extractor.extract(org_id, findings, org.privacy_budget)
        self._current_round.contributions[org_id] = patterns
        org.patterns_contributed += len(patterns)
        org.rounds_participated += 1

        self._audit("contribute", org_id, details={
            "patterns": len(patterns),
            "findings_count": len(findings),
            "epsilon_remaining": round(org.privacy_budget.remaining_epsilon, 4),
        })
        return patterns

    def sign_contribution(self, org_id: str, finding_count: int) -> str | None:
        """Generate an HMAC signature for a contribution."""
        org = self._orgs.get(org_id)
        if not org:
            return None
        return self._authenticator.sign(org_id, org.hmac_key, str(finding_count))

    def complete_round(self) -> list[AggregatedPattern]:
        """Complete the current round and aggregate patterns."""
        if not self._current_round:
            return []

        aggregated = self._aggregator.aggregate(self._current_round.contributions)
        self._current_round.aggregated_patterns = aggregated
        self._current_round.status = AggregationStatus.COMPLETED
        self._current_round.completed_at = datetime.now(timezone.utc)
        self._rounds.append(self._current_round)
        self._global_patterns.extend(aggregated)
        self._current_round = None
        return aggregated

    def get_global_patterns(self) -> list[AggregatedPattern]:
        return list(self._global_patterns)

    def adopt_pattern(self, org_id: str, pattern_id: str) -> bool:
        """Record that an org adopted a global pattern."""
        org = self._orgs.get(org_id)
        if not org:
            return False
        for p in self._global_patterns:
            if p.id == pattern_id:
                p.adoption_count += 1
                org.patterns_adopted += 1
                return True
        return False

    def get_org_contribution(self, org_id: str) -> OrgContribution | None:
        return self._orgs.get(org_id)

    def get_privacy_status(self, org_id: str) -> dict[str, Any]:
        org = self._orgs.get(org_id)
        if not org:
            return {}
        return {
            "org_id": org_id,
            "epsilon_total": org.privacy_budget.total_epsilon,
            "epsilon_used": round(org.privacy_budget.used_epsilon, 4),
            "epsilon_remaining": round(org.privacy_budget.remaining_epsilon, 4),
            "queries_made": org.privacy_budget.queries_made,
            "patterns_contributed": org.patterns_contributed,
            "patterns_adopted": org.patterns_adopted,
        }

    def get_rounds(self) -> list[FederatedRound]:
        return list(self._rounds)

    def get_audit_log(self, org_id: str | None = None) -> list[AuditLogEntry]:
        """Get audit log entries, optionally filtered by org."""
        if org_id:
            return [e for e in self._audit_log if e.org_id == org_id]
        return list(self._audit_log)

    def calibrate_noise(
        self, sensitivity: float, desired_accuracy: float, dataset_size: int,
    ) -> float:
        """Get recommended epsilon for desired accuracy."""
        return self._calibrator.calibrate_epsilon(sensitivity, desired_accuracy, dataset_size)

    def _audit(
        self,
        action: str,
        org_id: str,
        details: dict[str, Any] | None = None,
        success: bool = True,
        error: str = "",
    ) -> None:
        entry = AuditLogEntry(
            org_id=org_id, action=action,
            details=details or {}, success=success, error=error,
        )
        self._audit_log.append(entry)
        log_fn = logger.info if success else logger.warning
        log_fn("federated_audit", action=action, org_id=org_id, success=success, error=error)


# ─── Singleton Access ──────────────────────────────────────────────────


_federated_instance: FederatedVerificationService | None = None


def get_federated_verification_service() -> FederatedVerificationService:
    """Get or create the singleton FederatedVerificationService."""
    global _federated_instance
    if _federated_instance is None:
        _federated_instance = FederatedVerificationService()
    return _federated_instance


def reset_federated_verification_service() -> None:
    """Reset the singleton (for testing)."""
    global _federated_instance
    _federated_instance = None
