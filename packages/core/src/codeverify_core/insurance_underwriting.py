"""AI Code Insurance Underwriting Platform.

Risk assessment APIs with actuarial models, premium calculation based on trust
scores, claim validation through verification proofs, and insurance partner
integration for AI-generated code liability transfer.

Features:
- Actuarial risk scoring from code quality metrics
- Premium calculation with configurable models
- Claim submission and automated triage
- Policy lifecycle management
- Insurance partner API abstractions
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class RiskTier(str, Enum):
    """Actuarial risk tier for a codebase."""

    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class PolicyStatus(str, Enum):
    """Status of an insurance policy."""

    DRAFT = "draft"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    CLAIMED = "claimed"


class ClaimStatus(str, Enum):
    """Status of an insurance claim."""

    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    VALIDATED = "validated"
    REJECTED = "rejected"
    PAID = "paid"
    APPEALED = "appealed"


class ClaimRejectionReason(str, Enum):
    """Reasons a claim may be rejected."""

    UNVERIFIED_CODE = "unverified_code"
    POLICY_EXPIRED = "policy_expired"
    OUTSIDE_COVERAGE = "outside_coverage"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FRAUDULENT = "fraudulent"
    PRE_EXISTING = "pre_existing"


class CoverageType(str, Enum):
    """Types of coverage offered."""

    SECURITY_VULNERABILITY = "security_vulnerability"
    LOGIC_ERROR = "logic_error"
    DATA_LOSS = "data_loss"
    COMPLIANCE_VIOLATION = "compliance_violation"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    FULL = "full"


@dataclass
class RiskProfile:
    """Actuarial risk profile for a codebase or repository."""

    repo_id: str
    trust_score: float = 50.0
    verification_coverage: float = 0.0
    lines_of_code: int = 0
    ai_generated_ratio: float = 0.0
    language: str = "python"
    industry: str = "technology"
    open_findings: int = 0
    critical_findings: int = 0
    historical_incidents: int = 0
    last_assessed: datetime = field(
        default_factory=lambda: datetime.now(UTC),
    )

    @property
    def risk_tier(self) -> RiskTier:
        """Calculate risk tier from profile metrics."""
        score = self.composite_risk_score
        if score < 0.1:
            return RiskTier.MINIMAL
        elif score < 0.3:
            return RiskTier.LOW
        elif score < 0.5:
            return RiskTier.MODERATE
        elif score < 0.75:
            return RiskTier.HIGH
        return RiskTier.CRITICAL

    @property
    def composite_risk_score(self) -> float:
        """Compute composite risk 0.0 (safe) to 1.0 (risky)."""
        trust_risk = max(0.0, 1.0 - (self.trust_score / 100.0))
        coverage_risk = max(0.0, 1.0 - self.verification_coverage)
        ai_risk = self.ai_generated_ratio * 0.5
        finding_risk = min(1.0, (self.critical_findings * 0.2 + self.open_findings * 0.05))
        incident_risk = min(1.0, self.historical_incidents * 0.15)

        weights = {
            "trust": 0.30,
            "coverage": 0.25,
            "ai": 0.15,
            "findings": 0.20,
            "incidents": 0.10,
        }
        return (
            trust_risk * weights["trust"]
            + coverage_risk * weights["coverage"]
            + ai_risk * weights["ai"]
            + finding_risk * weights["findings"]
            + incident_risk * weights["incidents"]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "trust_score": self.trust_score,
            "verification_coverage": self.verification_coverage,
            "lines_of_code": self.lines_of_code,
            "ai_generated_ratio": self.ai_generated_ratio,
            "language": self.language,
            "industry": self.industry,
            "open_findings": self.open_findings,
            "critical_findings": self.critical_findings,
            "historical_incidents": self.historical_incidents,
            "risk_tier": self.risk_tier.value,
            "composite_risk_score": round(self.composite_risk_score, 4),
        }


@dataclass
class PremiumCalculation:
    """Result of premium calculation for a policy."""

    base_premium: float = 0.0
    risk_multiplier: float = 1.0
    coverage_discount: float = 0.0
    industry_factor: float = 1.0
    volume_discount: float = 0.0
    final_monthly_premium: float = 0.0
    currency: str = "USD"

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_premium": round(self.base_premium, 2),
            "risk_multiplier": round(self.risk_multiplier, 4),
            "coverage_discount": round(self.coverage_discount, 2),
            "industry_factor": round(self.industry_factor, 4),
            "volume_discount": round(self.volume_discount, 2),
            "final_monthly_premium": round(self.final_monthly_premium, 2),
            "currency": self.currency,
        }


INDUSTRY_FACTORS: dict[str, float] = {
    "healthcare": 1.5,
    "finance": 1.4,
    "government": 1.3,
    "defense": 1.6,
    "technology": 1.0,
    "education": 0.8,
    "retail": 1.1,
    "manufacturing": 1.2,
}


@dataclass
class InsurancePolicy:
    """An active insurance policy for AI-generated code."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    holder_org: str = ""
    repo_ids: list[str] = field(default_factory=list)
    coverage_types: list[CoverageType] = field(
        default_factory=lambda: [CoverageType.FULL],
    )
    coverage_limit: float = 100_000.0
    deductible: float = 5_000.0
    monthly_premium: float = 0.0
    status: PolicyStatus = PolicyStatus.DRAFT
    risk_profile: RiskProfile | None = None
    effective_date: datetime = field(
        default_factory=lambda: datetime.now(UTC),
    )
    expiry_date: datetime | None = None
    claims: list[str] = field(default_factory=list)
    total_paid_claims: float = 0.0
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC),
    )

    @property
    def remaining_coverage(self) -> float:
        return max(0.0, self.coverage_limit - self.total_paid_claims)

    @property
    def is_active(self) -> bool:
        now = datetime.now(UTC)
        if self.status != PolicyStatus.ACTIVE:
            return False
        return not (self.expiry_date and now > self.expiry_date)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "holder_org": self.holder_org,
            "repo_ids": self.repo_ids,
            "coverage_types": [ct.value for ct in self.coverage_types],
            "coverage_limit": self.coverage_limit,
            "deductible": self.deductible,
            "monthly_premium": round(self.monthly_premium, 2),
            "status": self.status.value,
            "remaining_coverage": round(self.remaining_coverage, 2),
            "claims_count": len(self.claims),
            "total_paid_claims": round(self.total_paid_claims, 2),
        }


@dataclass
class InsuranceClaim:
    """An insurance claim against a policy."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    policy_id: str = ""
    repo_id: str = ""
    description: str = ""
    claimed_amount: float = 0.0
    coverage_type: CoverageType = CoverageType.LOGIC_ERROR
    evidence_proof_ids: list[str] = field(default_factory=list)
    verification_state_at_time: str = ""
    status: ClaimStatus = ClaimStatus.SUBMITTED
    rejection_reason: ClaimRejectionReason | None = None
    paid_amount: float = 0.0
    submitted_at: datetime = field(
        default_factory=lambda: datetime.now(UTC),
    )
    resolved_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "policy_id": self.policy_id,
            "repo_id": self.repo_id,
            "description": self.description,
            "claimed_amount": round(self.claimed_amount, 2),
            "coverage_type": self.coverage_type.value,
            "evidence_count": len(self.evidence_proof_ids),
            "status": self.status.value,
            "rejection_reason": self.rejection_reason.value if self.rejection_reason else None,
            "paid_amount": round(self.paid_amount, 2),
        }


class PremiumCalculator:
    """Actuarial premium calculator based on risk profiles."""

    BASE_RATE_PER_1K_LOC = 2.50  # $2.50 per 1K LOC per month

    def calculate(
        self,
        profile: RiskProfile,
        coverage_limit: float = 100_000.0,
        deductible: float = 5_000.0,
    ) -> PremiumCalculation:
        """Calculate monthly premium for given risk profile."""
        loc_units = max(1, profile.lines_of_code / 1000)
        base = loc_units * self.BASE_RATE_PER_1K_LOC

        risk_score = profile.composite_risk_score
        risk_mult = 1.0 + (risk_score * 3.0)  # 1x to 4x

        coverage_disc = 0.0
        if profile.verification_coverage >= 0.8:
            coverage_disc = base * 0.20
        elif profile.verification_coverage >= 0.5:
            coverage_disc = base * 0.10

        industry_factor = INDUSTRY_FACTORS.get(profile.industry, 1.0)

        coverage_factor = coverage_limit / 100_000.0
        deductible_factor = max(0.5, 1.0 - (deductible / 50_000.0))

        volume_disc = 0.0
        if loc_units > 500:
            volume_disc = base * 0.15
        elif loc_units > 100:
            volume_disc = base * 0.05

        final = (
            base * risk_mult - coverage_disc
        ) * industry_factor * coverage_factor * deductible_factor - volume_disc
        final = max(10.0, final)  # Minimum $10/month

        return PremiumCalculation(
            base_premium=base,
            risk_multiplier=risk_mult,
            coverage_discount=coverage_disc,
            industry_factor=industry_factor,
            volume_discount=volume_disc,
            final_monthly_premium=final,
        )


class ClaimValidator:
    """Validates insurance claims against verification proofs."""

    def validate(
        self,
        claim: InsuranceClaim,
        policy: InsurancePolicy,
        proof_hashes: list[str] | None = None,
    ) -> tuple[bool, ClaimRejectionReason | None]:
        """Validate a claim. Returns (is_valid, rejection_reason)."""
        if not policy.is_active:
            return False, ClaimRejectionReason.POLICY_EXPIRED

        if claim.repo_id not in policy.repo_ids:
            return False, ClaimRejectionReason.OUTSIDE_COVERAGE

        if (
            claim.coverage_type not in policy.coverage_types
            and CoverageType.FULL not in policy.coverage_types
        ):
            return False, ClaimRejectionReason.OUTSIDE_COVERAGE

        if not claim.evidence_proof_ids:
            return False, ClaimRejectionReason.INSUFFICIENT_EVIDENCE

        if claim.claimed_amount > policy.remaining_coverage:
            return False, ClaimRejectionReason.OUTSIDE_COVERAGE

        if proof_hashes is not None:
            verified = any(h for h in proof_hashes if len(h) >= 32)
            if not verified:
                return False, ClaimRejectionReason.UNVERIFIED_CODE

        return True, None


class InsuranceUnderwriter:
    """Main insurance underwriting engine.

    Manages policies, risk assessment, premium calculation, and claims.
    """

    def __init__(self) -> None:
        self._policies: dict[str, InsurancePolicy] = {}
        self._claims: dict[str, InsuranceClaim] = {}
        self._risk_profiles: dict[str, RiskProfile] = {}
        self._premium_calculator = PremiumCalculator()
        self._claim_validator = ClaimValidator()

    def assess_risk(self, profile: RiskProfile) -> RiskProfile:
        """Assess and store a risk profile for a repository."""
        profile.last_assessed = datetime.now(UTC)
        self._risk_profiles[profile.repo_id] = profile
        logger.info(
            "risk_assessed",
            repo_id=profile.repo_id,
            risk_tier=profile.risk_tier.value,
            score=round(profile.composite_risk_score, 4),
        )
        return profile

    def calculate_premium(
        self,
        repo_id: str,
        coverage_limit: float = 100_000.0,
        deductible: float = 5_000.0,
    ) -> PremiumCalculation:
        """Calculate premium for a repo based on its risk profile."""
        profile = self._risk_profiles.get(repo_id)
        if profile is None:
            profile = RiskProfile(repo_id=repo_id)
        return self._premium_calculator.calculate(profile, coverage_limit, deductible)

    def create_policy(
        self,
        holder_org: str,
        repo_ids: list[str],
        coverage_limit: float = 100_000.0,
        deductible: float = 5_000.0,
        coverage_types: list[CoverageType] | None = None,
    ) -> InsurancePolicy:
        """Create a new insurance policy."""
        total_premium = 0.0
        for rid in repo_ids:
            calc = self.calculate_premium(rid, coverage_limit, deductible)
            total_premium += calc.final_monthly_premium

        policy = InsurancePolicy(
            holder_org=holder_org,
            repo_ids=repo_ids,
            coverage_types=coverage_types or [CoverageType.FULL],
            coverage_limit=coverage_limit,
            deductible=deductible,
            monthly_premium=total_premium,
            status=PolicyStatus.ACTIVE,
        )
        self._policies[policy.id] = policy
        logger.info(
            "policy_created",
            policy_id=policy.id,
            org=holder_org,
            premium=round(total_premium, 2),
        )
        return policy

    def submit_claim(
        self,
        policy_id: str,
        repo_id: str,
        description: str,
        claimed_amount: float,
        coverage_type: CoverageType = CoverageType.LOGIC_ERROR,
        evidence_proof_ids: list[str] | None = None,
    ) -> InsuranceClaim:
        """Submit an insurance claim."""
        claim = InsuranceClaim(
            policy_id=policy_id,
            repo_id=repo_id,
            description=description,
            claimed_amount=claimed_amount,
            coverage_type=coverage_type,
            evidence_proof_ids=evidence_proof_ids or [],
        )
        self._claims[claim.id] = claim

        policy = self._policies.get(policy_id)
        if policy:
            policy.claims.append(claim.id)

        logger.info(
            "claim_submitted",
            claim_id=claim.id,
            policy_id=policy_id,
            amount=claimed_amount,
        )
        return claim

    def process_claim(
        self,
        claim_id: str,
        proof_hashes: list[str] | None = None,
    ) -> InsuranceClaim:
        """Process and validate a claim."""
        claim = self._claims.get(claim_id)
        if not claim:
            raise ValueError(f"Claim {claim_id} not found")

        policy = self._policies.get(claim.policy_id)
        if not policy:
            claim.status = ClaimStatus.REJECTED
            claim.rejection_reason = ClaimRejectionReason.POLICY_EXPIRED
            return claim

        claim.status = ClaimStatus.UNDER_REVIEW
        is_valid, reason = self._claim_validator.validate(claim, policy, proof_hashes)

        if is_valid:
            net_amount = max(0.0, claim.claimed_amount - policy.deductible)
            paid = min(net_amount, policy.remaining_coverage)
            claim.status = ClaimStatus.VALIDATED
            claim.paid_amount = paid
            policy.total_paid_claims += paid
            claim.resolved_at = datetime.now(UTC)
            logger.info("claim_validated", claim_id=claim_id, paid=round(paid, 2))
        else:
            claim.status = ClaimStatus.REJECTED
            claim.rejection_reason = reason
            claim.resolved_at = datetime.now(UTC)
            logger.info("claim_rejected", claim_id=claim_id, reason=reason)

        return claim

    def get_policy(self, policy_id: str) -> InsurancePolicy | None:
        return self._policies.get(policy_id)

    def get_claim(self, claim_id: str) -> InsuranceClaim | None:
        return self._claims.get(claim_id)

    def get_portfolio_summary(self) -> dict[str, Any]:
        """Get summary of all policies and claims."""
        active = [p for p in self._policies.values() if p.is_active]
        total_premium = sum(p.monthly_premium for p in active)
        total_claims = sum(c.paid_amount for c in self._claims.values())
        return {
            "total_policies": len(self._policies),
            "active_policies": len(active),
            "total_monthly_premium": round(total_premium, 2),
            "total_claims_paid": round(total_claims, 2),
            "loss_ratio": round(total_claims / max(1.0, total_premium * 12), 4),
            "claims_count": len(self._claims),
            "pending_claims": len(
                [c for c in self._claims.values() if c.status == ClaimStatus.SUBMITTED]
            ),
        }


_default_underwriter: InsuranceUnderwriter | None = None


def get_insurance_underwriter() -> InsuranceUnderwriter:
    """Get the singleton insurance underwriter."""
    global _default_underwriter
    if _default_underwriter is None:
        _default_underwriter = InsuranceUnderwriter()
    return _default_underwriter


def reset_insurance_underwriter() -> None:
    """Reset the singleton (for testing)."""
    global _default_underwriter
    _default_underwriter = None
