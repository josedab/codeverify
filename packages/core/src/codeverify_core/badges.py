"""Verification badge attestation and certification system.

This module provides:
- Attestation schema for verification results (JSON-LD format)
- Cryptographic signing and verification of attestations
- Certification tiers (Bronze, Silver, Gold, Platinum)
- Badge generation and validation
"""

import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field


class CertificationTier(str, Enum):
    """Certification tier levels based on verification coverage and history."""

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"


class AttestationType(str, Enum):
    """Type of verification attestation."""

    ANALYSIS = "analysis"  # Single analysis attestation
    REPOSITORY = "repository"  # Repository-wide certification
    RELEASE = "release"  # Release/tag certification
    FILE = "file"  # Individual file attestation


class VerificationScope(BaseModel):
    """Scope of verification covered by attestation."""

    null_safety: bool = False
    array_bounds: bool = False
    integer_overflow: bool = False
    division_by_zero: bool = False
    security_vulnerabilities: bool = False
    type_safety: bool = False
    resource_leaks: bool = False
    ai_code_detection: bool = False


class AttestationSubject(BaseModel):
    """Subject of the attestation (what was verified)."""

    repo_full_name: str
    ref: str | None = None  # branch, tag, or commit SHA
    commit_sha: str | None = None
    file_paths: list[str] = Field(default_factory=list)
    pr_number: int | None = None


class VerificationEvidence(BaseModel):
    """Evidence supporting the verification claim."""

    analysis_id: UUID | None = None
    findings_count: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    formal_proofs_count: int = 0
    ai_analyses_count: int = 0
    files_verified: int = 0
    lines_of_code: int = 0
    verification_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    duration_ms: int = 0


class TierRequirements(BaseModel):
    """Requirements for achieving a certification tier."""

    min_verification_coverage: float
    max_critical_findings: int
    max_high_findings: int
    min_formal_proofs_ratio: float
    min_consecutive_clean_analyses: int
    validity_days: int


TIER_REQUIREMENTS: dict[CertificationTier, TierRequirements] = {
    CertificationTier.BRONZE: TierRequirements(
        min_verification_coverage=0.5,
        max_critical_findings=0,
        max_high_findings=5,
        min_formal_proofs_ratio=0.0,
        min_consecutive_clean_analyses=1,
        validity_days=30,
    ),
    CertificationTier.SILVER: TierRequirements(
        min_verification_coverage=0.7,
        max_critical_findings=0,
        max_high_findings=2,
        min_formal_proofs_ratio=0.3,
        min_consecutive_clean_analyses=5,
        validity_days=60,
    ),
    CertificationTier.GOLD: TierRequirements(
        min_verification_coverage=0.85,
        max_critical_findings=0,
        max_high_findings=0,
        min_formal_proofs_ratio=0.5,
        min_consecutive_clean_analyses=10,
        validity_days=90,
    ),
    CertificationTier.PLATINUM: TierRequirements(
        min_verification_coverage=0.95,
        max_critical_findings=0,
        max_high_findings=0,
        min_formal_proofs_ratio=0.8,
        min_consecutive_clean_analyses=25,
        validity_days=180,
    ),
}


class VerificationAttestation(BaseModel):
    """
    Verification attestation in JSON-LD format.

    This follows the in-toto attestation framework and SLSA provenance spec
    for software supply chain security.
    """

    # JSON-LD context
    context: str = Field(
        default="https://codeverify.dev/attestation/v1",
        alias="@context",
    )
    type_: str = Field(default="VerificationAttestation", alias="@type")

    # Attestation metadata
    id: UUID = Field(default_factory=uuid4)
    attestation_type: AttestationType
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    version: str = "1.0.0"

    # Issuer information
    issuer: str = "https://codeverify.dev"
    issuer_org_id: UUID | None = None

    # Subject and evidence
    subject: AttestationSubject
    scope: VerificationScope
    evidence: VerificationEvidence

    # Certification
    tier: CertificationTier | None = None
    passed: bool = False

    # Cryptographic binding
    signature: str | None = None
    signature_algorithm: str = "HMAC-SHA256"
    public_key_id: str | None = None

    # Audit trail
    parent_attestation_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field
    @property
    def attestation_hash(self) -> str:
        """Compute hash of attestation content for verification."""
        content = self._get_signable_content()
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_signable_content(self) -> str:
        """Get the content that should be signed."""
        signable = {
            "id": str(self.id),
            "attestation_type": self.attestation_type.value,
            "created_at": self.created_at.isoformat(),
            "subject": self.subject.model_dump(),
            "scope": self.scope.model_dump(),
            "evidence": self.evidence.model_dump(),
            "tier": self.tier.value if self.tier else None,
            "passed": self.passed,
        }
        return json.dumps(signable, sort_keys=True, default=str)

    def sign(self, secret_key: str) -> None:
        """Sign the attestation with HMAC-SHA256."""
        content = self._get_signable_content()
        self.signature = hmac.new(
            secret_key.encode(),
            content.encode(),
            hashlib.sha256,
        ).hexdigest()

    def verify_signature(self, secret_key: str) -> bool:
        """Verify the attestation signature."""
        if not self.signature:
            return False
        content = self._get_signable_content()
        expected = hmac.new(
            secret_key.encode(),
            content.encode(),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(self.signature, expected)

    def is_valid(self) -> bool:
        """Check if attestation is still valid (not expired)."""
        if self.expires_at is None:
            return True
        return datetime.utcnow() < self.expires_at

    def to_json_ld(self) -> dict[str, Any]:
        """Export as JSON-LD document."""
        return {
            "@context": self.context,
            "@type": self.type_,
            "id": f"urn:uuid:{self.id}",
            "attestationType": self.attestation_type.value,
            "createdAt": self.created_at.isoformat(),
            "expiresAt": self.expires_at.isoformat() if self.expires_at else None,
            "issuer": {
                "@type": "Organization",
                "name": "CodeVerify",
                "url": self.issuer,
            },
            "subject": {
                "@type": "SoftwareSourceCode",
                "repository": self.subject.repo_full_name,
                "ref": self.subject.ref,
                "commitHash": self.subject.commit_sha,
                "files": self.subject.file_paths,
            },
            "verificationScope": self.scope.model_dump(),
            "evidence": self.evidence.model_dump(),
            "certification": {
                "tier": self.tier.value if self.tier else None,
                "passed": self.passed,
            },
            "proof": {
                "type": self.signature_algorithm,
                "hash": self.attestation_hash,
                "signature": self.signature,
            },
        }

    class Config:
        populate_by_name = True


class CertificationResult(BaseModel):
    """Result of evaluating certification tier."""

    eligible: bool
    tier: CertificationTier | None = None
    next_tier: CertificationTier | None = None
    missing_requirements: list[str] = Field(default_factory=list)
    progress: dict[str, float] = Field(default_factory=dict)


class BadgeConfig(BaseModel):
    """Configuration for badge generation."""

    style: str = "flat"  # flat, flat-square, plastic, for-the-badge
    label: str = "CodeVerify"
    color_scheme: str = "auto"  # auto, dark, light
    include_tier: bool = True
    include_score: bool = False


class Badge(BaseModel):
    """Generated verification badge."""

    id: UUID = Field(default_factory=uuid4)
    attestation_id: UUID
    repo_full_name: str
    tier: CertificationTier | None = None
    passed: bool = False
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    config: BadgeConfig = Field(default_factory=BadgeConfig)
    token: str = Field(default_factory=lambda: secrets.token_urlsafe(16))

    @computed_field
    @property
    def embed_url(self) -> str:
        """URL to embed badge in markdown/HTML."""
        return f"https://codeverify.dev/badge/{self.token}.svg"

    @computed_field
    @property
    def verify_url(self) -> str:
        """URL to verify badge authenticity."""
        return f"https://codeverify.dev/verify/{self.token}"

    @computed_field
    @property
    def markdown(self) -> str:
        """Markdown snippet for embedding."""
        return f"[![CodeVerify]({self.embed_url})]({self.verify_url})"

    @computed_field
    @property
    def html(self) -> str:
        """HTML snippet for embedding."""
        return f'<a href="{self.verify_url}"><img src="{self.embed_url}" alt="CodeVerify" /></a>'


def evaluate_tier(
    evidence: VerificationEvidence,
    consecutive_clean: int = 1,
) -> CertificationResult:
    """
    Evaluate which certification tier the evidence qualifies for.

    Args:
        evidence: Verification evidence from analysis
        consecutive_clean: Number of consecutive clean analyses

    Returns:
        CertificationResult with tier evaluation
    """
    result = CertificationResult(eligible=False)

    # Check tiers from highest to lowest
    for tier in [
        CertificationTier.PLATINUM,
        CertificationTier.GOLD,
        CertificationTier.SILVER,
        CertificationTier.BRONZE,
    ]:
        reqs = TIER_REQUIREMENTS[tier]
        missing = []
        progress = {}

        # Check verification coverage
        if evidence.verification_coverage < reqs.min_verification_coverage:
            missing.append(
                f"Verification coverage {evidence.verification_coverage:.0%} "
                f"< required {reqs.min_verification_coverage:.0%}"
            )
        progress["coverage"] = min(
            evidence.verification_coverage / reqs.min_verification_coverage, 1.0
        )

        # Check critical findings
        if evidence.critical_count > reqs.max_critical_findings:
            missing.append(
                f"Critical findings {evidence.critical_count} "
                f"> allowed {reqs.max_critical_findings}"
            )
        progress["critical"] = 1.0 if evidence.critical_count <= reqs.max_critical_findings else 0.0

        # Check high findings
        if evidence.high_count > reqs.max_high_findings:
            missing.append(
                f"High findings {evidence.high_count} "
                f"> allowed {reqs.max_high_findings}"
            )
        progress["high"] = 1.0 if evidence.high_count <= reqs.max_high_findings else 0.0

        # Check formal proofs ratio
        total_checks = evidence.formal_proofs_count + evidence.ai_analyses_count
        formal_ratio = (
            evidence.formal_proofs_count / total_checks if total_checks > 0 else 0.0
        )
        if formal_ratio < reqs.min_formal_proofs_ratio:
            missing.append(
                f"Formal proof ratio {formal_ratio:.0%} "
                f"< required {reqs.min_formal_proofs_ratio:.0%}"
            )
        progress["formal_proofs"] = min(
            formal_ratio / reqs.min_formal_proofs_ratio
            if reqs.min_formal_proofs_ratio > 0
            else 1.0,
            1.0,
        )

        # Check consecutive clean analyses
        if consecutive_clean < reqs.min_consecutive_clean_analyses:
            missing.append(
                f"Consecutive clean analyses {consecutive_clean} "
                f"< required {reqs.min_consecutive_clean_analyses}"
            )
        progress["consecutive"] = min(
            consecutive_clean / reqs.min_consecutive_clean_analyses, 1.0
        )

        # If all requirements met, this is the tier
        if not missing:
            result.eligible = True
            result.tier = tier
            result.progress = progress
            # Set next tier if not platinum
            tier_order = list(CertificationTier)
            tier_idx = tier_order.index(tier)
            if tier_idx > 0:
                result.next_tier = tier_order[tier_idx - 1]
            return result

        # If this is the first tier we don't qualify for, record requirements
        if result.tier is None and tier == CertificationTier.BRONZE:
            result.missing_requirements = missing
            result.progress = progress
            result.next_tier = CertificationTier.BRONZE

    return result


def create_attestation(
    subject: AttestationSubject,
    evidence: VerificationEvidence,
    scope: VerificationScope,
    attestation_type: AttestationType = AttestationType.ANALYSIS,
    issuer_org_id: UUID | None = None,
    consecutive_clean: int = 1,
) -> VerificationAttestation:
    """
    Create a new verification attestation.

    Args:
        subject: What was verified
        evidence: Verification results
        scope: What verification checks were performed
        attestation_type: Type of attestation
        issuer_org_id: Organization ID that requested verification
        consecutive_clean: Number of consecutive clean analyses

    Returns:
        Unsigned VerificationAttestation
    """
    # Evaluate certification tier
    cert_result = evaluate_tier(evidence, consecutive_clean)

    # Determine expiration based on tier
    if cert_result.tier:
        validity_days = TIER_REQUIREMENTS[cert_result.tier].validity_days
        expires_at = datetime.utcnow() + timedelta(days=validity_days)
    else:
        expires_at = datetime.utcnow() + timedelta(days=7)  # Non-certified: 7 days

    # Determine if passed (no critical/high, coverage met)
    passed = (
        evidence.critical_count == 0
        and evidence.high_count <= 2
        and evidence.verification_coverage >= 0.5
    )

    return VerificationAttestation(
        attestation_type=attestation_type,
        expires_at=expires_at,
        issuer_org_id=issuer_org_id,
        subject=subject,
        scope=scope,
        evidence=evidence,
        tier=cert_result.tier,
        passed=passed,
    )


def create_badge(
    attestation: VerificationAttestation,
    config: BadgeConfig | None = None,
) -> Badge:
    """
    Create a badge from an attestation.

    Args:
        attestation: The verification attestation
        config: Badge configuration options

    Returns:
        Badge object with embed URLs
    """
    return Badge(
        attestation_id=attestation.id,
        repo_full_name=attestation.subject.repo_full_name,
        tier=attestation.tier,
        passed=attestation.passed,
        expires_at=attestation.expires_at,
        config=config or BadgeConfig(),
    )
