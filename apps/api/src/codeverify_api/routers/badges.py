"""Verification badges and attestation API router."""

import secrets
from datetime import datetime
from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from codeverify_api.auth.dependencies import get_current_user_optional
from codeverify_api.db.database import get_db
from codeverify_api.config import settings
from codeverify_api.db.models import (
    Analysis,
    CertificationHistory,
    Repository,
    VerificationAttestation as AttestationDB,
    VerificationBadge as BadgeDB,
)
from codeverify_core.badges import (
    AttestationSubject,
    AttestationType,
    Badge,
    BadgeConfig,
    CertificationResult,
    CertificationTier,
    TIER_REQUIREMENTS,
    TierRequirements,
    VerificationAttestation,
    VerificationEvidence,
    VerificationScope,
    create_attestation,
    create_badge,
    evaluate_tier,
)

logger = structlog.get_logger()

router = APIRouter()


# Request/Response models
class CreateAttestationRequest(BaseModel):
    """Request to create a verification attestation."""

    repo_full_name: str
    ref: str | None = None
    commit_sha: str | None = None
    file_paths: list[str] = Field(default_factory=list)
    pr_number: int | None = None
    analysis_id: UUID | None = None
    attestation_type: AttestationType = AttestationType.ANALYSIS
    scope: VerificationScope | None = None
    evidence: VerificationEvidence | None = None


class AttestationResponse(BaseModel):
    """Attestation response."""

    id: UUID
    attestation_type: str
    repo_full_name: str
    ref: str | None
    commit_sha: str | None
    tier: str | None
    passed: bool
    expires_at: datetime | None
    created_at: datetime
    attestation_hash: str
    signature: str | None
    verify_url: str
    json_ld: dict[str, Any]

    class Config:
        from_attributes = True


class BadgeResponse(BaseModel):
    """Badge response with embed URLs."""

    id: UUID
    attestation_id: UUID
    repo_full_name: str
    tier: str | None
    passed: bool
    token: str
    embed_url: str
    verify_url: str
    markdown: str
    html: str
    expires_at: datetime | None
    created_at: datetime

    class Config:
        from_attributes = True


class CreateBadgeRequest(BaseModel):
    """Request to create a badge."""

    attestation_id: UUID
    style: str = "flat"
    label: str = "CodeVerify"
    include_tier: bool = True
    include_score: bool = False


class TierProgressResponse(BaseModel):
    """Progress towards certification tiers."""

    current_tier: str | None
    next_tier: str | None
    progress: dict[str, float]
    missing_requirements: list[str]
    tier_requirements: dict[str, TierRequirements]


class VerifyAttestationResponse(BaseModel):
    """Response for attestation verification."""

    valid: bool
    attestation_id: UUID | None
    repo_full_name: str | None
    tier: str | None
    passed: bool | None
    expires_at: datetime | None
    expired: bool
    signature_valid: bool | None
    message: str


# Badge SVG generation
def generate_badge_svg(
    label: str,
    message: str,
    color: str,
    style: str = "flat",
) -> str:
    """Generate an SVG badge."""
    # Color mapping
    colors = {
        "brightgreen": "#4c1",
        "green": "#97ca00",
        "yellowgreen": "#a4a61d",
        "yellow": "#dfb317",
        "orange": "#fe7d37",
        "red": "#e05d44",
        "blue": "#007ec6",
        "lightgrey": "#9f9f9f",
        "gray": "#555",
        "bronze": "#cd7f32",
        "silver": "#c0c0c0",
        "gold": "#ffd700",
        "platinum": "#e5e4e2",
    }
    bg_color = colors.get(color, color)

    label_width = len(label) * 6.5 + 10
    message_width = len(message) * 6.5 + 10
    total_width = label_width + message_width

    if style == "flat-square":
        return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <rect rx="0" width="{total_width}" height="20" fill="#555"/>
  <rect rx="0" x="{label_width}" width="{message_width}" height="20" fill="{bg_color}"/>
  <rect rx="0" width="{total_width}" height="20" fill="url(#b)"/>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="{label_width/2}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{label_width/2}" y="14">{label}</text>
    <text x="{label_width + message_width/2}" y="15" fill="#010101" fill-opacity=".3">{message}</text>
    <text x="{label_width + message_width/2}" y="14">{message}</text>
  </g>
</svg>'''

    # Default flat style
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20" role="img" aria-label="{label}: {message}">
  <title>{label}: {message}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r">
    <rect width="{total_width}" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="{label_width}" height="20" fill="#555"/>
    <rect x="{label_width}" width="{message_width}" height="20" fill="{bg_color}"/>
    <rect width="{total_width}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="110">
    <text aria-hidden="true" x="{label_width*5}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)">{label}</text>
    <text x="{label_width*5}" y="140" transform="scale(.1)" fill="#fff">{label}</text>
    <text aria-hidden="true" x="{(label_width + message_width/2)*10}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)">{message}</text>
    <text x="{(label_width + message_width/2)*10}" y="140" transform="scale(.1)" fill="#fff">{message}</text>
  </g>
</svg>'''


def get_badge_color(tier: str | None, passed: bool) -> str:
    """Get badge color based on tier and status."""
    if not passed:
        return "red"
    if tier:
        return tier.lower()
    return "green"


def get_badge_message(tier: str | None, passed: bool) -> str:
    """Get badge message text."""
    if not passed:
        return "not verified"
    if tier:
        return f"{tier.lower()} certified"
    return "verified"


# API Endpoints
@router.post("/attestations", response_model=AttestationResponse)
async def create_verification_attestation(
    request: CreateAttestationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict | None = Depends(get_current_user_optional),
) -> AttestationResponse:
    """
    Create a new verification attestation.

    This creates a cryptographically signed attestation of verification results
    that can be used to generate badges.
    """
    # Look up repository if exists
    repo = None
    if request.repo_full_name:
        result = await db.execute(
            select(Repository).where(Repository.full_name == request.repo_full_name)
        )
        repo = result.scalar_one_or_none()

    # Get evidence from analysis if provided
    evidence = request.evidence
    if request.analysis_id and not evidence:
        result = await db.execute(
            select(Analysis).where(Analysis.id == request.analysis_id)
        )
        analysis = result.scalar_one_or_none()
        if analysis:
            # Build evidence from analysis
            from sqlalchemy import func as sqlfunc
            from codeverify_api.db.models import Finding

            findings_result = await db.execute(
                select(Finding).where(Finding.analysis_id == analysis.id)
            )
            findings = findings_result.scalars().all()

            critical = sum(1 for f in findings if f.severity == "critical")
            high = sum(1 for f in findings if f.severity == "high")
            medium = sum(1 for f in findings if f.severity == "medium")
            low = sum(1 for f in findings if f.severity == "low")
            formal = sum(1 for f in findings if f.verification_type == "formal")
            ai = sum(1 for f in findings if f.verification_type == "ai")

            evidence = VerificationEvidence(
                analysis_id=analysis.id,
                findings_count=len(findings),
                critical_count=critical,
                high_count=high,
                medium_count=medium,
                low_count=low,
                formal_proofs_count=formal,
                ai_analyses_count=ai,
                files_verified=len(set(f.file_path for f in findings)),
                verification_coverage=0.8 if analysis.status == "completed" else 0.0,
            )

    if not evidence:
        evidence = VerificationEvidence()

    scope = request.scope or VerificationScope(
        null_safety=True,
        array_bounds=True,
        integer_overflow=True,
        security_vulnerabilities=True,
    )

    subject = AttestationSubject(
        repo_full_name=request.repo_full_name,
        ref=request.ref,
        commit_sha=request.commit_sha,
        file_paths=request.file_paths,
        pr_number=request.pr_number,
    )

    # Count consecutive clean analyses for this repo
    consecutive_clean = 1
    if repo:
        result = await db.execute(
            select(AttestationDB)
            .where(AttestationDB.repo_id == repo.id)
            .where(AttestationDB.passed == True)
            .order_by(AttestationDB.created_at.desc())
            .limit(50)
        )
        previous_attestations = result.scalars().all()
        for att in previous_attestations:
            if att.passed:
                consecutive_clean += 1
            else:
                break

    # Create attestation
    attestation = create_attestation(
        subject=subject,
        evidence=evidence,
        scope=scope,
        attestation_type=request.attestation_type,
        consecutive_clean=consecutive_clean,
    )

    # Sign the attestation
    attestation.sign(settings.JWT_SECRET)

    # Store in database
    db_attestation = AttestationDB(
        id=attestation.id,
        org_id=repo.org_id if repo else None,
        repo_id=repo.id if repo else None,
        analysis_id=request.analysis_id,
        attestation_type=attestation.attestation_type.value,
        repo_full_name=subject.repo_full_name,
        ref=subject.ref,
        commit_sha=subject.commit_sha,
        file_paths=subject.file_paths,
        scope=scope.model_dump(),
        evidence=evidence.model_dump(),
        tier=attestation.tier.value if attestation.tier else None,
        passed=attestation.passed,
        signature=attestation.signature,
        attestation_hash=attestation.attestation_hash,
        expires_at=attestation.expires_at,
        metadata=attestation.metadata,
    )

    db.add(db_attestation)

    # Record certification history if tier changed
    if repo and attestation.tier:
        result = await db.execute(
            select(CertificationHistory)
            .where(CertificationHistory.repo_id == repo.id)
            .order_by(CertificationHistory.created_at.desc())
            .limit(1)
        )
        last_history = result.scalar_one_or_none()
        previous_tier = last_history.new_tier if last_history else None

        if previous_tier != (attestation.tier.value if attestation.tier else None):
            reason = "initial" if not previous_tier else (
                "upgrade" if _tier_rank(attestation.tier) > _tier_rank(previous_tier) else "downgrade"
            )
            history = CertificationHistory(
                repo_id=repo.id,
                attestation_id=attestation.id,
                previous_tier=previous_tier,
                new_tier=attestation.tier.value if attestation.tier else None,
                reason=reason,
                evidence_snapshot=evidence.model_dump(),
            )
            db.add(history)

    await db.commit()

    logger.info(
        "Created verification attestation",
        attestation_id=str(attestation.id),
        repo=subject.repo_full_name,
        tier=attestation.tier.value if attestation.tier else None,
        passed=attestation.passed,
    )

    return AttestationResponse(
        id=attestation.id,
        attestation_type=attestation.attestation_type.value,
        repo_full_name=subject.repo_full_name,
        ref=subject.ref,
        commit_sha=subject.commit_sha,
        tier=attestation.tier.value if attestation.tier else None,
        passed=attestation.passed,
        expires_at=attestation.expires_at,
        created_at=attestation.created_at,
        attestation_hash=attestation.attestation_hash,
        signature=attestation.signature,
        verify_url=f"https://codeverify.dev/verify/attestation/{attestation.id}",
        json_ld=attestation.to_json_ld(),
    )


def _tier_rank(tier: CertificationTier | str | None) -> int:
    """Get numeric rank for tier comparison."""
    if tier is None:
        return 0
    tier_str = tier.value if isinstance(tier, CertificationTier) else tier
    ranks = {"bronze": 1, "silver": 2, "gold": 3, "platinum": 4}
    return ranks.get(tier_str.lower(), 0)


@router.get("/attestations/{attestation_id}", response_model=AttestationResponse)
async def get_attestation(
    attestation_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> AttestationResponse:
    """Get an attestation by ID."""
    result = await db.execute(
        select(AttestationDB).where(AttestationDB.id == attestation_id)
    )
    attestation = result.scalar_one_or_none()

    if not attestation:
        raise HTTPException(status_code=404, detail="Attestation not found")

    # Reconstruct the attestation object for JSON-LD
    att_obj = VerificationAttestation(
        id=attestation.id,
        attestation_type=AttestationType(attestation.attestation_type),
        created_at=attestation.created_at,
        expires_at=attestation.expires_at,
        subject=AttestationSubject(
            repo_full_name=attestation.repo_full_name,
            ref=attestation.ref,
            commit_sha=attestation.commit_sha,
            file_paths=attestation.file_paths,
        ),
        scope=VerificationScope(**attestation.scope),
        evidence=VerificationEvidence(**attestation.evidence),
        tier=CertificationTier(attestation.tier) if attestation.tier else None,
        passed=attestation.passed,
        signature=attestation.signature,
    )

    return AttestationResponse(
        id=attestation.id,
        attestation_type=attestation.attestation_type,
        repo_full_name=attestation.repo_full_name,
        ref=attestation.ref,
        commit_sha=attestation.commit_sha,
        tier=attestation.tier,
        passed=attestation.passed,
        expires_at=attestation.expires_at,
        created_at=attestation.created_at,
        attestation_hash=attestation.attestation_hash,
        signature=attestation.signature,
        verify_url=f"https://codeverify.dev/verify/attestation/{attestation.id}",
        json_ld=att_obj.to_json_ld(),
    )


@router.post("/badges", response_model=BadgeResponse)
async def create_verification_badge(
    request: CreateBadgeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict | None = Depends(get_current_user_optional),
) -> BadgeResponse:
    """Create a new badge from an attestation."""
    # Get attestation
    result = await db.execute(
        select(AttestationDB).where(AttestationDB.id == request.attestation_id)
    )
    attestation = result.scalar_one_or_none()

    if not attestation:
        raise HTTPException(status_code=404, detail="Attestation not found")

    # Generate unique token
    token = secrets.token_urlsafe(16)

    # Create badge record
    db_badge = BadgeDB(
        attestation_id=attestation.id,
        repo_id=attestation.repo_id,
        repo_full_name=attestation.repo_full_name,
        token=token,
        tier=attestation.tier,
        passed=attestation.passed,
        style=request.style,
        config={
            "label": request.label,
            "include_tier": request.include_tier,
            "include_score": request.include_score,
        },
        expires_at=attestation.expires_at,
    )

    db.add(db_badge)
    await db.commit()

    base_url = "https://codeverify.dev"
    embed_url = f"{base_url}/badge/{token}.svg"
    verify_url = f"{base_url}/verify/{token}"

    return BadgeResponse(
        id=db_badge.id,
        attestation_id=attestation.id,
        repo_full_name=attestation.repo_full_name,
        tier=attestation.tier,
        passed=attestation.passed,
        token=token,
        embed_url=embed_url,
        verify_url=verify_url,
        markdown=f"[![CodeVerify]({embed_url})]({verify_url})",
        html=f'<a href="{verify_url}"><img src="{embed_url}" alt="CodeVerify" /></a>',
        expires_at=attestation.expires_at,
        created_at=db_badge.created_at,
    )


@router.get("/badge/{token}.svg")
async def get_badge_svg(
    token: str,
    style: str = Query(default="flat", pattern="^(flat|flat-square|plastic)$"),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Get badge SVG image."""
    # Remove .svg extension if present
    token = token.replace(".svg", "")

    result = await db.execute(
        select(BadgeDB).where(BadgeDB.token == token)
    )
    badge = result.scalar_one_or_none()

    if not badge:
        # Return a "not found" badge
        svg = generate_badge_svg("CodeVerify", "not found", "lightgrey", style)
        return Response(content=svg, media_type="image/svg+xml")

    # Update view count
    badge.view_count += 1
    badge.last_viewed_at = datetime.utcnow()
    await db.commit()

    # Check expiration
    if badge.expires_at and datetime.utcnow() > badge.expires_at:
        svg = generate_badge_svg("CodeVerify", "expired", "lightgrey", style)
        return Response(content=svg, media_type="image/svg+xml")

    label = badge.config.get("label", "CodeVerify")
    color = get_badge_color(badge.tier, badge.passed)
    message = get_badge_message(badge.tier, badge.passed)

    svg = generate_badge_svg(label, message, color, badge.style or style)

    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={
            "Cache-Control": "max-age=300",  # 5 minutes
            "ETag": f'"{badge.token}-{badge.view_count}"',
        },
    )


@router.get("/verify/{token}", response_model=VerifyAttestationResponse)
async def verify_badge(
    token: str,
    db: AsyncSession = Depends(get_db),
) -> VerifyAttestationResponse:
    """Verify a badge's authenticity and status."""
    result = await db.execute(
        select(BadgeDB).where(BadgeDB.token == token)
    )
    badge = result.scalar_one_or_none()

    if not badge:
        return VerifyAttestationResponse(
            valid=False,
            attestation_id=None,
            repo_full_name=None,
            tier=None,
            passed=None,
            expires_at=None,
            expired=False,
            signature_valid=None,
            message="Badge not found",
        )

    # Get attestation
    result = await db.execute(
        select(AttestationDB).where(AttestationDB.id == badge.attestation_id)
    )
    attestation = result.scalar_one_or_none()

    if not attestation:
        return VerifyAttestationResponse(
            valid=False,
            attestation_id=badge.attestation_id,
            repo_full_name=badge.repo_full_name,
            tier=badge.tier,
            passed=badge.passed,
            expires_at=badge.expires_at,
            expired=False,
            signature_valid=None,
            message="Underlying attestation not found",
        )

    # Check expiration
    expired = False
    if attestation.expires_at and datetime.utcnow() > attestation.expires_at:
        expired = True

    # Verify signature
    signature_valid = None
    if attestation.signature:
        # Reconstruct attestation to verify
        att_obj = VerificationAttestation(
            id=attestation.id,
            attestation_type=AttestationType(attestation.attestation_type),
            created_at=attestation.created_at,
            subject=AttestationSubject(
                repo_full_name=attestation.repo_full_name,
                ref=attestation.ref,
                commit_sha=attestation.commit_sha,
                file_paths=attestation.file_paths,
            ),
            scope=VerificationScope(**attestation.scope),
            evidence=VerificationEvidence(**attestation.evidence),
            tier=CertificationTier(attestation.tier) if attestation.tier else None,
            passed=attestation.passed,
            signature=attestation.signature,
        )
        signature_valid = att_obj.verify_signature(settings.JWT_SECRET)

    valid = (
        not expired
        and attestation.passed
        and (signature_valid is None or signature_valid)
    )

    return VerifyAttestationResponse(
        valid=valid,
        attestation_id=attestation.id,
        repo_full_name=attestation.repo_full_name,
        tier=attestation.tier,
        passed=attestation.passed,
        expires_at=attestation.expires_at,
        expired=expired,
        signature_valid=signature_valid,
        message="Valid" if valid else ("Expired" if expired else "Verification failed"),
    )


@router.get("/repos/{owner}/{repo}/certification", response_model=TierProgressResponse)
async def get_certification_progress(
    owner: str,
    repo: str,
    db: AsyncSession = Depends(get_db),
) -> TierProgressResponse:
    """Get certification progress for a repository."""
    repo_full_name = f"{owner}/{repo}"

    # Get latest attestation
    result = await db.execute(
        select(AttestationDB)
        .where(AttestationDB.repo_full_name == repo_full_name)
        .order_by(AttestationDB.created_at.desc())
        .limit(1)
    )
    attestation = result.scalar_one_or_none()

    if not attestation:
        return TierProgressResponse(
            current_tier=None,
            next_tier="bronze",
            progress={},
            missing_requirements=["No verification attestations found"],
            tier_requirements={t.value: r for t, r in TIER_REQUIREMENTS.items()},
        )

    evidence = VerificationEvidence(**attestation.evidence)

    # Count consecutive clean analyses
    result = await db.execute(
        select(AttestationDB)
        .where(AttestationDB.repo_full_name == repo_full_name)
        .order_by(AttestationDB.created_at.desc())
        .limit(50)
    )
    attestations = result.scalars().all()
    consecutive_clean = 0
    for att in attestations:
        if att.passed:
            consecutive_clean += 1
        else:
            break

    cert_result = evaluate_tier(evidence, consecutive_clean)

    return TierProgressResponse(
        current_tier=cert_result.tier.value if cert_result.tier else None,
        next_tier=cert_result.next_tier.value if cert_result.next_tier else None,
        progress=cert_result.progress,
        missing_requirements=cert_result.missing_requirements,
        tier_requirements={t.value: r for t, r in TIER_REQUIREMENTS.items()},
    )


@router.get("/repos/{owner}/{repo}/history")
async def get_certification_history(
    owner: str,
    repo: str,
    limit: int = Query(default=10, le=100),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """Get certification history for a repository."""
    # Find repository
    repo_full_name = f"{owner}/{repo}"
    result = await db.execute(
        select(Repository).where(Repository.full_name == repo_full_name)
    )
    repository = result.scalar_one_or_none()

    if not repository:
        return []

    result = await db.execute(
        select(CertificationHistory)
        .where(CertificationHistory.repo_id == repository.id)
        .order_by(CertificationHistory.created_at.desc())
        .limit(limit)
    )
    history = result.scalars().all()

    return [
        {
            "id": str(h.id),
            "previous_tier": h.previous_tier,
            "new_tier": h.new_tier,
            "reason": h.reason,
            "created_at": h.created_at.isoformat(),
        }
        for h in history
    ]


@router.get("/leaderboard")
async def get_certification_leaderboard(
    tier: str | None = Query(default=None),
    limit: int = Query(default=50, le=100),
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """Get leaderboard of certified repositories."""
    query = (
        select(AttestationDB)
        .where(AttestationDB.tier.isnot(None))
        .where(AttestationDB.passed == True)
    )

    if tier:
        query = query.where(AttestationDB.tier == tier.lower())

    # Get latest attestation per repo
    from sqlalchemy import func, distinct

    subquery = (
        select(
            AttestationDB.repo_full_name,
            func.max(AttestationDB.created_at).label("max_created"),
        )
        .where(AttestationDB.tier.isnot(None))
        .group_by(AttestationDB.repo_full_name)
        .subquery()
    )

    query = (
        select(AttestationDB)
        .join(
            subquery,
            (AttestationDB.repo_full_name == subquery.c.repo_full_name)
            & (AttestationDB.created_at == subquery.c.max_created),
        )
        .order_by(
            # Order by tier rank (platinum first)
            AttestationDB.tier.desc(),
            AttestationDB.created_at.desc(),
        )
        .limit(limit)
    )

    result = await db.execute(query)
    attestations = result.scalars().all()

    return [
        {
            "rank": idx + 1,
            "repo_full_name": a.repo_full_name,
            "tier": a.tier,
            "verified_at": a.created_at.isoformat(),
            "expires_at": a.expires_at.isoformat() if a.expires_at else None,
        }
        for idx, a in enumerate(attestations)
    ]


# CI/CD Status API
class CIStatusResponse(BaseModel):
    """CI/CD compatible status response."""

    state: str  # success, failure, pending, error
    description: str
    target_url: str
    context: str = "codeverify/verification"


@router.get("/repos/{owner}/{repo}/status", response_model=CIStatusResponse)
async def get_ci_status(
    owner: str,
    repo: str,
    ref: str | None = Query(default=None, description="Git ref (branch/tag/sha)"),
    db: AsyncSession = Depends(get_db),
) -> CIStatusResponse:
    """
    Get CI/CD compatible verification status for a repository.
    
    Use this endpoint in CI/CD pipelines to check verification status.
    Returns GitHub-compatible commit status format.
    """
    repo_full_name = f"{owner}/{repo}"

    # Build query for latest attestation
    query = (
        select(AttestationDB)
        .where(AttestationDB.repo_full_name == repo_full_name)
    )

    if ref:
        query = query.where(
            (AttestationDB.ref == ref) | (AttestationDB.commit_sha == ref)
        )

    query = query.order_by(AttestationDB.created_at.desc()).limit(1)

    result = await db.execute(query)
    attestation = result.scalar_one_or_none()

    base_url = "https://codeverify.dev"

    if not attestation:
        return CIStatusResponse(
            state="pending",
            description="No verification found",
            target_url=f"{base_url}/repos/{owner}/{repo}",
            context="codeverify/verification",
        )

    # Check if expired
    if attestation.expires_at and datetime.utcnow() > attestation.expires_at:
        return CIStatusResponse(
            state="error",
            description="Verification expired",
            target_url=f"{base_url}/verify/attestation/{attestation.id}",
            context="codeverify/verification",
        )

    if attestation.passed:
        tier_str = f" ({attestation.tier})" if attestation.tier else ""
        return CIStatusResponse(
            state="success",
            description=f"Verification passed{tier_str}",
            target_url=f"{base_url}/verify/attestation/{attestation.id}",
            context="codeverify/verification",
        )
    else:
        return CIStatusResponse(
            state="failure",
            description="Verification failed",
            target_url=f"{base_url}/verify/attestation/{attestation.id}",
            context="codeverify/verification",
        )


# Embeddable Widget Endpoints
@router.get("/repos/{owner}/{repo}/widget")
async def get_badge_widget(
    owner: str,
    repo: str,
    style: str = Query(default="flat", pattern="^(flat|flat-square|plastic|for-the-badge|social)$"),
    format: str = Query(default="svg", pattern="^(svg|html|markdown|json)$"),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """
    Get an embeddable widget for a repository.
    
    Formats:
    - svg: SVG image (default)
    - html: HTML embed snippet
    - markdown: Markdown embed snippet
    - json: JSON data for custom widgets
    """
    repo_full_name = f"{owner}/{repo}"

    # Get latest attestation
    result = await db.execute(
        select(AttestationDB)
        .where(AttestationDB.repo_full_name == repo_full_name)
        .order_by(AttestationDB.created_at.desc())
        .limit(1)
    )
    attestation = result.scalar_one_or_none()

    base_url = "https://codeverify.dev"

    if not attestation:
        if format == "svg":
            svg = generate_badge_svg("CodeVerify", "not verified", "lightgrey", style)
            return Response(content=svg, media_type="image/svg+xml")
        elif format == "html":
            return Response(
                content=f'<a href="{base_url}/repos/{owner}/{repo}"><img src="{base_url}/api/v1/badges/repos/{owner}/{repo}/widget?format=svg" alt="CodeVerify" /></a>',
                media_type="text/html",
            )
        elif format == "markdown":
            return Response(
                content=f'[![CodeVerify]({base_url}/api/v1/badges/repos/{owner}/{repo}/widget?format=svg)]({base_url}/repos/{owner}/{repo})',
                media_type="text/plain",
            )
        else:
            return Response(
                content='{"verified": false, "tier": null, "passed": false}',
                media_type="application/json",
            )

    # Check expiration
    expired = attestation.expires_at and datetime.utcnow() > attestation.expires_at
    color = get_badge_color(attestation.tier if not expired else None, attestation.passed and not expired)
    message = get_badge_message(attestation.tier if not expired else None, attestation.passed and not expired)

    if expired:
        message = "expired"
        color = "lightgrey"

    verify_url = f"{base_url}/verify/attestation/{attestation.id}"
    widget_url = f"{base_url}/api/v1/badges/repos/{owner}/{repo}/widget?format=svg&style={style}"

    if format == "svg":
        svg = generate_badge_svg("CodeVerify", message, color, style)
        return Response(
            content=svg,
            media_type="image/svg+xml",
            headers={"Cache-Control": "max-age=300"},
        )
    elif format == "html":
        return Response(
            content=f'<a href="{verify_url}"><img src="{widget_url}" alt="CodeVerify: {message}" /></a>',
            media_type="text/html",
        )
    elif format == "markdown":
        return Response(
            content=f'[![CodeVerify: {message}]({widget_url})]({verify_url})',
            media_type="text/plain",
        )
    else:
        import json
        data = {
            "verified": attestation.passed and not expired,
            "tier": attestation.tier,
            "passed": attestation.passed,
            "expired": expired,
            "expires_at": attestation.expires_at.isoformat() if attestation.expires_at else None,
            "verified_at": attestation.created_at.isoformat(),
            "verify_url": verify_url,
            "badge_url": widget_url,
        }
        return Response(content=json.dumps(data), media_type="application/json")


# Package Registry Integrations
class PackageVerificationResponse(BaseModel):
    """Package verification status for registry integrations."""

    package_name: str
    registry: str  # npm, pypi, maven, nuget, crates
    version: str | None
    verified: bool
    tier: str | None
    verification_url: str | None
    badge_url: str
    last_verified: datetime | None


@router.get("/packages/{registry}/{package_name:path}")
async def get_package_verification(
    registry: str,
    package_name: str,
    version: str | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
) -> PackageVerificationResponse:
    """
    Get verification status for a package on npm, PyPI, or other registries.
    
    Supported registries: npm, pypi, maven, nuget, crates
    
    Package name formats:
    - npm: @scope/package or package
    - pypi: package-name
    - maven: group.id:artifact-id
    - nuget: Package.Name
    - crates: crate-name
    """
    registry = registry.lower()
    valid_registries = ["npm", "pypi", "maven", "nuget", "crates"]
    
    if registry not in valid_registries:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid registry. Supported: {', '.join(valid_registries)}",
        )

    base_url = "https://codeverify.dev"

    # Look up attestation by package metadata
    result = await db.execute(
        select(AttestationDB)
        .where(AttestationDB.metadata["package_name"].astext == package_name)
        .where(AttestationDB.metadata["registry"].astext == registry)
        .order_by(AttestationDB.created_at.desc())
        .limit(1)
    )
    attestation = result.scalar_one_or_none()

    if not attestation:
        return PackageVerificationResponse(
            package_name=package_name,
            registry=registry,
            version=version,
            verified=False,
            tier=None,
            verification_url=None,
            badge_url=f"{base_url}/api/v1/badges/packages/{registry}/{package_name}/badge.svg",
            last_verified=None,
        )

    expired = attestation.expires_at and datetime.utcnow() > attestation.expires_at

    return PackageVerificationResponse(
        package_name=package_name,
        registry=registry,
        version=attestation.metadata.get("version") if attestation.metadata else version,
        verified=attestation.passed and not expired,
        tier=attestation.tier if not expired else None,
        verification_url=f"{base_url}/verify/attestation/{attestation.id}",
        badge_url=f"{base_url}/api/v1/badges/packages/{registry}/{package_name}/badge.svg",
        last_verified=attestation.created_at,
    )


@router.get("/packages/{registry}/{package_name:path}/badge.svg")
async def get_package_badge(
    registry: str,
    package_name: str,
    style: str = Query(default="flat"),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Get verification badge SVG for a package."""
    registry = registry.lower()

    # Look up attestation
    result = await db.execute(
        select(AttestationDB)
        .where(AttestationDB.metadata["package_name"].astext == package_name)
        .where(AttestationDB.metadata["registry"].astext == registry)
        .order_by(AttestationDB.created_at.desc())
        .limit(1)
    )
    attestation = result.scalar_one_or_none()

    if not attestation:
        svg = generate_badge_svg("CodeVerify", "unverified", "lightgrey", style)
        return Response(content=svg, media_type="image/svg+xml")

    expired = attestation.expires_at and datetime.utcnow() > attestation.expires_at
    color = get_badge_color(attestation.tier if not expired else None, attestation.passed and not expired)
    message = get_badge_message(attestation.tier if not expired else None, attestation.passed and not expired)

    if expired:
        message = "expired"
        color = "lightgrey"

    svg = generate_badge_svg("CodeVerify", message, color, style)
    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={"Cache-Control": "max-age=300"},
    )


@router.post("/packages/{registry}/{package_name:path}/attest")
async def create_package_attestation(
    registry: str,
    package_name: str,
    version: str = Query(...),
    repo_full_name: str = Query(...),
    analysis_id: UUID | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: dict | None = Depends(get_current_user_optional),
) -> AttestationResponse:
    """
    Create a verification attestation for a package.
    
    Links a package version to a repository verification.
    """
    # Create attestation with package metadata
    request = CreateAttestationRequest(
        repo_full_name=repo_full_name,
        attestation_type=AttestationType.RELEASE,
        analysis_id=analysis_id,
    )

    # Add package metadata
    attestation_response = await create_verification_attestation(
        request=request,
        db=db,
        current_user=current_user,
    )

    # Update metadata with package info
    result = await db.execute(
        select(AttestationDB).where(AttestationDB.id == attestation_response.id)
    )
    attestation = result.scalar_one_or_none()
    if attestation:
        attestation.metadata = {
            **attestation.metadata,
            "package_name": package_name,
            "registry": registry.lower(),
            "version": version,
        }
        await db.commit()

    return attestation_response
