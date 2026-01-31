"""Trust Score API endpoints."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from codeverify_api.db.database import get_db
from codeverify_api.db.models import TrustScoreCache, Repository

router = APIRouter()


class TrustScoreRequest(BaseModel):
    """Request to calculate trust score for code."""

    code: str = Field(..., description="The code to analyze")
    file_path: str = Field(default="unknown", description="Path to the file")
    language: str = Field(default="python", description="Programming language")
    author: str | None = Field(default=None, description="Code author for historical lookup")
    include_verification: bool = Field(
        default=False,
        description="Run formal verification for more accurate score",
    )


class TrustScoreFactorsResponse(BaseModel):
    """Trust score factors breakdown."""

    complexity_score: float
    pattern_confidence: float
    historical_accuracy: float
    verification_coverage: float
    code_quality_signals: float
    ai_detection_confidence: float


class TrustScoreResponse(BaseModel):
    """Trust score calculation response."""

    score: float = Field(..., ge=0.0, le=100.0, description="Trust score (0-100)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the score")
    risk_level: str = Field(..., description="Risk level: low, medium, high, critical")
    factors: TrustScoreFactorsResponse
    recommendations: list[str]
    is_ai_generated: bool
    code_hash: str


class TrustScoreFeedbackRequest(BaseModel):
    """Feedback on a trust score prediction."""

    code_hash: str
    was_accurate: bool = Field(..., description="Was the trust score prediction accurate?")
    actual_outcome: str | None = Field(
        default=None,
        description="Description of actual outcome (bug found, false positive, etc.)",
    )
    author: str | None = None
    file_path: str | None = None


class TrustScoreBatchRequest(BaseModel):
    """Request to calculate trust scores for multiple files."""

    files: list[TrustScoreRequest]


class TrustScoreBatchResponse(BaseModel):
    """Response with trust scores for multiple files."""

    scores: dict[str, TrustScoreResponse]
    overall_score: float
    overall_risk_level: str
    high_risk_files: list[str]


@router.post("", response_model=TrustScoreResponse)
async def calculate_trust_score(request: TrustScoreRequest) -> TrustScoreResponse:
    """
    Calculate trust score for a code snippet.

    The trust score indicates how reliable the code is likely to be,
    with special attention to AI-generated code patterns.
    """
    from codeverify_agents.trust_score import TrustScoreAgent, calculate_code_hash

    agent = TrustScoreAgent()

    context = {
        "file_path": request.file_path,
        "language": request.language,
        "author": request.author,
    }

    result = await agent.analyze(request.code, context)

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trust score calculation failed: {result.error}",
        )

    data = result.data
    code_hash = calculate_code_hash(request.code)

    return TrustScoreResponse(
        score=data["score"],
        confidence=data["confidence"],
        risk_level=data["risk_level"],
        factors=TrustScoreFactorsResponse(**data["factors"]),
        recommendations=data["recommendations"],
        is_ai_generated=data["is_ai_generated"],
        code_hash=code_hash,
    )


@router.post("/batch", response_model=TrustScoreBatchResponse)
async def calculate_trust_scores_batch(
    request: TrustScoreBatchRequest,
) -> TrustScoreBatchResponse:
    """Calculate trust scores for multiple code files."""
    from codeverify_agents.trust_score import TrustScoreAgent, calculate_code_hash

    agent = TrustScoreAgent()
    scores: dict[str, TrustScoreResponse] = {}
    total_score = 0.0
    high_risk_files: list[str] = []

    for file_request in request.files:
        context = {
            "file_path": file_request.file_path,
            "language": file_request.language,
            "author": file_request.author,
        }

        result = await agent.analyze(file_request.code, context)

        if result.success:
            data = result.data
            code_hash = calculate_code_hash(file_request.code)

            score_response = TrustScoreResponse(
                score=data["score"],
                confidence=data["confidence"],
                risk_level=data["risk_level"],
                factors=TrustScoreFactorsResponse(**data["factors"]),
                recommendations=data["recommendations"],
                is_ai_generated=data["is_ai_generated"],
                code_hash=code_hash,
            )

            scores[file_request.file_path] = score_response
            total_score += data["score"]

            if data["risk_level"] in ("high", "critical"):
                high_risk_files.append(file_request.file_path)

    # Calculate overall metrics
    overall_score = total_score / len(scores) if scores else 0.0

    if overall_score >= 80:
        overall_risk_level = "low"
    elif overall_score >= 60:
        overall_risk_level = "medium"
    elif overall_score >= 40:
        overall_risk_level = "high"
    else:
        overall_risk_level = "critical"

    return TrustScoreBatchResponse(
        scores=scores,
        overall_score=overall_score,
        overall_risk_level=overall_risk_level,
        high_risk_files=high_risk_files,
    )


@router.post("/feedback")
async def submit_trust_score_feedback(
    request: TrustScoreFeedbackRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Submit feedback on a trust score prediction.

    This helps improve future predictions by updating historical accuracy data.
    """
    # Store feedback for model improvement
    # In production, this would update a feedback table and trigger model retraining
    return {
        "status": "received",
        "code_hash": request.code_hash,
        "was_accurate": request.was_accurate,
        "message": "Feedback recorded. Thank you for helping improve trust score accuracy.",
    }


@router.get("/history/{file_path:path}")
async def get_trust_score_history(
    file_path: str,
    repo_id: UUID | None = None,
    limit: int = Query(default=10, le=100),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get historical trust scores for a file."""
    query = (
        select(TrustScoreCache)
        .where(TrustScoreCache.file_path == file_path)
        .order_by(desc(TrustScoreCache.computed_at))
        .limit(limit)
    )

    if repo_id:
        query = query.where(TrustScoreCache.repo_id == repo_id)

    result = await db.execute(query)
    scores = result.scalars().all()

    if not scores:
        return {
            "file_path": file_path,
            "history": [],
            "average_score": None,
            "trend": None,
        }

    history = [
        {
            "score": s.score,
            "risk_level": s.risk_level,
            "ai_probability": s.ai_probability,
            "commit_sha": s.commit_sha,
            "computed_at": s.computed_at.isoformat(),
        }
        for s in scores
    ]

    avg_score = sum(s.score for s in scores) / len(scores)

    # Calculate trend
    trend = None
    if len(scores) >= 2:
        recent = sum(s.score for s in scores[: len(scores) // 2]) / (len(scores) // 2)
        older = sum(s.score for s in scores[len(scores) // 2 :]) / (len(scores) - len(scores) // 2)
        if recent > older + 5:
            trend = "improving"
        elif recent < older - 5:
            trend = "declining"
        else:
            trend = "stable"

    return {
        "file_path": file_path,
        "history": history,
        "average_score": round(avg_score, 1),
        "trend": trend,
    }


@router.get("/stats")
async def get_trust_score_stats(
    repo_id: UUID | None = None,
    days: int = Query(default=30, le=90),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get trust score statistics for a repository or organization."""
    from datetime import timedelta, datetime

    cutoff = datetime.utcnow() - timedelta(days=days)

    base_filter = TrustScoreCache.computed_at >= cutoff
    if repo_id:
        base_filter = base_filter & (TrustScoreCache.repo_id == repo_id)

    # Total scores
    total_query = select(func.count(TrustScoreCache.id)).where(base_filter)
    total_result = await db.execute(total_query)
    total = total_result.scalar() or 0

    if total == 0:
        return {
            "period_days": days,
            "total_scores_calculated": 0,
            "average_score": 0.0,
            "ai_generated_percentage": 0.0,
            "risk_distribution": {"low": 0, "medium": 0, "high": 0, "critical": 0},
            "accuracy_rate": 0.0,
        }

    # Average score
    avg_query = select(func.avg(TrustScoreCache.score)).where(base_filter)
    avg_result = await db.execute(avg_query)
    avg_score = avg_result.scalar() or 0.0

    # AI generated percentage (using ai_probability > 0.5 as threshold)
    ai_query = select(func.count(TrustScoreCache.id)).where(
        base_filter & (TrustScoreCache.ai_probability > 0.5)
    )
    ai_result = await db.execute(ai_query)
    ai_count = ai_result.scalar() or 0

    # Risk distribution
    risk_dist = {}
    for risk in ["low", "medium", "high", "critical"]:
        risk_query = select(func.count(TrustScoreCache.id)).where(
            base_filter & (TrustScoreCache.risk_level == risk)
        )
        risk_result = await db.execute(risk_query)
        risk_dist[risk] = risk_result.scalar() or 0

    return {
        "period_days": days,
        "total_scores_calculated": total,
        "average_score": round(float(avg_score), 1),
        "ai_generated_percentage": round((ai_count / total) * 100, 1) if total > 0 else 0.0,
        "risk_distribution": risk_dist,
        "accuracy_rate": 0.0,  # Would need feedback data to calculate
    }
