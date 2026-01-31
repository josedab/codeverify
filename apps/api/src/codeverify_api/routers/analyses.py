"""Analyses API endpoints."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from codeverify_api.db.database import get_db
from codeverify_api.db.models import Analysis, Finding, AnalysisStage
from codeverify_api.db.repositories import AnalysisRepository

router = APIRouter()


class DismissRequest(BaseModel):
    """Request body for dismissing a finding."""
    reason: str | None = None


class AnalysisResponse(BaseModel):
    """Response model for analysis."""
    id: str
    repo_id: str
    pr_number: int
    pr_title: str | None
    head_sha: str
    base_sha: str | None
    status: str
    started_at: str | None
    completed_at: str | None
    findings_count: int
    created_at: str

    class Config:
        from_attributes = True


@router.get("")
async def list_analyses(
    repo_id: UUID | None = None,
    pr_number: int | None = None,
    analysis_status: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """List analyses with optional filters."""
    query = select(Analysis)
    count_query = select(func.count(Analysis.id))

    if repo_id:
        query = query.where(Analysis.repo_id == repo_id)
        count_query = count_query.where(Analysis.repo_id == repo_id)
    if pr_number:
        query = query.where(Analysis.pr_number == pr_number)
        count_query = count_query.where(Analysis.pr_number == pr_number)
    if analysis_status:
        query = query.where(Analysis.status == analysis_status)
        count_query = count_query.where(Analysis.status == analysis_status)

    query = query.order_by(Analysis.created_at.desc()).offset(offset).limit(limit)

    result = await db.execute(query)
    analyses = result.scalars().all()

    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    return {
        "analyses": [
            {
                "id": str(a.id),
                "repo_id": str(a.repo_id),
                "pr_number": a.pr_number,
                "pr_title": a.pr_title,
                "head_sha": a.head_sha,
                "base_sha": a.base_sha,
                "status": a.status,
                "started_at": a.started_at.isoformat() if a.started_at else None,
                "completed_at": a.completed_at.isoformat() if a.completed_at else None,
                "created_at": a.created_at.isoformat(),
            }
            for a in analyses
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/{analysis_id}")
async def get_analysis(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get analysis details including findings."""
    repo = AnalysisRepository(db)
    analysis = await repo.get_with_findings(analysis_id)

    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    return {
        "id": str(analysis.id),
        "repo_id": str(analysis.repo_id),
        "pr_number": analysis.pr_number,
        "pr_title": analysis.pr_title,
        "head_sha": analysis.head_sha,
        "base_sha": analysis.base_sha,
        "status": analysis.status,
        "started_at": analysis.started_at.isoformat() if analysis.started_at else None,
        "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None,
        "error_message": analysis.error_message,
        "metadata": analysis.metadata,
        "created_at": analysis.created_at.isoformat(),
        "findings": [
            {
                "id": str(f.id),
                "category": f.category,
                "severity": f.severity,
                "title": f.title,
                "description": f.description,
                "file_path": f.file_path,
                "line_start": f.line_start,
                "line_end": f.line_end,
                "code_snippet": f.code_snippet,
                "fix_suggestion": f.fix_suggestion,
                "confidence": f.confidence,
                "verification_type": f.verification_type,
                "dismissed": f.dismissed,
            }
            for f in analysis.findings
        ],
        "stages": [
            {
                "id": str(s.id),
                "stage_name": s.stage_name,
                "status": s.status,
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                "duration_ms": s.duration_ms,
                "error_message": s.error_message,
            }
            for s in analysis.stages
        ],
        "summary": {
            "total_findings": len(analysis.findings),
            "critical": sum(1 for f in analysis.findings if f.severity == "critical"),
            "high": sum(1 for f in analysis.findings if f.severity == "high"),
            "medium": sum(1 for f in analysis.findings if f.severity == "medium"),
            "low": sum(1 for f in analysis.findings if f.severity == "low"),
        },
    }


@router.get("/{analysis_id}/findings")
async def get_analysis_findings(
    analysis_id: UUID,
    category: str | None = None,
    severity: str | None = None,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get findings for an analysis."""
    repo = AnalysisRepository(db)

    # Verify analysis exists
    analysis = await repo.get(analysis_id)
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    # Build query
    query = select(Finding).where(Finding.analysis_id == analysis_id)
    count_query = select(func.count(Finding.id)).where(Finding.analysis_id == analysis_id)

    if category:
        query = query.where(Finding.category == category)
        count_query = count_query.where(Finding.category == category)
    if severity:
        query = query.where(Finding.severity == severity)
        count_query = count_query.where(Finding.severity == severity)

    query = query.offset(offset).limit(limit)

    result = await db.execute(query)
    findings = result.scalars().all()

    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    return {
        "findings": [
            {
                "id": str(f.id),
                "category": f.category,
                "severity": f.severity,
                "title": f.title,
                "description": f.description,
                "file_path": f.file_path,
                "line_start": f.line_start,
                "line_end": f.line_end,
                "code_snippet": f.code_snippet,
                "fix_suggestion": f.fix_suggestion,
                "fix_diff": f.fix_diff,
                "confidence": f.confidence,
                "verification_type": f.verification_type,
                "verification_proof": f.verification_proof,
                "metadata": f.metadata,
                "dismissed": f.dismissed,
                "dismissed_reason": f.dismissed_reason,
                "created_at": f.created_at.isoformat(),
            }
            for f in findings
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.post("/{analysis_id}/findings/{finding_id}/dismiss")
async def dismiss_finding(
    analysis_id: UUID,
    finding_id: UUID,
    request: DismissRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Dismiss a finding."""
    # Verify finding exists and belongs to analysis
    query = select(Finding).where(
        Finding.id == finding_id,
        Finding.analysis_id == analysis_id,
    )
    result = await db.execute(query)
    finding = result.scalar_one_or_none()

    if not finding:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Finding not found")

    finding.dismissed = True
    finding.dismissed_reason = request.reason
    await db.flush()
    await db.refresh(finding)

    return {
        "finding_id": str(finding_id),
        "dismissed": True,
        "reason": request.reason,
    }


@router.post("/{analysis_id}/rerun")
async def rerun_analysis(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Re-run an analysis."""
    repo = AnalysisRepository(db)
    original = await repo.get(analysis_id)

    if not original:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    # Create new analysis with same parameters
    new_analysis = await repo.create_analysis(
        repo_id=original.repo_id,
        pr_number=original.pr_number,
        pr_title=original.pr_title,
        head_sha=original.head_sha,
        base_sha=original.base_sha,
    )

    return {
        "original_analysis_id": str(analysis_id),
        "new_analysis_id": str(new_analysis.id),
        "status": "queued",
    }


@router.get("/{analysis_id}/stages")
async def get_analysis_stages(
    analysis_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> list[dict[str, Any]]:
    """Get detailed stage information for an analysis."""
    # Verify analysis exists
    analysis = await db.get(Analysis, analysis_id)
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")

    query = select(AnalysisStage).where(AnalysisStage.analysis_id == analysis_id)
    result = await db.execute(query)
    stages = result.scalars().all()

    return [
        {
            "id": str(s.id),
            "stage_name": s.stage_name,
            "status": s.status,
            "started_at": s.started_at.isoformat() if s.started_at else None,
            "completed_at": s.completed_at.isoformat() if s.completed_at else None,
            "duration_ms": s.duration_ms,
            "result": s.result,
            "error_message": s.error_message,
            "created_at": s.created_at.isoformat(),
        }
        for s in stages
    ]
