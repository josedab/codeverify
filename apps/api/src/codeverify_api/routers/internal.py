"""Internal API endpoints for worker communication."""

import os
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from codeverify_api.db.database import get_db
from codeverify_api.db.models import Analysis, Finding, AnalysisStage, Repository
from codeverify_api.db.repositories import AnalysisRepository, RepositoryRepository

router = APIRouter()


class FindingData(BaseModel):
    """Finding data from worker."""
    category: str
    severity: str
    title: str
    description: str | None = None
    file_path: str
    line_start: int | None = None
    line_end: int | None = None
    code_snippet: str | None = None
    fix_suggestion: str | None = None
    fix_diff: str | None = None
    confidence: float | None = None
    verification_type: str | None = None
    verification_proof: str | None = None
    metadata: dict[str, Any] | None = None


class StageData(BaseModel):
    """Stage data from worker."""
    stage_name: str
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: int | None = None
    result: dict[str, Any] | None = None
    error_message: str | None = None


class AnalysisData(BaseModel):
    """Analysis data from worker."""
    repo_id: int  # GitHub repo ID
    repo_full_name: str
    pr_number: int
    pr_title: str | None = None
    head_sha: str
    base_sha: str | None = None
    status: str
    started_at: str
    completed_at: str
    error_message: str | None = None
    findings: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}


async def verify_internal_key(
    authorization: str = Header(...),
) -> bool:
    """Verify internal API key for worker authentication."""
    expected_key = os.environ.get("INTERNAL_API_KEY", "")
    
    if not expected_key:
        # If no key configured, allow in development
        if os.environ.get("ENVIRONMENT", "development") == "development":
            return True
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Internal API not configured",
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header",
        )
    
    token = authorization[7:]  # Remove "Bearer " prefix
    
    if token != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal API key",
        )
    
    return True


@router.post("/analyses", status_code=status.HTTP_201_CREATED)
async def store_analysis(
    data: AnalysisData,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_internal_key),
) -> dict[str, Any]:
    """Store analysis results from worker.
    
    This endpoint is called by the worker to persist analysis results
    after completing a PR analysis.
    """
    # Get or create repository by GitHub ID
    repo_repo = RepositoryRepository(db)
    repo = await repo_repo.get_by_github_id(data.repo_id)
    
    if not repo:
        # Create repository if it doesn't exist
        owner, name = data.repo_full_name.split("/")
        repo = await repo_repo.create(
            github_id=data.repo_id,
            name=name,
            full_name=data.repo_full_name,
            private=False,  # Default, will be updated on next webhook
        )
    
    # Create analysis
    analysis_repo = AnalysisRepository(db)
    
    # Parse timestamps
    started_at = datetime.fromisoformat(data.started_at.replace("Z", "+00:00"))
    completed_at = datetime.fromisoformat(data.completed_at.replace("Z", "+00:00"))
    
    analysis = Analysis(
        repo_id=repo.id,
        pr_number=data.pr_number,
        pr_title=data.pr_title,
        head_sha=data.head_sha,
        base_sha=data.base_sha,
        status=data.status,
        started_at=started_at,
        completed_at=completed_at,
        error_message=data.error_message,
        metadata=data.summary,
    )
    db.add(analysis)
    await db.flush()
    await db.refresh(analysis)
    
    # Add findings
    for finding_data in data.findings:
        finding = Finding(
            analysis_id=analysis.id,
            category=finding_data.get("category", "unknown"),
            severity=finding_data.get("severity", "low"),
            title=finding_data.get("title", "Finding"),
            description=finding_data.get("description"),
            file_path=finding_data.get("file_path", ""),
            line_start=finding_data.get("line_start"),
            line_end=finding_data.get("line_end"),
            code_snippet=finding_data.get("code_snippet"),
            fix_suggestion=finding_data.get("fix_suggestion"),
            fix_diff=finding_data.get("fix_diff"),
            confidence=finding_data.get("confidence"),
            verification_type=finding_data.get("verification_type"),
            verification_proof=finding_data.get("verification_proof"),
            metadata=finding_data.get("metadata", {}),
        )
        db.add(finding)
    
    # Add stages
    for stage_data in data.stages:
        stage_started = None
        stage_completed = None
        
        if stage_data.get("started_at"):
            stage_started = datetime.fromisoformat(
                stage_data["started_at"].replace("Z", "+00:00")
            )
        if stage_data.get("completed_at"):
            stage_completed = datetime.fromisoformat(
                stage_data["completed_at"].replace("Z", "+00:00")
            )
        
        stage = AnalysisStage(
            analysis_id=analysis.id,
            stage_name=stage_data.get("stage_name", "unknown"),
            status=stage_data.get("status", "completed"),
            started_at=stage_started,
            completed_at=stage_completed,
            duration_ms=stage_data.get("duration_ms"),
            result=stage_data.get("result"),
            error_message=stage_data.get("error_message"),
        )
        db.add(stage)
    
    await db.flush()
    
    return {
        "id": str(analysis.id),
        "repo_id": str(repo.id),
        "pr_number": analysis.pr_number,
        "status": analysis.status,
        "findings_count": len(data.findings),
        "stages_count": len(data.stages),
        "created_at": analysis.created_at.isoformat(),
    }


@router.patch("/analyses/{analysis_id}/status")
async def update_analysis_status(
    analysis_id: UUID,
    status_value: str,
    error_message: str | None = None,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_internal_key),
) -> dict[str, Any]:
    """Update analysis status from worker."""
    analysis = await db.get(Analysis, analysis_id)
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found",
        )
    
    analysis.status = status_value
    if error_message:
        analysis.error_message = error_message
    
    if status_value == "running" and not analysis.started_at:
        analysis.started_at = datetime.utcnow()
    elif status_value in ("completed", "failed"):
        analysis.completed_at = datetime.utcnow()
    
    await db.flush()
    await db.refresh(analysis)
    
    return {
        "id": str(analysis.id),
        "status": analysis.status,
        "updated_at": analysis.updated_at.isoformat(),
    }
