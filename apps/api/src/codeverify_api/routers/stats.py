"""Stats router for dashboard metrics."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from codeverify_api.auth.dependencies import get_current_user
from codeverify_api.db.database import get_db
from codeverify_api.db.models import Analysis, Finding, Organization, Repository, User

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("/dashboard")
async def get_dashboard_stats(
    organization_id: UUID | None = Query(None, description="Filter by organization"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Get dashboard statistics."""
    # Base query filters
    repo_filter = []
    if organization_id:
        repo_filter.append(Repository.organization_id == organization_id)
    
    # Get repository IDs for filtering
    repo_query = select(Repository.id)
    if repo_filter:
        repo_query = repo_query.where(*repo_filter)
    
    # Total analyses
    total_query = select(func.count(Analysis.id))
    if repo_filter:
        total_query = total_query.join(Repository).where(*repo_filter)
    total_result = await db.execute(total_query)
    total_analyses = total_result.scalar() or 0
    
    # Passed analyses
    passed_query = select(func.count(Analysis.id)).where(
        Analysis.conclusion == "passed"
    )
    if repo_filter:
        passed_query = passed_query.join(Repository).where(*repo_filter)
    passed_result = await db.execute(passed_query)
    passed = passed_result.scalar() or 0
    
    # Failed analyses
    failed_query = select(func.count(Analysis.id)).where(
        Analysis.conclusion == "failed"
    )
    if repo_filter:
        failed_query = failed_query.join(Repository).where(*repo_filter)
    failed_result = await db.execute(failed_query)
    failed = failed_result.scalar() or 0
    
    # Total findings
    findings_query = select(func.count(Finding.id))
    findings_result = await db.execute(findings_query)
    total_findings = findings_result.scalar() or 0
    
    # Findings by severity
    by_severity = {}
    for severity in ["critical", "high", "medium", "low"]:
        severity_query = select(func.count(Finding.id)).where(
            Finding.severity == severity
        )
        severity_result = await db.execute(severity_query)
        by_severity[severity] = severity_result.scalar() or 0
    
    # Recent activity (last 10 completed analyses)
    recent_query = (
        select(Analysis)
        .where(Analysis.status == "completed")
        .order_by(Analysis.completed_at.desc())
        .limit(10)
    )
    if repo_filter:
        recent_query = recent_query.join(Repository).where(*repo_filter)
    recent_result = await db.execute(recent_query)
    recent_analyses = recent_result.scalars().all()
    
    recent_activity = [
        {
            "id": str(a.id),
            "repository_id": str(a.repository_id),
            "pr_number": a.pr_number,
            "conclusion": a.conclusion,
            "completed_at": a.completed_at.isoformat() if a.completed_at else None,
        }
        for a in recent_analyses
    ]
    
    # Calculate trends
    now = datetime.utcnow()
    seven_days_ago = now - timedelta(days=7)
    thirty_days_ago = now - timedelta(days=30)
    
    # Pass rate last 7 days
    rate_7d_total = await db.execute(
        select(func.count(Analysis.id)).where(
            Analysis.completed_at >= seven_days_ago,
            Analysis.status == "completed",
        )
    )
    rate_7d_passed = await db.execute(
        select(func.count(Analysis.id)).where(
            Analysis.completed_at >= seven_days_ago,
            Analysis.conclusion == "passed",
        )
    )
    total_7d = rate_7d_total.scalar() or 0
    passed_7d = rate_7d_passed.scalar() or 0
    pass_rate_7d = passed_7d / total_7d if total_7d > 0 else 0
    
    # Pass rate last 30 days
    rate_30d_total = await db.execute(
        select(func.count(Analysis.id)).where(
            Analysis.completed_at >= thirty_days_ago,
            Analysis.status == "completed",
        )
    )
    rate_30d_passed = await db.execute(
        select(func.count(Analysis.id)).where(
            Analysis.completed_at >= thirty_days_ago,
            Analysis.conclusion == "passed",
        )
    )
    total_30d = rate_30d_total.scalar() or 0
    passed_30d = rate_30d_passed.scalar() or 0
    pass_rate_30d = passed_30d / total_30d if total_30d > 0 else 0
    
    # Average issues per PR
    avg_issues_query = select(func.avg(
        func.coalesce(Analysis.summary["total_issues"].astext.cast(func.INTEGER), 0)
    )).where(Analysis.status == "completed")
    avg_result = await db.execute(avg_issues_query)
    avg_issues = avg_result.scalar() or 0
    
    return {
        "total_analyses": total_analyses,
        "passed": passed,
        "failed": failed,
        "total_findings": total_findings,
        "by_severity": by_severity,
        "recent_activity": recent_activity,
        "trends": {
            "pass_rate_7d": round(pass_rate_7d, 3),
            "pass_rate_30d": round(pass_rate_30d, 3),
            "avg_issues_per_pr": round(float(avg_issues), 1),
        },
    }


@router.get("/repository/{repository_id}")
async def get_repository_stats(
    repository_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Get statistics for a specific repository."""
    # Total analyses for this repo
    total_result = await db.execute(
        select(func.count(Analysis.id)).where(
            Analysis.repository_id == repository_id
        )
    )
    total = total_result.scalar() or 0
    
    # Passed/failed
    passed_result = await db.execute(
        select(func.count(Analysis.id)).where(
            Analysis.repository_id == repository_id,
            Analysis.conclusion == "passed",
        )
    )
    passed = passed_result.scalar() or 0
    
    failed_result = await db.execute(
        select(func.count(Analysis.id)).where(
            Analysis.repository_id == repository_id,
            Analysis.conclusion == "failed",
        )
    )
    failed = failed_result.scalar() or 0
    
    # Findings by category
    category_result = await db.execute(
        select(Finding.category, func.count(Finding.id))
        .join(Analysis)
        .where(Analysis.repository_id == repository_id)
        .group_by(Finding.category)
    )
    by_category = {row[0]: row[1] for row in category_result.all()}
    
    # Recent analyses
    recent_result = await db.execute(
        select(Analysis)
        .where(Analysis.repository_id == repository_id)
        .order_by(Analysis.created_at.desc())
        .limit(20)
    )
    recent = recent_result.scalars().all()
    
    return {
        "repository_id": str(repository_id),
        "total_analyses": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / total if total > 0 else 0,
        "by_category": by_category,
        "recent_analyses": [
            {
                "id": str(a.id),
                "pr_number": a.pr_number,
                "status": a.status,
                "conclusion": a.conclusion,
                "created_at": a.created_at.isoformat(),
            }
            for a in recent
        ],
    }
