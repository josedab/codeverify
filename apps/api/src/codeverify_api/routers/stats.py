"""Stats router for dashboard metrics."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select, and_, case
from sqlalchemy.ext.asyncio import AsyncSession

from codeverify_api.auth.dependencies import get_current_user
from codeverify_api.db.database import get_db
from codeverify_api.db.models import Analysis, Finding, Organization, Repository, User, OrgMembership

router = APIRouter(prefix="/stats", tags=["stats"])


class TeamMemberStats(BaseModel):
    """Individual team member statistics."""
    user_id: str
    username: str
    avatar_url: str | None
    analyses_triggered: int
    findings_created: int
    findings_dismissed: int
    last_active: str | None


class TeamStatsResponse(BaseModel):
    """Team statistics response."""
    organization_id: str
    total_members: int
    active_members_7d: int
    active_members_30d: int
    members: list[TeamMemberStats]
    activity_by_day: list[dict[str, Any]]


class TrendDataPoint(BaseModel):
    """Single data point for trend charts."""
    date: str
    analyses: int
    passed: int
    failed: int
    findings: int


class TrendsResponse(BaseModel):
    """Trends response for charts."""
    period: str
    data: list[TrendDataPoint]
    summary: dict[str, Any]


class LeaderboardEntry(BaseModel):
    """Leaderboard entry."""
    rank: int
    user_id: str
    username: str
    avatar_url: str | None
    score: int
    metric: str


class LeaderboardResponse(BaseModel):
    """Leaderboard response."""
    period: str
    entries: list[LeaderboardEntry]


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


@router.get("/team", response_model=TeamStatsResponse)
async def get_team_stats(
    organization_id: UUID = Query(..., description="Organization ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TeamStatsResponse:
    """Get team-level statistics for an organization."""
    now = datetime.utcnow()
    seven_days_ago = now - timedelta(days=7)
    thirty_days_ago = now - timedelta(days=30)
    
    # Get all members
    members_query = (
        select(User, OrgMembership)
        .join(OrgMembership, User.id == OrgMembership.user_id)
        .where(OrgMembership.org_id == organization_id)
    )
    members_result = await db.execute(members_query)
    members = members_result.all()
    
    total_members = len(members)
    
    # Get repo IDs for this org
    repo_query = select(Repository.id).where(Repository.org_id == organization_id)
    repo_result = await db.execute(repo_query)
    repo_ids = [r[0] for r in repo_result.all()]
    
    member_stats = []
    active_7d = set()
    active_30d = set()
    
    for user, membership in members:
        # Analyses triggered by this user
        analyses_count = await db.execute(
            select(func.count(Analysis.id))
            .where(
                Analysis.triggered_by == user.id,
                Analysis.repo_id.in_(repo_ids) if repo_ids else True
            )
        )
        analyses_triggered = analyses_count.scalar() or 0
        
        # Findings dismissed by this user
        dismissed_count = await db.execute(
            select(func.count(Finding.id))
            .where(Finding.dismissed_by == user.id)
        )
        findings_dismissed = dismissed_count.scalar() or 0
        
        # Last activity
        last_analysis = await db.execute(
            select(Analysis.created_at)
            .where(Analysis.triggered_by == user.id)
            .order_by(Analysis.created_at.desc())
            .limit(1)
        )
        last_active_row = last_analysis.first()
        last_active = last_active_row[0] if last_active_row else None
        
        if last_active:
            if last_active >= seven_days_ago:
                active_7d.add(user.id)
            if last_active >= thirty_days_ago:
                active_30d.add(user.id)
        
        member_stats.append(TeamMemberStats(
            user_id=str(user.id),
            username=user.username,
            avatar_url=user.avatar_url,
            analyses_triggered=analyses_triggered,
            findings_created=0,  # Findings are created by system
            findings_dismissed=findings_dismissed,
            last_active=last_active.isoformat() if last_active else None,
        ))
    
    # Activity by day (last 30 days)
    activity_by_day = []
    for i in range(30):
        day = now - timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        day_count = await db.execute(
            select(func.count(Analysis.id))
            .where(
                Analysis.created_at >= day_start,
                Analysis.created_at < day_end,
                Analysis.repo_id.in_(repo_ids) if repo_ids else True
            )
        )
        
        activity_by_day.append({
            "date": day_start.strftime("%Y-%m-%d"),
            "count": day_count.scalar() or 0,
        })
    
    activity_by_day.reverse()
    
    return TeamStatsResponse(
        organization_id=str(organization_id),
        total_members=total_members,
        active_members_7d=len(active_7d),
        active_members_30d=len(active_30d),
        members=member_stats,
        activity_by_day=activity_by_day,
    )


@router.get("/trends", response_model=TrendsResponse)
async def get_trends(
    organization_id: UUID | None = Query(None, description="Filter by organization"),
    period: str = Query("30d", description="Time period: 7d, 30d, 90d"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> TrendsResponse:
    """Get trend data for charts."""
    period_days = {"7d": 7, "30d": 30, "90d": 90}.get(period, 30)
    now = datetime.utcnow()
    start_date = now - timedelta(days=period_days)
    
    # Get repo IDs for filtering
    repo_ids = None
    if organization_id:
        repo_query = select(Repository.id).where(Repository.org_id == organization_id)
        repo_result = await db.execute(repo_query)
        repo_ids = [r[0] for r in repo_result.all()]
    
    data_points = []
    total_analyses = 0
    total_passed = 0
    total_failed = 0
    total_findings = 0
    
    for i in range(period_days):
        day = now - timedelta(days=period_days - 1 - i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        # Base filter
        base_filter = [
            Analysis.created_at >= day_start,
            Analysis.created_at < day_end,
        ]
        if repo_ids is not None:
            base_filter.append(Analysis.repo_id.in_(repo_ids))
        
        # Analyses count
        analyses_count = await db.execute(
            select(func.count(Analysis.id)).where(and_(*base_filter))
        )
        analyses = analyses_count.scalar() or 0
        
        # Passed count
        passed_count = await db.execute(
            select(func.count(Analysis.id)).where(
                and_(*base_filter, Analysis.conclusion == "passed")
            )
        )
        passed = passed_count.scalar() or 0
        
        # Failed count
        failed_count = await db.execute(
            select(func.count(Analysis.id)).where(
                and_(*base_filter, Analysis.conclusion == "failed")
            )
        )
        failed = failed_count.scalar() or 0
        
        # Findings count
        findings_query = (
            select(func.count(Finding.id))
            .join(Analysis, Finding.analysis_id == Analysis.id)
            .where(and_(*base_filter))
        )
        findings_count = await db.execute(findings_query)
        findings = findings_count.scalar() or 0
        
        data_points.append(TrendDataPoint(
            date=day_start.strftime("%Y-%m-%d"),
            analyses=analyses,
            passed=passed,
            failed=failed,
            findings=findings,
        ))
        
        total_analyses += analyses
        total_passed += passed
        total_failed += failed
        total_findings += findings
    
    return TrendsResponse(
        period=period,
        data=data_points,
        summary={
            "total_analyses": total_analyses,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_findings": total_findings,
            "pass_rate": total_passed / total_analyses if total_analyses > 0 else 0,
            "avg_findings_per_analysis": total_findings / total_analyses if total_analyses > 0 else 0,
        },
    )


@router.get("/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(
    organization_id: UUID = Query(..., description="Organization ID"),
    metric: str = Query("analyses", description="Metric: analyses, findings_fixed, activity"),
    period: str = Query("30d", description="Time period: 7d, 30d, 90d, all"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> LeaderboardResponse:
    """Get leaderboard for gamification."""
    period_days = {"7d": 7, "30d": 30, "90d": 90, "all": 365 * 10}.get(period, 30)
    start_date = datetime.utcnow() - timedelta(days=period_days)
    
    # Get members of org
    members_query = (
        select(User)
        .join(OrgMembership, User.id == OrgMembership.user_id)
        .where(OrgMembership.org_id == organization_id)
    )
    members_result = await db.execute(members_query)
    members = members_result.scalars().all()
    
    # Get repo IDs for this org
    repo_query = select(Repository.id).where(Repository.org_id == organization_id)
    repo_result = await db.execute(repo_query)
    repo_ids = [r[0] for r in repo_result.all()]
    
    user_scores = []
    
    for user in members:
        score = 0
        
        if metric == "analyses":
            count = await db.execute(
                select(func.count(Analysis.id))
                .where(
                    Analysis.triggered_by == user.id,
                    Analysis.created_at >= start_date,
                    Analysis.repo_id.in_(repo_ids) if repo_ids else True
                )
            )
            score = count.scalar() or 0
            
        elif metric == "findings_fixed":
            count = await db.execute(
                select(func.count(Finding.id))
                .where(
                    Finding.dismissed_by == user.id,
                    Finding.dismissed == True,
                    Finding.created_at >= start_date
                )
            )
            score = count.scalar() or 0
            
        elif metric == "activity":
            # Composite score: analyses + findings reviewed
            analyses = await db.execute(
                select(func.count(Analysis.id))
                .where(
                    Analysis.triggered_by == user.id,
                    Analysis.created_at >= start_date,
                    Analysis.repo_id.in_(repo_ids) if repo_ids else True
                )
            )
            dismissed = await db.execute(
                select(func.count(Finding.id))
                .where(
                    Finding.dismissed_by == user.id,
                    Finding.created_at >= start_date
                )
            )
            score = (analyses.scalar() or 0) * 10 + (dismissed.scalar() or 0) * 5
        
        user_scores.append({
            "user": user,
            "score": score,
        })
    
    # Sort and rank
    user_scores.sort(key=lambda x: x["score"], reverse=True)
    
    entries = [
        LeaderboardEntry(
            rank=i + 1,
            user_id=str(item["user"].id),
            username=item["user"].username,
            avatar_url=item["user"].avatar_url,
            score=item["score"],
            metric=metric,
        )
        for i, item in enumerate(user_scores[:20])
    ]
    
    return LeaderboardResponse(
        period=period,
        entries=entries,
    )
