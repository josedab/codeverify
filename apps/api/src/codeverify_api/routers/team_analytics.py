"""Team Analytics API router.

Provides org-wide metrics: defect density trends, verification coverage
heatmaps, top recurring bug patterns, team leaderboard, and export endpoints.
"""

import csv
import io
import random
import uuid
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class DefectDensityTrend(BaseModel):
    date: str
    total_findings: int
    critical: int
    high: int
    medium: int
    low: int
    defect_density: float = Field(description="Findings per 1K lines of code")


class CoverageHeatmapEntry(BaseModel):
    directory: str
    files_total: int
    files_verified: int
    coverage_pct: float
    findings_count: int
    risk_level: str


class BugPattern(BaseModel):
    pattern_id: str
    name: str
    category: str
    occurrences: int
    trend: str = Field(description="increasing, decreasing, stable")
    affected_repos: list[str]
    first_seen: str
    last_seen: str


class TeamMember(BaseModel):
    username: str
    avatar_url: str | None = None
    analyses_triggered: int
    findings_resolved: int
    fixes_applied: int
    verification_score: float


class AnalyticsSummary(BaseModel):
    org_id: str
    period: str
    total_analyses: int
    total_findings: int
    total_fixes_applied: int
    pass_rate: float
    avg_findings_per_analysis: float
    avg_resolution_time_hours: float
    verification_coverage_pct: float


class SlackSummaryConfig(BaseModel):
    org_id: str
    webhook_url: str
    channel: str = "#codeverify"
    schedule: str = Field(default="weekly", description="weekly or daily")
    enabled: bool = True


# ---------------------------------------------------------------------------
# Demo data generators
# ---------------------------------------------------------------------------

def _generate_defect_trends(days: int = 30) -> list[DefectDensityTrend]:
    trends = []
    base = datetime.utcnow() - timedelta(days=days)
    for i in range(days):
        date = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        critical = random.randint(0, 3)
        high = random.randint(1, 8)
        medium = random.randint(5, 20)
        low = random.randint(10, 30)
        total = critical + high + medium + low
        trends.append(DefectDensityTrend(
            date=date,
            total_findings=total,
            critical=critical,
            high=high,
            medium=medium,
            low=low,
            defect_density=round(total / 10.0, 2),
        ))
    return trends


def _generate_coverage_heatmap() -> list[CoverageHeatmapEntry]:
    dirs = [
        ("src/api", 24, 22, 5),
        ("src/auth", 8, 8, 2),
        ("src/models", 15, 12, 3),
        ("src/services", 18, 14, 8),
        ("src/utils", 12, 6, 1),
        ("src/workers", 10, 7, 4),
        ("tests/", 30, 5, 0),
        ("scripts/", 6, 1, 0),
    ]
    return [
        CoverageHeatmapEntry(
            directory=d,
            files_total=total,
            files_verified=verified,
            coverage_pct=round(verified / max(total, 1) * 100, 1),
            findings_count=findings,
            risk_level="high" if verified / max(total, 1) < 0.5 else "medium" if verified / max(total, 1) < 0.8 else "low",
        )
        for d, total, verified, findings in dirs
    ]


def _generate_bug_patterns() -> list[BugPattern]:
    return [
        BugPattern(
            pattern_id="bp-1", name="Null pointer dereference", category="null_safety",
            occurrences=45, trend="decreasing", affected_repos=["api-service", "web-app"],
            first_seen="2025-11-01", last_seen="2026-02-15",
        ),
        BugPattern(
            pattern_id="bp-2", name="SQL injection risk", category="injection",
            occurrences=12, trend="decreasing", affected_repos=["api-service"],
            first_seen="2025-12-10", last_seen="2026-01-20",
        ),
        BugPattern(
            pattern_id="bp-3", name="Unhandled promise rejection", category="async_safety",
            occurrences=28, trend="stable", affected_repos=["web-app", "mobile-app"],
            first_seen="2026-01-01", last_seen="2026-02-25",
        ),
        BugPattern(
            pattern_id="bp-4", name="Hardcoded credentials", category="secrets",
            occurrences=6, trend="increasing", affected_repos=["data-pipeline"],
            first_seen="2026-02-01", last_seen="2026-02-27",
        ),
        BugPattern(
            pattern_id="bp-5", name="Array index out of bounds", category="bounds_check",
            occurrences=18, trend="decreasing", affected_repos=["api-service", "data-pipeline"],
            first_seen="2025-10-15", last_seen="2026-02-20",
        ),
    ]


def _generate_leaderboard() -> list[TeamMember]:
    return [
        TeamMember(username="alice", analyses_triggered=156, findings_resolved=89, fixes_applied=72, verification_score=94.2),
        TeamMember(username="bob", analyses_triggered=123, findings_resolved=67, fixes_applied=45, verification_score=88.5),
        TeamMember(username="carol", analyses_triggered=98, findings_resolved=55, fixes_applied=38, verification_score=91.0),
        TeamMember(username="dave", analyses_triggered=87, findings_resolved=42, fixes_applied=31, verification_score=85.3),
        TeamMember(username="eve", analyses_triggered=145, findings_resolved=78, fixes_applied=60, verification_score=92.7),
    ]


# ---------------------------------------------------------------------------
# In-memory config store
# ---------------------------------------------------------------------------

_slack_configs: dict[str, SlackSummaryConfig] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/summary/{org_id}", response_model=AnalyticsSummary)
async def get_analytics_summary(
    org_id: str,
    period: str = Query(default="30d", description="7d, 30d, 90d"),
) -> AnalyticsSummary:
    """Get org-wide analytics summary."""
    return AnalyticsSummary(
        org_id=org_id,
        period=period,
        total_analyses=1247,
        total_findings=342,
        total_fixes_applied=186,
        pass_rate=89.2,
        avg_findings_per_analysis=2.7,
        avg_resolution_time_hours=4.2,
        verification_coverage_pct=73.5,
    )


@router.get("/defect-trends/{org_id}", response_model=list[DefectDensityTrend])
async def get_defect_trends(
    org_id: str,
    days: int = Query(default=30, ge=7, le=365),
) -> list[DefectDensityTrend]:
    """Get defect density trends over time."""
    return _generate_defect_trends(days)


@router.get("/coverage-heatmap/{org_id}", response_model=list[CoverageHeatmapEntry])
async def get_coverage_heatmap(org_id: str) -> list[CoverageHeatmapEntry]:
    """Get verification coverage heatmap by directory."""
    return _generate_coverage_heatmap()


@router.get("/bug-patterns/{org_id}", response_model=list[BugPattern])
async def get_bug_patterns(org_id: str) -> list[BugPattern]:
    """Get top recurring bug patterns across the org."""
    return _generate_bug_patterns()


@router.get("/leaderboard/{org_id}", response_model=list[TeamMember])
async def get_team_leaderboard(
    org_id: str,
    sort_by: str = Query(default="verification_score", description="Sort field"),
) -> list[TeamMember]:
    """Get team adoption leaderboard."""
    members = _generate_leaderboard()
    if sort_by == "analyses_triggered":
        members.sort(key=lambda m: m.analyses_triggered, reverse=True)
    elif sort_by == "findings_resolved":
        members.sort(key=lambda m: m.findings_resolved, reverse=True)
    else:
        members.sort(key=lambda m: m.verification_score, reverse=True)
    return members


@router.get("/export/{org_id}/csv")
async def export_analytics_csv(
    org_id: str,
    report_type: str = Query(default="defect_trends", description="defect_trends, coverage, bugs"),
) -> StreamingResponse:
    """Export analytics data as CSV."""
    output = io.StringIO()
    writer = csv.writer(output)

    if report_type == "defect_trends":
        writer.writerow(["date", "total", "critical", "high", "medium", "low", "density"])
        for t in _generate_defect_trends(30):
            writer.writerow([t.date, t.total_findings, t.critical, t.high, t.medium, t.low, t.defect_density])
    elif report_type == "coverage":
        writer.writerow(["directory", "files_total", "files_verified", "coverage_pct", "findings", "risk"])
        for h in _generate_coverage_heatmap():
            writer.writerow([h.directory, h.files_total, h.files_verified, h.coverage_pct, h.findings_count, h.risk_level])
    elif report_type == "bugs":
        writer.writerow(["pattern", "category", "occurrences", "trend", "repos", "first_seen", "last_seen"])
        for b in _generate_bug_patterns():
            writer.writerow([b.name, b.category, b.occurrences, b.trend, ";".join(b.affected_repos), b.first_seen, b.last_seen])

    output.seek(0)
    filename = f"codeverify-{report_type}-{org_id}-{datetime.utcnow().strftime('%Y%m%d')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.post("/slack-summary", response_model=SlackSummaryConfig)
async def configure_slack_summary(config: SlackSummaryConfig) -> SlackSummaryConfig:
    """Configure Slack weekly summary notifications."""
    _slack_configs[config.org_id] = config
    return config


@router.post("/slack-summary/{org_id}/send")
async def send_slack_summary(org_id: str) -> dict[str, Any]:
    """Trigger a Slack summary notification now."""
    config = _slack_configs.get(org_id)
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Slack config not found")

    # Build summary message (in production: send via webhook)
    summary = await get_analytics_summary(org_id, "7d")
    message = {
        "channel": config.channel,
        "text": (
            f"📊 *CodeVerify Weekly Summary*\n"
            f"• Total analyses: {summary.total_analyses}\n"
            f"• Pass rate: {summary.pass_rate}%\n"
            f"• Findings: {summary.total_findings} ({summary.total_fixes_applied} auto-fixed)\n"
            f"• Coverage: {summary.verification_coverage_pct}%\n"
            f"• Avg resolution: {summary.avg_resolution_time_hours}h"
        ),
        "sent_at": datetime.utcnow().isoformat(),
    }

    return {"status": "sent", "message": message}
