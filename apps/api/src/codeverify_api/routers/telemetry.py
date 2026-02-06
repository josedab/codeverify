"""Verification telemetry and ROI dashboard API router."""

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

router = APIRouter()


class RecordEventRequest(BaseModel):
    event_type: str = Field(description="Event type: finding_detected, finding_fixed, analysis_completed, etc.")
    repo_id: str
    finding_id: str | None = None
    severity: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecordEventResponse(BaseModel):
    event_id: str
    recorded: bool


class FindingLifecycleUpdate(BaseModel):
    finding_id: str
    new_status: str = Field(description="Status: detected, acknowledged, fix_in_progress, fixed, verified, false_positive, wont_fix")
    repo_id: str
    severity: str | None = None


class ROIReportResponse(BaseModel):
    period_days: int
    total_findings: int
    findings_by_severity: dict[str, int]
    bugs_prevented: int
    estimated_cost_saved: float
    total_analyses: int
    average_time_to_fix_hours: float
    fix_rate: float
    verification_coverage: float
    developer_hours_saved: float
    roi_multiplier: float
    trends: dict[str, list[float]]


class TrendDataResponse(BaseModel):
    metric: str
    weeks: int
    data: list[float]
    labels: list[str]


# NIST-based cost estimates per bug severity
SEVERITY_COSTS = {
    "critical": 15000.0,
    "high": 5000.0,
    "medium": 1500.0,
    "low": 500.0,
}

# In-memory telemetry storage
_events: list[dict[str, Any]] = []
_findings: dict[str, dict[str, Any]] = {}
_analyses_count: int = 0


@router.post("/events", response_model=RecordEventResponse)
async def record_event(request: RecordEventRequest) -> RecordEventResponse:
    """Record a telemetry event."""
    global _analyses_count
    event_id = str(uuid.uuid4())
    event = {
        "event_id": event_id,
        "timestamp": time.time(),
        "event_type": request.event_type,
        "repo_id": request.repo_id,
        "finding_id": request.finding_id,
        "severity": request.severity,
        "metadata": request.metadata,
    }
    _events.append(event)

    if request.event_type == "analysis_completed":
        _analyses_count += 1

    if request.event_type == "finding_detected" and request.finding_id:
        _findings[request.finding_id] = {
            "finding_id": request.finding_id,
            "detected_at": time.time(),
            "severity": request.severity or "medium",
            "lifecycle": "detected",
            "fixed_at": None,
            "repo_id": request.repo_id,
        }

    return RecordEventResponse(event_id=event_id, recorded=True)


@router.post("/findings/lifecycle")
async def update_finding_lifecycle(request: FindingLifecycleUpdate) -> dict[str, Any]:
    """Update the lifecycle status of a finding."""
    finding = _findings.get(request.finding_id)
    if not finding:
        # Auto-create if not tracked yet
        _findings[request.finding_id] = {
            "finding_id": request.finding_id,
            "detected_at": time.time(),
            "severity": request.severity or "medium",
            "lifecycle": request.new_status,
            "fixed_at": None,
            "repo_id": request.repo_id,
        }
        finding = _findings[request.finding_id]

    old_status = finding["lifecycle"]
    finding["lifecycle"] = request.new_status

    if request.new_status in ("fixed", "verified"):
        finding["fixed_at"] = time.time()

    return {
        "finding_id": request.finding_id,
        "previous_status": old_status,
        "new_status": request.new_status,
        "updated": True,
    }


@router.get("/roi-report", response_model=ROIReportResponse)
async def generate_roi_report(
    period_days: int = Query(default=30, ge=1, le=365, description="Report period in days"),
) -> ROIReportResponse:
    """Generate an ROI report for the specified period."""
    cutoff = time.time() - (period_days * 86400)

    # Filter findings within period
    period_findings = {
        fid: f for fid, f in _findings.items()
        if f["detected_at"] >= cutoff
    }

    # Aggregate by severity
    by_severity: dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for f in period_findings.values():
        sev = f.get("severity", "medium")
        if sev in by_severity:
            by_severity[sev] += 1

    total = len(period_findings)
    fixed = sum(1 for f in period_findings.values() if f["lifecycle"] in ("fixed", "verified"))
    false_pos = sum(1 for f in period_findings.values() if f["lifecycle"] == "false_positive")

    bugs_prevented = fixed
    fix_rate = fixed / max(total - false_pos, 1)

    # Time to fix
    fix_times = []
    for f in period_findings.values():
        if f.get("fixed_at") and f["detected_at"]:
            hours = (f["fixed_at"] - f["detected_at"]) / 3600
            fix_times.append(hours)
    avg_ttf = sum(fix_times) / len(fix_times) if fix_times else 0.0

    # Cost savings estimate
    cost_saved = sum(
        SEVERITY_COSTS.get(f.get("severity", "medium"), 1500.0)
        for f in period_findings.values()
        if f["lifecycle"] in ("fixed", "verified")
    )

    # Estimate developer hours saved (1 hour per finding detected early vs 4 hours in production)
    dev_hours_saved = bugs_prevented * 3.0

    # Assume tool cost of $500/month for ROI calculation
    tool_cost = (period_days / 30) * 500
    roi_multiplier = cost_saved / max(tool_cost, 1)

    # Generate trend data (weekly buckets)
    weeks = min(period_days // 7, 12)
    findings_trend = [0.0] * max(weeks, 1)
    fixes_trend = [0.0] * max(weeks, 1)

    for f in period_findings.values():
        week_idx = min(int((time.time() - f["detected_at"]) / (7 * 86400)), max(weeks - 1, 0))
        if week_idx < len(findings_trend):
            findings_trend[week_idx] += 1
        if f.get("fixed_at"):
            fix_week = min(int((time.time() - f["fixed_at"]) / (7 * 86400)), max(weeks - 1, 0))
            if fix_week < len(fixes_trend):
                fixes_trend[fix_week] += 1

    # Reverse so most recent is last
    findings_trend.reverse()
    fixes_trend.reverse()

    return ROIReportResponse(
        period_days=period_days,
        total_findings=total,
        findings_by_severity=by_severity,
        bugs_prevented=bugs_prevented,
        estimated_cost_saved=round(cost_saved, 2),
        total_analyses=_analyses_count,
        average_time_to_fix_hours=round(avg_ttf, 2),
        fix_rate=round(fix_rate, 3),
        verification_coverage=0.75,  # placeholder - would need real data
        developer_hours_saved=round(dev_hours_saved, 1),
        roi_multiplier=round(roi_multiplier, 2),
        trends={
            "findings": findings_trend,
            "fixes": fixes_trend,
        },
    )


@router.get("/executive-summary")
async def get_executive_summary(
    period_days: int = Query(default=30, ge=1, le=365),
) -> dict[str, str]:
    """Generate a markdown-formatted executive summary."""
    report = await generate_roi_report(period_days)

    summary = f"""# CodeVerify ROI Report — Last {report.period_days} Days

## Key Metrics

| Metric | Value |
|--------|-------|
| Total Findings | {report.total_findings} |
| Bugs Prevented | {report.bugs_prevented} |
| Estimated Cost Saved | ${report.estimated_cost_saved:,.2f} |
| Fix Rate | {report.fix_rate:.1%} |
| Avg. Time to Fix | {report.average_time_to_fix_hours:.1f} hours |
| Developer Hours Saved | {report.developer_hours_saved:.0f} hours |
| ROI Multiplier | {report.roi_multiplier:.1f}x |

## Findings by Severity

| Severity | Count |
|----------|-------|
| Critical | {report.findings_by_severity.get('critical', 0)} |
| High | {report.findings_by_severity.get('high', 0)} |
| Medium | {report.findings_by_severity.get('medium', 0)} |
| Low | {report.findings_by_severity.get('low', 0)} |

## Summary

CodeVerify prevented an estimated **{report.bugs_prevented} bugs** from reaching production,
saving approximately **${report.estimated_cost_saved:,.2f}** in potential remediation costs.
The tool delivered a **{report.roi_multiplier:.1f}x return on investment** over the period.
"""
    return {"markdown": summary, "period_days": str(period_days)}


@router.get("/trends", response_model=TrendDataResponse)
async def get_trend_data(
    metric: str = Query(description="Metric: findings, fixes, cost_saved"),
    weeks: int = Query(default=12, ge=1, le=52),
) -> TrendDataResponse:
    """Get weekly trend data for a specific metric."""
    import datetime

    data = [0.0] * weeks
    labels = []
    for i in range(weeks):
        week_start = datetime.datetime.now() - datetime.timedelta(weeks=weeks - i - 1)
        labels.append(week_start.strftime("%Y-%m-%d"))

    cutoff = time.time() - (weeks * 7 * 86400)

    if metric in ("findings", "fixes"):
        for f in _findings.values():
            ts = f["detected_at"] if metric == "findings" else f.get("fixed_at")
            if ts and ts >= cutoff:
                week_idx = int((ts - cutoff) / (7 * 86400))
                if 0 <= week_idx < weeks:
                    data[week_idx] += 1
    elif metric == "cost_saved":
        for f in _findings.values():
            if f.get("fixed_at") and f["fixed_at"] >= cutoff:
                week_idx = int((f["fixed_at"] - cutoff) / (7 * 86400))
                if 0 <= week_idx < weeks:
                    data[week_idx] += SEVERITY_COSTS.get(f.get("severity", "medium"), 1500.0)

    return TrendDataResponse(metric=metric, weeks=weeks, data=data, labels=labels)


@router.get("/events")
async def list_events(
    repo_id: str | None = Query(default=None),
    event_type: str | None = Query(default=None),
    limit: int = Query(default=50, le=200),
) -> dict[str, Any]:
    """List telemetry events with optional filters."""
    filtered = _events
    if repo_id:
        filtered = [e for e in filtered if e["repo_id"] == repo_id]
    if event_type:
        filtered = [e for e in filtered if e["event_type"] == event_type]

    return {
        "events": filtered[-limit:],
        "total": len(filtered),
        "returned": min(len(filtered), limit),
    }
