"""Codebase Scanning API endpoints."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

router = APIRouter()


class ScanConfigRequest(BaseModel):
    """Request to configure a scan."""

    repo_full_name: str
    scan_type: str = Field(default="full", description="Scan type: full, incremental, security, verification, quality")
    schedule: str = Field(default="manual", description="Schedule: manual, daily, weekly, monthly, on_push")
    branch: str | None = None

    include_security: bool = True
    include_verification: bool = True
    include_quality: bool = True
    include_tech_debt: bool = True

    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)

    max_files: int | None = None
    timeout_minutes: int = 60

    notify_on_complete: bool = True
    notify_channels: list[str] = Field(default_factory=list)


class TriggerScanRequest(BaseModel):
    """Request to trigger an immediate scan."""

    repo_full_name: str
    scan_type: str = "full"
    branch: str | None = None


class ScanResultSummary(BaseModel):
    """Summary of scan results."""

    scan_id: str
    repo_full_name: str
    scan_type: str
    status: str
    started_at: str | None
    completed_at: str | None
    duration_seconds: float | None

    files_scanned: int
    lines_of_code: int
    total_findings: int

    security_score: float | None
    quality_score: float | None
    verification_coverage: float | None
    tech_debt_hours: float


@router.post("/trigger", response_model=ScanResultSummary)
async def trigger_scan(request: TriggerScanRequest) -> ScanResultSummary:
    """
    Trigger an immediate codebase scan.

    Returns the scan ID which can be used to poll for results.
    """
    from codeverify_core.scanning import (
        ScanConfiguration,
        ScanType,
        ScanSchedule,
        trigger_scan as do_trigger_scan,
    )

    config = ScanConfiguration(
        repo_full_name=request.repo_full_name,
        scan_type=ScanType(request.scan_type),
        schedule=ScanSchedule.MANUAL,
        branch=request.branch,
    )

    result = await do_trigger_scan(config)

    return ScanResultSummary(
        scan_id=str(result.scan_id),
        repo_full_name=result.repo_full_name,
        scan_type=result.scan_type.value,
        status=result.status.value,
        started_at=result.started_at.isoformat() if result.started_at else None,
        completed_at=result.completed_at.isoformat() if result.completed_at else None,
        duration_seconds=result.duration_seconds,
        files_scanned=result.files_scanned,
        lines_of_code=result.lines_of_code,
        total_findings=result.total_findings,
        security_score=result.security_score.score if result.security_score else None,
        quality_score=result.quality_metrics.overall_score if result.quality_metrics else None,
        verification_coverage=result.verification_coverage.coverage_percentage if result.verification_coverage else None,
        tech_debt_hours=result.tech_debt_hours,
    )


@router.get("/{scan_id}")
async def get_scan_result(scan_id: str) -> dict[str, Any]:
    """Get detailed scan results by ID."""
    from codeverify_core.scanning import get_scan_result as do_get_result

    try:
        scan_uuid = UUID(scan_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid scan ID format",
        )

    result = await do_get_result(scan_uuid)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan not found: {scan_id}",
        )

    return {
        "scan_id": str(result.scan_id),
        "repo_full_name": result.repo_full_name,
        "scan_type": result.scan_type.value,
        "status": result.status.value,
        "started_at": result.started_at.isoformat() if result.started_at else None,
        "completed_at": result.completed_at.isoformat() if result.completed_at else None,
        "duration_seconds": result.duration_seconds,
        "files_scanned": result.files_scanned,
        "lines_of_code": result.lines_of_code,
        "security_score": result.security_score.dict() if result.security_score else None,
        "quality_metrics": result.quality_metrics.dict() if result.quality_metrics else None,
        "verification_coverage": result.verification_coverage.dict() if result.verification_coverage else None,
        "total_findings": result.total_findings,
        "findings_by_severity": result.findings_by_severity,
        "findings_by_category": result.findings_by_category,
        "tech_debt_items": [item.dict() for item in result.tech_debt_items],
        "tech_debt_hours": result.tech_debt_hours,
        "error_message": result.error_message,
    }


@router.post("/{scan_id}/simulate")
async def simulate_scan(scan_id: str) -> dict[str, Any]:
    """
    Simulate scan execution (for demo purposes).

    In production, scans run asynchronously via background workers.
    """
    from codeverify_core.scanning import simulate_scan_execution

    try:
        scan_uuid = UUID(scan_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid scan ID format",
        )

    try:
        result = await simulate_scan_execution(scan_uuid)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    return {
        "scan_id": str(result.scan_id),
        "status": result.status.value,
        "message": "Scan simulation completed",
    }


@router.get("/repo/{repo_full_name:path}/history")
async def get_scan_history(
    repo_full_name: str,
    limit: int = Query(default=10, le=50),
) -> dict[str, Any]:
    """Get scan history for a repository."""
    from codeverify_core.scanning import get_scan_history as do_get_history

    results = await do_get_history(repo_full_name, limit)

    return {
        "repo_full_name": repo_full_name,
        "scans": [
            {
                "scan_id": str(r.scan_id),
                "scan_type": r.scan_type.value,
                "status": r.status.value,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "total_findings": r.total_findings,
                "security_score": r.security_score.score if r.security_score else None,
            }
            for r in results
        ],
    }


@router.get("/repo/{repo_full_name:path}/trends")
async def get_trends(
    repo_full_name: str,
    days: int = Query(default=30, le=90),
) -> dict[str, Any]:
    """Get trend data for a repository over time."""
    from codeverify_core.scanning import get_trend_data

    return await get_trend_data(repo_full_name, days)


@router.post("/schedule", status_code=status.HTTP_201_CREATED)
async def create_scheduled_scan(request: ScanConfigRequest) -> dict[str, Any]:
    """Create a scheduled scan for a repository."""
    from codeverify_core.scanning import (
        ScanConfiguration,
        ScanType,
        ScanSchedule,
        create_scheduled_scan as do_create_scheduled,
    )

    config = ScanConfiguration(
        repo_full_name=request.repo_full_name,
        scan_type=ScanType(request.scan_type),
        schedule=ScanSchedule(request.schedule),
        branch=request.branch,
        include_security=request.include_security,
        include_verification=request.include_verification,
        include_quality=request.include_quality,
        include_tech_debt=request.include_tech_debt,
        include_patterns=request.include_patterns,
        exclude_patterns=request.exclude_patterns,
        languages=request.languages,
        max_files=request.max_files,
        timeout_minutes=request.timeout_minutes,
        notify_on_complete=request.notify_on_complete,
        notify_channels=request.notify_channels,
    )

    scheduled = await do_create_scheduled(config)

    return {
        "id": str(scheduled.id),
        "repo_full_name": scheduled.config.repo_full_name,
        "schedule": scheduled.schedule.value,
        "next_run": scheduled.next_run.isoformat() if scheduled.next_run else None,
        "enabled": scheduled.enabled,
    }


@router.get("/schedules")
async def list_scheduled_scans(
    repo_full_name: str | None = None,
) -> dict[str, Any]:
    """List all scheduled scans."""
    from codeverify_core.scanning import _scheduled_scans

    schedules = list(_scheduled_scans.values())

    if repo_full_name:
        schedules = [s for s in schedules if s.config.repo_full_name == repo_full_name]

    return {
        "schedules": [
            {
                "id": str(s.id),
                "repo_full_name": s.config.repo_full_name,
                "scan_type": s.config.scan_type.value,
                "schedule": s.schedule.value,
                "next_run": s.next_run.isoformat() if s.next_run else None,
                "last_run": s.last_run.isoformat() if s.last_run else None,
                "enabled": s.enabled,
            }
            for s in schedules
        ],
    }


@router.delete("/schedule/{schedule_id}")
async def delete_scheduled_scan(schedule_id: str) -> dict[str, Any]:
    """Delete a scheduled scan."""
    from codeverify_core.scanning import _scheduled_scans

    if schedule_id not in _scheduled_scans:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Schedule not found: {schedule_id}",
        )

    del _scheduled_scans[schedule_id]
    return {"deleted": True, "schedule_id": schedule_id}


@router.post("/schedule/{schedule_id}/toggle")
async def toggle_scheduled_scan(
    schedule_id: str,
    enabled: bool,
) -> dict[str, Any]:
    """Enable or disable a scheduled scan."""
    from codeverify_core.scanning import _scheduled_scans

    if schedule_id not in _scheduled_scans:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Schedule not found: {schedule_id}",
        )

    _scheduled_scans[schedule_id].enabled = enabled
    return {"schedule_id": schedule_id, "enabled": enabled}
