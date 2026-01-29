"""Codebase-Wide Analysis - Scheduled full repository scans."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ScanStatus(str, Enum):
    """Status of a codebase scan."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScanType(str, Enum):
    """Type of codebase scan."""

    FULL = "full"  # Complete repository scan
    INCREMENTAL = "incremental"  # Only changed files since last scan
    SECURITY = "security"  # Security-focused scan
    VERIFICATION = "verification"  # Formal verification focus
    QUALITY = "quality"  # Code quality metrics


class ScanSchedule(str, Enum):
    """Schedule frequency for scans."""

    MANUAL = "manual"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ON_PUSH = "on_push"  # Triggered on push to default branch


class SecurityScore(BaseModel):
    """Security posture score."""

    score: float = Field(ge=0.0, le=100.0)
    critical_issues: int = 0
    high_issues: int = 0
    medium_issues: int = 0
    low_issues: int = 0
    trend: str = "stable"  # "improving", "declining", "stable"


class QualityMetrics(BaseModel):
    """Code quality metrics."""

    overall_score: float = Field(ge=0.0, le=100.0)
    maintainability_index: float = Field(ge=0.0, le=100.0)
    test_coverage: float | None = None
    documentation_coverage: float | None = None
    type_coverage: float | None = None
    complexity_score: float = Field(ge=0.0, le=100.0)
    duplication_percentage: float = Field(ge=0.0, le=100.0)


class VerificationCoverage(BaseModel):
    """Formal verification coverage metrics."""

    total_functions: int = 0
    verified_functions: int = 0
    coverage_percentage: float = 0.0
    checks_passed: int = 0
    checks_failed: int = 0
    checks_timeout: int = 0


class TechDebtItem(BaseModel):
    """A tech debt item."""

    id: str
    category: str  # "security", "performance", "maintainability", "deprecated"
    severity: str
    file_path: str
    line: int | None = None
    description: str
    estimated_effort: str | None = None  # "1h", "1d", "1w"
    created_at: datetime


class CodebaseScanResult(BaseModel):
    """Results from a codebase scan."""

    scan_id: UUID
    repo_full_name: str
    scan_type: ScanType
    status: ScanStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None

    # Metrics
    files_scanned: int = 0
    lines_of_code: int = 0
    security_score: SecurityScore | None = None
    quality_metrics: QualityMetrics | None = None
    verification_coverage: VerificationCoverage | None = None

    # Findings
    total_findings: int = 0
    findings_by_severity: dict[str, int] = Field(default_factory=dict)
    findings_by_category: dict[str, int] = Field(default_factory=dict)

    # Tech debt
    tech_debt_items: list[TechDebtItem] = Field(default_factory=list)
    tech_debt_hours: float = 0.0

    # Trends
    trend_data: dict[str, Any] = Field(default_factory=dict)

    # Error info
    error_message: str | None = None


class ScanConfiguration(BaseModel):
    """Configuration for a codebase scan."""

    repo_full_name: str
    scan_type: ScanType = ScanType.FULL
    schedule: ScanSchedule = ScanSchedule.MANUAL
    branch: str | None = None  # Default branch if not specified

    # What to include
    include_security: bool = True
    include_verification: bool = True
    include_quality: bool = True
    include_tech_debt: bool = True

    # Filters
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)

    # Limits
    max_files: int | None = None
    timeout_minutes: int = 60

    # Notification
    notify_on_complete: bool = True
    notify_channels: list[str] = Field(default_factory=list)  # ["slack", "email"]


class ScheduledScan(BaseModel):
    """A scheduled scan configuration."""

    id: UUID
    config: ScanConfiguration
    schedule: ScanSchedule
    next_run: datetime | None = None
    last_run: datetime | None = None
    last_result_id: UUID | None = None
    enabled: bool = True
    created_at: datetime
    updated_at: datetime


# In-memory storage (should use database in production)
_scheduled_scans: dict[str, ScheduledScan] = {}
_scan_results: dict[str, CodebaseScanResult] = {}


async def create_scheduled_scan(config: ScanConfiguration) -> ScheduledScan:
    """Create a new scheduled scan."""
    scan_id = uuid4()
    now = datetime.utcnow()

    # Calculate next run time
    next_run = _calculate_next_run(config.schedule, now)

    scheduled = ScheduledScan(
        id=scan_id,
        config=config,
        schedule=config.schedule,
        next_run=next_run,
        enabled=True,
        created_at=now,
        updated_at=now,
    )

    _scheduled_scans[str(scan_id)] = scheduled
    return scheduled


async def trigger_scan(config: ScanConfiguration) -> CodebaseScanResult:
    """Trigger an immediate scan."""
    scan_id = uuid4()
    now = datetime.utcnow()

    result = CodebaseScanResult(
        scan_id=scan_id,
        repo_full_name=config.repo_full_name,
        scan_type=config.scan_type,
        status=ScanStatus.QUEUED,
        started_at=now,
    )

    _scan_results[str(scan_id)] = result

    # In production, this would queue a background job
    # For now, we'll return the queued status
    return result


async def get_scan_result(scan_id: UUID) -> CodebaseScanResult | None:
    """Get scan result by ID."""
    return _scan_results.get(str(scan_id))


async def get_scan_history(
    repo_full_name: str,
    limit: int = 10,
) -> list[CodebaseScanResult]:
    """Get scan history for a repository."""
    results = [
        r for r in _scan_results.values()
        if r.repo_full_name == repo_full_name
    ]
    results.sort(key=lambda r: r.started_at or datetime.min, reverse=True)
    return results[:limit]


async def get_trend_data(
    repo_full_name: str,
    days: int = 30,
) -> dict[str, Any]:
    """Get trend data for a repository over time."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    results = [
        r for r in _scan_results.values()
        if r.repo_full_name == repo_full_name
        and r.started_at
        and r.started_at > cutoff
        and r.status == ScanStatus.COMPLETED
    ]

    if not results:
        return {"message": "No scan data available for trend analysis"}

    results.sort(key=lambda r: r.started_at or datetime.min)

    # Build trend data
    security_scores = []
    quality_scores = []
    finding_counts = []

    for r in results:
        date_str = r.started_at.strftime("%Y-%m-%d") if r.started_at else ""

        if r.security_score:
            security_scores.append({
                "date": date_str,
                "score": r.security_score.score,
            })

        if r.quality_metrics:
            quality_scores.append({
                "date": date_str,
                "score": r.quality_metrics.overall_score,
            })

        finding_counts.append({
            "date": date_str,
            "total": r.total_findings,
            "by_severity": r.findings_by_severity,
        })

    return {
        "period_days": days,
        "scan_count": len(results),
        "security_trend": security_scores,
        "quality_trend": quality_scores,
        "findings_trend": finding_counts,
    }


def _calculate_next_run(schedule: ScanSchedule, from_time: datetime) -> datetime | None:
    """Calculate the next run time for a schedule."""
    if schedule == ScanSchedule.MANUAL:
        return None
    elif schedule == ScanSchedule.DAILY:
        return from_time + timedelta(days=1)
    elif schedule == ScanSchedule.WEEKLY:
        return from_time + timedelta(weeks=1)
    elif schedule == ScanSchedule.MONTHLY:
        return from_time + timedelta(days=30)
    elif schedule == ScanSchedule.ON_PUSH:
        return None  # Triggered by webhook
    return None


async def simulate_scan_execution(scan_id: UUID) -> CodebaseScanResult:
    """
    Simulate scan execution for demo purposes.

    In production, this would be handled by a background worker.
    """
    import random

    result = _scan_results.get(str(scan_id))
    if not result:
        raise ValueError(f"Scan not found: {scan_id}")

    result.status = ScanStatus.RUNNING

    # Simulate scan results
    result.files_scanned = random.randint(100, 1000)
    result.lines_of_code = result.files_scanned * random.randint(50, 200)

    # Security score
    result.security_score = SecurityScore(
        score=random.uniform(60, 95),
        critical_issues=random.randint(0, 2),
        high_issues=random.randint(0, 5),
        medium_issues=random.randint(2, 15),
        low_issues=random.randint(5, 30),
        trend=random.choice(["improving", "stable", "declining"]),
    )

    # Quality metrics
    result.quality_metrics = QualityMetrics(
        overall_score=random.uniform(60, 90),
        maintainability_index=random.uniform(50, 85),
        test_coverage=random.uniform(40, 80),
        documentation_coverage=random.uniform(30, 70),
        type_coverage=random.uniform(60, 95),
        complexity_score=random.uniform(20, 50),
        duplication_percentage=random.uniform(2, 15),
    )

    # Verification coverage
    total_functions = random.randint(50, 300)
    verified = int(total_functions * random.uniform(0.3, 0.8))
    result.verification_coverage = VerificationCoverage(
        total_functions=total_functions,
        verified_functions=verified,
        coverage_percentage=(verified / total_functions) * 100,
        checks_passed=verified - random.randint(0, 5),
        checks_failed=random.randint(0, 3),
        checks_timeout=random.randint(0, 2),
    )

    # Findings
    result.total_findings = (
        result.security_score.critical_issues
        + result.security_score.high_issues
        + result.security_score.medium_issues
        + result.security_score.low_issues
    )
    result.findings_by_severity = {
        "critical": result.security_score.critical_issues,
        "high": result.security_score.high_issues,
        "medium": result.security_score.medium_issues,
        "low": result.security_score.low_issues,
    }

    # Tech debt
    result.tech_debt_items = [
        TechDebtItem(
            id=str(uuid4()),
            category="deprecated",
            severity="medium",
            file_path="src/legacy/old_module.py",
            description="Uses deprecated API",
            estimated_effort="4h",
            created_at=datetime.utcnow(),
        ),
        TechDebtItem(
            id=str(uuid4()),
            category="maintainability",
            severity="low",
            file_path="src/utils/helpers.py",
            line=145,
            description="Function too complex (cyclomatic complexity: 25)",
            estimated_effort="2h",
            created_at=datetime.utcnow(),
        ),
    ]
    result.tech_debt_hours = random.uniform(20, 100)

    result.status = ScanStatus.COMPLETED
    result.completed_at = datetime.utcnow()
    if result.started_at:
        result.duration_seconds = (result.completed_at - result.started_at).total_seconds()

    return result
