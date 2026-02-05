"""
Code Evolution Tracker API Router

Provides REST API endpoints for tracking code evolution:
- Record commit snapshots
- Analyze metric trends
- Detect regressions
- Generate evolution reports
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Import Code Evolution Tracker
try:
    from codeverify_agents.code_evolution import (
        CodeEvolutionTracker,
        MetricType,
        TrendDirection,
        RegressionSeverity,
    )
    CODE_EVOLUTION_AVAILABLE = True
except ImportError:
    CODE_EVOLUTION_AVAILABLE = False
    CodeEvolutionTracker = None
    MetricType = None
    TrendDirection = None
    RegressionSeverity = None


router = APIRouter(prefix="/api/v1/evolution", tags=["code-evolution"])

# Singleton tracker instance
_tracker: Optional[CodeEvolutionTracker] = None


def get_tracker() -> CodeEvolutionTracker:
    """Get or create the tracker singleton."""
    global _tracker
    if _tracker is None and CODE_EVOLUTION_AVAILABLE:
        _tracker = CodeEvolutionTracker()
    return _tracker


# =============================================================================
# Request/Response Models
# =============================================================================


class RecordSnapshotRequest(BaseModel):
    """Request to record a commit snapshot."""
    repository: str = Field(..., description="Repository identifier")
    commit_sha: str = Field(..., description="Git commit SHA")
    commit_message: str = Field(..., description="Commit message")
    author: str = Field(..., description="Commit author")
    timestamp: Optional[str] = Field(None, description="Commit timestamp (ISO format)")
    metrics: Optional[Dict[str, float]] = Field(None, description="Quality metrics")
    files_changed: int = Field(0, ge=0, description="Number of files changed")
    lines_added: int = Field(0, ge=0, description="Lines added")
    lines_removed: int = Field(0, ge=0, description="Lines removed")
    findings_count: int = Field(0, ge=0, description="Number of findings")
    verified_functions: int = Field(0, ge=0, description="Verified functions count")
    total_functions: int = Field(0, ge=0, description="Total functions count")


class SnapshotResponse(BaseModel):
    """Response with snapshot information."""
    id: str
    commit_sha: str
    repository: str
    recorded_at: str


class AnalyzeTrendsRequest(BaseModel):
    """Request to analyze trends."""
    repository: str = Field(..., description="Repository identifier")
    metric_types: Optional[List[str]] = Field(None, description="Metrics to analyze")
    days: int = Field(30, ge=1, le=365, description="Days to analyze")


class CompareCommitsRequest(BaseModel):
    """Request to compare commits."""
    repository: str = Field(..., description="Repository identifier")
    commit_sha_1: str = Field(..., description="First commit SHA")
    commit_sha_2: str = Field(..., description="Second commit SHA")


# =============================================================================
# API Endpoints
# =============================================================================


@router.post(
    "/snapshot",
    response_model=SnapshotResponse,
    summary="Record Snapshot",
    description="Record a commit snapshot with quality metrics"
)
async def record_snapshot(request: RecordSnapshotRequest) -> SnapshotResponse:
    """
    Record a snapshot of code quality at a specific commit.

    This should be called after each commit analysis to track evolution.
    """
    if not CODE_EVOLUTION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Code Evolution Tracker is not available"
        )

    tracker = get_tracker()
    if not tracker:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Code Evolution Tracker"
        )

    timestamp = None
    if request.timestamp:
        try:
            timestamp = datetime.fromisoformat(request.timestamp)
        except ValueError:
            pass

    snapshot = tracker.record_snapshot(
        repository=request.repository,
        commit_sha=request.commit_sha,
        commit_message=request.commit_message,
        author=request.author,
        timestamp=timestamp,
        metrics=request.metrics,
        files_changed=request.files_changed,
        lines_added=request.lines_added,
        lines_removed=request.lines_removed,
        findings_count=request.findings_count,
        verified_functions=request.verified_functions,
        total_functions=request.total_functions,
    )

    return SnapshotResponse(
        id=snapshot.id,
        commit_sha=snapshot.commit_sha,
        repository=request.repository,
        recorded_at=snapshot.timestamp.isoformat(),
    )


@router.get(
    "/snapshots/{repository}",
    summary="Get Snapshots",
    description="Get snapshots for a repository"
)
async def get_snapshots(
    repository: str,
    days: int = 30,
    limit: int = 50,
) -> Dict[str, Any]:
    """Get snapshots for a repository."""
    if not CODE_EVOLUTION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Code Evolution Tracker is not available"
        )

    tracker = get_tracker()
    if not tracker:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Code Evolution Tracker"
        )

    from datetime import timedelta
    start_date = datetime.now() - timedelta(days=days)

    snapshots = tracker.get_snapshots(
        repository=repository,
        start_date=start_date,
        limit=limit,
    )

    return {
        "snapshots": [s.to_dict() for s in snapshots],
        "count": len(snapshots),
        "repository": repository,
    }


@router.post(
    "/trends",
    summary="Analyze Trends",
    description="Analyze metric trends over time"
)
async def analyze_trends(request: AnalyzeTrendsRequest) -> Dict[str, Any]:
    """
    Analyze trends for specified metrics.

    Returns trend direction, slope, and volatility for each metric.
    """
    if not CODE_EVOLUTION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Code Evolution Tracker is not available"
        )

    tracker = get_tracker()
    if not tracker:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Code Evolution Tracker"
        )

    metric_types = None
    if request.metric_types:
        metric_types = []
        for mt in request.metric_types:
            try:
                metric_types.append(MetricType(mt))
            except ValueError:
                pass

    trends = tracker.analyze_trends(
        repository=request.repository,
        metric_types=metric_types,
        days=request.days,
    )

    return {
        "trends": [t.to_dict() for t in trends],
        "count": len(trends),
        "repository": request.repository,
        "days_analyzed": request.days,
    }


@router.get(
    "/regressions/{repository}",
    summary="Detect Regressions",
    description="Detect quality regressions in recent commits"
)
async def detect_regressions(
    repository: str,
    days: int = 7,
) -> Dict[str, Any]:
    """
    Detect quality regressions in recent commits.

    Returns regressions with severity, affected commits, and suggested actions.
    """
    if not CODE_EVOLUTION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Code Evolution Tracker is not available"
        )

    tracker = get_tracker()
    if not tracker:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Code Evolution Tracker"
        )

    regressions = tracker.detect_regressions(
        repository=repository,
        days=days,
    )

    return {
        "regressions": [r.to_dict() for r in regressions],
        "count": len(regressions),
        "critical_count": sum(1 for r in regressions if r.severity == RegressionSeverity.CRITICAL),
        "high_count": sum(1 for r in regressions if r.severity == RegressionSeverity.HIGH),
        "repository": repository,
    }


@router.get(
    "/report/{repository}",
    summary="Generate Report",
    description="Generate comprehensive evolution report"
)
async def generate_report(
    repository: str,
    days: int = 30,
) -> Dict[str, Any]:
    """
    Generate comprehensive evolution report.

    Includes trends, regressions, contributor stats, and quality summary.
    """
    if not CODE_EVOLUTION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Code Evolution Tracker is not available"
        )

    tracker = get_tracker()
    if not tracker:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Code Evolution Tracker"
        )

    report = tracker.generate_report(
        repository=repository,
        days=days,
    )

    return report.to_dict()


@router.post(
    "/compare",
    summary="Compare Commits",
    description="Compare two specific commits"
)
async def compare_commits(request: CompareCommitsRequest) -> Dict[str, Any]:
    """
    Compare two specific commits.

    Returns metric differences between the commits.
    """
    if not CODE_EVOLUTION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Code Evolution Tracker is not available"
        )

    tracker = get_tracker()
    if not tracker:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Code Evolution Tracker"
        )

    result = tracker.compare_commits(
        repository=request.repository,
        commit_sha_1=request.commit_sha_1,
        commit_sha_2=request.commit_sha_2,
    )

    return result


@router.get(
    "/metric/{repository}/{metric_type}",
    summary="Get Metric History",
    description="Get historical values for a specific metric"
)
async def get_metric_history(
    repository: str,
    metric_type: str,
    days: int = 30,
) -> Dict[str, Any]:
    """Get historical values for a specific metric."""
    if not CODE_EVOLUTION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Code Evolution Tracker is not available"
        )

    tracker = get_tracker()
    if not tracker:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Code Evolution Tracker"
        )

    try:
        mt = MetricType(metric_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric type: {metric_type}"
        )

    history = tracker.get_metric_history(
        repository=repository,
        metric_type=mt,
        days=days,
    )

    return {
        "metric_type": metric_type,
        "history": history,
        "count": len(history),
        "repository": repository,
    }


@router.get(
    "/repositories",
    summary="List Repositories",
    description="List all tracked repositories"
)
async def list_repositories() -> Dict[str, Any]:
    """List all tracked repositories."""
    if not CODE_EVOLUTION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Code Evolution Tracker is not available"
        )

    tracker = get_tracker()
    if not tracker:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Code Evolution Tracker"
        )

    repos = tracker.get_repositories()

    return {
        "repositories": repos,
        "count": len(repos),
    }


@router.get(
    "/metric-types",
    summary="Get Metric Types",
    description="Get available metric types"
)
async def get_metric_types() -> Dict[str, Any]:
    """Get available metric types."""
    if not CODE_EVOLUTION_AVAILABLE:
        return {
            "available": False,
            "types": [],
        }

    types = [
        {"value": "complexity", "name": "Complexity", "higher_is_worse": True},
        {"value": "coverage", "name": "Coverage", "higher_is_worse": False},
        {"value": "security_score", "name": "Security Score", "higher_is_worse": False},
        {"value": "maintainability", "name": "Maintainability", "higher_is_worse": False},
        {"value": "verification_rate", "name": "Verification Rate", "higher_is_worse": False},
        {"value": "bug_density", "name": "Bug Density", "higher_is_worse": True},
        {"value": "tech_debt", "name": "Tech Debt", "higher_is_worse": True},
        {"value": "ai_code_ratio", "name": "AI Code Ratio", "higher_is_worse": None},
    ]

    return {
        "available": True,
        "types": types,
    }


@router.get(
    "/stats",
    summary="Get Statistics",
    description="Get tracker statistics"
)
async def get_stats() -> Dict[str, Any]:
    """Get tracker statistics."""
    if not CODE_EVOLUTION_AVAILABLE:
        return {
            "available": False,
            "message": "Code Evolution Tracker is not available",
        }

    tracker = get_tracker()
    if not tracker:
        return {
            "available": False,
            "message": "Failed to initialize tracker",
        }

    stats = tracker.get_statistics()
    stats["available"] = True

    return stats


@router.delete(
    "/repository/{repository}",
    summary="Clear Repository",
    description="Clear evolution data for a repository"
)
async def clear_repository(repository: str) -> Dict[str, Any]:
    """Clear evolution data for a repository."""
    if not CODE_EVOLUTION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Code Evolution Tracker is not available"
        )

    tracker = get_tracker()
    if not tracker:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Code Evolution Tracker"
        )

    success = tracker.clear_repository(repository)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Repository not found: {repository}"
        )

    return {
        "cleared": True,
        "repository": repository,
    }
