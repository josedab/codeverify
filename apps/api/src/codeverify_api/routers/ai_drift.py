"""
AI Drift Detection API Router

Provides REST API endpoints for monitoring AI code quality drift:
- Record AI code snapshots
- Generate drift reports
- Get active alerts
- Manage baselines
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Import AI Drift Detector
try:
    from codeverify_agents.ai_drift_detector import (
        AIDriftDetector,
        AICodeSnapshot,
        DriftCategory,
        DriftSeverity,
    )
    AI_DRIFT_DETECTOR_AVAILABLE = True
except ImportError:
    AI_DRIFT_DETECTOR_AVAILABLE = False
    AIDriftDetector = None
    AICodeSnapshot = None
    DriftCategory = None
    DriftSeverity = None


router = APIRouter(prefix="/api/v1/drift", tags=["ai-drift"])

# Singleton detector instance
_drift_detector: Optional[AIDriftDetector] = None


def get_drift_detector() -> AIDriftDetector:
    """Get or create the drift detector singleton."""
    global _drift_detector
    if _drift_detector is None and AI_DRIFT_DETECTOR_AVAILABLE:
        _drift_detector = AIDriftDetector()
    return _drift_detector


# =============================================================================
# Request/Response Models
# =============================================================================


class RecordSnapshotRequest(BaseModel):
    """Request to record an AI code snapshot."""
    file_path: str = Field(..., description="Path to the file")
    code: str = Field(..., description="The code content")
    trust_score: float = Field(..., ge=0, le=100, description="Trust score (0-100)")
    ai_probability: float = Field(..., ge=0, le=1, description="AI probability (0-1)")
    findings: List[Dict[str, Any]] = Field(default_factory=list, description="Analysis findings")

    # Optional metadata
    detected_model: Optional[str] = Field(None, description="Detected AI model")
    complexity_score: Optional[float] = Field(None, ge=0, le=100)
    security_score: Optional[float] = Field(None, ge=0, le=100)
    test_coverage: Optional[float] = Field(None, ge=0, le=100)
    documentation_score: Optional[float] = Field(None, ge=0, le=100)
    was_reviewed: Optional[bool] = Field(False, description="Whether code was reviewed")
    review_depth: Optional[float] = Field(None, ge=0, le=1)
    time_to_accept: Optional[float] = Field(None, ge=0)
    author: Optional[str] = Field(None, description="Code author")
    commit_hash: Optional[str] = Field(None, description="Git commit hash")


class SnapshotResponse(BaseModel):
    """Response after recording a snapshot."""
    snapshot_id: str
    timestamp: str
    immediate_alerts: List[Dict[str, Any]]


class DriftReportRequest(BaseModel):
    """Request to generate a drift report."""
    days: int = Field(30, ge=1, le=365, description="Number of days to analyze")


class EstablishBaselineRequest(BaseModel):
    """Request to establish baseline metrics."""
    days: int = Field(30, ge=7, le=90, description="Days of data to use for baseline")


class AlertsRequest(BaseModel):
    """Request to filter alerts."""
    severity: Optional[str] = Field(None, description="Filter by severity")
    category: Optional[str] = Field(None, description="Filter by category")


class UpdateThresholdsRequest(BaseModel):
    """Request to update detection thresholds."""
    trust_score_min: Optional[float] = Field(None, ge=0, le=100)
    trust_score_decline: Optional[float] = Field(None, ge=0, le=100)
    security_score_min: Optional[float] = Field(None, ge=0, le=100)
    review_rate_min: Optional[float] = Field(None, ge=0, le=100)
    acceptance_spike: Optional[float] = Field(None, ge=0, le=100)
    critical_finding_max: Optional[float] = Field(None, ge=0, le=1)
    complexity_increase: Optional[float] = Field(None, ge=0, le=100)
    test_coverage_min: Optional[float] = Field(None, ge=0, le=100)


# =============================================================================
# API Endpoints
# =============================================================================


@router.post(
    "/snapshot",
    response_model=SnapshotResponse,
    summary="Record AI Code Snapshot",
    description="Record an AI-generated code snapshot for drift tracking"
)
async def record_snapshot(request: RecordSnapshotRequest) -> SnapshotResponse:
    """
    Record an AI code snapshot.

    This endpoint should be called whenever AI-generated code is detected
    or analyzed. The detector uses these snapshots to track quality trends.
    """
    if not AI_DRIFT_DETECTOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="AI Drift Detector is not available"
        )

    detector = get_drift_detector()
    if not detector:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize AI Drift Detector"
        )

    # Build optional metadata
    metadata = {}
    if request.detected_model:
        metadata["detected_model"] = request.detected_model
    if request.complexity_score is not None:
        metadata["complexity_score"] = request.complexity_score
    if request.security_score is not None:
        metadata["security_score"] = request.security_score
    if request.test_coverage is not None:
        metadata["test_coverage"] = request.test_coverage
    if request.documentation_score is not None:
        metadata["documentation_score"] = request.documentation_score
    if request.was_reviewed is not None:
        metadata["was_reviewed"] = request.was_reviewed
    if request.review_depth is not None:
        metadata["review_depth"] = request.review_depth
    if request.time_to_accept is not None:
        metadata["time_to_accept"] = request.time_to_accept
    if request.author:
        metadata["author"] = request.author
    if request.commit_hash:
        metadata["commit_hash"] = request.commit_hash

    # Record the snapshot
    snapshot = detector.record_from_analysis(
        file_path=request.file_path,
        code=request.code,
        trust_score=request.trust_score,
        ai_probability=request.ai_probability,
        findings=request.findings,
        **metadata,
    )

    # Get any immediate alerts generated
    recent_alerts = detector.get_active_alerts()[-5:]  # Last 5 alerts

    return SnapshotResponse(
        snapshot_id=snapshot.snapshot_id,
        timestamp=snapshot.timestamp.isoformat(),
        immediate_alerts=[a.to_dict() for a in recent_alerts],
    )


@router.post(
    "/report",
    summary="Generate Drift Report",
    description="Generate a comprehensive drift analysis report"
)
async def generate_drift_report(request: DriftReportRequest) -> Dict[str, Any]:
    """
    Generate a drift analysis report.

    Returns comprehensive analysis of AI code quality trends,
    active alerts, and recommendations.
    """
    if not AI_DRIFT_DETECTOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="AI Drift Detector is not available"
        )

    detector = get_drift_detector()
    if not detector:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize AI Drift Detector"
        )

    report = detector.generate_report(request.days)
    return report.to_dict()


@router.post(
    "/baseline",
    summary="Establish Baseline",
    description="Establish baseline metrics from historical data"
)
async def establish_baseline(request: EstablishBaselineRequest) -> Dict[str, Any]:
    """
    Establish baseline metrics.

    Uses historical data to establish what "normal" looks like,
    enabling drift detection by comparison.
    """
    if not AI_DRIFT_DETECTOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="AI Drift Detector is not available"
        )

    detector = get_drift_detector()
    if not detector:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize AI Drift Detector"
        )

    metrics = detector.establish_baseline(request.days)

    return {
        "baseline_established": True,
        "period_days": request.days,
        "metrics": metrics.to_dict(),
    }


@router.get(
    "/alerts",
    summary="Get Active Alerts",
    description="Get active drift alerts"
)
async def get_alerts(
    severity: Optional[str] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get active drift alerts.

    Can be filtered by severity (low, medium, high, critical)
    or category (quality_degradation, security_risk_increase, etc.)
    """
    if not AI_DRIFT_DETECTOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="AI Drift Detector is not available"
        )

    detector = get_drift_detector()
    if not detector:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize AI Drift Detector"
        )

    # Parse filters
    severity_filter = None
    category_filter = None

    if severity:
        try:
            severity_filter = DriftSeverity(severity.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid severity: {severity}. Valid values: low, medium, high, critical"
            )

    if category:
        try:
            category_filter = DriftCategory(category.lower())
        except ValueError:
            valid = [c.value for c in DriftCategory]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid category: {category}. Valid values: {valid}"
            )

    alerts = detector.get_active_alerts(severity_filter, category_filter)

    return {
        "alerts": [a.to_dict() for a in alerts],
        "count": len(alerts),
        "filters": {
            "severity": severity,
            "category": category,
        },
    }


@router.get(
    "/stats",
    summary="Get Detector Statistics",
    description="Get drift detector statistics"
)
async def get_stats() -> Dict[str, Any]:
    """Get detector statistics."""
    if not AI_DRIFT_DETECTOR_AVAILABLE:
        return {
            "available": False,
            "message": "AI Drift Detector is not available",
        }

    detector = get_drift_detector()
    if not detector:
        return {
            "available": False,
            "message": "Failed to initialize detector",
        }

    stats = detector.get_statistics()
    stats["available"] = True

    return stats


@router.put(
    "/thresholds",
    summary="Update Detection Thresholds",
    description="Update thresholds for drift detection"
)
async def update_thresholds(request: UpdateThresholdsRequest) -> Dict[str, Any]:
    """
    Update detection thresholds.

    Allows customizing when alerts are triggered.
    """
    if not AI_DRIFT_DETECTOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="AI Drift Detector is not available"
        )

    detector = get_drift_detector()
    if not detector:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize AI Drift Detector"
        )

    # Update thresholds
    updates = request.dict(exclude_none=True)
    for key, value in updates.items():
        if key in detector.thresholds:
            detector.thresholds[key] = value

    return {
        "updated": True,
        "thresholds": detector.thresholds,
    }


@router.delete(
    "/data",
    summary="Clear Drift Data",
    description="Clear all stored drift data"
)
async def clear_data() -> Dict[str, Any]:
    """
    Clear all drift data.

    Use with caution - this removes all historical snapshots and alerts.
    """
    if not AI_DRIFT_DETECTOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="AI Drift Detector is not available"
        )

    detector = get_drift_detector()
    if not detector:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize AI Drift Detector"
        )

    detector.clear_data()

    return {
        "cleared": True,
        "message": "All drift data has been cleared",
    }


@router.get(
    "/health",
    summary="Get Current Health Score",
    description="Get the current AI code health score"
)
async def get_health_score(days: int = 7) -> Dict[str, Any]:
    """
    Get current health score.

    Quick endpoint to check overall AI code health.
    """
    if not AI_DRIFT_DETECTOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="AI Drift Detector is not available"
        )

    detector = get_drift_detector()
    if not detector:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize AI Drift Detector"
        )

    report = detector.generate_report(days)

    return {
        "health_score": report.health_score,
        "health_trend": report.health_trend,
        "alert_count": len(report.alerts),
        "critical_alerts": sum(1 for a in report.alerts if a.severity == DriftSeverity.CRITICAL),
        "period_days": days,
    }
