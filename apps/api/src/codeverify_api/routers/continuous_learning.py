"""
Continuous Learning API Router

Provides REST API endpoints for continuous learning from user feedback:
- Record feedback on findings
- Learn patterns from feedback
- Trigger model fine-tuning
- Get recommendations based on learned patterns
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Import Continuous Learning Engine
try:
    from codeverify_agents.continuous_learning import (
        ContinuousLearningEngine,
        FeedbackType,
        FindingCategory,
        LearningStatus,
    )
    CONTINUOUS_LEARNING_AVAILABLE = True
except ImportError:
    CONTINUOUS_LEARNING_AVAILABLE = False
    ContinuousLearningEngine = None
    FeedbackType = None
    FindingCategory = None
    LearningStatus = None


router = APIRouter(prefix="/api/v1/learning", tags=["continuous-learning"])

# Singleton engine instance
_learning_engine: Optional[ContinuousLearningEngine] = None


def get_learning_engine() -> ContinuousLearningEngine:
    """Get or create the learning engine singleton."""
    global _learning_engine
    if _learning_engine is None and CONTINUOUS_LEARNING_AVAILABLE:
        _learning_engine = ContinuousLearningEngine()
    return _learning_engine


# =============================================================================
# Request/Response Models
# =============================================================================


class FeedbackRequest(BaseModel):
    """Request to record feedback."""
    finding_id: str = Field(..., description="ID of the finding")
    finding_type: str = Field(..., description="Type of the finding")
    category: str = Field("other", description="Category: security, performance, style, bug, etc.")
    feedback_type: str = Field(..., description="Feedback: accepted, rejected, false_positive, etc.")
    user_id: Optional[str] = Field(None, description="User ID if available")
    code_snippet: Optional[str] = Field(None, description="Related code snippet")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class FeedbackResponse(BaseModel):
    """Response with feedback record."""
    id: str
    finding_id: str
    feedback_type: str
    recorded_at: str
    triggers_updated: bool


class LearnRequest(BaseModel):
    """Request to learn patterns."""
    hours: int = Field(168, ge=1, le=720, description="Hours of feedback to analyze")


class RecommendationRequest(BaseModel):
    """Request for a recommendation."""
    finding_type: str = Field(..., description="Type of the finding")
    code_snippet: Optional[str] = Field(None, description="Code snippet for context")


class TrainingRequest(BaseModel):
    """Request to trigger training."""
    force: bool = Field(False, description="Force training even if cooldown active")


# =============================================================================
# API Endpoints
# =============================================================================


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Record Feedback",
    description="Record user feedback on a finding"
)
async def record_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Record user feedback on a finding.

    This feedback is used to learn patterns and improve detection accuracy.
    """
    if not CONTINUOUS_LEARNING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Continuous Learning Engine is not available"
        )

    engine = get_learning_engine()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Continuous Learning Engine"
        )

    record = engine.record_feedback(
        finding_id=request.finding_id,
        finding_type=request.finding_type,
        category=request.category,
        feedback_type=request.feedback_type,
        user_id=request.user_id,
        code_snippet=request.code_snippet,
        context=request.context,
    )

    return FeedbackResponse(
        id=record.id,
        finding_id=record.finding_id,
        feedback_type=record.feedback_type.value,
        recorded_at=record.timestamp.isoformat(),
        triggers_updated=True,
    )


@router.get(
    "/feedback",
    summary="Get Recent Feedback",
    description="Get recent feedback records"
)
async def get_feedback(
    hours: int = 24,
    category: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """Get recent feedback records."""
    if not CONTINUOUS_LEARNING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Continuous Learning Engine is not available"
        )

    engine = get_learning_engine()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Continuous Learning Engine"
        )

    cat = None
    if category:
        try:
            cat = FindingCategory(category)
        except ValueError:
            pass

    records = engine.collector.get_recent_feedback(hours=hours, category=cat)
    records = records[:limit]

    return {
        "records": [r.to_dict() for r in records],
        "count": len(records),
        "hours": hours,
    }


@router.post(
    "/learn",
    summary="Learn Patterns",
    description="Learn patterns from recent feedback"
)
async def learn_patterns(request: LearnRequest) -> Dict[str, Any]:
    """
    Learn patterns from recent feedback.

    Analyzes feedback to identify false positive patterns, rejection patterns,
    and code patterns that can improve detection accuracy.
    """
    if not CONTINUOUS_LEARNING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Continuous Learning Engine is not available"
        )

    engine = get_learning_engine()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Continuous Learning Engine"
        )

    patterns = engine.learn_patterns(hours=request.hours)

    return {
        "patterns_learned": len(patterns),
        "patterns": [p.to_dict() for p in patterns],
        "hours_analyzed": request.hours,
    }


@router.get(
    "/patterns",
    summary="Get Learned Patterns",
    description="Get all learned patterns"
)
async def get_patterns(
    active_only: bool = True,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """Get learned patterns."""
    if not CONTINUOUS_LEARNING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Continuous Learning Engine is not available"
        )

    engine = get_learning_engine()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Continuous Learning Engine"
        )

    if category:
        try:
            cat = FindingCategory(category)
            patterns = engine.learner.get_patterns_by_category(cat)
        except ValueError:
            patterns = []
    elif active_only:
        patterns = engine.learner.get_active_patterns()
    else:
        patterns = list(engine.learner.patterns.values())

    return {
        "patterns": [p.to_dict() for p in patterns],
        "count": len(patterns),
    }


@router.post(
    "/patterns/{pattern_id}/deactivate",
    summary="Deactivate Pattern",
    description="Deactivate a learned pattern"
)
async def deactivate_pattern(pattern_id: str) -> Dict[str, Any]:
    """Deactivate a learned pattern."""
    if not CONTINUOUS_LEARNING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Continuous Learning Engine is not available"
        )

    engine = get_learning_engine()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Continuous Learning Engine"
        )

    success = engine.learner.deactivate_pattern(pattern_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Pattern not found: {pattern_id}"
        )

    return {
        "deactivated": True,
        "pattern_id": pattern_id,
    }


@router.post(
    "/recommend",
    summary="Get Recommendation",
    description="Get recommendation based on learned patterns"
)
async def get_recommendation(request: RecommendationRequest) -> Dict[str, Any]:
    """
    Get recommendation for a finding based on learned patterns.

    Returns whether the finding should be suppressed, highlighted, or shown normally.
    """
    if not CONTINUOUS_LEARNING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Continuous Learning Engine is not available"
        )

    engine = get_learning_engine()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Continuous Learning Engine"
        )

    recommendation = engine.get_recommendation(
        finding_type=request.finding_type,
        code_snippet=request.code_snippet,
    )

    return recommendation


@router.post(
    "/train",
    summary="Trigger Training",
    description="Trigger model fine-tuning if conditions are met"
)
async def trigger_training(request: TrainingRequest) -> Dict[str, Any]:
    """
    Trigger model fine-tuning.

    Training is triggered when sufficient feedback has been collected or
    false positive rate exceeds threshold.
    """
    if not CONTINUOUS_LEARNING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Continuous Learning Engine is not available"
        )

    engine = get_learning_engine()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Continuous Learning Engine"
        )

    if not request.force:
        can_train, reason = engine.trainer.can_train()
        if not can_train:
            return {
                "triggered": False,
                "reason": reason,
            }

    job = await engine.trigger_training()

    if not job:
        return {
            "triggered": False,
            "reason": "No active triggers or training not needed",
        }

    return {
        "triggered": True,
        "job": job.to_dict(),
    }


@router.get(
    "/train/status",
    summary="Get Training Status",
    description="Get current training status and triggers"
)
async def get_training_status() -> Dict[str, Any]:
    """Get training status and trigger information."""
    if not CONTINUOUS_LEARNING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Continuous Learning Engine is not available"
        )

    engine = get_learning_engine()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Continuous Learning Engine"
        )

    can_train, reason = engine.trainer.can_train()
    triggers = {tid: t.to_dict() for tid, t in engine.trainer.triggers.items()}

    recent_jobs = engine.trainer.get_recent_jobs(limit=5)

    return {
        "can_train": can_train,
        "reason": reason,
        "triggers": triggers,
        "recent_jobs": [j.to_dict() for j in recent_jobs],
    }


@router.get(
    "/train/jobs",
    summary="Get Training Jobs",
    description="Get training job history"
)
async def get_training_jobs(limit: int = 10) -> Dict[str, Any]:
    """Get training job history."""
    if not CONTINUOUS_LEARNING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Continuous Learning Engine is not available"
        )

    engine = get_learning_engine()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Continuous Learning Engine"
        )

    jobs = engine.trainer.get_recent_jobs(limit=limit)

    return {
        "jobs": [j.to_dict() for j in jobs],
        "count": len(jobs),
    }


@router.get(
    "/metrics",
    summary="Get Learning Metrics",
    description="Get learning metrics and improvements"
)
async def get_metrics() -> Dict[str, Any]:
    """Get learning metrics."""
    if not CONTINUOUS_LEARNING_AVAILABLE:
        return {
            "available": False,
            "message": "Continuous Learning Engine is not available",
        }

    engine = get_learning_engine()
    if not engine:
        return {
            "available": False,
            "message": "Failed to initialize engine",
        }

    metrics = engine.get_metrics()

    return {
        "available": True,
        **metrics.to_dict(),
    }


@router.get(
    "/stats",
    summary="Get Statistics",
    description="Get comprehensive learning statistics"
)
async def get_stats() -> Dict[str, Any]:
    """Get comprehensive learning statistics."""
    if not CONTINUOUS_LEARNING_AVAILABLE:
        return {
            "available": False,
            "message": "Continuous Learning Engine is not available",
        }

    engine = get_learning_engine()
    if not engine:
        return {
            "available": False,
            "message": "Failed to initialize engine",
        }

    stats = engine.get_statistics()
    stats["available"] = True

    return stats


@router.get(
    "/export",
    summary="Export Data",
    description="Export all learning data"
)
async def export_data() -> Dict[str, Any]:
    """Export all learning data."""
    if not CONTINUOUS_LEARNING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Continuous Learning Engine is not available"
        )

    engine = get_learning_engine()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Continuous Learning Engine"
        )

    return engine.export_data()


@router.delete(
    "/data",
    summary="Clear Data",
    description="Clear all learning data"
)
async def clear_data() -> Dict[str, Any]:
    """Clear all learning data."""
    if not CONTINUOUS_LEARNING_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Continuous Learning Engine is not available"
        )

    engine = get_learning_engine()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Continuous Learning Engine"
        )

    engine.clear_data()

    return {
        "cleared": True,
        "message": "All learning data cleared",
    }
