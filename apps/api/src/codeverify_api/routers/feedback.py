"""Feedback router for collecting user feedback on findings."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID
import hashlib

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from codeverify_api.auth.dependencies import get_current_user
from codeverify_api.db.database import get_db
from codeverify_api.db.models import User

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackCreate(BaseModel):
    """Schema for creating feedback."""
    
    finding_id: UUID = Field(..., description="ID of the finding")
    feedback_type: str = Field(
        ...,
        description="Type of feedback: 'false_positive', 'helpful', 'not_helpful', 'incorrect_severity'"
    )
    comment: str | None = Field(None, description="Optional comment")
    suggested_severity: str | None = Field(
        None,
        description="Suggested severity if incorrect_severity"
    )
    # Additional context for learning
    finding_title: str | None = Field(None, description="Title of the finding")
    finding_category: str | None = Field(None, description="Category of the finding")
    code_pattern: str | None = Field(None, description="Code pattern that triggered the finding")


class FeedbackResponse(BaseModel):
    """Schema for feedback response."""
    
    id: UUID
    finding_id: UUID
    user_id: UUID
    feedback_type: str
    comment: str | None
    suggested_severity: str | None
    created_at: datetime


class FalsePositivePattern(BaseModel):
    """A learned false positive pattern."""
    
    pattern_hash: str
    finding_title: str
    finding_category: str
    code_pattern: str | None
    occurrence_count: int
    confidence_adjustment: float
    last_seen: datetime


# In-memory storage (would use database in production)
_feedback_store: list[dict[str, Any]] = []
_false_positive_patterns: dict[str, dict[str, Any]] = {}


def _compute_pattern_hash(title: str, category: str, code_pattern: str | None) -> str:
    """Compute a hash for a finding pattern."""
    pattern_str = f"{title}:{category}:{code_pattern or ''}"
    return hashlib.sha256(pattern_str.encode()).hexdigest()[:16]


@router.post("", response_model=FeedbackResponse)
async def create_feedback(
    feedback: FeedbackCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Submit feedback on a finding.
    
    This feeds into the false positive learning system.
    """
    import uuid
    
    # Validate feedback type
    valid_types = ["false_positive", "helpful", "not_helpful", "incorrect_severity"]
    if feedback.feedback_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid feedback type. Must be one of: {valid_types}"
        )
    
    # Create feedback record
    feedback_record = {
        "id": uuid.uuid4(),
        "finding_id": feedback.finding_id,
        "user_id": current_user.id if hasattr(current_user, 'id') else uuid.uuid4(),
        "feedback_type": feedback.feedback_type,
        "comment": feedback.comment,
        "suggested_severity": feedback.suggested_severity,
        "finding_title": feedback.finding_title,
        "finding_category": feedback.finding_category,
        "code_pattern": feedback.code_pattern,
        "created_at": datetime.utcnow(),
    }
    
    _feedback_store.append(feedback_record)
    
    # Update false positive patterns if applicable
    if feedback.feedback_type == "false_positive" and feedback.finding_title:
        _update_false_positive_pattern(
            title=feedback.finding_title,
            category=feedback.finding_category or "unknown",
            code_pattern=feedback.code_pattern,
        )
    
    return feedback_record


def _update_false_positive_pattern(
    title: str,
    category: str,
    code_pattern: str | None,
) -> None:
    """Update the false positive pattern database."""
    pattern_hash = _compute_pattern_hash(title, category, code_pattern)
    
    if pattern_hash in _false_positive_patterns:
        # Increment occurrence count
        _false_positive_patterns[pattern_hash]["occurrence_count"] += 1
        _false_positive_patterns[pattern_hash]["last_seen"] = datetime.utcnow()
        
        # Adjust confidence based on occurrences
        occurrences = _false_positive_patterns[pattern_hash]["occurrence_count"]
        # More reports = lower confidence in this finding type
        _false_positive_patterns[pattern_hash]["confidence_adjustment"] = min(
            0.5, occurrences * 0.1  # Max 50% confidence reduction
        )
    else:
        # New pattern
        _false_positive_patterns[pattern_hash] = {
            "pattern_hash": pattern_hash,
            "finding_title": title,
            "finding_category": category,
            "code_pattern": code_pattern,
            "occurrence_count": 1,
            "confidence_adjustment": 0.1,
            "last_seen": datetime.utcnow(),
        }


@router.get("/finding/{finding_id}")
async def get_finding_feedback(
    finding_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """Get all feedback for a finding."""
    return [f for f in _feedback_store if f["finding_id"] == finding_id]


@router.get("/stats")
async def get_feedback_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Get feedback statistics including false positive analysis."""
    total = len(_feedback_store)
    
    by_type = {}
    for feedback in _feedback_store:
        ftype = feedback["feedback_type"]
        by_type[ftype] = by_type.get(ftype, 0) + 1
    
    false_positive_count = by_type.get("false_positive", 0)
    false_positive_rate = (
        false_positive_count / total * 100
        if total > 0 else 0
    )
    
    # Get top false positive patterns
    top_patterns = sorted(
        _false_positive_patterns.values(),
        key=lambda x: x["occurrence_count"],
        reverse=True
    )[:10]
    
    return {
        "total_feedback": total,
        "by_type": by_type,
        "false_positive_rate": round(false_positive_rate, 2),
        "false_positive_count": false_positive_count,
        "learned_patterns_count": len(_false_positive_patterns),
        "top_false_positive_patterns": top_patterns,
    }


@router.get("/patterns")
async def get_learned_patterns(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Get all learned false positive patterns.
    
    These patterns are used to adjust confidence scores in future analyses.
    """
    patterns = list(_false_positive_patterns.values())
    
    # Sort by occurrence count
    patterns.sort(key=lambda x: x["occurrence_count"], reverse=True)
    
    return {
        "total_patterns": len(patterns),
        "patterns": patterns,
        "usage": "These patterns reduce confidence scores for matching findings",
    }


@router.get("/adjustment/{finding_title}")
async def get_confidence_adjustment(
    finding_title: str,
    category: str = "unknown",
    code_pattern: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Check if a finding matches a known false positive pattern.
    
    Returns the confidence adjustment to apply.
    """
    pattern_hash = _compute_pattern_hash(finding_title, category, code_pattern)
    
    if pattern_hash in _false_positive_patterns:
        pattern = _false_positive_patterns[pattern_hash]
        return {
            "matches_pattern": True,
            "pattern_hash": pattern_hash,
            "confidence_adjustment": pattern["confidence_adjustment"],
            "occurrence_count": pattern["occurrence_count"],
            "recommendation": "Consider reducing confidence score" if pattern["occurrence_count"] > 3 else "Monitor pattern",
        }
    
    return {
        "matches_pattern": False,
        "confidence_adjustment": 0.0,
        "recommendation": "No adjustment needed",
    }


@router.post("/dismiss/{finding_id}")
async def dismiss_finding(
    finding_id: UUID,
    reason: str,
    learn_pattern: bool = True,
    finding_title: str | None = None,
    finding_category: str | None = None,
    code_pattern: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Dismiss a finding and optionally learn the pattern.
    
    This is a convenience endpoint that combines feedback creation
    with pattern learning.
    """
    import uuid
    
    # Create dismiss feedback
    feedback_record = {
        "id": uuid.uuid4(),
        "finding_id": finding_id,
        "user_id": current_user.id if hasattr(current_user, 'id') else uuid.uuid4(),
        "feedback_type": "false_positive",
        "comment": reason,
        "suggested_severity": None,
        "finding_title": finding_title,
        "finding_category": finding_category,
        "code_pattern": code_pattern,
        "created_at": datetime.utcnow(),
        "dismissed": True,
    }
    
    _feedback_store.append(feedback_record)
    
    # Learn pattern if requested
    pattern_learned = False
    if learn_pattern and finding_title:
        _update_false_positive_pattern(
            title=finding_title,
            category=finding_category or "unknown",
            code_pattern=code_pattern,
        )
        pattern_learned = True
    
    return {
        "dismissed": True,
        "finding_id": str(finding_id),
        "reason": reason,
        "pattern_learned": pattern_learned,
    }
