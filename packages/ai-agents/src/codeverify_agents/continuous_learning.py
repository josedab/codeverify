"""
Continuous Learning Engine

Learns from user feedback (accepted/rejected findings, false positives)
to improve detection accuracy over time. Tracks patterns and triggers
model fine-tuning when appropriate.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4


class FeedbackType(str, Enum):
    """Types of user feedback."""
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"


class FindingCategory(str, Enum):
    """Categories of findings."""
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    BUG = "bug"
    COMPLEXITY = "complexity"
    DOCUMENTATION = "documentation"
    AI_GENERATED = "ai_generated"
    OTHER = "other"


class LearningStatus(str, Enum):
    """Status of learning operations."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FeedbackRecord:
    """Record of user feedback on a finding."""
    id: str
    finding_id: str
    finding_type: str
    finding_category: FindingCategory
    feedback_type: FeedbackType
    user_id: Optional[str]
    code_snippet: Optional[str]
    context: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "finding_id": self.finding_id,
            "finding_type": self.finding_type,
            "finding_category": self.finding_category.value,
            "feedback_type": self.feedback_type.value,
            "user_id": self.user_id,
            "code_snippet": self.code_snippet,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class LearnedPattern:
    """A pattern learned from feedback."""
    id: str
    pattern_type: str
    description: str
    category: FindingCategory
    confidence: float
    support_count: int  # Number of feedback records supporting this pattern
    examples: List[str]
    learned_at: datetime
    last_updated: datetime
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "category": self.category.value,
            "confidence": self.confidence,
            "support_count": self.support_count,
            "examples": self.examples[:5],  # Limit examples
            "learned_at": self.learned_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "is_active": self.is_active,
        }


@dataclass
class LearningMetrics:
    """Metrics for the learning engine."""
    total_feedback: int
    feedback_by_type: Dict[str, int]
    feedback_by_category: Dict[str, int]
    patterns_learned: int
    active_patterns: int
    accuracy_improvement: float
    false_positive_reduction: float
    last_training_time: Optional[datetime]
    next_training_eligible: Optional[datetime]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_feedback": self.total_feedback,
            "feedback_by_type": self.feedback_by_type,
            "feedback_by_category": self.feedback_by_category,
            "patterns_learned": self.patterns_learned,
            "active_patterns": self.active_patterns,
            "accuracy_improvement": self.accuracy_improvement,
            "false_positive_reduction": self.false_positive_reduction,
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
            "next_training_eligible": self.next_training_eligible.isoformat() if self.next_training_eligible else None,
        }


@dataclass
class TrainingTrigger:
    """Criteria for triggering model fine-tuning."""
    id: str
    trigger_type: str
    threshold: float
    current_value: float
    is_triggered: bool
    triggered_at: Optional[datetime]
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "trigger_type": self.trigger_type,
            "threshold": self.threshold,
            "current_value": self.current_value,
            "is_triggered": self.is_triggered,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "description": self.description,
        }


@dataclass
class TrainingJob:
    """A model fine-tuning job."""
    id: str
    status: LearningStatus
    trigger_id: str
    feedback_count: int
    patterns_included: int
    started_at: datetime
    completed_at: Optional[datetime]
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "status": self.status.value,
            "trigger_id": self.trigger_id,
            "feedback_count": self.feedback_count,
            "patterns_included": self.patterns_included,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metrics": self.metrics,
            "error": self.error,
        }


class FeedbackCollector:
    """Collects and stores user feedback."""

    def __init__(self):
        self.feedback_records: Dict[str, FeedbackRecord] = {}
        self.feedback_by_finding: Dict[str, List[str]] = defaultdict(list)
        self.feedback_by_user: Dict[str, List[str]] = defaultdict(list)
        self.feedback_by_category: Dict[str, List[str]] = defaultdict(list)

    def record_feedback(
        self,
        finding_id: str,
        finding_type: str,
        category: FindingCategory,
        feedback_type: FeedbackType,
        user_id: Optional[str] = None,
        code_snippet: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeedbackRecord:
        """Record user feedback on a finding."""
        record_id = str(uuid4())

        record = FeedbackRecord(
            id=record_id,
            finding_id=finding_id,
            finding_type=finding_type,
            finding_category=category,
            feedback_type=feedback_type,
            user_id=user_id,
            code_snippet=code_snippet,
            context=context or {},
            timestamp=datetime.now(),
            metadata=metadata or {},
        )

        self.feedback_records[record_id] = record
        self.feedback_by_finding[finding_id].append(record_id)
        if user_id:
            self.feedback_by_user[user_id].append(record_id)
        self.feedback_by_category[category.value].append(record_id)

        return record

    def get_feedback(self, record_id: str) -> Optional[FeedbackRecord]:
        """Get a feedback record by ID."""
        return self.feedback_records.get(record_id)

    def get_feedback_for_finding(self, finding_id: str) -> List[FeedbackRecord]:
        """Get all feedback for a finding."""
        record_ids = self.feedback_by_finding.get(finding_id, [])
        return [self.feedback_records[rid] for rid in record_ids if rid in self.feedback_records]

    def get_feedback_by_category(self, category: FindingCategory) -> List[FeedbackRecord]:
        """Get all feedback for a category."""
        record_ids = self.feedback_by_category.get(category.value, [])
        return [self.feedback_records[rid] for rid in record_ids if rid in self.feedback_records]

    def get_recent_feedback(
        self,
        hours: int = 24,
        category: Optional[FindingCategory] = None,
    ) -> List[FeedbackRecord]:
        """Get recent feedback records."""
        cutoff = datetime.now() - timedelta(hours=hours)
        records = []

        for record in self.feedback_records.values():
            if record.timestamp >= cutoff:
                if category is None or record.finding_category == category:
                    records.append(record)

        return sorted(records, key=lambda r: r.timestamp, reverse=True)

    def get_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        by_type: Dict[str, int] = defaultdict(int)
        by_category: Dict[str, int] = defaultdict(int)

        for record in self.feedback_records.values():
            by_type[record.feedback_type.value] += 1
            by_category[record.finding_category.value] += 1

        return {
            "total_records": len(self.feedback_records),
            "by_type": dict(by_type),
            "by_category": dict(by_category),
            "unique_findings": len(self.feedback_by_finding),
            "unique_users": len(self.feedback_by_user),
        }


class PatternLearner:
    """Learns patterns from feedback data."""

    def __init__(self, min_support: int = 3, min_confidence: float = 0.7):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.patterns: Dict[str, LearnedPattern] = {}
        self.pattern_by_category: Dict[str, List[str]] = defaultdict(list)

    def analyze_feedback(
        self,
        feedback_records: List[FeedbackRecord],
    ) -> List[LearnedPattern]:
        """Analyze feedback records to learn patterns."""
        # Group feedback by finding type and outcome
        type_outcomes: Dict[str, Dict[str, List[FeedbackRecord]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for record in feedback_records:
            type_outcomes[record.finding_type][record.feedback_type.value].append(record)

        new_patterns = []

        for finding_type, outcomes in type_outcomes.items():
            # Check for false positive patterns
            false_positives = outcomes.get("false_positive", [])
            if len(false_positives) >= self.min_support:
                pattern = self._create_pattern(
                    finding_type,
                    "false_positive_pattern",
                    false_positives,
                    f"Finding type '{finding_type}' frequently marked as false positive",
                )
                if pattern:
                    new_patterns.append(pattern)

            # Check for rejection patterns
            rejected = outcomes.get("rejected", [])
            if len(rejected) >= self.min_support:
                pattern = self._create_pattern(
                    finding_type,
                    "rejection_pattern",
                    rejected,
                    f"Finding type '{finding_type}' frequently rejected",
                )
                if pattern:
                    new_patterns.append(pattern)

            # Check for acceptance patterns
            accepted = outcomes.get("accepted", [])
            if len(accepted) >= self.min_support:
                pattern = self._create_pattern(
                    finding_type,
                    "acceptance_pattern",
                    accepted,
                    f"Finding type '{finding_type}' frequently accepted",
                )
                if pattern:
                    new_patterns.append(pattern)

        # Analyze code snippet patterns
        snippet_patterns = self._analyze_code_snippets(feedback_records)
        new_patterns.extend(snippet_patterns)

        return new_patterns

    def _create_pattern(
        self,
        finding_type: str,
        pattern_type: str,
        records: List[FeedbackRecord],
        description: str,
    ) -> Optional[LearnedPattern]:
        """Create a learned pattern from feedback records."""
        if len(records) < self.min_support:
            return None

        # Determine dominant category
        category_counts: Dict[str, int] = defaultdict(int)
        for record in records:
            category_counts[record.finding_category.value] += 1

        dominant_category = max(category_counts.items(), key=lambda x: x[1])[0]

        # Calculate confidence
        total_for_type = len(records)
        confidence = min(1.0, total_for_type / (self.min_support * 2))

        if confidence < self.min_confidence:
            return None

        # Create pattern
        pattern_id = str(uuid4())
        examples = [r.code_snippet for r in records[:5] if r.code_snippet]

        pattern = LearnedPattern(
            id=pattern_id,
            pattern_type=pattern_type,
            description=description,
            category=FindingCategory(dominant_category),
            confidence=confidence,
            support_count=len(records),
            examples=examples,
            learned_at=datetime.now(),
            last_updated=datetime.now(),
        )

        self.patterns[pattern_id] = pattern
        self.pattern_by_category[dominant_category].append(pattern_id)

        return pattern

    def _analyze_code_snippets(
        self,
        records: List[FeedbackRecord],
    ) -> List[LearnedPattern]:
        """Analyze code snippets for common patterns."""
        patterns = []

        # Group by code snippet hash
        snippet_groups: Dict[str, List[FeedbackRecord]] = defaultdict(list)

        for record in records:
            if record.code_snippet:
                # Normalize and hash the snippet
                normalized = self._normalize_code(record.code_snippet)
                snippet_hash = hashlib.md5(normalized.encode()).hexdigest()[:16]
                snippet_groups[snippet_hash].append(record)

        # Find patterns in frequently occurring snippets
        for snippet_hash, group in snippet_groups.items():
            if len(group) >= self.min_support:
                # Check if predominantly false positives
                fp_count = sum(1 for r in group if r.feedback_type == FeedbackType.FALSE_POSITIVE)
                if fp_count / len(group) >= self.min_confidence:
                    pattern = LearnedPattern(
                        id=str(uuid4()),
                        pattern_type="code_pattern_fp",
                        description=f"Code pattern frequently marked as false positive",
                        category=group[0].finding_category,
                        confidence=fp_count / len(group),
                        support_count=len(group),
                        examples=[r.code_snippet for r in group[:3] if r.code_snippet],
                        learned_at=datetime.now(),
                        last_updated=datetime.now(),
                    )
                    patterns.append(pattern)
                    self.patterns[pattern.id] = pattern

        return patterns

    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison."""
        # Remove whitespace and normalize
        lines = code.strip().split("\n")
        normalized = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                normalized.append(stripped)
        return "\n".join(normalized)

    def get_pattern(self, pattern_id: str) -> Optional[LearnedPattern]:
        """Get a pattern by ID."""
        return self.patterns.get(pattern_id)

    def get_patterns_by_category(self, category: FindingCategory) -> List[LearnedPattern]:
        """Get patterns for a category."""
        pattern_ids = self.pattern_by_category.get(category.value, [])
        return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]

    def get_active_patterns(self) -> List[LearnedPattern]:
        """Get all active patterns."""
        return [p for p in self.patterns.values() if p.is_active]

    def deactivate_pattern(self, pattern_id: str) -> bool:
        """Deactivate a pattern."""
        pattern = self.patterns.get(pattern_id)
        if pattern:
            pattern.is_active = False
            return True
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get pattern statistics."""
        by_type: Dict[str, int] = defaultdict(int)
        by_category: Dict[str, int] = defaultdict(int)

        for pattern in self.patterns.values():
            by_type[pattern.pattern_type] += 1
            by_category[pattern.category.value] += 1

        active_count = sum(1 for p in self.patterns.values() if p.is_active)

        return {
            "total_patterns": len(self.patterns),
            "active_patterns": active_count,
            "by_type": dict(by_type),
            "by_category": dict(by_category),
            "avg_confidence": sum(p.confidence for p in self.patterns.values()) / max(1, len(self.patterns)),
        }


class TrainingManager:
    """Manages model fine-tuning triggers and jobs."""

    def __init__(
        self,
        feedback_threshold: int = 100,
        fp_rate_threshold: float = 0.2,
        training_cooldown_hours: int = 24,
    ):
        self.feedback_threshold = feedback_threshold
        self.fp_rate_threshold = fp_rate_threshold
        self.training_cooldown_hours = training_cooldown_hours

        self.triggers: Dict[str, TrainingTrigger] = {}
        self.jobs: Dict[str, TrainingJob] = {}
        self.last_training_time: Optional[datetime] = None

        # Initialize default triggers
        self._init_default_triggers()

    def _init_default_triggers(self):
        """Initialize default training triggers."""
        self.triggers["feedback_volume"] = TrainingTrigger(
            id="feedback_volume",
            trigger_type="volume",
            threshold=self.feedback_threshold,
            current_value=0,
            is_triggered=False,
            triggered_at=None,
            description=f"Trigger when feedback count reaches {self.feedback_threshold}",
        )

        self.triggers["fp_rate"] = TrainingTrigger(
            id="fp_rate",
            trigger_type="rate",
            threshold=self.fp_rate_threshold,
            current_value=0,
            is_triggered=False,
            triggered_at=None,
            description=f"Trigger when false positive rate exceeds {self.fp_rate_threshold*100}%",
        )

        self.triggers["pattern_drift"] = TrainingTrigger(
            id="pattern_drift",
            trigger_type="drift",
            threshold=0.15,
            current_value=0,
            is_triggered=False,
            triggered_at=None,
            description="Trigger when pattern accuracy drifts significantly",
        )

    def update_triggers(
        self,
        feedback_count: int,
        fp_rate: float,
        pattern_drift: float = 0.0,
    ) -> List[TrainingTrigger]:
        """Update trigger values and check if any are triggered."""
        triggered = []

        # Update feedback volume trigger
        vol_trigger = self.triggers["feedback_volume"]
        vol_trigger.current_value = feedback_count
        if feedback_count >= vol_trigger.threshold and not vol_trigger.is_triggered:
            vol_trigger.is_triggered = True
            vol_trigger.triggered_at = datetime.now()
            triggered.append(vol_trigger)

        # Update FP rate trigger
        fp_trigger = self.triggers["fp_rate"]
        fp_trigger.current_value = fp_rate
        if fp_rate >= fp_trigger.threshold and not fp_trigger.is_triggered:
            fp_trigger.is_triggered = True
            fp_trigger.triggered_at = datetime.now()
            triggered.append(fp_trigger)

        # Update pattern drift trigger
        drift_trigger = self.triggers["pattern_drift"]
        drift_trigger.current_value = pattern_drift
        if pattern_drift >= drift_trigger.threshold and not drift_trigger.is_triggered:
            drift_trigger.is_triggered = True
            drift_trigger.triggered_at = datetime.now()
            triggered.append(drift_trigger)

        return triggered

    def can_train(self) -> Tuple[bool, str]:
        """Check if training is allowed."""
        if self.last_training_time:
            cooldown_end = self.last_training_time + timedelta(hours=self.training_cooldown_hours)
            if datetime.now() < cooldown_end:
                remaining = cooldown_end - datetime.now()
                return False, f"Training cooldown active. {remaining.seconds // 3600} hours remaining."

        # Check if any trigger is active
        active_triggers = [t for t in self.triggers.values() if t.is_triggered]
        if not active_triggers:
            return False, "No triggers are active"

        return True, "Training is allowed"

    def create_training_job(
        self,
        trigger_id: str,
        feedback_count: int,
        patterns_count: int,
    ) -> TrainingJob:
        """Create a new training job."""
        job = TrainingJob(
            id=str(uuid4()),
            status=LearningStatus.PENDING,
            trigger_id=trigger_id,
            feedback_count=feedback_count,
            patterns_included=patterns_count,
            started_at=datetime.now(),
            completed_at=None,
        )

        self.jobs[job.id] = job
        return job

    async def run_training(
        self,
        job_id: str,
        feedback_records: List[FeedbackRecord],
        patterns: List[LearnedPattern],
    ) -> TrainingJob:
        """Run a training job (simulated)."""
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        job.status = LearningStatus.PROCESSING

        try:
            # Simulate training process
            # In production, this would call the actual fine-tuning pipeline

            # Calculate improvement metrics
            fp_before = sum(1 for r in feedback_records if r.feedback_type == FeedbackType.FALSE_POSITIVE)
            fp_rate_before = fp_before / max(1, len(feedback_records))

            # Simulated improvement
            improvement = min(0.3, len(patterns) * 0.02)

            job.metrics = {
                "feedback_processed": len(feedback_records),
                "patterns_applied": len(patterns),
                "fp_rate_before": fp_rate_before,
                "fp_rate_after": max(0, fp_rate_before - improvement),
                "accuracy_improvement": improvement,
            }

            job.status = LearningStatus.COMPLETED
            job.completed_at = datetime.now()

            # Reset triggers
            for trigger in self.triggers.values():
                trigger.is_triggered = False
                trigger.current_value = 0

            self.last_training_time = datetime.now()

        except Exception as e:
            job.status = LearningStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()

        return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get a training job by ID."""
        return self.jobs.get(job_id)

    def get_recent_jobs(self, limit: int = 10) -> List[TrainingJob]:
        """Get recent training jobs."""
        jobs = sorted(self.jobs.values(), key=lambda j: j.started_at, reverse=True)
        return jobs[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        completed = [j for j in self.jobs.values() if j.status == LearningStatus.COMPLETED]

        total_improvement = sum(j.metrics.get("accuracy_improvement", 0) for j in completed)
        avg_improvement = total_improvement / max(1, len(completed))

        return {
            "total_jobs": len(self.jobs),
            "completed_jobs": len(completed),
            "failed_jobs": sum(1 for j in self.jobs.values() if j.status == LearningStatus.FAILED),
            "avg_improvement": avg_improvement,
            "last_training": self.last_training_time.isoformat() if self.last_training_time else None,
            "triggers": {tid: t.to_dict() for tid, t in self.triggers.items()},
        }


class ContinuousLearningEngine:
    """Main engine for continuous learning from feedback."""

    def __init__(
        self,
        min_pattern_support: int = 3,
        min_pattern_confidence: float = 0.7,
        feedback_threshold: int = 100,
        fp_rate_threshold: float = 0.2,
    ):
        self.collector = FeedbackCollector()
        self.learner = PatternLearner(min_pattern_support, min_pattern_confidence)
        self.trainer = TrainingManager(feedback_threshold, fp_rate_threshold)

        self._accuracy_history: List[Tuple[datetime, float]] = []
        self._fp_rate_history: List[Tuple[datetime, float]] = []

    def record_feedback(
        self,
        finding_id: str,
        finding_type: str,
        category: str,
        feedback_type: str,
        user_id: Optional[str] = None,
        code_snippet: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> FeedbackRecord:
        """Record user feedback."""
        cat = FindingCategory(category) if category in [c.value for c in FindingCategory] else FindingCategory.OTHER
        fb_type = FeedbackType(feedback_type)

        record = self.collector.record_feedback(
            finding_id=finding_id,
            finding_type=finding_type,
            category=cat,
            feedback_type=fb_type,
            user_id=user_id,
            code_snippet=code_snippet,
            context=context,
        )

        # Update triggers
        stats = self.collector.get_statistics()
        total = stats["total_records"]
        fp_count = stats["by_type"].get("false_positive", 0)
        fp_rate = fp_count / max(1, total)

        self.trainer.update_triggers(total, fp_rate)

        return record

    def learn_patterns(self, hours: int = 168) -> List[LearnedPattern]:
        """Learn patterns from recent feedback."""
        recent = self.collector.get_recent_feedback(hours=hours)
        if not recent:
            return []

        patterns = self.learner.analyze_feedback(recent)
        return patterns

    async def trigger_training(self) -> Optional[TrainingJob]:
        """Trigger training if conditions are met."""
        can_train, reason = self.trainer.can_train()
        if not can_train:
            return None

        # Get active trigger
        active_triggers = [t for t in self.trainer.triggers.values() if t.is_triggered]
        if not active_triggers:
            return None

        trigger = active_triggers[0]

        # Get all feedback and patterns
        feedback = list(self.collector.feedback_records.values())
        patterns = self.learner.get_active_patterns()

        # Create and run job
        job = self.trainer.create_training_job(
            trigger_id=trigger.id,
            feedback_count=len(feedback),
            patterns_count=len(patterns),
        )

        job = await self.trainer.run_training(job.id, feedback, patterns)

        # Record improvement
        if job.status == LearningStatus.COMPLETED:
            improvement = job.metrics.get("accuracy_improvement", 0)
            self._accuracy_history.append((datetime.now(), improvement))

        return job

    def get_recommendation(self, finding_type: str, code_snippet: Optional[str] = None) -> Dict[str, Any]:
        """Get recommendation based on learned patterns."""
        patterns = self.learner.get_active_patterns()

        # Find relevant patterns
        relevant = []
        for pattern in patterns:
            if pattern.pattern_type == "false_positive_pattern":
                if finding_type in pattern.description:
                    relevant.append({
                        "type": "suppress",
                        "confidence": pattern.confidence,
                        "reason": pattern.description,
                    })
            elif pattern.pattern_type == "acceptance_pattern":
                if finding_type in pattern.description:
                    relevant.append({
                        "type": "highlight",
                        "confidence": pattern.confidence,
                        "reason": pattern.description,
                    })

        if not relevant:
            return {"recommendation": "standard", "patterns_checked": len(patterns)}

        # Return highest confidence recommendation
        best = max(relevant, key=lambda r: r["confidence"])
        return {
            "recommendation": best["type"],
            "confidence": best["confidence"],
            "reason": best["reason"],
            "patterns_checked": len(patterns),
        }

    def get_metrics(self) -> LearningMetrics:
        """Get learning metrics."""
        collector_stats = self.collector.get_statistics()
        learner_stats = self.learner.get_statistics()
        trainer_stats = self.trainer.get_statistics()

        # Calculate improvements
        total_improvement = sum(imp for _, imp in self._accuracy_history)

        fp_stats = collector_stats["by_type"]
        initial_fp = fp_stats.get("false_positive", 0)
        total_feedback = collector_stats["total_records"]
        fp_reduction = min(1.0, total_improvement) if initial_fp > 0 else 0

        next_eligible = None
        if self.trainer.last_training_time:
            next_eligible = self.trainer.last_training_time + timedelta(
                hours=self.trainer.training_cooldown_hours
            )

        return LearningMetrics(
            total_feedback=total_feedback,
            feedback_by_type=fp_stats,
            feedback_by_category=collector_stats["by_category"],
            patterns_learned=learner_stats["total_patterns"],
            active_patterns=learner_stats["active_patterns"],
            accuracy_improvement=total_improvement,
            false_positive_reduction=fp_reduction,
            last_training_time=self.trainer.last_training_time,
            next_training_eligible=next_eligible,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "feedback": self.collector.get_statistics(),
            "patterns": self.learner.get_statistics(),
            "training": self.trainer.get_statistics(),
            "metrics": self.get_metrics().to_dict(),
        }

    def export_data(self) -> Dict[str, Any]:
        """Export all learning data."""
        return {
            "feedback": [r.to_dict() for r in self.collector.feedback_records.values()],
            "patterns": [p.to_dict() for p in self.learner.patterns.values()],
            "jobs": [j.to_dict() for j in self.trainer.jobs.values()],
            "metrics": self.get_metrics().to_dict(),
        }

    def clear_data(self):
        """Clear all learning data."""
        self.collector = FeedbackCollector()
        self.learner = PatternLearner()
        self.trainer = TrainingManager()
        self._accuracy_history = []
        self._fp_rate_history = []
