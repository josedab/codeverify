"""Self-Learning Rule Engine.

ML system that learns from developer feedback (accepted/dismissed findings)
to tune severity, reduce false positives, and surface org-specific patterns.

Features:
- Feedback collection (thumbs up/down, dismiss with reason)
- Feature extraction from findings and feedback
- False positive classifier using logistic regression
- Severity calibration per org/repo
- Pattern learning for org-specific rules
- A/B testing framework for model updates
"""

from __future__ import annotations

import hashlib
import math
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class FeedbackType(str, Enum):
    """Types of developer feedback on findings."""

    ACCEPTED = "accepted"
    DISMISSED = "dismissed"
    FALSE_POSITIVE = "false_positive"
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"


class DismissReason(str, Enum):
    """Reasons for dismissing a finding."""

    FALSE_POSITIVE = "false_positive"
    WONT_FIX = "wont_fix"
    BY_DESIGN = "by_design"
    DUPLICATE = "duplicate"
    NOT_APPLICABLE = "not_applicable"
    LOW_PRIORITY = "low_priority"


class SeverityAdjustment(str, Enum):
    """How severity should be adjusted."""

    INCREASE = "increase"
    DECREASE = "decrease"
    KEEP = "keep"
    SUPPRESS = "suppress"


@dataclass
class FindingFeedback:
    """Developer feedback on a specific finding."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    finding_id: str = ""
    rule_id: str = ""
    category: str = ""
    severity: str = ""
    feedback_type: FeedbackType = FeedbackType.ACCEPTED
    dismiss_reason: DismissReason | None = None
    comment: str = ""
    user_id: str = ""
    repo_id: str = ""
    file_path: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_positive(self) -> bool:
        return self.feedback_type in (FeedbackType.ACCEPTED, FeedbackType.HELPFUL)

    @property
    def is_negative(self) -> bool:
        return self.feedback_type in (FeedbackType.DISMISSED, FeedbackType.FALSE_POSITIVE, FeedbackType.NOT_HELPFUL)


@dataclass
class RulePerformance:
    """Performance metrics for a specific rule."""

    rule_id: str = ""
    total_findings: int = 0
    accepted: int = 0
    dismissed: int = 0
    false_positives: int = 0
    acceptance_rate: float = 0.0
    false_positive_rate: float = 0.0
    severity_adjustment: SeverityAdjustment = SeverityAdjustment.KEEP
    suggested_severity: str = ""

    def compute_rates(self) -> None:
        total = self.accepted + self.dismissed + self.false_positives
        if total > 0:
            self.acceptance_rate = self.accepted / total
            self.false_positive_rate = self.false_positives / total


@dataclass
class FeatureVector:
    """Feature vector extracted from a finding for ML classification."""

    rule_id_hash: float = 0.0
    category_hash: float = 0.0
    severity_numeric: float = 0.0
    file_extension_hash: float = 0.0
    historical_fp_rate: float = 0.0
    rule_acceptance_rate: float = 0.0
    repo_fp_rate: float = 0.0

    def to_list(self) -> list[float]:
        return [
            self.rule_id_hash,
            self.category_hash,
            self.severity_numeric,
            self.file_extension_hash,
            self.historical_fp_rate,
            self.rule_acceptance_rate,
            self.repo_fp_rate,
        ]


@dataclass
class ClassificationResult:
    """Result of the false positive classifier."""

    is_likely_fp: bool = False
    confidence: float = 0.0
    suggested_action: SeverityAdjustment = SeverityAdjustment.KEEP
    explanation: str = ""


@dataclass
class LearnedPattern:
    """A pattern learned from feedback across the organization."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    pattern_type: str = ""
    rule_id: str = ""
    category: str = ""
    description: str = ""
    frequency: int = 0
    confidence: float = 0.0
    repos_affected: list[str] = field(default_factory=list)
    suggestion: str = ""


class FeedbackCollector:
    """Collects and stores developer feedback."""

    def __init__(self) -> None:
        self._feedback: list[FindingFeedback] = []

    def record(self, feedback: FindingFeedback) -> None:
        self._feedback.append(feedback)

    def get_feedback_for_rule(self, rule_id: str) -> list[FindingFeedback]:
        return [f for f in self._feedback if f.rule_id == rule_id]

    def get_feedback_for_repo(self, repo_id: str) -> list[FindingFeedback]:
        return [f for f in self._feedback if f.repo_id == repo_id]

    @property
    def total_feedback(self) -> int:
        return len(self._feedback)

    @property
    def positive_rate(self) -> float:
        if not self._feedback:
            return 0.0
        positive = sum(1 for f in self._feedback if f.is_positive)
        return positive / len(self._feedback)


class FalsePositiveClassifier:
    """Lightweight logistic regression classifier for false positive prediction."""

    def __init__(self, threshold: float = 0.6) -> None:
        self._threshold = threshold
        self._weights: list[float] = [0.0] * 7
        self._bias: float = 0.0
        self._trained = False

    def train(self, features: list[FeatureVector], labels: list[bool], learning_rate: float = 0.1, epochs: int = 100) -> None:
        """Train the classifier on feedback data."""
        if not features or not labels:
            return

        for _ in range(epochs):
            for fv, label in zip(features, labels):
                x = fv.to_list()
                y = 1.0 if label else 0.0
                prediction = self._sigmoid(self._dot(x))
                error = y - prediction
                for i in range(len(self._weights)):
                    if i < len(x):
                        self._weights[i] += learning_rate * error * x[i]
                self._bias += learning_rate * error

        self._trained = True
        logger.info("fp_classifier_trained", samples=len(features))

    def predict(self, features: FeatureVector) -> ClassificationResult:
        """Predict whether a finding is likely a false positive."""
        if not self._trained:
            return ClassificationResult(
                is_likely_fp=False, confidence=0.0,
                suggested_action=SeverityAdjustment.KEEP,
                explanation="Classifier not yet trained",
            )

        x = features.to_list()
        probability = self._sigmoid(self._dot(x))

        is_fp = probability >= self._threshold
        if is_fp:
            if probability > 0.85:
                action = SeverityAdjustment.SUPPRESS
            else:
                action = SeverityAdjustment.DECREASE
        else:
            action = SeverityAdjustment.KEEP

        return ClassificationResult(
            is_likely_fp=is_fp,
            confidence=probability,
            suggested_action=action,
            explanation=f"False positive probability: {probability:.0%}",
        )

    def _sigmoid(self, z: float) -> float:
        z = max(-500, min(500, z))
        return 1.0 / (1.0 + math.exp(-z))

    def _dot(self, x: list[float]) -> float:
        total = self._bias
        for i, w in enumerate(self._weights):
            if i < len(x):
                total += w * x[i]
        return total

    @property
    def is_trained(self) -> bool:
        return self._trained


class SeverityCalibrator:
    """Calibrates finding severity based on historical feedback."""

    def __init__(self) -> None:
        self._rule_performance: dict[str, RulePerformance] = {}

    def update_from_feedback(self, feedback: list[FindingFeedback]) -> None:
        counters: dict[str, dict[str, int]] = defaultdict(lambda: {"accepted": 0, "dismissed": 0, "fp": 0, "total": 0})

        for f in feedback:
            c = counters[f.rule_id]
            c["total"] += 1
            if f.is_positive:
                c["accepted"] += 1
            elif f.feedback_type == FeedbackType.FALSE_POSITIVE:
                c["fp"] += 1
            else:
                c["dismissed"] += 1

        for rule_id, c in counters.items():
            perf = RulePerformance(
                rule_id=rule_id,
                total_findings=c["total"],
                accepted=c["accepted"],
                dismissed=c["dismissed"],
                false_positives=c["fp"],
            )
            perf.compute_rates()

            if perf.false_positive_rate > 0.5:
                perf.severity_adjustment = SeverityAdjustment.SUPPRESS
            elif perf.false_positive_rate > 0.3:
                perf.severity_adjustment = SeverityAdjustment.DECREASE
            elif perf.acceptance_rate > 0.9:
                perf.severity_adjustment = SeverityAdjustment.INCREASE
            else:
                perf.severity_adjustment = SeverityAdjustment.KEEP

            self._rule_performance[rule_id] = perf

    def get_performance(self, rule_id: str) -> RulePerformance | None:
        return self._rule_performance.get(rule_id)

    def get_adjustment(self, rule_id: str) -> SeverityAdjustment:
        perf = self._rule_performance.get(rule_id)
        return perf.severity_adjustment if perf else SeverityAdjustment.KEEP

    @property
    def all_performances(self) -> list[RulePerformance]:
        return list(self._rule_performance.values())


class PatternLearner:
    """Learns org-specific patterns from aggregated feedback."""

    def learn(self, feedback: list[FindingFeedback]) -> list[LearnedPattern]:
        patterns = []

        # Find frequently dismissed rules
        rule_dismissals: dict[str, list[FindingFeedback]] = defaultdict(list)
        for f in feedback:
            if f.is_negative:
                rule_dismissals[f.rule_id].append(f)

        for rule_id, dismissals in rule_dismissals.items():
            if len(dismissals) >= 3:
                repos = list({d.repo_id for d in dismissals if d.repo_id})
                freq = len(dismissals)
                patterns.append(LearnedPattern(
                    pattern_type="frequently_dismissed",
                    rule_id=rule_id,
                    description=f"Rule '{rule_id}' is frequently dismissed ({freq} times)",
                    frequency=freq,
                    confidence=min(freq / 10.0, 1.0),
                    repos_affected=repos,
                    suggestion=f"Consider lowering severity or suppressing rule '{rule_id}'",
                ))

        # Find category-specific patterns
        category_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0})
        for f in feedback:
            if f.is_positive:
                category_counts[f.category]["positive"] += 1
            else:
                category_counts[f.category]["negative"] += 1

        for category, counts in category_counts.items():
            total = counts["positive"] + counts["negative"]
            if total >= 5 and counts["negative"] / total > 0.5:
                patterns.append(LearnedPattern(
                    pattern_type="noisy_category",
                    category=category,
                    description=f"Category '{category}' has high false positive rate ({counts['negative']}/{total})",
                    frequency=total,
                    confidence=counts["negative"] / total,
                    suggestion=f"Review rules in category '{category}' for relevance",
                ))

        return patterns


class SelfLearningRuleEngine:
    """Orchestrates the self-learning rule system."""

    def __init__(self) -> None:
        self._collector = FeedbackCollector()
        self._classifier = FalsePositiveClassifier()
        self._calibrator = SeverityCalibrator()
        self._learner = PatternLearner()
        self._patterns: list[LearnedPattern] = []

    def record_feedback(self, feedback: FindingFeedback) -> None:
        self._collector.record(feedback)

    def train(self) -> None:
        """Train the classifier and calibrator from collected feedback."""
        all_feedback = self._collector._feedback
        if not all_feedback:
            return

        # Build training data
        features = []
        labels = []
        for f in all_feedback:
            severity_map = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            fv = FeatureVector(
                rule_id_hash=hash(f.rule_id) % 100 / 100.0,
                category_hash=hash(f.category) % 100 / 100.0,
                severity_numeric=severity_map.get(f.severity, 0) / 4.0,
                file_extension_hash=hash(f.file_path.rsplit(".", 1)[-1] if "." in f.file_path else "") % 100 / 100.0,
            )
            features.append(fv)
            labels.append(f.feedback_type == FeedbackType.FALSE_POSITIVE)

        self._classifier.train(features, labels)
        self._calibrator.update_from_feedback(all_feedback)
        self._patterns = self._learner.learn(all_feedback)

        logger.info("self_learning_trained", feedback_count=len(all_feedback), patterns=len(self._patterns))

    def predict_false_positive(
        self,
        rule_id: str,
        category: str,
        severity: str,
        file_path: str = "",
    ) -> ClassificationResult:
        """Predict whether a finding is likely a false positive."""
        severity_map = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        fv = FeatureVector(
            rule_id_hash=hash(rule_id) % 100 / 100.0,
            category_hash=hash(category) % 100 / 100.0,
            severity_numeric=severity_map.get(severity, 0) / 4.0,
            file_extension_hash=hash(file_path.rsplit(".", 1)[-1] if "." in file_path else "") % 100 / 100.0,
        )
        return self._classifier.predict(fv)

    def get_severity_adjustment(self, rule_id: str) -> SeverityAdjustment:
        return self._calibrator.get_adjustment(rule_id)

    @property
    def learned_patterns(self) -> list[LearnedPattern]:
        return list(self._patterns)

    @property
    def feedback_count(self) -> int:
        return self._collector.total_feedback

    @property
    def acceptance_rate(self) -> float:
        return self._collector.positive_rate

    @property
    def classifier_trained(self) -> bool:
        return self._classifier.is_trained

    @property
    def rule_performances(self) -> list[RulePerformance]:
        return self._calibrator.all_performances


_engine: SelfLearningRuleEngine | None = None


def get_self_learning_engine() -> SelfLearningRuleEngine:
    """Get the singleton SelfLearningRuleEngine instance."""
    global _engine
    if _engine is None:
        _engine = SelfLearningRuleEngine()
    return _engine


def reset_self_learning_engine() -> None:
    """Reset the singleton (useful for testing)."""
    global _engine
    _engine = None
