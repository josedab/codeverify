"""Organizational Learning Engine.

ML-powered false-positive classifier from feedback, severity
calibration, predictive quality scoring, and per-org pattern learning.

Features:
- Feedback collection (accept/dismiss/false-positive/helpful)
- Logistic-regression-style false positive classifier
- Severity calibration from historical feedback
- Per-org pattern learning (noisy rules, frequent dismissals)
- Predictive quality scoring for code changes
- A/B testing support for rule tuning
"""

from __future__ import annotations

import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

import structlog

logger = structlog.get_logger()


class FeedbackType(str, Enum):
    ACCEPTED = "accepted"
    DISMISSED = "dismissed"
    FALSE_POSITIVE = "false_positive"
    HELPFUL = "helpful"
    FIXED = "fixed"


class PredictionOutcome(str, Enum):
    HIGH_RISK = "high_risk"
    MEDIUM_RISK = "medium_risk"
    LOW_RISK = "low_risk"
    CLEAN = "clean"


@dataclass
class FindingFeedback:
    """Feedback on a specific finding."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    org_id: str = ""
    finding_category: str = ""
    finding_severity: str = ""
    rule_id: str = ""
    file_path: str = ""
    feedback_type: FeedbackType = FeedbackType.ACCEPTED
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class RulePerformance:
    """Performance metrics for a single rule."""

    rule_id: str = ""
    total_findings: int = 0
    accepted: int = 0
    dismissed: int = 0
    false_positives: int = 0
    helpful: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_findings == 0:
            return 0.0
        return round(self.accepted / self.total_findings, 3)

    @property
    def fp_rate(self) -> float:
        if self.total_findings == 0:
            return 0.0
        return round(self.false_positives / self.total_findings, 3)


@dataclass
class SeverityCalibration:
    """Calibrated severity for a rule based on feedback."""

    rule_id: str = ""
    original_severity: str = ""
    calibrated_severity: str = ""
    confidence: float = 0.0
    sample_size: int = 0


@dataclass
class QualityPrediction:
    """Predicted quality outcome for a code change."""

    file_path: str = ""
    outcome: PredictionOutcome = PredictionOutcome.MEDIUM_RISK
    risk_score: float = 0.5
    predicted_findings: int = 0
    confidence: float = 0.5
    factors: dict[str, float] = field(default_factory=dict)


@dataclass
class OrgLearningProfile:
    """Learned profile for an organization."""

    org_id: str = ""
    total_feedback: int = 0
    rule_performance: dict[str, RulePerformance] = field(default_factory=dict)
    severity_calibrations: list[SeverityCalibration] = field(default_factory=list)
    suppressed_rules: list[str] = field(default_factory=list)
    noisy_categories: list[str] = field(default_factory=list)
    model_weights: dict[str, float] = field(default_factory=dict)
    last_trained: datetime | None = None


class FPClassifier:
    """Simple false-positive classifier using weighted features."""

    DEFAULT_WEIGHTS = {
        "historical_fp_rate": -2.0,
        "acceptance_rate": 1.5,
        "severity_weight": 1.0,
        "file_risk": 0.5,
        "rule_age": -0.3,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights = weights or dict(self.DEFAULT_WEIGHTS)

    def predict_fp_probability(self, features: dict[str, float]) -> float:
        """Predict probability of false positive (0-1)."""
        score = 0.0
        for feat, val in features.items():
            weight = self._weights.get(feat, 0.0)
            score += weight * val
        prob = 1.0 / (1.0 + math.exp(-score))  # sigmoid
        return round(prob, 3)

    def should_suppress(self, features: dict[str, float], threshold: float = 0.7) -> bool:
        return self.predict_fp_probability(features) >= threshold

    def update_weights(self, feedback_data: list[tuple[dict[str, float], bool]]) -> None:
        """Simple weight update from labeled data (gradient-style)."""
        lr = 0.01
        for features, is_fp in feedback_data:
            pred = self.predict_fp_probability(features)
            target = 1.0 if is_fp else 0.0
            error = pred - target
            for feat, val in features.items():
                if feat in self._weights:
                    self._weights[feat] -= lr * error * val


class SeverityCalibrator:
    """Calibrates finding severity based on historical feedback."""

    SEVERITY_LEVELS = ["critical", "high", "medium", "low", "info"]

    def calibrate(
        self, rule_id: str, original: str, feedback_history: list[FindingFeedback]
    ) -> SeverityCalibration:
        """Calibrate severity based on feedback patterns."""
        if not feedback_history:
            return SeverityCalibration(
                rule_id=rule_id,
                original_severity=original,
                calibrated_severity=original,
                confidence=0.0,
                sample_size=0,
            )

        fp_count = sum(
            1 for f in feedback_history if f.feedback_type == FeedbackType.FALSE_POSITIVE
        )
        dismiss_count = sum(
            1 for f in feedback_history if f.feedback_type == FeedbackType.DISMISSED
        )
        total = len(feedback_history)
        noise_rate = (fp_count + dismiss_count) / total

        orig_idx = self.SEVERITY_LEVELS.index(original) if original in self.SEVERITY_LEVELS else 2
        if noise_rate > 0.6 and orig_idx < len(self.SEVERITY_LEVELS) - 1:
            calibrated = self.SEVERITY_LEVELS[orig_idx + 1]
        elif noise_rate < 0.1 and orig_idx > 0:
            calibrated = self.SEVERITY_LEVELS[orig_idx - 1]
        else:
            calibrated = original

        return SeverityCalibration(
            rule_id=rule_id,
            original_severity=original,
            calibrated_severity=calibrated,
            confidence=round(1.0 - noise_rate, 3),
            sample_size=total,
        )


class QualityPredictor:
    """Predicts quality outcomes for code changes."""

    def predict(
        self,
        file_path: str,
        change_size: int,
        author_history: dict[str, float] | None = None,
        file_history: dict[str, float] | None = None,
    ) -> QualityPrediction:
        """Predict quality outcome for a file change."""
        factors: dict[str, float] = {}

        factors["change_size"] = min(1.0, change_size / 500)
        factors["author_fp_rate"] = (author_history or {}).get("fp_rate", 0.5)
        factors["file_bug_rate"] = (file_history or {}).get("bug_rate", 0.3)

        risk_score = sum(factors.values()) / len(factors) if factors else 0.5

        if risk_score >= 0.7:
            outcome = PredictionOutcome.HIGH_RISK
        elif risk_score >= 0.4:
            outcome = PredictionOutcome.MEDIUM_RISK
        elif risk_score >= 0.1:
            outcome = PredictionOutcome.LOW_RISK
        else:
            outcome = PredictionOutcome.CLEAN

        predicted_findings = int(risk_score * 10 * (change_size / 100))

        return QualityPrediction(
            file_path=file_path,
            outcome=outcome,
            risk_score=round(risk_score, 3),
            predicted_findings=predicted_findings,
            confidence=0.6,
            factors=factors,
        )


class OrgLearningService:
    """Main service for organizational learning engine."""

    def __init__(self) -> None:
        self._profiles: dict[str, OrgLearningProfile] = {}
        self._classifier = FPClassifier()
        self._calibrator = SeverityCalibrator()
        self._predictor = QualityPredictor()
        self._all_feedback: list[FindingFeedback] = []

    def record_feedback(self, feedback: FindingFeedback) -> None:
        """Record finding feedback and update org profile."""
        self._all_feedback.append(feedback)
        profile = self._get_or_create_profile(feedback.org_id)
        profile.total_feedback += 1

        rule_perf = profile.rule_performance.get(feedback.rule_id)
        if not rule_perf:
            rule_perf = RulePerformance(rule_id=feedback.rule_id)
            profile.rule_performance[feedback.rule_id] = rule_perf

        rule_perf.total_findings += 1
        if feedback.feedback_type == FeedbackType.ACCEPTED:
            rule_perf.accepted += 1
        elif feedback.feedback_type == FeedbackType.DISMISSED:
            rule_perf.dismissed += 1
        elif feedback.feedback_type == FeedbackType.FALSE_POSITIVE:
            rule_perf.false_positives += 1
        elif feedback.feedback_type == FeedbackType.HELPFUL:
            rule_perf.helpful += 1

    def train(self, org_id: str) -> OrgLearningProfile:
        """Train the model for an org based on accumulated feedback."""
        profile = self._get_or_create_profile(org_id)
        org_feedback = [f for f in self._all_feedback if f.org_id == org_id]

        # Calibrate severities
        rule_feedback: dict[str, list[FindingFeedback]] = defaultdict(list)
        for f in org_feedback:
            rule_feedback[f.rule_id].append(f)

        calibrations: list[SeverityCalibration] = []
        for rule_id, feedbacks in rule_feedback.items():
            severity = feedbacks[0].finding_severity if feedbacks else "medium"
            cal = self._calibrator.calibrate(rule_id, severity, feedbacks)
            calibrations.append(cal)

        profile.severity_calibrations = calibrations

        # Identify noisy rules
        noisy = [r_id for r_id, rp in profile.rule_performance.items() if rp.fp_rate > 0.5]
        profile.noisy_categories = noisy
        profile.suppressed_rules = [
            r_id for r_id, rp in profile.rule_performance.items() if rp.fp_rate > 0.8
        ]
        profile.last_trained = datetime.now(UTC)

        return profile

    def should_suppress(
        self, org_id: str, rule_id: str, features: dict[str, float] | None = None
    ) -> bool:
        """Check if a finding should be suppressed based on learning."""
        profile = self._profiles.get(org_id)
        if profile and rule_id in profile.suppressed_rules:
            return True
        if features:
            return self._classifier.should_suppress(features)
        return False

    def predict_quality(
        self, file_path: str, change_size: int, _org_id: str = ""
    ) -> QualityPrediction:
        return self._predictor.predict(file_path, change_size)

    def get_profile(self, org_id: str) -> OrgLearningProfile | None:
        return self._profiles.get(org_id)

    def get_rule_stats(self, org_id: str) -> list[RulePerformance]:
        profile = self._profiles.get(org_id)
        if not profile:
            return []
        return list(profile.rule_performance.values())

    def _get_or_create_profile(self, org_id: str) -> OrgLearningProfile:
        if org_id not in self._profiles:
            self._profiles[org_id] = OrgLearningProfile(org_id=org_id)
        return self._profiles[org_id]


# ─── Singleton Access ──────────────────────────────────────────────────


_org_learning_instance: OrgLearningService | None = None


def get_org_learning_service() -> OrgLearningService:
    global _org_learning_instance
    if _org_learning_instance is None:
        _org_learning_instance = OrgLearningService()
    return _org_learning_instance


def reset_org_learning_service() -> None:
    global _org_learning_instance
    _org_learning_instance = None
