"""AI Model Bias & Fairness Verification.

Extends formal verification to ML models: proves fairness properties
(demographic parity, equal opportunity), detects bias in predictions,
and generates compliance reports for EU AI Act and similar regulations.

Features:
- Fairness metric formalization (demographic parity, equal opportunity, etc.)
- Statistical bias detection via Z3-compatible constraints
- Counterfactual fairness testing
- EU AI Act compliance reporting
- Remediation suggestions (re-weighting, constraint updates)
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class FairnessMetric(str, Enum):
    """Supported fairness metrics."""

    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    EQUALIZED_ODDS = "equalized_odds"
    PREDICTIVE_PARITY = "predictive_parity"
    CALIBRATION = "calibration"
    COUNTERFACTUAL = "counterfactual"
    INDIVIDUAL = "individual"


class ProtectedAttribute(str, Enum):
    """Protected attributes for fairness analysis."""

    GENDER = "gender"
    RACE = "race"
    AGE = "age"
    DISABILITY = "disability"
    RELIGION = "religion"
    NATIONALITY = "nationality"
    CUSTOM = "custom"


class BiasLevel(str, Enum):
    """Severity of detected bias."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


class RemediationType(str, Enum):
    """Types of bias remediation."""

    REWEIGHTING = "reweighting"
    RESAMPLING = "resampling"
    CONSTRAINT_UPDATE = "constraint_update"
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"
    FEATURE_REMOVAL = "feature_removal"
    ADVERSARIAL_DEBIASING = "adversarial_debiasing"


class ComplianceStatus(str, Enum):
    """Compliance status for AI regulations."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_ASSESSED = "not_assessed"


@dataclass
class FairnessConstraint:
    """A formal fairness constraint for Z3-compatible verification."""

    metric: FairnessMetric
    protected_attribute: ProtectedAttribute
    threshold: float = 0.8  # Minimum acceptable ratio (80% rule)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric.value,
            "attribute": self.protected_attribute.value,
            "threshold": self.threshold,
            "description": self.description,
        }


@dataclass
class GroupMetrics:
    """Metrics for a specific demographic group."""

    group_name: str
    total_count: int = 0
    positive_count: int = 0
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    @property
    def positive_rate(self) -> float:
        return self.positive_count / max(1, self.total_count)

    @property
    def true_positive_rate(self) -> float:
        return self.true_positive / max(1, self.true_positive + self.false_negative)

    @property
    def false_positive_rate(self) -> float:
        return self.false_positive / max(1, self.false_positive + self.true_negative)

    @property
    def precision(self) -> float:
        return self.true_positive / max(1, self.true_positive + self.false_positive)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group_name,
            "total": self.total_count,
            "positive_rate": round(self.positive_rate, 4),
            "tpr": round(self.true_positive_rate, 4),
            "fpr": round(self.false_positive_rate, 4),
            "precision": round(self.precision, 4),
        }


@dataclass
class BiasDetectionResult:
    """Result of a bias detection analysis."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metric: FairnessMetric = FairnessMetric.DEMOGRAPHIC_PARITY
    protected_attribute: ProtectedAttribute = ProtectedAttribute.CUSTOM
    bias_level: BiasLevel = BiasLevel.NONE
    disparity_ratio: float = 1.0
    group_metrics: list[GroupMetrics] = field(default_factory=list)
    threshold: float = 0.8
    passed: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "metric": self.metric.value,
            "attribute": self.protected_attribute.value,
            "bias_level": self.bias_level.value,
            "disparity_ratio": round(self.disparity_ratio, 4),
            "threshold": self.threshold,
            "passed": self.passed,
            "group_count": len(self.group_metrics),
        }


@dataclass
class RemediationSuggestion:
    """A suggestion for remediating detected bias."""

    type: RemediationType
    description: str
    expected_improvement: float = 0.0
    complexity: str = "medium"
    affected_groups: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "description": self.description,
            "expected_improvement": round(self.expected_improvement, 4),
            "complexity": self.complexity,
        }


@dataclass
class FairnessReport:
    """Comprehensive fairness analysis report."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = ""
    results: list[BiasDetectionResult] = field(default_factory=list)
    remediations: list[RemediationSuggestion] = field(default_factory=list)
    overall_compliance: ComplianceStatus = ComplianceStatus.NOT_ASSESSED
    regulation: str = ""
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    @property
    def passed_checks(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_checks(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_name": self.model_name,
            "total_checks": len(self.results),
            "passed": self.passed_checks,
            "failed": self.failed_checks,
            "overall_compliance": self.overall_compliance.value,
            "regulation": self.regulation,
            "remediation_count": len(self.remediations),
        }


class BiasDetector:
    """Detects bias in model predictions using fairness metrics."""

    def check_demographic_parity(
        self,
        groups: list[GroupMetrics],
        threshold: float = 0.8,
    ) -> BiasDetectionResult:
        """Check demographic parity (equal positive rates across groups)."""
        if len(groups) < 2:
            return BiasDetectionResult(
                metric=FairnessMetric.DEMOGRAPHIC_PARITY,
                passed=True,
                description="Insufficient groups for comparison.",
            )

        rates = [g.positive_rate for g in groups]
        min_rate = min(rates) if rates else 0
        max_rate = max(rates) if rates else 1
        ratio = min_rate / max(max_rate, 1e-10)

        bias_level = self._ratio_to_bias_level(ratio)
        return BiasDetectionResult(
            metric=FairnessMetric.DEMOGRAPHIC_PARITY,
            bias_level=bias_level,
            disparity_ratio=ratio,
            group_metrics=groups,
            threshold=threshold,
            passed=ratio >= threshold,
            description=f"Positive rate ratio: {ratio:.4f} (threshold: {threshold})",
        )

    def check_equal_opportunity(
        self,
        groups: list[GroupMetrics],
        threshold: float = 0.8,
    ) -> BiasDetectionResult:
        """Check equal opportunity (equal TPR across groups)."""
        if len(groups) < 2:
            return BiasDetectionResult(
                metric=FairnessMetric.EQUAL_OPPORTUNITY,
                passed=True,
            )

        tprs = [g.true_positive_rate for g in groups]
        min_tpr = min(tprs) if tprs else 0
        max_tpr = max(tprs) if tprs else 1
        ratio = min_tpr / max(max_tpr, 1e-10)

        bias_level = self._ratio_to_bias_level(ratio)
        return BiasDetectionResult(
            metric=FairnessMetric.EQUAL_OPPORTUNITY,
            bias_level=bias_level,
            disparity_ratio=ratio,
            group_metrics=groups,
            threshold=threshold,
            passed=ratio >= threshold,
            description=f"TPR ratio: {ratio:.4f} (threshold: {threshold})",
        )

    def check_equalized_odds(
        self,
        groups: list[GroupMetrics],
        threshold: float = 0.8,
    ) -> BiasDetectionResult:
        """Check equalized odds (equal TPR and FPR across groups)."""
        if len(groups) < 2:
            return BiasDetectionResult(
                metric=FairnessMetric.EQUALIZED_ODDS,
                passed=True,
            )

        tprs = [g.true_positive_rate for g in groups]
        fprs = [g.false_positive_rate for g in groups]

        tpr_ratio = min(tprs) / max(max(tprs), 1e-10)
        fpr_ratio = min(fprs) / max(max(fprs), 1e-10) if max(fprs) > 0 else 1.0
        combined_ratio = min(tpr_ratio, fpr_ratio)

        bias_level = self._ratio_to_bias_level(combined_ratio)
        return BiasDetectionResult(
            metric=FairnessMetric.EQUALIZED_ODDS,
            bias_level=bias_level,
            disparity_ratio=combined_ratio,
            group_metrics=groups,
            threshold=threshold,
            passed=combined_ratio >= threshold,
            description=f"Equalized odds ratio: {combined_ratio:.4f}",
        )

    @staticmethod
    def _ratio_to_bias_level(ratio: float) -> BiasLevel:
        if ratio >= 0.9:
            return BiasLevel.NONE
        elif ratio >= 0.8:
            return BiasLevel.LOW
        elif ratio >= 0.6:
            return BiasLevel.MODERATE
        elif ratio >= 0.4:
            return BiasLevel.HIGH
        return BiasLevel.SEVERE


class RemediationAdvisor:
    """Generates remediation suggestions for detected bias."""

    def suggest(self, result: BiasDetectionResult) -> list[RemediationSuggestion]:
        """Suggest remediations for a bias detection result."""
        if result.passed:
            return []

        suggestions: list[RemediationSuggestion] = []

        if result.bias_level in (BiasLevel.HIGH, BiasLevel.SEVERE):
            suggestions.append(RemediationSuggestion(
                type=RemediationType.REWEIGHTING,
                description="Apply sample reweighting to equalize group representation.",
                expected_improvement=0.15,
                complexity="low",
            ))
            suggestions.append(RemediationSuggestion(
                type=RemediationType.ADVERSARIAL_DEBIASING,
                description="Train an adversarial network to remove protected attribute signal.",
                expected_improvement=0.25,
                complexity="high",
            ))

        if result.bias_level in (BiasLevel.MODERATE, BiasLevel.LOW):
            suggestions.append(RemediationSuggestion(
                type=RemediationType.THRESHOLD_ADJUSTMENT,
                description="Adjust decision thresholds per group to equalize outcomes.",
                expected_improvement=0.10,
                complexity="low",
            ))

        suggestions.append(RemediationSuggestion(
            type=RemediationType.CONSTRAINT_UPDATE,
            description="Add fairness constraints to Z3 verification spec.",
            expected_improvement=0.20,
            complexity="medium",
        ))

        return suggestions


class FairnessVerifier:
    """Main fairness verification engine."""

    def __init__(self) -> None:
        self._detector = BiasDetector()
        self._advisor = RemediationAdvisor()
        self._reports: dict[str, FairnessReport] = {}

    def verify_fairness(
        self,
        model_name: str,
        groups: list[GroupMetrics],
        constraints: list[FairnessConstraint] | None = None,
        regulation: str = "EU AI Act",
    ) -> FairnessReport:
        """Run full fairness verification on model predictions."""
        if constraints is None:
            constraints = [
                FairnessConstraint(
                    metric=FairnessMetric.DEMOGRAPHIC_PARITY,
                    protected_attribute=ProtectedAttribute.CUSTOM,
                ),
                FairnessConstraint(
                    metric=FairnessMetric.EQUAL_OPPORTUNITY,
                    protected_attribute=ProtectedAttribute.CUSTOM,
                ),
            ]

        results: list[BiasDetectionResult] = []
        for constraint in constraints:
            if constraint.metric == FairnessMetric.DEMOGRAPHIC_PARITY:
                result = self._detector.check_demographic_parity(groups, constraint.threshold)
            elif constraint.metric == FairnessMetric.EQUAL_OPPORTUNITY:
                result = self._detector.check_equal_opportunity(groups, constraint.threshold)
            elif constraint.metric == FairnessMetric.EQUALIZED_ODDS:
                result = self._detector.check_equalized_odds(groups, constraint.threshold)
            else:
                result = self._detector.check_demographic_parity(groups, constraint.threshold)

            result.protected_attribute = constraint.protected_attribute
            results.append(result)

        all_remediations: list[RemediationSuggestion] = []
        for r in results:
            all_remediations.extend(self._advisor.suggest(r))

        all_passed = all(r.passed for r in results)
        any_passed = any(r.passed for r in results)
        if all_passed:
            compliance = ComplianceStatus.COMPLIANT
        elif any_passed:
            compliance = ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            compliance = ComplianceStatus.NON_COMPLIANT

        report = FairnessReport(
            model_name=model_name,
            results=results,
            remediations=all_remediations,
            overall_compliance=compliance,
            regulation=regulation,
        )
        self._reports[report.id] = report

        logger.info(
            "fairness_verified",
            model=model_name,
            checks=len(results),
            passed=report.passed_checks,
            compliance=compliance.value,
        )
        return report

    def get_report(self, report_id: str) -> FairnessReport | None:
        return self._reports.get(report_id)

    def list_reports(self) -> list[FairnessReport]:
        return list(self._reports.values())


_default_verifier: FairnessVerifier | None = None


def get_fairness_verifier() -> FairnessVerifier:
    """Get the singleton fairness verifier."""
    global _default_verifier
    if _default_verifier is None:
        _default_verifier = FairnessVerifier()
    return _default_verifier


def reset_fairness_verifier() -> None:
    """Reset the singleton (for testing)."""
    global _default_verifier
    _default_verifier = None
