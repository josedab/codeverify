"""AI Drift Detector - Monitor AI-generated code degradation over time.

This module detects when AI-generated code patterns in a codebase start
degrading in quality, helping teams identify when Copilot/AI suggestions
are no longer being properly reviewed or are being accepted uncritically.

Features:
- Track AI code quality metrics over time
- Detect declining review patterns
- Identify acceptance rate anomalies
- Alert on quality degradation trends
- Generate remediation recommendations
"""

import hashlib
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import structlog

from .base import AgentConfig, AgentResult, BaseAgent

logger = structlog.get_logger()


# =============================================================================
# Enums and Data Classes
# =============================================================================


class DriftSeverity(str, Enum):
    """Severity levels for detected drift."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftCategory(str, Enum):
    """Categories of AI code drift."""

    QUALITY_DEGRADATION = "quality_degradation"
    ACCEPTANCE_RATE_SPIKE = "acceptance_rate_spike"
    REVIEW_DEPTH_DECLINE = "review_depth_decline"
    SECURITY_RISK_INCREASE = "security_risk_increase"
    TEST_COVERAGE_DROP = "test_coverage_drop"
    DOCUMENTATION_DECLINE = "documentation_decline"
    COMPLEXITY_INCREASE = "complexity_increase"
    PATTERN_DEVIATION = "pattern_deviation"


class AlertType(str, Enum):
    """Types of drift alerts."""

    TREND = "trend"  # Gradual degradation over time
    ANOMALY = "anomaly"  # Sudden spike or drop
    THRESHOLD = "threshold"  # Crossed predefined threshold
    PATTERN = "pattern"  # Repeated problematic pattern


@dataclass
class AICodeSnapshot:
    """Snapshot of AI-generated code at a point in time."""

    snapshot_id: str
    timestamp: datetime
    file_path: str
    code_hash: str

    # AI detection metrics
    ai_probability: float
    detected_model: str | None = None

    # Quality metrics
    trust_score: float = 0.0
    complexity_score: float = 0.0
    security_score: float = 0.0
    test_coverage: float = 0.0
    documentation_score: float = 0.0

    # Review metrics
    was_reviewed: bool = False
    review_depth: float = 0.0  # 0-1 scale
    time_to_accept: float = 0.0  # seconds

    # Findings at time of snapshot
    findings_count: int = 0
    critical_findings: int = 0
    high_findings: int = 0

    # Metadata
    author: str | None = None
    commit_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp.isoformat(),
            "file_path": self.file_path,
            "code_hash": self.code_hash,
            "ai_probability": self.ai_probability,
            "detected_model": self.detected_model,
            "trust_score": self.trust_score,
            "complexity_score": self.complexity_score,
            "security_score": self.security_score,
            "test_coverage": self.test_coverage,
            "documentation_score": self.documentation_score,
            "was_reviewed": self.was_reviewed,
            "review_depth": self.review_depth,
            "time_to_accept": self.time_to_accept,
            "findings_count": self.findings_count,
            "critical_findings": self.critical_findings,
            "high_findings": self.high_findings,
            "author": self.author,
            "commit_hash": self.commit_hash,
        }


@dataclass
class DriftAlert:
    """Alert for detected AI code drift."""

    alert_id: str
    category: DriftCategory
    alert_type: AlertType
    severity: DriftSeverity
    message: str
    details: str

    # Metrics that triggered the alert
    metric_name: str
    current_value: float
    baseline_value: float
    threshold: float | None = None

    # Time range
    detected_at: datetime = field(default_factory=datetime.now)
    period_start: datetime | None = None
    period_end: datetime | None = None

    # Affected files/areas
    affected_files: list[str] = field(default_factory=list)
    affected_authors: list[str] = field(default_factory=list)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "category": self.category.value,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "baseline_value": self.baseline_value,
            "threshold": self.threshold,
            "detected_at": self.detected_at.isoformat(),
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "affected_files": self.affected_files,
            "affected_authors": self.affected_authors,
            "recommendations": self.recommendations,
        }


@dataclass
class DriftMetrics:
    """Aggregated drift metrics for a time period."""

    period_start: datetime
    period_end: datetime

    # Aggregate metrics
    total_ai_snippets: int = 0
    avg_trust_score: float = 0.0
    avg_ai_probability: float = 0.0
    avg_complexity: float = 0.0
    avg_security_score: float = 0.0
    avg_test_coverage: float = 0.0
    avg_documentation: float = 0.0

    # Review metrics
    review_rate: float = 0.0  # % of AI code reviewed
    avg_review_depth: float = 0.0
    avg_time_to_accept: float = 0.0

    # Finding metrics
    total_findings: int = 0
    critical_finding_rate: float = 0.0
    high_finding_rate: float = 0.0

    # Trend indicators (-1 to 1, negative = declining)
    trust_trend: float = 0.0
    security_trend: float = 0.0
    quality_trend: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_ai_snippets": self.total_ai_snippets,
            "avg_trust_score": self.avg_trust_score,
            "avg_ai_probability": self.avg_ai_probability,
            "avg_complexity": self.avg_complexity,
            "avg_security_score": self.avg_security_score,
            "avg_test_coverage": self.avg_test_coverage,
            "avg_documentation": self.avg_documentation,
            "review_rate": self.review_rate,
            "avg_review_depth": self.avg_review_depth,
            "avg_time_to_accept": self.avg_time_to_accept,
            "total_findings": self.total_findings,
            "critical_finding_rate": self.critical_finding_rate,
            "high_finding_rate": self.high_finding_rate,
            "trust_trend": self.trust_trend,
            "security_trend": self.security_trend,
            "quality_trend": self.quality_trend,
        }


@dataclass
class DriftReport:
    """Comprehensive drift analysis report."""

    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime

    # Overall health
    health_score: float  # 0-100
    health_trend: str  # "improving", "stable", "declining"

    # Active alerts
    alerts: list[DriftAlert]

    # Metrics comparison
    current_metrics: DriftMetrics
    baseline_metrics: DriftMetrics | None = None

    # Per-author analysis
    author_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    # Per-file hotspots
    hotspots: list[dict[str, Any]] = field(default_factory=list)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "health_score": self.health_score,
            "health_trend": self.health_trend,
            "alerts": [a.to_dict() for a in self.alerts],
            "current_metrics": self.current_metrics.to_dict(),
            "baseline_metrics": self.baseline_metrics.to_dict() if self.baseline_metrics else None,
            "author_metrics": self.author_metrics,
            "hotspots": self.hotspots,
            "recommendations": self.recommendations,
        }


# =============================================================================
# AI Drift Detector Agent
# =============================================================================


class AIDriftDetector(BaseAgent):
    """Agent for detecting AI code quality drift over time.

    Monitors AI-generated code patterns and detects when quality,
    security, or review practices are degrading.

    Example usage:
        detector = AIDriftDetector()
        detector.record_snapshot(snapshot)  # Record AI code
        report = detector.generate_report()  # Get drift analysis
    """

    # Thresholds for alert generation
    DEFAULT_THRESHOLDS = {
        "trust_score_min": 60.0,
        "trust_score_decline": 10.0,  # % decline triggers alert
        "security_score_min": 70.0,
        "review_rate_min": 80.0,  # % of AI code reviewed
        "acceptance_spike": 50.0,  # % increase in acceptance rate
        "critical_finding_max": 0.1,  # Max 10% critical findings
        "complexity_increase": 20.0,  # % increase triggers alert
        "test_coverage_min": 60.0,
        "time_to_accept_spike": 100.0,  # % decrease in review time
    }

    def __init__(
        self,
        config: AgentConfig | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        """Initialize the drift detector."""
        super().__init__(config)
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}

        # Storage for snapshots (in production, use a database)
        self._snapshots: list[AICodeSnapshot] = []
        self._alerts: list[DriftAlert] = []

        # Baseline metrics (established from initial period)
        self._baseline_metrics: DriftMetrics | None = None
        self._baseline_established = False

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """Analyze code for AI drift indicators."""
        start_time = time.time()

        try:
            # Generate report for analysis period
            days = context.get("days", 30)
            report = self.generate_report(days)

            return AgentResult(
                success=True,
                data=report.to_dict(),
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error("Drift analysis failed", error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def record_snapshot(self, snapshot: AICodeSnapshot) -> None:
        """Record an AI code snapshot for drift tracking."""
        self._snapshots.append(snapshot)

        # Check for immediate alerts
        alerts = self._check_snapshot_alerts(snapshot)
        self._alerts.extend(alerts)

        logger.debug(
            "Recorded AI code snapshot",
            file=snapshot.file_path,
            ai_probability=snapshot.ai_probability,
            trust_score=snapshot.trust_score,
        )

    def record_from_analysis(
        self,
        file_path: str,
        code: str,
        trust_score: float,
        ai_probability: float,
        findings: list[dict[str, Any]],
        **metadata: Any,
    ) -> AICodeSnapshot:
        """Record a snapshot from analysis results."""
        snapshot = AICodeSnapshot(
            snapshot_id=self._generate_id(),
            timestamp=datetime.now(),
            file_path=file_path,
            code_hash=hashlib.sha256(code.encode()).hexdigest()[:16],
            ai_probability=ai_probability,
            trust_score=trust_score,
            findings_count=len(findings),
            critical_findings=sum(1 for f in findings if f.get("severity") == "critical"),
            high_findings=sum(1 for f in findings if f.get("severity") == "high"),
            **metadata,
        )

        self.record_snapshot(snapshot)
        return snapshot

    def establish_baseline(self, days: int = 30) -> DriftMetrics:
        """Establish baseline metrics from historical data."""
        cutoff = datetime.now() - timedelta(days=days)
        baseline_snapshots = [s for s in self._snapshots if s.timestamp >= cutoff]

        if not baseline_snapshots:
            logger.warning("No snapshots available for baseline")
            return DriftMetrics(
                period_start=cutoff,
                period_end=datetime.now(),
            )

        self._baseline_metrics = self._calculate_metrics(
            baseline_snapshots,
            cutoff,
            datetime.now(),
        )
        self._baseline_established = True

        logger.info(
            "Baseline established",
            snapshots=len(baseline_snapshots),
            avg_trust_score=self._baseline_metrics.avg_trust_score,
        )

        return self._baseline_metrics

    def generate_report(self, days: int = 30) -> DriftReport:
        """Generate a comprehensive drift analysis report."""
        now = datetime.now()
        period_start = now - timedelta(days=days)

        # Get snapshots for the period
        period_snapshots = [
            s for s in self._snapshots
            if s.timestamp >= period_start
        ]

        # Calculate current metrics
        current_metrics = self._calculate_metrics(period_snapshots, period_start, now)

        # Detect drift and generate alerts
        alerts = self._detect_drift(current_metrics)

        # Calculate health score
        health_score = self._calculate_health_score(current_metrics, alerts)

        # Determine trend
        health_trend = self._determine_health_trend(current_metrics)

        # Get hotspots
        hotspots = self._identify_hotspots(period_snapshots)

        # Generate recommendations
        recommendations = self._generate_recommendations(alerts, current_metrics)

        # Per-author analysis
        author_metrics = self._analyze_by_author(period_snapshots)

        report = DriftReport(
            report_id=self._generate_id(),
            generated_at=now,
            period_start=period_start,
            period_end=now,
            health_score=health_score,
            health_trend=health_trend,
            alerts=alerts,
            current_metrics=current_metrics,
            baseline_metrics=self._baseline_metrics,
            author_metrics=author_metrics,
            hotspots=hotspots,
            recommendations=recommendations,
        )

        logger.info(
            "Generated drift report",
            health_score=health_score,
            alerts=len(alerts),
            recommendations=len(recommendations),
        )

        return report

    def get_active_alerts(
        self,
        severity: DriftSeverity | None = None,
        category: DriftCategory | None = None,
    ) -> list[DriftAlert]:
        """Get active drift alerts."""
        alerts = self._alerts

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if category:
            alerts = [a for a in alerts if a.category == category]

        return alerts

    def _check_snapshot_alerts(self, snapshot: AICodeSnapshot) -> list[DriftAlert]:
        """Check a single snapshot for immediate alerts."""
        alerts = []

        # Low trust score alert
        if snapshot.trust_score < self.thresholds["trust_score_min"]:
            alerts.append(DriftAlert(
                alert_id=self._generate_id(),
                category=DriftCategory.QUALITY_DEGRADATION,
                alert_type=AlertType.THRESHOLD,
                severity=DriftSeverity.HIGH if snapshot.trust_score < 40 else DriftSeverity.MEDIUM,
                message=f"Low trust score detected: {snapshot.trust_score:.1f}",
                details=f"File {snapshot.file_path} has trust score below threshold",
                metric_name="trust_score",
                current_value=snapshot.trust_score,
                baseline_value=self.thresholds["trust_score_min"],
                threshold=self.thresholds["trust_score_min"],
                affected_files=[snapshot.file_path],
                recommendations=[
                    "Review the AI-generated code carefully",
                    "Add tests for the new functionality",
                    "Consider refactoring complex sections",
                ],
            ))

        # Critical findings alert
        if snapshot.critical_findings > 0:
            alerts.append(DriftAlert(
                alert_id=self._generate_id(),
                category=DriftCategory.SECURITY_RISK_INCREASE,
                alert_type=AlertType.THRESHOLD,
                severity=DriftSeverity.CRITICAL,
                message=f"Critical security findings in AI code: {snapshot.critical_findings}",
                details=f"File {snapshot.file_path} contains critical security issues",
                metric_name="critical_findings",
                current_value=float(snapshot.critical_findings),
                baseline_value=0.0,
                threshold=0.0,
                affected_files=[snapshot.file_path],
                recommendations=[
                    "Address critical security issues immediately",
                    "Review AI suggestion acceptance practices",
                    "Enable stricter pre-commit checks",
                ],
            ))

        # Quick acceptance alert (no review)
        if snapshot.ai_probability > 0.7 and not snapshot.was_reviewed:
            alerts.append(DriftAlert(
                alert_id=self._generate_id(),
                category=DriftCategory.REVIEW_DEPTH_DECLINE,
                alert_type=AlertType.ANOMALY,
                severity=DriftSeverity.MEDIUM,
                message="AI code accepted without review",
                details=f"High-probability AI code in {snapshot.file_path} was not reviewed",
                metric_name="review_depth",
                current_value=0.0,
                baseline_value=1.0,
                affected_files=[snapshot.file_path],
                affected_authors=[snapshot.author] if snapshot.author else [],
                recommendations=[
                    "Establish code review requirements for AI-generated code",
                    "Use tools like CodeVerify to flag unreviewed AI code",
                    "Add review checklists for AI suggestions",
                ],
            ))

        return alerts

    def _calculate_metrics(
        self,
        snapshots: list[AICodeSnapshot],
        start: datetime,
        end: datetime,
    ) -> DriftMetrics:
        """Calculate aggregated metrics from snapshots."""
        if not snapshots:
            return DriftMetrics(period_start=start, period_end=end)

        metrics = DriftMetrics(
            period_start=start,
            period_end=end,
            total_ai_snippets=len(snapshots),
        )

        # Calculate averages
        metrics.avg_trust_score = statistics.mean(s.trust_score for s in snapshots)
        metrics.avg_ai_probability = statistics.mean(s.ai_probability for s in snapshots)
        metrics.avg_complexity = statistics.mean(s.complexity_score for s in snapshots)
        metrics.avg_security_score = statistics.mean(s.security_score for s in snapshots)
        metrics.avg_test_coverage = statistics.mean(s.test_coverage for s in snapshots)
        metrics.avg_documentation = statistics.mean(s.documentation_score for s in snapshots)

        # Review metrics
        reviewed = [s for s in snapshots if s.was_reviewed]
        metrics.review_rate = len(reviewed) / len(snapshots) * 100 if snapshots else 0

        if reviewed:
            metrics.avg_review_depth = statistics.mean(s.review_depth for s in reviewed)
            metrics.avg_time_to_accept = statistics.mean(s.time_to_accept for s in reviewed)

        # Finding metrics
        metrics.total_findings = sum(s.findings_count for s in snapshots)
        if snapshots:
            metrics.critical_finding_rate = sum(s.critical_findings for s in snapshots) / len(snapshots)
            metrics.high_finding_rate = sum(s.high_findings for s in snapshots) / len(snapshots)

        # Calculate trends using linear regression on time-ordered data
        metrics.trust_trend = self._calculate_trend([s.trust_score for s in sorted(snapshots, key=lambda x: x.timestamp)])
        metrics.security_trend = self._calculate_trend([s.security_score for s in sorted(snapshots, key=lambda x: x.timestamp)])

        # Quality trend is combination of trust and documentation
        quality_values = [(s.trust_score + s.documentation_score) / 2 for s in sorted(snapshots, key=lambda x: x.timestamp)]
        metrics.quality_trend = self._calculate_trend(quality_values)

        return metrics

    def _calculate_trend(self, values: list[float]) -> float:
        """Calculate trend (-1 to 1) from a list of values over time."""
        if len(values) < 2:
            return 0.0

        # Simple linear regression
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator

        # Normalize to -1 to 1 range (assuming typical value ranges)
        max_expected_change = max(values) - min(values) if max(values) != min(values) else 1
        normalized = slope * n / max_expected_change

        return max(-1.0, min(1.0, normalized))

    def _detect_drift(self, current: DriftMetrics) -> list[DriftAlert]:
        """Detect drift by comparing current metrics to baseline."""
        alerts = []

        if not self._baseline_metrics:
            return alerts

        baseline = self._baseline_metrics

        # Trust score decline
        if baseline.avg_trust_score > 0:
            trust_decline = (baseline.avg_trust_score - current.avg_trust_score) / baseline.avg_trust_score * 100
            if trust_decline > self.thresholds["trust_score_decline"]:
                alerts.append(DriftAlert(
                    alert_id=self._generate_id(),
                    category=DriftCategory.QUALITY_DEGRADATION,
                    alert_type=AlertType.TREND,
                    severity=DriftSeverity.HIGH if trust_decline > 20 else DriftSeverity.MEDIUM,
                    message=f"Trust score declined by {trust_decline:.1f}%",
                    details="AI code quality is degrading compared to baseline period",
                    metric_name="avg_trust_score",
                    current_value=current.avg_trust_score,
                    baseline_value=baseline.avg_trust_score,
                    threshold=self.thresholds["trust_score_decline"],
                    period_start=current.period_start,
                    period_end=current.period_end,
                    recommendations=[
                        "Increase code review rigor for AI suggestions",
                        "Consider additional verification tools",
                        "Review team training on AI code assessment",
                    ],
                ))

        # Review rate decline
        if baseline.review_rate > 0:
            review_decline = (baseline.review_rate - current.review_rate)
            if current.review_rate < self.thresholds["review_rate_min"]:
                alerts.append(DriftAlert(
                    alert_id=self._generate_id(),
                    category=DriftCategory.REVIEW_DEPTH_DECLINE,
                    alert_type=AlertType.THRESHOLD,
                    severity=DriftSeverity.HIGH,
                    message=f"Review rate below threshold: {current.review_rate:.1f}%",
                    details=f"Only {current.review_rate:.1f}% of AI code is being reviewed (baseline: {baseline.review_rate:.1f}%)",
                    metric_name="review_rate",
                    current_value=current.review_rate,
                    baseline_value=baseline.review_rate,
                    threshold=self.thresholds["review_rate_min"],
                    recommendations=[
                        "Mandate code reviews for all AI-generated code",
                        "Set up automated review reminders",
                        "Track and reward thorough reviews",
                    ],
                ))

        # Security score decline
        if baseline.avg_security_score > 0:
            security_decline = (baseline.avg_security_score - current.avg_security_score)
            if current.avg_security_score < self.thresholds["security_score_min"]:
                alerts.append(DriftAlert(
                    alert_id=self._generate_id(),
                    category=DriftCategory.SECURITY_RISK_INCREASE,
                    alert_type=AlertType.THRESHOLD,
                    severity=DriftSeverity.HIGH,
                    message=f"Security score below threshold: {current.avg_security_score:.1f}",
                    details="AI-generated code security is declining",
                    metric_name="avg_security_score",
                    current_value=current.avg_security_score,
                    baseline_value=baseline.avg_security_score,
                    threshold=self.thresholds["security_score_min"],
                    recommendations=[
                        "Enable security-focused AI code analysis",
                        "Add security review checkpoints",
                        "Provide security training for the team",
                    ],
                ))

        # Critical findings increase
        if current.critical_finding_rate > self.thresholds["critical_finding_max"]:
            alerts.append(DriftAlert(
                alert_id=self._generate_id(),
                category=DriftCategory.SECURITY_RISK_INCREASE,
                alert_type=AlertType.THRESHOLD,
                severity=DriftSeverity.CRITICAL,
                message=f"Critical finding rate too high: {current.critical_finding_rate:.2f}",
                details="Too many critical security issues in AI-generated code",
                metric_name="critical_finding_rate",
                current_value=current.critical_finding_rate,
                baseline_value=baseline.critical_finding_rate,
                threshold=self.thresholds["critical_finding_max"],
                recommendations=[
                    "Immediately review all critical findings",
                    "Block AI code acceptance until issues resolved",
                    "Investigate root cause of security issues",
                ],
            ))

        # Test coverage decline
        if baseline.avg_test_coverage > 0 and current.avg_test_coverage < self.thresholds["test_coverage_min"]:
            alerts.append(DriftAlert(
                alert_id=self._generate_id(),
                category=DriftCategory.TEST_COVERAGE_DROP,
                alert_type=AlertType.THRESHOLD,
                severity=DriftSeverity.MEDIUM,
                message=f"Test coverage below threshold: {current.avg_test_coverage:.1f}%",
                details="AI-generated code lacks adequate test coverage",
                metric_name="avg_test_coverage",
                current_value=current.avg_test_coverage,
                baseline_value=baseline.avg_test_coverage,
                threshold=self.thresholds["test_coverage_min"],
                recommendations=[
                    "Require tests for AI-generated code",
                    "Use AI to generate test cases",
                    "Add test coverage checks to CI",
                ],
            ))

        # Complexity increase
        if baseline.avg_complexity > 0:
            complexity_increase = (current.avg_complexity - baseline.avg_complexity) / baseline.avg_complexity * 100
            if complexity_increase > self.thresholds["complexity_increase"]:
                alerts.append(DriftAlert(
                    alert_id=self._generate_id(),
                    category=DriftCategory.COMPLEXITY_INCREASE,
                    alert_type=AlertType.TREND,
                    severity=DriftSeverity.MEDIUM,
                    message=f"Code complexity increased by {complexity_increase:.1f}%",
                    details="AI-generated code is becoming more complex",
                    metric_name="avg_complexity",
                    current_value=current.avg_complexity,
                    baseline_value=baseline.avg_complexity,
                    threshold=self.thresholds["complexity_increase"],
                    recommendations=[
                        "Refactor complex AI-generated code",
                        "Set complexity limits for AI suggestions",
                        "Prefer simpler solutions in code reviews",
                    ],
                ))

        return alerts

    def _calculate_health_score(self, metrics: DriftMetrics, alerts: list[DriftAlert]) -> float:
        """Calculate overall health score (0-100)."""
        score = 100.0

        # Deduct for low trust score
        if metrics.avg_trust_score < 80:
            score -= (80 - metrics.avg_trust_score) * 0.5

        # Deduct for low review rate
        if metrics.review_rate < 80:
            score -= (80 - metrics.review_rate) * 0.3

        # Deduct for alerts
        severity_penalties = {
            DriftSeverity.LOW: 2,
            DriftSeverity.MEDIUM: 5,
            DriftSeverity.HIGH: 10,
            DriftSeverity.CRITICAL: 20,
        }

        for alert in alerts:
            score -= severity_penalties.get(alert.severity, 0)

        # Deduct for negative trends
        if metrics.trust_trend < 0:
            score -= abs(metrics.trust_trend) * 10

        if metrics.security_trend < 0:
            score -= abs(metrics.security_trend) * 15

        return max(0.0, min(100.0, score))

    def _determine_health_trend(self, metrics: DriftMetrics) -> str:
        """Determine overall health trend."""
        combined_trend = (metrics.trust_trend + metrics.security_trend + metrics.quality_trend) / 3

        if combined_trend > 0.1:
            return "improving"
        elif combined_trend < -0.1:
            return "declining"
        else:
            return "stable"

    def _identify_hotspots(self, snapshots: list[AICodeSnapshot]) -> list[dict[str, Any]]:
        """Identify files/areas with most drift issues."""
        file_issues: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "issues": 0,
            "critical": 0,
            "avg_trust": [],
            "low_reviews": 0,
        })

        for snapshot in snapshots:
            file_issues[snapshot.file_path]["issues"] += snapshot.findings_count
            file_issues[snapshot.file_path]["critical"] += snapshot.critical_findings
            file_issues[snapshot.file_path]["avg_trust"].append(snapshot.trust_score)
            if not snapshot.was_reviewed:
                file_issues[snapshot.file_path]["low_reviews"] += 1

        hotspots = []
        for file_path, data in file_issues.items():
            avg_trust = statistics.mean(data["avg_trust"]) if data["avg_trust"] else 0
            hotspots.append({
                "file_path": file_path,
                "total_issues": data["issues"],
                "critical_issues": data["critical"],
                "avg_trust_score": avg_trust,
                "unreviewed_count": data["low_reviews"],
                "risk_score": data["critical"] * 3 + data["issues"] + data["low_reviews"] * 2,
            })

        # Sort by risk score descending
        hotspots.sort(key=lambda x: x["risk_score"], reverse=True)

        return hotspots[:10]  # Top 10 hotspots

    def _analyze_by_author(self, snapshots: list[AICodeSnapshot]) -> dict[str, dict[str, float]]:
        """Analyze metrics by author."""
        author_data: dict[str, list[AICodeSnapshot]] = defaultdict(list)

        for snapshot in snapshots:
            if snapshot.author:
                author_data[snapshot.author].append(snapshot)

        result = {}
        for author, author_snapshots in author_data.items():
            result[author] = {
                "total_ai_code": len(author_snapshots),
                "avg_trust_score": statistics.mean(s.trust_score for s in author_snapshots),
                "review_rate": sum(1 for s in author_snapshots if s.was_reviewed) / len(author_snapshots) * 100,
                "critical_findings": sum(s.critical_findings for s in author_snapshots),
            }

        return result

    def _generate_recommendations(
        self,
        alerts: list[DriftAlert],
        metrics: DriftMetrics,
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []
        seen = set()

        # Collect recommendations from alerts (deduplicated)
        for alert in alerts:
            for rec in alert.recommendations:
                if rec not in seen:
                    recommendations.append(rec)
                    seen.add(rec)

        # Add metric-based recommendations
        if metrics.review_rate < 50:
            rec = "Establish mandatory code review process for all AI-generated code"
            if rec not in seen:
                recommendations.append(rec)
                seen.add(rec)

        if metrics.avg_test_coverage < 30:
            rec = "Implement automated test generation for AI code"
            if rec not in seen:
                recommendations.append(rec)
                seen.add(rec)

        if metrics.trust_trend < -0.3:
            rec = "Schedule team training on AI code review best practices"
            if rec not in seen:
                recommendations.append(rec)
                seen.add(rec)

        return recommendations[:10]  # Limit to top 10

    def _generate_id(self) -> str:
        """Generate a unique ID."""
        return hashlib.sha256(
            f"{time.time()}{len(self._snapshots)}".encode()
        ).hexdigest()[:12]

    def get_statistics(self) -> dict[str, Any]:
        """Get detector statistics."""
        return {
            "total_snapshots": len(self._snapshots),
            "active_alerts": len(self._alerts),
            "baseline_established": self._baseline_established,
            "thresholds": self.thresholds,
        }

    def clear_data(self) -> None:
        """Clear all stored data."""
        self._snapshots.clear()
        self._alerts.clear()
        self._baseline_metrics = None
        self._baseline_established = False
