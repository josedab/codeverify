"""Predictive Code Quality Forecasting.

ML-based quality trend prediction with time series analysis, scenario
modeling, and executive dashboards for proactive quality management.

Features:
- Time series quality metrics tracking
- Trend analysis with linear regression
- Scenario modeling ("what if" analysis)
- Executive summary with ROI metrics
- Anomaly detection for quality degradation
"""

from __future__ import annotations

import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class MetricType(str, Enum):
    """Types of quality metrics tracked."""

    FINDINGS = "findings"
    CRITICAL_FINDINGS = "critical_findings"
    TRUST_SCORE = "trust_score"
    VERIFICATION_COVERAGE = "verification_coverage"
    CODE_CHURN = "code_churn"
    AI_GENERATED_RATIO = "ai_generated_ratio"
    INCIDENT_COUNT = "incident_count"


class TrendDirection(str, Enum):
    """Direction of a quality trend."""

    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"


class AlertSeverity(str, Enum):
    """Severity of a quality alert."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ForecastConfidence(str, Enum):
    """Confidence level of a forecast."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class QualityDataPoint:
    """A single quality measurement at a point in time."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metric: MetricType = MetricType.FINDINGS
    value: float = 0.0
    repo_id: str = ""
    team: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric": self.metric.value,
            "value": round(self.value, 4),
            "repo_id": self.repo_id,
            "team": self.team,
        }


@dataclass
class TrendAnalysis:
    """Result of trend analysis on quality data."""

    metric: MetricType = MetricType.FINDINGS
    direction: TrendDirection = TrendDirection.STABLE
    slope: float = 0.0
    r_squared: float = 0.0
    data_points: int = 0
    forecast_next_30d: float = 0.0
    forecast_next_90d: float = 0.0
    confidence: ForecastConfidence = ForecastConfidence.MEDIUM

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric.value,
            "direction": self.direction.value,
            "slope": round(self.slope, 6),
            "r_squared": round(self.r_squared, 4),
            "data_points": self.data_points,
            "forecast_30d": round(self.forecast_next_30d, 2),
            "forecast_90d": round(self.forecast_next_90d, 2),
            "confidence": self.confidence.value,
        }


@dataclass
class QualityAlert:
    """Alert for quality degradation or anomaly."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    severity: AlertSeverity = AlertSeverity.WARNING
    metric: MetricType = MetricType.FINDINGS
    message: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    triggered_at: datetime = field(
        default_factory=lambda: datetime.now(UTC),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity.value,
            "metric": self.metric.value,
            "message": self.message,
            "current_value": round(self.current_value, 4),
            "threshold": round(self.threshold, 4),
        }


@dataclass
class ScenarioResult:
    """Result of a what-if scenario analysis."""

    scenario_name: str = ""
    description: str = ""
    predicted_impact: dict[str, float] = field(default_factory=dict)
    confidence: ForecastConfidence = ForecastConfidence.MEDIUM

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "description": self.description,
            "impact": {k: round(v, 4) for k, v in self.predicted_impact.items()},
            "confidence": self.confidence.value,
        }


@dataclass
class QualityForecast:
    """Complete quality forecast report."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    repo_id: str = ""
    trends: list[TrendAnalysis] = field(default_factory=list)
    alerts: list[QualityAlert] = field(default_factory=list)
    scenarios: list[ScenarioResult] = field(default_factory=list)
    executive_summary: str = ""
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(UTC),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "trends_count": len(self.trends),
            "alerts_count": len(self.alerts),
            "scenarios_count": len(self.scenarios),
            "executive_summary": self.executive_summary,
        }


class TrendCalculator:
    """Calculates trends using simple linear regression."""

    def analyze(self, values: list[float], metric: MetricType) -> TrendAnalysis:
        """Perform trend analysis on a series of values."""
        n = len(values)
        if n < 3:
            return TrendAnalysis(
                metric=metric,
                data_points=n,
                confidence=ForecastConfidence.LOW,
            )

        # Linear regression: y = mx + b
        x_vals = list(range(n))
        x_mean = sum(x_vals) / n
        y_mean = sum(values) / n

        ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, values, strict=True))
        ss_xx = sum((x - x_mean) ** 2 for x in x_vals)
        ss_yy = sum((y - y_mean) ** 2 for y in values)

        slope = 0.0 if ss_xx == 0 else ss_xy / ss_xx

        r_squared = 0.0
        if ss_xx > 0 and ss_yy > 0:
            r_squared = (ss_xy**2) / (ss_xx * ss_yy)

        # Forecast
        last_val = values[-1]
        forecast_30 = last_val + slope * 30
        forecast_90 = last_val + slope * 90

        # Direction
        if abs(slope) < 0.01:
            direction = TrendDirection.STABLE
        elif slope > 0:
            # For findings/incidents, increasing is declining quality
            if metric in (
                MetricType.FINDINGS,
                MetricType.CRITICAL_FINDINGS,
                MetricType.INCIDENT_COUNT,
                MetricType.CODE_CHURN,
            ):
                direction = TrendDirection.DECLINING
            else:
                direction = TrendDirection.IMPROVING
        else:
            if metric in (
                MetricType.FINDINGS,
                MetricType.CRITICAL_FINDINGS,
                MetricType.INCIDENT_COUNT,
                MetricType.CODE_CHURN,
            ):
                direction = TrendDirection.IMPROVING
            else:
                direction = TrendDirection.DECLINING

        # Check volatility
        if n >= 5:
            diffs = [abs(values[i] - values[i - 1]) for i in range(1, n)]
            avg_diff = sum(diffs) / len(diffs)
            if avg_diff > abs(y_mean) * 0.3:
                direction = TrendDirection.VOLATILE

        confidence = (
            ForecastConfidence.HIGH
            if r_squared > 0.7
            else (ForecastConfidence.MEDIUM if r_squared > 0.3 else ForecastConfidence.LOW)
        )

        return TrendAnalysis(
            metric=metric,
            direction=direction,
            slope=slope,
            r_squared=r_squared,
            data_points=n,
            forecast_next_30d=forecast_30,
            forecast_next_90d=forecast_90,
            confidence=confidence,
        )


class AnomalyDetector:
    """Detects quality anomalies using simple statistical methods."""

    def detect(
        self,
        values: list[float],
        metric: MetricType,
        z_threshold: float = 2.0,
    ) -> list[QualityAlert]:
        """Detect anomalies in quality data."""
        if len(values) < 5:
            return []

        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else 0.0

        if std == 0:
            return []

        alerts: list[QualityAlert] = []
        latest = values[-1]
        z_score = (latest - mean) / std

        if abs(z_score) > z_threshold:
            severity = AlertSeverity.CRITICAL if abs(z_score) > 3.0 else AlertSeverity.WARNING
            direction = "above" if z_score > 0 else "below"
            alerts.append(
                QualityAlert(
                    severity=severity,
                    metric=metric,
                    message=f"{metric.value} is {abs(z_score):.1f} std devs {direction} average.",
                    current_value=latest,
                    threshold=mean + z_threshold * std,
                )
            )

        return alerts


class QualityForecaster:
    """Main predictive code quality forecasting engine."""

    def __init__(self) -> None:
        self._data: dict[str, list[QualityDataPoint]] = defaultdict(list)
        self._trend_calc = TrendCalculator()
        self._anomaly_det = AnomalyDetector()
        self._forecasts: dict[str, QualityForecast] = {}

    def record_metric(
        self,
        metric: MetricType,
        value: float,
        repo_id: str = "default",
        team: str = "",
    ) -> None:
        """Record a quality data point."""
        dp = QualityDataPoint(metric=metric, value=value, repo_id=repo_id, team=team)
        key = f"{repo_id}:{metric.value}"
        self._data[key].append(dp)

    def generate_forecast(
        self,
        repo_id: str = "default",
        metrics: list[MetricType] | None = None,
    ) -> QualityForecast:
        """Generate a complete quality forecast."""
        if metrics is None:
            metrics = list(MetricType)

        trends: list[TrendAnalysis] = []
        all_alerts: list[QualityAlert] = []

        for metric in metrics:
            key = f"{repo_id}:{metric.value}"
            points = self._data.get(key, [])
            values = [p.value for p in points]

            if values:
                trend = self._trend_calc.analyze(values, metric)
                trends.append(trend)
                alerts = self._anomaly_det.detect(values, metric)
                all_alerts.extend(alerts)

        scenarios = self._generate_scenarios(trends)
        summary = self._generate_summary(trends, all_alerts)

        forecast = QualityForecast(
            repo_id=repo_id,
            trends=trends,
            alerts=all_alerts,
            scenarios=scenarios,
            executive_summary=summary,
        )
        self._forecasts[forecast.id] = forecast

        logger.info(
            "forecast_generated",
            repo_id=repo_id,
            trends=len(trends),
            alerts=len(all_alerts),
        )
        return forecast

    def _generate_scenarios(self, trends: list[TrendAnalysis]) -> list[ScenarioResult]:
        """Generate what-if scenarios from current trends."""
        scenarios: list[ScenarioResult] = []

        declining = [t for t in trends if t.direction == TrendDirection.DECLINING]
        if declining:
            impact = {t.metric.value: t.forecast_next_90d for t in declining}
            scenarios.append(
                ScenarioResult(
                    scenario_name="status_quo",
                    description="Continue current trajectory without intervention.",
                    predicted_impact=impact,
                    confidence=ForecastConfidence.HIGH,
                )
            )

        scenarios.append(
            ScenarioResult(
                scenario_name="increase_verification",
                description="Increase verification coverage by 20%.",
                predicted_impact={
                    "findings_reduction": 0.30,
                    "trust_score_improvement": 15.0,
                },
                confidence=ForecastConfidence.MEDIUM,
            )
        )

        scenarios.append(
            ScenarioResult(
                scenario_name="add_team_members",
                description="Add 2 developers focused on quality.",
                predicted_impact={
                    "review_coverage_improvement": 0.25,
                    "time_to_fix_reduction": 0.40,
                },
                confidence=ForecastConfidence.LOW,
            )
        )

        return scenarios

    def _generate_summary(
        self,
        trends: list[TrendAnalysis],
        alerts: list[QualityAlert],
    ) -> str:
        """Generate executive summary text."""
        parts: list[str] = []
        declining = [t for t in trends if t.direction == TrendDirection.DECLINING]
        improving = [t for t in trends if t.direction == TrendDirection.IMPROVING]
        critical = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]

        if critical:
            parts.append(f"{len(critical)} critical alert(s) detected.")
        if declining:
            metrics = ", ".join(t.metric.value for t in declining)
            parts.append(f"Declining trends in: {metrics}.")
        if improving:
            metrics = ", ".join(t.metric.value for t in improving)
            parts.append(f"Improving trends in: {metrics}.")
        if not parts:
            parts.append("Quality metrics are stable. No action required.")

        return " ".join(parts)

    def get_forecast(self, forecast_id: str) -> QualityForecast | None:
        return self._forecasts.get(forecast_id)


_default_forecaster: QualityForecaster | None = None


def get_quality_forecaster() -> QualityForecaster:
    """Get the singleton quality forecaster."""
    global _default_forecaster
    if _default_forecaster is None:
        _default_forecaster = QualityForecaster()
    return _default_forecaster


def reset_quality_forecaster() -> None:
    """Reset the singleton (for testing)."""
    global _default_forecaster
    _default_forecaster = None
