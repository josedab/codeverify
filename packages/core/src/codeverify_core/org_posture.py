"""Organization Security Posture Score.

Aggregates org-wide verification coverage, finding density, fix rate,
and compliance status into a single health score with DORA metrics
integration and executive dashboard data.

Features:
- Composite org-wide security posture score (0-100)
- DORA metrics integration (deployment freq, lead time, MTTR, change failure rate)
- Repository risk heatmap with weighted scoring
- Trend detection (improving/stable/declining)
- Weekly digest generation for executive reporting
- Alert thresholds for score drops
"""

from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

import structlog

logger = structlog.get_logger()


class PostureTrend(str, Enum):
    """Trend direction for posture score."""

    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


class RiskLevel(str, Enum):
    """Risk level for a repository."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class AlertSeverity(str, Enum):
    """Severity of posture alerts."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class DORAMetricLevel(str, Enum):
    """DORA performance levels."""

    ELITE = "elite"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RepositoryMetrics:
    """Verification metrics for a single repository."""

    repo_id: str = ""
    repo_name: str = ""
    verification_coverage: float = 0.0  # 0-1
    finding_density: float = 0.0  # findings per 1000 LOC
    fix_rate: float = 0.0  # 0-1
    mean_time_to_fix_hours: float = 0.0
    critical_findings: int = 0
    high_findings: int = 0
    medium_findings: int = 0
    low_findings: int = 0
    total_loc: int = 0
    last_scan_at: datetime | None = None
    is_compliant: bool = True


@dataclass
class DORAMetrics:
    """DORA (DevOps Research & Assessment) metrics."""

    deployment_frequency_per_day: float = 0.0
    lead_time_hours: float = 0.0
    mean_time_to_restore_hours: float = 0.0
    change_failure_rate: float = 0.0  # 0-1

    @property
    def deployment_level(self) -> DORAMetricLevel:
        if self.deployment_frequency_per_day >= 1.0:
            return DORAMetricLevel.ELITE
        if self.deployment_frequency_per_day >= 0.14:  # ~weekly
            return DORAMetricLevel.HIGH
        if self.deployment_frequency_per_day >= 0.033:  # ~monthly
            return DORAMetricLevel.MEDIUM
        return DORAMetricLevel.LOW

    @property
    def lead_time_level(self) -> DORAMetricLevel:
        if self.lead_time_hours < 24:
            return DORAMetricLevel.ELITE
        if self.lead_time_hours < 168:  # 1 week
            return DORAMetricLevel.HIGH
        if self.lead_time_hours < 720:  # 1 month
            return DORAMetricLevel.MEDIUM
        return DORAMetricLevel.LOW

    @property
    def mttr_level(self) -> DORAMetricLevel:
        if self.mean_time_to_restore_hours < 1:
            return DORAMetricLevel.ELITE
        if self.mean_time_to_restore_hours < 24:
            return DORAMetricLevel.HIGH
        if self.mean_time_to_restore_hours < 168:
            return DORAMetricLevel.MEDIUM
        return DORAMetricLevel.LOW

    @property
    def change_failure_level(self) -> DORAMetricLevel:
        if self.change_failure_rate < 0.05:
            return DORAMetricLevel.ELITE
        if self.change_failure_rate < 0.10:
            return DORAMetricLevel.HIGH
        if self.change_failure_rate < 0.15:
            return DORAMetricLevel.MEDIUM
        return DORAMetricLevel.LOW

    @property
    def overall_level(self) -> DORAMetricLevel:
        levels = [
            self.deployment_level,
            self.lead_time_level,
            self.mttr_level,
            self.change_failure_level,
        ]
        level_order = [
            DORAMetricLevel.LOW,
            DORAMetricLevel.MEDIUM,
            DORAMetricLevel.HIGH,
            DORAMetricLevel.ELITE,
        ]
        indices = [level_order.index(level) for level in levels]
        median_idx = sorted(indices)[len(indices) // 2]
        return level_order[median_idx]


@dataclass
class PostureScore:
    """Composite organization security posture score."""

    overall_score: float = 0.0  # 0-100
    coverage_score: float = 0.0  # 0-100
    finding_score: float = 0.0  # 0-100
    fix_rate_score: float = 0.0  # 0-100
    compliance_score: float = 0.0  # 0-100
    dora_score: float = 0.0  # 0-100
    trend: PostureTrend = PostureTrend.STABLE
    risk_level: RiskLevel = RiskLevel.MEDIUM
    calculated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class PostureAlert:
    """Alert for posture score changes."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    severity: AlertSeverity = AlertSeverity.INFO
    title: str = ""
    message: str = ""
    score_before: float = 0.0
    score_after: float = 0.0
    triggered_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class HeatmapEntry:
    """Risk heatmap entry for a repository."""

    repo_name: str = ""
    risk_level: RiskLevel = RiskLevel.MEDIUM
    score: float = 0.0
    top_issue: str = ""
    finding_count: int = 0


@dataclass
class ExecutiveDigest:
    """Weekly executive digest."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    org_name: str = ""
    period_start: datetime = field(default_factory=lambda: datetime.now(UTC))
    period_end: datetime = field(default_factory=lambda: datetime.now(UTC))
    posture_score: PostureScore | None = None
    dora_metrics: DORAMetrics | None = None
    heatmap: list[HeatmapEntry] = field(default_factory=list)
    alerts: list[PostureAlert] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    summary_markdown: str = ""


class PostureCalculator:
    """Calculates the composite posture score."""

    WEIGHTS = {
        "coverage": 0.25,
        "findings": 0.25,
        "fix_rate": 0.20,
        "compliance": 0.15,
        "dora": 0.15,
    }

    def calculate(
        self,
        repos: list[RepositoryMetrics],
        dora: DORAMetrics | None = None,
    ) -> PostureScore:
        """Calculate the composite posture score."""
        if not repos:
            return PostureScore()

        coverage_score = self._score_coverage(repos)
        finding_score = self._score_findings(repos)
        fix_rate_score = self._score_fix_rate(repos)
        compliance_score = self._score_compliance(repos)
        dora_score = self._score_dora(dora) if dora else 50.0

        overall = (
            coverage_score * self.WEIGHTS["coverage"]
            + finding_score * self.WEIGHTS["findings"]
            + fix_rate_score * self.WEIGHTS["fix_rate"]
            + compliance_score * self.WEIGHTS["compliance"]
            + dora_score * self.WEIGHTS["dora"]
        )

        risk_level = self._classify_risk(overall)

        return PostureScore(
            overall_score=round(overall, 1),
            coverage_score=round(coverage_score, 1),
            finding_score=round(finding_score, 1),
            fix_rate_score=round(fix_rate_score, 1),
            compliance_score=round(compliance_score, 1),
            dora_score=round(dora_score, 1),
            risk_level=risk_level,
        )

    def _score_coverage(self, repos: list[RepositoryMetrics]) -> float:
        if not repos:
            return 0.0
        avg = statistics.mean(r.verification_coverage for r in repos)
        return min(100.0, avg * 100)

    def _score_findings(self, repos: list[RepositoryMetrics]) -> float:
        if not repos:
            return 100.0
        total_critical = sum(r.critical_findings for r in repos)
        total_high = sum(r.high_findings for r in repos)
        penalty = total_critical * 20 + total_high * 5
        return max(0.0, 100.0 - penalty)

    def _score_fix_rate(self, repos: list[RepositoryMetrics]) -> float:
        if not repos:
            return 0.0
        rates = [r.fix_rate for r in repos if r.fix_rate > 0]
        if not rates:
            return 50.0
        return min(100.0, statistics.mean(rates) * 100)

    def _score_compliance(self, repos: list[RepositoryMetrics]) -> float:
        if not repos:
            return 0.0
        compliant = sum(1 for r in repos if r.is_compliant)
        return (compliant / len(repos)) * 100

    def _score_dora(self, dora: DORAMetrics) -> float:
        level_scores = {
            DORAMetricLevel.ELITE: 100,
            DORAMetricLevel.HIGH: 75,
            DORAMetricLevel.MEDIUM: 50,
            DORAMetricLevel.LOW: 25,
        }
        return level_scores.get(dora.overall_level, 50)

    def _classify_risk(self, score: float) -> RiskLevel:
        if score >= 90:
            return RiskLevel.MINIMAL
        if score >= 70:
            return RiskLevel.LOW
        if score >= 50:
            return RiskLevel.MEDIUM
        if score >= 30:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL


class TrendDetector:
    """Detects trends in posture scores over time."""

    def detect(
        self,
        history: list[PostureScore],
        window: int = 5,
    ) -> PostureTrend:
        """Detect trend from score history."""
        if len(history) < 2:
            return PostureTrend.STABLE

        recent = history[-window:]
        scores = [s.overall_score for s in recent]

        if len(scores) < 2:
            return PostureTrend.STABLE

        first_half = statistics.mean(scores[: len(scores) // 2])
        second_half = statistics.mean(scores[len(scores) // 2 :])
        diff = second_half - first_half

        if diff > 3.0:
            return PostureTrend.IMPROVING
        if diff < -3.0:
            return PostureTrend.DECLINING
        return PostureTrend.STABLE


class OrgSecurityPostureService:
    """Main service for organization security posture scoring."""

    def __init__(self, org_name: str = "") -> None:
        self._org_name = org_name
        self._calculator = PostureCalculator()
        self._trend_detector = TrendDetector()
        self._repos: dict[str, RepositoryMetrics] = {}
        self._dora: DORAMetrics | None = None
        self._history: list[PostureScore] = []
        self._alerts: list[PostureAlert] = []
        self._alert_threshold_drop: float = 5.0

    def set_repo_metrics(self, metrics: RepositoryMetrics) -> None:
        """Set or update metrics for a repository."""
        self._repos[metrics.repo_id] = metrics

    def set_dora_metrics(self, dora: DORAMetrics) -> None:
        self._dora = dora

    def calculate_posture(self) -> PostureScore:
        """Calculate current posture score."""
        repos = list(self._repos.values())
        score = self._calculator.calculate(repos, self._dora)

        if self._history:
            score.trend = self._trend_detector.detect(self._history)
            last = self._history[-1]
            drop = last.overall_score - score.overall_score
            if drop >= self._alert_threshold_drop:
                self._alerts.append(
                    PostureAlert(
                        severity=AlertSeverity.WARNING if drop < 10 else AlertSeverity.CRITICAL,
                        title="Posture Score Drop",
                        message=f"Score dropped {drop:.1f} points ({last.overall_score:.1f} → {score.overall_score:.1f})",
                        score_before=last.overall_score,
                        score_after=score.overall_score,
                    )
                )

        self._history.append(score)
        return score

    def get_heatmap(self) -> list[HeatmapEntry]:
        """Generate risk heatmap across repositories."""
        entries: list[HeatmapEntry] = []
        for repo in self._repos.values():
            total_findings = (
                repo.critical_findings
                + repo.high_findings
                + repo.medium_findings
                + repo.low_findings
            )
            repo_score = repo.verification_coverage * 50 + repo.fix_rate * 50
            risk = (
                RiskLevel.CRITICAL
                if repo.critical_findings > 0
                else RiskLevel.HIGH
                if repo.high_findings > 3
                else RiskLevel.MEDIUM
                if total_findings > 10
                else RiskLevel.LOW
                if total_findings > 0
                else RiskLevel.MINIMAL
            )
            top_issue = (
                "Critical findings"
                if repo.critical_findings > 0
                else "High findings"
                if repo.high_findings > 0
                else "Low coverage"
                if repo.verification_coverage < 0.5
                else "Healthy"
            )
            entries.append(
                HeatmapEntry(
                    repo_name=repo.repo_name,
                    risk_level=risk,
                    score=round(repo_score, 1),
                    top_issue=top_issue,
                    finding_count=total_findings,
                )
            )

        entries.sort(key=lambda e: e.score)
        return entries

    def generate_digest(self) -> ExecutiveDigest:
        """Generate an executive digest."""
        score = self.calculate_posture()
        heatmap = self.get_heatmap()

        recommendations: list[str] = []
        if score.coverage_score < 50:
            recommendations.append("Increase verification coverage across repositories")
        if score.finding_score < 70:
            recommendations.append("Prioritize fixing critical and high-severity findings")
        if score.fix_rate_score < 60:
            recommendations.append("Improve finding fix rate with autofix suggestions")
        if score.compliance_score < 80:
            recommendations.append("Address compliance gaps in non-compliant repositories")

        critical_repos = [e for e in heatmap if e.risk_level == RiskLevel.CRITICAL]
        if critical_repos:
            names = ", ".join(e.repo_name for e in critical_repos[:3])
            recommendations.append(f"Immediate attention needed: {names}")

        summary = (
            f"# Security Posture Digest — {self._org_name}\n\n"
            f"**Overall Score: {score.overall_score}/100** ({score.risk_level.value})\n"
            f"**Trend: {score.trend.value}**\n\n"
            f"| Dimension | Score |\n|---|---|\n"
            f"| Coverage | {score.coverage_score} |\n"
            f"| Findings | {score.finding_score} |\n"
            f"| Fix Rate | {score.fix_rate_score} |\n"
            f"| Compliance | {score.compliance_score} |\n"
            f"| DORA | {score.dora_score} |\n\n"
            f"**Repos at Risk:** {len(critical_repos)}\n"
        )

        return ExecutiveDigest(
            org_name=self._org_name,
            posture_score=score,
            dora_metrics=self._dora,
            heatmap=heatmap,
            alerts=self._alerts[-10:],
            recommendations=recommendations,
            summary_markdown=summary,
        )

    def get_alerts(self) -> list[PostureAlert]:
        return list(self._alerts)

    def get_history(self) -> list[PostureScore]:
        return list(self._history)


# ─── Singleton Access ──────────────────────────────────────────────────


_org_posture_instance: OrgSecurityPostureService | None = None


def get_org_posture_service() -> OrgSecurityPostureService:
    """Get or create the singleton OrgSecurityPostureService."""
    global _org_posture_instance
    if _org_posture_instance is None:
        _org_posture_instance = OrgSecurityPostureService()
    return _org_posture_instance


def reset_org_posture_service() -> None:
    """Reset the singleton (for testing)."""
    global _org_posture_instance
    _org_posture_instance = None
