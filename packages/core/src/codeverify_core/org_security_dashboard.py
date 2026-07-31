"""Organization Security Posture Dashboard.

Executive-facing dashboard providing org-wide verification coverage,
AI code risk heatmap, compliance status, DORA metrics integration,
and trend forecasting.

Features:
- Org-wide verification coverage aggregation
- Risk heatmap across repositories
- Compliance status tracking (SOC2, HIPAA, PCI-DSS)
- DORA metrics integration (deployment freq, lead time, MTTR, change failure)
- Trend analysis and forecasting
- Executive summary generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class RiskLevel(str, Enum):
    """Risk levels for the heatmap."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceFramework(str, Enum):
    """Compliance frameworks tracked."""

    SOC2 = "soc2"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    GDPR = "gdpr"
    ISO_27001 = "iso_27001"


class ComplianceStatus(str, Enum):
    """Status of compliance for a framework."""

    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"


class TrendDirection(str, Enum):
    """Direction of a trend metric."""

    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


@dataclass
class RepoMetrics:
    """Verification metrics for a single repository."""

    repo_id: str = ""
    repo_name: str = ""
    total_findings: int = 0
    critical_findings: int = 0
    high_findings: int = 0
    medium_findings: int = 0
    low_findings: int = 0
    verification_coverage: float = 0.0
    ai_code_ratio: float = 0.0
    trust_score: float = 0.0
    last_scan: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def risk_level(self) -> RiskLevel:
        if self.critical_findings > 0:
            return RiskLevel.CRITICAL
        if self.high_findings > 3:
            return RiskLevel.HIGH
        if self.high_findings > 0 or self.medium_findings > 5:
            return RiskLevel.MEDIUM
        if self.medium_findings > 0:
            return RiskLevel.LOW
        return RiskLevel.MINIMAL

    @property
    def risk_score(self) -> float:
        score = (
            self.critical_findings * 10
            + self.high_findings * 5
            + self.medium_findings * 2
            + self.low_findings * 0.5
        )
        coverage_factor = max(0.1, 1.0 - self.verification_coverage)
        return min(score * coverage_factor, 100.0)


@dataclass
class DORAMetrics:
    """DORA (DevOps Research and Assessment) metrics."""

    deployment_frequency_per_day: float = 0.0
    lead_time_hours: float = 0.0
    mean_time_to_recovery_hours: float = 0.0
    change_failure_rate: float = 0.0

    @property
    def deployment_frequency_rating(self) -> str:
        if self.deployment_frequency_per_day >= 1.0:
            return "Elite"
        if self.deployment_frequency_per_day >= 0.14:  # ~weekly
            return "High"
        if self.deployment_frequency_per_day >= 0.033:  # ~monthly
            return "Medium"
        return "Low"

    @property
    def lead_time_rating(self) -> str:
        if self.lead_time_hours < 24:
            return "Elite"
        if self.lead_time_hours < 168:  # 1 week
            return "High"
        if self.lead_time_hours < 720:  # 1 month
            return "Medium"
        return "Low"

    @property
    def overall_rating(self) -> str:
        ratings = [self.deployment_frequency_rating, self.lead_time_rating]
        score_map = {"Elite": 4, "High": 3, "Medium": 2, "Low": 1}
        avg = sum(score_map.get(r, 1) for r in ratings) / len(ratings)
        if avg >= 3.5:
            return "Elite"
        if avg >= 2.5:
            return "High"
        if avg >= 1.5:
            return "Medium"
        return "Low"


@dataclass
class ComplianceRecord:
    """Compliance status for a specific framework."""

    framework: ComplianceFramework = ComplianceFramework.SOC2
    status: ComplianceStatus = ComplianceStatus.NOT_ASSESSED
    controls_total: int = 0
    controls_met: int = 0
    last_assessed: datetime = field(default_factory=lambda: datetime.now(UTC))
    notes: str = ""

    @property
    def coverage_pct(self) -> float:
        return (self.controls_met / self.controls_total * 100) if self.controls_total > 0 else 0.0


@dataclass
class TrendDataPoint:
    """A data point in a trend series."""

    date: datetime = field(default_factory=lambda: datetime.now(UTC))
    value: float = 0.0
    label: str = ""


@dataclass
class TrendSeries:
    """A time-series of metric values."""

    metric_name: str = ""
    data_points: list[TrendDataPoint] = field(default_factory=list)

    @property
    def direction(self) -> TrendDirection:
        if len(self.data_points) < 2:
            return TrendDirection.STABLE
        recent = self.data_points[-1].value
        previous = self.data_points[-2].value
        if recent > previous * 1.05:
            return TrendDirection.IMPROVING
        if recent < previous * 0.95:
            return TrendDirection.DECLINING
        return TrendDirection.STABLE

    @property
    def latest_value(self) -> float:
        return self.data_points[-1].value if self.data_points else 0.0

    def add_point(self, value: float, date: datetime | None = None) -> None:
        self.data_points.append(
            TrendDataPoint(
                date=date or datetime.now(UTC),
                value=value,
            )
        )


@dataclass
class OrgSecurityPosture:
    """Complete organization security posture snapshot."""

    org_name: str = ""
    total_repos: int = 0
    repos_scanned: int = 0
    overall_risk_score: float = 0.0
    overall_risk_level: RiskLevel = RiskLevel.MINIMAL
    total_findings: int = 0
    critical_findings: int = 0
    avg_trust_score: float = 0.0
    avg_verification_coverage: float = 0.0
    repo_metrics: list[RepoMetrics] = field(default_factory=list)
    compliance: list[ComplianceRecord] = field(default_factory=list)
    dora: DORAMetrics = field(default_factory=DORAMetrics)
    trends: list[TrendSeries] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def scan_coverage_pct(self) -> float:
        return (self.repos_scanned / self.total_repos * 100) if self.total_repos > 0 else 0.0

    def to_executive_summary(self) -> dict[str, Any]:
        return {
            "organization": self.org_name,
            "risk_level": self.overall_risk_level.value,
            "risk_score": round(self.overall_risk_score, 1),
            "repos_scanned": f"{self.repos_scanned}/{self.total_repos}",
            "scan_coverage": f"{self.scan_coverage_pct:.0f}%",
            "critical_findings": self.critical_findings,
            "total_findings": self.total_findings,
            "avg_trust_score": round(self.avg_trust_score, 1),
            "avg_verification_coverage": f"{self.avg_verification_coverage:.0f}%",
            "dora_rating": self.dora.overall_rating,
            "compliance_summary": {c.framework.value: c.status.value for c in self.compliance},
        }


class OrgSecurityDashboard:
    """Aggregates and computes organization-wide security posture."""

    def __init__(self, org_name: str = "") -> None:
        self._org_name = org_name
        self._repos: dict[str, RepoMetrics] = {}
        self._compliance: dict[ComplianceFramework, ComplianceRecord] = {}
        self._dora = DORAMetrics()
        self._trends: dict[str, TrendSeries] = {}

    def add_repo_metrics(self, metrics: RepoMetrics) -> None:
        self._repos[metrics.repo_id] = metrics

    def set_compliance(self, record: ComplianceRecord) -> None:
        self._compliance[record.framework] = record

    def set_dora_metrics(self, dora: DORAMetrics) -> None:
        self._dora = dora

    def add_trend_point(self, metric_name: str, value: float, date: datetime | None = None) -> None:
        if metric_name not in self._trends:
            self._trends[metric_name] = TrendSeries(metric_name=metric_name)
        self._trends[metric_name].add_point(value, date)

    def generate_posture(self) -> OrgSecurityPosture:
        """Generate the complete security posture snapshot."""
        repos = list(self._repos.values())
        total_findings = sum(r.total_findings for r in repos)
        critical = sum(r.critical_findings for r in repos)
        avg_trust = sum(r.trust_score for r in repos) / len(repos) if repos else 0.0
        avg_coverage = sum(r.verification_coverage for r in repos) / len(repos) if repos else 0.0
        overall_risk = sum(r.risk_score for r in repos) / len(repos) if repos else 0.0

        risk_level = RiskLevel.MINIMAL
        if overall_risk > 50:
            risk_level = RiskLevel.CRITICAL
        elif overall_risk > 30:
            risk_level = RiskLevel.HIGH
        elif overall_risk > 15:
            risk_level = RiskLevel.MEDIUM
        elif overall_risk > 5:
            risk_level = RiskLevel.LOW

        posture = OrgSecurityPosture(
            org_name=self._org_name,
            total_repos=len(repos),
            repos_scanned=sum(1 for r in repos if r.verification_coverage > 0),
            overall_risk_score=overall_risk,
            overall_risk_level=risk_level,
            total_findings=total_findings,
            critical_findings=critical,
            avg_trust_score=avg_trust,
            avg_verification_coverage=avg_coverage * 100,
            repo_metrics=repos,
            compliance=list(self._compliance.values()),
            dora=self._dora,
            trends=list(self._trends.values()),
        )
        logger.info("posture_generated", org=self._org_name, risk=risk_level.value)
        return posture

    def get_risk_heatmap(self) -> list[dict[str, Any]]:
        """Generate a risk heatmap across repositories."""
        return [
            {
                "repo": r.repo_name,
                "risk_level": r.risk_level.value,
                "risk_score": round(r.risk_score, 1),
                "coverage": f"{r.verification_coverage * 100:.0f}%",
                "critical": r.critical_findings,
                "ai_ratio": f"{r.ai_code_ratio * 100:.0f}%",
            }
            for r in sorted(self._repos.values(), key=lambda r: r.risk_score, reverse=True)
        ]

    @property
    def total_repos(self) -> int:
        return len(self._repos)


_dashboard: OrgSecurityDashboard | None = None


def get_org_security_dashboard(org_name: str = "default") -> OrgSecurityDashboard:
    """Get the singleton OrgSecurityDashboard instance."""
    global _dashboard
    if _dashboard is None:
        _dashboard = OrgSecurityDashboard(org_name)
    return _dashboard


def reset_org_security_dashboard() -> None:
    """Reset the singleton (useful for testing)."""
    global _dashboard
    _dashboard = None
