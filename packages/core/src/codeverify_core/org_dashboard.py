"""Organization Intelligence Dashboard.

Provides data aggregation, org-level metrics, team comparisons,
risk heatmaps, and ROI calculations for enterprise reporting.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class MetricPeriod(str, Enum):
    """Time period for metric aggregation."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class RiskLevel(str, Enum):
    """Risk level for heatmap."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TeamMetrics:
    """Metrics for a single team."""

    team_id: str = ""
    team_name: str = ""
    total_analyses: int = 0
    total_findings: int = 0
    critical_findings: int = 0
    high_findings: int = 0
    medium_findings: int = 0
    low_findings: int = 0
    verification_coverage_pct: float = 0.0
    ai_generated_code_pct: float = 0.0
    fix_acceptance_rate: float = 0.0
    avg_findings_per_pr: float = 0.0
    repos: list[str] = field(default_factory=list)


@dataclass
class RepoMetrics:
    """Metrics for a single repository."""

    repo_id: str = ""
    repo_name: str = ""
    total_analyses: int = 0
    total_findings: int = 0
    findings_by_severity: dict[str, int] = field(default_factory=dict)
    findings_by_category: dict[str, int] = field(default_factory=dict)
    verification_coverage_pct: float = 0.0
    trend: str = "stable"  # improving, stable, degrading


@dataclass
class RiskHeatmapEntry:
    """A single cell in the risk heatmap."""

    repo_name: str = ""
    team_name: str = ""
    risk_level: RiskLevel = RiskLevel.LOW
    risk_score: float = 0.0
    top_issues: list[str] = field(default_factory=list)


@dataclass
class ROIMetrics:
    """Return on Investment metrics."""

    bugs_caught_pre_production: int = 0
    estimated_cost_avoided_usd: float = 0.0
    security_vulns_prevented: int = 0
    avg_time_to_fix_hours: float = 0.0
    developer_hours_saved: float = 0.0
    cost_per_finding_usd: float = 0.0

    # Industry standard cost estimates per severity
    BUG_COSTS: dict[str, float] = field(
        default_factory=lambda: {
            "critical": 10000.0,
            "high": 5000.0,
            "medium": 1000.0,
            "low": 50.0,
        }
    )


@dataclass
class TrendPoint:
    """A single data point in a trend series."""

    date: str = ""
    value: float = 0.0


@dataclass
class OrgDashboardData:
    """Complete dashboard data for an organization."""

    org_name: str = ""
    period: MetricPeriod = MetricPeriod.MONTHLY
    team_metrics: list[TeamMetrics] = field(default_factory=list)
    repo_metrics: list[RepoMetrics] = field(default_factory=list)
    risk_heatmap: list[RiskHeatmapEntry] = field(default_factory=list)
    roi: ROIMetrics = field(default_factory=ROIMetrics)
    finding_trends: list[TrendPoint] = field(default_factory=list)
    coverage_trends: list[TrendPoint] = field(default_factory=list)
    top_recurring_issues: list[dict[str, Any]] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class MetricsAggregator:
    """Aggregates raw analysis data into dashboard metrics."""

    def __init__(self) -> None:
        self._analyses: list[dict[str, Any]] = []
        self._findings: list[dict[str, Any]] = []

    def ingest_analysis(self, analysis: dict[str, Any]) -> None:
        """Ingest a raw analysis result."""
        self._analyses.append(analysis)

    def ingest_finding(self, finding: dict[str, Any]) -> None:
        """Ingest a raw finding."""
        self._findings.append(finding)

    def compute_team_metrics(self, team_id: str, team_name: str, repos: list[str]) -> TeamMetrics:
        """Compute metrics for a team based on their repos."""
        team_findings = [f for f in self._findings if f.get("repo") in repos]
        team_analyses = [a for a in self._analyses if a.get("repo") in repos]

        sev_counts: defaultdict[str, int] = defaultdict(int)
        for f in team_findings:
            sev_counts[f.get("severity", "low")] += 1

        total_prs = max(len(team_analyses), 1)
        return TeamMetrics(
            team_id=team_id,
            team_name=team_name,
            total_analyses=len(team_analyses),
            total_findings=len(team_findings),
            critical_findings=sev_counts["critical"],
            high_findings=sev_counts["high"],
            medium_findings=sev_counts["medium"],
            low_findings=sev_counts["low"],
            avg_findings_per_pr=len(team_findings) / total_prs,
            repos=repos,
        )

    def compute_repo_metrics(self, repo_name: str) -> RepoMetrics:
        """Compute metrics for a single repository."""
        repo_findings = [f for f in self._findings if f.get("repo") == repo_name]
        repo_analyses = [a for a in self._analyses if a.get("repo") == repo_name]

        by_severity: dict[str, int] = defaultdict(int)
        by_category: dict[str, int] = defaultdict(int)
        for f in repo_findings:
            by_severity[f.get("severity", "low")] += 1
            by_category[f.get("category", "other")] += 1

        return RepoMetrics(
            repo_name=repo_name,
            total_analyses=len(repo_analyses),
            total_findings=len(repo_findings),
            findings_by_severity=dict(by_severity),
            findings_by_category=dict(by_category),
        )

    def compute_risk_heatmap(self, entries: list[dict[str, str]]) -> list[RiskHeatmapEntry]:
        """Compute risk heatmap for repo/team combinations."""
        result: list[RiskHeatmapEntry] = []
        for entry in entries:
            repo = entry.get("repo", "")
            team = entry.get("team", "")
            repo_findings = [f for f in self._findings if f.get("repo") == repo]

            crit = sum(1 for f in repo_findings if f.get("severity") == "critical")
            high = sum(1 for f in repo_findings if f.get("severity") == "high")
            score = crit * 4 + high * 2 + len(repo_findings) * 0.1

            if crit > 0:
                level = RiskLevel.CRITICAL
            elif high > 2:
                level = RiskLevel.HIGH
            elif len(repo_findings) > 10:
                level = RiskLevel.MEDIUM
            else:
                level = RiskLevel.LOW

            top = [f.get("rule_id", "") for f in repo_findings[:3]]
            result.append(
                RiskHeatmapEntry(
                    repo_name=repo,
                    team_name=team,
                    risk_level=level,
                    risk_score=score,
                    top_issues=top,
                )
            )
        return result

    def compute_roi(self) -> ROIMetrics:
        """Calculate ROI metrics from findings data."""
        roi = ROIMetrics()
        roi.bugs_caught_pre_production = len(self._findings)

        for f in self._findings:
            sev = f.get("severity", "low")
            roi.estimated_cost_avoided_usd += roi.BUG_COSTS.get(sev, 50.0)
            if f.get("category") in ("security", "vulnerability"):
                roi.security_vulns_prevented += 1

        if self._findings:
            roi.cost_per_finding_usd = roi.estimated_cost_avoided_usd / len(self._findings)
        roi.developer_hours_saved = len(self._findings) * 0.5  # ~30 min per finding

        return roi

    def compute_top_recurring(self, top_n: int = 10) -> list[dict[str, Any]]:
        """Find the most frequently occurring issue types."""
        rule_counts: dict[str, int] = defaultdict(int)
        for f in self._findings:
            rule_counts[f.get("rule_id", "unknown")] += 1

        sorted_rules = sorted(rule_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"rule_id": r, "count": c} for r, c in sorted_rules]


class OrgDashboard:
    """Organization intelligence dashboard."""

    def __init__(self, org_name: str = "") -> None:
        self.org_name = org_name
        self._aggregator = MetricsAggregator()

    def ingest_analysis(self, analysis: dict[str, Any]) -> None:
        self._aggregator.ingest_analysis(analysis)

    def ingest_finding(self, finding: dict[str, Any]) -> None:
        self._aggregator.ingest_finding(finding)

    def generate_dashboard(
        self,
        teams: list[dict[str, Any]] | None = None,
        period: MetricPeriod = MetricPeriod.MONTHLY,
    ) -> OrgDashboardData:
        """Generate complete dashboard data."""
        teams = teams or []

        team_metrics = [
            self._aggregator.compute_team_metrics(
                t.get("id", ""), t.get("name", ""), t.get("repos", [])
            )
            for t in teams
        ]

        all_repos = set()
        for t in teams:
            all_repos.update(t.get("repos", []))
        repo_metrics = [self._aggregator.compute_repo_metrics(r) for r in all_repos]

        heatmap_entries = [
            {"repo": repo, "team": t.get("name", "")} for t in teams for repo in t.get("repos", [])
        ]
        risk_heatmap = self._aggregator.compute_risk_heatmap(heatmap_entries)

        roi = self._aggregator.compute_roi()
        top_recurring = self._aggregator.compute_top_recurring()

        return OrgDashboardData(
            org_name=self.org_name,
            period=period,
            team_metrics=team_metrics,
            repo_metrics=repo_metrics,
            risk_heatmap=risk_heatmap,
            roi=roi,
            top_recurring_issues=top_recurring,
        )

    def export_csv(self, dashboard: OrgDashboardData) -> str:
        """Export dashboard summary as CSV."""
        lines = ["metric,value"]
        lines.append(f"org_name,{dashboard.org_name}")
        lines.append(f"total_teams,{len(dashboard.team_metrics)}")
        lines.append(f"total_repos,{len(dashboard.repo_metrics)}")
        lines.append(f"bugs_caught,{dashboard.roi.bugs_caught_pre_production}")
        lines.append(f"cost_avoided_usd,{dashboard.roi.estimated_cost_avoided_usd:.2f}")
        lines.append(f"security_vulns_prevented,{dashboard.roi.security_vulns_prevented}")
        lines.append(f"developer_hours_saved,{dashboard.roi.developer_hours_saved:.1f}")
        return "\n".join(lines)
