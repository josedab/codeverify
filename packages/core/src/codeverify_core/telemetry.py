"""Verification Telemetry & ROI Dashboard - Track findings lifecycle and measure value.

Provides event-based telemetry, finding lifecycle tracking, NIST-based cost
estimation, and an ROI dashboard with executive summaries and CSV export.
"""

from __future__ import annotations

import csv
import io
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class FindingLifecycle(str, Enum):
    """Lifecycle stages of a verification finding."""
    DETECTED = "detected"
    ACKNOWLEDGED = "acknowledged"
    FIX_IN_PROGRESS = "fix_in_progress"
    FIXED = "fixed"
    VERIFIED = "verified"
    FALSE_POSITIVE = "false_positive"
    WONT_FIX = "wont_fix"


class MetricType(str, Enum):
    """Types of metrics available for trend analysis."""
    BUGS_PREVENTED = "bugs_prevented"
    TIME_SAVED_HOURS = "time_saved_hours"
    COST_PER_FINDING = "cost_per_finding"
    FINDINGS_BY_SEVERITY = "findings_by_severity"
    FIX_RATE = "fix_rate"
    MEAN_TIME_TO_FIX_HOURS = "mean_time_to_fix_hours"
    VERIFICATION_COVERAGE = "verification_coverage"


@dataclass
class TelemetryEvent:
    """A single telemetry event emitted during verification."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""  # e.g. "finding_detected", "finding_fixed", "analysis_completed"
    repo_id: str = ""
    finding_id: str | None = None
    severity: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FindingMetrics:
    """Metrics tracked for a single finding across its lifecycle."""
    finding_id: str = ""
    detected_at: float = 0.0
    severity: str = "medium"
    lifecycle: FindingLifecycle = FindingLifecycle.DETECTED
    fixed_at: float | None = None
    time_to_fix_hours: float | None = None
    estimated_cost_if_production: float = 0.0  # NIST-based bug cost


@dataclass
class ROIReport:
    """Aggregated ROI report for a time period."""
    period_start: float = 0.0
    period_end: float = 0.0
    total_findings: int = 0
    findings_by_severity: dict[str, int] = field(default_factory=dict)
    bugs_prevented: int = 0
    estimated_cost_saved: float = 0.0
    total_analyses: int = 0
    average_time_to_fix_hours: float = 0.0
    fix_rate: float = 0.0  # 0-1
    verification_coverage: float = 0.0  # 0-1, percentage of code verified
    developer_hours_saved: float = 0.0
    roi_multiplier: float = 0.0  # cost_saved / tool_cost
    trends: dict[str, list[float]] = field(default_factory=dict)


# NIST / IBM Systems Sciences Institute average cost of a bug found in
# production, broken down by severity.
_DEFAULT_NIST_COSTS: dict[str, float] = {
    "critical": 15_000.0,
    "high": 5_000.0,
    "medium": 1_500.0,
    "low": 500.0,
}


class CostEstimator:
    """Estimate the monetary value of catching a bug before production.

    Uses NIST-based baseline costs that can be adjusted with per-severity
    multipliers (e.g. for regulated industries).
    """

    def __init__(
        self,
        base_costs: dict[str, float] | None = None,
        multipliers: dict[str, float] | None = None,
    ) -> None:
        self._base_costs = dict(base_costs or _DEFAULT_NIST_COSTS)
        self._multipliers = dict(multipliers or {})

    def estimate_bug_cost(self, severity: str) -> float:
        """Return the estimated production-bug cost for *severity*."""
        severity_lower = severity.lower()
        base = self._base_costs.get(severity_lower, self._base_costs.get("medium", 1_500.0))
        multiplier = self._multipliers.get(severity_lower, 1.0)
        return base * multiplier

    def estimate_total_savings(self, findings: list[FindingMetrics]) -> float:
        """Sum the estimated cost avoidance across all *findings*."""
        return sum(self.estimate_bug_cost(f.severity) for f in findings)


class TelemetryCollector:
    """Collects telemetry events and finding lifecycle data."""

    def __init__(self) -> None:
        self._events: list[TelemetryEvent] = []
        self._findings: dict[str, FindingMetrics] = {}

    @property
    def events(self) -> list[TelemetryEvent]:
        return list(self._events)

    @property
    def findings(self) -> dict[str, FindingMetrics]:
        return dict(self._findings)

    def record_event(
        self,
        event_type: str,
        repo_id: str,
        finding_id: str | None = None,
        severity: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TelemetryEvent:
        """Record an arbitrary telemetry event."""
        event = TelemetryEvent(
            event_type=event_type,
            repo_id=repo_id,
            finding_id=finding_id,
            severity=severity,
            metadata=metadata or {},
        )
        self._events.append(event)
        logger.debug(
            "Telemetry event recorded",
            event_type=event_type,
            repo_id=repo_id,
            finding_id=finding_id,
        )
        return event

    def record_finding_detected(
        self,
        finding_id: str,
        repo_id: str,
        severity: str,
    ) -> FindingMetrics:
        """Record that a new finding has been detected."""
        now = time.time()
        cost_estimator = CostEstimator()
        metrics = FindingMetrics(
            finding_id=finding_id,
            detected_at=now,
            severity=severity,
            lifecycle=FindingLifecycle.DETECTED,
            estimated_cost_if_production=cost_estimator.estimate_bug_cost(severity),
        )
        self._findings[finding_id] = metrics
        self.record_event(
            event_type="finding_detected",
            repo_id=repo_id,
            finding_id=finding_id,
            severity=severity,
            metadata={"estimated_cost": metrics.estimated_cost_if_production},
        )
        logger.info(
            "Finding detected",
            finding_id=finding_id,
            severity=severity,
            estimated_cost=metrics.estimated_cost_if_production,
        )
        return metrics

    def record_finding_fixed(self, finding_id: str) -> FindingMetrics | None:
        """Mark a finding as fixed and compute time-to-fix."""
        metrics = self._findings.get(finding_id)
        if metrics is None:
            logger.warning("Cannot mark unknown finding as fixed", finding_id=finding_id)
            return None
        now = time.time()
        metrics.lifecycle = FindingLifecycle.FIXED
        metrics.fixed_at = now
        metrics.time_to_fix_hours = (now - metrics.detected_at) / 3600.0
        self.record_event(
            event_type="finding_fixed",
            repo_id="",
            finding_id=finding_id,
            severity=metrics.severity,
            metadata={"time_to_fix_hours": metrics.time_to_fix_hours},
        )
        logger.info(
            "Finding fixed",
            finding_id=finding_id,
            time_to_fix_hours=round(metrics.time_to_fix_hours, 2),
        )
        return metrics

    def record_finding_false_positive(self, finding_id: str) -> FindingMetrics | None:
        """Mark a finding as a false positive."""
        metrics = self._findings.get(finding_id)
        if metrics is None:
            logger.warning("Cannot mark unknown finding as false positive", finding_id=finding_id)
            return None
        metrics.lifecycle = FindingLifecycle.FALSE_POSITIVE
        self.record_event(
            event_type="finding_false_positive",
            repo_id="",
            finding_id=finding_id,
            severity=metrics.severity,
        )
        logger.info("Finding marked as false positive", finding_id=finding_id)
        return metrics

    def record_analysis_completed(
        self,
        repo_id: str,
        duration_ms: float,
        findings_count: int,
    ) -> TelemetryEvent:
        """Record that a verification analysis run has completed."""
        return self.record_event(
            event_type="analysis_completed",
            repo_id=repo_id,
            metadata={"duration_ms": duration_ms, "findings_count": findings_count},
        )


_DEFAULT_DEVELOPER_HOURLY_RATE = 75.0

# Rough estimate of developer-hours a detected bug saves.
_DEFAULT_HOURS_SAVED_PER_BUG: dict[str, float] = {
    "critical": 40.0,
    "high": 16.0,
    "medium": 6.0,
    "low": 2.0,
}

_DEFAULT_TOOL_MONTHLY_COST = 500.0

_SECONDS_PER_WEEK = 7 * 24 * 3600.0
_SECONDS_PER_DAY = 24 * 3600.0

_TERMINAL_LIFECYCLES = (
    FindingLifecycle.FIXED,
    FindingLifecycle.VERIFIED,
    FindingLifecycle.FALSE_POSITIVE,
    FindingLifecycle.WONT_FIX,
)
_NON_ACTIONABLE = (FindingLifecycle.FALSE_POSITIVE, FindingLifecycle.WONT_FIX)


class ROIDashboard:
    """Aggregates telemetry into ROI reports, summaries, and trend data."""

    def __init__(
        self,
        collector: TelemetryCollector,
        cost_estimator: CostEstimator | None = None,
        developer_hourly_rate: float = _DEFAULT_DEVELOPER_HOURLY_RATE,
        hours_saved_per_bug: dict[str, float] | None = None,
        tool_monthly_cost: float = _DEFAULT_TOOL_MONTHLY_COST,
    ) -> None:
        self.collector = collector
        self.cost_estimator = cost_estimator or CostEstimator()
        self._developer_hourly_rate = developer_hourly_rate
        self._hours_saved_per_bug = dict(hours_saved_per_bug or _DEFAULT_HOURS_SAVED_PER_BUG)
        self._tool_monthly_cost = tool_monthly_cost

    # -- report generation ---------------------------------------------------

    def generate_report(self, period_days: int = 30) -> ROIReport:
        """Build an :class:`ROIReport` covering the last *period_days*."""
        now = time.time()
        period_start = now - (period_days * _SECONDS_PER_DAY)

        period_findings = [f for f in self.collector.findings.values() if f.detected_at >= period_start]
        period_events = [e for e in self.collector.events if e.timestamp >= period_start]

        by_severity: dict[str, int] = {}
        for f in period_findings:
            sev = f.severity.lower()
            by_severity[sev] = by_severity.get(sev, 0) + 1

        actionable = [f for f in period_findings if f.lifecycle not in _NON_ACTIONABLE]
        bugs_prevented = len(actionable)
        estimated_cost_saved = self.cost_estimator.estimate_total_savings(actionable)
        total_analyses = sum(1 for e in period_events if e.event_type == "analysis_completed")

        fixed = [f for f in period_findings if f.lifecycle == FindingLifecycle.FIXED]
        resolved = [f for f in period_findings if f.lifecycle in _TERMINAL_LIFECYCLES]
        fix_rate = len(resolved) / max(len(period_findings), 1)

        fix_times = [f.time_to_fix_hours for f in fixed if f.time_to_fix_hours is not None]
        average_ttf = sum(fix_times) / max(len(fix_times), 1)

        dev_hours = sum(self._hours_saved_per_bug.get(f.severity.lower(), 4.0) for f in actionable)

        all_repos = {e.repo_id for e in self.collector.events if e.repo_id}
        period_repos = {e.repo_id for e in period_events if e.repo_id}
        coverage = len(period_repos) / max(len(all_repos), 1)

        tool_cost = self._tool_monthly_cost * (period_days / 30.0)
        roi_mult = estimated_cost_saved / max(tool_cost, 1.0)

        return ROIReport(
            period_start=period_start,
            period_end=now,
            total_findings=len(period_findings),
            findings_by_severity=by_severity,
            bugs_prevented=bugs_prevented,
            estimated_cost_saved=round(estimated_cost_saved, 2),
            total_analyses=total_analyses,
            average_time_to_fix_hours=round(average_ttf, 2),
            fix_rate=round(fix_rate, 4),
            verification_coverage=round(coverage, 4),
            developer_hours_saved=round(dev_hours, 2),
            roi_multiplier=round(roi_mult, 2),
            trends=self._build_trends(period_start, now),
        )

    # -- executive summary ---------------------------------------------------

    def generate_executive_summary(self, report: ROIReport) -> str:
        """Return a Markdown-formatted executive summary of *report*."""
        lines: list[str] = [
            "# Verification ROI Report",
            "",
            f"**Period:** {_ts_to_date(report.period_start)} -- {_ts_to_date(report.period_end)}",
            "",
            "## Key Metrics",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Total findings | {report.total_findings} |",
            f"| Bugs prevented | {report.bugs_prevented} |",
            f"| Estimated cost saved | ${report.estimated_cost_saved:,.2f} |",
            f"| Developer hours saved | {report.developer_hours_saved:.1f} h |",
            f"| ROI multiplier | {report.roi_multiplier:.1f}x |",
            f"| Fix rate | {report.fix_rate * 100:.1f}% |",
            f"| Avg time to fix | {report.average_time_to_fix_hours:.1f} h |",
            f"| Verification coverage | {report.verification_coverage * 100:.1f}% |",
            f"| Total analyses | {report.total_analyses} |",
            "",
            "## Findings by Severity",
            "",
        ]

        if report.findings_by_severity:
            lines += ["| Severity | Count |", "|---|---|"]
            for sev in ("critical", "high", "medium", "low"):
                count = report.findings_by_severity.get(sev, 0)
                if count:
                    lines.append(f"| {sev.capitalize()} | {count} |")
        else:
            lines.append("No findings in this period.")

        lines += ["", "## Summary", ""]
        if report.roi_multiplier >= 1.0:
            lines.append(
                f"Verification delivered **{report.roi_multiplier:.1f}x** return on investment, "
                f"saving an estimated **${report.estimated_cost_saved:,.2f}** by catching "
                f"**{report.bugs_prevented}** bugs before production."
            )
        else:
            lines.append(
                f"Verification is currently below break-even ({report.roi_multiplier:.1f}x). "
                "Consider expanding coverage to higher-risk repositories to increase value."
            )
        return "\n".join(lines)

    # -- CSV export ----------------------------------------------------------

    def export_csv(self, report: ROIReport) -> str:
        """Export *report* as a CSV string."""
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["Metric", "Value"])
        rows: list[tuple[str, object]] = [
            ("Period Start", _ts_to_date(report.period_start)),
            ("Period End", _ts_to_date(report.period_end)),
            ("Total Findings", report.total_findings),
            ("Bugs Prevented", report.bugs_prevented),
            ("Estimated Cost Saved ($)", f"{report.estimated_cost_saved:.2f}"),
            ("Total Analyses", report.total_analyses),
            ("Average Time to Fix (h)", f"{report.average_time_to_fix_hours:.2f}"),
            ("Fix Rate", f"{report.fix_rate:.4f}"),
            ("Verification Coverage", f"{report.verification_coverage:.4f}"),
            ("Developer Hours Saved", f"{report.developer_hours_saved:.2f}"),
            ("ROI Multiplier", f"{report.roi_multiplier:.2f}"),
        ]
        for label, val in rows:
            w.writerow([label, val])
        w.writerow([])
        w.writerow(["Severity", "Count"])
        for sev in ("critical", "high", "medium", "low"):
            w.writerow([sev.capitalize(), report.findings_by_severity.get(sev, 0)])
        if report.trends:
            w.writerow([])
            w.writerow(["Trend Metric", "Week Values (comma-separated)"])
            for name, values in report.trends.items():
                w.writerow([name, ",".join(f"{v:.2f}" for v in values)])
        return buf.getvalue()

    # -- trend data ----------------------------------------------------------

    def get_trend_data(self, metric: MetricType, weeks: int = 12) -> list[float]:
        """Return weekly data points for *metric* over the last *weeks*."""
        now = time.time()
        results: list[float] = []
        for week_idx in range(weeks, 0, -1):
            week_end = now - ((week_idx - 1) * _SECONDS_PER_WEEK)
            week_start = week_end - _SECONDS_PER_WEEK
            findings = [f for f in self.collector.findings.values()
                        if week_start <= f.detected_at < week_end]
            events = [e for e in self.collector.events
                      if week_start <= e.timestamp < week_end]
            results.append(round(
                self._compute_metric_for_window(metric, findings, events, week_start, week_end), 4,
            ))
        return results

    # -- private helpers -----------------------------------------------------

    def _compute_metric_for_window(
        self,
        metric: MetricType,
        findings: list[FindingMetrics],
        events: list[TelemetryEvent],
        window_start: float,
        window_end: float,
    ) -> float:
        """Compute a single metric value for one time window."""
        actionable = [f for f in findings if f.lifecycle not in _NON_ACTIONABLE]

        if metric == MetricType.BUGS_PREVENTED:
            return float(len(actionable))

        if metric == MetricType.TIME_SAVED_HOURS:
            return sum(self._hours_saved_per_bug.get(f.severity.lower(), 4.0) for f in actionable)

        if metric == MetricType.COST_PER_FINDING:
            window_days = (window_end - window_start) / _SECONDS_PER_DAY
            tool_cost = self._tool_monthly_cost * (window_days / 30.0)
            return tool_cost / max(len(findings), 1)

        if metric == MetricType.FINDINGS_BY_SEVERITY:
            return float(len(findings))

        if metric == MetricType.FIX_RATE:
            resolved = [f for f in findings if f.lifecycle in _TERMINAL_LIFECYCLES]
            return len(resolved) / max(len(findings), 1)

        if metric == MetricType.MEAN_TIME_TO_FIX_HOURS:
            ttfs = [f.time_to_fix_hours for f in findings if f.time_to_fix_hours is not None]
            return sum(ttfs) / max(len(ttfs), 1)

        if metric == MetricType.VERIFICATION_COVERAGE:
            all_repos = {e.repo_id for e in self.collector.events if e.repo_id}
            window_repos = {e.repo_id for e in events if e.repo_id}
            return len(window_repos) / max(len(all_repos), 1)

        return 0.0

    def _build_trends(self, period_start: float, period_end: float) -> dict[str, list[float]]:
        """Build weekly trend data for key metrics over the report period."""
        total_seconds = period_end - period_start
        num_weeks = max(int(total_seconds / _SECONDS_PER_WEEK), 1)
        key_metrics = [MetricType.BUGS_PREVENTED, MetricType.FIX_RATE, MetricType.MEAN_TIME_TO_FIX_HOURS]

        trends: dict[str, list[float]] = {}
        for metric in key_metrics:
            weekly: list[float] = []
            for week_idx in range(num_weeks):
                w_start = period_start + (week_idx * _SECONDS_PER_WEEK)
                w_end = min(w_start + _SECONDS_PER_WEEK, period_end)
                findings = [f for f in self.collector.findings.values()
                            if w_start <= f.detected_at < w_end]
                events = [e for e in self.collector.events
                          if w_start <= e.timestamp < w_end]
                weekly.append(round(
                    self._compute_metric_for_window(metric, findings, events, w_start, w_end), 4,
                ))
            trends[metric.value] = weekly
        return trends


def _ts_to_date(ts: float) -> str:
    """Convert a UNIX timestamp to a YYYY-MM-DD string."""
    import datetime as _dt

    return _dt.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
