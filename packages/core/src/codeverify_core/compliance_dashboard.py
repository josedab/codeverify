"""Compliance Dashboard & Reporter — rich compliance reporting and audit export.

Generates dashboard views with compliance scoring, evidence visualization,
audit lifecycle management, remediation tracking, and multi-format report
export (Markdown, HTML, CSV, PDF data). Companion to compliance_as_code.py.
"""

from __future__ import annotations

import csv
import io
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from html import escape as html_escape
from typing import Any

import structlog

logger = structlog.get_logger()

# =============================================================================
# Enumerations
# =============================================================================


class ReportFormat(str, Enum):
    """Supported output formats for compliance reports."""

    JSON = "json"
    HTML = "html"
    CSV = "csv"
    MARKDOWN = "markdown"
    PDF_DATA = "pdf_data"


class ComplianceStatus(str, Enum):
    """Assessment status for a compliance control."""

    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"
    EXEMPT = "exempt"


class ControlPriority(str, Enum):
    """Priority classification for compliance controls."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TrendDirection(str, Enum):
    """Direction of compliance score trend over time."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    UNKNOWN = "unknown"


class AuditType(str, Enum):
    """Types of compliance audits."""

    INTERNAL = "internal"
    EXTERNAL = "external"
    SOC2_TYPE_II = "soc2_type_ii"
    ISO_27001 = "iso_27001"
    HIPAA_REVIEW = "hipaa_review"
    PCI_DSS = "pci_dss"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ControlStatus:
    """Status of a single compliance control."""

    control_id: str
    control_name: str
    framework: str
    status: ComplianceStatus
    priority: ControlPriority
    evidence_count: int = 0
    last_assessed: datetime | None = None
    gap_description: str | None = None
    remediation_plan: str | None = None
    owner: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id,
            "control_name": self.control_name,
            "framework": self.framework,
            "status": self.status.value,
            "priority": self.priority.value,
            "evidence_count": self.evidence_count,
            "last_assessed": self.last_assessed.isoformat() if self.last_assessed else None,
            "gap_description": self.gap_description,
            "remediation_plan": self.remediation_plan,
            "owner": self.owner,
        }


@dataclass
class ComplianceScore:
    """Compliance score for a single framework."""

    framework: str
    overall_score: float
    controls_total: int
    controls_compliant: int
    controls_partial: int
    controls_non_compliant: int
    controls_exempt: int
    trend: TrendDirection = TrendDirection.UNKNOWN
    score_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "overall_score": round(self.overall_score, 2),
            "controls_total": self.controls_total,
            "controls_compliant": self.controls_compliant,
            "controls_partial": self.controls_partial,
            "controls_non_compliant": self.controls_non_compliant,
            "controls_exempt": self.controls_exempt,
            "trend": self.trend.value,
            "score_history": self.score_history,
        }


@dataclass
class DashboardWidget:
    """A single widget for the compliance dashboard."""

    id: str
    title: str
    widget_type: str
    data: dict[str, Any]
    position: tuple[int, int] = (0, 0)
    size: tuple[int, int] = (1, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "widget_type": self.widget_type,
            "data": self.data,
            "position": list(self.position),
            "size": list(self.size),
        }


@dataclass
class DashboardView:
    """Complete dashboard view with widgets and scores."""

    title: str
    generated_at: datetime
    widgets: list[DashboardWidget]
    scores: list[ComplianceScore]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "generated_at": self.generated_at.isoformat(),
            "widgets": [w.to_dict() for w in self.widgets],
            "scores": [s.to_dict() for s in self.scores],
            "summary": self.summary,
        }


@dataclass
class AuditRecord:
    """Record of a compliance audit."""

    id: str
    audit_type: AuditType
    framework: str
    auditor: str
    started_at: datetime
    completed_at: datetime | None = None
    findings: list[dict[str, Any]] = field(default_factory=list)
    status: str = "in_progress"
    overall_result: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "audit_type": self.audit_type.value,
            "framework": self.framework,
            "auditor": self.auditor,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "findings": self.findings,
            "status": self.status,
            "overall_result": self.overall_result,
        }


@dataclass
class RemediationItem:
    """A remediation action item for a non-compliant control."""

    id: str
    control_id: str
    framework: str
    description: str
    priority: ControlPriority
    status: str = "open"
    assignee: str | None = None
    due_date: datetime | None = None
    estimated_effort: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "control_id": self.control_id,
            "framework": self.framework,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status,
            "assignee": self.assignee,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "estimated_effort": self.estimated_effort,
        }


@dataclass
class ComplianceReport:
    """A full compliance report for a framework."""

    id: str
    title: str
    framework: str
    generated_at: datetime
    score: ComplianceScore
    controls: list[ControlStatus]
    remediations: list[RemediationItem]
    executive_summary: str
    format: ReportFormat = ReportFormat.JSON

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "framework": self.framework,
            "generated_at": self.generated_at.isoformat(),
            "score": self.score.to_dict(),
            "controls": [c.to_dict() for c in self.controls],
            "remediations": [r.to_dict() for r in self.remediations],
            "executive_summary": self.executive_summary,
            "format": self.format.value,
        }


# =============================================================================
# Compliance Scorer
# =============================================================================

# Weight multipliers for priority-based risk adjustment
_PRIORITY_WEIGHTS: dict[str, float] = {
    ControlPriority.CRITICAL.value: 4.0,
    ControlPriority.HIGH.value: 3.0,
    ControlPriority.MEDIUM.value: 2.0,
    ControlPriority.LOW.value: 1.0,
}


class ComplianceScorer:
    """Calculate compliance scores across frameworks."""

    def __init__(self) -> None:
        self._history: dict[str, list[float]] = {}

    def calculate_score(self, controls: list[ControlStatus], framework: str) -> ComplianceScore:
        """Calculate an overall compliance score for a framework."""
        fw = [c for c in controls if c.framework == framework]
        total = len(fw)
        if total == 0:
            return ComplianceScore(
                framework=framework,
                overall_score=0.0,
                controls_total=0,
                controls_compliant=0,
                controls_partial=0,
                controls_non_compliant=0,
                controls_exempt=0,
            )
        compliant = sum(1 for c in fw if c.status == ComplianceStatus.COMPLIANT)
        partial = sum(1 for c in fw if c.status == ComplianceStatus.PARTIALLY_COMPLIANT)
        non_compliant = sum(1 for c in fw if c.status == ComplianceStatus.NON_COMPLIANT)
        exempt = sum(1 for c in fw if c.status == ComplianceStatus.EXEMPT)
        assessable = total - exempt
        score = ((compliant + partial * 0.5) / assessable * 100.0) if assessable > 0 else 100.0
        history = self._history.get(framework, [])
        trend = self.calculate_trend(score, history)
        history.append(score)
        self._history[framework] = history
        score_history = [{"score": s, "index": i} for i, s in enumerate(history)]
        logger.info("Compliance score calculated", framework=framework, score=round(score, 2))
        return ComplianceScore(
            framework=framework,
            overall_score=score,
            controls_total=total,
            controls_compliant=compliant,
            controls_partial=partial,
            controls_non_compliant=non_compliant,
            controls_exempt=exempt,
            trend=trend,
            score_history=score_history,
        )

    def calculate_trend(self, current_score: float, history: list[float]) -> TrendDirection:
        """Determine trend direction from score history."""
        if len(history) < 2:
            return TrendDirection.UNKNOWN
        recent_avg = sum(history[-3:]) / len(history[-3:])
        delta = current_score - recent_avg
        if delta > 2.0:
            return TrendDirection.IMPROVING
        if delta < -2.0:
            return TrendDirection.DEGRADING
        return TrendDirection.STABLE

    def compare_frameworks(self, scores: list[ComplianceScore]) -> dict[str, Any]:
        """Compare compliance scores across multiple frameworks."""
        if not scores:
            return {"frameworks": [], "average_score": 0.0}
        avg = sum(s.overall_score for s in scores) / len(scores)
        ranked = sorted(scores, key=lambda s: s.overall_score, reverse=True)
        return {
            "frameworks": [
                {
                    "framework": s.framework,
                    "score": round(s.overall_score, 2),
                    "trend": s.trend.value,
                }
                for s in ranked
            ],
            "average_score": round(avg, 2),
            "best": ranked[0].framework,
            "worst": ranked[-1].framework,
        }

    def risk_adjusted_score(self, score: ComplianceScore, controls: list[ControlStatus]) -> float:
        """Calculate a risk-adjusted score weighting critical controls higher."""
        fw = [c for c in controls if c.framework == score.framework]
        if not fw:
            return score.overall_score
        weighted_sum, weight_total = 0.0, 0.0
        for ctrl in fw:
            if ctrl.status == ComplianceStatus.EXEMPT:
                continue
            w = _PRIORITY_WEIGHTS.get(ctrl.priority.value, 1.0)
            weight_total += w
            if ctrl.status == ComplianceStatus.COMPLIANT:
                weighted_sum += w
            elif ctrl.status == ComplianceStatus.PARTIALLY_COMPLIANT:
                weighted_sum += w * 0.5
        return round((weighted_sum / weight_total * 100.0) if weight_total else 0.0, 2)


# =============================================================================
# Dashboard Builder
# =============================================================================


class DashboardBuilder:
    """Build compliance dashboard views with widgets."""

    def __init__(self) -> None:
        self._widget_counter = 0

    def build_dashboard(
        self,
        controls: list[ControlStatus],
        scores: list[ComplianceScore],
        remediations: list[RemediationItem] | None = None,
    ) -> DashboardView:
        """Build a complete dashboard view from controls and scores."""
        widgets: list[DashboardWidget] = []
        for score in scores:
            fw_ctrls = [c for c in controls if c.framework == score.framework]
            widgets += [
                self._score_gauge_widget(score),
                self._controls_breakdown_widget(fw_ctrls, score.framework),
                self._trend_chart_widget(score),
            ]
        if remediations:
            widgets.append(self._remediation_tracker_widget(remediations))
        if len(scores) > 1:
            widgets.append(self._framework_comparison_widget(scores))
        avg = sum(s.overall_score for s in scores) / len(scores) if scores else 0.0
        summary = {
            "total_frameworks": len(scores),
            "total_controls": sum(s.controls_total for s in scores),
            "total_compliant": sum(s.controls_compliant for s in scores),
            "average_score": round(avg, 2),
            "widgets_count": len(widgets),
        }
        logger.info("Dashboard built", frameworks=len(scores), widgets=len(widgets))
        return DashboardView(
            title="Compliance Dashboard",
            generated_at=datetime.now(UTC),
            widgets=widgets,
            scores=scores,
            summary=summary,
        )

    def _next_widget_id(self) -> str:
        self._widget_counter += 1
        return f"widget-{self._widget_counter}"

    def _score_gauge_widget(self, score: ComplianceScore) -> DashboardWidget:
        """Gauge widget showing overall score for a framework."""
        return DashboardWidget(
            id=self._next_widget_id(),
            title=f"{score.framework} Compliance Score",
            widget_type="gauge",
            data={
                "score": round(score.overall_score, 2),
                "max": 100.0,
                "thresholds": {"red": 50, "yellow": 80, "green": 90},
                "trend": score.trend.value,
            },
            position=(0, 0),
            size=(1, 1),
        )

    def _controls_breakdown_widget(
        self,
        controls: list[ControlStatus],
        framework: str,
    ) -> DashboardWidget:
        """Breakdown of control statuses for a framework."""
        counts: dict[str, int] = {}
        for c in controls:
            counts[c.status.value] = counts.get(c.status.value, 0) + 1
        return DashboardWidget(
            id=self._next_widget_id(),
            title=f"{framework} Controls Breakdown",
            widget_type="pie_chart",
            data={"framework": framework, "breakdown": counts, "total": len(controls)},
            position=(0, 1),
            size=(1, 1),
        )

    def _trend_chart_widget(self, score: ComplianceScore) -> DashboardWidget:
        """Trend chart showing score history."""
        return DashboardWidget(
            id=self._next_widget_id(),
            title=f"{score.framework} Score Trend",
            widget_type="line_chart",
            data={
                "framework": score.framework,
                "current_score": round(score.overall_score, 2),
                "trend": score.trend.value,
                "history": score.score_history,
            },
            position=(1, 0),
            size=(1, 2),
        )

    def _remediation_tracker_widget(self, remediations: list[RemediationItem]) -> DashboardWidget:
        """Widget tracking open remediation items."""
        by_pri: dict[str, int] = {}
        by_st: dict[str, int] = {}
        for r in remediations:
            by_pri[r.priority.value] = by_pri.get(r.priority.value, 0) + 1
            by_st[r.status] = by_st.get(r.status, 0) + 1
        return DashboardWidget(
            id=self._next_widget_id(),
            title="Remediation Tracker",
            widget_type="table",
            data={
                "total": len(remediations),
                "by_priority": by_pri,
                "by_status": by_st,
                "items": [r.to_dict() for r in remediations[:10]],
            },
            position=(2, 0),
            size=(1, 2),
        )

    def _framework_comparison_widget(self, scores: list[ComplianceScore]) -> DashboardWidget:
        """Comparison widget across frameworks."""
        return DashboardWidget(
            id=self._next_widget_id(),
            title="Framework Comparison",
            widget_type="bar_chart",
            data={
                "frameworks": [
                    {"name": s.framework, "score": round(s.overall_score, 2)} for s in scores
                ]
            },
            position=(3, 0),
            size=(1, 2),
        )


# =============================================================================
# Report Generator
# =============================================================================

_STATUS_BADGES: dict[str, str] = {
    ComplianceStatus.COMPLIANT.value: "✅",
    ComplianceStatus.PARTIALLY_COMPLIANT.value: "⚠️",
    ComplianceStatus.NON_COMPLIANT.value: "❌",
    ComplianceStatus.NOT_ASSESSED.value: "⬜",
    ComplianceStatus.EXEMPT.value: "🔘",
}
_PRIORITY_BADGES: dict[str, str] = {
    ControlPriority.CRITICAL.value: "🔴",
    ControlPriority.HIGH.value: "🟠",
    ControlPriority.MEDIUM.value: "🟡",
    ControlPriority.LOW.value: "🟢",
}


class ReportGenerator:
    """Generate compliance reports in multiple formats."""

    def __init__(self) -> None:
        self._scorer = ComplianceScorer()

    def generate_report(
        self,
        controls: list[ControlStatus],
        score: ComplianceScore,
        remediations: list[RemediationItem],
        framework: str,
        format: ReportFormat = ReportFormat.MARKDOWN,
    ) -> ComplianceReport:
        """Generate a compliance report for a framework."""
        fw_controls = [c for c in controls if c.framework == framework]
        fw_remediations = [r for r in remediations if r.framework == framework]
        report = ComplianceReport(
            id=str(uuid.uuid4()),
            title=f"{framework} Compliance Report",
            framework=framework,
            generated_at=datetime.now(UTC),
            score=score,
            controls=fw_controls,
            remediations=fw_remediations,
            executive_summary=self._executive_summary(score, fw_controls),
            format=format,
        )
        logger.info("Report generated", framework=framework, format=format.value)
        return report

    def to_markdown(self, report: ComplianceReport) -> str:
        """Render a compliance report as Markdown."""
        s = report.score
        ts = report.generated_at.strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            f"# {report.title}",
            "",
            f"**Generated:** {ts}  ",
            f"**Framework:** {report.framework}  ",
            f"**Report ID:** `{report.id}`",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            report.executive_summary,
            "",
            "## Compliance Score",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Overall Score | **{s.overall_score:.1f}%** |",
            f"| Trend | {s.trend.value.title()} |",
            f"| Total Controls | {s.controls_total} |",
            f"| Compliant | {s.controls_compliant} |",
            f"| Partially Compliant | {s.controls_partial} |",
            f"| Non-Compliant | {s.controls_non_compliant} |",
            f"| Exempt | {s.controls_exempt} |",
            "",
            "## Control Details",
            "",
            "| Status | Priority | Control ID | Name | Evidence | Owner |",
            "|--------|----------|------------|------|----------|-------|",
        ]
        for c in report.controls:
            b, p = (
                _STATUS_BADGES.get(c.status.value, ""),
                _PRIORITY_BADGES.get(c.priority.value, ""),
            )
            lines.append(
                f"| {b} {c.status.value} | {p} {c.priority.value} | `{c.control_id}` "
                f"| {c.control_name} | {c.evidence_count} | {c.owner or '—'} |"
            )
        if report.remediations:
            lines += [
                "",
                "## Remediation Plan",
                "",
                "| Priority | Control | Description | Status | Assignee | Due Date |",
                "|----------|---------|-------------|--------|----------|----------|",
            ]
            for r in report.remediations:
                p = _PRIORITY_BADGES.get(r.priority.value, "")
                due = r.due_date.strftime("%Y-%m-%d") if r.due_date else "—"
                lines.append(
                    f"| {p} {r.priority.value} | `{r.control_id}` | {r.description} "
                    f"| {r.status} | {r.assignee or '—'} | {due} |"
                )
        lines += ["", "---", "*Report generated by CodeVerify Compliance Dashboard*"]
        return "\n".join(lines)

    def to_html(self, report: ComplianceReport) -> str:
        """Render a compliance report as HTML with inline CSS."""
        s = report.score
        color = (
            "#22c55e"
            if s.overall_score >= 90
            else ("#eab308" if s.overall_score >= 70 else "#ef4444")
        )

        def _ctrl_row(c: ControlStatus) -> str:
            b = _STATUS_BADGES.get(c.status.value, "")
            return (
                f"<tr><td>{b} {html_escape(c.status.value)}</td><td>{html_escape(c.priority.value)}</td>"
                f"<td><code>{html_escape(c.control_id)}</code></td><td>{html_escape(c.control_name)}</td>"
                f"<td>{c.evidence_count}</td><td>{html_escape(c.owner or '—')}</td></tr>"
            )

        ctrl_rows = "\n".join(_ctrl_row(c) for c in report.controls)
        rem_html = ""
        if report.remediations:

            def _rem_row(r: RemediationItem) -> str:
                due = r.due_date.strftime("%Y-%m-%d") if r.due_date else "—"
                return (
                    f"<tr><td>{html_escape(r.priority.value)}</td><td><code>{html_escape(r.control_id)}</code></td>"
                    f"<td>{html_escape(r.description)}</td><td>{html_escape(r.status)}</td>"
                    f"<td>{html_escape(r.assignee or '—')}</td><td>{due}</td></tr>"
                )

            rem_html = (
                "<h2>Remediation Plan</h2><table><thead><tr><th>Priority</th><th>Control</th>"
                "<th>Description</th><th>Status</th><th>Assignee</th><th>Due Date</th>"
                "</tr></thead><tbody>"
                + "\n".join(_rem_row(r) for r in report.remediations)
                + "</tbody></table>"
            )
        css = (
            "body{font-family:system-ui,sans-serif;max-width:960px;margin:0 auto;padding:20px;color:#1e293b;}"
            "h1{border-bottom:2px solid #3b82f6;padding-bottom:8px;}"
            "table{border-collapse:collapse;width:100%;margin:16px 0;}"
            "th,td{border:1px solid #e2e8f0;padding:8px 12px;text-align:left;}"
            "th{background:#f1f5f9;font-weight:600;}tr:nth-child(even){background:#f8fafc;}"
            f".score-box{{display:inline-block;font-size:2em;font-weight:700;color:{color};"
            f"border:3px solid {color};border-radius:12px;padding:16px 24px;margin:8px 0;}}"
            ".meta{color:#64748b;font-size:0.9em;}"
            "code{background:#f1f5f9;padding:2px 6px;border-radius:4px;}"
        )
        ts = report.generated_at.strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{html_escape(report.title)}</title><style>{css}</style></head><body>"
            f"<h1>{html_escape(report.title)}</h1>"
            f"<p class='meta'>Generated: {ts} | Framework: {html_escape(report.framework)}"
            f" | Report ID: <code>{report.id}</code></p>"
            f"<h2>Executive Summary</h2><p>{html_escape(report.executive_summary)}</p>"
            f"<h2>Compliance Score</h2><div class='score-box'>{s.overall_score:.1f}%</div>"
            f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>"
            f"<tr><td>Trend</td><td>{s.trend.value.title()}</td></tr>"
            f"<tr><td>Total Controls</td><td>{s.controls_total}</td></tr>"
            f"<tr><td>Compliant</td><td>{s.controls_compliant}</td></tr>"
            f"<tr><td>Partially Compliant</td><td>{s.controls_partial}</td></tr>"
            f"<tr><td>Non-Compliant</td><td>{s.controls_non_compliant}</td></tr>"
            f"<tr><td>Exempt</td><td>{s.controls_exempt}</td></tr></tbody></table>"
            f"<h2>Control Details</h2><table><thead><tr><th>Status</th><th>Priority</th>"
            f"<th>Control ID</th><th>Name</th><th>Evidence</th><th>Owner</th>"
            f"</tr></thead><tbody>{ctrl_rows}</tbody></table>"
            f"{rem_html}<hr><p class='meta'>Report generated by CodeVerify "
            f"Compliance Dashboard</p></body></html>"
        )

    def to_csv(self, report: ComplianceReport) -> str:
        """Render control data as CSV with proper escaping."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "control_id",
                "control_name",
                "framework",
                "status",
                "priority",
                "evidence_count",
                "last_assessed",
                "gap_description",
                "owner",
            ]
        )
        for c in report.controls:
            writer.writerow(
                [
                    c.control_id,
                    c.control_name,
                    c.framework,
                    c.status.value,
                    c.priority.value,
                    c.evidence_count,
                    c.last_assessed.isoformat() if c.last_assessed else "",
                    c.gap_description or "",
                    c.owner or "",
                ]
            )
        return buf.getvalue()

    def to_pdf_data(self, report: ComplianceReport) -> dict[str, Any]:
        """Return structured data suitable for PDF rendering."""
        return {
            "metadata": {
                "title": report.title,
                "framework": report.framework,
                "generated_at": report.generated_at.isoformat(),
                "report_id": report.id,
            },
            "executive_summary": report.executive_summary,
            "score": report.score.to_dict(),
            "controls": [c.to_dict() for c in report.controls],
            "remediations": [r.to_dict() for r in report.remediations],
            "page_settings": {"orientation": "portrait", "margin_mm": 20, "font": "Helvetica"},
        }

    def _executive_summary(self, score: ComplianceScore, controls: list[ControlStatus]) -> str:
        """Generate an executive summary from score and controls."""
        if score.controls_total == 0:
            return f"No controls have been assessed for {score.framework}."
        pct, total = score.overall_score, score.controls_total
        critical_gaps = [
            c
            for c in controls
            if c.status == ComplianceStatus.NON_COMPLIANT
            and c.priority in (ControlPriority.CRITICAL, ControlPriority.HIGH)
        ]
        level = "strong" if pct >= 90 else ("moderate" if pct >= 70 else "insufficient")
        trend_map = {
            TrendDirection.IMPROVING: " The trend is positive with scores improving.",
            TrendDirection.STABLE: " The compliance posture is stable.",
            TrendDirection.DEGRADING: " Attention required — scores are declining.",
        }
        summary = (
            f"The {score.framework} compliance posture is {level} at {pct:.1f}%. "
            f"Of {total} controls assessed, {score.controls_compliant} are fully compliant, "
            f"{score.controls_partial} are partially compliant, and "
            f"{score.controls_non_compliant} are non-compliant."
        )
        if critical_gaps:
            ids = ", ".join(c.control_id for c in critical_gaps[:5])
            summary += (
                f" {len(critical_gaps)} high-priority gap(s) require immediate attention: {ids}."
            )
        summary += trend_map.get(score.trend, "")
        return summary


# =============================================================================
# Audit Manager
# =============================================================================


class AuditManager:
    """Manage audit lifecycle and findings."""

    def __init__(self) -> None:
        self._audits: dict[str, AuditRecord] = {}

    def start_audit(self, audit_type: AuditType, framework: str, auditor: str) -> AuditRecord:
        """Start a new audit."""
        record = AuditRecord(
            id=str(uuid.uuid4()),
            audit_type=audit_type,
            framework=framework,
            auditor=auditor,
            started_at=datetime.now(UTC),
        )
        self._audits[record.id] = record
        logger.info("Audit started", audit_id=record.id, framework=framework, auditor=auditor)
        return record

    def add_finding(
        self,
        audit_id: str,
        control_id: str,
        finding_type: str,
        description: str,
    ) -> dict[str, Any]:
        """Record a finding within an audit."""
        audit = self._audits.get(audit_id)
        if audit is None:
            raise ValueError(f"Audit {audit_id} not found")
        if audit.status != "in_progress":
            raise ValueError(f"Audit {audit_id} is not in progress")
        finding = {
            "id": str(uuid.uuid4()),
            "control_id": control_id,
            "finding_type": finding_type,
            "description": description,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        audit.findings.append(finding)
        logger.info("Audit finding added", audit_id=audit_id, control_id=control_id)
        return finding

    def complete_audit(self, audit_id: str, result: str) -> AuditRecord:
        """Complete an audit with an overall result."""
        audit = self._audits.get(audit_id)
        if audit is None:
            raise ValueError(f"Audit {audit_id} not found")
        audit.completed_at, audit.status, audit.overall_result = (
            datetime.now(UTC),
            "completed",
            result,
        )
        logger.info("Audit completed", audit_id=audit_id, result=result)
        return audit

    def get_audit_history(self, framework: str | None = None) -> list[AuditRecord]:
        """Return audit history, optionally filtered by framework."""
        audits = list(self._audits.values())
        if framework:
            audits = [a for a in audits if a.framework == framework]
        return sorted(audits, key=lambda a: a.started_at, reverse=True)


# =============================================================================
# Remediation Tracker
# =============================================================================


class RemediationTracker:
    """Track and manage remediation items."""

    def __init__(self) -> None:
        self._items: dict[str, RemediationItem] = {}

    def create_item(
        self,
        control_id: str,
        framework: str,
        description: str,
        priority: ControlPriority,
    ) -> RemediationItem:
        """Create a new remediation item."""
        item = RemediationItem(
            id=str(uuid.uuid4()),
            control_id=control_id,
            framework=framework,
            description=description,
            priority=priority,
        )
        self._items[item.id] = item
        logger.info("Remediation created", item_id=item.id, control_id=control_id)
        return item

    def assign(self, item_id: str, assignee: str) -> RemediationItem:
        """Assign a remediation item to an owner."""
        item = self._items.get(item_id)
        if item is None:
            raise ValueError(f"Remediation item {item_id} not found")
        item.assignee, item.status = assignee, "assigned"
        logger.info("Remediation assigned", item_id=item_id, assignee=assignee)
        return item

    def update_status(self, item_id: str, status: str) -> RemediationItem:
        """Update the status of a remediation item."""
        item = self._items.get(item_id)
        if item is None:
            raise ValueError(f"Remediation item {item_id} not found")
        item.status = status
        logger.info("Remediation status updated", item_id=item_id, status=status)
        return item

    def get_overdue(self) -> list[RemediationItem]:
        """Return remediation items past their due date."""
        now = datetime.now(UTC)
        return [
            item
            for item in self._items.values()
            if item.due_date and item.due_date < now and item.status not in ("done", "closed")
        ]

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of all remediation items."""
        by_status: dict[str, int] = {}
        by_priority: dict[str, int] = {}
        for item in self._items.values():
            by_status[item.status] = by_status.get(item.status, 0) + 1
            by_priority[item.priority.value] = by_priority.get(item.priority.value, 0) + 1
        return {
            "total": len(self._items),
            "by_status": by_status,
            "by_priority": by_priority,
            "overdue_count": len(self.get_overdue()),
        }

    def all_items(self) -> list[RemediationItem]:
        """Return all remediation items."""
        return list(self._items.values())


# =============================================================================
# Compliance Dashboard — Main Orchestrator
# =============================================================================


class ComplianceDashboard:
    """Main orchestrator for compliance dashboard operations."""

    def __init__(self) -> None:
        self._controls: list[ControlStatus] = []
        self._scorer = ComplianceScorer()
        self._dashboard_builder = DashboardBuilder()
        self._report_generator = ReportGenerator()
        self._audit_manager = AuditManager()
        self._remediation_tracker = RemediationTracker()

    def get_dashboard(self, frameworks: list[str] | None = None) -> DashboardView:
        """Build and return a dashboard view."""
        fw_list = frameworks or list({c.framework for c in self._controls})
        scores = [self._scorer.calculate_score(self._controls, fw) for fw in fw_list]
        remediations = self._remediation_tracker.all_items()
        return self._dashboard_builder.build_dashboard(self._controls, scores, remediations)

    def add_control(
        self,
        control_id: str,
        control_name: str,
        framework: str,
        status: ComplianceStatus,
        priority: ControlPriority = ControlPriority.MEDIUM,
        **kwargs: Any,
    ) -> ControlStatus:
        """Register a control status entry."""
        ctrl = ControlStatus(
            control_id=control_id,
            control_name=control_name,
            framework=framework,
            status=status,
            priority=priority,
            evidence_count=kwargs.get("evidence_count", 0),
            last_assessed=kwargs.get("last_assessed"),
            gap_description=kwargs.get("gap_description"),
            remediation_plan=kwargs.get("remediation_plan"),
            owner=kwargs.get("owner"),
        )
        self._controls.append(ctrl)
        logger.info("Control added", control_id=control_id, framework=framework)
        return ctrl

    def generate_report(
        self,
        framework: str,
        format: ReportFormat = ReportFormat.MARKDOWN,
    ) -> ComplianceReport:
        """Generate a compliance report for a framework."""
        score = self._scorer.calculate_score(self._controls, framework)
        remediations = self._remediation_tracker.all_items()
        return self._report_generator.generate_report(
            self._controls,
            score,
            remediations,
            framework,
            format,
        )

    def start_audit(self, audit_type: AuditType, framework: str, auditor: str) -> AuditRecord:
        """Start a new compliance audit."""
        return self._audit_manager.start_audit(audit_type, framework, auditor)

    def get_remediation_summary(self) -> dict[str, Any]:
        """Return a summary of remediation items."""
        return self._remediation_tracker.get_summary()

    def export_evidence(self, framework: str) -> dict[str, Any]:
        """Export evidence summary for a framework."""
        fw_controls = [c for c in self._controls if c.framework == framework]
        return {
            "framework": framework,
            "exported_at": datetime.now(UTC).isoformat(),
            "controls_count": len(fw_controls),
            "total_evidence_items": sum(c.evidence_count for c in fw_controls),
            "controls": [c.to_dict() for c in fw_controls],
        }
