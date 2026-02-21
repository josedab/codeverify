"""Compliance Report Generator.

Generates HTML, text, and JSON compliance reports from verification results
with framework-specific control mappings, evidence linking, and audit trails.

Supported frameworks: SOC 2, HIPAA, PCI-DSS, GDPR, ISO 27001.

.. deprecated::
    This module is superseded by ``codeverify_core.compliance_engine``.
    It remains importable for backward compatibility but will be
    removed in a future release.
"""

from __future__ import annotations

import warnings as _warnings
_warnings.warn(
    "codeverify_core.compliance_reports is deprecated. Use codeverify_core.compliance_engine instead.",
    DeprecationWarning,
    stacklevel=2,
)


import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# =============================================================================
# Compliance Framework Control Definitions
# =============================================================================


class ComplianceFramework(str, Enum):
    SOC2 = "soc2"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    GDPR = "gdpr"
    ISO_27001 = "iso_27001"


class ControlStatus(str, Enum):
    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"


@dataclass
class ControlMapping:
    """Maps a verification finding category to a compliance control."""

    framework: ComplianceFramework
    control_id: str
    control_name: str
    description: str
    finding_categories: list[str]  # Categories of findings that map to this control


@dataclass
class ControlAssessment:
    """Assessment of a single control."""

    control_id: str
    control_name: str
    framework: str
    status: ControlStatus
    evidence_count: int = 0
    findings_count: int = 0
    details: str = ""


@dataclass
class ComplianceReport:
    """A generated compliance report."""

    organization: str
    framework: ComplianceFramework
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    period_start: str = ""
    period_end: str = ""
    assessments: list[ControlAssessment] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    audit_entries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def compliance_score(self) -> float:
        if not self.assessments:
            return 0.0
        compliant = sum(1 for a in self.assessments if a.status == ControlStatus.COMPLIANT)
        assessed = sum(1 for a in self.assessments if a.status != ControlStatus.NOT_ASSESSED)
        return compliant / assessed if assessed > 0 else 0.0

    @property
    def total_controls(self) -> int:
        return len(self.assessments)

    @property
    def compliant_controls(self) -> int:
        return sum(1 for a in self.assessments if a.status == ControlStatus.COMPLIANT)

    @property
    def non_compliant_controls(self) -> int:
        return sum(1 for a in self.assessments if a.status == ControlStatus.NON_COMPLIANT)


# =============================================================================
# Framework Control Mappings
# =============================================================================

# Mapping from finding categories → compliance controls
FRAMEWORK_CONTROLS: dict[ComplianceFramework, list[ControlMapping]] = {
    ComplianceFramework.SOC2: [
        ControlMapping(
            ComplianceFramework.SOC2,
            "CC6.1",
            "Logical and Physical Access Controls",
            "Access control mechanisms prevent unauthorized access.",
            ["null_safety", "security", "authentication"],
        ),
        ControlMapping(
            ComplianceFramework.SOC2,
            "CC6.6",
            "System Boundary Security",
            "Input validation and output encoding at system boundaries.",
            ["security", "injection", "xss", "input_validation"],
        ),
        ControlMapping(
            ComplianceFramework.SOC2,
            "CC7.1",
            "Vulnerability Management",
            "System vulnerabilities are identified and remediated.",
            ["security", "dependency", "supply_chain"],
        ),
        ControlMapping(
            ComplianceFramework.SOC2,
            "CC7.2",
            "Security Monitoring",
            "Anomalies and security events are monitored.",
            ["error_handling", "logging"],
        ),
        ControlMapping(
            ComplianceFramework.SOC2,
            "CC8.1",
            "Change Management",
            "Changes are authorized, tested, and approved.",
            ["type_safety", "formal_verification"],
        ),
    ],
    ComplianceFramework.HIPAA: [
        ControlMapping(
            ComplianceFramework.HIPAA,
            "164.312(a)(1)",
            "Access Control",
            "Implement technical policies for electronic PHI access.",
            ["authentication", "null_safety", "security"],
        ),
        ControlMapping(
            ComplianceFramework.HIPAA,
            "164.312(a)(2)(iv)",
            "Encryption",
            "Encrypt/decrypt electronic PHI.",
            ["security", "encryption"],
        ),
        ControlMapping(
            ComplianceFramework.HIPAA,
            "164.312(b)",
            "Audit Controls",
            "Record and examine information system activity.",
            ["logging", "error_handling"],
        ),
        ControlMapping(
            ComplianceFramework.HIPAA,
            "164.312(c)(1)",
            "Integrity Controls",
            "Protect electronic PHI from improper alteration.",
            ["type_safety", "formal_verification", "input_validation"],
        ),
        ControlMapping(
            ComplianceFramework.HIPAA,
            "164.312(e)(1)",
            "Transmission Security",
            "Guard against unauthorized access during transmission.",
            ["security", "encryption", "network"],
        ),
    ],
    ComplianceFramework.PCI_DSS: [
        ControlMapping(
            ComplianceFramework.PCI_DSS,
            "6.5.1",
            "Injection Flaws",
            "Address common coding vulnerabilities - injection.",
            ["injection", "sql_injection", "command_injection"],
        ),
        ControlMapping(
            ComplianceFramework.PCI_DSS,
            "6.5.2",
            "Buffer Overflows",
            "Address common coding vulnerabilities - overflows.",
            ["integer_overflow", "array_bounds", "memory_safety"],
        ),
        ControlMapping(
            ComplianceFramework.PCI_DSS,
            "6.5.5",
            "Improper Error Handling",
            "Address common coding vulnerabilities - error handling.",
            ["error_handling", "null_safety"],
        ),
        ControlMapping(
            ComplianceFramework.PCI_DSS,
            "6.5.7",
            "XSS",
            "Address cross-site scripting vulnerabilities.",
            ["xss", "security", "input_validation"],
        ),
        ControlMapping(
            ComplianceFramework.PCI_DSS,
            "6.5.10",
            "Authentication",
            "Broken authentication and session management.",
            ["authentication", "security"],
        ),
    ],
    ComplianceFramework.GDPR: [
        ControlMapping(
            ComplianceFramework.GDPR,
            "Art.25",
            "Data Protection by Design",
            "Implement appropriate technical measures.",
            ["security", "type_safety", "formal_verification"],
        ),
        ControlMapping(
            ComplianceFramework.GDPR,
            "Art.32",
            "Security of Processing",
            "Implement appropriate technical and organizational measures.",
            ["security", "encryption", "authentication"],
        ),
        ControlMapping(
            ComplianceFramework.GDPR,
            "Art.33",
            "Breach Notification",
            "Notify supervisory authority of breaches.",
            ["logging", "error_handling", "monitoring"],
        ),
    ],
    ComplianceFramework.ISO_27001: [
        ControlMapping(
            ComplianceFramework.ISO_27001,
            "A.12.6",
            "Vulnerability Management",
            "Technical vulnerability management.",
            ["security", "dependency", "supply_chain"],
        ),
        ControlMapping(
            ComplianceFramework.ISO_27001,
            "A.14.2",
            "Secure Development",
            "Security in development and support processes.",
            ["security", "type_safety", "formal_verification", "null_safety"],
        ),
        ControlMapping(
            ComplianceFramework.ISO_27001,
            "A.12.4",
            "Logging and Monitoring",
            "Logging and monitoring controls.",
            ["logging", "error_handling"],
        ),
    ],
}


# =============================================================================
# Report Generator
# =============================================================================


class ComplianceReportGenerator:
    """Generates compliance reports from verification findings.

    Usage:
        gen = ComplianceReportGenerator()
        report = gen.generate(
            framework=ComplianceFramework.SOC2,
            organization="TestCorp",
            findings=[{"category": "security", "severity": "high", ...}],
            audit_entries=[...],
        )
        html = gen.to_html(report)
        text = gen.to_text(report)
    """

    def generate(
        self,
        framework: ComplianceFramework,
        organization: str,
        findings: list[dict[str, Any]],
        audit_entries: list[dict[str, Any]] | None = None,
        period_start: str = "",
        period_end: str = "",
    ) -> ComplianceReport:
        """Generate a compliance report from verification findings."""
        controls = FRAMEWORK_CONTROLS.get(framework, [])
        finding_categories = {f.get("category", "") for f in findings}

        assessments: list[ControlAssessment] = []
        for control in controls:
            matching_categories = set(control.finding_categories) & finding_categories
            matching_findings = [
                f for f in findings if f.get("category", "") in control.finding_categories
            ]

            # Assess control status
            critical_findings = [
                f for f in matching_findings if f.get("severity") in ("critical", "high")
            ]
            if matching_categories and not critical_findings:
                status = ControlStatus.COMPLIANT
            elif critical_findings:
                status = ControlStatus.NON_COMPLIANT
            elif matching_categories:
                status = ControlStatus.PARTIAL
            else:
                status = ControlStatus.NOT_ASSESSED

            assessments.append(
                ControlAssessment(
                    control_id=control.control_id,
                    control_name=control.control_name,
                    framework=framework.value,
                    status=status,
                    evidence_count=len(matching_findings),
                    findings_count=len(critical_findings),
                    details=control.description,
                )
            )

        report = ComplianceReport(
            organization=organization,
            framework=framework,
            period_start=period_start,
            period_end=period_end,
            assessments=assessments,
            audit_entries=audit_entries or [],
        )

        report.summary = {
            "total_controls": report.total_controls,
            "compliant": report.compliant_controls,
            "non_compliant": report.non_compliant_controls,
            "partial": sum(1 for a in assessments if a.status == ControlStatus.PARTIAL),
            "not_assessed": sum(1 for a in assessments if a.status == ControlStatus.NOT_ASSESSED),
            "compliance_score": round(report.compliance_score * 100, 1),
            "total_findings": len(findings),
        }

        return report

    def to_text(self, report: ComplianceReport) -> str:
        """Render the report as plain text."""
        lines: list[str] = []
        lines.append("=" * 70)
        lines.append(f"COMPLIANCE REPORT: {report.framework.value.upper()}")
        lines.append(f"Organization: {report.organization}")
        lines.append(f"Generated: {report.generated_at}")
        if report.period_start:
            lines.append(f"Period: {report.period_start} — {report.period_end}")
        lines.append("=" * 70)
        lines.append("")

        # Summary
        s = report.summary
        lines.append(f"COMPLIANCE SCORE: {s.get('compliance_score', 0)}%")
        lines.append(f"  Controls Assessed: {s.get('total_controls', 0)}")
        lines.append(f"  Compliant: {s.get('compliant', 0)}")
        lines.append(f"  Non-Compliant: {s.get('non_compliant', 0)}")
        lines.append(f"  Partial: {s.get('partial', 0)}")
        lines.append(f"  Findings: {s.get('total_findings', 0)}")
        lines.append("")

        # Control Details
        lines.append("-" * 70)
        lines.append("CONTROL ASSESSMENTS")
        lines.append("-" * 70)
        for a in report.assessments:
            status_icon = {
                ControlStatus.COMPLIANT: "✓",
                ControlStatus.PARTIAL: "~",
                ControlStatus.NON_COMPLIANT: "✗",
                ControlStatus.NOT_ASSESSED: "-",
            }.get(a.status, "?")
            lines.append(f"  [{status_icon}] {a.control_id}: {a.control_name} ({a.status.value})")
            if a.findings_count > 0:
                lines.append(f"      {a.findings_count} critical/high finding(s)")
        lines.append("")

        # Audit Trail
        if report.audit_entries:
            lines.append("-" * 70)
            lines.append(f"AUDIT TRAIL ({len(report.audit_entries)} entries)")
            lines.append("-" * 70)
            for entry in report.audit_entries[:10]:
                lines.append(
                    f"  {entry.get('timestamp', '')} | "
                    f"{entry.get('action', '')} | "
                    f"{entry.get('actor', 'system')}"
                )

        lines.append("")
        lines.append("=" * 70)
        lines.append("END OF REPORT")
        return "\n".join(lines)

    def to_html(self, report: ComplianceReport) -> str:
        """Render the report as HTML."""
        s = report.summary
        score = s.get("compliance_score", 0)
        score_color = "#22c55e" if score >= 80 else "#f59e0b" if score >= 60 else "#ef4444"

        rows = ""
        for a in report.assessments:
            status_class = {
                ControlStatus.COMPLIANT: "compliant",
                ControlStatus.PARTIAL: "partial",
                ControlStatus.NON_COMPLIANT: "non-compliant",
                ControlStatus.NOT_ASSESSED: "not-assessed",
            }.get(a.status, "")
            rows += f"""
            <tr class="{status_class}">
                <td>{a.control_id}</td>
                <td>{a.control_name}</td>
                <td><span class="status-badge {status_class}">{a.status.value}</span></td>
                <td>{a.evidence_count}</td>
                <td>{a.findings_count}</td>
            </tr>"""

        audit_rows = ""
        for entry in report.audit_entries[:20]:
            audit_rows += f"""
            <tr>
                <td>{entry.get("timestamp", "")[:19]}</td>
                <td>{entry.get("action", "")}</td>
                <td>{entry.get("actor", "system")}</td>
                <td>{entry.get("outcome", "success")}</td>
            </tr>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Compliance Report — {report.framework.value.upper()}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; color: #1f2937; }}
  h1 {{ color: #111827; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }}
  .score {{ font-size: 48px; font-weight: bold; color: {score_color}; }}
  .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }}
  .summary-card {{ background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; text-align: center; }}
  .summary-card .value {{ font-size: 24px; font-weight: bold; }}
  .summary-card .label {{ color: #6b7280; font-size: 14px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
  th {{ background: #f3f4f6; text-align: left; padding: 12px; border-bottom: 2px solid #e5e7eb; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; }}
  .status-badge {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }}
  .compliant .status-badge {{ background: #dcfce7; color: #166534; }}
  .partial .status-badge {{ background: #fef9c3; color: #854d0e; }}
  .non-compliant .status-badge {{ background: #fee2e2; color: #991b1b; }}
  .not-assessed .status-badge {{ background: #f3f4f6; color: #6b7280; }}
  .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #e5e7eb; color: #9ca3af; font-size: 12px; }}
</style>
</head>
<body>
  <h1>Compliance Report: {report.framework.value.upper()}</h1>
  <p><strong>Organization:</strong> {report.organization} | <strong>Generated:</strong> {report.generated_at[:19]}</p>

  <div class="score">{score}%</div>
  <p>Compliance Score</p>

  <div class="summary">
    <div class="summary-card"><div class="value">{s.get("total_controls", 0)}</div><div class="label">Total Controls</div></div>
    <div class="summary-card"><div class="value" style="color:#22c55e">{s.get("compliant", 0)}</div><div class="label">Compliant</div></div>
    <div class="summary-card"><div class="value" style="color:#ef4444">{s.get("non_compliant", 0)}</div><div class="label">Non-Compliant</div></div>
    <div class="summary-card"><div class="value">{s.get("total_findings", 0)}</div><div class="label">Total Findings</div></div>
  </div>

  <h2>Control Assessments</h2>
  <table>
    <thead><tr><th>Control ID</th><th>Control Name</th><th>Status</th><th>Evidence</th><th>Findings</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>

  {"<h2>Audit Trail</h2><table><thead><tr><th>Timestamp</th><th>Action</th><th>Actor</th><th>Outcome</th></tr></thead><tbody>" + audit_rows + "</tbody></table>" if audit_rows else ""}

  <div class="footer">
    <p>Generated by CodeVerify Compliance Engine | {report.generated_at}</p>
    <p>This report is auto-generated from verification results. Consult your compliance officer for official attestations.</p>
  </div>
</body>
</html>"""
        return html

    def to_json(self, report: ComplianceReport) -> str:
        """Serialize the report to JSON."""
        return json.dumps(
            {
                "organization": report.organization,
                "framework": report.framework.value,
                "generated_at": report.generated_at,
                "period_start": report.period_start,
                "period_end": report.period_end,
                "summary": report.summary,
                "assessments": [
                    {
                        "control_id": a.control_id,
                        "control_name": a.control_name,
                        "status": a.status.value,
                        "evidence_count": a.evidence_count,
                        "findings_count": a.findings_count,
                        "details": a.details,
                    }
                    for a in report.assessments
                ],
                "audit_entries": report.audit_entries,
            },
            indent=2,
        )
