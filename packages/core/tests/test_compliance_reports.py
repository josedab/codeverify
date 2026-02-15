"""Tests for compliance_reports module."""

from __future__ import annotations

import json

import pytest

from codeverify_core.compliance_reports import (
    FRAMEWORK_CONTROLS,
    ComplianceFramework,
    ComplianceReport,
    ComplianceReportGenerator,
    ControlAssessment,
    ControlStatus,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def generator():
    return ComplianceReportGenerator()


@pytest.fixture
def sample_findings():
    return [
        {
            "category": "security",
            "severity": "high",
            "message": "SQL injection risk",
            "file": "app.py",
        },
        {
            "category": "null_safety",
            "severity": "medium",
            "message": "Unchecked None",
            "file": "utils.py",
        },
        {
            "category": "error_handling",
            "severity": "low",
            "message": "Broad except",
            "file": "handler.py",
        },
        {
            "category": "type_safety",
            "severity": "medium",
            "message": "Missing annotation",
            "file": "models.py",
        },
        {"category": "logging", "severity": "low", "message": "Missing log", "file": "api.py"},
    ]


@pytest.fixture
def sample_audit_entries():
    return [
        {
            "timestamp": "2025-01-15T10:00:00Z",
            "action": "scan_started",
            "actor": "ci-bot",
            "outcome": "success",
        },
        {
            "timestamp": "2025-01-15T10:05:00Z",
            "action": "scan_completed",
            "actor": "ci-bot",
            "outcome": "success",
        },
        {
            "timestamp": "2025-01-15T10:06:00Z",
            "action": "report_generated",
            "actor": "system",
            "outcome": "success",
        },
    ]


@pytest.fixture
def critical_findings():
    """Findings that should cause non-compliance."""
    return [
        {"category": "security", "severity": "critical", "message": "RCE vulnerability"},
        {"category": "injection", "severity": "high", "message": "SQL injection"},
        {"category": "authentication", "severity": "high", "message": "Broken auth"},
    ]


# =============================================================================
# ControlMapping tests
# =============================================================================


class TestControlMapping:
    def test_framework_controls_defined(self):
        for framework in ComplianceFramework:
            assert framework in FRAMEWORK_CONTROLS
            assert len(FRAMEWORK_CONTROLS[framework]) > 0

    def test_soc2_has_expected_controls(self):
        controls = FRAMEWORK_CONTROLS[ComplianceFramework.SOC2]
        ids = [c.control_id for c in controls]
        assert "CC6.1" in ids
        assert "CC8.1" in ids

    def test_hipaa_has_expected_controls(self):
        controls = FRAMEWORK_CONTROLS[ComplianceFramework.HIPAA]
        ids = [c.control_id for c in controls]
        assert "164.312(a)(1)" in ids
        assert "164.312(b)" in ids

    def test_pci_dss_has_expected_controls(self):
        controls = FRAMEWORK_CONTROLS[ComplianceFramework.PCI_DSS]
        ids = [c.control_id for c in controls]
        assert "6.5.1" in ids  # Injection
        assert "6.5.7" in ids  # XSS


# =============================================================================
# ComplianceReport model tests
# =============================================================================


class TestComplianceReport:
    def test_compliance_score_all_compliant(self):
        report = ComplianceReport(
            organization="TestCorp",
            framework=ComplianceFramework.SOC2,
            assessments=[
                ControlAssessment("C1", "Control 1", "soc2", ControlStatus.COMPLIANT),
                ControlAssessment("C2", "Control 2", "soc2", ControlStatus.COMPLIANT),
            ],
        )
        assert report.compliance_score == 1.0
        assert report.compliant_controls == 2

    def test_compliance_score_mixed(self):
        report = ComplianceReport(
            organization="TestCorp",
            framework=ComplianceFramework.SOC2,
            assessments=[
                ControlAssessment("C1", "Control 1", "soc2", ControlStatus.COMPLIANT),
                ControlAssessment("C2", "Control 2", "soc2", ControlStatus.NON_COMPLIANT),
            ],
        )
        assert report.compliance_score == 0.5
        assert report.non_compliant_controls == 1

    def test_compliance_score_empty(self):
        report = ComplianceReport(
            organization="TestCorp",
            framework=ComplianceFramework.SOC2,
        )
        assert report.compliance_score == 0.0
        assert report.total_controls == 0

    def test_not_assessed_excluded_from_score(self):
        report = ComplianceReport(
            organization="TestCorp",
            framework=ComplianceFramework.SOC2,
            assessments=[
                ControlAssessment("C1", "Control 1", "soc2", ControlStatus.COMPLIANT),
                ControlAssessment("C2", "Control 2", "soc2", ControlStatus.NOT_ASSESSED),
            ],
        )
        # Only 1 assessed, 1 compliant → 100%
        assert report.compliance_score == 1.0


# =============================================================================
# Report Generation tests
# =============================================================================


class TestReportGeneration:
    def test_generate_soc2_report(self, generator, sample_findings):
        report = generator.generate(
            framework=ComplianceFramework.SOC2,
            organization="TestCorp",
            findings=sample_findings,
        )
        assert report.organization == "TestCorp"
        assert report.framework == ComplianceFramework.SOC2
        assert report.total_controls == 5
        assert "compliance_score" in report.summary

    def test_generate_with_no_findings(self, generator):
        report = generator.generate(
            framework=ComplianceFramework.HIPAA,
            organization="HealthOrg",
            findings=[],
        )
        # All controls should be NOT_ASSESSED when no findings match
        for a in report.assessments:
            assert a.status == ControlStatus.NOT_ASSESSED

    def test_critical_findings_cause_non_compliance(self, generator, critical_findings):
        report = generator.generate(
            framework=ComplianceFramework.SOC2,
            organization="VulnCorp",
            findings=critical_findings,
        )
        non_compliant = [a for a in report.assessments if a.status == ControlStatus.NON_COMPLIANT]
        assert len(non_compliant) > 0

    def test_low_severity_yields_compliance(self, generator):
        findings = [
            {"category": "security", "severity": "low", "message": "Minor info leak"},
            {"category": "logging", "severity": "low", "message": "Missing debug log"},
        ]
        report = generator.generate(
            framework=ComplianceFramework.SOC2,
            organization="SafeCorp",
            findings=findings,
        )
        # Controls matched by low-severity findings should be COMPLIANT
        compliant = [a for a in report.assessments if a.status == ControlStatus.COMPLIANT]
        assert len(compliant) > 0

    def test_audit_entries_included(self, generator, sample_findings, sample_audit_entries):
        report = generator.generate(
            framework=ComplianceFramework.PCI_DSS,
            organization="PayCorp",
            findings=sample_findings,
            audit_entries=sample_audit_entries,
        )
        assert len(report.audit_entries) == 3

    def test_period_range(self, generator, sample_findings):
        report = generator.generate(
            framework=ComplianceFramework.GDPR,
            organization="EUCorp",
            findings=sample_findings,
            period_start="2025-01-01",
            period_end="2025-03-31",
        )
        assert report.period_start == "2025-01-01"
        assert report.period_end == "2025-03-31"

    def test_all_frameworks_generate(self, generator, sample_findings):
        for fw in ComplianceFramework:
            report = generator.generate(
                framework=fw,
                organization="MultiCorp",
                findings=sample_findings,
            )
            assert report.total_controls > 0
            assert isinstance(report.compliance_score, float)


# =============================================================================
# Report Rendering tests
# =============================================================================


class TestTextReport:
    def test_text_contains_header(self, generator, sample_findings):
        report = generator.generate(ComplianceFramework.SOC2, "TestCorp", sample_findings)
        text = generator.to_text(report)
        assert "SOC2" in text
        assert "TestCorp" in text
        assert "COMPLIANCE SCORE" in text

    def test_text_shows_control_statuses(self, generator, sample_findings):
        report = generator.generate(ComplianceFramework.SOC2, "TestCorp", sample_findings)
        text = generator.to_text(report)
        assert "CC6.1" in text
        assert "CC8.1" in text

    def test_text_shows_audit_trail(self, generator, sample_findings, sample_audit_entries):
        report = generator.generate(
            ComplianceFramework.SOC2,
            "TestCorp",
            sample_findings,
            audit_entries=sample_audit_entries,
        )
        text = generator.to_text(report)
        assert "AUDIT TRAIL" in text
        assert "scan_started" in text


class TestHtmlReport:
    def test_html_is_valid_doc(self, generator, sample_findings):
        report = generator.generate(ComplianceFramework.SOC2, "TestCorp", sample_findings)
        html = generator.to_html(report)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "TestCorp" in html

    def test_html_contains_score(self, generator, sample_findings):
        report = generator.generate(ComplianceFramework.SOC2, "TestCorp", sample_findings)
        html = generator.to_html(report)
        score = report.summary["compliance_score"]
        assert f"{score}%" in html

    def test_html_has_controls_table(self, generator, sample_findings):
        report = generator.generate(ComplianceFramework.PCI_DSS, "PayCorp", sample_findings)
        html = generator.to_html(report)
        assert "6.5.1" in html
        assert "Control Assessments" in html

    def test_html_with_audit_trail(self, generator, sample_findings, sample_audit_entries):
        report = generator.generate(
            ComplianceFramework.SOC2,
            "TestCorp",
            sample_findings,
            audit_entries=sample_audit_entries,
        )
        html = generator.to_html(report)
        assert "Audit Trail" in html
        assert "ci-bot" in html


class TestJsonReport:
    def test_json_valid(self, generator, sample_findings):
        report = generator.generate(ComplianceFramework.SOC2, "TestCorp", sample_findings)
        json_str = generator.to_json(report)
        data = json.loads(json_str)
        assert data["organization"] == "TestCorp"
        assert data["framework"] == "soc2"

    def test_json_has_assessments(self, generator, sample_findings):
        report = generator.generate(ComplianceFramework.SOC2, "TestCorp", sample_findings)
        data = json.loads(generator.to_json(report))
        assert len(data["assessments"]) == 5
        assert all("control_id" in a for a in data["assessments"])

    def test_json_roundtrip_summary(self, generator, sample_findings):
        report = generator.generate(ComplianceFramework.HIPAA, "HealthOrg", sample_findings)
        data = json.loads(generator.to_json(report))
        assert "compliance_score" in data["summary"]
        assert data["summary"]["total_controls"] == report.total_controls
