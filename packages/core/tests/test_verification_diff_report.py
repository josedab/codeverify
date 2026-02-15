"""Tests for verification_diff_report module."""

from __future__ import annotations

import pytest

from codeverify_core.verification_diff_report import (
    DiffReport,
    FindingSummary,
    VerificationDiffReporter,
)


@pytest.fixture
def reporter():
    return VerificationDiffReporter()


BASELINE = [
    {
        "rule_id": "null_safety",
        "file_path": "a.py",
        "line": 10,
        "severity": "high",
        "message": "Null deref",
    },
    {
        "rule_id": "overflow",
        "file_path": "b.py",
        "line": 20,
        "severity": "medium",
        "message": "Overflow risk",
    },
]


class TestFindingSummary:
    def test_identity_key(self):
        s = FindingSummary("rule1", "a.py", 10, "high", "msg")
        assert s.identity_key == "rule1|a.py|10"

    def test_different_findings_different_keys(self):
        s1 = FindingSummary("rule1", "a.py", 10, "high", "msg")
        s2 = FindingSummary("rule2", "a.py", 10, "high", "msg")
        assert s1.identity_key != s2.identity_key


class TestDiffReport:
    def test_net_change_positive(self):
        report = DiffReport(
            new_findings=[FindingSummary("r1", "a.py", 1, "high", "")],
            fixed_findings=[],
        )
        assert report.net_change == 1

    def test_net_change_negative(self):
        report = DiffReport(
            new_findings=[],
            fixed_findings=[FindingSummary("r1", "a.py", 1, "high", "")],
        )
        assert report.net_change == -1

    def test_is_improvement(self):
        report = DiffReport(
            new_findings=[],
            fixed_findings=[FindingSummary("r1", "a.py", 1, "high", "")],
        )
        assert report.is_improvement is True

    def test_has_regressions(self):
        report = DiffReport(
            regressions=[FindingSummary("r1", "a.py", 1, "critical", "")],
        )
        assert report.has_regressions is True

    def test_to_markdown_improvement(self):
        report = DiffReport(
            new_findings=[],
            fixed_findings=[FindingSummary("r1", "a.py", 1, "high", "msg")],
        )
        md = report.to_markdown()
        assert "Improvement" in md
        assert "Fixed" in md

    def test_to_markdown_regressions(self):
        reg = FindingSummary("r1", "a.py", 1, "critical", "bad")
        report = DiffReport(
            new_findings=[reg],
            regressions=[reg],
        )
        md = report.to_markdown()
        assert "Regressions" in md

    def test_to_markdown_no_changes(self):
        report = DiffReport()
        md = report.to_markdown()
        assert "No changes" in md


class TestVerificationDiffReporter:
    def test_no_changes(self, reporter):
        report = reporter.compare(BASELINE, BASELINE)
        assert report.total_new == 0
        assert report.total_fixed == 0
        assert report.total_unchanged == 2

    def test_new_finding(self, reporter):
        current = BASELINE + [
            {
                "rule_id": "xss",
                "file_path": "c.py",
                "line": 5,
                "severity": "high",
                "message": "XSS risk",
            },
        ]
        report = reporter.compare(BASELINE, current)
        assert report.total_new == 1
        assert report.new_findings[0].rule_id == "xss"

    def test_fixed_finding(self, reporter):
        current = [BASELINE[0]]  # Only first finding remains
        report = reporter.compare(BASELINE, current)
        assert report.total_fixed == 1
        assert report.fixed_findings[0].rule_id == "overflow"

    def test_regression_detection(self, reporter):
        current = BASELINE + [
            {
                "rule_id": "injection",
                "file_path": "d.py",
                "line": 1,
                "severity": "critical",
                "message": "SQL injection",
            },
        ]
        report = reporter.compare(BASELINE, current)
        assert report.has_regressions is True
        assert report.regressions[0].severity == "critical"

    def test_custom_regression_severities(self, reporter):
        current = BASELINE + [
            {
                "rule_id": "style",
                "file_path": "e.py",
                "line": 1,
                "severity": "low",
                "message": "Style issue",
            },
        ]
        report = reporter.compare(BASELINE, current, regression_severities=("low",))
        assert report.has_regressions is True

    def test_empty_baseline(self, reporter):
        current = [
            {"rule_id": "r1", "file_path": "a.py", "line": 1, "severity": "high", "message": "new"},
        ]
        report = reporter.compare([], current)
        assert report.total_new == 1
        assert report.total_fixed == 0

    def test_empty_current(self, reporter):
        report = reporter.compare(BASELINE, [])
        assert report.total_fixed == 2
        assert report.total_new == 0
        assert report.is_improvement is True

    def test_both_empty(self, reporter):
        report = reporter.compare([], [])
        assert report.total_new == 0
        assert report.total_fixed == 0
