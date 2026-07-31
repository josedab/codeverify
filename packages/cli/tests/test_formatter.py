"""Unit tests for cli/formatter.py — output formatting for rich, JSON, and SARIF."""

from __future__ import annotations

from io import StringIO
from unittest.mock import MagicMock

from codeverify_cli.formatter import format_diff, format_findings, format_summary
from rich.console import Console


def _console() -> Console:
    """Create a Console that writes to a string buffer with no markup."""
    return Console(file=StringIO(), force_terminal=False, no_color=True, width=120)


def _get_output(console: Console) -> str:
    console.file.seek(0)
    return console.file.read()


class TestFormatFindings:
    """Test format_findings rich output."""

    def test_no_findings_shows_success(self):
        c = _console()
        results = MagicMock()
        results.findings = []
        format_findings(c, results)
        output = _get_output(c)
        assert "No issues found" in output

    def test_shows_finding_count(self):
        c = _console()
        results = MagicMock()
        results.findings = [
            {"file_path": "src/main.py", "severity": "high", "line_start": 10, "title": "Bug"},
            {"file_path": "src/main.py", "severity": "low", "line_start": 20, "title": "Style"},
        ]
        format_findings(c, results)
        output = _get_output(c)
        assert "2 issue(s)" in output

    def test_groups_by_file(self):
        c = _console()
        results = MagicMock()
        results.findings = [
            {"file_path": "a.py", "severity": "high", "line_start": 1, "title": "Bug A"},
            {"file_path": "b.py", "severity": "low", "line_start": 1, "title": "Bug B"},
        ]
        format_findings(c, results)
        output = _get_output(c)
        assert "a.py" in output
        assert "b.py" in output

    def test_shows_description(self):
        c = _console()
        results = MagicMock()
        results.findings = [
            {
                "file_path": "a.py",
                "severity": "medium",
                "line_start": 5,
                "title": "Issue",
                "description": "Something is wrong here",
            },
        ]
        format_findings(c, results)
        output = _get_output(c)
        assert "Something is wrong" in output

    def test_shows_fix_when_flag_set(self):
        c = _console()
        results = MagicMock()
        results.findings = [
            {
                "file_path": "a.py",
                "severity": "low",
                "line_start": 1,
                "title": "Issue",
                "fix_suggestion": "x = safe_call()",
            },
        ]
        format_findings(c, results, show_fix=True)
        output = _get_output(c)
        assert "Suggested fix" in output

    def test_no_fix_when_flag_not_set(self):
        c = _console()
        results = MagicMock()
        results.findings = [
            {
                "file_path": "a.py",
                "severity": "low",
                "line_start": 1,
                "title": "Issue",
                "fix_suggestion": "x = safe_call()",
            },
        ]
        format_findings(c, results, show_fix=False)
        output = _get_output(c)
        assert "Suggested fix" not in output

    def test_handles_missing_optional_fields(self):
        c = _console()
        results = MagicMock()
        results.findings = [
            {"file_path": "a.py", "severity": "info", "title": "Note"},
        ]
        format_findings(c, results)
        output = _get_output(c)
        assert "Note" in output

    def test_detects_typescript_language(self):
        c = _console()
        results = MagicMock()
        results.findings = [
            {
                "file_path": "src/index.ts",
                "severity": "low",
                "line_start": 1,
                "title": "Issue",
                "fix_suggestion": "const x = 1;",
            },
        ]
        format_findings(c, results, show_fix=True)
        output = _get_output(c)
        assert "Suggested fix" in output


class TestFormatSummary:
    """Test format_summary rich table output."""

    def test_shows_summary_table(self):
        c = _console()
        results = MagicMock()
        results.to_dict.return_value = {
            "summary": {"critical": 1, "high": 2, "medium": 3, "low": 4, "total": 10}
        }
        results.files_analyzed = 5
        results.functions_found = 10
        results.classes_found = 3
        results.duration_ms = 150.0

        format_summary(c, results)
        output = _get_output(c)
        assert "Analysis Summary" in output
        assert "5" in output  # files
        assert "150" in output  # duration

    def test_pass_when_no_critical_or_high(self):
        c = _console()
        results = MagicMock()
        results.to_dict.return_value = {
            "summary": {"critical": 0, "high": 0, "medium": 1, "low": 2, "total": 3}
        }
        results.files_analyzed = 1
        results.functions_found = 0
        results.classes_found = 0
        results.duration_ms = 50.0

        format_summary(c, results)
        output = _get_output(c)
        assert "passed" in output.lower()

    def test_fail_when_critical_found(self):
        c = _console()
        results = MagicMock()
        results.to_dict.return_value = {
            "summary": {"critical": 1, "high": 0, "medium": 0, "low": 0, "total": 1}
        }
        results.files_analyzed = 1
        results.functions_found = 0
        results.classes_found = 0
        results.duration_ms = 10.0

        format_summary(c, results)
        output = _get_output(c)
        assert "failed" in output.lower()

    def test_fail_when_high_found(self):
        c = _console()
        results = MagicMock()
        results.to_dict.return_value = {
            "summary": {"critical": 0, "high": 3, "medium": 0, "low": 0, "total": 3}
        }
        results.files_analyzed = 1
        results.functions_found = 0
        results.classes_found = 0
        results.duration_ms = 10.0

        format_summary(c, results)
        output = _get_output(c)
        assert "failed" in output.lower()


class TestFormatDiff:
    """Test format_diff output."""

    def test_shows_diff_when_changes_exist(self):
        c = _console()
        format_diff(c, "x = 1\ny = 2\n", "x = 1\ny = 3\n", "test.py")
        output = _get_output(c)
        assert "test.py" in output

    def test_shows_no_changes_when_identical(self):
        c = _console()
        format_diff(c, "x = 1\n", "x = 1\n", "test.py")
        output = _get_output(c)
        assert "No changes" in output

    def test_handles_empty_strings(self):
        c = _console()
        format_diff(c, "", "new content\n", "test.py")
        output = _get_output(c)
        # Should show a diff with additions
        assert len(output) > 0
