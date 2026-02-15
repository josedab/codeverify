"""Verification Diff Reporter.

Compares verification results between two runs to show new findings,
fixed findings, and unchanged findings. Detects regressions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FindingSummary:
    """Compact summary of a finding for comparison."""

    rule_id: str
    file_path: str
    line: int
    severity: str
    message: str

    @property
    def identity_key(self) -> str:
        """Key for identity comparison across runs."""
        return f"{self.rule_id}|{self.file_path}|{self.line}"


@dataclass
class DiffReport:
    """Report comparing two verification runs."""

    new_findings: list[FindingSummary] = field(default_factory=list)
    fixed_findings: list[FindingSummary] = field(default_factory=list)
    unchanged_findings: list[FindingSummary] = field(default_factory=list)
    regressions: list[FindingSummary] = field(default_factory=list)

    @property
    def total_new(self) -> int:
        return len(self.new_findings)

    @property
    def total_fixed(self) -> int:
        return len(self.fixed_findings)

    @property
    def total_unchanged(self) -> int:
        return len(self.unchanged_findings)

    @property
    def has_regressions(self) -> bool:
        return len(self.regressions) > 0

    @property
    def is_improvement(self) -> bool:
        return self.total_fixed > 0 and self.total_new == 0

    @property
    def net_change(self) -> int:
        """Positive = more findings, negative = fewer."""
        return self.total_new - self.total_fixed

    def to_markdown(self) -> str:
        """Render as a markdown summary."""
        lines: list[str] = []
        lines.append("## Verification Diff Report")
        lines.append("")

        if self.is_improvement:
            lines.append(f"🎉 **Improvement**: {self.total_fixed} finding(s) fixed!")
        elif self.has_regressions:
            lines.append(
                f"⚠️ **Regressions detected**: {len(self.regressions)} critical/high finding(s) introduced"
            )
        elif self.total_new > 0:
            lines.append(f"📝 {self.total_new} new finding(s), {self.total_fixed} fixed")
        else:
            lines.append("✅ No changes in findings")

        lines.append("")
        lines.append("| Metric | Count |")
        lines.append("|--------|-------|")
        lines.append(f"| New | {self.total_new} |")
        lines.append(f"| Fixed | {self.total_fixed} |")
        lines.append(f"| Unchanged | {self.total_unchanged} |")
        lines.append(f"| Net change | {self.net_change:+d} |")

        if self.new_findings:
            lines.append("")
            lines.append("### New Findings")
            for f in self.new_findings:
                lines.append(
                    f"- **{f.severity}** `{f.rule_id}` in `{f.file_path}:{f.line}`: {f.message}"
                )

        if self.fixed_findings:
            lines.append("")
            lines.append("### Fixed Findings")
            for f in self.fixed_findings:
                lines.append(f"- ~~{f.rule_id} in {f.file_path}:{f.line}~~")

        return "\n".join(lines)


def _to_summaries(findings: list[dict[str, Any]]) -> list[FindingSummary]:
    return [
        FindingSummary(
            rule_id=f.get("rule_id", ""),
            file_path=f.get("file_path", ""),
            line=f.get("line", 0),
            severity=f.get("severity", "medium"),
            message=f.get("message", ""),
        )
        for f in findings
    ]


class VerificationDiffReporter:
    """Compare two verification runs and produce a diff report.

    Usage:
        reporter = VerificationDiffReporter()
        report = reporter.compare(baseline_findings, current_findings)
        print(report.to_markdown())
    """

    def compare(
        self,
        baseline: list[dict[str, Any]],
        current: list[dict[str, Any]],
        regression_severities: tuple[str, ...] = ("critical", "high"),
    ) -> DiffReport:
        """Compare baseline and current findings.

        Args:
            baseline: Previous run's findings.
            current: Current run's findings.
            regression_severities: Severities that count as regressions.
        """
        base_summaries = _to_summaries(baseline)
        curr_summaries = _to_summaries(current)

        base_keys = {s.identity_key: s for s in base_summaries}
        curr_keys = {s.identity_key: s for s in curr_summaries}

        new_findings = [s for k, s in curr_keys.items() if k not in base_keys]
        fixed_findings = [s for k, s in base_keys.items() if k not in curr_keys]
        unchanged = [s for k, s in curr_keys.items() if k in base_keys]
        regressions = [f for f in new_findings if f.severity in regression_severities]

        return DiffReport(
            new_findings=new_findings,
            fixed_findings=fixed_findings,
            unchanged_findings=unchanged,
            regressions=regressions,
        )
