"""PR Impact Scorer & Labeler.

Computes a risk / impact score for a pull request based on file criticality,
change size, and verification findings.  Produces label recommendations for
GitHub/GitLab PR labeling.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    TRIVIAL = "trivial"


@dataclass
class FileRiskProfile:
    """Risk profile for a single file."""

    path: str
    criticality: float = 0.0  # 0-1
    change_lines: int = 0
    findings_count: int = 0
    critical_findings: int = 0


@dataclass
class PRImpactScore:
    """Impact assessment for a pull request."""

    risk_level: RiskLevel
    score: float  # 0-100
    labels: list[str] = field(default_factory=list)
    summary: str = ""
    file_risks: list[FileRiskProfile] = field(default_factory=list)
    findings_total: int = 0
    findings_critical: int = 0
    change_size: int = 0


# Default criticality patterns (path regex → criticality 0-1)
DEFAULT_CRITICALITY_PATTERNS: list[tuple[str, float]] = [
    (r"(test_|\.test\.|spec\.|_test\.)", 0.2),
    (r"(docs?/|README|CHANGELOG|\.md$)", 0.1),
    (r"(\.gitignore|\.eslintrc|prettier)", 0.05),
    (r"(auth|security|crypto|password|token|secret)", 1.0),
    (r"(payment|billing|checkout|stripe)", 0.95),
    (r"(database|migration|schema|model)", 0.85),
    (r"(api/|routes/|endpoint|handler)", 0.75),
    (r"(config|settings|\.env)", 0.7),
    (r"(middleware|interceptor)", 0.65),
    (r"(service|core|engine|manager)", 0.6),
    (r"(util|helper|lib/)", 0.4),
]


def compute_file_criticality(
    path: str,
    patterns: list[tuple[str, float]] | None = None,
) -> float:
    """Compute criticality score (0-1) for a file path."""
    patterns = patterns or DEFAULT_CRITICALITY_PATTERNS
    for regex, score in patterns:
        if re.search(regex, path, re.IGNORECASE):
            return score
    return 0.3  # Default for unmatched files


def _risk_level_from_score(score: float) -> RiskLevel:
    if score >= 80:
        return RiskLevel.CRITICAL
    if score >= 60:
        return RiskLevel.HIGH
    if score >= 40:
        return RiskLevel.MEDIUM
    if score >= 20:
        return RiskLevel.LOW
    return RiskLevel.TRIVIAL


def _labels_for_risk(risk: RiskLevel, findings_critical: int) -> list[str]:
    labels = [f"risk:{risk.value}"]
    if findings_critical > 0:
        labels.append("security-review-needed")
    if risk in (RiskLevel.CRITICAL, RiskLevel.HIGH):
        labels.append("needs-senior-review")
    return labels


# =============================================================================
# Impact Scorer
# =============================================================================


class PRImpactScorer:
    """Scores PR impact from file changes and verification findings.

    Usage:
        scorer = PRImpactScorer()
        impact = scorer.score(
            changed_files={"src/auth.py": 45, "tests/test_auth.py": 20},
            findings=[{"severity": "high", "file_path": "src/auth.py"}],
        )
        print(impact.risk_level, impact.labels)
    """

    def __init__(
        self,
        criticality_patterns: list[tuple[str, float]] | None = None,
        size_weight: float = 0.3,
        criticality_weight: float = 0.4,
        findings_weight: float = 0.3,
    ):
        self._patterns = criticality_patterns or DEFAULT_CRITICALITY_PATTERNS
        self._w_size = size_weight
        self._w_crit = criticality_weight
        self._w_find = findings_weight

    def score(
        self,
        changed_files: dict[str, int],
        findings: list[dict[str, Any]] | None = None,
    ) -> PRImpactScore:
        """Compute impact score for a PR.

        Args:
            changed_files: Map of file path → number of changed lines.
            findings: List of verification findings with severity, file_path.

        Returns:
            PRImpactScore with risk level, labels, and per-file breakdown.
        """
        findings = findings or []
        file_risks: list[FileRiskProfile] = []
        total_lines = sum(changed_files.values())
        findings_total = len(findings)
        findings_critical = sum(1 for f in findings if f.get("severity") in ("critical", "high"))

        # Build per-file risk
        for path, lines_changed in changed_files.items():
            crit = compute_file_criticality(path, self._patterns)
            file_findings = [f for f in findings if f.get("file_path") == path]
            file_crit = sum(1 for f in file_findings if f.get("severity") in ("critical", "high"))
            file_risks.append(
                FileRiskProfile(
                    path=path,
                    criticality=crit,
                    change_lines=lines_changed,
                    findings_count=len(file_findings),
                    critical_findings=file_crit,
                )
            )

        # Compute composite score (0-100)
        size_score = min(100.0, (total_lines / 500) * 100)

        max_crit = max((fr.criticality for fr in file_risks), default=0.0)
        crit_score = max_crit * 100

        findings_score = min(100.0, findings_critical * 30 + findings_total * 5)

        composite = (
            self._w_size * size_score + self._w_crit * crit_score + self._w_find * findings_score
        )
        composite = min(100.0, max(0.0, composite))

        risk = _risk_level_from_score(composite)
        labels = _labels_for_risk(risk, findings_critical)

        summary_parts = [f"Risk: {risk.value}"]
        summary_parts.append(f"{total_lines} lines changed across {len(changed_files)} files")
        if findings_total > 0:
            summary_parts.append(f"{findings_total} findings ({findings_critical} critical)")

        return PRImpactScore(
            risk_level=risk,
            score=round(composite, 1),
            labels=labels,
            summary=" | ".join(summary_parts),
            file_risks=file_risks,
            findings_total=findings_total,
            findings_critical=findings_critical,
            change_size=total_lines,
        )
