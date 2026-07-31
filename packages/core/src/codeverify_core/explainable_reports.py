"""Explainable AI Verification Reports.

Auto-generates plain-English explanations of Z3 verification results with
visual diagrams. Makes formal methods accessible to non-experts by translating
SMT-LIB output into narrative explanations with analogies.

Features:
- Z3 output → plain English summarization
- Visual counterexamples with Mermaid diagrams
- Severity explanations with CVE references
- Tiered explanations (beginner/intermediate/expert)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ExplanationLevel(str, Enum):
    """Detail level for explanations."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    EXPERT = "expert"


class VerificationOutcome(str, Enum):
    """Outcome of a verification check."""

    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"


@dataclass
class CounterExample:
    """A concrete counterexample showing how code can fail."""

    variables: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    execution_path: list[str] = field(default_factory=list)

    def format_table(self) -> str:
        if not self.variables:
            return "No counterexample available"
        lines = ["| Variable | Value |", "|----------|-------|"]
        for var, val in self.variables.items():
            lines.append(f"| `{var}` | `{val}` |")
        return "\n".join(lines)


@dataclass
class VerificationFinding:
    """A single verification finding to be explained."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    check_type: str = ""
    outcome: VerificationOutcome = VerificationOutcome.UNSAFE
    constraint: str = ""
    file_path: str = ""
    function_name: str = ""
    line: int = 0
    severity: str = "medium"
    counterexample: CounterExample | None = None
    raw_z3_output: str = ""


@dataclass
class ExplainedFinding:
    """A finding with human-readable explanation."""

    finding: VerificationFinding
    title: str = ""
    summary: str = ""
    explanation: str = ""
    analogy: str = ""
    visual_diagram: str = ""
    fix_suggestion: str = ""
    severity_reason: str = ""
    similar_cves: list[str] = field(default_factory=list)
    level: ExplanationLevel = ExplanationLevel.INTERMEDIATE

    def to_markdown(self) -> str:
        lines = [f"### {self.title}", ""]
        if self.summary:
            lines.extend([f"**Summary:** {self.summary}", ""])
        if self.explanation:
            lines.extend(["**What happened:**", self.explanation, ""])
        if self.analogy:
            lines.extend([f"💡 **Analogy:** {self.analogy}", ""])
        if self.finding.counterexample:
            lines.extend(
                [
                    "**Counterexample:**",
                    self.finding.counterexample.format_table(),
                    "",
                ]
            )
        if self.visual_diagram:
            lines.extend(["**Execution trace:**", "```mermaid", self.visual_diagram, "```", ""])
        if self.fix_suggestion:
            lines.extend([f"**Suggested fix:** {self.fix_suggestion}", ""])
        if self.severity_reason:
            lines.extend([f"**Why {self.finding.severity}:** {self.severity_reason}", ""])
        if self.similar_cves:
            lines.extend(["**Similar CVEs:** " + ", ".join(self.similar_cves), ""])
        return "\n".join(lines)


# Explanation templates per check type
_CHECK_EXPLANATIONS: dict[str, dict[str, str]] = {
    "null_safety": {
        "title": "Null Safety Violation",
        "beginner": (
            "Your code tries to use a value that might be empty (null/None). "
            "This is like trying to open an envelope that might not exist — "
            "your program will crash."
        ),
        "intermediate": (
            "The variable could be None at this point in the code. "
            "If the code tries to access an attribute or call a method on it, "
            "a NullPointerException/AttributeError will be raised."
        ),
        "expert": (
            "Z3 found a satisfying assignment where the variable is null "
            "at the dereference point. The null-safety constraint is unsatisfiable "
            "under the current path conditions."
        ),
        "analogy": "Like trying to read a page from a book that hasn't been printed yet.",
        "fix": "Add a null check before accessing the variable: `if x is not None:`",
    },
    "division_by_zero": {
        "title": "Division by Zero",
        "beginner": (
            "Your code divides a number by something that could be zero. "
            "Dividing by zero is undefined in math and will crash your program."
        ),
        "intermediate": (
            "The divisor expression can evaluate to zero under certain inputs. "
            "This will raise a ZeroDivisionError at runtime."
        ),
        "expert": (
            "Z3 demonstrated a counterexample where the divisor constraint "
            "evaluates to zero. The non-zero precondition is not enforced."
        ),
        "analogy": "Like trying to split a pizza among zero people — it doesn't make sense.",
        "fix": "Guard the division: `if divisor != 0: result = x / divisor`",
    },
    "array_bounds": {
        "title": "Array Index Out of Bounds",
        "beginner": (
            "Your code tries to access an item in a list at a position that "
            "might not exist. Like asking for seat 100 on a bus with 50 seats."
        ),
        "intermediate": (
            "The array index can exceed the array length under certain inputs. "
            "This will raise an IndexError at runtime."
        ),
        "expert": (
            "Z3 found an assignment where index >= len(array) or index < 0. "
            "The bounds constraint is violated under the discovered path conditions."
        ),
        "analogy": "Like trying to go to floor 15 in a 10-story building.",
        "fix": "Check bounds: `if 0 <= index < len(array):`",
    },
    "integer_overflow": {
        "title": "Integer Overflow",
        "beginner": (
            "Your calculation produces a number too large to store. "
            "Like a car odometer rolling over from 999,999 back to 000,000."
        ),
        "intermediate": (
            "The arithmetic operation can produce a value exceeding the "
            "maximum representable integer, causing wrap-around or silent corruption."
        ),
        "expert": (
            "Z3 demonstrated inputs where the result exceeds 2^63-1 (signed) "
            "or 2^64-1 (unsigned). No overflow check exists on the arithmetic path."
        ),
        "analogy": "Like a scoreboard that only goes to 99 — score 100 shows as 00.",
        "fix": "Use checked arithmetic or validate inputs: `if a + b > MAX_INT: raise`",
    },
    "resource_leak": {
        "title": "Resource Leak",
        "beginner": (
            "Your code opens a file or connection but might not close it. "
            "Like leaving the water tap running — eventually resources run out."
        ),
        "intermediate": (
            "A resource is acquired but not released on all code paths. "
            "If an exception occurs between open and close, the resource leaks."
        ),
        "expert": (
            "Z3 found a path where the resource acquisition is not paired with "
            "a corresponding release. The exception path bypasses the cleanup."
        ),
        "analogy": "Like borrowing a library book and forgetting to return it.",
        "fix": "Use a context manager: `with open(file) as f:`",
    },
    "race_condition": {
        "title": "Potential Race Condition",
        "beginner": (
            "Two parts of your code might try to change the same thing "
            "at the same time, like two people editing the same document."
        ),
        "intermediate": (
            "Shared mutable state is accessed without synchronization. "
            "Concurrent access can lead to data corruption or lost updates."
        ),
        "expert": (
            "The shared variable has a read-modify-write pattern without "
            "atomic guarantees. Interleaving of concurrent accesses can "
            "violate the expected invariant."
        ),
        "analogy": "Like two cashiers both trying to make change from the same register.",
        "fix": "Add synchronization: use a lock, mutex, or atomic operation",
    },
}

# CVE references per check type
_RELATED_CVES: dict[str, list[str]] = {
    "null_safety": ["CVE-2021-44228 (Log4Shell null context)", "CWE-476"],
    "division_by_zero": ["CWE-369"],
    "array_bounds": ["CVE-2021-3156 (sudo heap overflow)", "CWE-125", "CWE-787"],
    "integer_overflow": ["CVE-2014-0160 (Heartbleed)", "CWE-190"],
    "resource_leak": ["CWE-404", "CWE-772"],
    "race_condition": ["CWE-362", "CWE-367"],
}


class FindingExplainer:
    """Generates human-readable explanations for verification findings."""

    def __init__(
        self,
        default_level: ExplanationLevel = ExplanationLevel.INTERMEDIATE,
    ) -> None:
        self.default_level = default_level

    def explain(
        self,
        finding: VerificationFinding,
        level: ExplanationLevel | None = None,
    ) -> ExplainedFinding:
        """Generate a human-readable explanation for a finding."""
        lvl = level or self.default_level
        templates = _CHECK_EXPLANATIONS.get(finding.check_type, {})

        title = templates.get("title", f"Verification Issue: {finding.check_type}")
        explanation = templates.get(lvl.value, templates.get("intermediate", ""))
        analogy = templates.get("analogy", "")
        fix = templates.get("fix", "")
        cves = _RELATED_CVES.get(finding.check_type, [])

        # Generate summary
        summary = self._generate_summary(finding)

        # Generate visual diagram
        diagram = self._generate_diagram(finding)

        # Severity reasoning
        severity_reason = self._explain_severity(finding)

        return ExplainedFinding(
            finding=finding,
            title=title,
            summary=summary,
            explanation=explanation if explanation else self._fallback_explanation(finding),
            analogy=analogy,
            visual_diagram=diagram,
            fix_suggestion=fix,
            severity_reason=severity_reason,
            similar_cves=cves,
            level=lvl,
        )

    def explain_batch(
        self,
        findings: list[VerificationFinding],
        level: ExplanationLevel | None = None,
    ) -> list[ExplainedFinding]:
        """Explain multiple findings."""
        return [self.explain(f, level) for f in findings]

    def _generate_summary(self, finding: VerificationFinding) -> str:
        outcome_text = {
            VerificationOutcome.SAFE: "verified as safe",
            VerificationOutcome.UNSAFE: "found to be potentially unsafe",
            VerificationOutcome.UNKNOWN: "could not be determined",
            VerificationOutcome.TIMEOUT: "timed out during verification",
        }
        status = outcome_text.get(finding.outcome, "analyzed")
        location = ""
        if finding.function_name:
            location = f" in `{finding.function_name}()`"
        if finding.file_path:
            location += f" at {finding.file_path}:{finding.line}"
        return f"The {finding.check_type.replace('_', ' ')} check was {status}{location}."

    def _fallback_explanation(self, finding: VerificationFinding) -> str:
        return (
            f"A {finding.check_type.replace('_', ' ')} issue was detected. "
            f"The verification constraint `{finding.constraint}` could not be satisfied "
            f"under all possible inputs."
        )

    def _generate_diagram(self, finding: VerificationFinding) -> str:
        """Generate a Mermaid diagram for the finding."""
        if not finding.counterexample or not finding.counterexample.execution_path:
            # Simple diagram
            return (
                f"graph TD\n"
                f'    A["Input"] --> B["{finding.check_type}"]\n'
                f'    B --> C{{"{finding.outcome.value}"}}\n'
                f'    C -->|"unsafe"| D["⚠️ Issue at L{finding.line}"]'
            )

        # Detailed diagram from execution path
        lines = ["graph TD"]
        for i, step in enumerate(finding.counterexample.execution_path):
            safe_step = step.replace('"', "'")
            node_id = f"S{i}"
            next_id = f"S{i + 1}" if i < len(finding.counterexample.execution_path) - 1 else "END"
            lines.append(f'    {node_id}["{safe_step}"]')
            if i < len(finding.counterexample.execution_path) - 1:
                lines.append(f"    {node_id} --> {next_id}")
        lines.append(f'    END["⚠️ {finding.outcome.value}"]')
        if finding.counterexample.execution_path:
            lines.append(f"    S{len(finding.counterexample.execution_path) - 1} --> END")
        return "\n".join(lines)

    def _explain_severity(self, finding: VerificationFinding) -> str:
        reasons = {
            "critical": (
                "This issue can be exploited remotely without authentication "
                "and may lead to arbitrary code execution or data loss."
            ),
            "high": (
                "This issue has a high probability of causing runtime failures "
                "and could lead to security vulnerabilities if exposed to user input."
            ),
            "medium": (
                "This issue may cause unexpected behavior under specific conditions "
                "but requires particular inputs to trigger."
            ),
            "low": (
                "This issue is unlikely to cause significant problems but "
                "represents a code quality concern."
            ),
        }
        return reasons.get(finding.severity, "Severity based on impact and exploitability.")


class ExplainableReport:
    """Generates a complete explainable verification report."""

    def __init__(
        self,
        explainer: FindingExplainer | None = None,
        level: ExplanationLevel = ExplanationLevel.INTERMEDIATE,
    ) -> None:
        self.explainer = explainer or FindingExplainer(default_level=level)
        self.level = level
        self._findings: list[VerificationFinding] = []
        self._explained: list[ExplainedFinding] = []

    def add_finding(self, finding: VerificationFinding) -> ExplainedFinding:
        """Add a finding and generate its explanation."""
        self._findings.append(finding)
        explained = self.explainer.explain(finding, self.level)
        self._explained.append(explained)
        return explained

    def add_findings(self, findings: list[VerificationFinding]) -> list[ExplainedFinding]:
        return [self.add_finding(f) for f in findings]

    @property
    def findings(self) -> list[ExplainedFinding]:
        return list(self._explained)

    def to_markdown(self) -> str:
        """Generate a full markdown report."""
        lines = [
            "# Verification Report",
            "",
            f"**Findings:** {len(self._explained)}",
            f"**Explanation Level:** {self.level.value}",
            "",
        ]

        severity_order = ["critical", "high", "medium", "low", "info"]
        sorted_findings = sorted(
            self._explained,
            key=lambda e: (
                severity_order.index(e.finding.severity)
                if e.finding.severity in severity_order
                else 99
            ),
        )

        for explained in sorted_findings:
            lines.append(explained.to_markdown())
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_findings": len(self._explained),
            "level": self.level.value,
            "findings": [
                {
                    "title": e.title,
                    "summary": e.summary,
                    "severity": e.finding.severity,
                    "check_type": e.finding.check_type,
                    "explanation": e.explanation,
                    "analogy": e.analogy,
                    "fix_suggestion": e.fix_suggestion,
                    "similar_cves": e.similar_cves,
                }
                for e in self._explained
            ],
        }


# Singleton
_finding_explainer_instance: FindingExplainer | None = None


def get_finding_explainer() -> FindingExplainer:
    global _finding_explainer_instance
    if _finding_explainer_instance is None:
        _finding_explainer_instance = FindingExplainer()
    return _finding_explainer_instance


def reset_finding_explainer() -> None:
    global _finding_explainer_instance
    _finding_explainer_instance = None
