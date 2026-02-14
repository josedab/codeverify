"""AI Code Generation Firewall.

Real-time proxy that intercepts AI-generated code suggestions (from Copilot,
ChatGPT, Claude, etc.), analyses them for security and quality risks, and
blocks or sanitizes dangerous patterns before they reach the codebase.

Key capabilities:
1. Interception: Capture code suggestions from multiple AI sources
2. Risk Analysis: Detect secrets, injections, unsafe imports, and quality issues
3. Sanitization: Automatically fix dangerous patterns while preserving intent
4. Policy Enforcement: Configurable rules for blocking, warning, or quarantining
5. Metrics: Track firewall effectiveness and risk trends
"""

from __future__ import annotations

import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


# =============================================================================
# Enumerations
# =============================================================================


class SuggestionSource(str, Enum):
    """Origin of the AI code suggestion."""

    COPILOT = "copilot"
    CHATGPT = "chatgpt"
    CLAUDE = "claude"
    CODEWHISPERER = "codewhisperer"
    TABNINE = "tabnine"
    UNKNOWN = "unknown"


class FirewallAction(str, Enum):
    """Action the firewall takes on an intercepted suggestion."""

    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    SANITIZE = "sanitize"
    QUARANTINE = "quarantine"


class RiskLevel(str, Enum):
    """Assessed risk level of a code suggestion."""

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def severity_order(self) -> int:
        return {"safe": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}[self.value]

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.severity_order >= other.severity_order

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, RiskLevel):
            return NotImplemented
        return self.severity_order > other.severity_order


class SanitizationType(str, Enum):
    """Type of sanitization applied to dangerous code."""

    REMOVE_SECRETS = "remove_secrets"
    ESCAPE_INJECTION = "escape_injection"
    FIX_VULNERABILITY = "fix_vulnerability"
    ADD_VALIDATION = "add_validation"
    REMOVE_UNSAFE_IMPORT = "remove_unsafe_import"


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class SuggestionInterception:
    """An intercepted AI code suggestion with its surrounding context."""

    id: str
    source: SuggestionSource
    code: str
    language: str
    file_path: str
    cursor_line: int
    cursor_column: int
    timestamp: float
    context_before: str = ""
    context_after: str = ""
    model_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source.value,
            "code": self.code,
            "language": self.language,
            "file_path": self.file_path,
            "cursor_line": self.cursor_line,
            "cursor_column": self.cursor_column,
            "timestamp": self.timestamp,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "model_id": self.model_id,
        }


@dataclass
class FirewallPolicy:
    """Policy governing how the firewall handles suggestions at various risk levels."""

    id: str
    name: str
    description: str
    enabled: bool = True
    risk_threshold: RiskLevel = RiskLevel.HIGH
    action_on_block: FirewallAction = FirewallAction.BLOCK
    allow_override: bool = True
    max_override_count: int = 3
    blocked_patterns: list[str] = field(default_factory=list)
    blocked_imports: list[str] = field(default_factory=list)
    require_review_above: RiskLevel = RiskLevel.MEDIUM

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "risk_threshold": self.risk_threshold.value,
            "action_on_block": self.action_on_block.value,
            "allow_override": self.allow_override,
            "max_override_count": self.max_override_count,
            "blocked_patterns": self.blocked_patterns,
            "blocked_imports": self.blocked_imports,
            "require_review_above": self.require_review_above.value,
        }


@dataclass
class RiskAssessment:
    """Result of analysing a suggestion for security and quality risks."""

    risk_level: RiskLevel
    confidence: float
    risk_factors: list[dict[str, Any]]
    security_issues: list[str]
    quality_issues: list[str]
    overall_score: float  # 0-100, higher = more risky

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level.value,
            "confidence": round(self.confidence, 3),
            "risk_factors": self.risk_factors,
            "security_issues": self.security_issues,
            "quality_issues": self.quality_issues,
            "overall_score": round(self.overall_score, 2),
        }


@dataclass
class SanitizationAction:
    """A single sanitization operation applied to a code suggestion."""

    sanitization_type: SanitizationType
    original_code: str
    sanitized_code: str
    description: str
    line: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sanitization_type": self.sanitization_type.value,
            "original_code": self.original_code,
            "sanitized_code": self.sanitized_code,
            "description": self.description,
            "line": self.line,
        }


@dataclass
class FirewallDecision:
    """The firewall's final decision on an intercepted suggestion."""

    interception_id: str
    action: FirewallAction
    risk_assessment: RiskAssessment
    sanitizations: list[SanitizationAction] = field(default_factory=list)
    original_code: str = ""
    modified_code: str | None = None
    override_available: bool = True
    reason: str = ""
    processing_time_ms: float = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "interception_id": self.interception_id,
            "action": self.action.value,
            "risk_assessment": self.risk_assessment.to_dict(),
            "sanitizations": [s.to_dict() for s in self.sanitizations],
            "original_code": self.original_code,
            "modified_code": self.modified_code,
            "override_available": self.override_available,
            "reason": self.reason,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


@dataclass
class FirewallMetrics:
    """Aggregated metrics for firewall activity."""

    total_interceptions: int = 0
    allowed: int = 0
    warned: int = 0
    blocked: int = 0
    sanitized: int = 0
    overrides: int = 0
    avg_processing_time_ms: float = 0
    top_risk_factors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_interceptions": self.total_interceptions,
            "allowed": self.allowed,
            "warned": self.warned,
            "blocked": self.blocked,
            "sanitized": self.sanitized,
            "overrides": self.overrides,
            "avg_processing_time_ms": round(self.avg_processing_time_ms, 2),
            "top_risk_factors": self.top_risk_factors,
        }


# =============================================================================
# Secret detection patterns
# =============================================================================

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS Access Key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("AWS Secret Key", re.compile(r"""(?:aws)?_?secret_?(?:access)?_?key\s*[=:]\s*['"][A-Za-z0-9/+=]{40}['"]""", re.IGNORECASE)),
    ("Generic API Key", re.compile(r"""(?:api[_-]?key|apikey)\s*[=:]\s*['"][A-Za-z0-9_\-]{20,}['"]""", re.IGNORECASE)),
    ("Generic Secret", re.compile(r"""(?:secret|token|password|passwd|pwd)\s*[=:]\s*['"][^\s'"]{8,}['"]""", re.IGNORECASE)),
    ("GitHub Token", re.compile(r"gh[pousr]_[A-Za-z0-9_]{36,}")),
    ("Slack Token", re.compile(r"xox[baprs]-[0-9A-Za-z\-]{10,}")),
    ("Private Key Header", re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----")),
    ("JWT Token", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_\-]+")),
    ("Hardcoded IP + Port", re.compile(r"\b(?:password|secret|key)\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE)),
]

# =============================================================================
# Injection patterns
# =============================================================================

_SQL_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("String-formatted SQL", re.compile(r"""(?:execute|cursor\.execute|query)\s*\(\s*(?:f['\"]|['\"].*%s|['\"].*\bformat\b)""", re.IGNORECASE)),
    ("Raw SQL concatenation", re.compile(r"""(?:SELECT|INSERT|UPDATE|DELETE|DROP|ALTER)\s+.*\+\s*(?:str\(|request\.|input\(|user)""", re.IGNORECASE)),
    ("SQL string interpolation", re.compile(r"""(?:SELECT|INSERT|UPDATE|DELETE)\s+.*\{.*\}""", re.IGNORECASE)),
]

_COMMAND_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("os.system call", re.compile(r"\bos\.system\s*\(")),
    ("os.popen call", re.compile(r"\bos\.popen\s*\(")),
    ("subprocess shell=True", re.compile(r"\bsubprocess\.\w+\s*\([^)]*shell\s*=\s*True")),
    ("eval() usage", re.compile(r"\beval\s*\(\s*(?!.*\bliteral_eval\b)")),
    ("exec() usage", re.compile(r"\bexec\s*\(")),
    ("compile() with exec", re.compile(r"\bcompile\s*\([^)]*['\"]exec['\"]")),
]

_JS_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("innerHTML assignment", re.compile(r"\.innerHTML\s*=\s*(?!['\"]\s*['\"]\s*;)")),
    ("document.write", re.compile(r"\bdocument\.write\s*\(")),
    ("eval() usage", re.compile(r"\beval\s*\(")),
    ("Function constructor", re.compile(r"\bnew\s+Function\s*\(")),
]

# =============================================================================
# Unsafe import patterns
# =============================================================================

_UNSAFE_IMPORTS: dict[str, list[str]] = {
    "python": [
        "pickle",
        "shelve",
        "marshal",
        "os.system",
        "commands",
        "tempfile.mktemp",
    ],
    "javascript": [
        "child_process",
        "vm",
    ],
}


# =============================================================================
# Risk analyzer
# =============================================================================


class SuggestionRiskAnalyzer:
    """Analyse AI code suggestions for security and quality risks.

    Inspects intercepted suggestions against a curated set of patterns
    covering hardcoded secrets, injection vectors, unsafe imports, and
    general code-quality heuristics.
    """

    def __init__(self) -> None:
        self._secret_patterns = _SECRET_PATTERNS
        self._sql_patterns = _SQL_INJECTION_PATTERNS
        self._cmd_patterns = _COMMAND_INJECTION_PATTERNS
        self._js_patterns = _JS_INJECTION_PATTERNS
        self._unsafe_imports = _UNSAFE_IMPORTS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, interception: SuggestionInterception) -> RiskAssessment:
        """Return a full risk assessment for the intercepted suggestion."""
        code = interception.code
        language = interception.language.lower()

        factors: list[dict[str, Any]] = []

        factors.extend(self._check_secrets(code))
        factors.extend(self._check_injection(code, language))
        factors.extend(self._check_unsafe_patterns(code, language))
        factors.extend(self._check_quality(code))

        risk_level, confidence, score = self._calculate_risk_score(factors)

        security_issues = [
            f["description"]
            for f in factors
            if f.get("category") in ("secret", "injection", "unsafe_pattern")
        ]
        quality_issues = [
            f["description"] for f in factors if f.get("category") == "quality"
        ]

        logger.debug(
            "risk_analysis_complete",
            interception_id=interception.id,
            risk_level=risk_level.value,
            factor_count=len(factors),
            score=round(score, 2),
        )

        return RiskAssessment(
            risk_level=risk_level,
            confidence=confidence,
            risk_factors=factors,
            security_issues=security_issues,
            quality_issues=quality_issues,
            overall_score=score,
        )

    # ------------------------------------------------------------------
    # Secret detection
    # ------------------------------------------------------------------

    def _check_secrets(self, code: str) -> list[dict[str, Any]]:
        """Detect hardcoded secrets, API keys, and credentials."""
        factors: list[dict[str, Any]] = []
        for label, pattern in self._secret_patterns:
            for match in pattern.finditer(code):
                ln = code[: match.start()].count("\n") + 1
                factors.append({"category": "secret", "severity": "critical",
                                "description": f"Hardcoded {label} detected on line {ln}",
                                "pattern": label, "line": ln,
                                "matched_text": _redact(match.group()), "weight": 30.0})
        return factors

    # ------------------------------------------------------------------
    # Injection detection
    # ------------------------------------------------------------------

    def _check_injection(self, code: str, language: str) -> list[dict[str, Any]]:
        """Detect SQL injection, command injection, and XSS vectors."""
        factors: list[dict[str, Any]] = []

        def _scan(
            patterns: list[tuple[str, re.Pattern[str]]],
            category_label: str,
            weight: float,
        ) -> None:
            for label, pattern in patterns:
                for match in pattern.finditer(code):
                    ln = code[: match.start()].count("\n") + 1
                    factors.append({
                        "category": "injection",
                        "severity": "high",
                        "description": f"{category_label} ({label}) on line {ln}",
                        "pattern": label, "line": ln,
                        "matched_text": _truncate(match.group(), 80),
                        "weight": weight,
                    })

        _scan(self._sql_patterns, "Potential SQL injection", 25.0)

        if language in ("python", "py"):
            _scan(self._cmd_patterns, "Command injection risk", 25.0)

        if language in ("javascript", "js", "typescript", "ts", "jsx", "tsx"):
            _scan(self._js_patterns, "XSS / JS injection risk", 22.0)

        return factors

    # ------------------------------------------------------------------
    # Unsafe patterns
    # ------------------------------------------------------------------

    def _check_unsafe_patterns(self, code: str, language: str) -> list[dict[str, Any]]:
        """Detect unsafe imports and dangerous function usage."""
        factors: list[dict[str, Any]] = []
        normalized = "python" if language in ("python", "py") else language

        unsafe_list = self._unsafe_imports.get(normalized, [])
        for module in unsafe_list:
            import_pat = re.compile(
                rf"\b(?:import\s+{re.escape(module)}|from\s+{re.escape(module)}\s+import)\b"
            )
            for m in import_pat.finditer(code):
                ln = code[: m.start()].count("\n") + 1
                factors.append({"category": "unsafe_pattern", "severity": "medium",
                                "description": f"Unsafe import '{module}' on line {ln}",
                                "pattern": module, "line": ln,
                                "matched_text": _truncate(m.group(), 80), "weight": 15.0})

        # Additional unsafe patterns: path traversal and insecure crypto
        extra: list[tuple[str, re.Pattern[str], str, float]] = [
            ("path_traversal", re.compile(r"""(?:\.\./|\.\.\\)"""),
             "Path traversal pattern", 12.0),
            ("insecure_crypto", re.compile(
                r"""\b(?:hashlib\.md5|hashlib\.sha1|MD5\.new|SHA\.new)\s*\(""", re.IGNORECASE),
             "Insecure hash algorithm", 10.0),
        ]
        for pat_name, regex, desc, weight in extra:
            for m in regex.finditer(code):
                ln = code[: m.start()].count("\n") + 1
                factors.append({"category": "unsafe_pattern", "severity": "medium",
                                "description": f"{desc} on line {ln}",
                                "pattern": pat_name, "line": ln,
                                "matched_text": _truncate(m.group(), 80), "weight": weight})

        return factors

        return factors

    # ------------------------------------------------------------------
    # Quality checks
    # ------------------------------------------------------------------

    def _check_quality(self, code: str) -> list[dict[str, Any]]:
        """Check for general code-quality issues in the suggestion."""
        factors: list[dict[str, Any]] = []
        lines = code.splitlines()

        for idx, line in enumerate(lines, start=1):
            if len(line) > 200:
                factors.append({"category": "quality", "severity": "low", "line": idx,
                                "description": f"Line {idx} exceeds 200 chars ({len(line)})",
                                "pattern": "long_line", "weight": 2.0})
            stripped = line.lstrip()
            if stripped and (len(line) - len(stripped)) >= 16:
                factors.append({"category": "quality", "severity": "low", "line": idx,
                                "description": f"Deeply nested code on line {idx}",
                                "pattern": "deep_nesting", "weight": 2.0})

        # Regex-based quality checks
        _quality_patterns: list[tuple[str, re.Pattern[str], str, float]] = [
            ("todo_marker", re.compile(r"\b(TODO|FIXME|HACK|XXX)\b", re.IGNORECASE), "Unresolved marker", 1.0),
            ("bare_except", re.compile(r"\bexcept\s*:"), "Bare except clause", 3.0),
        ]
        for pattern_name, regex, desc_prefix, weight in _quality_patterns:
            for match in regex.finditer(code):
                ln = code[: match.start()].count("\n") + 1
                factors.append({"category": "quality", "severity": "low", "line": ln,
                                "description": f"{desc_prefix} on line {ln}",
                                "pattern": pattern_name, "weight": weight})

        return factors

    # ------------------------------------------------------------------
    # Score calculation
    # ------------------------------------------------------------------

    def _calculate_risk_score(
        self, factors: list[dict[str, Any]]
    ) -> tuple[RiskLevel, float, float]:
        """Derive risk level, confidence, and numeric score from risk factors.

        Returns (risk_level, confidence, overall_score).
        """
        if not factors:
            return RiskLevel.SAFE, 1.0, 0.0

        total_weight = sum(f.get("weight", 1.0) for f in factors)
        score = min(total_weight, 100.0)

        severity_counts: dict[str, int] = Counter(
            f.get("severity", "low") for f in factors
        )

        if severity_counts.get("critical", 0) > 0 or score >= 80:
            risk_level = RiskLevel.CRITICAL
        elif severity_counts.get("high", 0) > 0 or score >= 50:
            risk_level = RiskLevel.HIGH
        elif severity_counts.get("medium", 0) > 0 or score >= 25:
            risk_level = RiskLevel.MEDIUM
        elif score >= 5:
            risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.SAFE

        # Confidence increases with more factors confirming the assessment
        confidence = min(0.5 + len(factors) * 0.1, 1.0)

        return risk_level, confidence, score


# =============================================================================
# Code sanitizer
# =============================================================================


class CodeSanitizer:
    """Sanitize dangerous code patterns while preserving functionality.

    Applies targeted transformations to remove or neutralise risky constructs
    without changing the surrounding logic.
    """

    def __init__(self) -> None:
        self._secret_patterns = _SECRET_PATTERNS
        self._placeholder = '"<REDACTED>"'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sanitize(
        self,
        code: str,
        risk_assessment: RiskAssessment,
        language: str = "python",
    ) -> tuple[str, list[SanitizationAction]]:
        """Apply all relevant sanitizations and return (sanitized_code, actions)."""
        actions: list[SanitizationAction] = []
        current_code = code

        has_secrets = any(
            f.get("category") == "secret" for f in risk_assessment.risk_factors
        )
        has_injection = any(
            f.get("category") == "injection" for f in risk_assessment.risk_factors
        )
        has_unsafe = any(
            f.get("category") == "unsafe_pattern"
            and f.get("pattern") not in ("path_traversal", "insecure_crypto")
            for f in risk_assessment.risk_factors
        )

        if has_secrets:
            current_code, secret_actions = self._remove_secrets(current_code)
            actions.extend(secret_actions)

        if has_injection:
            current_code, injection_actions = self._escape_injection(
                current_code, language
            )
            actions.extend(injection_actions)

        if has_unsafe:
            current_code, import_actions = self._fix_unsafe_imports(current_code)
            actions.extend(import_actions)

        logger.debug(
            "code_sanitized",
            action_count=len(actions),
            code_changed=current_code != code,
        )

        return current_code, actions

    # ------------------------------------------------------------------
    # Secret removal
    # ------------------------------------------------------------------

    def _remove_secrets(
        self, code: str, line_offset: int = 0
    ) -> tuple[str, list[SanitizationAction]]:
        """Replace detected secrets with redacted placeholders."""
        actions: list[SanitizationAction] = []
        result = code
        changed = True
        while changed:
            changed = False
            for label, pattern in self._secret_patterns:
                match = pattern.search(result)
                if match:
                    snippet = match.group()
                    ln = result[: match.start()].count("\n") + 1 + line_offset
                    replacement = pattern.sub(self._placeholder, snippet)
                    result = result[: match.start()] + replacement + result[match.end() :]
                    actions.append(SanitizationAction(
                        sanitization_type=SanitizationType.REMOVE_SECRETS,
                        original_code=_redact(snippet), sanitized_code=replacement,
                        description=f"Removed {label} and replaced with placeholder", line=ln))
                    changed = True
                    break
        return result, actions

    # ------------------------------------------------------------------
    # Injection escaping
    # ------------------------------------------------------------------

    def _escape_injection(
        self, code: str, language: str, line_offset: int = 0
    ) -> tuple[str, list[SanitizationAction]]:
        """Replace injection-prone patterns with parameterised equivalents."""
        actions: list[SanitizationAction] = []
        result = code

        fstring_sql = re.compile(
            r"""((?:cursor\.execute|\.execute)\s*\(\s*)f(['"])(.*?)\2""", re.DOTALL)
        for match in fstring_sql.finditer(result):
            ln = result[: match.start()].count("\n") + 1 + line_offset
            original = match.group()
            safe = f"{match.group(1)}{match.group(2)}/* SANITIZED: use parameterised query */{match.group(2)}"
            result = result.replace(original, safe, 1)
            actions.append(SanitizationAction(
                sanitization_type=SanitizationType.ESCAPE_INJECTION,
                original_code=_truncate(original, 120), sanitized_code=_truncate(safe, 120),
                description="Replaced f-string SQL with parameterised query placeholder", line=ln))

        for func_name in ("eval", "exec"):
            for match in re.compile(rf"\b{func_name}\s*\(").finditer(result):
                ln = result[: match.start()].count("\n") + 1 + line_offset
                original = match.group()
                safe = f"# SANITIZED: {func_name} removed for safety\n# {func_name}("
                result = result.replace(original, safe, 1)
                actions.append(SanitizationAction(
                    sanitization_type=SanitizationType.ESCAPE_INJECTION,
                    original_code=original, sanitized_code=safe,
                    description=f"Commented out unsafe {func_name}() call", line=ln))

        return result, actions

    # ------------------------------------------------------------------
    # Unsafe import fixing
    # ------------------------------------------------------------------

    def _fix_unsafe_imports(
        self, code: str, line_offset: int = 0
    ) -> tuple[str, list[SanitizationAction]]:
        """Comment out or replace unsafe imports with safe alternatives."""
        actions: list[SanitizationAction] = []
        result = code
        replacements = {"os.system": "subprocess.run", "pickle": "json",
                        "marshal": "json", "commands": "subprocess"}

        for unsafe_mod, safe_mod in replacements.items():
            import_re = re.compile(
                rf"^(\s*)(import\s+{re.escape(unsafe_mod)}|from\s+{re.escape(unsafe_mod)}\s+import\s+\w+)",
                re.MULTILINE)
            for match in import_re.finditer(result):
                ln = result[: match.start()].count("\n") + 1 + line_offset
                original_line = match.group()
                indent = match.group(1)
                safe_line = f"{indent}# SANITIZED: {unsafe_mod} replaced with {safe_mod}\n{indent}import {safe_mod}"
                result = result.replace(original_line, safe_line, 1)
                actions.append(SanitizationAction(
                    sanitization_type=SanitizationType.REMOVE_UNSAFE_IMPORT,
                    original_code=original_line.strip(), sanitized_code=safe_line.strip(),
                    description=f"Replaced unsafe '{unsafe_mod}' import with '{safe_mod}'", line=ln))

        return result, actions


# =============================================================================
# Main firewall engine
# =============================================================================


_DEFAULT_POLICY = FirewallPolicy(
    id="default",
    name="Default Firewall Policy",
    description="Blocks critical and high-risk suggestions; warns on medium risk.",
    risk_threshold=RiskLevel.HIGH,
    action_on_block=FirewallAction.BLOCK,
    require_review_above=RiskLevel.MEDIUM,
)


class AICodeFirewall:
    """Main firewall engine that intercepts, analyses, and controls AI code suggestions.

    Combines risk analysis, policy evaluation, and optional sanitization into a
    single ``intercept`` call that returns a :class:`FirewallDecision`.
    """

    def __init__(self, policy: FirewallPolicy | None = None) -> None:
        self._policy = policy or _DEFAULT_POLICY
        self._analyzer = SuggestionRiskAnalyzer()
        self._sanitizer = CodeSanitizer()
        self._decisions: dict[str, FirewallDecision] = {}
        self._override_counts: dict[str, int] = {}
        self._processing_times: list[float] = []
        self._action_counts: Counter[str] = Counter()

        logger.info(
            "firewall_initialized",
            policy_id=self._policy.id,
            policy_name=self._policy.name,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def intercept(self, interception: SuggestionInterception) -> FirewallDecision:
        """Analyse an intercepted suggestion and return a firewall decision."""
        start = time.monotonic()

        risk = self._analyzer.analyze(interception)
        action = self._apply_policy(risk, self._policy)

        sanitizations: list[SanitizationAction] = []
        modified_code: str | None = None

        if action == FirewallAction.SANITIZE:
            modified_code, sanitizations = self._sanitizer.sanitize(
                interception.code,
                risk,
                language=interception.language,
            )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        override_available = (
            self._policy.allow_override
            and self._override_counts.get(interception.id, 0)
            < self._policy.max_override_count
        )

        reason = self._build_reason(action, risk)

        decision = FirewallDecision(
            interception_id=interception.id,
            action=action,
            risk_assessment=risk,
            sanitizations=sanitizations,
            original_code=interception.code,
            modified_code=modified_code,
            override_available=override_available,
            reason=reason,
            processing_time_ms=elapsed_ms,
        )

        self._decisions[interception.id] = decision
        self._action_counts[action.value] += 1
        self._processing_times.append(elapsed_ms)

        logger.info(
            "firewall_decision",
            interception_id=interception.id,
            action=action.value,
            risk_level=risk.risk_level.value,
            score=round(risk.overall_score, 2),
            processing_ms=round(elapsed_ms, 2),
            source=interception.source.value,
        )

        return decision

    def override(self, decision_id: str, reason: str) -> bool:
        """Attempt to override a blocked/warned decision. Returns success."""
        decision = self._decisions.get(decision_id)
        if decision is None:
            logger.warning("override_not_found", decision_id=decision_id)
            return False
        current_count = self._override_counts.get(decision_id, 0)
        if not decision.override_available or current_count >= self._policy.max_override_count:
            logger.warning("override_denied", decision_id=decision_id)
            return False

        self._override_counts[decision_id] = current_count + 1
        decision.action = FirewallAction.ALLOW
        decision.reason = f"Override accepted: {reason}"
        decision.override_available = (
            self._override_counts[decision_id] < self._policy.max_override_count
        )
        self._action_counts["overrides"] += 1
        logger.info("firewall_override", decision_id=decision_id, override_reason=reason)
        return True

    def get_metrics(self) -> FirewallMetrics:
        """Return aggregated firewall metrics."""
        avg_time = (sum(self._processing_times) / len(self._processing_times)
                    if self._processing_times else 0.0)
        factor_counter: Counter[str] = Counter()
        for decision in self._decisions.values():
            for f in decision.risk_assessment.risk_factors:
                factor_counter[f.get("pattern", "unknown")] += 1
        top_factors = [{"pattern": p, "count": c} for p, c in factor_counter.most_common(10)]

        return FirewallMetrics(
            total_interceptions=len(self._decisions),
            allowed=self._action_counts.get(FirewallAction.ALLOW.value, 0),
            warned=self._action_counts.get(FirewallAction.WARN.value, 0),
            blocked=self._action_counts.get(FirewallAction.BLOCK.value, 0),
            sanitized=self._action_counts.get(FirewallAction.SANITIZE.value, 0),
            overrides=self._action_counts.get("overrides", 0),
            avg_processing_time_ms=avg_time, top_risk_factors=top_factors,
        )

    def update_policy(self, policy: FirewallPolicy) -> None:
        """Replace the active policy with *policy*."""
        logger.info("policy_updated", old_policy=self._policy.id, new_policy=policy.id)
        self._policy = policy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_policy(
        self, risk: RiskAssessment, policy: FirewallPolicy
    ) -> FirewallAction:
        """Determine the firewall action based on risk assessment and policy."""
        if not policy.enabled:
            return FirewallAction.ALLOW

        # Check explicit blocked patterns
        for bp in policy.blocked_patterns:
            try:
                if re.search(bp, risk.risk_factors.__repr__()):
                    return policy.action_on_block
            except re.error:
                logger.warning("invalid_blocked_pattern", pattern=bp)

        # Map risk level to action
        if risk.risk_level >= policy.risk_threshold:
            # Try to sanitize if the action would be block
            if policy.action_on_block == FirewallAction.BLOCK:
                has_fixable = any(
                    f.get("category") in ("secret", "injection", "unsafe_pattern")
                    for f in risk.risk_factors
                )
                if has_fixable:
                    return FirewallAction.SANITIZE
            return policy.action_on_block

        if risk.risk_level >= policy.require_review_above:
            return FirewallAction.WARN

        return FirewallAction.ALLOW

    def _build_reason(self, action: FirewallAction, risk: RiskAssessment) -> str:
        """Build a human-readable reason string for the decision."""
        if action == FirewallAction.ALLOW:
            return "Suggestion passed all checks."
        parts: list[str] = []
        if risk.security_issues:
            parts.append(f"{len(risk.security_issues)} security issue(s)")
        if risk.quality_issues:
            parts.append(f"{len(risk.quality_issues)} quality issue(s)")
        issues_text = " and ".join(parts) if parts else "risk factors"
        verb = {FirewallAction.WARN: "flagged for review", FirewallAction.BLOCK: "blocked",
                FirewallAction.SANITIZE: "sanitized",
                FirewallAction.QUARANTINE: "quarantined for manual review"}.get(action, action.value)
        return f"Suggestion {verb} due to {issues_text} (risk={risk.risk_level.value}, score={risk.overall_score:.1f})."


# =============================================================================
# Utility helpers
# =============================================================================


def _redact(text: str, visible_chars: int = 6) -> str:
    """Partially redact a string, keeping only the first few characters."""
    if len(text) <= visible_chars:
        return "***"
    return text[:visible_chars] + "***"


def _truncate(text: str, max_length: int = 80) -> str:
    """Truncate text to *max_length* characters, appending '…' if trimmed."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "…"
