"""Verification Policy-as-Code Engine (OPA-style).

Defines and evaluates verification policies that control how CodeVerify
behaves for different files, modules, and repositories.  Policies are
declarative rules with conditions, actions, and scopes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PolicyAction(str, Enum):
    """Action to take when a policy rule matches."""
    ALLOW = "allow"
    DENY = "deny"
    WARN = "warn"


class PolicyScope(str, Enum):
    """Scope at which a policy rule is evaluated."""
    FILE = "file"
    FUNCTION = "function"
    MODULE = "module"
    REPOSITORY = "repository"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

VALID_OPERATORS = frozenset(
    {"equals", "not_equals", "contains", "matches", "greater_than", "less_than", "in"}
)


@dataclass
class PolicyCondition:
    """A single condition within a policy rule."""
    field: str
    operator: str
    value: Any

    def __post_init__(self) -> None:
        if self.operator not in VALID_OPERATORS:
            raise ValueError(
                f"Invalid operator '{self.operator}'. "
                f"Must be one of: {', '.join(sorted(VALID_OPERATORS))}"
            )


@dataclass
class PolicyRule:
    """A single policy rule composed of conditions and an action."""
    id: str
    name: str
    description: str
    conditions: list[PolicyCondition]
    action: PolicyAction
    scope: PolicyScope
    verification_depth: str | None = None
    priority: int = 0
    enabled: bool = True


@dataclass
class PolicySet:
    """A named, versioned collection of policy rules."""
    name: str
    version: str
    rules: list[PolicyRule]
    description: str
    default_action: PolicyAction = PolicyAction.WARN


@dataclass
class PolicyEvaluationResult:
    """The outcome of evaluating a single policy rule."""
    rule_id: str
    rule_name: str
    action: PolicyAction
    matched: bool
    reason: str
    matched_conditions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PolicyEngine:
    """Evaluates policy rules against runtime context dictionaries."""

    def load_from_yaml(self, yaml_content: str) -> PolicySet:
        """Parse a YAML string into a *PolicySet*."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load policies from YAML. "
                "Install it with: pip install pyyaml"
            ) from exc
        data = yaml.safe_load(yaml_content)
        if not isinstance(data, dict):
            raise ValueError("YAML content must be a mapping at the top level")
        return self.load_from_dict(data)

    def load_from_dict(self, data: dict) -> PolicySet:
        """Build a *PolicySet* from a plain dictionary."""
        rules: list[PolicyRule] = []
        for rd in data.get("rules", []):
            conditions = [
                PolicyCondition(field=c["field"], operator=c["operator"], value=c["value"])
                for c in rd.get("conditions", [])
            ]
            try:
                action = PolicyAction(rd.get("action", "warn"))
            except ValueError:
                logger.warning("Unknown action, defaulting to warn", rule_id=rd.get("id"))
                action = PolicyAction.WARN
            try:
                scope = PolicyScope(rd.get("scope", "file"))
            except ValueError:
                scope = PolicyScope.FILE
            rules.append(PolicyRule(
                id=rd.get("id", ""), name=rd.get("name", ""),
                description=rd.get("description", ""), conditions=conditions,
                action=action, scope=scope,
                verification_depth=rd.get("verification_depth"),
                priority=int(rd.get("priority", 0)),
                enabled=bool(rd.get("enabled", True)),
            ))
        try:
            default_action = PolicyAction(data.get("default_action", "warn"))
        except ValueError:
            default_action = PolicyAction.WARN
        return PolicySet(
            name=data.get("name", "unnamed"), version=data.get("version", "0.0.0"),
            rules=rules, description=data.get("description", ""),
            default_action=default_action,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, policy_set: PolicySet, context: dict) -> list[PolicyEvaluationResult]:
        """Evaluate all enabled rules against *context* (descending priority)."""
        results: list[PolicyEvaluationResult] = []
        sorted_rules = sorted(policy_set.rules, key=lambda r: r.priority, reverse=True)
        for rule in sorted_rules:
            if not rule.enabled:
                continue
            matched_conds: list[str] = []
            all_matched = True
            for cond in rule.conditions:
                if self._match_condition(cond, context):
                    matched_conds.append(f"{cond.field} {cond.operator} {cond.value!r}")
                else:
                    all_matched = False
            if all_matched and rule.conditions:
                reason = f"Rule '{rule.name}' matched: {', '.join(matched_conds)}"
                results.append(PolicyEvaluationResult(
                    rule_id=rule.id, rule_name=rule.name, action=rule.action,
                    matched=True, reason=reason, matched_conditions=matched_conds,
                ))
                logger.debug("Policy rule matched", rule_id=rule.id, action=rule.action.value)
            else:
                results.append(PolicyEvaluationResult(
                    rule_id=rule.id, rule_name=rule.name, action=rule.action,
                    matched=False, reason=f"Rule '{rule.name}' did not match",
                ))
        return results

    def evaluate_file(
        self, policy_set: PolicySet, file_path: str,
        findings: list[dict[str, Any]], language: str | None = None,
    ) -> PolicyEvaluationResult:
        """Evaluate policies for a single file and return the top match."""
        severity_counts: dict[str, int] = {}
        max_severity = "info"
        sev_order = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        for f in findings:
            sev = str(f.get("severity", "info")).lower()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
            if sev_order.get(sev, 0) > sev_order.get(max_severity, 0):
                max_severity = sev
        context: dict[str, Any] = {
            "file_path": file_path,
            "finding_count": len(findings),
            "severity": max_severity,
            "severity_counts": severity_counts,
            "language": language or _guess_language(file_path),
            "is_security_file": _is_security_related(file_path),
            "is_test_file": _is_test_file(file_path),
            "is_generated": _is_generated_code(file_path),
            "is_config_file": _is_config_file(file_path),
            "is_migration_file": _is_migration_file(file_path),
            "is_documentation_file": _is_documentation_file(file_path),
        }
        results = self.evaluate(policy_set, context)
        matched = [r for r in results if r.matched]
        if matched:
            return matched[0]
        return PolicyEvaluationResult(
            rule_id="default", rule_name="default",
            action=policy_set.default_action, matched=False,
            reason="No policy rules matched; using default action",
        )

    def get_verification_depth(
        self, policy_set: PolicySet, file_path: str, context: dict,
    ) -> str:
        """Return the verification depth for *file_path* or ``"static"``."""
        enriched = dict(context)
        enriched.setdefault("file_path", file_path)
        results = self.evaluate(policy_set, enriched)
        for result in results:
            if not result.matched:
                continue
            for rule in policy_set.rules:
                if rule.id == result.rule_id and rule.verification_depth:
                    logger.debug("Verification depth resolved",
                                 file_path=file_path, depth=rule.verification_depth)
                    return rule.verification_depth
        return "static"

    # ------------------------------------------------------------------
    # Condition matching
    # ------------------------------------------------------------------

    def _match_condition(self, condition: PolicyCondition, context: dict) -> bool:
        """Evaluate a single condition against the context."""
        ctx_value = context.get(condition.field)
        if ctx_value is None:
            return False
        op = condition.operator
        expected = condition.value
        try:
            if op == "equals":
                return _coerce_compare(ctx_value, expected) == 0
            if op == "not_equals":
                return _coerce_compare(ctx_value, expected) != 0
            if op == "contains":
                return str(expected) in str(ctx_value)
            if op == "matches":
                return bool(re.search(str(expected), str(ctx_value)))
            if op == "greater_than":
                return float(ctx_value) > float(expected)
            if op == "less_than":
                return float(ctx_value) < float(expected)
            if op == "in":
                if isinstance(expected, list):
                    return ctx_value in expected
                if isinstance(expected, str):
                    return str(ctx_value) in [i.strip() for i in expected.split(",")]
                return False
        except (TypeError, ValueError) as exc:
            logger.warning("Condition evaluation error", field=condition.field, error=str(exc))
            return False
        return False


# ---------------------------------------------------------------------------
# Built-in policies
# ---------------------------------------------------------------------------

def _rule(id: str, name: str, desc: str, field: str, op: str, value: Any,
          action: PolicyAction, depth: str | None, priority: int) -> PolicyRule:
    """Shorthand factory for built-in policy rules."""
    return PolicyRule(
        id=id, name=name, description=desc,
        conditions=[PolicyCondition(field=field, operator=op, value=value)],
        action=action, scope=PolicyScope.FILE,
        verification_depth=depth, priority=priority,
    )


BUILT_IN_POLICIES: list[PolicyRule] = [
    _rule("auth-files-require-formal", "Auth files require formal verification",
          "Auth/security files must undergo formal verification.",
          "is_security_file", "equals", True, PolicyAction.DENY, "formal", 100),
    _rule("test-files-allow-warnings", "Test files only warn",
          "Test files produce warnings, never block.",
          "is_test_file", "equals", True, PolicyAction.WARN, "pattern", 90),
    _rule("critical-findings-block", "Critical findings block PR",
          "Any file with critical-severity findings blocks the PR.",
          "severity", "equals", "critical", PolicyAction.DENY, "full", 95),
    _rule("generated-code-strict", "AI-generated code gets strict verification",
          "AI-generated code requires stricter verification.",
          "is_generated", "equals", True, PolicyAction.DENY, "ai", 85),
    _rule("config-files-skip", "Skip verification for config files",
          "Configuration files are allowed without deep verification.",
          "is_config_file", "equals", True, PolicyAction.ALLOW, "pattern", 80),
    _rule("api-endpoints-security", "API files require security scan",
          "API endpoint files must pass a security-focused verification.",
          "file_path", "matches",
          r"(routes|endpoints|controllers|views|api)[/\\]",
          PolicyAction.DENY, "ai", 75),
    _rule("migration-files-skip", "Skip migration files",
          "Database migration files are allowed without verification.",
          "is_migration_file", "equals", True, PolicyAction.ALLOW, "pattern", 70),
    _rule("high-complexity-formal", "High complexity files get formal verification",
          "Files exceeding the finding-count threshold require formal methods.",
          "finding_count", "greater_than", 10, PolicyAction.DENY, "formal", 65),
    _rule("dependency-changes-strict", "Dependency lock file changes need scrutiny",
          "Lock file changes require strict verification for supply-chain safety.",
          "file_path", "matches",
          r"(package-lock\.json|yarn\.lock|Pipfile\.lock|poetry\.lock|"
          r"Gemfile\.lock|go\.sum|Cargo\.lock|pnpm-lock\.yaml)",
          PolicyAction.DENY, "static", 60),
    _rule("documentation-skip", "Skip documentation files",
          "Documentation files do not require verification.",
          "is_documentation_file", "equals", True, PolicyAction.ALLOW, None, 55),
    _rule("high-severity-warn", "High severity findings warn",
          "High-severity findings produce a warning for reviewer triage.",
          "severity", "equals", "high", PolicyAction.WARN, "ai", 50),
    _rule("low-severity-allow", "Low severity findings allowed",
          "Low/info severity findings are allowed to proceed.",
          "severity", "in", ["low", "info"], PolicyAction.ALLOW, "pattern", 10),
]


def get_default_policy_set() -> PolicySet:
    """Return a *PolicySet* populated with all built-in rules."""
    return PolicySet(
        name="codeverify-defaults", version="1.0.0",
        rules=list(BUILT_IN_POLICIES),
        description="Default CodeVerify verification policies",
        default_action=PolicyAction.WARN,
    )


# ---------------------------------------------------------------------------
# YAML parsing convenience
# ---------------------------------------------------------------------------

def parse_policy_yaml(yaml_content: str) -> PolicySet:
    """Module-level convenience wrapper around *PolicyEngine.load_from_yaml*."""
    return PolicyEngine().load_from_yaml(yaml_content)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _coerce_compare(left: Any, right: Any) -> int:
    """Compare two values with best-effort type coercion (-1 / 0 / 1)."""
    if isinstance(right, bool) or isinstance(left, bool):
        lb, rb = bool(left), bool(right)
        return 0 if lb == rb else (-1 if lb < rb else 1)
    try:
        lf, rf = float(left), float(right)
        return 0 if lf == rf else (-1 if lf < rf else 1)
    except (TypeError, ValueError):
        pass
    ls, rs = str(left).lower(), str(right).lower()
    return 0 if ls == rs else (-1 if ls < rs else 1)


def _is_security_related(file_path: str) -> bool:
    """Heuristic check for security / auth files."""
    patterns = (r"auth", r"login", r"password", r"credential", r"security",
                r"permission", r"access.?control", r"oauth", r"token",
                r"crypto", r"encrypt", r"jwt")
    lower = file_path.lower()
    return any(re.search(p, lower) for p in patterns)


def _is_test_file(file_path: str) -> bool:
    """Heuristic check for test files."""
    lower = file_path.lower()
    return bool(
        re.search(r"(^|[/\\])(tests?|__tests__|spec)[/\\]", lower)
        or re.search(r"[._](test|spec)\.\w+$", lower)
        or re.search(r"test_[^/\\]+\.py$", lower)
    )


def _is_generated_code(file_path: str) -> bool:
    """Heuristic check for generated code."""
    lower = file_path.lower()
    return bool(
        re.search(r"(generated|auto.?gen|\.g\.)", lower)
        or re.search(r"(pb2|_pb2_grpc)\.py$", lower)
        or re.search(r"\.generated\.\w+$", lower)
    )


def _is_config_file(file_path: str) -> bool:
    """Heuristic check for configuration files."""
    lower = file_path.lower()
    return bool(
        re.search(r"\.(ya?ml|toml|ini|cfg|conf|json|env|properties)$", lower)
        or re.search(r"(^|[/\\])\.(eslintrc|prettierrc|babelrc)", lower)
        or re.search(r"(Makefile|Dockerfile|Procfile)$", file_path)
    )


def _is_migration_file(file_path: str) -> bool:
    """Heuristic check for database migration files."""
    lower = file_path.lower()
    return bool(
        re.search(r"(^|[/\\])migrations?[/\\]", lower)
        or re.search(r"(^|[/\\])alembic[/\\]versions[/\\]", lower)
        or re.search(r"\d{3,}_\w+\.py$", lower)
    )


def _is_documentation_file(file_path: str) -> bool:
    """Heuristic check for documentation files."""
    lower = file_path.lower()
    return bool(
        re.search(r"\.(md|rst|txt|adoc|rdoc)$", lower)
        or re.search(r"(^|[/\\])(docs?|documentation)[/\\]", lower)
        or re.search(r"(CHANGELOG|LICENSE|NOTICE|AUTHORS|CONTRIBUTORS)", file_path)
    )


_EXT_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "typescript", ".jsx": "javascript", ".java": "java",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".c": "c",
    ".cpp": "cpp", ".cs": "csharp", ".swift": "swift",
    ".kt": "kotlin", ".scala": "scala", ".php": "php",
    ".sh": "shell", ".sql": "sql",
}


def _guess_language(file_path: str) -> str:
    """Best-guess language identifier from the file extension."""
    lower = file_path.lower()
    for ext, lang in _EXT_LANGUAGE_MAP.items():
        if lower.endswith(ext):
            return lang
    return "unknown"
