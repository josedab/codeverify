"""Verification-as-Code Infrastructure.

Terraform/Pulumi-style declarative verification policies with GitOps workflows.
Version-control verification rules, detect policy drift, and enforce standards
at scale across an organization.

Features:
- HCL-like DSL for verification policies
- Git-tracked policy repo with auto-apply
- Policy drift detection and alerting
- Compliance templates (SOC2, HIPAA, PCI-DSS)
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class PolicySeverity(str, Enum):
    """Enforcement severity of a policy rule."""

    BLOCK = "block"
    WARN = "warn"
    INFO = "info"


class PolicyScope(str, Enum):
    """Scope of a policy."""

    ORGANIZATION = "organization"
    TEAM = "team"
    REPOSITORY = "repository"
    FILE_PATTERN = "file_pattern"


class DriftStatus(str, Enum):
    """Status of policy drift detection."""

    IN_SYNC = "in_sync"
    DRIFTED = "drifted"
    UNKNOWN = "unknown"
    OVERRIDE = "override"


@dataclass
class PolicyRule:
    """A single verification policy rule."""

    id: str = ""
    name: str = ""
    check: str = ""
    severity: PolicySeverity = PolicySeverity.WARN
    message: str = ""
    enabled: bool = True
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "check": self.check,
            "severity": self.severity.value,
            "message": self.message,
            "enabled": self.enabled,
            "tags": self.tags,
        }


@dataclass
class PolicyModule:
    """A named module of policy rules (e.g., SOC2, HIPAA)."""

    name: str = ""
    version: str = "1.0.0"
    description: str = ""
    rules: list[PolicyRule] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    scope: PolicyScope = PolicyScope.ORGANIZATION

    @property
    def rule_count(self) -> int:
        return len(self.rules)

    @property
    def enabled_count(self) -> int:
        return sum(1 for r in self.rules if r.enabled)

    def fingerprint(self) -> str:
        content = f"{self.name}:{self.version}:" + ":".join(
            f"{r.id}={r.severity.value}" for r in self.rules
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "rules": [r.to_dict() for r in self.rules],
            "variables": self.variables,
            "scope": self.scope.value,
        }


@dataclass
class PolicyViolation:
    """A violation of a policy rule."""

    rule_id: str = ""
    rule_name: str = ""
    severity: PolicySeverity = PolicySeverity.WARN
    file_path: str = ""
    line: int = 0
    message: str = ""
    repository: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "file_path": self.file_path,
            "line": self.line,
            "message": self.message,
            "repository": self.repository,
        }


@dataclass
class DriftReport:
    """Report of policy drift between central and local configs."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    repository: str = ""
    central_fingerprint: str = ""
    local_fingerprint: str = ""
    status: DriftStatus = DriftStatus.UNKNOWN
    differences: list[dict[str, Any]] = field(default_factory=list)
    checked_at: float = field(default_factory=time.time)

    @property
    def is_drifted(self) -> bool:
        return self.status == DriftStatus.DRIFTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "repository": self.repository,
            "status": self.status.value,
            "differences_count": len(self.differences),
            "checked_at": self.checked_at,
        }


# Built-in compliance templates
_COMPLIANCE_TEMPLATES: dict[str, PolicyModule] = {
    "soc2": PolicyModule(
        name="soc2",
        version="1.0.0",
        description="SOC 2 Type II compliance requirements",
        rules=[
            PolicyRule(
                id="soc2-auth", name="Authentication required",
                check="null_safety", severity=PolicySeverity.BLOCK,
                message="All authentication paths must be null-safe",
                tags=["security", "soc2"],
            ),
            PolicyRule(
                id="soc2-audit", name="Audit logging",
                check="audit_logging", severity=PolicySeverity.BLOCK,
                message="Sensitive operations must have audit logging",
                tags=["compliance", "soc2"],
            ),
            PolicyRule(
                id="soc2-crypto", name="Strong cryptography",
                check="crypto_strength", severity=PolicySeverity.BLOCK,
                message="Use AES-256 or stronger encryption",
                tags=["security", "soc2"],
            ),
            PolicyRule(
                id="soc2-access", name="Access control",
                check="access_control", severity=PolicySeverity.WARN,
                message="Implement principle of least privilege",
                tags=["security", "soc2"],
            ),
        ],
        scope=PolicyScope.ORGANIZATION,
    ),
    "hipaa": PolicyModule(
        name="hipaa",
        version="1.0.0",
        description="HIPAA security requirements for healthcare",
        rules=[
            PolicyRule(
                id="hipaa-phi", name="PHI protection",
                check="data_classification", severity=PolicySeverity.BLOCK,
                message="Protected Health Information must be encrypted at rest and in transit",
                tags=["healthcare", "hipaa"],
            ),
            PolicyRule(
                id="hipaa-access-log", name="Access logging",
                check="audit_logging", severity=PolicySeverity.BLOCK,
                message="All PHI access must be logged with user identity",
                tags=["compliance", "hipaa"],
            ),
            PolicyRule(
                id="hipaa-retention", name="Data retention",
                check="data_retention", severity=PolicySeverity.WARN,
                message="PHI retention must comply with 6-year minimum",
                tags=["compliance", "hipaa"],
            ),
        ],
        scope=PolicyScope.ORGANIZATION,
    ),
    "pci_dss": PolicyModule(
        name="pci_dss",
        version="1.0.0",
        description="PCI DSS requirements for payment processing",
        rules=[
            PolicyRule(
                id="pci-card-data", name="Card data protection",
                check="sensitive_data", severity=PolicySeverity.BLOCK,
                message="Card numbers must never be stored in plaintext",
                tags=["payment", "pci"],
            ),
            PolicyRule(
                id="pci-input-val", name="Input validation",
                check="input_validation", severity=PolicySeverity.BLOCK,
                message="All payment inputs must be validated and sanitized",
                tags=["security", "pci"],
            ),
            PolicyRule(
                id="pci-crypto", name="Strong cryptography",
                check="crypto_strength", severity=PolicySeverity.BLOCK,
                message="Use industry-standard encryption for card data",
                tags=["security", "pci"],
            ),
        ],
        scope=PolicyScope.ORGANIZATION,
    ),
}


class PolicyDSLParser:
    """Parses the CodeVerify policy DSL into PolicyModule objects.

    DSL syntax:
        module "name" {
          version = "1.0.0"
          scope = "organization"

          rule "rule-id" {
            check = "null_safety"
            severity = "block"
            message = "Must be null-safe"
            tags = ["security"]
          }

          variable "max_complexity" {
            default = 10
          }
        }
    """

    def parse(self, dsl_text: str) -> list[PolicyModule]:
        """Parse DSL text into policy modules."""
        modules: list[PolicyModule] = []
        module_blocks = re.finditer(
            r'module\s+"([^"]+)"\s*\{(.*?)\n\}',
            dsl_text,
            re.DOTALL,
        )

        for match in module_blocks:
            name = match.group(1)
            body = match.group(2)
            module = self._parse_module(name, body)
            modules.append(module)

        return modules

    def _parse_module(self, name: str, body: str) -> PolicyModule:
        module = PolicyModule(name=name)

        # Parse simple attributes
        version_match = re.search(r'version\s*=\s*"([^"]+)"', body)
        if version_match:
            module.version = version_match.group(1)

        scope_match = re.search(r'scope\s*=\s*"([^"]+)"', body)
        if scope_match:
            try:
                module.scope = PolicyScope(scope_match.group(1))
            except ValueError:
                pass

        desc_match = re.search(r'description\s*=\s*"([^"]+)"', body)
        if desc_match:
            module.description = desc_match.group(1)

        # Parse rules
        rule_blocks = re.finditer(
            r'rule\s+"([^"]+)"\s*\{(.*?)\}',
            body,
            re.DOTALL,
        )
        for rule_match in rule_blocks:
            rule_id = rule_match.group(1)
            rule_body = rule_match.group(2)
            module.rules.append(self._parse_rule(rule_id, rule_body))

        # Parse variables
        var_blocks = re.finditer(
            r'variable\s+"([^"]+)"\s*\{(.*?)\}',
            body,
            re.DOTALL,
        )
        for var_match in var_blocks:
            var_name = var_match.group(1)
            var_body = var_match.group(2)
            default_match = re.search(r'default\s*=\s*(\S+)', var_body)
            if default_match:
                val = default_match.group(1).strip('"')
                try:
                    val = int(val)
                except (ValueError, TypeError):
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        pass
                module.variables[var_name] = val

        return module

    def _parse_rule(self, rule_id: str, body: str) -> PolicyRule:
        rule = PolicyRule(id=rule_id, name=rule_id)

        check_match = re.search(r'check\s*=\s*"([^"]+)"', body)
        if check_match:
            rule.check = check_match.group(1)

        severity_match = re.search(r'severity\s*=\s*"([^"]+)"', body)
        if severity_match:
            try:
                rule.severity = PolicySeverity(severity_match.group(1))
            except ValueError:
                pass

        msg_match = re.search(r'message\s*=\s*"([^"]+)"', body)
        if msg_match:
            rule.message = msg_match.group(1)

        name_match = re.search(r'name\s*=\s*"([^"]+)"', body)
        if name_match:
            rule.name = name_match.group(1)

        tags_match = re.search(r'tags\s*=\s*\[([^\]]*)\]', body)
        if tags_match:
            rule.tags = [t.strip().strip('"') for t in tags_match.group(1).split(",") if t.strip()]

        return rule


class PolicyEngine:
    """Evaluates code against policy modules."""

    def __init__(self) -> None:
        self._modules: dict[str, PolicyModule] = {}
        self._parser = PolicyDSLParser()

    def load_template(self, template_name: str) -> PolicyModule | None:
        """Load a built-in compliance template."""
        template = _COMPLIANCE_TEMPLATES.get(template_name)
        if template:
            self._modules[template.name] = template
        return template

    def load_dsl(self, dsl_text: str) -> list[PolicyModule]:
        """Load policies from DSL text."""
        modules = self._parser.parse(dsl_text)
        for m in modules:
            self._modules[m.name] = m
        return modules

    def add_module(self, module: PolicyModule) -> None:
        self._modules[module.name] = module

    def get_module(self, name: str) -> PolicyModule | None:
        return self._modules.get(name)

    @property
    def modules(self) -> list[PolicyModule]:
        return list(self._modules.values())

    def get_all_rules(self) -> list[PolicyRule]:
        """Get all enabled rules across all modules."""
        rules = []
        for module in self._modules.values():
            for rule in module.rules:
                if rule.enabled:
                    rules.append(rule)
        return rules

    def evaluate(
        self,
        source: str,
        file_path: str = "",
        repository: str = "",
    ) -> list[PolicyViolation]:
        """Evaluate source code against all loaded policies."""
        violations: list[PolicyViolation] = []
        lines = source.split("\n")

        for rule in self.get_all_rules():
            rule_violations = self._check_rule(rule, source, lines, file_path, repository)
            violations.extend(rule_violations)

        return violations

    def _check_rule(
        self,
        rule: PolicyRule,
        source: str,
        lines: list[str],
        file_path: str,
        repository: str,
    ) -> list[PolicyViolation]:
        """Check a single rule against source code."""
        violations: list[PolicyViolation] = []

        # Map check types to patterns
        check_patterns: dict[str, list[tuple[str, str]]] = {
            "null_safety": [
                (r"\.(\w+)\s*(?!\s*is\s+not\s+None)", "Potential null dereference without check"),
            ],
            "input_validation": [
                (r"request\.\w+\s*(?!\s*\.strip\()", "User input not validated"),
            ],
            "sensitive_data": [
                (r"(?:password|secret|api_key|token)\s*=\s*['\"]", "Hardcoded sensitive data"),
            ],
            "crypto_strength": [
                (r"\bmd5\b|\bsha1\b", "Weak cryptographic algorithm"),
                (r"DES\b|RC4\b", "Weak encryption algorithm"),
            ],
            "audit_logging": [],  # Structural check, not pattern-based
            "data_classification": [],
            "data_retention": [],
            "access_control": [],
        }

        patterns = check_patterns.get(rule.check, [])
        for pattern, desc in patterns:
            for i, line in enumerate(lines, 1):
                if line.strip().startswith("#") or line.strip().startswith("//"):
                    continue
                try:
                    if re.search(pattern, line):
                        violations.append(PolicyViolation(
                            rule_id=rule.id,
                            rule_name=rule.name,
                            severity=rule.severity,
                            file_path=file_path,
                            line=i,
                            message=f"{rule.message}: {desc}",
                            repository=repository,
                        ))
                except re.error:
                    continue

        return violations

    def check_drift(
        self,
        central_module: PolicyModule,
        local_module: PolicyModule,
        repository: str = "",
    ) -> DriftReport:
        """Check for policy drift between central and local configs."""
        central_fp = central_module.fingerprint()
        local_fp = local_module.fingerprint()

        differences: list[dict[str, Any]] = []

        if central_fp == local_fp:
            return DriftReport(
                repository=repository,
                central_fingerprint=central_fp,
                local_fingerprint=local_fp,
                status=DriftStatus.IN_SYNC,
            )

        # Find specific differences
        central_rules = {r.id: r for r in central_module.rules}
        local_rules = {r.id: r for r in local_module.rules}

        for rule_id, central_rule in central_rules.items():
            if rule_id not in local_rules:
                differences.append({
                    "type": "rule_missing_locally",
                    "rule_id": rule_id,
                    "central_severity": central_rule.severity.value,
                })
            else:
                local_rule = local_rules[rule_id]
                if central_rule.severity != local_rule.severity:
                    differences.append({
                        "type": "severity_changed",
                        "rule_id": rule_id,
                        "central": central_rule.severity.value,
                        "local": local_rule.severity.value,
                    })
                if central_rule.enabled != local_rule.enabled:
                    differences.append({
                        "type": "enabled_changed",
                        "rule_id": rule_id,
                        "central": central_rule.enabled,
                        "local": local_rule.enabled,
                    })

        for rule_id in local_rules:
            if rule_id not in central_rules:
                differences.append({
                    "type": "rule_added_locally",
                    "rule_id": rule_id,
                })

        return DriftReport(
            repository=repository,
            central_fingerprint=central_fp,
            local_fingerprint=local_fp,
            status=DriftStatus.DRIFTED,
            differences=differences,
        )


# Singleton
_policy_engine_instance: PolicyEngine | None = None


def get_policy_engine() -> PolicyEngine:
    global _policy_engine_instance
    if _policy_engine_instance is None:
        _policy_engine_instance = PolicyEngine()
    return _policy_engine_instance


def reset_policy_engine() -> None:
    global _policy_engine_instance
    _policy_engine_instance = None
