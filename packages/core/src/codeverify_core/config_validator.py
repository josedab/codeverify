"""Configuration Validator.

Validates .codeverify.yml configuration files, provides helpful error
messages, supports default generation and version migration.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

CONFIG_SCHEMA: dict[str, Any] = {
    "version": {"type": "str", "required": True, "allowed": ["1", "2"]},
    "verification": {
        "type": "dict",
        "schema": {
            "enabled": {"type": "bool"},
            "timeout_seconds": {"type": "int", "min": 1, "max": 600},
            "checks": {
                "type": "list",
                "allowed_items": [
                    "null_safety",
                    "array_bounds",
                    "integer_overflow",
                    "division_by_zero",
                    "type_safety",
                ],
            },
        },
    },
    "ai": {
        "type": "dict",
        "schema": {
            "enabled": {"type": "bool"},
            "model": {"type": "str"},
            "temperature": {"type": "float", "min": 0.0, "max": 2.0},
        },
    },
    "languages": {
        "type": "list",
        "allowed_items": ["python", "typescript", "go", "java"],
    },
    "severity_thresholds": {
        "type": "dict",
        "schema": {
            "critical": {"type": "int", "min": 0},
            "high": {"type": "int", "min": 0},
            "medium": {"type": "int", "min": 0},
            "low": {"type": "int", "min": 0},
        },
    },
    "exclude_paths": {"type": "list"},
    "custom_rules": {
        "type": "list",
        "item_schema": {
            "id": {"type": "str", "required": True},
            "pattern": {"type": "str", "required": True},
            "severity": {
                "type": "str",
                "allowed": ["critical", "high", "medium", "low"],
            },
        },
    },
}

DEFAULT_CONFIG: dict[str, Any] = {
    "version": "2",
    "verification": {
        "enabled": True,
        "timeout_seconds": 30,
        "checks": ["null_safety", "array_bounds", "integer_overflow", "division_by_zero"],
    },
    "ai": {
        "enabled": True,
        "model": "gpt-4",
        "temperature": 0.1,
    },
    "languages": ["python", "typescript"],
    "severity_thresholds": {
        "critical": 0,
        "high": 0,
        "medium": 5,
        "low": 10,
    },
    "exclude_paths": ["node_modules/", ".venv/", "__pycache__/"],
}


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue."""

    path: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    suggestion: str = ""


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]


class ConfigValidator:
    """Validates .codeverify.yml configuration.

    Usage:
        validator = ConfigValidator()
        result = validator.validate(config_dict)
        if not result.valid:
            for issue in result.errors:
                print(f"{issue.path}: {issue.message}")
    """

    def __init__(self, schema: dict[str, Any] | None = None):
        self._schema = schema or CONFIG_SCHEMA

    def validate(self, config: dict[str, Any]) -> ValidationResult:
        """Validate a configuration dictionary."""
        issues: list[ValidationIssue] = []
        self._validate_dict(config, self._schema, "", issues)
        return ValidationResult(
            valid=all(i.severity != ValidationSeverity.ERROR for i in issues),
            issues=issues,
        )

    def _validate_dict(
        self,
        data: dict[str, Any],
        schema: dict[str, Any],
        path: str,
        issues: list[ValidationIssue],
    ) -> None:
        # Check for unknown keys
        for key in data:
            if key not in schema:
                issues.append(
                    ValidationIssue(
                        path=f"{path}.{key}" if path else key,
                        message=f"Unknown configuration key: '{key}'",
                        severity=ValidationSeverity.WARNING,
                        suggestion=f"Valid keys: {', '.join(schema.keys())}",
                    )
                )

        # Check required keys
        for key, spec in schema.items():
            if isinstance(spec, dict) and spec.get("required") and key not in data:
                issues.append(
                    ValidationIssue(
                        path=f"{path}.{key}" if path else key,
                        message=f"Required key '{key}' is missing",
                        severity=ValidationSeverity.ERROR,
                    )
                )

        # Validate each key
        for key, value in data.items():
            key_path = f"{path}.{key}" if path else key
            spec = schema.get(key)
            if spec is None:
                continue
            self._validate_value(value, spec, key_path, issues)

    def _validate_value(
        self,
        value: Any,
        spec: dict[str, Any],
        path: str,
        issues: list[ValidationIssue],
    ) -> None:
        expected_type = spec.get("type")

        if expected_type == "str" and not isinstance(value, str):
            issues.append(
                ValidationIssue(
                    path=path,
                    message=f"Expected string, got {type(value).__name__}",
                    severity=ValidationSeverity.ERROR,
                )
            )
            return

        if expected_type == "int" and not isinstance(value, int):
            issues.append(
                ValidationIssue(
                    path=path,
                    message=f"Expected integer, got {type(value).__name__}",
                    severity=ValidationSeverity.ERROR,
                )
            )
            return

        if expected_type == "float" and not isinstance(value, (int, float)):
            issues.append(
                ValidationIssue(
                    path=path,
                    message=f"Expected number, got {type(value).__name__}",
                    severity=ValidationSeverity.ERROR,
                )
            )
            return

        if expected_type == "bool" and not isinstance(value, bool):
            issues.append(
                ValidationIssue(
                    path=path,
                    message=f"Expected boolean, got {type(value).__name__}",
                    severity=ValidationSeverity.ERROR,
                )
            )
            return

        if expected_type == "list" and not isinstance(value, list):
            issues.append(
                ValidationIssue(
                    path=path,
                    message=f"Expected list, got {type(value).__name__}",
                    severity=ValidationSeverity.ERROR,
                )
            )
            return

        if expected_type == "dict" and not isinstance(value, dict):
            issues.append(
                ValidationIssue(
                    path=path,
                    message=f"Expected mapping, got {type(value).__name__}",
                    severity=ValidationSeverity.ERROR,
                )
            )
            return

        # Range checks
        if expected_type in ("int", "float"):
            if "min" in spec and value < spec["min"]:
                issues.append(
                    ValidationIssue(
                        path=path,
                        message=f"Value {value} is below minimum {spec['min']}",
                        severity=ValidationSeverity.ERROR,
                    )
                )
            if "max" in spec and value > spec["max"]:
                issues.append(
                    ValidationIssue(
                        path=path,
                        message=f"Value {value} exceeds maximum {spec['max']}",
                        severity=ValidationSeverity.ERROR,
                    )
                )

        # Allowed values
        if "allowed" in spec and isinstance(value, str) and value not in spec["allowed"]:
            issues.append(
                ValidationIssue(
                    path=path,
                    message=f"Value '{value}' not in allowed values: {spec['allowed']}",
                    severity=ValidationSeverity.ERROR,
                )
            )

        # List item validation
        if expected_type == "list" and isinstance(value, list):
            if "allowed_items" in spec:
                for i, item in enumerate(value):
                    if item not in spec["allowed_items"]:
                        issues.append(
                            ValidationIssue(
                                path=f"{path}[{i}]",
                                message=f"Value '{item}' not in allowed items",
                                severity=ValidationSeverity.ERROR,
                                suggestion=f"Allowed: {spec['allowed_items']}",
                            )
                        )
            if "item_schema" in spec:
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._validate_dict(item, spec["item_schema"], f"{path}[{i}]", issues)

        # Nested dict validation
        if expected_type == "dict" and isinstance(value, dict) and "schema" in spec:
            self._validate_dict(value, spec["schema"], path, issues)

    def generate_default(self) -> dict[str, Any]:
        """Generate default configuration."""
        return copy.deepcopy(DEFAULT_CONFIG)

    def migrate_v1_to_v2(self, config: dict[str, Any]) -> dict[str, Any]:
        """Migrate a v1 config to v2 format."""
        result = copy.deepcopy(config)
        result["version"] = "2"

        # v1 had flat timeout; v2 nests under verification
        if "timeout" in result:
            result.setdefault("verification", {})["timeout_seconds"] = result.pop("timeout")

        # v1 had "checks" at top level
        if "checks" in result and "verification" not in result:
            result["verification"] = {"checks": result.pop("checks")}
        elif "checks" in result:
            result["verification"]["checks"] = result.pop("checks")

        return result
