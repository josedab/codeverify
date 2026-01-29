"""Models and data types for custom rule definitions.

This module contains the core data structures used for defining
custom verification rules.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


class RuleType(str, Enum):
    """Type of custom rule."""

    PATTERN = "pattern"  # Regex pattern matching
    AST = "ast"  # AST-based matching
    SEMANTIC = "semantic"  # AI-powered semantic matching
    COMPOSITE = "composite"  # Combination of rules


class RuleSeverity(str, Enum):
    """Severity level for rule violations."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RuleScope(str, Enum):
    """Scope where rule applies."""

    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    BLOCK = "block"
    LINE = "line"


class ConditionOperator(str, Enum):
    """Operators for rule conditions."""

    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    MATCHES = "matches"
    NOT_MATCHES = "not_matches"
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"


@dataclass
class RuleCondition:
    """A single condition in a rule."""

    id: str
    field: str  # What to check: "code", "line", "function_name", "imports", etc.
    operator: ConditionOperator
    value: str | int | float | bool | list[str]
    case_sensitive: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value,
            "case_sensitive": self.case_sensitive,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuleCondition":
        """Create from dictionary."""
        return cls(
            id=data.get("id", str(uuid4())),
            field=data["field"],
            operator=ConditionOperator(data["operator"]),
            value=data["value"],
            case_sensitive=data.get("case_sensitive", True),
            description=data.get("description", ""),
        )


@dataclass
class RuleAction:
    """Action to take when rule matches."""

    action_type: str  # "report", "suggest_fix", "block", "warn"
    message: str
    fix_template: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_type": self.action_type,
            "message": self.message,
            "fix_template": self.fix_template,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuleAction":
        """Create from dictionary."""
        return cls(
            action_type=data["action_type"],
            message=data["message"],
            fix_template=data.get("fix_template"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CustomRule:
    """A custom verification rule."""

    id: UUID
    name: str
    description: str
    rule_type: RuleType
    severity: RuleSeverity
    scope: RuleScope
    conditions: list[RuleCondition]
    actions: list[RuleAction]
    condition_logic: str = "AND"  # "AND" or "OR"
    enabled: bool = True
    languages: list[str] = field(default_factory=list)  # Empty = all languages
    file_patterns: list[str] = field(default_factory=list)  # Glob patterns
    exclude_patterns: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    author: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "rule_type": self.rule_type.value,
            "severity": self.severity.value,
            "scope": self.scope.value,
            "conditions": [c.to_dict() for c in self.conditions],
            "actions": [a.to_dict() for a in self.actions],
            "condition_logic": self.condition_logic,
            "enabled": self.enabled,
            "languages": self.languages,
            "file_patterns": self.file_patterns,
            "exclude_patterns": self.exclude_patterns,
            "tags": self.tags,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CustomRule":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if isinstance(data.get("id"), str) else data.get("id", uuid4()),
            name=data["name"],
            description=data.get("description", ""),
            rule_type=RuleType(data.get("rule_type", "pattern")),
            severity=RuleSeverity(data.get("severity", "medium")),
            scope=RuleScope(data.get("scope", "line")),
            conditions=[RuleCondition.from_dict(c) for c in data.get("conditions", [])],
            actions=[RuleAction.from_dict(a) for a in data.get("actions", [])],
            condition_logic=data.get("condition_logic", "AND"),
            enabled=data.get("enabled", True),
            languages=data.get("languages", []),
            file_patterns=data.get("file_patterns", []),
            exclude_patterns=data.get("exclude_patterns", []),
            tags=data.get("tags", []),
            author=data.get("author"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
            version=data.get("version", 1),
            metadata=data.get("metadata", {}),
        )

    def to_yaml(self) -> str:
        """Convert to YAML format for .codeverify.yml."""
        import yaml
        rule_dict = {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "enabled": self.enabled,
        }

        # Add pattern if it's a pattern-based rule
        if self.rule_type == RuleType.PATTERN:
            for condition in self.conditions:
                if condition.operator == ConditionOperator.MATCHES:
                    rule_dict["pattern"] = condition.value

        # Add semantic prompt if it's a semantic rule
        if self.rule_type == RuleType.SEMANTIC:
            rule_dict["prompt"] = self.description

        if self.languages:
            rule_dict["languages"] = self.languages

        return yaml.dump(rule_dict, default_flow_style=False)


@dataclass
class RuleViolation:
    """Represents a violation of a rule found during evaluation."""

    rule_id: UUID
    rule_name: str
    severity: RuleSeverity
    message: str
    file_path: str
    line_number: int
    column: int = 0
    end_line: int | None = None
    end_column: int | None = None
    code_snippet: str = ""
    suggested_fix: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": str(self.rule_id),
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
            "code_snippet": self.code_snippet,
            "suggested_fix": self.suggested_fix,
            "metadata": self.metadata,
            "detected_at": self.detected_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuleViolation":
        """Create from dictionary."""
        return cls(
            rule_id=UUID(data["rule_id"]),
            rule_name=data["rule_name"],
            severity=RuleSeverity(data["severity"]),
            message=data["message"],
            file_path=data["file_path"],
            line_number=data["line_number"],
            column=data.get("column", 0),
            end_line=data.get("end_line"),
            end_column=data.get("end_column"),
            code_snippet=data.get("code_snippet", ""),
            suggested_fix=data.get("suggested_fix"),
            metadata=data.get("metadata", {}),
            detected_at=datetime.fromisoformat(data["detected_at"])
            if "detected_at" in data
            else datetime.utcnow(),
        )
