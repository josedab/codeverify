"""Custom Rule Builder - No-code interface for creating verification rules."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger()


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


# Strategy pattern for rule evaluation
from abc import ABC, abstractmethod


class RuleEvaluationStrategy(ABC):
    """Abstract strategy for evaluating rules of a specific type."""

    @abstractmethod
    def can_evaluate(self, rule_type: RuleType) -> bool:
        """Check if this strategy can evaluate the given rule type."""
        pass

    @abstractmethod
    def evaluate(
        self,
        rule: CustomRule,
        code: str,
        file_path: str,
        evaluator: "RuleEvaluator",
    ) -> list[dict[str, Any]]:
        """Evaluate the rule and return violations."""
        pass


class PatternRuleStrategy(RuleEvaluationStrategy):
    """Strategy for evaluating pattern-based rules."""

    def can_evaluate(self, rule_type: RuleType) -> bool:
        return rule_type == RuleType.PATTERN

    def evaluate(
        self,
        rule: CustomRule,
        code: str,
        file_path: str,
        evaluator: "RuleEvaluator",
    ) -> list[dict[str, Any]]:
        violations = []
        lines = code.split("\n")

        for condition in rule.conditions:
            if condition.operator not in (
                ConditionOperator.MATCHES,
                ConditionOperator.CONTAINS,
            ):
                continue

            pattern = condition.value
            if not isinstance(pattern, str):
                continue

            flags = 0 if condition.case_sensitive else re.IGNORECASE

            for line_num, line in enumerate(lines, 1):
                if condition.operator == ConditionOperator.MATCHES:
                    if re.search(pattern, line, flags):
                        violations.append(
                            evaluator._create_violation(rule, file_path, line_num, line)
                        )
                elif condition.operator == ConditionOperator.CONTAINS:
                    search_line = line if condition.case_sensitive else line.lower()
                    search_val = pattern if condition.case_sensitive else pattern.lower()
                    if search_val in search_line:
                        violations.append(
                            evaluator._create_violation(rule, file_path, line_num, line)
                        )

        return violations


class CompositeRuleStrategy(RuleEvaluationStrategy):
    """Strategy for evaluating composite rules with multiple conditions."""

    def can_evaluate(self, rule_type: RuleType) -> bool:
        return rule_type == RuleType.COMPOSITE

    def evaluate(
        self,
        rule: CustomRule,
        code: str,
        file_path: str,
        evaluator: "RuleEvaluator",
    ) -> list[dict[str, Any]]:
        condition_results = []

        for condition in rule.conditions:
            result = evaluator._evaluate_condition(condition, code)
            condition_results.append(result)

        # Apply logic
        if rule.condition_logic == "AND":
            all_match = all(condition_results)
        else:  # OR
            all_match = any(condition_results)

        if all_match:
            return [evaluator._create_violation(rule, file_path, 1, code[:100])]

        return []


class ASTRuleStrategy(RuleEvaluationStrategy):
    """Strategy for evaluating AST-based rules (placeholder for future implementation)."""

    def can_evaluate(self, rule_type: RuleType) -> bool:
        return rule_type == RuleType.AST

    def evaluate(
        self,
        rule: CustomRule,
        code: str,
        file_path: str,
        evaluator: "RuleEvaluator",
    ) -> list[dict[str, Any]]:
        # AST rules require additional infrastructure
        logger.warning("AST rule evaluation not yet implemented", rule_id=str(rule.id))
        return []


class SemanticRuleStrategy(RuleEvaluationStrategy):
    """Strategy for evaluating semantic AI-powered rules (placeholder)."""

    def can_evaluate(self, rule_type: RuleType) -> bool:
        return rule_type == RuleType.SEMANTIC

    def evaluate(
        self,
        rule: CustomRule,
        code: str,
        file_path: str,
        evaluator: "RuleEvaluator",
    ) -> list[dict[str, Any]]:
        # Semantic rules require AI agent infrastructure
        logger.warning("Semantic rule evaluation not yet implemented", rule_id=str(rule.id))
        return []


# Default strategies
_default_strategies: list[RuleEvaluationStrategy] = [
    PatternRuleStrategy(),
    CompositeRuleStrategy(),
    ASTRuleStrategy(),
    SemanticRuleStrategy(),
]


class RuleEvaluator:
    """Evaluates custom rules against code using pluggable strategies."""

    def __init__(
        self,
        rules: list[CustomRule],
        strategies: list[RuleEvaluationStrategy] | None = None,
    ) -> None:
        """Initialize with rules and optional custom strategies."""
        self.rules = [r for r in rules if r.enabled]
        self._strategies = strategies or _default_strategies

    def register_strategy(self, strategy: RuleEvaluationStrategy) -> None:
        """Register a new evaluation strategy."""
        self._strategies.append(strategy)

    def evaluate(
        self,
        code: str,
        file_path: str,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Evaluate all rules against the code.

        Returns list of violations.
        """
        violations = []

        for rule in self.rules:
            # Check if rule applies to this file
            if not self._rule_applies(rule, file_path, language):
                continue

            # Evaluate the rule using appropriate strategy
            rule_violations = self._evaluate_rule(rule, code, file_path)
            violations.extend(rule_violations)

        return violations

    def _rule_applies(
        self,
        rule: CustomRule,
        file_path: str,
        language: str | None,
    ) -> bool:
        """Check if a rule applies to a file."""
        from fnmatch import fnmatch

        # Check language
        if rule.languages and language:
            if language.lower() not in [l.lower() for l in rule.languages]:
                return False

        # Check file patterns
        if rule.file_patterns:
            if not any(fnmatch(file_path, p) for p in rule.file_patterns):
                return False

        # Check exclusions
        if rule.exclude_patterns:
            if any(fnmatch(file_path, p) for p in rule.exclude_patterns):
                return False

        return True

    def _evaluate_rule(
        self,
        rule: CustomRule,
        code: str,
        file_path: str,
    ) -> list[dict[str, Any]]:
        """Evaluate a single rule using registered strategies."""
        # Find the appropriate strategy for this rule type
        for strategy in self._strategies:
            if strategy.can_evaluate(rule.rule_type):
                return strategy.evaluate(rule, code, file_path, self)

        logger.warning(
            "No strategy found for rule type",
            rule_type=rule.rule_type.value,
            rule_id=str(rule.id),
        )
        return []

    # Keep legacy methods for backward compatibility (used by strategies)
    def _evaluate_pattern_rule(
        self,
        rule: CustomRule,
        code: str,
        file_path: str,
    ) -> list[dict[str, Any]]:
        """Evaluate a pattern-based rule. Deprecated: Use PatternRuleStrategy."""
        return PatternRuleStrategy().evaluate(rule, code, file_path, self)

    def _evaluate_composite_rule(
        self,
        rule: CustomRule,
        code: str,
        file_path: str,
    ) -> list[dict[str, Any]]:
        """Evaluate a composite rule. Deprecated: Use CompositeRuleStrategy."""
        return CompositeRuleStrategy().evaluate(rule, code, file_path, self)

    def _evaluate_condition(
        self,
        condition: RuleCondition,
        code: str,
    ) -> bool:
        """Evaluate a single condition."""
        value = condition.value
        field_value = self._get_field_value(condition.field, code)

        if condition.operator == ConditionOperator.CONTAINS:
            return str(value) in str(field_value)
        elif condition.operator == ConditionOperator.NOT_CONTAINS:
            return str(value) not in str(field_value)
        elif condition.operator == ConditionOperator.MATCHES:
            return bool(re.search(str(value), str(field_value)))
        elif condition.operator == ConditionOperator.NOT_MATCHES:
            return not bool(re.search(str(value), str(field_value)))
        elif condition.operator == ConditionOperator.EQUALS:
            return field_value == value
        elif condition.operator == ConditionOperator.NOT_EQUALS:
            return field_value != value
        elif condition.operator == ConditionOperator.EXISTS:
            return bool(field_value)
        elif condition.operator == ConditionOperator.NOT_EXISTS:
            return not bool(field_value)

        return False

    def _get_field_value(self, field: str, code: str) -> Any:
        """Get the value of a field from code."""
        if field == "code":
            return code
        elif field == "line_count":
            return len(code.split("\n"))
        elif field == "imports":
            imports = re.findall(r"^(?:import|from)\s+(\S+)", code, re.MULTILINE)
            return imports
        elif field == "functions":
            functions = re.findall(r"^\s*(?:async\s+)?def\s+(\w+)", code, re.MULTILINE)
            return functions
        elif field == "classes":
            classes = re.findall(r"^\s*class\s+(\w+)", code, re.MULTILINE)
            return classes

        return None

    def _create_violation(
        self,
        rule: CustomRule,
        file_path: str,
        line_num: int,
        line_content: str,
    ) -> dict[str, Any]:
        """Create a violation record."""
        action = rule.actions[0] if rule.actions else None

        return {
            "rule_id": str(rule.id),
            "rule_name": rule.name,
            "severity": rule.severity.value,
            "message": action.message if action else rule.description,
            "file_path": file_path,
            "line": line_num,
            "code_snippet": line_content.strip()[:200],
            "fix_suggestion": action.fix_template if action else None,
            "tags": rule.tags,
        }


class RuleBuilder:
    """Builder class for creating custom rules with a fluent interface."""

    def __init__(self) -> None:
        """Initialize the builder."""
        self._id = uuid4()
        self._name = ""
        self._description = ""
        self._rule_type = RuleType.PATTERN
        self._severity = RuleSeverity.MEDIUM
        self._scope = RuleScope.LINE
        self._conditions: list[RuleCondition] = []
        self._actions: list[RuleAction] = []
        self._condition_logic = "AND"
        self._enabled = True
        self._languages: list[str] = []
        self._file_patterns: list[str] = []
        self._exclude_patterns: list[str] = []
        self._tags: list[str] = []

    def name(self, name: str) -> "RuleBuilder":
        """Set the rule name."""
        self._name = name
        return self

    def description(self, description: str) -> "RuleBuilder":
        """Set the rule description."""
        self._description = description
        return self

    def severity(self, severity: RuleSeverity | str) -> "RuleBuilder":
        """Set the severity level."""
        if isinstance(severity, str):
            severity = RuleSeverity(severity)
        self._severity = severity
        return self

    def pattern(self, pattern: str, description: str = "") -> "RuleBuilder":
        """Add a regex pattern condition."""
        self._rule_type = RuleType.PATTERN
        self._conditions.append(
            RuleCondition(
                id=str(uuid4()),
                field="code",
                operator=ConditionOperator.MATCHES,
                value=pattern,
                description=description,
            )
        )
        return self

    def contains(self, text: str, case_sensitive: bool = True) -> "RuleBuilder":
        """Add a contains condition."""
        self._conditions.append(
            RuleCondition(
                id=str(uuid4()),
                field="code",
                operator=ConditionOperator.CONTAINS,
                value=text,
                case_sensitive=case_sensitive,
            )
        )
        return self

    def action(
        self,
        message: str,
        fix_template: str | None = None,
        action_type: str = "report",
    ) -> "RuleBuilder":
        """Add an action."""
        self._actions.append(
            RuleAction(
                action_type=action_type,
                message=message,
                fix_template=fix_template,
            )
        )
        return self

    def for_languages(self, *languages: str) -> "RuleBuilder":
        """Set applicable languages."""
        self._languages = list(languages)
        return self

    def for_files(self, *patterns: str) -> "RuleBuilder":
        """Set file patterns."""
        self._file_patterns = list(patterns)
        return self

    def exclude_files(self, *patterns: str) -> "RuleBuilder":
        """Set exclusion patterns."""
        self._exclude_patterns = list(patterns)
        return self

    def with_tags(self, *tags: str) -> "RuleBuilder":
        """Add tags."""
        self._tags = list(tags)
        return self

    def build(self) -> CustomRule:
        """Build the custom rule."""
        return CustomRule(
            id=self._id,
            name=self._name,
            description=self._description,
            rule_type=self._rule_type,
            severity=self._severity,
            scope=self._scope,
            conditions=self._conditions,
            actions=self._actions,
            condition_logic=self._condition_logic,
            enabled=self._enabled,
            languages=self._languages,
            file_patterns=self._file_patterns,
            exclude_patterns=self._exclude_patterns,
            tags=self._tags,
        )


# Pre-built rule templates
RULE_TEMPLATES = {
    "no-print": RuleBuilder()
        .name("No Print Statements")
        .description("Use logging instead of print statements")
        .severity(RuleSeverity.LOW)
        .pattern(r"\bprint\s*\(")
        .action("Replace print() with proper logging", fix_template="logger.info({args})")
        .for_languages("python")
        .with_tags("style", "logging")
        .build(),

    "no-hardcoded-secrets": RuleBuilder()
        .name("No Hardcoded Secrets")
        .description("Detect hardcoded passwords and API keys")
        .severity(RuleSeverity.CRITICAL)
        .pattern(r"(?i)(password|api_key|secret|token)\s*=\s*['\"][^'\"]+['\"]")
        .action("Use environment variables for sensitive data")
        .with_tags("security", "secrets")
        .build(),

    "no-eval": RuleBuilder()
        .name("No Eval Usage")
        .description("Avoid using eval() which can execute arbitrary code")
        .severity(RuleSeverity.HIGH)
        .pattern(r"\beval\s*\(")
        .action("Replace eval() with safer alternatives like ast.literal_eval()")
        .for_languages("python")
        .with_tags("security")
        .build(),

    "require-type-hints": RuleBuilder()
        .name("Require Type Hints")
        .description("Function parameters should have type hints")
        .severity(RuleSeverity.LOW)
        .pattern(r"def\s+\w+\s*\([^)]*[a-zA-Z_]\w*\s*[,)]")
        .action("Add type hints to function parameters")
        .for_languages("python")
        .with_tags("style", "typing")
        .build(),
}


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


def get_builtin_rules() -> dict[str, CustomRule]:
    """Get all built-in rule templates.
    
    Returns:
        Dictionary mapping rule IDs to CustomRule instances.
    """
    return RULE_TEMPLATES.copy()
