"""Fluent builder interface for creating custom rules.

This module provides RuleBuilder for constructing rules with a
clean, chainable API.
"""

from uuid import uuid4

from codeverify_core.rules.models import (
    ConditionOperator,
    CustomRule,
    RuleAction,
    RuleCondition,
    RuleScope,
    RuleSeverity,
    RuleType,
)


class RuleBuilder:
    """Builder class for creating custom rules with a fluent interface.
    
    Example:
        >>> rule = (RuleBuilder()
        ...     .name("No Print Statements")
        ...     .description("Use logging instead of print")
        ...     .severity("low")
        ...     .pattern(r"\\bprint\\s*\\(")
        ...     .action("Replace print() with logger.info()")
        ...     .for_languages("python")
        ...     .build())
    """

    def __init__(self) -> None:
        """Initialize the builder with default values."""
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
        """Set the rule name.
        
        Args:
            name: Human-readable rule name
        """
        self._name = name
        return self

    def description(self, description: str) -> "RuleBuilder":
        """Set the rule description.
        
        Args:
            description: Detailed description of what the rule checks
        """
        self._description = description
        return self

    def severity(self, severity: RuleSeverity | str) -> "RuleBuilder":
        """Set the severity level.
        
        Args:
            severity: Severity level (RuleSeverity enum or string)
        """
        if isinstance(severity, str):
            severity = RuleSeverity(severity)
        self._severity = severity
        return self

    def scope(self, scope: RuleScope | str) -> "RuleBuilder":
        """Set the rule scope.
        
        Args:
            scope: Scope where rule applies (RuleScope enum or string)
        """
        if isinstance(scope, str):
            scope = RuleScope(scope)
        self._scope = scope
        return self

    def pattern(self, pattern: str, description: str = "") -> "RuleBuilder":
        """Add a regex pattern condition.
        
        Args:
            pattern: Regular expression pattern to match
            description: Optional description of what the pattern matches
        """
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
        """Add a contains condition.
        
        Args:
            text: Text to search for
            case_sensitive: Whether the search is case-sensitive
        """
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

    def not_contains(self, text: str, case_sensitive: bool = True) -> "RuleBuilder":
        """Add a not-contains condition.
        
        Args:
            text: Text that should not be present
            case_sensitive: Whether the search is case-sensitive
        """
        self._conditions.append(
            RuleCondition(
                id=str(uuid4()),
                field="code",
                operator=ConditionOperator.NOT_CONTAINS,
                value=text,
                case_sensitive=case_sensitive,
            )
        )
        return self

    def condition(
        self,
        field: str,
        operator: ConditionOperator | str,
        value: str | int | float | bool | list[str],
        case_sensitive: bool = True,
        description: str = "",
    ) -> "RuleBuilder":
        """Add a custom condition.
        
        Args:
            field: Field to check (code, line_count, imports, etc.)
            operator: Condition operator
            value: Value to compare against
            case_sensitive: Whether string comparisons are case-sensitive
            description: Optional description
        """
        if isinstance(operator, str):
            operator = ConditionOperator(operator)
        self._conditions.append(
            RuleCondition(
                id=str(uuid4()),
                field=field,
                operator=operator,
                value=value,
                case_sensitive=case_sensitive,
                description=description,
            )
        )
        return self

    def action(
        self,
        message: str,
        fix_template: str | None = None,
        action_type: str = "report",
    ) -> "RuleBuilder":
        """Add an action for when the rule matches.
        
        Args:
            message: Message to display when rule matches
            fix_template: Optional template for auto-fix suggestion
            action_type: Type of action (report, suggest_fix, block, warn)
        """
        self._actions.append(
            RuleAction(
                action_type=action_type,
                message=message,
                fix_template=fix_template,
            )
        )
        return self

    def for_languages(self, *languages: str) -> "RuleBuilder":
        """Set applicable languages.
        
        Args:
            *languages: Language identifiers (e.g., "python", "typescript")
        """
        self._languages = list(languages)
        return self

    def for_files(self, *patterns: str) -> "RuleBuilder":
        """Set file patterns (glob syntax).
        
        Args:
            *patterns: Glob patterns for files to include
        """
        self._file_patterns = list(patterns)
        return self

    def exclude_files(self, *patterns: str) -> "RuleBuilder":
        """Set exclusion patterns.
        
        Args:
            *patterns: Glob patterns for files to exclude
        """
        self._exclude_patterns = list(patterns)
        return self

    def with_tags(self, *tags: str) -> "RuleBuilder":
        """Add tags for categorization.
        
        Args:
            *tags: Tag strings
        """
        self._tags = list(tags)
        return self

    def with_logic(self, logic: str) -> "RuleBuilder":
        """Set condition logic (AND/OR).
        
        Args:
            logic: "AND" or "OR"
        """
        self._condition_logic = logic.upper()
        return self

    def enabled(self, enabled: bool = True) -> "RuleBuilder":
        """Set whether the rule is enabled.
        
        Args:
            enabled: Whether the rule should be active
        """
        self._enabled = enabled
        return self

    def as_composite(self) -> "RuleBuilder":
        """Mark this rule as a composite rule."""
        self._rule_type = RuleType.COMPOSITE
        return self

    def as_semantic(self) -> "RuleBuilder":
        """Mark this rule as a semantic (AI-powered) rule."""
        self._rule_type = RuleType.SEMANTIC
        return self

    def as_ast(self) -> "RuleBuilder":
        """Mark this rule as an AST-based rule."""
        self._rule_type = RuleType.AST
        return self

    def build(self) -> CustomRule:
        """Build the custom rule.
        
        Returns:
            Configured CustomRule instance
        """
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
