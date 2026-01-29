"""Rule evaluator for executing custom rules against code.

This module provides the RuleEvaluator class that orchestrates rule
evaluation using pluggable strategies.
"""

import re
from fnmatch import fnmatch
from typing import Any

import structlog

from codeverify_core.rules.models import (
    ConditionOperator,
    CustomRule,
    RuleCondition,
)
from codeverify_core.rules.strategies import (
    PatternRuleStrategy,
    CompositeRuleStrategy,
    RuleEvaluationStrategy,
    get_default_strategies,
)

logger = structlog.get_logger()


class RuleEvaluator:
    """Evaluates custom rules against code using pluggable strategies.
    
    The evaluator uses the Strategy pattern to support different rule types.
    New rule types can be added by implementing RuleEvaluationStrategy and
    registering it with the evaluator.
    
    Example:
        >>> rules = [my_rule1, my_rule2]
        >>> evaluator = RuleEvaluator(rules)
        >>> violations = evaluator.evaluate(code, "src/main.py", "python")
    """

    def __init__(
        self,
        rules: list[CustomRule],
        strategies: list[RuleEvaluationStrategy] | None = None,
    ) -> None:
        """Initialize with rules and optional custom strategies.
        
        Args:
            rules: List of CustomRule instances to evaluate
            strategies: Optional list of evaluation strategies.
                       If None, uses default strategies.
        """
        self.rules = [r for r in rules if r.enabled]
        self._strategies = strategies or get_default_strategies()

    def register_strategy(self, strategy: RuleEvaluationStrategy) -> None:
        """Register a new evaluation strategy.
        
        Args:
            strategy: Strategy instance to register
        """
        self._strategies.append(strategy)

    def evaluate(
        self,
        code: str,
        file_path: str,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate all rules against the code.
        
        Args:
            code: The source code to evaluate
            file_path: Path to the file being evaluated
            language: Optional language identifier (e.g., "python", "typescript")
            
        Returns:
            List of violation dictionaries
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
        """Check if a rule applies to a file.
        
        Args:
            rule: The rule to check
            file_path: Path to the file
            language: Language of the file
            
        Returns:
            True if the rule should be evaluated for this file
        """
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
        """Evaluate a single rule using registered strategies.
        
        Args:
            rule: The rule to evaluate
            code: The source code
            file_path: Path to the file
            
        Returns:
            List of violation dictionaries
        """
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

    # Legacy methods for backward compatibility (used by strategies)
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
        """Evaluate a single condition.
        
        Args:
            condition: The condition to evaluate
            code: The source code
            
        Returns:
            True if the condition matches
        """
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
        """Get the value of a field from code.
        
        Args:
            field: Field name to extract
            code: The source code
            
        Returns:
            Extracted field value
        """
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
        """Create a violation record.
        
        Args:
            rule: The violated rule
            file_path: Path to the file
            line_num: Line number of the violation
            line_content: Content of the violating line
            
        Returns:
            Violation dictionary
        """
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
