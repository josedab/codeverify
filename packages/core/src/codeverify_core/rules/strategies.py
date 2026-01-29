"""Strategy pattern implementations for rule evaluation.

This module contains the abstract strategy interface and concrete
implementations for different rule types (pattern, composite, AST, semantic).
"""

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import structlog

from codeverify_core.rules.models import (
    ConditionOperator,
    CustomRule,
    RuleType,
)

if TYPE_CHECKING:
    from codeverify_core.rules.evaluator import RuleEvaluator

logger = structlog.get_logger()


class RuleEvaluationStrategy(ABC):
    """Abstract strategy for evaluating rules of a specific type.
    
    Implement this interface to add support for new rule types.
    """

    @abstractmethod
    def can_evaluate(self, rule_type: RuleType) -> bool:
        """Check if this strategy can evaluate the given rule type.
        
        Args:
            rule_type: The type of rule to check
            
        Returns:
            True if this strategy handles the given rule type
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        rule: CustomRule,
        code: str,
        file_path: str,
        evaluator: "RuleEvaluator",
    ) -> list[dict[str, Any]]:
        """Evaluate the rule and return violations.
        
        Args:
            rule: The rule to evaluate
            code: The code to check
            file_path: Path to the file being checked
            evaluator: The evaluator instance (for accessing helper methods)
            
        Returns:
            List of violation dictionaries
        """
        pass


class PatternRuleStrategy(RuleEvaluationStrategy):
    """Strategy for evaluating pattern-based (regex) rules."""

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
    """Strategy for evaluating composite rules with multiple conditions.
    
    Supports AND/OR logic between conditions.
    """

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
    """Strategy for evaluating AST-based rules.
    
    Currently a placeholder - full implementation requires AST parsing infrastructure.
    """

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
    """Strategy for evaluating semantic AI-powered rules.
    
    Currently a placeholder - full implementation requires AI agent infrastructure.
    """

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


# Default strategy instances for registration
_default_strategies: list[RuleEvaluationStrategy] = [
    PatternRuleStrategy(),
    CompositeRuleStrategy(),
    ASTRuleStrategy(),
    SemanticRuleStrategy(),
]


def get_default_strategies() -> list[RuleEvaluationStrategy]:
    """Get a copy of the default evaluation strategies.
    
    Returns:
        List of default RuleEvaluationStrategy instances
    """
    return _default_strategies.copy()
