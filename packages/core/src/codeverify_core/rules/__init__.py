"""Custom Rule Builder - No-code interface for creating verification rules.

This package provides a comprehensive system for defining and evaluating
custom code verification rules using a fluent builder interface and
pluggable evaluation strategies.

Modules:
    models: Core data structures (CustomRule, RuleCondition, RuleViolation, etc.)
    strategies: Strategy pattern implementations for rule evaluation
    evaluator: Rule evaluator orchestrating strategy-based evaluation
    builder: Fluent builder interface for constructing rules
    templates: Pre-built rule templates for common checks

Example:
    >>> from codeverify_core.rules import RuleBuilder, RuleEvaluator
    >>> 
    >>> # Create a custom rule
    >>> rule = (RuleBuilder()
    ...     .name("No Print Statements")
    ...     .severity("low")
    ...     .pattern(r"\\bprint\\s*\\(")
    ...     .action("Use logging instead")
    ...     .for_languages("python")
    ...     .build())
    >>> 
    >>> # Evaluate code
    >>> evaluator = RuleEvaluator([rule])
    >>> violations = evaluator.evaluate(code, "main.py", "python")
"""

# Models
from codeverify_core.rules.models import (
    ConditionOperator,
    CustomRule,
    RuleAction,
    RuleCondition,
    RuleScope,
    RuleSeverity,
    RuleType,
    RuleViolation,
)

# Strategies
from codeverify_core.rules.strategies import (
    ASTRuleStrategy,
    CompositeRuleStrategy,
    PatternRuleStrategy,
    RuleEvaluationStrategy,
    SemanticRuleStrategy,
    get_default_strategies,
)

# Evaluator
from codeverify_core.rules.evaluator import RuleEvaluator

# Builder
from codeverify_core.rules.builder import RuleBuilder

# Templates
from codeverify_core.rules.templates import (
    RULE_TEMPLATES,
    get_builtin_rules,
    get_rule_by_name,
    get_rules_by_tag,
    get_security_rules,
    get_style_rules,
)

__all__ = [
    # Models
    "ConditionOperator",
    "CustomRule",
    "RuleAction",
    "RuleCondition",
    "RuleScope",
    "RuleSeverity",
    "RuleType",
    "RuleViolation",
    # Strategies
    "ASTRuleStrategy",
    "CompositeRuleStrategy",
    "PatternRuleStrategy",
    "RuleEvaluationStrategy",
    "SemanticRuleStrategy",
    "get_default_strategies",
    # Evaluator
    "RuleEvaluator",
    # Builder
    "RuleBuilder",
    # Templates
    "RULE_TEMPLATES",
    "get_builtin_rules",
    "get_rule_by_name",
    "get_rules_by_tag",
    "get_security_rules",
    "get_style_rules",
]
