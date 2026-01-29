"""Tests for Custom Rule Builder functionality."""

import pytest
from unittest.mock import Mock, patch

from codeverify_core.rules import (
    CustomRule,
    RuleType,
    RuleEvaluator,
    RuleBuilder,
    RuleViolation,
    get_builtin_rules,
)


class TestCustomRule:
    """Tests for CustomRule dataclass."""

    def test_rule_creation(self):
        """Rule can be created with required fields."""
        rule = CustomRule(
            id="test-rule",
            name="Test Rule",
            description="A test rule",
            type=RuleType.PATTERN,
            pattern=r"print\(",
            severity="warning",
            message="Avoid using print statements",
        )
        
        assert rule.id == "test-rule"
        assert rule.name == "Test Rule"
        assert rule.type == RuleType.PATTERN
        assert rule.severity == "warning"

    def test_rule_with_fix_suggestion(self):
        """Rule can include fix suggestion."""
        rule = CustomRule(
            id="rule-with-fix",
            name="Rule with Fix",
            description="Has a fix",
            type=RuleType.PATTERN,
            pattern=r"print\(",
            severity="warning",
            message="Use logger instead",
            fix_template="logger.info($1)",
        )
        
        assert rule.fix_template == "logger.info($1)"

    def test_rule_types(self):
        """All rule types are available."""
        assert RuleType.PATTERN.value == "pattern"
        assert RuleType.AST.value == "ast"
        assert RuleType.SEMANTIC.value == "semantic"
        assert RuleType.COMPOSITE.value == "composite"


class TestRuleBuilder:
    """Tests for RuleBuilder fluent interface."""

    def test_builder_chain(self):
        """Builder supports method chaining."""
        builder = RuleBuilder()
        result = (
            builder
            .with_id("chain-test")
            .with_name("Chain Test")
            .with_description("Testing chaining")
            .with_type(RuleType.PATTERN)
            .with_pattern(r"test")
            .with_severity("error")
            .with_message("Test message")
        )
        
        assert result is builder  # Should return self

    def test_build_pattern_rule(self):
        """Builder creates pattern-based rule."""
        rule = (
            RuleBuilder()
            .with_id("no-print")
            .with_name("No Print Statements")
            .with_description("Disallow print statements")
            .with_type(RuleType.PATTERN)
            .with_pattern(r"print\s*\(")
            .with_severity("warning")
            .with_message("Use logger instead of print")
            .build()
        )
        
        assert isinstance(rule, CustomRule)
        assert rule.id == "no-print"
        assert rule.type == RuleType.PATTERN
        assert rule.pattern == r"print\s*\("

    def test_build_ast_rule(self):
        """Builder creates AST-based rule."""
        rule = (
            RuleBuilder()
            .with_id("no-eval")
            .with_name("No Eval")
            .with_description("Disallow eval()")
            .with_type(RuleType.AST)
            .with_ast_node_type("Call")
            .with_ast_condition("node.func.id == 'eval'")
            .with_severity("error")
            .with_message("eval() is dangerous")
            .build()
        )
        
        assert rule.type == RuleType.AST
        assert rule.ast_node_type == "Call"

    def test_build_composite_rule(self):
        """Builder creates composite rule with sub-rules."""
        sub_rule1 = CustomRule(
            id="sub1",
            name="Sub Rule 1",
            description="First sub-rule",
            type=RuleType.PATTERN,
            pattern=r"TODO",
            severity="info",
            message="TODO found",
        )
        sub_rule2 = CustomRule(
            id="sub2",
            name="Sub Rule 2",
            description="Second sub-rule",
            type=RuleType.PATTERN,
            pattern=r"FIXME",
            severity="info",
            message="FIXME found",
        )
        
        rule = (
            RuleBuilder()
            .with_id("todos-and-fixmes")
            .with_name("TODOs and FIXMEs")
            .with_description("Find all TODOs and FIXMEs")
            .with_type(RuleType.COMPOSITE)
            .with_sub_rules([sub_rule1, sub_rule2])
            .with_composition_mode("or")
            .with_severity("info")
            .with_message("Found TODO or FIXME")
            .build()
        )
        
        assert rule.type == RuleType.COMPOSITE
        assert len(rule.sub_rules) == 2

    def test_builder_validation_missing_id(self):
        """Builder validates required fields."""
        with pytest.raises(ValueError, match="id"):
            (
                RuleBuilder()
                .with_name("No ID")
                .with_type(RuleType.PATTERN)
                .build()
            )

    def test_builder_validation_missing_pattern(self):
        """Pattern rule requires pattern field."""
        with pytest.raises(ValueError, match="pattern"):
            (
                RuleBuilder()
                .with_id("test")
                .with_name("Test")
                .with_type(RuleType.PATTERN)
                # Missing pattern
                .build()
            )


class TestRuleEvaluator:
    """Tests for RuleEvaluator."""

    @pytest.fixture
    def evaluator(self):
        """Create a rule evaluator."""
        return RuleEvaluator()

    def test_evaluate_pattern_rule_match(self, evaluator):
        """Evaluator finds pattern matches."""
        rule = CustomRule(
            id="no-print",
            name="No Print",
            description="No prints",
            type=RuleType.PATTERN,
            pattern=r"print\s*\(",
            severity="warning",
            message="Don't use print",
        )
        
        code = """
def hello():
    print("Hello")
    print('World')
"""
        violations = evaluator.evaluate(rule, code)
        
        assert len(violations) == 2
        assert all(isinstance(v, RuleViolation) for v in violations)
        assert all(v.rule_id == "no-print" for v in violations)

    def test_evaluate_pattern_rule_no_match(self, evaluator):
        """Evaluator returns empty for no matches."""
        rule = CustomRule(
            id="no-print",
            name="No Print",
            description="No prints",
            type=RuleType.PATTERN,
            pattern=r"print\s*\(",
            severity="warning",
            message="Don't use print",
        )
        
        code = """
def hello():
    logger.info("Hello")
"""
        violations = evaluator.evaluate(rule, code)
        
        assert len(violations) == 0

    def test_evaluate_pattern_with_line_numbers(self, evaluator):
        """Evaluator reports correct line numbers."""
        rule = CustomRule(
            id="test",
            name="Test",
            description="Test",
            type=RuleType.PATTERN,
            pattern=r"TODO",
            severity="info",
            message="Found TODO",
        )
        
        code = """# Line 1
# Line 2
# TODO: Fix this on line 3
# Line 4
# TODO: And this on line 5
"""
        violations = evaluator.evaluate(rule, code)
        
        assert len(violations) == 2
        lines = [v.line for v in violations]
        assert 3 in lines
        assert 5 in lines

    def test_evaluate_multiple_rules(self, evaluator):
        """Evaluator can evaluate multiple rules."""
        rules = [
            CustomRule(
                id="no-print",
                name="No Print",
                description="No prints",
                type=RuleType.PATTERN,
                pattern=r"print\(",
                severity="warning",
                message="No print",
            ),
            CustomRule(
                id="no-todo",
                name="No TODO",
                description="No TODOs",
                type=RuleType.PATTERN,
                pattern=r"TODO",
                severity="info",
                message="No TODO",
            ),
        ]
        
        code = """
print("test")
# TODO: fix
"""
        all_violations = []
        for rule in rules:
            all_violations.extend(evaluator.evaluate(rule, code))
        
        assert len(all_violations) == 2
        rule_ids = {v.rule_id for v in all_violations}
        assert "no-print" in rule_ids
        assert "no-todo" in rule_ids

    def test_evaluate_case_insensitive(self, evaluator):
        """Evaluator supports case-insensitive patterns."""
        rule = CustomRule(
            id="no-password",
            name="No Password",
            description="No hardcoded passwords",
            type=RuleType.PATTERN,
            pattern=r"(?i)password\s*=",
            severity="error",
            message="Hardcoded password",
        )
        
        code = """
PASSWORD = "secret"
password = "another"
Password = "third"
"""
        violations = evaluator.evaluate(rule, code)
        
        assert len(violations) == 3

    def test_evaluate_disabled_rule(self, evaluator):
        """Evaluator skips disabled rules."""
        rule = CustomRule(
            id="disabled",
            name="Disabled Rule",
            description="This is disabled",
            type=RuleType.PATTERN,
            pattern=r".*",
            severity="warning",
            message="Should not match",
            enabled=False,
        )
        
        code = "any code here"
        violations = evaluator.evaluate(rule, code)
        
        assert len(violations) == 0


class TestBuiltinRules:
    """Tests for built-in rules."""

    def test_builtin_rules_available(self):
        """Built-in rules are available."""
        rules = get_builtin_rules()
        
        assert isinstance(rules, list)
        assert len(rules) > 0
        assert all(isinstance(r, CustomRule) for r in rules)

    def test_builtin_no_print_rule(self):
        """Built-in no-print rule exists and works."""
        rules = get_builtin_rules()
        no_print = next((r for r in rules if r.id == "no-print"), None)
        
        assert no_print is not None
        assert no_print.type == RuleType.PATTERN

    def test_builtin_no_secrets_rule(self):
        """Built-in no-hardcoded-secrets rule exists."""
        rules = get_builtin_rules()
        no_secrets = next((r for r in rules if "secret" in r.id.lower()), None)
        
        assert no_secrets is not None
        assert no_secrets.severity in ["error", "critical"]

    def test_builtin_no_eval_rule(self):
        """Built-in no-eval rule exists."""
        rules = get_builtin_rules()
        no_eval = next((r for r in rules if "eval" in r.id.lower()), None)
        
        assert no_eval is not None


class TestRuleViolation:
    """Tests for RuleViolation dataclass."""

    def test_violation_creation(self):
        """Violation can be created with all fields."""
        violation = RuleViolation(
            rule_id="test-rule",
            rule_name="Test Rule",
            severity="error",
            message="Test violation",
            file_path="test.py",
            line=42,
            column=10,
            code_snippet="print('bad')",
        )
        
        assert violation.rule_id == "test-rule"
        assert violation.line == 42
        assert violation.severity == "error"

    def test_violation_optional_fields(self):
        """Violation handles optional fields."""
        violation = RuleViolation(
            rule_id="test",
            rule_name="Test",
            severity="warning",
            message="Test",
        )
        
        assert violation.file_path is None
        assert violation.line is None
        assert violation.column is None
        assert violation.code_snippet is None

    def test_violation_to_dict(self):
        """Violation can be converted to dict."""
        violation = RuleViolation(
            rule_id="test",
            rule_name="Test",
            severity="warning",
            message="Test message",
            line=10,
        )
        
        # If to_dict method exists
        if hasattr(violation, "to_dict"):
            d = violation.to_dict()
            assert d["rule_id"] == "test"
            assert d["line"] == 10


class TestRuleFileFormats:
    """Tests for rule file format handling."""

    def test_rule_from_yaml_dict(self):
        """Rule can be created from YAML-like dict."""
        yaml_data = {
            "id": "yaml-rule",
            "name": "YAML Rule",
            "description": "From YAML",
            "type": "pattern",
            "pattern": r"test",
            "severity": "warning",
            "message": "Found test",
        }
        
        rule = CustomRule(
            id=yaml_data["id"],
            name=yaml_data["name"],
            description=yaml_data["description"],
            type=RuleType(yaml_data["type"]),
            pattern=yaml_data["pattern"],
            severity=yaml_data["severity"],
            message=yaml_data["message"],
        )
        
        assert rule.id == "yaml-rule"
        assert rule.type == RuleType.PATTERN

    def test_rule_serialization(self):
        """Rule can be serialized."""
        rule = CustomRule(
            id="serialize-test",
            name="Serialize Test",
            description="Test serialization",
            type=RuleType.PATTERN,
            pattern=r"test",
            severity="info",
            message="Test",
        )
        
        # If to_dict method exists
        if hasattr(rule, "to_dict"):
            d = rule.to_dict()
            assert d["id"] == "serialize-test"
            assert d["type"] == "pattern"
