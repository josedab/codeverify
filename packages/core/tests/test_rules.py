"""Tests for Custom Rule Builder functionality."""

from uuid import UUID, uuid4

from codeverify_core.rules import (
    ConditionOperator,
    CustomRule,
    RuleAction,
    RuleBuilder,
    RuleCondition,
    RuleEvaluator,
    RuleScope,
    RuleSeverity,
    RuleType,
    RuleViolation,
    get_builtin_rules,
)


def _make_rule(
    name="Test Rule",
    description="A test rule",
    rule_type=RuleType.PATTERN,
    severity=RuleSeverity.MEDIUM,
    scope=RuleScope.LINE,
    pattern=r"print\(",
    action_message="Avoid using print statements",
    enabled=True,
    languages=None,
):
    """Helper to create a CustomRule with a pattern condition and action."""
    conditions = [
        RuleCondition(
            id=str(uuid4()),
            field="code",
            operator=ConditionOperator.MATCHES,
            value=pattern,
        )
    ]
    actions = [RuleAction(action_type="report", message=action_message)]
    return CustomRule(
        id=uuid4(),
        name=name,
        description=description,
        rule_type=rule_type,
        severity=severity,
        scope=scope,
        conditions=conditions,
        actions=actions,
        enabled=enabled,
        languages=languages or [],
    )


class TestCustomRule:
    """Tests for CustomRule dataclass."""

    def test_rule_creation(self):
        """Rule can be created with required fields."""
        rule = _make_rule()

        assert isinstance(rule.id, UUID)
        assert rule.name == "Test Rule"
        assert rule.rule_type == RuleType.PATTERN
        assert rule.severity == RuleSeverity.MEDIUM

    def test_rule_with_fix_suggestion(self):
        """Rule can include fix suggestion in action."""
        rule_id = uuid4()
        rule = CustomRule(
            id=rule_id,
            name="Rule with Fix",
            description="Has a fix",
            rule_type=RuleType.PATTERN,
            severity=RuleSeverity.MEDIUM,
            scope=RuleScope.LINE,
            conditions=[
                RuleCondition(
                    id=str(uuid4()),
                    field="code",
                    operator=ConditionOperator.MATCHES,
                    value=r"print\(",
                )
            ],
            actions=[
                RuleAction(
                    action_type="suggest_fix",
                    message="Use logger instead",
                    fix_template="logger.info($1)",
                )
            ],
        )

        assert rule.actions[0].fix_template == "logger.info($1)"

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
            builder.name("Chain Test")
            .description("Testing chaining")
            .severity("medium")
            .scope("line")
            .pattern(r"test")
            .action("Test message")
        )

        assert result is builder  # Should return self

    def test_build_pattern_rule(self):
        """Builder creates pattern-based rule."""
        rule = (
            RuleBuilder()
            .name("No Print Statements")
            .description("Disallow print statements")
            .severity(RuleSeverity.MEDIUM)
            .scope(RuleScope.LINE)
            .pattern(r"print\s*\(")
            .action("Use logger instead of print")
            .build()
        )

        assert isinstance(rule, CustomRule)
        assert isinstance(rule.id, UUID)
        assert rule.rule_type == RuleType.PATTERN
        assert len(rule.conditions) == 1
        assert rule.conditions[0].value == r"print\s*\("

    def test_build_ast_rule(self):
        """Builder creates AST-based rule."""
        rule = (
            RuleBuilder()
            .name("No Eval")
            .description("Disallow eval()")
            .severity(RuleSeverity.HIGH)
            .as_ast()
            .condition("code", ConditionOperator.MATCHES, r"\beval\s*\(")
            .action("eval() is dangerous")
            .build()
        )

        assert rule.rule_type == RuleType.AST

    def test_build_composite_rule(self):
        """Builder creates composite rule."""
        rule = (
            RuleBuilder()
            .name("TODOs and FIXMEs")
            .description("Find all TODOs and FIXMEs")
            .severity(RuleSeverity.INFO)
            .condition("code", ConditionOperator.MATCHES, r"TODO")
            .condition("code", ConditionOperator.MATCHES, r"FIXME")
            .as_composite()
            .with_logic("OR")
            .action("Found TODO or FIXME")
            .build()
        )

        assert rule.rule_type == RuleType.COMPOSITE
        assert len(rule.conditions) == 2

    def test_builder_with_languages_and_tags(self):
        """Builder supports language and tag configuration."""
        rule = (
            RuleBuilder()
            .name("Python Only")
            .description("Python specific rule")
            .severity("low")
            .pattern(r"print\(")
            .action("Use logging")
            .for_languages("python")
            .with_tags("style", "logging")
            .build()
        )

        assert rule.languages == ["python"]
        assert rule.tags == ["style", "logging"]


class TestRuleEvaluator:
    """Tests for RuleEvaluator."""

    def test_evaluate_pattern_rule_match(self):
        """Evaluator finds pattern matches."""
        rule = _make_rule(
            pattern=r"print\s*\(",
            action_message="Don't use print",
        )

        code = """\
def hello():
    print("Hello")
    print('World')
"""
        evaluator = RuleEvaluator([rule])
        violations = evaluator.evaluate(code, "test.py", "python")

        assert len(violations) == 2
        assert all(isinstance(v, dict) for v in violations)
        assert all(v["rule_name"] == rule.name for v in violations)

    def test_evaluate_pattern_rule_no_match(self):
        """Evaluator returns empty for no matches."""
        rule = _make_rule(
            pattern=r"print\s*\(",
            action_message="Don't use print",
        )

        code = """\
def hello():
    logger.info("Hello")
"""
        evaluator = RuleEvaluator([rule])
        violations = evaluator.evaluate(code, "test.py", "python")

        assert len(violations) == 0

    def test_evaluate_pattern_with_line_numbers(self):
        """Evaluator reports correct line numbers."""
        rule = _make_rule(
            pattern=r"TODO",
            action_message="Found TODO",
            severity=RuleSeverity.INFO,
        )

        code = (
            "# Line 1\n# Line 2\n# TODO: Fix this on line 3\n# Line 4\n# TODO: And this on line 5\n"
        )
        evaluator = RuleEvaluator([rule])
        violations = evaluator.evaluate(code, "test.py", "python")

        assert len(violations) == 2
        lines = [v["line"] for v in violations]
        assert 3 in lines
        assert 5 in lines

    def test_evaluate_multiple_rules(self):
        """Evaluator can evaluate multiple rules."""
        rule_print = _make_rule(
            name="No Print",
            pattern=r"print\(",
            action_message="No print",
        )
        rule_todo = _make_rule(
            name="No TODO",
            pattern=r"TODO",
            action_message="No TODO",
            severity=RuleSeverity.INFO,
        )

        code = """\
print("test")
# TODO: fix
"""
        evaluator = RuleEvaluator([rule_print, rule_todo])
        violations = evaluator.evaluate(code, "test.py", "python")

        assert len(violations) == 2
        rule_names = {v["rule_name"] for v in violations}
        assert "No Print" in rule_names
        assert "No TODO" in rule_names

    def test_evaluate_case_insensitive(self):
        """Evaluator supports case-insensitive patterns."""
        rule = _make_rule(
            pattern=r"(?i)password\s*=",
            action_message="Hardcoded password",
            severity=RuleSeverity.CRITICAL,
        )

        code = """\
PASSWORD = "secret"
password = "another"
Password = "third"
"""
        evaluator = RuleEvaluator([rule])
        violations = evaluator.evaluate(code, "test.py", "python")

        assert len(violations) == 3

    def test_evaluate_disabled_rule(self):
        """Evaluator skips disabled rules."""
        rule = _make_rule(
            pattern=r".*",
            action_message="Should not match",
            enabled=False,
        )

        code = "any code here"
        evaluator = RuleEvaluator([rule])
        violations = evaluator.evaluate(code, "test.py", "python")

        assert len(violations) == 0


class TestBuiltinRules:
    """Tests for built-in rules."""

    def test_builtin_rules_available(self):
        """Built-in rules are available."""
        rules = get_builtin_rules()

        assert isinstance(rules, dict)
        assert len(rules) > 0
        assert all(isinstance(r, CustomRule) for r in rules.values())

    def test_builtin_no_print_rule(self):
        """Built-in no-print rule exists and works."""
        rules = get_builtin_rules()
        assert "no-print" in rules
        no_print = rules["no-print"]
        assert no_print.rule_type == RuleType.PATTERN

    def test_builtin_no_secrets_rule(self):
        """Built-in no-hardcoded-secrets rule exists."""
        rules = get_builtin_rules()
        assert "no-hardcoded-secrets" in rules
        no_secrets = rules["no-hardcoded-secrets"]
        assert no_secrets.severity == RuleSeverity.CRITICAL

    def test_builtin_no_eval_rule(self):
        """Built-in no-eval rule exists."""
        rules = get_builtin_rules()
        assert "no-eval" in rules


class TestRuleViolation:
    """Tests for RuleViolation dataclass."""

    def test_violation_creation(self):
        """Violation can be created with all fields."""
        rule_id = uuid4()
        violation = RuleViolation(
            rule_id=rule_id,
            rule_name="Test Rule",
            severity=RuleSeverity.HIGH,
            message="Test violation",
            file_path="test.py",
            line_number=42,
            column=10,
            code_snippet="print('bad')",
        )

        assert violation.rule_id == rule_id
        assert violation.line_number == 42
        assert violation.severity == RuleSeverity.HIGH

    def test_violation_default_fields(self):
        """Violation uses defaults for optional fields."""
        rule_id = uuid4()
        violation = RuleViolation(
            rule_id=rule_id,
            rule_name="Test",
            severity=RuleSeverity.MEDIUM,
            message="Test",
            file_path="test.py",
            line_number=1,
        )

        assert violation.column == 0
        assert violation.end_line is None
        assert violation.end_column is None
        assert violation.code_snippet == ""
        assert violation.suggested_fix is None

    def test_violation_to_dict(self):
        """Violation can be converted to dict."""
        rule_id = uuid4()
        violation = RuleViolation(
            rule_id=rule_id,
            rule_name="Test",
            severity=RuleSeverity.MEDIUM,
            message="Test message",
            file_path="test.py",
            line_number=10,
        )

        d = violation.to_dict()
        assert d["rule_id"] == str(rule_id)
        assert d["line_number"] == 10
        assert d["severity"] == "medium"


class TestRuleFileFormats:
    """Tests for rule file format handling."""

    def test_rule_from_dict(self):
        """Rule can be created from a dictionary via from_dict."""
        rule_id = str(uuid4())
        data = {
            "id": rule_id,
            "name": "Dict Rule",
            "description": "From dict",
            "rule_type": "pattern",
            "severity": "medium",
            "scope": "line",
            "conditions": [
                {
                    "id": "c1",
                    "field": "code",
                    "operator": "matches",
                    "value": r"test",
                }
            ],
            "actions": [
                {
                    "action_type": "report",
                    "message": "Found test",
                }
            ],
        }

        rule = CustomRule.from_dict(data)

        assert str(rule.id) == rule_id
        assert rule.rule_type == RuleType.PATTERN

    def test_rule_serialization(self):
        """Rule can be serialized and deserialized."""
        rule = _make_rule()

        d = rule.to_dict()
        assert d["id"] == str(rule.id)
        assert d["rule_type"] == "pattern"

        restored = CustomRule.from_dict(d)
        assert restored.id == rule.id
        assert restored.name == rule.name
