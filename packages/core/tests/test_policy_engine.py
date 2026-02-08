"""Tests for Policy Engine / CI-CD Gates module."""

import pytest

from codeverify_core.policy_engine import (
    PolicyAction,
    PolicyCondition,
    PolicyEngine,
    PolicyEvaluationResult,
    PolicyRule,
    PolicyScope,
    PolicySet,
    get_default_policy_set,
    parse_policy_yaml,
)


class TestPolicyAction:
    def test_actions_exist(self):
        assert PolicyAction.ALLOW is not None
        assert PolicyAction.DENY is not None
        assert PolicyAction.WARN is not None


class TestPolicyScope:
    def test_scopes_exist(self):
        assert PolicyScope.REPOSITORY is not None
        assert PolicyScope.FILE is not None
        assert PolicyScope.FUNCTION is not None


class TestPolicyCondition:
    def test_creation(self):
        cond = PolicyCondition(field="severity", operator="equals", value="critical")
        assert cond.field == "severity"
        assert cond.operator == "equals"


class TestPolicyRule:
    def test_creation(self):
        rule = PolicyRule(
            id="block-critical",
            name="Block critical findings",
            description="Block all critical",
            conditions=[PolicyCondition(field="severity", operator="equals", value="critical")],
            action=PolicyAction.DENY,
            scope=PolicyScope.REPOSITORY,
        )
        assert rule.id == "block-critical"
        assert rule.action == PolicyAction.DENY


class TestPolicySet:
    def test_creation(self):
        policy_set = PolicySet(
            name="default",
            version="1.0",
            description="Default policy set",
            rules=[
                PolicyRule(
                    id="r1", name="Rule 1",
                    description="First rule",
                    conditions=[PolicyCondition(field="severity", operator="equals", value="critical")],
                    action=PolicyAction.DENY,
                    scope=PolicyScope.REPOSITORY,
                ),
            ],
        )
        assert len(policy_set.rules) == 1
        assert policy_set.name == "default"


class TestPolicyEngine:
    def test_creation(self):
        engine = PolicyEngine()
        assert engine is not None

    def test_evaluate_returns_result(self):
        engine = PolicyEngine()
        policy_set = get_default_policy_set()
        context = {"severity": "critical", "category": "security", "finding_count": 1}
        result = engine.evaluate(policy_set, context)
        assert isinstance(result, list)

    def test_evaluate_empty_context(self):
        engine = PolicyEngine()
        policy_set = get_default_policy_set()
        result = engine.evaluate(policy_set, {})
        assert isinstance(result, list)

    def test_evaluate_with_custom_rules(self):
        engine = PolicyEngine()
        policy_set = PolicySet(
            name="strict",
            version="1.0",
            description="Strict policy set",
            rules=[
                PolicyRule(
                    id="block-all-high",
                    name="Block high findings",
            description="Block high severity",
                    conditions=[PolicyCondition(field="severity", operator="equals", value="high")],
                    action=PolicyAction.DENY,
                    scope=PolicyScope.REPOSITORY,
                ),
            ],
        )
        context = {"severity": "high", "category": "logic", "finding_count": 1}
        result = engine.evaluate(policy_set, context)
        assert isinstance(result, list)


class TestGetDefaultPolicySet:
    def test_returns_policy_set(self):
        ps = get_default_policy_set()
        assert isinstance(ps, PolicySet)
        assert len(ps.rules) > 0


class TestParsePolicyYaml:
    def test_parse_valid_yaml(self):
        yaml_content = """
name: test-policy
rules:
  - id: block-critical
    name: Block critical
    action: deny
    scope: repository
    conditions:
      - field: severity
        operator: equals
        value: critical
"""
        result = parse_policy_yaml(yaml_content)
        assert isinstance(result, PolicySet)
        assert result.name == "test-policy"
