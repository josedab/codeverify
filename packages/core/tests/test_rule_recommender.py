"""Tests for rule_recommender module."""

from __future__ import annotations

import pytest

from codeverify_core.rule_recommender import (
    AVAILABLE_RULES,
    RuleRecommender,
)


@pytest.fixture
def recommender():
    return RuleRecommender()


SAMPLE_FINDINGS = [
    {
        "rule_id": "null_safety",
        "file_path": "a.py",
        "severity": "high",
        "category": "safety",
        "message": "null",
    },
    {
        "rule_id": "null_safety",
        "file_path": "b.py",
        "severity": "high",
        "category": "safety",
        "message": "null",
    },
    {
        "rule_id": "array_bounds",
        "file_path": "c.py",
        "severity": "medium",
        "category": "safety",
        "message": "bounds",
    },
]


class TestRuleRecommender:
    def test_recommends_missing_security_rules(self, recommender):
        recs = recommender.analyze(
            findings_history=[],
            enabled_rules=["null_safety"],
            languages=["python"],
        )
        rec_ids = [r.rule_id for r in recs]
        assert "sql_injection" in rec_ids
        assert "command_injection" in rec_ids

    def test_no_recommendations_for_enabled_rules(self, recommender):
        all_rules = list(AVAILABLE_RULES.keys())
        recs = recommender.analyze(
            findings_history=[],
            enabled_rules=all_rules,
            languages=["python"],
        )
        enable_recs = [r for r in recs if r.action == "enable"]
        assert len(enable_recs) == 0

    def test_language_filtering(self, recommender):
        recs = recommender.analyze(
            findings_history=[],
            enabled_rules=[],
            languages=["python"],
        )
        rec_ids = [r.rule_id for r in recs]
        # xss is only for typescript/java, not python
        assert "xss" not in rec_ids

    def test_severity_increase_recommendation(self, recommender):
        findings = [{"rule_id": "error_handling", "severity": "low", "category": "quality"}] * 15
        recs = recommender.analyze(
            findings_history=findings,
            enabled_rules=["error_handling"],
            languages=["python"],
        )
        severity_recs = [r for r in recs if r.action == "increase_severity"]
        assert len(severity_recs) >= 1
        assert severity_recs[0].rule_id == "error_handling"

    def test_no_severity_increase_for_few_findings(self, recommender):
        findings = [{"rule_id": "error_handling", "severity": "low", "category": "quality"}] * 3
        recs = recommender.analyze(
            findings_history=findings,
            enabled_rules=["error_handling"],
            languages=["python"],
        )
        severity_recs = [r for r in recs if r.action == "increase_severity"]
        assert len(severity_recs) == 0

    def test_security_pattern_enables_security_rules(self, recommender):
        findings = [
            {"rule_id": "custom_sec", "severity": "high", "category": "security"},
        ]
        recs = recommender.analyze(
            findings_history=findings,
            enabled_rules=["null_safety"],
            languages=["python"],
        )
        security_enables = [
            r for r in recs if r.action == "enable" and "security" in r.reason.lower()
        ]
        assert len(security_enables) > 0

    def test_confidence_sorted(self, recommender):
        recs = recommender.analyze(
            findings_history=SAMPLE_FINDINGS,
            enabled_rules=[],
            languages=["python", "typescript"],
        )
        if len(recs) >= 2:
            for i in range(len(recs) - 1):
                assert recs[i].confidence >= recs[i + 1].confidence

    def test_empty_inputs(self, recommender):
        recs = recommender.analyze([], [], [])
        assert isinstance(recs, list)


class TestPatternInsights:
    def test_hot_file_detection(self, recommender):
        findings = [
            {"rule_id": f"r{i}", "file_path": "hot_file.py", "severity": "medium"} for i in range(5)
        ]
        insights = recommender.get_pattern_insights(findings)
        hot = [i for i in insights if i.pattern == "hot_file"]
        assert len(hot) >= 1
        assert "hot_file.py" in hot[0].affected_files

    def test_frequent_violation_detection(self, recommender):
        findings = [
            {"rule_id": "null_safety", "file_path": f"f{i}.py", "severity": "high"}
            for i in range(10)
        ]
        insights = recommender.get_pattern_insights(findings)
        freq = [i for i in insights if i.pattern == "frequent_violation"]
        assert len(freq) >= 1
        assert "null_safety" in freq[0].suggestion

    def test_no_insights_for_few_findings(self, recommender):
        findings = [
            {"rule_id": "r1", "file_path": "a.py", "severity": "low"},
        ]
        insights = recommender.get_pattern_insights(findings)
        assert len(insights) == 0

    def test_empty_findings(self, recommender):
        insights = recommender.get_pattern_insights([])
        assert insights == []
