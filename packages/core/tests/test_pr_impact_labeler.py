"""Tests for pr_impact_labeler module."""

from __future__ import annotations

from codeverify_core.pr_impact_labeler import (
    PRImpactScorer,
    RiskLevel,
    compute_file_criticality,
)


class TestFileCriticality:
    def test_auth_file_is_critical(self):
        assert compute_file_criticality("src/auth/login.py") == 1.0

    def test_payment_file_high(self):
        assert compute_file_criticality("services/payment_handler.py") >= 0.9

    def test_test_file_low(self):
        assert compute_file_criticality("tests/test_auth.py") <= 0.3

    def test_docs_very_low(self):
        assert compute_file_criticality("docs/README.md") <= 0.15

    def test_unknown_file_default(self):
        assert compute_file_criticality("src/foo.py") == 0.3

    def test_database_migration(self):
        assert compute_file_criticality("migrations/001_schema.py") >= 0.8

    def test_api_route(self):
        assert compute_file_criticality("api/users.py") >= 0.7


class TestPRImpactScorer:
    def test_trivial_change(self):
        scorer = PRImpactScorer()
        impact = scorer.score(
            changed_files={"docs/README.md": 3},
            findings=[],
        )
        assert impact.risk_level == RiskLevel.TRIVIAL
        assert impact.score < 20

    def test_critical_security_change(self):
        scorer = PRImpactScorer()
        impact = scorer.score(
            changed_files={"src/auth/jwt.py": 100},
            findings=[
                {"severity": "critical", "file_path": "src/auth/jwt.py"},
                {"severity": "high", "file_path": "src/auth/jwt.py"},
            ],
        )
        assert impact.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)
        assert "security-review-needed" in impact.labels

    def test_medium_change(self):
        scorer = PRImpactScorer()
        impact = scorer.score(
            changed_files={"src/utils.py": 50},
            findings=[{"severity": "medium", "file_path": "src/utils.py"}],
        )
        assert impact.risk_level in (RiskLevel.MEDIUM, RiskLevel.LOW)

    def test_labels_include_risk(self):
        scorer = PRImpactScorer()
        impact = scorer.score(
            changed_files={"src/auth.py": 200},
            findings=[],
        )
        assert any(label.startswith("risk:") for label in impact.labels)

    def test_senior_review_on_high_risk(self):
        scorer = PRImpactScorer()
        impact = scorer.score(
            changed_files={"src/auth/sso.py": 500},
            findings=[
                {"severity": "critical", "file_path": "src/auth/sso.py"},
            ],
        )
        assert "needs-senior-review" in impact.labels

    def test_summary_contains_stats(self):
        scorer = PRImpactScorer()
        impact = scorer.score(
            changed_files={"a.py": 10, "b.py": 20},
            findings=[{"severity": "low", "file_path": "a.py"}],
        )
        assert "30 lines" in impact.summary
        assert "2 files" in impact.summary
        assert "1 findings" in impact.summary

    def test_per_file_risks(self):
        scorer = PRImpactScorer()
        impact = scorer.score(
            changed_files={"src/auth.py": 10, "tests/test_auth.py": 5},
        )
        assert len(impact.file_risks) == 2
        auth_risk = next(r for r in impact.file_risks if r.path == "src/auth.py")
        test_risk = next(r for r in impact.file_risks if r.path == "tests/test_auth.py")
        assert auth_risk.criticality > test_risk.criticality

    def test_score_clamped_to_100(self):
        scorer = PRImpactScorer()
        impact = scorer.score(
            changed_files={"src/auth.py": 10000},
            findings=[
                {"severity": "critical", "file_path": "src/auth.py"},
            ]
            * 50,
        )
        assert impact.score <= 100.0

    def test_empty_pr(self):
        scorer = PRImpactScorer()
        impact = scorer.score(changed_files={})
        assert impact.risk_level == RiskLevel.TRIVIAL
        assert impact.change_size == 0

    def test_custom_weights(self):
        scorer = PRImpactScorer(
            size_weight=0.0,
            criticality_weight=1.0,
            findings_weight=0.0,
        )
        impact = scorer.score(changed_files={"src/auth.py": 1})
        # Score driven entirely by criticality
        assert impact.score > 50  # auth file has 1.0 criticality
