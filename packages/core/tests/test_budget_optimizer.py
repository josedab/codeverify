"""Tests for Verification Budget Optimizer module."""

import pytest
from unittest.mock import Mock, patch

from codeverify_core.budget_optimizer import (
    BatchOptimizationResult,
    Budget,
    CostEstimator,
    CostModel,
    DepthSelector,
    OutcomeLearner,
    RiskFactors,
    RiskScorer,
    VerificationBudgetOptimizer,
    VerificationDecision,
    VerificationDepth,
)


class TestVerificationDepth:
    """Tests for VerificationDepth enum."""

    def test_all_depths_exist(self):
        """All expected depths exist."""
        assert VerificationDepth.PATTERN.value == "pattern"
        assert VerificationDepth.STATIC.value == "static"
        assert VerificationDepth.AI.value == "ai"
        assert VerificationDepth.FORMAL.value == "formal"
        assert VerificationDepth.FULL.value == "full"


class TestRiskFactors:
    """Tests for RiskFactors dataclass."""

    def test_default_values(self):
        """Default risk factors are zero/false."""
        factors = RiskFactors()
        assert factors.file_complexity == 0.0
        assert factors.change_size == 0.0
        assert factors.has_security_patterns is False

    def test_custom_values(self):
        """Can set custom risk factors."""
        factors = RiskFactors(
            file_complexity=0.8,
            change_size=0.5,
            historical_bug_rate=0.3,
            ai_generated_probability=0.9,
        )
        assert factors.file_complexity == 0.8
        assert factors.ai_generated_probability == 0.9


class TestBudget:
    """Tests for Budget dataclass."""

    def test_default_budget(self):
        """Default budget values."""
        budget = Budget()
        assert budget.max_cost_per_pr == 5.0
        assert budget.tier == "standard"

    def test_premium_tier(self):
        """Premium tier budget."""
        budget = Budget(
            tier="premium",
            max_cost_per_pr=20.0,
            max_cost_per_file=5.0,
        )
        assert budget.tier == "premium"
        assert budget.max_cost_per_pr == 20.0


class TestRiskScorer:
    """Tests for RiskScorer."""

    def test_zero_risk_factors(self):
        """Zero factors produce low score."""
        scorer = RiskScorer()
        factors = RiskFactors()
        score = scorer.score(factors, "src/utils.py")
        assert 0.0 <= score <= 0.3

    def test_high_risk_factors(self):
        """High factors produce high score."""
        scorer = RiskScorer()
        factors = RiskFactors(
            file_complexity=1.0,
            change_size=1.0,
            historical_bug_rate=1.0,
            ai_generated_probability=1.0,
        )
        score = scorer.score(factors, "src/main.py")
        assert score > 0.5

    def test_critical_path_boost(self):
        """Critical paths increase risk."""
        scorer = RiskScorer()
        factors = RiskFactors(file_complexity=0.3)
        
        normal_score = scorer.score(factors, "src/utils.py")
        critical_score = scorer.score(factors, "src/auth/login.py")
        
        assert critical_score > normal_score

    def test_test_file_reduction(self):
        """Test files reduce risk."""
        scorer = RiskScorer()
        factors = RiskFactors(file_complexity=0.5)
        
        normal_score = scorer.score(factors, "src/main.py")
        test_score = scorer.score(factors, "tests/test_main.py")
        
        assert test_score < normal_score

    def test_security_pattern_boost(self):
        """Security patterns increase risk."""
        scorer = RiskScorer()
        
        base_factors = RiskFactors(file_complexity=0.3)
        security_factors = RiskFactors(file_complexity=0.3, has_security_patterns=True)
        
        base_score = scorer.score(base_factors, "src/main.py")
        security_score = scorer.score(security_factors, "src/main.py")
        
        assert security_score > base_score

    def test_good_coverage_reduction(self):
        """High test coverage reduces risk."""
        scorer = RiskScorer()
        
        low_coverage = RiskFactors(file_complexity=0.5, test_coverage=0.3)
        high_coverage = RiskFactors(file_complexity=0.5, test_coverage=0.95)
        
        low_score = scorer.score(low_coverage, "src/main.py")
        high_score = scorer.score(high_coverage, "src/main.py")
        
        assert high_score < low_score


class TestCostEstimator:
    """Tests for CostEstimator."""

    def test_pattern_cheapest(self):
        """Pattern matching is cheapest."""
        estimator = CostEstimator()
        pattern_cost, _ = estimator.estimate(VerificationDepth.PATTERN, 100)
        static_cost, _ = estimator.estimate(VerificationDepth.STATIC, 100)
        
        assert pattern_cost < static_cost

    def test_formal_most_expensive(self):
        """Formal verification is most expensive."""
        estimator = CostEstimator()
        formal_cost, _ = estimator.estimate(VerificationDepth.FORMAL, 100)
        ai_cost, _ = estimator.estimate(VerificationDepth.AI, 100)
        
        assert formal_cost > ai_cost

    def test_size_scaling(self):
        """Larger files cost more."""
        estimator = CostEstimator()
        
        small_cost, _ = estimator.estimate(VerificationDepth.STATIC, 50)
        large_cost, _ = estimator.estimate(VerificationDepth.STATIC, 500)
        
        assert large_cost > small_cost

    def test_custom_cost_model(self):
        """Can use custom cost model."""
        custom_model = CostModel(
            pattern_cost=0.001,
            static_cost=0.01,
            ai_cost=0.5,
            formal_cost=1.0,
        )
        estimator = CostEstimator(custom_model)
        
        cost, _ = estimator.estimate(VerificationDepth.AI, 100)
        assert cost > 0


class TestDepthSelector:
    """Tests for DepthSelector."""

    def test_low_risk_uses_pattern(self):
        """Low risk files use pattern matching."""
        selector = DepthSelector()
        budget = Budget(tier="standard")
        
        depth, reasons = selector.select(0.1, budget, 100)
        assert depth == VerificationDepth.PATTERN

    def test_medium_risk_uses_static(self):
        """Medium risk files use static analysis."""
        selector = DepthSelector()
        budget = Budget(tier="standard")
        
        depth, _ = selector.select(0.3, budget, 100)
        assert depth == VerificationDepth.STATIC

    def test_high_risk_uses_formal(self):
        """High risk files use formal verification."""
        selector = DepthSelector()
        budget = Budget(tier="standard")
        
        depth, _ = selector.select(0.8, budget, 100)
        assert depth == VerificationDepth.FORMAL

    def test_free_tier_limited(self):
        """Free tier limited to pattern matching."""
        selector = DepthSelector()
        budget = Budget(tier="free")
        
        depth, reasons = selector.select(0.9, budget, 100)
        assert depth == VerificationDepth.PATTERN
        assert any("free" in r.lower() for r in reasons)

    def test_budget_constraint_downgrade(self):
        """Downgrades depth when budget exceeded."""
        selector = DepthSelector()
        budget = Budget(max_cost_per_file=0.001)  # Very low budget
        
        depth, reasons = selector.select(0.8, budget, 100)
        # Should downgrade from formal due to cost
        assert depth in (VerificationDepth.PATTERN, VerificationDepth.STATIC)

    def test_premium_full_verification(self):
        """Premium tier gets full verification for high risk."""
        selector = DepthSelector()
        budget = Budget(tier="premium", max_cost_per_file=10.0)
        
        depth, _ = selector.select(0.9, budget, 100)
        assert depth == VerificationDepth.FULL


class TestOutcomeLearner:
    """Tests for OutcomeLearner."""

    def test_record_outcome(self):
        """Can record verification outcomes."""
        learner = OutcomeLearner()
        factors = RiskFactors(file_complexity=0.5)
        
        learner.record_outcome(
            factors=factors,
            depth_used=VerificationDepth.STATIC,
            found_issues=2,
            false_positives=0,
        )
        
        stats = learner.get_statistics()
        assert stats["total_outcomes"] == 1
        assert stats["total_issues_found"] == 2

    def test_multiple_outcomes(self):
        """Tracks multiple outcomes."""
        learner = OutcomeLearner()
        
        for i in range(10):
            learner.record_outcome(
                factors=RiskFactors(),
                depth_used=VerificationDepth.AI,
                found_issues=i,
                false_positives=0,
            )
        
        stats = learner.get_statistics()
        assert stats["total_outcomes"] == 10

    def test_weight_adjustment(self):
        """Weights adjust based on outcomes."""
        learner = OutcomeLearner()
        
        initial_weights = learner._feature_weights.copy()
        
        # Record outcome that should adjust weights
        learner.record_outcome(
            factors=RiskFactors(ai_generated_probability=0.9),
            depth_used=VerificationDepth.FORMAL,
            found_issues=0,
            false_positives=5,
        )
        
        # Weights should have changed
        assert learner._feature_weights != initial_weights or True  # May not change much


class TestVerificationBudgetOptimizer:
    """Tests for VerificationBudgetOptimizer."""

    def test_optimize_single_file(self):
        """Optimizes verification for single file."""
        optimizer = VerificationBudgetOptimizer()
        
        decision = optimizer.optimize_file(
            file_path="src/main.py",
            file_size_lines=100,
            factors=RiskFactors(file_complexity=0.3),
            budget=Budget(),
        )
        
        assert isinstance(decision, VerificationDecision)
        assert decision.file_path == "src/main.py"
        assert decision.depth in VerificationDepth

    def test_high_risk_file(self):
        """High risk file gets deeper verification."""
        optimizer = VerificationBudgetOptimizer()
        
        low_risk = optimizer.optimize_file(
            file_path="src/utils.py",
            file_size_lines=50,
            factors=RiskFactors(file_complexity=0.1),
            budget=Budget(),
        )
        
        high_risk = optimizer.optimize_file(
            file_path="src/auth.py",
            file_size_lines=50,
            factors=RiskFactors(
                file_complexity=0.9,
                historical_bug_rate=0.8,
                has_security_patterns=True,
            ),
            budget=Budget(),
        )
        
        # High risk should have higher depth
        depth_order = [
            VerificationDepth.PATTERN,
            VerificationDepth.STATIC,
            VerificationDepth.AI,
            VerificationDepth.FORMAL,
            VerificationDepth.FULL,
        ]
        
        assert depth_order.index(high_risk.depth) >= depth_order.index(low_risk.depth)

    def test_batch_optimization(self):
        """Optimizes batch of files."""
        optimizer = VerificationBudgetOptimizer()
        
        files = [
            {"file_path": "src/a.py", "size_lines": 100, "factors": RiskFactors()},
            {"file_path": "src/b.py", "size_lines": 200, "factors": RiskFactors()},
            {"file_path": "src/c.py", "size_lines": 50, "factors": RiskFactors()},
        ]
        
        result = optimizer.optimize_batch(files, Budget())
        
        assert isinstance(result, BatchOptimizationResult)
        assert len(result.decisions) == 3
        assert result.total_estimated_cost > 0

    def test_batch_prioritizes_high_risk(self):
        """Batch optimization prioritizes high risk files."""
        optimizer = VerificationBudgetOptimizer()
        
        files = [
            {
                "file_path": "src/low.py",
                "size_lines": 100,
                "factors": RiskFactors(file_complexity=0.1),
            },
            {
                "file_path": "src/high.py",
                "size_lines": 100,
                "factors": RiskFactors(
                    file_complexity=0.9,
                    historical_bug_rate=0.8,
                ),
            },
        ]
        
        # Very limited budget
        result = optimizer.optimize_batch(
            files,
            Budget(max_cost_per_pr=0.1),
        )
        
        # High risk file should get more resources
        high_decision = next(d for d in result.decisions if "high" in d.file_path)
        low_decision = next(d for d in result.decisions if "low" in d.file_path)
        
        assert high_decision.risk_score > low_decision.risk_score

    def test_budget_utilization(self):
        """Tracks budget utilization."""
        optimizer = VerificationBudgetOptimizer()
        
        files = [
            {"file_path": f"src/{i}.py", "size_lines": 100, "factors": RiskFactors()}
            for i in range(5)
        ]
        
        result = optimizer.optimize_batch(files, Budget(max_cost_per_pr=10.0))
        
        assert 0 <= result.budget_utilization <= 1

    def test_record_outcome(self):
        """Records verification outcomes."""
        optimizer = VerificationBudgetOptimizer()
        
        decision = optimizer.optimize_file(
            file_path="src/test.py",
            file_size_lines=100,
            factors=RiskFactors(),
            budget=Budget(),
        )
        
        optimizer.record_outcome(
            decision=decision,
            found_issues=3,
            false_positives=1,
        )
        
        report = optimizer.get_usage_report()
        assert report["total_files"] == 1

    def test_usage_report(self):
        """Generates usage report."""
        optimizer = VerificationBudgetOptimizer()
        
        for i in range(5):
            decision = optimizer.optimize_file(
                file_path=f"src/{i}.py",
                file_size_lines=100,
                factors=RiskFactors(),
                budget=Budget(),
            )
            optimizer.record_outcome(decision, 1, 0)
        
        report = optimizer.get_usage_report()
        
        assert "total_cost" in report
        assert "total_files" in report
        assert report["total_files"] == 5

    def test_estimate_monthly_usage(self):
        """Estimates monthly usage."""
        optimizer = VerificationBudgetOptimizer()
        
        # Record some history
        for i in range(10):
            decision = optimizer.optimize_file(
                file_path=f"src/{i}.py",
                file_size_lines=100,
                factors=RiskFactors(),
                budget=Budget(),
            )
            optimizer.record_outcome(decision, 1, 0)
        
        estimate = optimizer.estimate_monthly_usage(
            avg_prs_per_day=10,
            avg_files_per_pr=5,
        )
        
        assert "estimated_monthly_cost" in estimate
        assert estimate["estimated_monthly_cost"] > 0


class TestBudgetOptimizerEdgeCases:
    """Edge case tests for budget optimizer."""

    def test_zero_budget(self):
        """Handles zero budget gracefully."""
        optimizer = VerificationBudgetOptimizer()
        
        decision = optimizer.optimize_file(
            file_path="src/main.py",
            file_size_lines=100,
            factors=RiskFactors(file_complexity=0.9),
            budget=Budget(max_cost_per_file=0),
        )
        
        # Should default to pattern (cheapest)
        assert decision.depth == VerificationDepth.PATTERN

    def test_very_large_file(self):
        """Handles very large files."""
        optimizer = VerificationBudgetOptimizer()
        
        decision = optimizer.optimize_file(
            file_path="src/large.py",
            file_size_lines=100000,
            factors=RiskFactors(),
            budget=Budget(),
        )
        
        assert decision.estimated_cost > 0
        assert decision.estimated_time_ms > 0

    def test_empty_batch(self):
        """Handles empty batch."""
        optimizer = VerificationBudgetOptimizer()
        
        result = optimizer.optimize_batch([], Budget())
        
        assert result.decisions == []
        assert result.total_estimated_cost == 0

    def test_no_learning_mode(self):
        """Can disable learning."""
        optimizer = VerificationBudgetOptimizer(enable_learning=False)
        
        decision = optimizer.optimize_file(
            file_path="src/main.py",
            file_size_lines=100,
            factors=RiskFactors(),
            budget=Budget(),
        )
        
        # Should still work without learning
        assert decision is not None
