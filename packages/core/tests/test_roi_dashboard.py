"""Tests for ROI Dashboard & Cost Transparency."""

import pytest
from datetime import datetime, timedelta

from codeverify_core.roi_dashboard import (
    BUG_COST_ESTIMATES,
    BugCaught,
    BugSeverity,
    BugValueCalculator,
    CostCategory,
    CostConfig,
    CostTracker,
    ROIDashboard,
    ROIMetrics,
    VerificationCost,
    create_dashboard,
)


class TestCostConfig:
    """Tests for CostConfig."""
    
    def test_default_config(self):
        config = CostConfig()
        
        assert config.input_token_cost == 0.003
        assert config.output_token_cost == 0.015
        assert config.industry == "technology"
    
    def test_industry_multiplier(self):
        tech_config = CostConfig(industry="technology")
        finance_config = CostConfig(industry="finance")
        healthcare_config = CostConfig(industry="healthcare")
        
        assert tech_config.get_industry_multiplier() == 1.0
        assert finance_config.get_industry_multiplier() == 2.5
        assert healthcare_config.get_industry_multiplier() == 3.0
    
    def test_custom_config(self):
        config = CostConfig(
            input_token_cost=0.01,
            output_token_cost=0.03,
            bug_cost_multiplier=2.0,
            industry="finance",
        )
        
        assert config.input_token_cost == 0.01
        assert config.bug_cost_multiplier == 2.0


class TestCostTracker:
    """Tests for CostTracker."""
    
    @pytest.fixture
    def tracker(self):
        return CostTracker()
    
    def test_record_cost(self, tracker):
        cost = tracker.record_cost(
            category=CostCategory.LLM_TOKENS,
            amount_usd=0.05,
            description="Test cost",
            repository="test/repo",
            pr_number=123,
        )
        
        assert cost.id == "cost-1"
        assert cost.amount_usd == 0.05
        assert cost.repository == "test/repo"
        assert len(tracker.costs) == 1
    
    def test_record_llm_usage(self, tracker):
        cost = tracker.record_llm_usage(
            input_tokens=1000,
            output_tokens=500,
            model="gpt-4",
            repository="test/repo",
        )
        
        # Expected: (1000/1000 * 0.003) + (500/1000 * 0.015) = 0.003 + 0.0075 = 0.0105
        assert cost.amount_usd == pytest.approx(0.0105, rel=1e-3)
        assert cost.category == CostCategory.LLM_TOKENS
    
    def test_record_z3_compute(self, tracker):
        cost = tracker.record_z3_compute(
            compute_seconds=120,  # 2 minutes
            repository="test/repo",
        )
        
        # Expected: 2 * 0.01 = 0.02
        assert cost.amount_usd == pytest.approx(0.02, rel=1e-3)
        assert cost.category == CostCategory.Z3_COMPUTE
    
    def test_get_total_cost(self, tracker):
        tracker.record_cost(CostCategory.LLM_TOKENS, 0.10, "Cost 1")
        tracker.record_cost(CostCategory.Z3_COMPUTE, 0.05, "Cost 2")
        tracker.record_cost(CostCategory.API_CALLS, 0.03, "Cost 3")
        
        total = tracker.get_total_cost()
        
        assert total == pytest.approx(0.18, rel=1e-3)
    
    def test_get_cost_by_category(self, tracker):
        tracker.record_cost(CostCategory.LLM_TOKENS, 0.10, "Cost 1")
        tracker.record_cost(CostCategory.LLM_TOKENS, 0.20, "Cost 2")
        tracker.record_cost(CostCategory.Z3_COMPUTE, 0.05, "Cost 3")
        
        by_category = tracker.get_cost_by_category()
        
        assert by_category["llm_tokens"] == pytest.approx(0.30, rel=1e-3)
        assert by_category["z3_compute"] == pytest.approx(0.05, rel=1e-3)
    
    def test_filter_by_period(self, tracker):
        # Record some costs
        tracker.record_cost(CostCategory.LLM_TOKENS, 0.10, "Old cost")
        tracker.costs[0].timestamp = datetime.utcnow() - timedelta(days=10)
        
        tracker.record_cost(CostCategory.LLM_TOKENS, 0.20, "Recent cost")
        
        # Filter to last 5 days
        since = datetime.utcnow() - timedelta(days=5)
        total = tracker.get_total_cost(since=since)
        
        assert total == pytest.approx(0.20, rel=1e-3)


class TestBugValueCalculator:
    """Tests for BugValueCalculator."""
    
    @pytest.fixture
    def calculator(self):
        return BugValueCalculator()
    
    def test_record_bug(self, calculator):
        bug = calculator.record_bug(
            severity=BugSeverity.CRITICAL,
            title="SQL Injection",
            description="Unsanitized input",
            repository="test/repo",
            pr_number=123,
        )
        
        assert bug.id == "bug-1"
        assert bug.severity == BugSeverity.CRITICAL
        assert bug.estimated_cost_avoided is not None
        assert bug.estimated_cost_avoided > 0
    
    def test_critical_bug_cost_estimate(self, calculator):
        bug = calculator.record_bug(
            severity=BugSeverity.CRITICAL,
            title="Critical bug",
            description="",
        )
        
        # Should be around avg critical cost (150000) with default multiplier
        expected = BUG_COST_ESTIMATES[BugSeverity.CRITICAL]["avg"]
        assert bug.estimated_cost_avoided == pytest.approx(expected, rel=0.1)
    
    def test_production_bug_multiplier(self, calculator):
        dev_bug = calculator.record_bug(
            severity=BugSeverity.HIGH,
            title="Dev bug",
            description="",
            was_production=False,
        )
        
        prod_bug = calculator.record_bug(
            severity=BugSeverity.HIGH,
            title="Prod bug",
            description="",
            was_production=True,
        )
        
        # Production bugs should cost more
        assert prod_bug.estimated_cost_avoided > dev_bug.estimated_cost_avoided
    
    def test_industry_multiplier(self):
        tech_calc = BugValueCalculator(CostConfig(industry="technology"))
        finance_calc = BugValueCalculator(CostConfig(industry="finance"))
        
        tech_bug = tech_calc.record_bug(BugSeverity.HIGH, "Bug", "")
        finance_bug = finance_calc.record_bug(BugSeverity.HIGH, "Bug", "")
        
        # Finance bugs should be more expensive
        assert finance_bug.estimated_cost_avoided > tech_bug.estimated_cost_avoided
    
    def test_get_bugs_by_severity(self, calculator):
        calculator.record_bug(BugSeverity.CRITICAL, "Bug 1", "")
        calculator.record_bug(BugSeverity.CRITICAL, "Bug 2", "")
        calculator.record_bug(BugSeverity.HIGH, "Bug 3", "")
        calculator.record_bug(BugSeverity.LOW, "Bug 4", "")
        
        by_severity = calculator.get_bugs_by_severity()
        
        assert by_severity["critical"] == 2
        assert by_severity["high"] == 1
        assert by_severity["low"] == 1
    
    def test_get_total_value(self, calculator):
        calculator.record_bug(BugSeverity.HIGH, "Bug 1", "")
        calculator.record_bug(BugSeverity.MEDIUM, "Bug 2", "")
        
        avg, min_val, max_val = calculator.get_total_value()
        
        assert avg > 0
        assert min_val <= avg <= max_val


class TestROIMetrics:
    """Tests for ROIMetrics."""
    
    def test_to_dict(self):
        metrics = ROIMetrics(
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 31),
            total_cost_usd=100.0,
            cost_by_category={"llm_tokens": 80.0, "z3_compute": 20.0},
            cost_per_pr=10.0,
            cost_per_1k_loc=0.5,
            bugs_caught=5,
            bugs_by_severity={"critical": 1, "high": 2, "medium": 2},
            estimated_cost_avoided=50000.0,
            estimated_cost_avoided_min=25000.0,
            estimated_cost_avoided_max=100000.0,
            roi_percentage=49900.0,
            net_savings=49900.0,
            payback_period_days=0.1,
            bugs_per_week=1.2,
            cost_trend="stable",
        )
        
        d = metrics.to_dict()
        
        assert d["costs"]["total_usd"] == 100.0
        assert d["value"]["bugs_caught"] == 5
        assert d["roi"]["percentage"] == 49900.0
    
    def test_to_summary_string(self):
        metrics = ROIMetrics(
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 1, 31),
            total_cost_usd=100.0,
            cost_by_category={},
            cost_per_pr=10.0,
            cost_per_1k_loc=0.5,
            bugs_caught=5,
            bugs_by_severity={"critical": 1},
            estimated_cost_avoided=50000.0,
            estimated_cost_avoided_min=25000.0,
            estimated_cost_avoided_max=100000.0,
            roi_percentage=49900.0,
            net_savings=49900.0,
            payback_period_days=0.1,
            bugs_per_week=1.2,
            cost_trend="stable",
        )
        
        summary = metrics.to_summary_string()
        
        assert "ROI Summary" in summary
        assert "$100.00" in summary
        assert "Critical: 1" in summary


class TestROIDashboard:
    """Tests for ROIDashboard."""
    
    @pytest.fixture
    def dashboard(self):
        return ROIDashboard()
    
    @pytest.fixture
    def dashboard_with_storage(self, tmp_path):
        return ROIDashboard(storage_path=str(tmp_path / "roi"))
    
    def test_record_pr_analysis(self, dashboard):
        result = dashboard.record_pr_analysis(
            repository="test/repo",
            pr_number=123,
            lines_of_code=500,
            input_tokens=2000,
            output_tokens=1000,
            z3_seconds=30,
            bugs_found=[
                {"severity": "high", "title": "Null pointer", "description": "NPE risk"}
            ],
        )
        
        assert result["pr"] == "test/repo#123"
        assert result["cost_usd"] > 0
        assert result["bugs_found"] == 1
        assert dashboard.prs_analyzed == 1
        assert dashboard.lines_of_code_analyzed == 500
    
    def test_calculate_roi(self, dashboard):
        # Record some activity
        dashboard.record_pr_analysis(
            repository="test/repo",
            pr_number=1,
            lines_of_code=1000,
            input_tokens=5000,
            output_tokens=2000,
            z3_seconds=60,
            bugs_found=[
                {"severity": "critical", "title": "Security bug"},
                {"severity": "high", "title": "Logic error"},
            ],
        )
        
        metrics = dashboard.calculate_roi()
        
        assert metrics.total_cost_usd > 0
        assert metrics.bugs_caught == 2
        assert metrics.estimated_cost_avoided > 0
        assert metrics.roi_percentage > 0  # Should be positive with bugs caught
    
    def test_generate_report_dict(self, dashboard):
        dashboard.record_pr_analysis(
            repository="test/repo",
            pr_number=1,
            lines_of_code=100,
            input_tokens=1000,
            output_tokens=500,
            z3_seconds=10,
        )
        
        report = dashboard.generate_report(format="dict")
        
        assert "period" in report
        assert "costs" in report
        assert "value" in report
        assert "roi" in report
    
    def test_generate_report_summary(self, dashboard):
        dashboard.record_pr_analysis(
            repository="test/repo",
            pr_number=1,
            lines_of_code=100,
            input_tokens=1000,
            output_tokens=500,
            z3_seconds=10,
        )
        
        summary = dashboard.generate_report(format="summary")
        
        assert isinstance(summary, str)
        assert "ROI Summary" in summary
    
    def test_get_recent_activity(self, dashboard):
        dashboard.record_pr_analysis(
            repository="test/repo",
            pr_number=1,
            lines_of_code=100,
            input_tokens=1000,
            output_tokens=500,
            z3_seconds=10,
            bugs_found=[{"severity": "medium", "title": "Bug"}],
        )
        
        activity = dashboard.get_recent_activity(limit=5)
        
        assert "recent_costs" in activity
        assert "recent_bugs" in activity
        assert len(activity["recent_bugs"]) == 1
    
    def test_export_for_audit(self, dashboard):
        dashboard.record_pr_analysis(
            repository="test/repo",
            pr_number=1,
            lines_of_code=100,
            input_tokens=1000,
            output_tokens=500,
            z3_seconds=10,
        )
        
        since = datetime.utcnow() - timedelta(days=1)
        until = datetime.utcnow() + timedelta(days=1)
        
        export = dashboard.export_for_audit(since, until)
        
        assert "export_timestamp" in export
        assert "summary" in export
        assert "detailed_costs" in export
        assert "metadata" in export
    
    def test_persistence(self, dashboard_with_storage):
        dashboard_with_storage.record_pr_analysis(
            repository="test/repo",
            pr_number=1,
            lines_of_code=200,
            input_tokens=1000,
            output_tokens=500,
            z3_seconds=10,
        )
        
        # Create new dashboard with same storage
        new_dashboard = ROIDashboard(
            storage_path=dashboard_with_storage.storage_path
        )
        
        assert new_dashboard.prs_analyzed == 1
        assert new_dashboard.lines_of_code_analyzed == 200


class TestVerificationCost:
    """Tests for VerificationCost dataclass."""
    
    def test_to_dict(self):
        cost = VerificationCost(
            id="cost-1",
            timestamp=datetime(2024, 1, 15, 12, 0),
            category=CostCategory.LLM_TOKENS,
            amount_usd=0.0123,
            description="Test cost",
            repository="test/repo",
            pr_number=42,
        )
        
        d = cost.to_dict()
        
        assert d["id"] == "cost-1"
        assert d["category"] == "llm_tokens"
        assert d["amount_usd"] == 0.0123
        assert d["pr_number"] == 42


class TestBugCaught:
    """Tests for BugCaught dataclass."""
    
    def test_to_dict(self):
        bug = BugCaught(
            id="bug-1",
            timestamp=datetime(2024, 1, 15, 12, 0),
            severity=BugSeverity.CRITICAL,
            title="SQL Injection",
            description="User input not sanitized",
            repository="test/repo",
            pr_number=42,
            estimated_cost_avoided=150000.0,
            was_production=True,
        )
        
        d = bug.to_dict()
        
        assert d["id"] == "bug-1"
        assert d["severity"] == "critical"
        assert d["title"] == "SQL Injection"
        assert d["estimated_cost_avoided"] == 150000.0


class TestCreateDashboard:
    """Tests for create_dashboard function."""
    
    def test_create_default(self):
        dashboard = create_dashboard()
        
        assert dashboard is not None
        assert dashboard.config is not None
    
    def test_create_with_config(self):
        config = CostConfig(industry="finance")
        dashboard = create_dashboard(config=config)
        
        assert dashboard.config.industry == "finance"
