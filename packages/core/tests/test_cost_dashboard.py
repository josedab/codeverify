"""Tests for CI/CD Cost & Time Dashboard."""

from codeverify_core.cost_dashboard import (
    BudgetConfig,
    CostMetricsCollector,
    CostPrometheusMetrics,
    get_cost_collector,
    reset_cost_collector,
)


class TestCostMetricsCollector:
    def test_record_llm_usage(self):
        c = CostMetricsCollector()
        record = c.record_llm_usage("gpt-4", input_tokens=1000, output_tokens=500)
        assert record.amount_usd > 0
        assert record.category == "llm_tokens"
        assert c._llm_tokens_total == 1500

    def test_record_z3_time(self):
        c = CostMetricsCollector()
        record = c.record_z3_time(5000.0)  # 5 seconds
        assert record.amount_usd > 0
        assert c._z3_time_total_ms == 5000.0

    def test_cache_tracking(self):
        c = CostMetricsCollector()
        c.record_cache_hit(saved_ms=50)
        c.record_cache_hit(saved_ms=30)
        c.record_cache_miss()
        summary = c.get_summary()
        assert summary["cache_hits"] == 2
        assert summary["cache_misses"] == 1
        assert abs(summary["cache_hit_rate"] - 0.667) < 0.01

    def test_total_cost(self):
        c = CostMetricsCollector()
        c.record_llm_usage("gpt-4", 1000, 500)
        c.record_z3_time(2000)
        total = c.get_total_cost()
        assert total > 0

    def test_cost_by_category(self):
        c = CostMetricsCollector()
        c.record_llm_usage("gpt-4", 500, 200)
        c.record_z3_time(1000)
        summary = c.get_summary()
        assert "llm_tokens" in summary["cost_by_category"]
        assert "z3_compute" in summary["cost_by_category"]

    def test_cost_by_model(self):
        c = CostMetricsCollector()
        c.record_llm_usage("gpt-4", 500, 200)
        c.record_llm_usage("claude", 300, 100)
        summary = c.get_summary()
        assert "gpt-4" in summary["cost_by_model"]
        assert "claude" in summary["cost_by_model"]

    def test_get_records(self):
        c = CostMetricsCollector()
        c.record_llm_usage("gpt-4", 100, 50)
        c.record_z3_time(500)
        records = c.get_records()
        assert len(records) == 2

    def test_get_records_filtered(self):
        c = CostMetricsCollector()
        c.record_llm_usage("gpt-4", 100, 50)
        c.record_z3_time(500)
        records = c.get_records(category="llm_tokens")
        assert len(records) == 1


class TestBudgetTracking:
    def test_within_budget(self):
        budget = BudgetConfig(monthly_budget_usd=10.0, hard_limit=True)
        c = CostMetricsCollector(budget)
        c.record_llm_usage("gpt-4", 100, 50)
        assert c.is_within_budget() is True

    def test_over_budget_hard_limit(self):
        budget = BudgetConfig(monthly_budget_usd=0.001, hard_limit=True)
        c = CostMetricsCollector(budget)
        c.record_llm_usage("gpt-4", 10000, 5000)
        assert c.is_within_budget() is False

    def test_budget_usage_report(self):
        budget = BudgetConfig(monthly_budget_usd=100.0)
        c = CostMetricsCollector(budget)
        c.record_llm_usage("gpt-4", 1000, 500)
        usage = c.get_budget_usage()
        assert usage["monthly_budget_usd"] == 100.0
        assert usage["usage_pct"] >= 0
        assert usage["remaining_usd"] > 0

    def test_alert_threshold(self):
        budget = BudgetConfig(monthly_budget_usd=0.01, alert_threshold_pct=50.0)
        c = CostMetricsCollector(budget)
        c.record_llm_usage("gpt-4", 5000, 2000)
        usage = c.get_budget_usage()
        assert usage["alert_triggered"] is True


class TestCostPrometheusMetrics:
    def test_export_format(self):
        c = CostMetricsCollector()
        c.record_llm_usage("gpt-4", 500, 200)
        c.record_z3_time(1000)
        c.record_cache_hit()
        c.record_analysis_start()

        metrics = CostPrometheusMetrics(c)
        text = metrics.export()

        assert "codeverify_cost_total_usd" in text
        assert "codeverify_cost_llm_tokens_total" in text
        assert "codeverify_cost_z3_time_ms_total" in text
        assert "codeverify_cost_cache_hits_total" in text
        assert "codeverify_cost_analyses_total" in text
        assert "# TYPE" in text
        assert 'model="gpt-4"' in text


class TestSingleton:
    def test_get_and_reset(self):
        reset_cost_collector()
        c1 = get_cost_collector()
        c2 = get_cost_collector()
        assert c1 is c2
        reset_cost_collector()
        c3 = get_cost_collector()
        assert c3 is not c1
