"""Tests for Telemetry & ROI Analytics module."""

from codeverify_core.telemetry import (
    CostEstimator,
    FindingLifecycle,
    FindingMetrics,
    ROIDashboard,
    ROIReport,
    TelemetryCollector,
    TelemetryEvent,
)


class TestFindingLifecycle:
    def test_stages_exist(self):
        assert FindingLifecycle.DETECTED is not None
        assert FindingLifecycle.FIXED is not None


class TestTelemetryEvent:
    def test_creation(self):
        event = TelemetryEvent(event_type="analysis_complete", repo_id="test/repo")
        assert event.event_type == "analysis_complete"
        assert event.repo_id == "test/repo"
        assert len(event.event_id) > 0


class TestFindingMetrics:
    def test_creation(self):
        metrics = FindingMetrics(finding_id="f1")
        assert metrics.finding_id == "f1"


class TestCostEstimator:
    def test_creation(self):
        estimator = CostEstimator()
        assert estimator is not None

    def test_estimate_bug_cost(self):
        estimator = CostEstimator()
        cost = estimator.estimate_bug_cost("critical")
        assert cost > 0

    def test_estimate_bug_cost_low(self):
        estimator = CostEstimator()
        cost_low = estimator.estimate_bug_cost("low")
        cost_critical = estimator.estimate_bug_cost("critical")
        assert cost_critical > cost_low


class TestTelemetryCollector:
    def test_creation(self):
        collector = TelemetryCollector()
        assert collector is not None

    def test_record_event(self):
        collector = TelemetryCollector()
        collector.record_event(event_type="scan_started", repo_id="test/repo")
        assert len(collector.events) >= 1

    def test_record_multiple_events(self):
        collector = TelemetryCollector()
        for i in range(5):
            collector.record_event(event_type=f"event_{i}", repo_id="test/repo")
        assert len(collector.events) >= 5

    def test_record_finding_detected(self):
        collector = TelemetryCollector()
        metrics = collector.record_finding_detected(
            finding_id="f1", repo_id="test/repo", severity="high"
        )
        assert metrics is not None
        assert metrics.finding_id == "f1"


class TestROIReport:
    def test_has_attributes(self):
        collector = TelemetryCollector()
        dashboard = ROIDashboard(collector=collector)
        report = dashboard.generate_report(period_days=30)
        assert isinstance(report, ROIReport)


class TestROIDashboard:
    def test_creation(self):
        collector = TelemetryCollector()
        dashboard = ROIDashboard(collector=collector)
        assert dashboard is not None

    def test_generate_report(self):
        collector = TelemetryCollector()
        dashboard = ROIDashboard(collector=collector)
        report = dashboard.generate_report(period_days=30)
        assert isinstance(report, ROIReport)

    def test_generate_executive_summary(self):
        collector = TelemetryCollector()
        dashboard = ROIDashboard(collector=collector)
        report = dashboard.generate_report(period_days=30)
        summary = dashboard.generate_executive_summary(report)
        assert isinstance(summary, str)
