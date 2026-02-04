"""Tests for Prometheus Metrics."""

import time
import pytest
import threading
from unittest.mock import patch, MagicMock

from codeverify_verifier.prometheus_metrics import (
    MetricType,
    MetricLabels,
    Counter,
    Gauge,
    Histogram,
    Summary,
    RuntimeProbeMetrics,
    format_prometheus,
    format_json,
    MetricsExporter,
    MetricsServer,
    observe_spec_check,
    get_global_metrics,
    reset_global_metrics,
    MetricsIntegration,
)


class TestCounter:
    """Tests for Counter metric."""

    def test_basic_increment(self):
        """Test basic counter increment."""
        counter = Counter("test_counter", "Test counter")
        
        counter.inc()
        assert counter.get() == 1.0
        
        counter.inc(5)
        assert counter.get() == 6.0

    def test_increment_with_labels(self):
        """Test counter with labels."""
        counter = Counter(
            "test_counter",
            "Test counter",
            labels=["method", "status"]
        )
        
        counter.inc(method="GET", status="200")
        counter.inc(method="GET", status="200")
        counter.inc(method="POST", status="200")
        
        assert counter.get(method="GET", status="200") == 2.0
        assert counter.get(method="POST", status="200") == 1.0
        assert counter.get(method="DELETE", status="404") == 0.0

    def test_negative_increment_raises(self):
        """Test that negative increment raises error."""
        counter = Counter("test_counter", "Test counter")
        
        with pytest.raises(ValueError):
            counter.inc(-1)

    def test_collect(self):
        """Test collecting counter values."""
        counter = Counter(
            "test_counter",
            "Test counter",
            labels=["label1"]
        )
        
        counter.inc(label1="a")
        counter.inc(label1="b")
        counter.inc(2, label1="a")
        
        collected = counter.collect()
        
        assert len(collected) == 2
        assert any(m["labels"]["label1"] == "a" and m["value"] == 3.0 for m in collected)
        assert any(m["labels"]["label1"] == "b" and m["value"] == 1.0 for m in collected)


class TestGauge:
    """Tests for Gauge metric."""

    def test_set_value(self):
        """Test setting gauge value."""
        gauge = Gauge("test_gauge", "Test gauge")
        
        gauge.set(42.0)
        assert gauge.get() == 42.0
        
        gauge.set(10.0)
        assert gauge.get() == 10.0

    def test_increment_decrement(self):
        """Test incrementing and decrementing gauge."""
        gauge = Gauge("test_gauge", "Test gauge")
        
        gauge.set(10.0)
        gauge.inc(5.0)
        assert gauge.get() == 15.0
        
        gauge.dec(3.0)
        assert gauge.get() == 12.0

    def test_gauge_with_labels(self):
        """Test gauge with labels."""
        gauge = Gauge(
            "test_gauge",
            "Test gauge",
            labels=["instance"]
        )
        
        gauge.set(100, instance="a")
        gauge.set(200, instance="b")
        
        assert gauge.get(instance="a") == 100
        assert gauge.get(instance="b") == 200


class TestHistogram:
    """Tests for Histogram metric."""

    def test_observe(self):
        """Test observing values."""
        histogram = Histogram(
            "test_histogram",
            "Test histogram",
            buckets=(0.1, 0.5, 1.0)
        )
        
        histogram.observe(0.05)  # In 0.1 bucket
        histogram.observe(0.3)   # In 0.5 bucket
        histogram.observe(0.8)   # In 1.0 bucket
        histogram.observe(2.0)   # Only in +Inf
        
        collected = histogram.collect()
        
        # Find bucket counts
        bucket_values = {
            m["labels"]["le"]: m["value"]
            for m in collected
            if "le" in m.get("labels", {})
        }
        
        assert bucket_values["0.1"] == 1  # 0.05
        assert bucket_values["0.5"] == 2  # 0.05, 0.3
        assert bucket_values["1.0"] == 3  # 0.05, 0.3, 0.8
        assert bucket_values["+Inf"] == 4  # all

    def test_observe_with_labels(self):
        """Test histogram with labels."""
        histogram = Histogram(
            "test_histogram",
            "Test histogram",
            labels=["method"],
            buckets=(0.1, 1.0)
        )
        
        histogram.observe(0.05, method="GET")
        histogram.observe(0.5, method="POST")
        
        collected = histogram.collect()
        
        # Should have separate series for each label combination
        get_metrics = [m for m in collected if m.get("labels", {}).get("method") == "GET"]
        post_metrics = [m for m in collected if m.get("labels", {}).get("method") == "POST"]
        
        assert len(get_metrics) > 0
        assert len(post_metrics) > 0

    def test_time_context_manager(self):
        """Test timing context manager."""
        histogram = Histogram(
            "test_histogram",
            "Test histogram",
            buckets=(0.001, 0.01, 0.1, 1.0)
        )
        
        with histogram.time():
            time.sleep(0.005)
        
        collected = histogram.collect()
        
        # Should have recorded something
        count_metric = next(m for m in collected if m["name"] == "test_histogram_count")
        assert count_metric["value"] == 1


class TestSummary:
    """Tests for Summary metric."""

    def test_observe(self):
        """Test observing values."""
        summary = Summary(
            "test_summary",
            "Test summary",
            quantiles=(0.5, 0.9)
        )
        
        # Observe values 1-10
        for i in range(1, 11):
            summary.observe(float(i))
        
        collected = summary.collect()
        
        # Should have quantiles
        assert any(m.get("labels", {}).get("quantile") == "0.5" for m in collected)
        assert any(m.get("labels", {}).get("quantile") == "0.9" for m in collected)

    def test_max_age(self):
        """Test that old observations expire."""
        summary = Summary(
            "test_summary",
            "Test summary",
            max_age_seconds=1  # 1 second
        )
        
        summary.observe(100.0)
        collected1 = summary.collect()
        
        # Wait for expiry
        time.sleep(1.5)
        summary.observe(50.0)
        
        collected2 = summary.collect()
        
        # Old value should be expired
        count_metric = next(
            (m for m in collected2 if m["name"] == "test_summary_count"),
            None
        )
        assert count_metric is not None
        assert count_metric["value"] == 1  # Only new observation


class TestRuntimeProbeMetrics:
    """Tests for RuntimeProbeMetrics."""

    def test_record_check(self):
        """Test recording specification checks."""
        metrics = RuntimeProbeMetrics()
        
        metrics.record_check(
            spec_id="auth_check",
            probe_type="precondition",
            passed=True,
            duration_seconds=0.001
        )
        
        collected = metrics.collect_all()
        
        # Should have counter and histogram entries
        assert any(m["name"].endswith("_total") for m in collected)
        assert any("_duration_" in m["name"] for m in collected)

    def test_record_violation(self):
        """Test recording violations."""
        metrics = RuntimeProbeMetrics()
        
        metrics.record_violation(
            spec_id="auth_check",
            probe_type="precondition",
            severity="high"
        )
        
        collected = metrics.collect_all()
        
        violation_metrics = [m for m in collected if "violations" in m["name"]]
        assert len(violation_metrics) > 0

    def test_set_active_specs(self):
        """Test setting active specs gauge."""
        metrics = RuntimeProbeMetrics()
        
        metrics.set_active_specs("precondition", 5)
        metrics.set_active_specs("postcondition", 3)
        
        assert metrics.active_specs.get(probe_type="precondition") == 5
        assert metrics.active_specs.get(probe_type="postcondition") == 3

    def test_record_error(self):
        """Test recording errors."""
        metrics = RuntimeProbeMetrics()
        
        metrics.record_error("ValueError")
        metrics.record_error("ValueError")
        metrics.record_error("TypeError")
        
        collected = metrics.collect_all()
        
        error_metrics = [m for m in collected if "errors" in m["name"]]
        assert len(error_metrics) > 0


class TestFormatFunctions:
    """Tests for formatting functions."""

    def test_format_prometheus(self):
        """Test Prometheus format output."""
        metrics = [
            {
                "name": "test_counter",
                "type": "counter",
                "value": 42.0,
                "labels": {"label1": "value1"}
            }
        ]
        
        output = format_prometheus(metrics)
        
        assert "# HELP test_counter" in output
        assert "# TYPE test_counter counter" in output
        assert 'test_counter{label1="value1"} 42.0' in output

    def test_format_prometheus_no_labels(self):
        """Test Prometheus format without labels."""
        metrics = [
            {
                "name": "test_gauge",
                "type": "gauge",
                "value": 100.0,
                "labels": {}
            }
        ]
        
        output = format_prometheus(metrics)
        
        assert "test_gauge 100.0" in output

    def test_format_json(self):
        """Test JSON format output."""
        import json
        
        metrics = [
            {
                "name": "test_counter",
                "type": "counter",
                "value": 42.0,
                "labels": {"label1": "value1"}
            }
        ]
        
        output = format_json(metrics)
        parsed = json.loads(output)
        
        assert len(parsed) == 1
        assert parsed[0]["name"] == "test_counter"
        assert parsed[0]["value"] == 42.0


class TestMetricsExporter:
    """Tests for MetricsExporter."""

    def test_to_prometheus(self):
        """Test exporting to Prometheus format."""
        metrics = RuntimeProbeMetrics()
        metrics.record_check("test", "precondition", True, 0.001)
        
        exporter = MetricsExporter(metrics)
        output = exporter.to_prometheus()
        
        assert "# HELP" in output
        assert "# TYPE" in output

    def test_to_json(self):
        """Test exporting to JSON format."""
        import json
        
        metrics = RuntimeProbeMetrics()
        metrics.record_check("test", "precondition", True, 0.001)
        
        exporter = MetricsExporter(metrics)
        output = exporter.to_json()
        
        parsed = json.loads(output)
        assert isinstance(parsed, list)


class TestObserveSpecCheckDecorator:
    """Tests for observe_spec_check decorator."""

    def test_successful_check(self):
        """Test decorator with successful check."""
        metrics = RuntimeProbeMetrics()
        
        @observe_spec_check(metrics, "test_spec", "assertion")
        def check_positive(x):
            return x > 0
        
        result = check_positive(5)
        
        assert result is True
        assert metrics.spec_checks_total.get(
            spec_id="test_spec",
            probe_type="assertion",
            result="pass"
        ) == 1

    def test_failed_check(self):
        """Test decorator with failed check."""
        metrics = RuntimeProbeMetrics()
        
        @observe_spec_check(metrics, "test_spec", "assertion")
        def check_positive(x):
            return x > 0
        
        result = check_positive(-5)
        
        assert result is False
        assert metrics.spec_checks_total.get(
            spec_id="test_spec",
            probe_type="assertion",
            result="fail"
        ) == 1

    def test_exception_handling(self):
        """Test decorator handles exceptions."""
        metrics = RuntimeProbeMetrics()
        
        @observe_spec_check(metrics, "test_spec", "assertion")
        def check_raises(x):
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            check_raises(5)
        
        # Should have recorded the error
        assert metrics.verification_errors_total.get(error_type="ValueError") == 1


class TestGlobalMetrics:
    """Tests for global metrics instance."""

    def test_get_global_metrics(self):
        """Test getting global metrics instance."""
        reset_global_metrics()
        
        metrics1 = get_global_metrics()
        metrics2 = get_global_metrics()
        
        assert metrics1 is metrics2

    def test_reset_global_metrics(self):
        """Test resetting global metrics."""
        metrics1 = get_global_metrics()
        reset_global_metrics()
        metrics2 = get_global_metrics()
        
        assert metrics1 is not metrics2


class TestMetricsIntegration:
    """Tests for MetricsIntegration."""

    def test_on_spec_registered(self):
        """Test spec registration tracking."""
        metrics = RuntimeProbeMetrics()
        integration = MetricsIntegration(metrics)
        
        integration.on_spec_registered("spec1", "precondition")
        integration.on_spec_registered("spec2", "precondition")
        
        assert metrics.active_specs.get(probe_type="precondition") == 2

    def test_on_spec_unregistered(self):
        """Test spec unregistration tracking."""
        metrics = RuntimeProbeMetrics()
        integration = MetricsIntegration(metrics)
        
        integration.on_spec_registered("spec1", "precondition")
        integration.on_spec_registered("spec2", "precondition")
        integration.on_spec_unregistered("spec1", "precondition")
        
        assert metrics.active_specs.get(probe_type="precondition") == 1

    def test_on_check_performed(self):
        """Test check tracking."""
        metrics = RuntimeProbeMetrics()
        integration = MetricsIntegration(metrics)
        
        integration.on_check_performed(
            spec_id="test",
            probe_type="precondition",
            passed=True,
            duration_seconds=0.001,
            function_name="my_func"
        )
        
        assert metrics.spec_checks_total.get(
            spec_id="test",
            probe_type="precondition",
            result="pass"
        ) == 1

    def test_on_violation(self):
        """Test violation tracking."""
        metrics = RuntimeProbeMetrics()
        integration = MetricsIntegration(metrics)
        
        integration.on_violation("test", "precondition", "high")
        
        assert metrics.spec_violations_total.get(
            spec_id="test",
            probe_type="precondition",
            severity="high"
        ) == 1

    def test_on_error(self):
        """Test error tracking."""
        metrics = RuntimeProbeMetrics()
        integration = MetricsIntegration(metrics)
        
        integration.on_error(ValueError("test"))
        
        assert metrics.verification_errors_total.get(error_type="ValueError") == 1


class TestMetricLabels:
    """Tests for MetricLabels."""

    def test_to_dict(self):
        """Test conversion to dict."""
        labels = MetricLabels(
            spec_id="test",
            probe_type="precondition",
            result="pass"
        )
        
        d = labels.to_dict()
        
        assert d["spec_id"] == "test"
        assert d["probe_type"] == "precondition"
        assert d["result"] == "pass"
        assert "severity" not in d  # Empty values excluded

    def test_to_label_string(self):
        """Test conversion to Prometheus label string."""
        labels = MetricLabels(
            spec_id="test",
            result="fail"
        )
        
        label_str = labels.to_label_string()
        
        assert 'spec_id="test"' in label_str
        assert 'result="fail"' in label_str


class TestThreadSafety:
    """Tests for thread safety."""

    def test_counter_thread_safety(self):
        """Test counter is thread-safe."""
        counter = Counter("test", "Test", labels=["id"])
        
        def increment():
            for _ in range(1000):
                counter.inc(id="test")
        
        threads = [threading.Thread(target=increment) for _ in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert counter.get(id="test") == 10000

    def test_gauge_thread_safety(self):
        """Test gauge is thread-safe."""
        gauge = Gauge("test", "Test")
        
        def update():
            for i in range(1000):
                gauge.set(float(i))
        
        threads = [threading.Thread(target=update) for _ in range(5)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Should be a valid value (no corruption)
        value = gauge.get()
        assert 0 <= value < 1000
