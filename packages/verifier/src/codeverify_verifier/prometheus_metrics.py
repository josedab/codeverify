"""
Prometheus Metrics for Runtime Probes

Provides observability for runtime verification through Prometheus metrics:
- Specification check counts (pass/fail)
- Violation rates by spec type
- Latency histograms for verification checks
- Gauge for active specs and violations
- Integration with common exporters

Compatible with:
- Prometheus Python client
- OpenMetrics format
- Pushgateway for batch jobs
- Service discovery
"""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional
from functools import wraps
import json
import http.server
import socketserver


class MetricType(Enum):
    """Prometheus metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricLabels:
    """Labels for a metric."""
    spec_id: str = ""
    probe_type: str = ""
    function_name: str = ""
    result: str = ""  # "pass", "fail", "error"
    severity: str = ""  # "critical", "high", "medium", "low"

    def to_dict(self) -> dict[str, str]:
        return {k: v for k, v in vars(self).items() if v}

    def to_label_string(self) -> str:
        """Convert to Prometheus label format."""
        pairs = [f'{k}="{v}"' for k, v in self.to_dict().items()]
        return "{" + ",".join(pairs) + "}" if pairs else ""


@dataclass
class HistogramBucket:
    """A histogram bucket with upper bound and count."""
    upper_bound: float
    count: int = 0


@dataclass
class MetricValue:
    """Value container for a metric."""
    value: float = 0.0
    labels: MetricLabels = field(default_factory=MetricLabels)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    histogram_buckets: list[HistogramBucket] = field(default_factory=list)
    histogram_sum: float = 0.0
    histogram_count: int = 0


class Counter:
    """Prometheus-style counter metric."""

    def __init__(self, name: str, description: str, labels: list[str] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0, **labels) -> None:
        """Increment counter."""
        if value < 0:
            raise ValueError("Counter can only increase")
        
        label_key = self._label_key(labels)
        with self._lock:
            self._values[label_key] += value

    def _label_key(self, labels: dict) -> tuple:
        """Create hashable key from labels."""
        return tuple(labels.get(l, "") for l in self.label_names)

    def get(self, **labels) -> float:
        """Get current counter value."""
        label_key = self._label_key(labels)
        return self._values.get(label_key, 0.0)

    def collect(self) -> list[dict]:
        """Collect all metric values for export."""
        results = []
        with self._lock:
            for label_key, value in self._values.items():
                label_dict = dict(zip(self.label_names, label_key))
                results.append({
                    "name": self.name,
                    "type": "counter",
                    "value": value,
                    "labels": label_dict
                })
        return results


class Gauge:
    """Prometheus-style gauge metric."""

    def __init__(self, name: str, description: str, labels: list[str] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: dict[tuple, float] = defaultdict(float)
        self._lock = threading.Lock()

    def set(self, value: float, **labels) -> None:
        """Set gauge value."""
        label_key = self._label_key(labels)
        with self._lock:
            self._values[label_key] = value

    def inc(self, value: float = 1.0, **labels) -> None:
        """Increment gauge."""
        label_key = self._label_key(labels)
        with self._lock:
            self._values[label_key] += value

    def dec(self, value: float = 1.0, **labels) -> None:
        """Decrement gauge."""
        label_key = self._label_key(labels)
        with self._lock:
            self._values[label_key] -= value

    def _label_key(self, labels: dict) -> tuple:
        return tuple(labels.get(l, "") for l in self.label_names)

    def get(self, **labels) -> float:
        label_key = self._label_key(labels)
        return self._values.get(label_key, 0.0)

    def collect(self) -> list[dict]:
        results = []
        with self._lock:
            for label_key, value in self._values.items():
                label_dict = dict(zip(self.label_names, label_key))
                results.append({
                    "name": self.name,
                    "type": "gauge",
                    "value": value,
                    "labels": label_dict
                })
        return results


class Histogram:
    """Prometheus-style histogram metric."""

    DEFAULT_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

    def __init__(
        self, 
        name: str, 
        description: str, 
        labels: list[str] = None,
        buckets: tuple[float, ...] = None
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self._values: dict[tuple, dict] = {}
        self._lock = threading.Lock()

    def observe(self, value: float, **labels) -> None:
        """Observe a value."""
        label_key = self._label_key(labels)
        
        with self._lock:
            if label_key not in self._values:
                self._values[label_key] = {
                    "buckets": {b: 0 for b in self.buckets},
                    "sum": 0.0,
                    "count": 0
                }
            
            data = self._values[label_key]
            data["sum"] += value
            data["count"] += 1
            
            for bucket in self.buckets:
                if value <= bucket:
                    data["buckets"][bucket] += 1

    def _label_key(self, labels: dict) -> tuple:
        return tuple(labels.get(l, "") for l in self.label_names)

    def time(self, **labels):
        """Context manager for timing operations."""
        return _HistogramTimer(self, labels)

    def collect(self) -> list[dict]:
        results = []
        with self._lock:
            for label_key, data in self._values.items():
                label_dict = dict(zip(self.label_names, label_key))
                
                # Bucket metrics
                cumulative = 0
                for bucket, count in sorted(data["buckets"].items()):
                    cumulative += count
                    results.append({
                        "name": f"{self.name}_bucket",
                        "type": "histogram",
                        "value": cumulative,
                        "labels": {**label_dict, "le": str(bucket)}
                    })
                
                # +Inf bucket
                results.append({
                    "name": f"{self.name}_bucket",
                    "type": "histogram", 
                    "value": data["count"],
                    "labels": {**label_dict, "le": "+Inf"}
                })
                
                # Sum and count
                results.append({
                    "name": f"{self.name}_sum",
                    "type": "histogram",
                    "value": data["sum"],
                    "labels": label_dict
                })
                results.append({
                    "name": f"{self.name}_count",
                    "type": "histogram",
                    "value": data["count"],
                    "labels": label_dict
                })
        
        return results


class _HistogramTimer:
    """Context manager for histogram timing."""

    def __init__(self, histogram: Histogram, labels: dict):
        self.histogram = histogram
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        self.histogram.observe(elapsed, **self.labels)
        return False


class Summary:
    """Prometheus-style summary metric (quantiles)."""

    def __init__(
        self,
        name: str,
        description: str,
        labels: list[str] = None,
        max_age_seconds: int = 600,
        quantiles: tuple[float, ...] = (0.5, 0.9, 0.99)
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.max_age_seconds = max_age_seconds
        self.quantiles = quantiles
        self._values: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
        self._lock = threading.Lock()

    def observe(self, value: float, **labels) -> None:
        """Observe a value."""
        label_key = self._label_key(labels)
        now = time.time()
        
        with self._lock:
            # Add new observation
            self._values[label_key].append((now, value))
            
            # Remove old observations
            cutoff = now - self.max_age_seconds
            self._values[label_key] = [
                (t, v) for t, v in self._values[label_key]
                if t > cutoff
            ]

    def _label_key(self, labels: dict) -> tuple:
        return tuple(labels.get(l, "") for l in self.label_names)

    def _calculate_quantile(self, values: list[float], q: float) -> float:
        """Calculate quantile value."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * q)
        idx = min(idx, len(sorted_values) - 1)
        return sorted_values[idx]

    def collect(self) -> list[dict]:
        results = []
        
        with self._lock:
            for label_key, observations in self._values.items():
                label_dict = dict(zip(self.label_names, label_key))
                values = [v for _, v in observations]
                
                if values:
                    # Quantiles
                    for q in self.quantiles:
                        results.append({
                            "name": self.name,
                            "type": "summary",
                            "value": self._calculate_quantile(values, q),
                            "labels": {**label_dict, "quantile": str(q)}
                        })
                    
                    # Sum and count
                    results.append({
                        "name": f"{self.name}_sum",
                        "type": "summary",
                        "value": sum(values),
                        "labels": label_dict
                    })
                    results.append({
                        "name": f"{self.name}_count",
                        "type": "summary",
                        "value": len(values),
                        "labels": label_dict
                    })
        
        return results


class RuntimeProbeMetrics:
    """
    Prometheus metrics specifically for CodeVerify runtime probes.
    
    Provides comprehensive observability for:
    - Specification check performance
    - Violation tracking
    - System health monitoring
    """

    def __init__(self, prefix: str = "codeverify"):
        self.prefix = prefix
        
        # Core verification metrics
        self.spec_checks_total = Counter(
            f"{prefix}_spec_checks_total",
            "Total number of specification checks performed",
            labels=["spec_id", "probe_type", "result"]
        )
        
        self.spec_violations_total = Counter(
            f"{prefix}_spec_violations_total",
            "Total number of specification violations detected",
            labels=["spec_id", "probe_type", "severity"]
        )
        
        self.spec_check_duration_seconds = Histogram(
            f"{prefix}_spec_check_duration_seconds",
            "Time spent checking specifications",
            labels=["spec_id", "probe_type"],
            buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)
        )
        
        # Active state metrics
        self.active_specs = Gauge(
            f"{prefix}_active_specs",
            "Number of currently active specifications",
            labels=["probe_type"]
        )
        
        self.pending_violations = Gauge(
            f"{prefix}_pending_violations",
            "Number of unresolved violations",
            labels=["severity"]
        )
        
        # System health metrics
        self.verification_errors_total = Counter(
            f"{prefix}_verification_errors_total",
            "Total number of errors during verification",
            labels=["error_type"]
        )
        
        self.spec_compilation_duration_seconds = Histogram(
            f"{prefix}_spec_compilation_duration_seconds",
            "Time spent compiling specifications",
            labels=["spec_id"]
        )
        
        # Latency summary for critical path
        self.critical_path_latency = Summary(
            f"{prefix}_critical_path_latency_seconds",
            "Latency of specification checks on critical code paths",
            labels=["function_name"]
        )
        
        # Rate metrics (derived from counters)
        self.violation_rate = Gauge(
            f"{prefix}_violation_rate_per_minute",
            "Rate of violations per minute",
            labels=["spec_id"]
        )

    def record_check(
        self,
        spec_id: str,
        probe_type: str,
        passed: bool,
        duration_seconds: float
    ) -> None:
        """Record a specification check."""
        result = "pass" if passed else "fail"
        
        self.spec_checks_total.inc(
            spec_id=spec_id,
            probe_type=probe_type,
            result=result
        )
        
        self.spec_check_duration_seconds.observe(
            duration_seconds,
            spec_id=spec_id,
            probe_type=probe_type
        )

    def record_violation(
        self,
        spec_id: str,
        probe_type: str,
        severity: str = "medium"
    ) -> None:
        """Record a specification violation."""
        self.spec_violations_total.inc(
            spec_id=spec_id,
            probe_type=probe_type,
            severity=severity
        )

    def record_error(self, error_type: str) -> None:
        """Record a verification error."""
        self.verification_errors_total.inc(error_type=error_type)

    def set_active_specs(self, probe_type: str, count: int) -> None:
        """Set the number of active specs."""
        self.active_specs.set(count, probe_type=probe_type)

    def set_pending_violations(self, severity: str, count: int) -> None:
        """Set the number of pending violations."""
        self.pending_violations.set(count, severity=severity)

    def collect_all(self) -> list[dict]:
        """Collect all metrics for export."""
        metrics = []
        metrics.extend(self.spec_checks_total.collect())
        metrics.extend(self.spec_violations_total.collect())
        metrics.extend(self.spec_check_duration_seconds.collect())
        metrics.extend(self.active_specs.collect())
        metrics.extend(self.pending_violations.collect())
        metrics.extend(self.verification_errors_total.collect())
        metrics.extend(self.spec_compilation_duration_seconds.collect())
        metrics.extend(self.critical_path_latency.collect())
        metrics.extend(self.violation_rate.collect())
        return metrics


def format_prometheus(metrics: list[dict]) -> str:
    """
    Format metrics in Prometheus exposition format.
    
    Example output:
    # HELP codeverify_spec_checks_total Total number of specification checks
    # TYPE codeverify_spec_checks_total counter
    codeverify_spec_checks_total{spec_id="auth_check",probe_type="precondition",result="pass"} 1234
    """
    lines = []
    seen_help = set()
    
    for metric in metrics:
        name = metric["name"]
        
        # Add HELP and TYPE once per metric
        if name not in seen_help:
            seen_help.add(name)
            lines.append(f"# HELP {name} Metric {name}")
            lines.append(f"# TYPE {name} {metric['type']}")
        
        # Format labels
        labels = metric.get("labels", {})
        if labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
            lines.append(f"{name}{{{label_str}}} {metric['value']}")
        else:
            lines.append(f"{name} {metric['value']}")
    
    return "\n".join(lines)


def format_json(metrics: list[dict]) -> str:
    """Format metrics as JSON for easier parsing."""
    return json.dumps(metrics, indent=2, default=str)


class MetricsExporter:
    """Export metrics to various destinations."""

    def __init__(self, metrics: RuntimeProbeMetrics):
        self.metrics = metrics

    def to_prometheus(self) -> str:
        """Export in Prometheus format."""
        return format_prometheus(self.metrics.collect_all())

    def to_json(self) -> str:
        """Export in JSON format."""
        return format_json(self.metrics.collect_all())

    def push_to_pushgateway(
        self,
        gateway_url: str,
        job: str = "codeverify",
        grouping_key: dict = None
    ) -> bool:
        """
        Push metrics to Prometheus Pushgateway.
        
        Args:
            gateway_url: URL of the pushgateway (e.g., "localhost:9091")
            job: Job name for grouping
            grouping_key: Additional grouping labels
            
        Returns:
            True if push succeeded
        """
        import urllib.request
        import urllib.error
        
        # Build URL
        url = f"http://{gateway_url}/metrics/job/{job}"
        if grouping_key:
            for key, value in grouping_key.items():
                url += f"/{key}/{value}"
        
        # Prepare data
        data = self.to_prometheus().encode('utf-8')
        
        try:
            req = urllib.request.Request(url, data=data, method='POST')
            req.add_header('Content-Type', 'text/plain')
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200
                
        except urllib.error.URLError as e:
            print(f"Failed to push to pushgateway: {e}")
            return False


class MetricsHTTPHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for metrics endpoint."""

    metrics_instance: RuntimeProbeMetrics = None

    def do_GET(self):
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            
            if self.metrics_instance:
                output = format_prometheus(self.metrics_instance.collect_all())
            else:
                output = "# No metrics available\n"
            
            self.wfile.write(output.encode('utf-8'))
            
        elif self.path == "/metrics/json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            if self.metrics_instance:
                output = format_json(self.metrics_instance.collect_all())
            else:
                output = "[]"
            
            self.wfile.write(output.encode('utf-8'))
            
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')
            
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress logging


class MetricsServer:
    """
    HTTP server for exposing metrics endpoint.
    
    Usage:
        metrics = RuntimeProbeMetrics()
        server = MetricsServer(metrics, port=8000)
        server.start()
        # ... application runs ...
        server.stop()
    """

    def __init__(self, metrics: RuntimeProbeMetrics, port: int = 8000, host: str = "0.0.0.0"):
        self.metrics = metrics
        self.port = port
        self.host = host
        self._server = None
        self._thread = None

    def start(self) -> None:
        """Start the metrics server in a background thread."""
        MetricsHTTPHandler.metrics_instance = self.metrics
        
        self._server = socketserver.TCPServer(
            (self.host, self.port),
            MetricsHTTPHandler
        )
        self._server.socket.setsockopt(
            socketserver.socket.SOL_SOCKET,
            socketserver.socket.SO_REUSEADDR,
            1
        )
        
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"Metrics server started on http://{self.host}:{self.port}/metrics")

    def stop(self) -> None:
        """Stop the metrics server."""
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None


def observe_spec_check(
    metrics: RuntimeProbeMetrics,
    spec_id: str,
    probe_type: str = "assertion"
) -> Callable[[Callable], Callable]:
    """
    Decorator to observe specification check metrics.
    
    Usage:
        @observe_spec_check(metrics, "auth_check", "precondition")
        def check_authentication(user):
            return user.is_authenticated
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                passed = bool(result)
                duration = time.perf_counter() - start
                
                metrics.record_check(spec_id, probe_type, passed, duration)
                
                if not passed:
                    metrics.record_violation(spec_id, probe_type)
                
                return result
                
            except Exception as e:
                duration = time.perf_counter() - start
                metrics.record_check(spec_id, probe_type, False, duration)
                metrics.record_error(type(e).__name__)
                raise
        
        return wrapper
    return decorator


# Global metrics instance
_global_metrics: Optional[RuntimeProbeMetrics] = None


def get_global_metrics() -> RuntimeProbeMetrics:
    """Get or create global metrics instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = RuntimeProbeMetrics()
    return _global_metrics


def reset_global_metrics() -> None:
    """Reset global metrics (for testing)."""
    global _global_metrics
    _global_metrics = None


# Example integration with RuntimeMonitor
class MetricsIntegration:
    """
    Integration layer between RuntimeMonitor and Prometheus metrics.
    
    Automatically records metrics for all spec checks.
    """

    def __init__(self, metrics: RuntimeProbeMetrics = None):
        self.metrics = metrics or get_global_metrics()
        self._violation_counts: dict[str, int] = defaultdict(int)
        self._last_rate_update = time.time()

    def on_spec_registered(self, spec_id: str, probe_type: str) -> None:
        """Called when a spec is registered."""
        # Increment active specs gauge
        current = self.metrics.active_specs.get(probe_type=probe_type)
        self.metrics.active_specs.set(current + 1, probe_type=probe_type)

    def on_spec_unregistered(self, spec_id: str, probe_type: str) -> None:
        """Called when a spec is unregistered."""
        current = self.metrics.active_specs.get(probe_type=probe_type)
        self.metrics.active_specs.set(max(0, current - 1), probe_type=probe_type)

    def on_check_performed(
        self,
        spec_id: str,
        probe_type: str,
        passed: bool,
        duration_seconds: float,
        function_name: str = ""
    ) -> None:
        """Called after each spec check."""
        self.metrics.record_check(spec_id, probe_type, passed, duration_seconds)
        
        if function_name:
            self.metrics.critical_path_latency.observe(
                duration_seconds,
                function_name=function_name
            )

    def on_violation(
        self,
        spec_id: str,
        probe_type: str,
        severity: str = "medium"
    ) -> None:
        """Called when a violation is detected."""
        self.metrics.record_violation(spec_id, probe_type, severity)
        self._violation_counts[spec_id] += 1
        
        # Update rate metric
        self._update_rates()

    def on_error(self, error: Exception) -> None:
        """Called when an error occurs during verification."""
        self.metrics.record_error(type(error).__name__)

    def _update_rates(self) -> None:
        """Update rate metrics."""
        now = time.time()
        elapsed_minutes = (now - self._last_rate_update) / 60.0
        
        if elapsed_minutes >= 1.0:
            for spec_id, count in self._violation_counts.items():
                rate = count / elapsed_minutes
                self.metrics.violation_rate.set(rate, spec_id=spec_id)
            
            self._violation_counts.clear()
            self._last_rate_update = now


# Example usage and testing
if __name__ == "__main__":
    # Create metrics
    metrics = RuntimeProbeMetrics()
    
    # Simulate some activity
    print("Simulating runtime probe activity...")
    
    for i in range(100):
        # Record checks
        passed = i % 10 != 0  # 10% failure rate
        duration = 0.001 + (i % 5) * 0.001
        
        metrics.record_check(
            spec_id="auth_validation",
            probe_type="precondition",
            passed=passed,
            duration_seconds=duration
        )
        
        if not passed:
            metrics.record_violation(
                spec_id="auth_validation",
                probe_type="precondition",
                severity="high"
            )
    
    # Record some additional metrics
    metrics.set_active_specs("precondition", 5)
    metrics.set_active_specs("postcondition", 3)
    metrics.set_pending_violations("high", 2)
    metrics.set_pending_violations("medium", 8)
    
    # Export metrics
    print("\n=== Prometheus Format ===")
    print(format_prometheus(metrics.collect_all())[:2000])  # First 2000 chars
    
    print("\n=== JSON Format ===")
    print(format_json(metrics.collect_all())[:2000])  # First 2000 chars
    
    # Test decorator
    @observe_spec_check(metrics, "example_spec", "assertion")
    def example_check(value: int) -> bool:
        return value > 0
    
    print("\n=== Testing Decorator ===")
    print(f"Check positive: {example_check(5)}")
    print(f"Check zero: {example_check(0)}")
    print(f"Check negative: {example_check(-1)}")
    
    print("\n=== Starting Metrics Server ===")
    server = MetricsServer(metrics, port=9090)
    server.start()
    
    print("Visit http://localhost:9090/metrics for Prometheus format")
    print("Visit http://localhost:9090/metrics/json for JSON format")
    print("Visit http://localhost:9090/health for health check")
    print("\nPress Ctrl+C to stop...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.stop()
