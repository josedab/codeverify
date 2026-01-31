"""Prometheus metrics configuration for CodeVerify API."""
from __future__ import annotations

from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_client import Counter, Histogram, Gauge
from fastapi import FastAPI


# Custom metrics
ANALYSES_TOTAL = Counter(
    "codeverify_analyses_total",
    "Total number of analyses",
    ["status", "conclusion"]
)

ANALYSES_DURATION = Histogram(
    "codeverify_analysis_duration_seconds",
    "Analysis duration in seconds",
    ["repository", "stage"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600)
)

FINDINGS_TOTAL = Counter(
    "codeverify_findings_total",
    "Total number of findings",
    ["severity", "category", "verification_type"]
)

ACTIVE_ANALYSES = Gauge(
    "codeverify_active_analyses",
    "Number of currently running analyses"
)

GITHUB_API_REQUESTS = Counter(
    "codeverify_github_api_requests_total",
    "Total GitHub API requests",
    ["endpoint", "status"]
)

LLM_API_REQUESTS = Counter(
    "codeverify_llm_api_requests_total",
    "Total LLM API requests",
    ["provider", "model", "status"]
)

LLM_API_LATENCY = Histogram(
    "codeverify_llm_api_latency_seconds",
    "LLM API request latency",
    ["provider", "model"],
    buckets=(0.5, 1, 2, 5, 10, 30, 60)
)

Z3_VERIFICATION_DURATION = Histogram(
    "codeverify_z3_verification_seconds",
    "Z3 verification duration",
    ["check_type", "result"],
    buckets=(0.1, 0.5, 1, 5, 10, 30)
)

WEBHOOK_EVENTS = Counter(
    "codeverify_webhook_events_total",
    "Total webhook events received",
    ["event_type", "action"]
)


def setup_metrics(app: FastAPI) -> Instrumentator:
    """Configure Prometheus metrics for the application."""
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/health", "/metrics"],
        inprogress_name="codeverify_http_requests_inprogress",
        inprogress_labels=True,
    )
    
    # Add default metrics
    instrumentator.add(
        metrics.default(
            metric_namespace="codeverify",
            metric_subsystem="http",
        )
    )
    
    # Add latency histogram
    instrumentator.add(
        metrics.latency(
            metric_namespace="codeverify",
            metric_subsystem="http",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
        )
    )
    
    # Add request size
    instrumentator.add(
        metrics.request_size(
            metric_namespace="codeverify",
            metric_subsystem="http",
        )
    )
    
    # Add response size
    instrumentator.add(
        metrics.response_size(
            metric_namespace="codeverify",
            metric_subsystem="http",
        )
    )
    
    # Instrument the app
    instrumentator.instrument(app)
    
    # Expose metrics endpoint
    instrumentator.expose(app, endpoint="/metrics", include_in_schema=False)
    
    return instrumentator


# Helper functions for recording custom metrics
def record_analysis_started():
    """Record that an analysis has started."""
    ACTIVE_ANALYSES.inc()


def record_analysis_completed(status: str, conclusion: str, duration_seconds: float):
    """Record that an analysis has completed."""
    ACTIVE_ANALYSES.dec()
    ANALYSES_TOTAL.labels(status=status, conclusion=conclusion).inc()


def record_analysis_stage(repository: str, stage: str, duration_seconds: float):
    """Record analysis stage duration."""
    ANALYSES_DURATION.labels(repository=repository, stage=stage).observe(duration_seconds)


def record_finding(severity: str, category: str, verification_type: str):
    """Record a finding."""
    FINDINGS_TOTAL.labels(
        severity=severity,
        category=category,
        verification_type=verification_type
    ).inc()


def record_github_request(endpoint: str, status: str):
    """Record a GitHub API request."""
    GITHUB_API_REQUESTS.labels(endpoint=endpoint, status=status).inc()


def record_llm_request(provider: str, model: str, status: str, latency_seconds: float):
    """Record an LLM API request."""
    LLM_API_REQUESTS.labels(provider=provider, model=model, status=status).inc()
    LLM_API_LATENCY.labels(provider=provider, model=model).observe(latency_seconds)


def record_z3_verification(check_type: str, result: str, duration_seconds: float):
    """Record a Z3 verification."""
    Z3_VERIFICATION_DURATION.labels(check_type=check_type, result=result).observe(duration_seconds)


def record_webhook_event(event_type: str, action: str):
    """Record a webhook event."""
    WEBHOOK_EVENTS.labels(event_type=event_type, action=action).inc()
