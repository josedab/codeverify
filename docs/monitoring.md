# Monitoring & Observability Guide

This guide covers setting up monitoring, logging, and alerting for CodeVerify.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Observability Stack                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │
│   │  Prometheus │    │    Loki     │    │   Tempo     │                    │
│   │  (Metrics)  │    │   (Logs)    │    │  (Traces)   │                    │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                    │
│          │                  │                  │                            │
│          └──────────────────┼──────────────────┘                            │
│                             │                                               │
│                             ▼                                               │
│                      ┌─────────────┐                                        │
│                      │   Grafana   │                                        │
│                      │ (Dashboards)│                                        │
│                      └──────┬──────┘                                        │
│                             │                                               │
│                             ▼                                               │
│                      ┌─────────────┐                                        │
│                      │  Alerting   │                                        │
│                      │ PagerDuty   │                                        │
│                      │   Slack     │                                        │
│                      └─────────────┘                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Using Docker Compose (Development)

```bash
# Start monitoring stack
docker compose -f docker-compose.monitoring.yml up -d

# Access Grafana
open http://localhost:3000
# Default credentials: admin / admin
```

### Using Helm (Production)

```bash
# Add Helm repos
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install Prometheus stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  -f deploy/monitoring/prometheus-values.yaml

# Install Loki
helm install loki grafana/loki-stack \
  --namespace monitoring \
  -f deploy/monitoring/loki-values.yaml
```

## Metrics

### Application Metrics

CodeVerify exposes Prometheus metrics at `/metrics`:

#### API Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `codeverify_http_requests_total` | Counter | Total HTTP requests |
| `codeverify_http_request_duration_seconds` | Histogram | Request latency |
| `codeverify_http_requests_in_progress` | Gauge | Active requests |
| `codeverify_db_queries_total` | Counter | Database queries |
| `codeverify_db_query_duration_seconds` | Histogram | Query latency |

#### Worker Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `codeverify_analyses_total` | Counter | Total analyses |
| `codeverify_analyses_in_progress` | Gauge | Active analyses |
| `codeverify_analysis_duration_seconds` | Histogram | Analysis duration |
| `codeverify_findings_total` | Counter | Findings by severity |
| `codeverify_verification_duration_seconds` | Histogram | Z3 verification time |
| `codeverify_llm_requests_total` | Counter | LLM API calls |
| `codeverify_llm_tokens_total` | Counter | LLM tokens used |

#### Queue Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `codeverify_queue_length` | Gauge | Jobs in queue |
| `codeverify_queue_latency_seconds` | Histogram | Time in queue |
| `codeverify_job_retries_total` | Counter | Job retries |

### Prometheus Configuration

```yaml
# prometheus-values.yaml
prometheus:
  prometheusSpec:
    serviceMonitorSelector:
      matchLabels:
        app: codeverify
    
    additionalScrapeConfigs:
      - job_name: 'codeverify-api'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            regex: codeverify-api
            action: keep
          - source_labels: [__meta_kubernetes_pod_container_port_number]
            regex: "8000"
            action: keep

      - job_name: 'codeverify-worker'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_label_app]
            regex: codeverify-worker
            action: keep
```

### ServiceMonitor

```yaml
# kubernetes/servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: codeverify
  labels:
    app: codeverify
spec:
  selector:
    matchLabels:
      app: codeverify
  endpoints:
    - port: http
      path: /metrics
      interval: 30s
```

## Logging

### Log Format

CodeVerify uses structured JSON logging:

```json
{
  "timestamp": "2026-01-31T00:00:00.000Z",
  "level": "INFO",
  "logger": "codeverify.api",
  "message": "Analysis completed",
  "analysis_id": "abc123",
  "duration_ms": 4523,
  "findings_count": 3,
  "trace_id": "xyz789"
}
```

### Loki Configuration

```yaml
# loki-values.yaml
loki:
  config:
    table_manager:
      retention_deletes_enabled: true
      retention_period: 720h  # 30 days

promtail:
  config:
    clients:
      - url: http://loki:3100/loki/api/v1/push
    
    scrape_configs:
      - job_name: kubernetes-pods
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_namespace]
            target_label: namespace
          - source_labels: [__meta_kubernetes_pod_name]
            target_label: pod
          - source_labels: [__meta_kubernetes_pod_label_app]
            target_label: app
```

### Log Queries (LogQL)

**Recent errors:**
```logql
{app="codeverify-api"} |= "ERROR" | json | line_format "{{.message}}"
```

**Analysis failures:**
```logql
{app="codeverify-worker"} | json | status="failed" | line_format "{{.analysis_id}}: {{.error_message}}"
```

**Slow requests:**
```logql
{app="codeverify-api"} | json | duration_ms > 1000
```

## Tracing

### OpenTelemetry Configuration

```python
# codeverify_api/tracing.py
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def setup_tracing():
    provider = TracerProvider()
    processor = BatchSpanProcessor(OTLPSpanExporter(
        endpoint="http://tempo:4317",
        insecure=True,
    ))
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
```

### Environment Variables

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://tempo:4317
OTEL_SERVICE_NAME=codeverify-api
OTEL_TRACES_SAMPLER=parentbased_traceidratio
OTEL_TRACES_SAMPLER_ARG=0.1  # Sample 10%
```

### Tempo Configuration

```yaml
# tempo-values.yaml
tempo:
  config:
    distributor:
      receivers:
        otlp:
          protocols:
            grpc:
              endpoint: 0.0.0.0:4317
    
    storage:
      trace:
        backend: s3
        s3:
          bucket: codeverify-traces
          endpoint: s3.amazonaws.com
```

## Dashboards

### Grafana Dashboards

Import the provided dashboards:

```bash
kubectl apply -f deploy/monitoring/dashboards/
```

**Available Dashboards:**

1. **CodeVerify Overview**
   - Analysis pass rate
   - Request throughput
   - Error rate
   - Queue depth

2. **API Performance**
   - Request latency (p50, p95, p99)
   - Requests per second
   - Error breakdown
   - Endpoint heatmap

3. **Worker Performance**
   - Analysis duration
   - Queue wait time
   - Verification success rate
   - LLM API latency

4. **Infrastructure**
   - CPU/Memory utilization
   - Pod restarts
   - Database connections
   - Redis operations

### Dashboard JSON

```json
{
  "dashboard": {
    "title": "CodeVerify Overview",
    "panels": [
      {
        "title": "Analysis Pass Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(codeverify_analyses_total{status=\"passed\"}[1h])) / sum(rate(codeverify_analyses_total[1h])) * 100"
          }
        ]
      },
      {
        "title": "Analyses per Hour",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(increase(codeverify_analyses_total[1h]))"
          }
        ]
      },
      {
        "title": "Findings by Severity",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (severity) (codeverify_findings_total)"
          }
        ]
      }
    ]
  }
}
```

## Alerting

### Prometheus Alert Rules

```yaml
# alerting-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: codeverify-alerts
spec:
  groups:
    - name: codeverify.rules
      rules:
        # High error rate
        - alert: HighErrorRate
          expr: |
            sum(rate(codeverify_http_requests_total{status=~"5.."}[5m])) 
            / sum(rate(codeverify_http_requests_total[5m])) > 0.05
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "High API error rate"
            description: "Error rate is {{ $value | humanizePercentage }}"

        # High latency
        - alert: HighLatency
          expr: |
            histogram_quantile(0.95, 
              sum(rate(codeverify_http_request_duration_seconds_bucket[5m])) by (le)
            ) > 2
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "High API latency"
            description: "P95 latency is {{ $value | humanizeDuration }}"

        # Queue backup
        - alert: QueueBacklog
          expr: codeverify_queue_length > 100
          for: 10m
          labels:
            severity: warning
          annotations:
            summary: "Analysis queue backing up"
            description: "{{ $value }} jobs in queue"

        # Worker down
        - alert: WorkerDown
          expr: |
            sum(up{app="codeverify-worker"}) < 2
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "Insufficient workers"
            description: "Only {{ $value }} workers running"

        # Analysis failures
        - alert: HighAnalysisFailureRate
          expr: |
            sum(rate(codeverify_analyses_total{status="failed"}[1h])) 
            / sum(rate(codeverify_analyses_total[1h])) > 0.1
          for: 15m
          labels:
            severity: warning
          annotations:
            summary: "High analysis failure rate"
            description: "{{ $value | humanizePercentage }} of analyses failing"

        # Database connection pool
        - alert: DatabaseConnectionPoolExhausted
          expr: codeverify_db_pool_available < 5
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "Database connection pool nearly exhausted"
            description: "Only {{ $value }} connections available"
```

### Alert Routing

```yaml
# alertmanager-config.yaml
alertmanager:
  config:
    route:
      receiver: 'default'
      group_by: ['alertname', 'severity']
      group_wait: 30s
      group_interval: 5m
      repeat_interval: 4h
      routes:
        - match:
            severity: critical
          receiver: 'pagerduty-critical'
        - match:
            severity: warning
          receiver: 'slack-warnings'

    receivers:
      - name: 'default'
        slack_configs:
          - api_url: 'https://hooks.slack.com/services/xxx'
            channel: '#codeverify-alerts'

      - name: 'pagerduty-critical'
        pagerduty_configs:
          - service_key: 'your-pagerduty-key'
            severity: critical

      - name: 'slack-warnings'
        slack_configs:
          - api_url: 'https://hooks.slack.com/services/xxx'
            channel: '#codeverify-alerts'
            title: '{{ .GroupLabels.alertname }}'
            text: '{{ .CommonAnnotations.description }}'
```

## Health Checks

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health` | Basic health check |
| `/health/ready` | Readiness (dependencies OK) |
| `/health/live` | Liveness (process running) |

### Kubernetes Probes

```yaml
# kubernetes/deployment.yaml
spec:
  containers:
    - name: api
      livenessProbe:
        httpGet:
          path: /health/live
          port: 8000
        initialDelaySeconds: 10
        periodSeconds: 10
        failureThreshold: 3
      
      readinessProbe:
        httpGet:
          path: /health/ready
          port: 8000
        initialDelaySeconds: 5
        periodSeconds: 5
        failureThreshold: 3
```

## SLIs/SLOs

### Service Level Indicators

| SLI | Calculation | Target |
|-----|-------------|--------|
| Availability | `1 - (error_requests / total_requests)` | 99.9% |
| Latency (p95) | `histogram_quantile(0.95, ...)` | < 500ms |
| Analysis Success | `passed_analyses / total_analyses` | > 95% |
| Queue Wait | `p95(time_in_queue)` | < 60s |

### SLO Dashboard

```promql
# Availability SLO (99.9%)
1 - (
  sum(rate(codeverify_http_requests_total{status=~"5.."}[30d]))
  / sum(rate(codeverify_http_requests_total[30d]))
)

# Error Budget Remaining
(0.001 - (
  sum(rate(codeverify_http_requests_total{status=~"5.."}[30d]))
  / sum(rate(codeverify_http_requests_total[30d]))
)) / 0.001 * 100
```

## Debugging

### Common Queries

**Find slow analyses:**
```promql
topk(10, 
  codeverify_analysis_duration_seconds{quantile="0.95"}
) by (repository)
```

**LLM API errors:**
```promql
sum by (provider, error_type) (
  rate(codeverify_llm_requests_total{status="error"}[1h])
)
```

**Memory usage by pod:**
```promql
container_memory_working_set_bytes{namespace="codeverify"}
/ container_spec_memory_limit_bytes{namespace="codeverify"} * 100
```

### Troubleshooting Runbooks

See individual runbooks in `deploy/monitoring/runbooks/`:

- `high-error-rate.md`
- `queue-backup.md`
- `database-connection-issues.md`
- `llm-api-failures.md`
