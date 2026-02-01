# Load Testing for CodeVerify

This directory contains load testing configuration using [Locust](https://locust.io/).

## Installation

```bash
pip install locust
```

## Running Load Tests

### Local Development

```bash
# Start the API server first
cd apps/api && uvicorn codeverify_api.main:app --port 8000

# Run Locust with web UI
locust -f tests/load/locustfile.py --host=http://localhost:8000

# Then open http://localhost:8089
```

### Headless Mode

```bash
# Run with specific user count and spawn rate
locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users=100 \
    --spawn-rate=10 \
    --run-time=5m \
    --headless
```

### CI/CD Integration

```bash
# Run quick smoke test
locust -f tests/load/locustfile.py \
    --host=http://localhost:8000 \
    --users=10 \
    --spawn-rate=5 \
    --run-time=1m \
    --headless \
    --only-summary
```

## Test Scenarios

### CodeVerifyUser (Default)
- Simulates typical API usage
- Reads repositories, analyses, stats
- Creates analyses occasionally

### WebhookSimulator
- Simulates GitHub webhook traffic
- Tests webhook endpoint under load

### HighVolumeUser
- Stress testing with rapid requests
- Tests rate limiting and performance

## Performance Targets

| Metric | Target |
|--------|--------|
| P50 Latency | < 100ms |
| P95 Latency | < 500ms |
| P99 Latency | < 1s |
| Error Rate | < 0.1% |
| Throughput | > 500 req/s |

## Interpreting Results

1. **Response Times**: Watch P95/P99 latencies
2. **Failure Rate**: Should stay below 0.1%
3. **Requests/sec**: Monitor throughput stability
4. **CPU/Memory**: Monitor server resources

## Custom Metrics

The load tests record custom metrics that can be exported to Prometheus/Grafana for analysis.
