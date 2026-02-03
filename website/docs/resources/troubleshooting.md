---
sidebar_position: 1
---

# Troubleshooting

Common issues and how to resolve them.

## Installation Issues

### pip install fails

**Error:** `ERROR: Could not find a version that satisfies the requirement codeverify`

**Solution:**
```bash
# Ensure Python 3.9+
python --version

# Upgrade pip
pip install --upgrade pip

# Try again
pip install codeverify
```

### Z3 installation fails

**Error:** `Failed building wheel for z3-solver`

**Solution:**

On macOS:
```bash
brew install z3
pip install codeverify
```

On Ubuntu:
```bash
sudo apt-get install libz3-dev
pip install codeverify
```

On Windows:
```bash
# Use pre-built wheel
pip install z3-solver --only-binary :all:
pip install codeverify
```

### Permission denied

**Error:** `PermissionError: [Errno 13] Permission denied`

**Solution:**
```bash
# Use user install
pip install --user codeverify

# Or use virtual environment
python -m venv venv
source venv/bin/activate
pip install codeverify
```

## Analysis Issues

### Analysis times out

**Error:** `Analysis timed out after 300 seconds`

**Solutions:**

1. Increase timeout:
```yaml
# .codeverify.yml
verification:
  timeout: 600
```

2. Reduce scope:
```yaml
include:
  - "src/**"  # Only analyze src/

exclude:
  - "**/*.test.*"  # Skip tests
```

3. Disable expensive checks:
```yaml
verification:
  checks:
    - null_safety
    - division_by_zero
    # Disable: array_bounds, integer_overflow
```

### Out of memory

**Error:** `MemoryError` or `Killed`

**Solutions:**

1. Limit file size:
```yaml
verification:
  advanced:
    max_file_size: 100000  # 100KB
```

2. Reduce worker memory:
```yaml
verification:
  advanced:
    memory_limit: 2048  # 2GB
```

3. Analyze in batches:
```bash
codeverify analyze src/module1
codeverify analyze src/module2
```

### "Unknown" verification result

**Error:** `Verification result: unknown (solver timeout)`

Z3 couldn't determine satisfiability within the timeout.

**Solutions:**

1. Increase Z3 timeout:
```yaml
verification:
  advanced:
    z3_timeout: 60
```

2. Simplify the code:
   - Break large functions into smaller ones
   - Add type annotations
   - Reduce nesting depth

3. Add assertions to help Z3:
```python
assert 0 <= index < len(array)
```

## Configuration Issues

### Config file not found

**Error:** `Configuration file not found: .codeverify.yml`

**Solutions:**

1. Create the file:
```bash
codeverify init
```

2. Specify path:
```bash
codeverify analyze --config path/to/config.yml
```

3. Use environment variable:
```bash
export CODEVERIFY_CONFIG=/path/to/config.yml
```

### Invalid YAML

**Error:** `yaml.scanner.ScannerError: mapping values are not allowed here`

**Solution:** Check YAML syntax:
```yaml
# Wrong
verification:
checks:
  - null_safety

# Correct
verification:
  checks:
    - null_safety
```

Use a YAML validator or:
```bash
codeverify config validate
```

### Unknown configuration key

**Error:** `Unknown configuration key: 'verfication'`

Usually a typo. Check spelling:
```yaml
# Wrong
verfication:
  enabled: true

# Correct
verification:
  enabled: true
```

## Integration Issues

### GitHub App not working

**Symptoms:** No status checks, no PR comments

**Solutions:**

1. Check installation:
   - Go to GitHub App settings
   - Verify repository is selected

2. Check permissions:
   - Checks: Read and write
   - Pull requests: Read and write
   - Contents: Read

3. Check webhook delivery:
   - Go to GitHub App > Advanced > Recent Deliveries
   - Look for failed deliveries

4. Verify configuration:
```yaml
github:
  app_id: 12345
  checks: true
  comment: true
```

### GitLab CI not running

**Symptoms:** Job skipped or not appearing

**Solutions:**

1. Check rules:
```yaml
codeverify:
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
```

2. Check image:
```yaml
codeverify:
  image: codeverify/cli:latest
```

3. Check variables:
   - Ensure `CODEVERIFY_API_KEY` is set in CI/CD settings

### VS Code extension not working

**Symptoms:** No diagnostics, status bar shows error

**Solutions:**

1. Check output:
   - View > Output > CodeVerify
   - Look for error messages

2. Verify CLI is installed:
```bash
codeverify --version
```

3. Check settings:
```json
{
  "codeverify.enable": true,
  "codeverify.path": "/path/to/codeverify"
}
```

4. Restart extension:
   - Command Palette > CodeVerify: Restart

## API Issues

### 401 Unauthorized

**Error:** `{"error":{"code":"unauthorized","message":"Invalid API key"}}`

**Solutions:**

1. Check API key format:
   - Should start with `cv_live_` or `cv_test_`

2. Verify environment variable:
```bash
echo $CODEVERIFY_API_KEY
```

3. Check key is active:
```bash
codeverify auth list-keys
```

### 429 Rate Limited

**Error:** `{"error":{"code":"rate_limited","message":"Too many requests"}}`

**Solutions:**

1. Check rate limit headers:
```
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1705315800
```

2. Implement backoff:
```python
import time

def analyze_with_backoff(repo):
    for attempt in range(5):
        try:
            return client.analyses.create(repository=repo)
        except RateLimitError as e:
            wait = 2 ** attempt
            time.sleep(wait)
    raise Exception("Rate limit exceeded")
```

3. Upgrade plan for higher limits

### 500 Internal Server Error

**Error:** `{"error":{"code":"internal_error"}}`

**Solutions:**

1. Retry with exponential backoff
2. Check [status.codeverify.dev](https://status.codeverify.dev)
3. Contact support with request ID

## Self-Hosting Issues

### Database connection failed

**Error:** `Connection refused to PostgreSQL`

**Solutions:**

1. Check database is running:
```bash
docker compose ps postgres
```

2. Verify connection string:
```bash
psql $DATABASE_URL
```

3. Check network:
```bash
docker compose exec api ping postgres
```

### Workers not processing

**Symptoms:** Analyses stuck in "queued" state

**Solutions:**

1. Check worker status:
```bash
docker compose exec worker celery -A app.worker inspect active
```

2. Check Redis connection:
```bash
docker compose exec worker redis-cli -h redis ping
```

3. Restart workers:
```bash
docker compose restart worker
```

### High memory usage

**Solutions:**

1. Limit worker memory:
```yaml
services:
  worker:
    deploy:
      resources:
        limits:
          memory: 4G
```

2. Reduce concurrency:
```bash
CELERY_CONCURRENCY=2 docker compose up worker
```

3. Enable swap (emergency):
```bash
sudo fallocate -l 4G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Getting Help

### Debug Mode

Enable verbose logging:
```bash
codeverify analyze --verbose --debug
```

Or set environment variable:
```bash
export CODEVERIFY_DEBUG=1
```

### Log Collection

Collect logs for support:
```bash
codeverify diagnose > diagnostic-report.txt
```

### Support Channels

- **GitHub Issues:** [github.com/codeverify/codeverify/issues](https://github.com/codeverify/codeverify/issues)
- **Discord:** [discord.gg/codeverify](https://discord.gg/codeverify)
- **Email:** support@codeverify.dev (Pro/Enterprise)

## Monitoring {#monitoring}

### Prometheus Metrics

CodeVerify exposes metrics at `/metrics`:

```bash
# API metrics
curl http://localhost:8000/metrics

# Worker metrics
curl http://localhost:8001/metrics
```

Key metrics:

| Metric | Description |
|--------|-------------|
| `codeverify_analyses_total` | Total analyses run |
| `codeverify_analyses_duration_seconds` | Analysis duration histogram |
| `codeverify_findings_total` | Findings by severity |
| `codeverify_z3_solver_duration_seconds` | Z3 solver time |
| `codeverify_worker_queue_length` | Pending jobs in queue |

### Grafana Dashboard

Import our dashboard:

1. Go to Grafana → Dashboards → Import
2. Use dashboard ID: `18432` or import from JSON:

```bash
curl -O https://raw.githubusercontent.com/codeverify/codeverify/main/deploy/grafana/dashboard.json
```

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Worker health
curl http://localhost:8001/health

# Database connectivity
curl http://localhost:8000/health/db

# Redis connectivity
curl http://localhost:8000/health/redis
```

### Alerting

Example Prometheus alerts:

```yaml
groups:
  - name: codeverify
    rules:
      - alert: CodeVerifyAPIDown
        expr: up{job="codeverify-api"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "CodeVerify API is down"

      - alert: CodeVerifyHighErrorRate
        expr: rate(codeverify_analyses_errors_total[5m]) > 0.1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High error rate in CodeVerify analyses"

      - alert: CodeVerifyQueueBacklog
        expr: codeverify_worker_queue_length > 100
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "CodeVerify analysis queue is backing up"
```
