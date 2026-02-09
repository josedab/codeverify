# Runbook: High Error Rate

**Alert:** `HighErrorRate`
**Severity:** Critical
**Condition:** API 5xx error rate > 5% for 5 minutes

## Symptoms

- Users see 500 errors on the dashboard or API calls
- GitHub PR checks fail with internal errors
- `codeverify_http_requests_total{status=~"5.."}` is elevated

## Investigation Steps

1. **Check API logs for the error pattern:**
   ```bash
   kubectl logs -l app=codeverify-api --tail=200 -n codeverify | \
     jq 'select(.level == "ERROR")'
   ```

2. **Identify the failing endpoint:**
   ```promql
   sum by (endpoint, status) (
     rate(codeverify_http_requests_total{status=~"5.."}[5m])
   )
   ```

3. **Check dependent services:**
   ```bash
   kubectl exec -it deploy/codeverify-api -n codeverify -- \
     python -c "import asyncpg, asyncio; asyncio.run(asyncpg.connect('$DATABASE_URL'))"
   ```

4. **Check pod resource usage:**
   ```bash
   kubectl top pods -l app=codeverify-api -n codeverify
   ```

## Common Causes

| Cause | Fix |
|-------|-----|
| Database connection pool exhausted | Restart API pods or increase `DB_POOL_SIZE` |
| Redis unreachable | Check Redis pod health, verify `REDIS_URL` |
| OOM kills | Increase memory limits in deployment |
| Bad deployment | Roll back: `kubectl rollout undo deploy/codeverify-api` |

## Resolution

1. If a single endpoint is failing, check that endpoint's handler for recent changes
2. If all endpoints fail, check database and Redis connectivity
3. If resource-related, scale horizontally: `kubectl scale deploy/codeverify-api --replicas=4`
4. If caused by a bad deploy, roll back and investigate
