# Runbook: Queue Backup

**Alert:** `QueueBacklog`
**Severity:** Warning
**Condition:** Analysis queue depth > 100 jobs for 10 minutes

## Symptoms

- PR checks remain in "pending" state for a long time
- `codeverify_queue_length` is growing steadily
- Users report slow analysis turnaround

## Investigation Steps

1. **Check current queue depth:**
   ```bash
   kubectl exec -it deploy/codeverify-api -n codeverify -- \
     python -c "import redis; r = redis.from_url('$REDIS_URL'); print(r.llen('codeverify:queue'))"
   ```

2. **Check worker count and status:**
   ```bash
   kubectl get pods -l app=codeverify-worker -n codeverify
   kubectl logs -l app=codeverify-worker --tail=50 -n codeverify
   ```

3. **Check worker processing rate:**
   ```promql
   sum(rate(codeverify_analyses_total[5m]))
   ```

4. **Check for stuck jobs:**
   ```promql
   codeverify_analyses_in_progress
   ```

## Common Causes

| Cause | Fix |
|-------|-----|
| Insufficient workers | Scale up: `kubectl scale deploy/codeverify-worker --replicas=6` |
| Workers crashing (OOM) | Increase memory limits, reduce `WORKER_CONCURRENCY` |
| LLM API rate limiting | Check LLM provider dashboard, reduce concurrency |
| Stuck analysis (infinite Z3 loop) | Reduce `verification.timeout` in config, restart workers |
| Spike in PR activity | Temporary — scale up workers, will self-resolve |

## Resolution

1. **Immediate:** Scale workers to drain the queue:
   ```bash
   kubectl scale deploy/codeverify-worker --replicas=8 -n codeverify
   ```

2. **If workers are crashing**, check logs and fix the root cause before scaling

3. **After queue drains**, scale workers back to normal:
   ```bash
   kubectl scale deploy/codeverify-worker --replicas=3 -n codeverify
   ```

4. **If recurring**, consider setting up HPA (Horizontal Pod Autoscaler) based on queue depth
