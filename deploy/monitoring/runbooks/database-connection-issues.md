# Runbook: Database Connection Issues

**Alert:** `DatabaseConnectionPoolExhausted`
**Severity:** Critical
**Condition:** Available database connections < 5 for 5 minutes

## Symptoms

- API returns 500 errors with "connection pool exhausted" in logs
- Slow API responses due to connection wait times
- `codeverify_db_pool_available` near zero

## Investigation Steps

1. **Check current connection count on PostgreSQL:**
   ```bash
   kubectl exec -it deploy/codeverify-postgres -n codeverify -- \
     psql -U codeverify -c "SELECT count(*) FROM pg_stat_activity WHERE datname = 'codeverify';"
   ```

2. **Check for long-running queries:**
   ```bash
   kubectl exec -it deploy/codeverify-postgres -n codeverify -- \
     psql -U codeverify -c "
       SELECT pid, now() - pg_stat_activity.query_start AS duration, query
       FROM pg_stat_activity
       WHERE state = 'active' AND datname = 'codeverify'
       ORDER BY duration DESC LIMIT 10;"
   ```

3. **Check API pod count (each holds a pool):**
   ```bash
   kubectl get pods -l app=codeverify-api -n codeverify
   ```

4. **Check pool metrics:**
   ```promql
   codeverify_db_pool_available
   codeverify_db_pool_in_use
   ```

## Common Causes

| Cause | Fix |
|-------|-----|
| Too many API replicas for max_connections | Reduce replicas or increase `max_connections` in PostgreSQL |
| Connection leak (unreturned connections) | Restart API pods, investigate code for missing `async with` |
| Long-running queries holding connections | Kill slow queries, add query timeouts |
| PostgreSQL max_connections too low | Increase in PostgreSQL config, restart |

## Resolution

1. **Immediate relief** — kill idle connections:
   ```sql
   SELECT pg_terminate_backend(pid)
   FROM pg_stat_activity
   WHERE datname = 'codeverify' AND state = 'idle' AND query_start < now() - interval '10 minutes';
   ```

2. **Increase pool size** if connections are legitimately needed:
   - Set `DB_POOL_SIZE` and `DB_MAX_OVERFLOW` environment variables
   - Ensure `DB_POOL_SIZE * num_replicas < max_connections`

3. **If recurring**, consider PgBouncer as a connection pooler in front of PostgreSQL
