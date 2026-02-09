# Runbook: LLM API Failures

**Alert:** `HighAnalysisFailureRate` (when caused by LLM errors)
**Severity:** Warning
**Condition:** LLM API error rate elevated or analysis failures > 10%

## Symptoms

- Analyses complete but with missing AI findings (formal verification still works)
- `codeverify_llm_requests_total{status="error"}` is elevated
- Worker logs show OpenAI/Anthropic API errors (429, 500, 503)
- Analysis results contain only Z3 verification findings, no semantic/security analysis

## Investigation Steps

1. **Check LLM error breakdown:**
   ```promql
   sum by (provider, error_type) (
     rate(codeverify_llm_requests_total{status="error"}[1h])
   )
   ```

2. **Check worker logs for API errors:**
   ```bash
   kubectl logs -l app=codeverify-worker --tail=200 -n codeverify | \
     jq 'select(.logger == "codeverify.agents" and .level == "ERROR")'
   ```

3. **Check rate limit status:**
   ```promql
   sum by (provider) (
     rate(codeverify_llm_requests_total{error_type="rate_limited"}[1h])
   )
   ```

4. **Verify API keys are valid:**
   ```bash
   # OpenAI
   curl -s -o /dev/null -w "%{http_code}" \
     -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models

   # Anthropic
   curl -s -o /dev/null -w "%{http_code}" \
     -H "x-api-key: $ANTHROPIC_API_KEY" \
     -H "anthropic-version: 2023-06-01" \
     https://api.anthropic.com/v1/messages
   ```

## Common Causes

| Cause | Fix |
|-------|-----|
| Rate limiting (429) | Reduce `WORKER_CONCURRENCY`, add backoff |
| API key expired/invalid | Rotate keys, check billing status |
| Provider outage (500/503) | Wait for recovery, switch to backup provider |
| Token quota exceeded | Check billing dashboard, upgrade plan |
| Network issues | Check DNS resolution, egress rules |

## Resolution

1. **Rate limiting:** Reduce concurrent workers:
   ```bash
   kubectl set env deploy/codeverify-worker WORKER_CONCURRENCY=2 -n codeverify
   ```

2. **Provider outage:** If both providers are configured, the system should fall back automatically. If only one is configured, set the other:
   ```bash
   kubectl set env deploy/codeverify-worker ANTHROPIC_API_KEY=sk-... -n codeverify
   ```

3. **Expired keys:** Rotate the API key in the Kubernetes secret:
   ```bash
   kubectl create secret generic codeverify-llm-keys \
     --from-literal=OPENAI_API_KEY=sk-new-key \
     --dry-run=client -o yaml | kubectl apply -f -
   kubectl rollout restart deploy/codeverify-worker -n codeverify
   ```

4. **If persistent**, enable verification-only mode by setting `ENABLE_AI_ANALYSIS=false` to maintain service while investigating
