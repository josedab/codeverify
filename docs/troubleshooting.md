# Troubleshooting Guide

Common issues and solutions for CodeVerify.

## Installation Issues

### "Z3 solver not found"

The Z3 solver is required for formal verification.

**Solution:**
```bash
pip install z3-solver
```

If using conda:
```bash
conda install -c conda-forge z3
```

### "Node.js version mismatch"

CodeVerify requires Node.js 20+.

**Solution:**
```bash
# Using nvm
nvm install 20
nvm use 20

# Verify
node --version  # Should be v20.x.x
```

### "Database connection refused"

PostgreSQL isn't running or connection string is wrong.

**Solutions:**
1. Start PostgreSQL:
   ```bash
   docker compose up -d postgres
   ```

2. Verify connection string in `.env`:
   ```bash
   DATABASE_URL=postgresql://codeverify:password@localhost:5432/codeverify
   ```

3. Check PostgreSQL is accepting connections:
   ```bash
   psql $DATABASE_URL -c "SELECT 1"
   ```

### "Redis connection refused"

Redis isn't running.

**Solution:**
```bash
docker compose up -d redis

# Verify
redis-cli ping  # Should return PONG
```

## Analysis Issues

### "Analysis timed out"

The analysis is taking too long, usually due to complex code or slow API responses.

**Solutions:**

1. Increase timeout in `.codeverify.yml`:
   ```yaml
   verification:
     timeout: 60  # seconds
   ```

2. Exclude large/generated files:
   ```yaml
   exclude:
     - "**/*.min.js"
     - "**/generated/**"
   ```

3. For local analysis, use `--timeout`:
   ```bash
   codeverify analyze --timeout 120
   ```

### "Too many findings"

Getting overwhelmed with findings.

**Solutions:**

1. Increase severity threshold:
   ```yaml
   thresholds:
     medium: 10
     low: 50
   ```

2. Filter by severity:
   ```bash
   codeverify analyze --severity high
   ```

3. Exclude test files:
   ```yaml
   exclude:
     - "tests/**"
     - "**/*.test.*"
   ```

### "False positive findings"

Getting incorrect findings.

**Solutions:**

1. Suppress inline:
   ```python
   result = a / b  # codeverify: ignore division_by_zero
   ```

2. Add ignore rules:
   ```yaml
   ignore:
     - pattern: "tests/**"
       reason: "Test code"
   ```

3. Report false positive (click 👎 in PR comment or dashboard)

### "AI analysis not working"

AI agents aren't returning results.

**Solutions:**

1. Check API key:
   ```bash
   echo $OPENAI_API_KEY
   # or
   echo $ANTHROPIC_API_KEY
   ```

2. Verify API connectivity:
   ```bash
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY"
   ```

3. Check rate limits - you may have exceeded your quota

### "Verification returns 'unknown'"

Z3 couldn't prove or disprove the property.

**Causes:**
- Code is too complex
- Timeout reached
- Unbounded loops

**Solutions:**

1. Increase timeout:
   ```yaml
   verification:
     timeout: 60
   ```

2. Break up complex functions into smaller pieces

3. Add type hints to help constraint generation:
   ```python
   def process(items: list[int], index: int) -> int:
       ...
   ```

## GitHub Integration Issues

### "Webhooks not received"

GitHub isn't sending webhooks.

**Solutions:**

1. Check webhook configuration in GitHub App settings
2. Verify webhook URL is publicly accessible
3. Check webhook secret matches `.env`
4. View Recent Deliveries in GitHub App settings for errors

### "Check status not updating"

PR checks remain pending.

**Solutions:**

1. Check worker logs:
   ```bash
   docker compose logs worker -f
   ```

2. Verify Redis queue:
   ```bash
   redis-cli LLEN codeverify:queue
   ```

3. Check GitHub App permissions include "Checks: Read & Write"

### "Comments not appearing on PR"

Analysis completes but no PR comment.

**Solutions:**

1. Check GitHub App has "Pull requests: Read & Write" permission
2. Verify `comment_on_pass: true` in config if expecting comments on passing analyses
3. Check API logs for GitHub API errors

### "App installation fails"

Can't install GitHub App.

**Solutions:**

1. Verify GitHub App is public or you're in the allowed organization
2. Check app permissions aren't requesting more than allowed by org policy
3. Verify callback URL matches what's configured in GitHub App

## Dashboard Issues

### "Login fails"

Can't authenticate with GitHub.

**Solutions:**

1. Check GitHub OAuth Client ID and Secret
2. Verify callback URL: `http://localhost:3000/auth/callback`
3. Clear browser cookies and try again

### "No data showing"

Dashboard is empty.

**Solutions:**

1. Verify you've run at least one analysis
2. Check you're viewing the correct organization
3. Verify API connection:
   ```bash
   curl http://localhost:8000/health
   ```

### "Charts not loading"

Analytics charts are broken.

**Solutions:**

1. Check browser console for JavaScript errors
2. Verify API is returning data
3. Clear browser cache

## CLI Issues

### "Command not found: codeverify"

CLI not in PATH.

**Solutions:**

1. Install globally:
   ```bash
   pip install codeverify-cli
   ```

2. Or use full path:
   ```bash
   python -m codeverify analyze
   ```

### "Configuration file not found"

CLI can't find `.codeverify.yml`.

**Solutions:**

1. Create config:
   ```bash
   codeverify init
   ```

2. Specify path:
   ```bash
   codeverify analyze -c /path/to/.codeverify.yml
   ```

### "Invalid configuration"

Config file has errors.

**Solution:**
```bash
codeverify validate
# Shows specific errors
```

## Performance Issues

### "Analysis is slow"

Analysis takes too long.

**Solutions:**

1. Reduce scope:
   ```yaml
   include:
     - "src/**/*"
   exclude:
     - "node_modules/**"
   ```

2. Disable unnecessary checks:
   ```yaml
   verification:
     checks:
       - null_safety  # Keep only needed checks
   ```

3. Use budget optimizer:
   ```yaml
   budget:
     tier: standard
     max_cost_per_file: 0.50
   ```

### "High memory usage"

Worker using too much memory.

**Solutions:**

1. Reduce concurrency:
   ```bash
   celery -A codeverify_worker.main worker --concurrency=2
   ```

2. Limit file size:
   ```yaml
   max_file_size_kb: 200
   ```

### "API rate limited"

Hitting rate limits.

**Solutions:**

1. Check your plan's rate limits
2. Reduce polling frequency
3. Use webhooks instead of polling
4. Contact support for limit increase

## Getting Help

If you can't resolve your issue:

1. **Check logs:**
   ```bash
   docker compose logs -f
   ```

2. **Search existing issues:**
   https://github.com/codeverify/codeverify/issues

3. **Ask in Discord:**
   [Link to Discord community]

4. **Open an issue:**
   Include:
   - CodeVerify version (`codeverify --version`)
   - Error messages
   - Configuration (redact secrets)
   - Steps to reproduce

5. **Security issues:**
   Email security@codeverify.dev (do not open public issue)
