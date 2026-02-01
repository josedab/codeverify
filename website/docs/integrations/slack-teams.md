---
sidebar_position: 6
---

# Slack & Teams Notifications

Get CodeVerify alerts in your team chat.

## Slack Integration

### Incoming Webhooks (Simple)

1. Go to [api.slack.com/apps](https://api.slack.com/apps)
2. Create a new app or select existing
3. Enable **Incoming Webhooks**
4. Add webhook to your channel
5. Copy the webhook URL

Configure CodeVerify:

```yaml
# .codeverify.yml
notifications:
  slack:
    webhook_url: ${SLACK_WEBHOOK_URL}
    channel: "#code-reviews"
    
    # When to notify
    on_failure: true
    on_success: false
    on_pr: true
    
    # Minimum severity to notify
    min_severity: high
```

Set the webhook URL in CI:

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/T.../B.../..."
```

### Message Format

CodeVerify sends rich Slack messages:

```
┌─────────────────────────────────────────────────────┐
│ 🔴 CodeVerify: 3 issues found                       │
├─────────────────────────────────────────────────────┤
│ Repository: acme/payments                           │
│ Branch: feature/checkout                            │
│ PR: #234 - Add checkout flow                        │
│ Trust Score: 62/100                                 │
├─────────────────────────────────────────────────────┤
│ ❌ Critical: SQL injection risk (billing.py:45)    │
│ ⚠️ High: Null dereference (checkout.py:78)         │
│ ⚠️ High: Array bounds (cart.py:23)                 │
├─────────────────────────────────────────────────────┤
│ [View Details] [View PR] [Dismiss]                  │
└─────────────────────────────────────────────────────┘
```

### Slack App (Advanced)

For richer integration, create a Slack App:

1. Create app at [api.slack.com/apps](https://api.slack.com/apps)
2. Add **Bot Token Scopes**:
   - `chat:write`
   - `chat:write.customize`
   - `reactions:write`
3. Install to workspace
4. Copy **Bot User OAuth Token**

Configure:

```yaml
notifications:
  slack:
    bot_token: ${SLACK_BOT_TOKEN}
    channel: "#code-reviews"
    
    # Thread replies for updates
    threading: true
    
    # React to messages on status change
    reactions: true
    
    # Mention users for critical issues
    mention_on_critical: true
    mention_users:
      - "@security-team"
```

### Channel Routing

Route notifications based on repository or severity:

```yaml
notifications:
  slack:
    routes:
      - match:
          repository: "*/payments*"
        channel: "#payments-alerts"
        
      - match:
          severity: critical
        channel: "#security-urgent"
        mention: "@security-team"
        
      - match:
          default: true
        channel: "#engineering"
```

## Microsoft Teams

### Incoming Webhook

1. In Teams, go to the channel
2. Click **...** → **Connectors**
3. Find **Incoming Webhook** → **Configure**
4. Name it "CodeVerify" and copy the URL

Configure:

```yaml
notifications:
  teams:
    webhook_url: ${TEAMS_WEBHOOK_URL}
    
    on_failure: true
    on_success: false
    min_severity: high
```

### Message Format

Teams receives Adaptive Cards:

```json
{
  "type": "AdaptiveCard",
  "body": [
    {
      "type": "TextBlock",
      "text": "🔴 CodeVerify: 3 issues found",
      "weight": "bolder",
      "size": "large"
    },
    {
      "type": "FactSet",
      "facts": [
        {"title": "Repository", "value": "acme/payments"},
        {"title": "Branch", "value": "feature/checkout"},
        {"title": "Trust Score", "value": "62/100"}
      ]
    }
  ],
  "actions": [
    {"type": "Action.OpenUrl", "title": "View Details", "url": "..."},
    {"type": "Action.OpenUrl", "title": "View PR", "url": "..."}
  ]
}
```

### Teams App (Power Automate)

For more control, use Power Automate:

1. Create a new Flow
2. Trigger: **When a HTTP request is received**
3. Action: **Post message in chat or channel**

Configure CodeVerify to use the HTTP trigger URL:

```yaml
notifications:
  webhook:
    url: ${POWER_AUTOMATE_URL}
    format: json
```

## Notification Settings

### Filtering

```yaml
notifications:
  filters:
    # Only notify for these severities
    severities:
      - critical
      - high
      
    # Only for certain categories
    categories:
      - security
      - null_safety
      
    # Only for these repositories
    repositories:
      - "acme/payments"
      - "acme/auth"
      
    # Ignore specific rules
    ignore_rules:
      - style-warning
```

### Rate Limiting

Prevent notification spam:

```yaml
notifications:
  rate_limit:
    # Maximum notifications per hour
    max_per_hour: 10
    
    # Aggregate multiple findings into one message
    aggregate: true
    aggregate_window: 300  # 5 minutes
```

### Scheduling

Quiet hours:

```yaml
notifications:
  schedule:
    # Timezone
    timezone: "America/New_York"
    
    # Quiet hours (no notifications)
    quiet_hours:
      - start: "22:00"
        end: "08:00"
      - start: "00:00"
        end: "23:59"
        days: [saturday, sunday]
    
    # During quiet hours, queue for summary
    queue_during_quiet: true
```

### Daily Summary

Receive a daily digest:

```yaml
notifications:
  daily_summary:
    enabled: true
    time: "09:00"
    timezone: "America/New_York"
    channel: "#code-quality"
```

## Email Notifications

```yaml
notifications:
  email:
    smtp_host: "smtp.company.com"
    smtp_port: 587
    smtp_user: ${SMTP_USER}
    smtp_password: ${SMTP_PASSWORD}
    from: "codeverify@company.com"
    
    recipients:
      - "security@company.com"
      - "tech-leads@company.com"
    
    # Per-repository recipients
    routes:
      - match:
          repository: "*/payments*"
        recipients:
          - "payments-team@company.com"
```

## Discord

```yaml
notifications:
  discord:
    webhook_url: ${DISCORD_WEBHOOK_URL}
    username: "CodeVerify"
    avatar_url: "https://codeverify.dev/logo.png"
```

## Generic Webhook

For custom integrations:

```yaml
notifications:
  webhook:
    url: "https://api.company.com/codeverify-events"
    method: POST
    headers:
      Authorization: "Bearer ${WEBHOOK_TOKEN}"
      Content-Type: "application/json"
    format: json
    
    # Include full findings data
    include_findings: true
    
    # Retry on failure
    retries: 3
    retry_delay: 30
```

Payload structure:

```json
{
  "event": "analysis_complete",
  "timestamp": "2024-01-15T10:30:00Z",
  "repository": "acme/payments",
  "branch": "feature/checkout",
  "commit": "abc123",
  "pr_number": 234,
  "status": "failed",
  "trust_score": 62,
  "findings": {
    "critical": 1,
    "high": 2,
    "medium": 0,
    "low": 0,
    "total": 3
  },
  "findings_details": [...]
}
```

## Troubleshooting

### No Notifications

1. Verify webhook URL is correct
2. Check environment variable is set
3. Review CI logs for errors
4. Test webhook manually with curl

### Rate Limited

Slack/Teams may rate limit:
- Enable `aggregate` mode
- Increase `aggregate_window`
- Reduce notification frequency

### Formatting Issues

If messages appear malformed:
- Check webhook type matches configuration
- Verify JSON is valid
- Test with minimal configuration first
