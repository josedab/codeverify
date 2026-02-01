---
sidebar_position: 5
---

# Team Learning

Track patterns across your organization to improve code quality over time.

## Overview

Team Learning aggregates insights from CodeVerify analyses across your organization:

- **Common bugs** — What issues appear most frequently?
- **Hotspots** — Which areas of code need attention?
- **Trends** — Is code quality improving or declining?
- **Benchmarks** — How do teams compare?

## Dashboard

Access the Team Learning dashboard at `https://app.codeverify.dev/team`.

### Overview Panel

```
┌─────────────────────────────────────────────────────────────┐
│  Organization: Acme Corp                                    │
│  Period: Last 30 days                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Analyses: 1,234        Findings: 892       Resolved: 756   │
│                                                             │
│  Trend: ↓ 15% fewer findings than last month               │
│                                                             │
│  Top Issues:                                                │
│    1. Null safety (234)                                     │
│    2. Missing error handling (189)                          │
│    3. Array bounds (156)                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Metrics

### Code Health Score

Organization-wide health metric:

```
Code Health Score: 78/100

Breakdown:
├─ Finding Density: 82/100
│  └─ 2.3 findings per 1000 lines (good: < 5)
├─ Resolution Rate: 85/100
│  └─ 85% of findings resolved within 7 days
├─ Recurrence: 70/100
│  └─ 12% of fixed bugs reintroduced
└─ Coverage: 75/100
   └─ 75% of repos have CodeVerify enabled
```

### Trend Charts

Track metrics over time:

```
Findings per Month
│
│        ╭─────╮
│   ╭────╯     ╰───╮
│───╯              ╰───╮
│                      ╰────
└────────────────────────────
  Jan  Feb  Mar  Apr  May  Jun
```

### Repository Comparison

```
Repository Health Comparison

Repository         Score   Trend    Top Issue
─────────────────────────────────────────────────
payments-api       92      ↑ +5     Type safety
user-service       85      → 0      Null safety
web-frontend       78      ↓ -3     Error handling
legacy-monolith    45      ↑ +12    Array bounds
mobile-app         88      ↑ +2     Division by zero
```

## Common Patterns

### Pattern Detection

CodeVerify identifies recurring issues:

```
Top Pattern: Unchecked Optional Return

Description: Functions returning Optional[T] are used
without null checks.

Occurrences: 89 across 12 repositories
Example:
    user = get_user(id)
    print(user.name)  # user could be None

Recommendation:
    Add null check or use optional chaining
```

### Pattern Library

Build a library of organization-specific patterns:

```yaml
# .codeverify/patterns/org-patterns.yml
patterns:
  - id: unchecked-api-response
    name: "Unchecked API Response"
    description: "API responses used without status check"
    occurrences: 156
    repositories: 8
    severity: high
    
  - id: missing-transaction
    name: "Missing Database Transaction"
    description: "Multiple DB operations without transaction"
    occurrences: 42
    repositories: 3
    severity: critical
```

## Hotspot Analysis

### File Hotspots

Files with the most issues:

```
File Hotspots (Last 90 days)

File                          Findings   Churn   Risk
─────────────────────────────────────────────────────
src/billing/calculator.py         45      High    🔴
src/api/handlers.py               38      Medium  🟡
lib/utils/validation.ts           32      Low     🟡
services/auth/oauth.py            28      High    🔴
```

### Function Hotspots

```
Function Hotspots

Function                    Findings   Last Finding
────────────────────────────────────────────────────
process_payment()               12     2 days ago
validate_user_input()            8     1 week ago
calculate_shipping()             7     3 days ago
```

## Team Insights

### Per-Team Metrics

```
Team Performance

Team           Repos   Score   Findings/Week   Resolution
──────────────────────────────────────────────────────────
Platform        8       85          12            4.2 days
Payments        4       92           5            1.8 days
Frontend        6       78          18            6.1 days
Mobile          3       88           8            3.2 days
```

### Individual Contributor Stats

```
Top Contributors (Findings Resolved)

Developer           Resolved   Introduced   Net
─────────────────────────────────────────────────
alice@company.com        45           12    +33
bob@company.com          38           15    +23
carol@company.com        32            8    +24
```

## Alerts and Notifications

### Threshold Alerts

```yaml
# Organization settings
alerts:
  # Alert when repository score drops
  score_drop:
    threshold: 10  # points
    period: 7      # days
    notify: ["#engineering", "tech-leads@company.com"]
    
  # Alert on new critical findings
  critical_findings:
    enabled: true
    notify: ["#security-alerts"]
    
  # Weekly digest
  weekly_digest:
    enabled: true
    day: monday
    notify: ["tech-leads@company.com"]
```

### Anomaly Detection

CodeVerify alerts on unusual patterns:

```
⚠️ Anomaly Detected

Repository: payments-api
Pattern: Sudden increase in null safety findings

Details:
- Last week: 2 findings
- This week: 15 findings (+650%)
- Affected files: src/handlers/*

Possible causes:
- New developer onboarded
- Dependency update
- Refactoring in progress

Recommended action: Review recent commits
```

## Integration

### Export Data

```bash
# Export team metrics as JSON
codeverify team export --format json --output team-metrics.json

# Export for specific period
codeverify team export --from 2024-01-01 --to 2024-06-30
```

### API Access

```bash
curl -H "Authorization: Bearer $TOKEN" \
  "https://api.codeverify.dev/v1/org/acme/metrics?period=30d"
```

### Webhooks

```yaml
webhooks:
  - url: "https://metrics.company.com/codeverify"
    events:
      - analysis_complete
      - score_change
      - alert_triggered
```

## Privacy and Access

### Data Retention

```yaml
team_learning:
  retention:
    metrics: 365      # days
    findings: 90      # days
    raw_code: 0       # don't store code
```

### Access Control

```yaml
team_learning:
  access:
    org_admins: full
    team_leads: team_only
    developers: self_only
```

## Configuration

### Enable Team Learning

```yaml
# .codeverify.yml (repository)
team_learning:
  enabled: true
  
  # Participate in org-wide metrics
  org_metrics: true
  
  # Share patterns with org
  share_patterns: true
```

### Organization Settings

```yaml
# Organization-level
team_learning:
  # Aggregate metrics across repos
  aggregate: true
  
  # Identify common patterns
  pattern_detection: true
  
  # Track resolution times
  resolution_tracking: true
  
  # Benchmark teams
  team_benchmarks: true
```

## Best Practices

1. **Review weekly** — Check dashboard for trends
2. **Address hotspots** — Focus on high-risk files
3. **Share learnings** — Distribute pattern insights
4. **Set goals** — Target score improvements
5. **Celebrate wins** — Recognize improving teams

## Next Steps

- [CI/CD Integration](/docs/integrations/ci-cd) — Enable analysis in pipelines
- [Notifications](/docs/integrations/slack-teams) — Set up alerts
- [API Reference](/docs/api/overview) — Access metrics programmatically
