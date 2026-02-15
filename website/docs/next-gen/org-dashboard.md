---
sidebar_position: 7
---

# Organization Dashboard

Aggregate metrics, team comparisons, risk heatmaps, and ROI tracking.

## Overview

The Organization Dashboard provides leadership and engineering managers with visibility into code quality trends across teams and repositories. It aggregates verification results into actionable metrics with export capabilities.

## Key Metrics

- **Team Comparison** — Side-by-side quality scores across teams
- **Risk Heatmap** — Visual map of high-risk files and directories
- **Trend Analysis** — Quality improvements over time
- **ROI Calculator** — Estimate bugs caught vs. cost of verification

## Usage

```python
from codeverify_core.org_dashboard import (
    OrgDashboard,
    TeamMetrics,
    RiskHeatmap,
    ROICalculator,
)

dashboard = OrgDashboard(org_name="Acme Corp")

# Add team data
dashboard.add_team_metrics(TeamMetrics(
    team_name="Platform",
    repos=["api-server", "auth-service"],
    total_analyses=250,
    critical_found=3,
    high_found=12,
    auto_fixed=8,
))

# Generate risk heatmap
heatmap = RiskHeatmap()
heatmap.add_entry("src/auth/", risk_score=8.5, reason="High complexity, frequent changes")
heatmap.add_entry("src/billing/", risk_score=6.2, reason="External API integration")

# Calculate ROI
roi = ROICalculator(
    monthly_cost=500.0,
    bugs_caught=45,
    avg_bug_cost=2500.0,
)
print(f"Monthly ROI: {roi.calculate():.0f}%")  # 22,400%
```

## CSV Export

```python
# Export metrics for external tools
csv_data = dashboard.export_csv()
with open("org-metrics.csv", "w") as f:
    f.write(csv_data)
```

## Dashboard Views

### Team Comparison Table

| Team       | Analyses | Critical | High | Auto-Fixed | Score |
|------------|:--------:|:--------:|:----:|:----------:|:-----:|
| Platform   | 250      | 3        | 12   | 8          | 8.2   |
| Frontend   | 180      | 0        | 5    | 4          | 9.1   |
| Data       | 120      | 1        | 8    | 3          | 7.8   |

### Risk Heatmap

```
src/
├── auth/      ████████░░ 8.5  ← High risk
├── billing/   ██████░░░░ 6.2
├── api/       ████░░░░░░ 4.0
└── utils/     ██░░░░░░░░ 1.5  ← Low risk
```

:::note
Dashboard data is aggregated from verification results stored in the database. For real-time dashboards, connect the metrics endpoint to Grafana or your preferred monitoring tool.
:::
