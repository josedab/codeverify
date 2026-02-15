---
sidebar_position: 6
---

# CI/CD Native Actions

Generate CI/CD configurations with SARIF output and quality gates.

## Overview

The CI/CD Actions module generates ready-to-use pipeline configurations for GitHub Actions, GitLab CI, Jenkins, and CircleCI. It includes SARIF 2.1.0 report generation and a quality gate evaluator for enforcing code standards.

## Supported Platforms

| Platform       | Config Format   | SARIF Upload | Quality Gates |
|---------------|-----------------|:------------:|:-------------:|
| GitHub Actions | YAML workflow   | ✅           | ✅            |
| GitLab CI      | `.gitlab-ci.yml`| ✅           | ✅            |
| Jenkins        | `Jenkinsfile`   | ✅           | ✅            |
| CircleCI       | `config.yml`    | ✅           | ✅            |

## Quick Start

```python
from codeverify_core.cicd_actions import CICDActionRunner

runner = CICDActionRunner(platform="github")

# Generate pipeline config
config = runner.generate_config(
    paths=["src/**/*.py", "src/**/*.ts"],
    fail_on="high",
    enable_sarif=True,
)

print(config)  # Ready-to-use YAML
```

## Quality Gates

Define pass/fail thresholds for your pipeline:

```python
from codeverify_core.cicd_actions import QualityGateEvaluator, QualityGateConfig

config = QualityGateConfig(
    max_critical=0,    # Zero tolerance for critical
    max_high=2,        # Allow up to 2 high severity
    max_medium=10,     # Allow up to 10 medium
    max_low=999,       # Effectively unlimited
)

evaluator = QualityGateEvaluator(config)

result = evaluator.evaluate({
    "critical": 0,
    "high": 1,
    "medium": 5,
    "low": 12,
})

print(f"Gate passed: {result.passed}")
print(f"Reason: {result.reason}")
```

## SARIF Reports

Generate standard SARIF 2.1.0 output for integration with GitHub Security tab:

```python
from codeverify_core.cicd_actions import SARIFReport

report = SARIFReport(tool_name="CodeVerify", version="0.8.0")

report.add_result(
    rule_id="null-deref",
    message="Potential null dereference",
    file_path="src/handler.py",
    line=42,
    severity="error",
)

sarif_json = report.to_dict()
```

## GitHub Action Integration

The GitHub Action automatically uses quality gates and SARIF:

```yaml
- uses: codeverify/action@v2
  with:
    tier: pro
    fail-on: high
    sarif: true
    paths: "src/**/*.py,src/**/*.ts,src/**/*.rs"
```

:::note
SARIF reports are automatically uploaded to GitHub's Security tab when running as a GitHub Action, making findings visible alongside Dependabot and CodeQL alerts.
:::
