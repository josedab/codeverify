---
sidebar_position: 2
---

# Monorepo Support

Efficiently analyze monorepos with affected-only analysis and per-package configuration.

## Overview

CodeVerify understands monorepo structures and provides:
- **Affected analysis** — Only analyze packages changed by a PR
- **Per-package config** — Different rules for different packages
- **Parallel verification** — Analyze packages concurrently
- **Unified reporting** — Single summary across all packages

## Configuration

### Enable Monorepo Mode

```yaml
# .codeverify.yml (root)
version: "1"

monorepo:
  enabled: true
  
  # Detect packages automatically
  auto_detect: true
  
  # Or specify package locations
  packages:
    - "packages/*"
    - "apps/*"
    - "services/*"
```

### Package Detection

CodeVerify auto-detects packages by looking for:
- `package.json`
- `pyproject.toml`
- `setup.py`
- `go.mod`
- `Cargo.toml`

Or specify explicitly:

```yaml
monorepo:
  packages:
    - path: "packages/api"
      name: "api"
      languages: [python]
      
    - path: "packages/web"
      name: "web"
      languages: [typescript]
      
    - path: "packages/shared"
      name: "shared"
      languages: [typescript]
```

## Affected Analysis

Only analyze packages affected by changes:

### How It Works

```
PR: Add new API endpoint

Changed files:
  - packages/api/src/handlers.py
  - packages/api/tests/test_handlers.py

Dependency graph:
  web → shared → api

Affected packages:
  - api (direct change)
  - shared (depends on api)
  - web (depends on shared)

CodeVerify analyzes: api, shared, web
```

### Configuration

```yaml
monorepo:
  affected_analysis: true
  
  # Consider test files when determining affected packages
  include_tests: true
  
  # Base branch for comparison
  base_branch: "main"
  
  # Dependency detection
  dependencies:
    # Detect from package.json, requirements.txt, etc.
    auto_detect: true
    
    # Or specify manually
    graph:
      web: [shared, api]
      shared: [api]
      api: []
```

### CLI Usage

```bash
# Analyze only affected packages
codeverify analyze --affected

# Compare against specific branch
codeverify analyze --affected --base origin/main

# Analyze specific packages
codeverify analyze --packages api,shared
```

## Per-Package Configuration

### Package-Level Config Files

Each package can have its own `.codeverify.yml`:

```
monorepo/
├── .codeverify.yml              # Root config (defaults)
├── packages/
│   ├── api/
│   │   ├── .codeverify.yml      # API-specific config
│   │   └── src/
│   ├── web/
│   │   ├── .codeverify.yml      # Web-specific config
│   │   └── src/
│   └── shared/
│       └── src/
```

### Root Config

```yaml
# .codeverify.yml (root)
version: "1"

monorepo:
  enabled: true
  packages:
    - "packages/*"

# Default settings for all packages
languages:
  - python
  - typescript

thresholds:
  critical: 0
  high: 0
  medium: 5
  low: 10
```

### Package Config (Override)

```yaml
# packages/api/.codeverify.yml
# Inherits from root, can override

languages:
  - python

# Stricter thresholds for API
thresholds:
  critical: 0
  high: 0
  medium: 0
  low: 5

# API-specific checks
verification:
  checks:
    - null_safety
    - division_by_zero
    - security    # Additional security focus
```

### Inheritance

Configs are merged with package taking precedence:

```yaml
# Effective config for packages/api:
languages: [python]                    # from package
thresholds:
  critical: 0                          # from package
  high: 0                              # from package
  medium: 0                            # from package (override)
  low: 5                               # from package (override)
verification:
  checks: [null_safety, division_by_zero, security]  # from package
```

## Parallel Analysis

```yaml
monorepo:
  parallel:
    enabled: true
    max_workers: 4
```

Analysis runs concurrently:

```
Analyzing monorepo...

[api]    ████████░░ 80% | 12s
[web]    ██████░░░░ 60% | 10s  
[shared] ██████████ 100% | 8s ✓

Completed: shared (0 findings)
Completed: api (2 findings)
Completed: web (1 finding)

Total: 3 findings across 3 packages
```

## Reporting

### Unified Summary

```
CodeVerify Monorepo Report
══════════════════════════════════════════════════════════════

Packages analyzed: 3 (5 affected, 2 unchanged)
Total findings: 7

Per-Package Summary:
┌──────────┬──────────┬────────┬────────┬─────┬─────┬───────┐
│ Package  │ Status   │ Crit   │ High   │ Med │ Low │ Score │
├──────────┼──────────┼────────┼────────┼─────┼─────┼───────┤
│ api      │ ❌ Fail  │ 0      │ 2      │ 1   │ 0   │ 65    │
│ web      │ ⚠ Warn   │ 0      │ 0      │ 3   │ 1   │ 78    │
│ shared   │ ✅ Pass  │ 0      │ 0      │ 0   │ 0   │ 95    │
└──────────┴──────────┴────────┴────────┴─────┴─────┴───────┘

Overall: ❌ FAIL (api has high severity findings)
```

### Per-Package Details

```bash
# Show findings for specific package
codeverify report --package api

# Export per-package reports
codeverify analyze --output-dir reports/
# Creates: reports/api.json, reports/web.json, reports/shared.json
```

### GitHub PR Comment

```markdown
## CodeVerify Monorepo Analysis

### Summary
| Package | Status | Findings | Trust Score |
|---------|--------|----------|-------------|
| `api` | ❌ Fail | 3 | 65 |
| `web` | ⚠️ Warn | 4 | 78 |
| `shared` | ✅ Pass | 0 | 95 |

### api (3 findings)
<details>
<summary>❌ High: SQL injection risk</summary>
...
</details>

### web (4 findings)
<details>
<summary>⚠️ Medium: Missing error handling</summary>
...
</details>
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/codeverify.yml
name: CodeVerify

on:
  pull_request:
    branches: [main]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for affected detection
      
      - name: Get changed packages
        id: changes
        run: |
          PACKAGES=$(git diff --name-only origin/main | 
            grep -E '^packages/' | 
            cut -d/ -f2 | 
            sort -u | 
            tr '\n' ',')
          echo "packages=$PACKAGES" >> $GITHUB_OUTPUT
      
      - name: Run CodeVerify
        uses: codeverify/action@v1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          mode: monorepo
          affected_only: true
```

### Matrix Strategy

Run packages in parallel jobs:

```yaml
jobs:
  detect:
    runs-on: ubuntu-latest
    outputs:
      packages: ${{ steps.detect.outputs.packages }}
    steps:
      - uses: actions/checkout@v4
      - id: detect
        run: |
          codeverify packages --affected --json > packages.json
          echo "packages=$(cat packages.json)" >> $GITHUB_OUTPUT

  analyze:
    needs: detect
    runs-on: ubuntu-latest
    strategy:
      matrix:
        package: ${{ fromJson(needs.detect.outputs.packages) }}
    steps:
      - uses: actions/checkout@v4
      - run: codeverify analyze packages/${{ matrix.package }}
```

## Best Practices

1. **Start with affected analysis** — Faster feedback on PRs
2. **Tune per-package thresholds** — Stricter for critical packages
3. **Use parallel analysis** — Speed up large monorepos
4. **Cache between runs** — Reuse Z3 results where possible
5. **Separate configs** — Don't force one-size-fits-all rules

## Next Steps

- [CI/CD Integration](/docs/integrations/ci-cd) — Full CI setup
- [Configuration Reference](/docs/configuration/overview) — All config options
- [Custom Rules](/docs/configuration/custom-rules) — Package-specific rules
