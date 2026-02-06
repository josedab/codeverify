# CodeVerify GitHub Action

AI-powered code verification with formal proofs using Z3 SMT solver.

## Quick Start

Add CodeVerify to your repository in just one step:

```yaml
# .github/workflows/codeverify.yml
name: CodeVerify
on: [pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: codeverify/codeverify-action@v1
```

## Verification Tiers

| Feature | Free | Pro | Enterprise |
|---------|------|-----|------------|
| Pattern-based analysis | ✅ | ✅ | ✅ |
| AI semantic analysis | ❌ | ✅ | ✅ |
| Z3 formal verification | ❌ | ❌ | ✅ |
| Supply chain audit | ❌ | ✅ | ✅ |
| Auto-fix suggestions | ❌ | ✅ | ✅ |
| Formal proof certificates | ❌ | ❌ | ✅ |
| Custom rules | ❌ | ❌ | ✅ |
| Files per run | 100 | 1,000 | 10,000 |
| Monthly runs | 100 | 1,000 | Unlimited |
| Price | Free | $29/mo | $199/mo |

## Usage Examples

### Free Tier (Pattern Analysis)

```yaml
- uses: codeverify/codeverify-action@v1
  with:
    tier: free
    paths: '**/*.py,**/*.ts'
    fail-on: high
```

### Pro Tier (AI Analysis)

```yaml
- uses: codeverify/codeverify-action@v1
  with:
    tier: pro
    api-key: ${{ secrets.CODEVERIFY_API_KEY }}
    paths: '**/*.py,**/*.ts,**/*.js'
    enable-supply-chain: 'true'
    fail-on: medium
```

### Enterprise Tier (Full Verification)

```yaml
- uses: codeverify/codeverify-action@v1
  with:
    tier: enterprise
    api-key: ${{ secrets.CODEVERIFY_API_KEY }}
    paths: '**/*.py,**/*.ts,**/*.js,**/*.go'
    enable-supply-chain: 'true'
    enable-sarif: 'true'
    fail-on: low
    config-file: 'codeverify.yml'
```

## Inputs

| Input | Description | Default |
|-------|-------------|---------|
| `tier` | Verification tier: `free`, `pro`, `enterprise` | `free` |
| `api-key` | CodeVerify API key (required for pro/enterprise) | - |
| `paths` | Glob patterns for files to verify | `**/*.py,**/*.ts,**/*.js` |
| `exclude-paths` | Glob patterns to exclude | `**/node_modules/**` |
| `fail-on` | Severity threshold: `critical`, `high`, `medium`, `low`, `never` | `high` |
| `config-file` | Path to codeverify.yml | - |
| `enable-supply-chain` | Enable dependency verification | `true` |
| `enable-pr-comments` | Post results as PR comments | `true` |
| `enable-sarif` | Generate SARIF for GitHub Security | `true` |
| `python-version` | Python version to use | `3.11` |

## Outputs

| Output | Description |
|--------|-------------|
| `status` | `passed`, `failed`, or `warning` |
| `issues-found` | Total number of issues |
| `critical-count` | Number of critical issues |
| `high-count` | Number of high severity issues |
| `report-url` | URL to full report |
| `sarif-file` | Path to SARIF file |

## Configuration File

Create a `codeverify.yml` in your repository root:

```yaml
# codeverify.yml
version: 1

# Severity threshold for failing checks
fail_on: high

# File patterns to include
include:
  - "src/**/*.py"
  - "src/**/*.ts"

# File patterns to exclude
exclude:
  - "**/test/**"
  - "**/tests/**"
  - "**/__pycache__/**"

# Custom rules (Enterprise only)
rules:
  # Disable specific rules
  disabled:
    - codeverify/pattern/EVAL_USAGE

  # Custom severity overrides
  severity_overrides:
    codeverify/pattern/ANY_CAST: low

# Supply chain configuration
supply_chain:
  enabled: true
  check_typosquatting: true
  check_version_drift: true
  max_age_days: 365

# Z3 verification settings (Enterprise only)
z3:
  enabled: true
  timeout_seconds: 30
  generate_proofs: true
```

## GitHub Security Integration

CodeVerify automatically uploads SARIF results to GitHub Security:

1. Enable SARIF output (enabled by default)
2. View results in the Security tab > Code scanning alerts
3. Filter by CodeVerify rules

## PR Comments

When enabled, CodeVerify posts detailed comments on PRs:

- Summary of issues by severity
- Expandable file-by-file breakdown
- Direct links to issue locations
- Fix suggestions (Pro/Enterprise)
- Proof status indicators (Enterprise)

## Workflow Templates

Copy one of these templates to get started:

- [`codeverify-free.yml`](workflow-templates/codeverify-free.yml) - Basic pattern analysis
- [`codeverify-pro.yml`](workflow-templates/codeverify-pro.yml) - AI-powered analysis
- [`codeverify-enterprise.yml`](workflow-templates/codeverify-enterprise.yml) - Full formal verification

## Getting an API Key

1. Sign up at [codeverify.dev](https://codeverify.dev)
2. Create a new API key in Settings
3. Add as `CODEVERIFY_API_KEY` in repository secrets

## Support

- [Documentation](https://docs.codeverify.dev)
- [GitHub Issues](https://github.com/codeverify/codeverify-action/issues)
- [Discord Community](https://discord.gg/codeverify)
- Enterprise support: enterprise@codeverify.dev
