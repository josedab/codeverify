---
sidebar_position: 1
---

# Configuration Overview

Configure CodeVerify using `.codeverify.yml` in your repository root.

## Quick Start

Create `.codeverify.yml`:

```yaml
version: "1"

languages:
  - python
  - typescript

verification:
  enabled: true
  checks:
    - null_safety
    - array_bounds
    - integer_overflow
    - division_by_zero

ai:
  enabled: true
  semantic: true
  security: true

exclude:
  - "node_modules/**"
  - "tests/**"
  - "**/*.test.*"

thresholds:
  critical: 0
  high: 0
  medium: 5
  low: 10
```

## Configuration Hierarchy

CodeVerify loads configuration from multiple sources (later overrides earlier):

1. **Built-in defaults**
2. **Global config**: `~/.codeverify/config.yml`
3. **Repository config**: `.codeverify.yml`
4. **Environment variables**: `CODEVERIFY_*`
5. **CLI arguments**: `--checks`, `--exclude`, etc.

## Complete Configuration Reference

```yaml
# Configuration version (required)
version: "1"

# Languages to analyze
languages:
  - python
  - typescript
  - javascript
  - go                    # Beta support

# Formal verification settings
verification:
  enabled: true
  timeout: 30             # Seconds per check
  checks:
    - null_safety
    - array_bounds
    - integer_overflow
    - division_by_zero
  advanced:
    max_path_depth: 10    # Max execution paths to explore
    incremental: true     # Use incremental solving
    log_proofs: false     # Log Z3 proofs for debugging

# AI analysis settings
ai:
  enabled: true
  semantic: true          # Code intent analysis
  security: true          # Vulnerability scanning
  model: "gpt-4"          # Or "claude-3-sonnet", "gpt-3.5-turbo"
  temperature: 0.1        # Lower = more deterministic
  max_tokens: 4096        # Max response length

# Trust Score for AI-generated code
trust_score:
  enabled: true
  minimum_score: 70
  strict_mode: false
  weights:
    type_safety: 0.20
    null_checks: 0.20
    security: 0.20
    style: 0.15
    documentation: 0.10
    error_handling: 0.15

# File patterns
include:
  - "src/**/*"
  - "lib/**/*"

exclude:
  - "node_modules/**"
  - "venv/**"
  - "dist/**"
  - "build/**"
  - "**/*.min.js"
  - "**/*.generated.*"
  - "tests/**"
  - "**/*.test.*"
  - "**/*.spec.*"

# Pass/fail thresholds
thresholds:
  critical: 0             # Max allowed critical findings
  high: 0                 # Max allowed high findings
  medium: 5               # Max allowed medium findings
  low: 10                 # Max allowed low findings

# Suppress specific findings
ignore:
  - pattern: "tests/**"
    reason: "Test code has different standards"
  - pattern: "migrations/**"
    categories:
      - null_safety
    reason: "Auto-generated code"

# Custom pattern rules
rules:
  - id: no-print
    name: "No print statements"
    pattern: "\\bprint\\s*\\("
    message: "Use logging instead of print"
    severity: low
    languages: [python]
    enabled: true

# PR behavior
pr:
  auto_approve: false
  comment_on_pass: true
  collapse_findings: true
  max_inline_comments: 10
  show_summary: true
  
# Reporting
reporting:
  format: "auto"          # "auto", "text", "json", "sarif", "github"
  output: "stdout"        # Or file path
  verbose: false

# Team/enterprise features
team:
  learning_mode: true
  report_schedule: "weekly"
  
# Monorepo settings
monorepo:
  enabled: true
  affected_analysis: true
  cycle_detection: true
```

## Environment Variables

Override any setting with environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `CODEVERIFY_CHECKS` | Comma-separated checks | `null_safety,division_by_zero` |
| `CODEVERIFY_TIMEOUT` | Verification timeout | `60` |
| `CODEVERIFY_AI_ENABLED` | Enable AI analysis | `true` |
| `CODEVERIFY_AI_MODEL` | LLM model to use | `gpt-4` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `CODEVERIFY_MIN_SEVERITY` | Minimum severity to report | `medium` |

## Configuration Sections

- **[Repository Config](./repository-config)** — Languages, files, thresholds
- **[Verification Settings](./verification-settings)** — Z3 configuration
- **[AI Settings](./ai-settings)** — LLM configuration
- **[Custom Rules](./custom-rules)** — Create your own patterns

## Validation

Validate your configuration:

```bash
codeverify config validate

# Or with a specific file
codeverify config validate --config custom.yml
```

## Generate Config

Generate a starter configuration:

```bash
# Interactive mode
codeverify init

# With defaults
codeverify init --defaults

# For specific languages
codeverify init --languages python,typescript
```
