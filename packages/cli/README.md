# CodeVerify CLI

Command-line interface for running CodeVerify analysis locally.

## Installation

```bash
pip install codeverify-cli
```

Or install from source:

```bash
cd packages/cli
pip install -e .
```

## Quick Start

```bash
# Initialize configuration
codeverify init

# Analyze current directory
codeverify analyze

# Analyze specific path
codeverify analyze src/

# Analyze only staged git files
codeverify analyze --staged

# Output as JSON
codeverify analyze -f json > report.json

# Output as SARIF (for IDE integration)
codeverify analyze -f sarif > report.sarif
```

## Commands

### `codeverify analyze`

Run analysis on code files.

```bash
codeverify analyze [PATH] [OPTIONS]

Options:
  -c, --config PATH     Path to .codeverify.yml
  -f, --format FORMAT   Output format: rich, json, sarif (default: rich)
  -s, --severity LEVEL  Minimum severity to report
  --fix                 Show fix suggestions
  --staged              Only analyze staged git files
  --fail-on LEVEL       Exit with error if findings at this level
```

### `codeverify init`

Create a new `.codeverify.yml` configuration file.

```bash
codeverify init
codeverify init --force  # Overwrite existing
```

### `codeverify validate`

Validate configuration file syntax.

```bash
codeverify validate
codeverify validate -c custom-config.yml
```

### `codeverify fix`

View and apply suggested fixes.

```bash
codeverify fix src/
codeverify fix src/ --fix      # Apply fixes
codeverify fix src/ --dry-run  # Preview changes
```

### `codeverify test-rule`

Test custom rule definitions.

```bash
codeverify test-rule my-rule.yml
codeverify test-rule my-rule.yml -t test-file.py
```

### `codeverify status`

Show analysis status and configuration.

```bash
codeverify status
```

## Configuration

Create a `.codeverify.yml` in your project root:

```yaml
version: "1"

languages:
  - python
  - typescript

include:
  - "src/**/*"

exclude:
  - "**/node_modules/**"
  - "**/*.test.*"

thresholds:
  critical: 0
  high: 0
  medium: 5
  low: 10

verification:
  enabled: true
  timeout_seconds: 30
  checks:
    - null_safety
    - array_bounds
    - integer_overflow

custom_rules:
  - id: no-print
    name: No Print Statements
    severity: low
    pattern: "print\\s*\\("
```

## Exit Codes

- `0`: Success, no issues at or above `--fail-on` level
- `1`: Issues found at or above `--fail-on` level
- `2`: Configuration or runtime error

## Integration

### Pre-commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: codeverify
        name: CodeVerify
        entry: codeverify analyze --staged --fail-on high
        language: system
        pass_filenames: false
```

### GitHub Actions

```yaml
- name: Run CodeVerify
  run: |
    pip install codeverify-cli
    codeverify analyze --fail-on high -f sarif > results.sarif

- name: Upload SARIF
  uses: github/codeql-action/upload-sarif@v2
  with:
    sarif_file: results.sarif
```

### VS Code

Use SARIF output with the SARIF Viewer extension:

```bash
codeverify analyze -f sarif > .codeverify-results.sarif
```
