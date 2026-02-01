---
sidebar_position: 2
---

# Repository Configuration

Configure basic repository settings in `.codeverify.yml`.

## Languages

Specify which languages to analyze:

```yaml
languages:
  - python           # .py files
  - typescript       # .ts, .tsx files
  - javascript       # .js, .jsx files
  - go               # .go files (beta)
```

CodeVerify auto-detects languages if not specified, but explicit configuration is recommended for:
- Faster analysis (skip detection)
- Excluding unwanted languages
- Ensuring consistent behavior

## File Patterns

### Include Patterns

Limit analysis to specific paths:

```yaml
include:
  - "src/**/*"
  - "lib/**/*"
  - "app/**/*.py"
```

If `include` is not specified, all files matching the languages are analyzed.

### Exclude Patterns

Skip files from analysis:

```yaml
exclude:
  # Dependencies
  - "node_modules/**"
  - "venv/**"
  - "vendor/**"
  
  # Build outputs
  - "dist/**"
  - "build/**"
  - "out/**"
  
  # Generated code
  - "**/*.generated.*"
  - "**/*.min.js"
  - "**/generated/**"
  
  # Tests (optional - you may want to analyze tests)
  - "tests/**"
  - "**/*.test.*"
  - "**/*.spec.*"
  - "**/__tests__/**"
  
  # Config files
  - "**/*.config.js"
  - "**/setup.py"
```

### Pattern Syntax

Patterns use glob syntax:

| Pattern | Matches |
|---------|---------|
| `*` | Any characters in a filename |
| `**` | Any directory depth |
| `?` | Single character |
| `[abc]` | Character class |
| `{a,b}` | Alternatives |

Examples:
- `src/**/*.py` — All Python files under `src/`
- `**/*.test.ts` — All TypeScript test files
- `lib/**` — Everything under `lib/`
- `*.{js,ts}` — All JS and TS files in root

## Thresholds

Set pass/fail thresholds by severity:

```yaml
thresholds:
  critical: 0    # Any critical finding = fail
  high: 0        # Any high finding = fail
  medium: 5      # Up to 5 medium findings allowed
  low: 10        # Up to 10 low findings allowed
```

### Threshold Strategies

**Strict (new projects):**
```yaml
thresholds:
  critical: 0
  high: 0
  medium: 0
  low: 0
```

**Balanced (most projects):**
```yaml
thresholds:
  critical: 0
  high: 0
  medium: 5
  low: 10
```

**Lenient (legacy projects):**
```yaml
thresholds:
  critical: 0
  high: 3
  medium: 10
  low: 999   # Effectively unlimited
```

**Warning-only (onboarding):**
```yaml
thresholds:
  critical: 999
  high: 999
  medium: 999
  low: 999
```

## Ignore Rules

Suppress findings for specific patterns:

```yaml
ignore:
  # Ignore all findings in tests
  - pattern: "tests/**"
    reason: "Test code has different standards"
    
  # Ignore specific categories in migrations
  - pattern: "migrations/**"
    categories:
      - null_safety
      - type_safety
    reason: "Auto-generated Django migrations"
    
  # Ignore specific severities
  - pattern: "scripts/**"
    severities:
      - low
      - medium
    reason: "Utility scripts, not production code"
    
  # Ignore by finding ID (for persistent false positives)
  - finding_id: "f_abc123"
    reason: "Known false positive, see ISSUE-456"
```

### Ignore Categories

| Category | Description |
|----------|-------------|
| `null_safety` | Null/undefined dereference checks |
| `array_bounds` | Array index bounds checks |
| `division_by_zero` | Division by zero checks |
| `integer_overflow` | Arithmetic overflow checks |
| `security` | All security findings |
| `style` | Style and convention findings |
| `type_safety` | Type-related findings |

## PR Behavior

Configure how CodeVerify interacts with pull requests:

```yaml
pr:
  # Never auto-approve PRs (recommended)
  auto_approve: false
  
  # Comment even when all checks pass
  comment_on_pass: true
  
  # Collapse finding details by default
  collapse_findings: true
  
  # Limit inline comments to reduce noise
  max_inline_comments: 10
  
  # Show summary table
  show_summary: true
  
  # Request changes vs just comment
  request_changes_on_fail: true
```

## Reporting

Configure output format:

```yaml
reporting:
  # Output format
  format: "auto"     # auto, text, json, sarif, github, junit
  
  # Output destination
  output: "stdout"   # Or file path like "reports/codeverify.json"
  
  # Show detailed output
  verbose: false
  
  # Include passing files in report
  include_passed: false
  
  # Include suppressed findings
  include_suppressed: false
```

### Format Options

| Format | Use Case |
|--------|----------|
| `auto` | Detects CI environment, uses appropriate format |
| `text` | Human-readable terminal output |
| `json` | Machine-readable, for custom tooling |
| `sarif` | GitHub Code Scanning integration |
| `github` | GitHub Actions annotations |
| `junit` | Test reporter integration |

## Examples

### Python Web App

```yaml
version: "1"

languages:
  - python

include:
  - "app/**"
  - "lib/**"

exclude:
  - "tests/**"
  - "migrations/**"
  - "**/*.pyc"

thresholds:
  critical: 0
  high: 0
  medium: 3
  low: 10

ignore:
  - pattern: "migrations/**"
    reason: "Auto-generated Django migrations"
```

### TypeScript React App

```yaml
version: "1"

languages:
  - typescript

include:
  - "src/**"

exclude:
  - "node_modules/**"
  - "build/**"
  - "**/*.test.tsx"
  - "**/*.stories.tsx"

thresholds:
  critical: 0
  high: 0
  medium: 5
  low: 20
```

### Monorepo

```yaml
version: "1"

languages:
  - python
  - typescript

exclude:
  - "node_modules/**"
  - "**/node_modules/**"
  - "**/dist/**"
  - "**/tests/**"

monorepo:
  enabled: true
  affected_analysis: true

thresholds:
  critical: 0
  high: 0
  medium: 10
  low: 50
```
