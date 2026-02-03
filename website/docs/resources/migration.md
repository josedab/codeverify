---
sidebar_position: 6
---

# Migration Guide

Moving to CodeVerify from other static analysis tools? This guide helps you transition smoothly.

## From ESLint / Pylint

### What Changes

| ESLint/Pylint | CodeVerify |
|---------------|------------|
| Pattern-based rules | Pattern + formal verification + AI |
| Fast, low coverage | Slower, high coverage |
| Many false positives | Fewer, proven issues |
| No counterexamples | Provides concrete counterexamples |

### Step 1: Keep Your Existing Linter

CodeVerify complements linters—it doesn't replace them. Keep ESLint/Pylint for:
- Style enforcement
- Import ordering  
- Naming conventions
- Quick feedback in editor

Use CodeVerify for:
- Bug detection (null, bounds, overflow)
- Security vulnerabilities
- AI-generated code review

### Step 2: Install CodeVerify

```bash
pip install codeverify
```

### Step 3: Create Initial Configuration

```yaml
# .codeverify.yml
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

# Start with lenient thresholds
thresholds:
  critical: 0
  high: 5
  medium: 20
  low: 100

exclude:
  - "tests/**"
  - "node_modules/**"
```

### Step 4: Run Initial Analysis

```bash
codeverify analyze src/ --format json > baseline.json
```

Review the findings. Many will be legitimate bugs your linter missed.

### Step 5: Tighten Thresholds Gradually

After fixing critical issues:

```yaml
thresholds:
  critical: 0
  high: 0      # Now enforce
  medium: 10   # Tighter
  low: 50
```

### ESLint Rule Mapping

Some ESLint rules overlap with CodeVerify:

| ESLint Rule | CodeVerify Equivalent |
|-------------|----------------------|
| `no-unused-vars` | Semantic analysis detects unused code |
| `no-undef` | Z3 null safety + type checking |
| `array-callback-return` | Z3 bounds checking |
| `no-unreachable` | Semantic analysis |

You can disable overlapping ESLint rules to reduce noise:

```js
// .eslintrc.js
module.exports = {
  rules: {
    // Let CodeVerify handle these with proofs
    'no-unused-vars': 'warn',  // Keep as warning
    // ... other rules
  }
};
```

## From mypy / TypeScript Strict Mode

### What CodeVerify Adds

mypy and TypeScript catch type errors. CodeVerify goes further:

| Issue | mypy/TS | CodeVerify |
|-------|---------|------------|
| `x: str = None` | ✅ Catches | ✅ Catches |
| `if x: return x.upper()` | ⚠️ Narrows type | ✅ Proves safety |
| `items[0]` (maybe empty) | ❌ Misses | ✅ Catches with proof |
| `a / b` (b might be 0) | ❌ Misses | ✅ Catches with proof |
| SQL injection | ❌ Misses | ✅ AI detection |

### Integration Strategy

Keep your type checker and add CodeVerify:

```yaml
# CI pipeline
steps:
  - run: mypy src/                    # Fast type checking
  - run: codeverify analyze src/      # Deep verification
```

### Handling Type Hints

CodeVerify uses your type hints to improve analysis. Good type hints = better verification:

```python
# Weak hints - CodeVerify must infer more
def process(data):
    return data.get("key")

# Strong hints - CodeVerify can verify more
def process(data: dict[str, str]) -> str | None:
    return data.get("key")
```

## From Semgrep / CodeQL

### Comparison

| Aspect | Semgrep/CodeQL | CodeVerify |
|--------|----------------|------------|
| Analysis type | Pattern matching | Formal + AI + patterns |
| Custom rules | YAML/QL language | YAML patterns + AI |
| Security focus | Primary | Included |
| Bug detection | Limited | Primary (with proofs) |
| Setup complexity | Medium | Low |

### Migration Path

1. **Keep Semgrep for custom patterns** you've invested time in
2. **Add CodeVerify for formal verification** Semgrep can't do
3. **Gradually replace Semgrep security rules** with CodeVerify's AI-powered detection

### Rule Conversion

Semgrep pattern rules can often be converted to CodeVerify:

```yaml
# Semgrep rule
rules:
  - id: hardcoded-password
    pattern: password = "..."
    message: Hardcoded password
    severity: ERROR

# CodeVerify equivalent (in .codeverify.yml)
custom_rules:
  - id: hardcoded-password
    pattern:
      type: assignment
      left: { name: { regex: "password|secret|key" } }
      right: { type: string_literal }
    message: Hardcoded credential detected
    severity: critical
```

## From SonarQube

### Key Differences

| SonarQube | CodeVerify |
|-----------|------------|
| Comprehensive quality metrics | Focused on correctness |
| Technical debt tracking | Issue tracking |
| Many language-specific rules | Formal verification |
| On-premise or cloud | Self-hosted or cloud |
| Quality gates | Thresholds |

### Coexistence Strategy

Many teams use both:

- **SonarQube** for: code smells, duplication, complexity metrics, coverage
- **CodeVerify** for: formal verification, AI review, security

```yaml
# CI pipeline with both
jobs:
  quality:
    steps:
      - run: sonar-scanner    # Quality metrics
      - run: codeverify analyze src/  # Verification
```

### Migrating Quality Gates

SonarQube quality gate → CodeVerify thresholds:

```yaml
# SonarQube: block on any new bugs
# CodeVerify equivalent:
thresholds:
  critical: 0
  high: 0
```

## Gradual Rollout Strategy

### Phase 1: Shadow Mode (Week 1-2)

Run CodeVerify without blocking:

```yaml
# GitHub Actions
- name: CodeVerify (shadow)
  continue-on-error: true
  run: codeverify analyze src/
```

Review findings to understand signal quality.

### Phase 2: Warning Mode (Week 3-4)

Add PR comments but don't block:

```yaml
thresholds:
  critical: 100  # Warn but don't fail
  high: 100
```

### Phase 3: Enforce Critical (Month 2)

Block on critical issues:

```yaml
thresholds:
  critical: 0
  high: 50
```

### Phase 4: Full Enforcement (Month 3+)

```yaml
thresholds:
  critical: 0
  high: 0
  medium: 10
```

## Handling Existing Issues

When migrating, you may have many existing findings.

### Option 1: Baseline File

Create a baseline to ignore existing issues:

```bash
codeverify analyze src/ --create-baseline > .codeverify-baseline.json
```

Then in config:

```yaml
baseline: .codeverify-baseline.json
```

New PRs only see new issues.

### Option 2: Suppress with Comments

For known false positives:

```python
# codeverify-disable-next-line null_safety
result = external_api()  # API guarantees non-null

result.process()  # codeverify-disable-line division_by_zero: handled upstream
```

### Option 3: Gradual Fix

Track issues and fix over time:

```bash
# Export to CSV for tracking
codeverify analyze src/ --format csv > issues.csv
```

## Common Migration Issues

### "Too Many Findings"

Start with strict exclusions and loosen over time:

```yaml
exclude:
  - "tests/**"
  - "scripts/**"
  - "migrations/**"
  - "generated/**"
```

### "Analysis Too Slow"

Analyze only changed files in CI:

```yaml
- uses: codeverify/action@v1
  with:
    changed_only: true
```

### "False Positives"

1. Check if it's actually a bug (CodeVerify is often right)
2. Add type hints to help the analyzer
3. Suppress with inline comments
4. Report to us for tuning

## Getting Help

- [GitHub Discussions](https://github.com/codeverify/codeverify/discussions) — Ask migration questions
- [Discord](https://discord.gg/codeverify) — Real-time help
- [FAQ](/docs/resources/faq) — Common questions
