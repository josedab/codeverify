---
sidebar_position: 2
---

# Quick Start

Get from zero to your first verified code in under 5 minutes.

## 1. Install CodeVerify

```bash
pip install codeverify
```

## 2. Set Up API Key (Optional)

For AI-powered analysis, set an LLM API key:

```bash
export OPENAI_API_KEY="sk-..."
```

:::note
Without an API key, you still get Z3 formal verification. AI analysis requires an API key.
:::

## 3. Analyze Your Code

### Single File

```bash
codeverify analyze src/main.py
```

### Directory

```bash
codeverify analyze src/
```

### With Specific Checks

```bash
codeverify analyze src/ --checks null_safety,division_by_zero
```

## 4. Understand the Output

CodeVerify outputs findings grouped by severity:

```
CodeVerify Analysis Results
═══════════════════════════

📁 Analyzed: src/calculator.py (3 functions, 45 lines)
⏱️  Duration: 2.3s

🔴 CRITICAL (1)
───────────────
Division by zero possible
  Location: src/calculator.py:15:12
  Function: divide(a, b)
  
  Verification: FORMAL (Z3 Proof)
  Counterexample: a=1, b=0
  
  def divide(a: int, b: int) -> float:
      return a / b  # ← Issue here
             ^^^^^
  
  Suggested Fix:
  + if b == 0:
  +     raise ValueError("divisor cannot be zero")
    return a / b

🟠 HIGH (1)
───────────
Array index may be out of bounds
  Location: src/calculator.py:22:12
  Function: get_element(items, index)
  
  Verification: FORMAL (Z3 Proof)
  Counterexample: items=[], index=0

Summary: 1 critical, 1 high, 0 medium, 0 low
Status: ❌ FAILED (threshold: 0 critical, 0 high)
```

## 5. Configure for Your Project

Create `.codeverify.yml` in your repository root:

```yaml title=".codeverify.yml"
version: "1"

# Languages to analyze
languages:
  - python
  - typescript

# Verification settings
verification:
  enabled: true
  timeout: 30
  checks:
    - null_safety
    - array_bounds
    - integer_overflow
    - division_by_zero

# AI analysis settings
ai:
  enabled: true
  semantic: true
  security: true

# Files to exclude
exclude:
  - "tests/**"
  - "node_modules/**"
  - "**/*.test.py"

# Pass/fail thresholds
thresholds:
  critical: 0   # Any critical = fail
  high: 0       # Any high = fail
  medium: 5     # Up to 5 medium allowed
  low: 10       # Up to 10 low allowed
```

## 6. Integrate with GitHub

Install the GitHub App for automatic PR analysis:

```bash
# Visit the GitHub App installation page
open https://github.com/apps/codeverify
```

Or add to your CI workflow:

```yaml title=".github/workflows/codeverify.yml"
name: CodeVerify
on: [pull_request]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install codeverify
      - run: codeverify analyze src/ --format github
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Common Commands

| Command | Description |
|---------|-------------|
| `codeverify analyze <path>` | Analyze files or directories |
| `codeverify analyze --watch` | Watch mode for continuous analysis |
| `codeverify check <file>` | Quick check without AI (faster) |
| `codeverify explain <finding-id>` | Get detailed explanation of a finding |
| `codeverify fix <file>` | Auto-fix issues where possible |
| `codeverify init` | Create `.codeverify.yml` interactively |
| `codeverify doctor` | Check installation and configuration |

## Output Formats

```bash
# Default (human-readable)
codeverify analyze src/

# JSON for CI/CD integration
codeverify analyze src/ --format json

# SARIF for GitHub Code Scanning
codeverify analyze src/ --format sarif > results.sarif

# GitHub Actions format
codeverify analyze src/ --format github

# JUnit XML for test reporters
codeverify analyze src/ --format junit
```

## Next Steps

- **[First Analysis](./first-analysis)** — Detailed walkthrough with real examples
- **[Configuration Guide](/docs/configuration/overview)** — All configuration options
- **[GitHub Integration](/docs/integrations/github)** — Set up automatic PR checks
- **[VS Code Extension](/docs/integrations/vscode)** — Real-time verification as you code
