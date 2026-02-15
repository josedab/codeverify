---
sidebar_position: 3
---

# Auto-Fix Pipeline

Automatically generate and verify fixes for detected issues.

## Overview

The Auto-Fix Pipeline goes beyond finding bugs — it generates fixes, verifies them against the original verification constraints, and presents only verified solutions. Each fix attempt goes through up to 3 verification iterations to ensure correctness.

## How It Works

```
Finding Detected
      │
      ▼
┌─────────────┐
│ FixGenerator │──── Generate candidate fix
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ FixVerifier  │──── Verify fix doesn't introduce new issues
└─────┬───────┘
      │
  ┌───┴───┐
  │ Pass? │
  └───┬───┘
   No │ Yes
      │  └──▶ Return verified fix
      ▼
  Retry (up to 3x)
```

## Usage

```python
from codeverify_core.autofix_loop import AutoFixPipeline, Finding

pipeline = AutoFixPipeline(max_iterations=3)

finding = Finding(
    rule_id="null-deref",
    message="Potential null dereference on line 42",
    severity="high",
    file_path="src/handler.py",
    line=42,
)

result = pipeline.fix_finding(finding, source_code)

if result.is_verified:
    print(f"Verified fix: {result.explanation}")
    print(f"Patched code:\n{result.patched_code}")
else:
    print(f"Could not verify fix after {result.iterations} attempts")
```

## Fix Confidence Levels

| Level    | Description                               | Action         |
|----------|-------------------------------------------|----------------|
| HIGH     | Fix verified against all constraints      | Auto-apply     |
| MEDIUM   | Fix passes basic checks, not full proof   | Suggest        |
| LOW      | Fix generated but not verified            | Review only    |

## Integration with CI/CD

In the worker pipeline, auto-fix runs automatically for critical and high severity findings:

```yaml
# .codeverify.yml
autofix:
  enabled: true
  max_iterations: 3
  apply_confidence: high  # Only auto-apply HIGH confidence fixes
  create_suggestions: true  # Create GitHub suggested changes for MEDIUM
```

:::note
Auto-fix currently generates textual explanations and suggestions. One-click apply via GitHub suggested changes is available when running through the GitHub integration.
:::
