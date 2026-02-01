---
sidebar_position: 3
---

# Understanding Findings

Every CodeVerify analysis produces findings—detailed reports of potential issues with actionable information.

## Finding Structure

Each finding contains:

```json
{
  "id": "f_abc123def456",
  "type": "division_by_zero",
  "severity": "critical",
  "confidence": 0.98,
  "verification_method": "formal",
  "location": {
    "file": "src/math.py",
    "line_start": 15,
    "line_end": 15,
    "column_start": 12,
    "column_end": 17
  },
  "title": "Division by zero possible",
  "description": "The divisor 'count' can be zero when this division executes.",
  "code_snippet": "return total / count",
  "counterexample": {
    "total": 100,
    "count": 0
  },
  "fix_suggestion": {
    "description": "Add a guard clause to check for zero",
    "code": "if count == 0:\n    raise ValueError('count cannot be zero')\nreturn total / count"
  }
}
```

## Severity Levels

CodeVerify uses four severity levels:

### 🔴 Critical

**Immediate crash, security breach, or data corruption.**

Examples:
- Division by zero
- SQL injection
- Null dereference on critical path
- Buffer overflow

**Default action:** Block PR, require immediate fix

### 🟠 High

**Significant bug or vulnerability that will cause problems.**

Examples:
- Array out-of-bounds access
- Missing authentication check
- Resource leak
- Race condition

**Default action:** Block PR, prioritize fix

### 🟡 Medium

**Potential issue that warrants review.**

Examples:
- Missing input validation
- Inefficient algorithm
- Potential null in edge case
- Deprecated API usage

**Default action:** Warn, allow merge with review

### 🔵 Low

**Minor suggestion or style improvement.**

Examples:
- Code smell
- Missing documentation
- Naming convention violation
- TODO/FIXME comment

**Default action:** Inform, allow merge

## Confidence Scores

Every finding includes a confidence score (0-100%):

| Range | Meaning | Recommended Action |
|-------|---------|-------------------|
| **90-100%** | Almost certainly a real issue | Fix immediately |
| **70-89%** | Very likely an issue | Investigate and fix |
| **50-69%** | Possibly an issue | Review carefully |
| **Below 50%** | Uncertain | Check context, may suppress |

### Confidence by Verification Method

| Method | Typical Confidence | Why |
|--------|-------------------|-----|
| Formal (Z3) | 95-99% | Mathematical proof |
| AI Security | 85-95% | Trained on known patterns |
| AI Semantic | 75-90% | Context-dependent |
| Pattern | 60-80% | Simple matching |

## Counterexamples

For formal verification findings, CodeVerify provides **counterexamples**—specific input values that trigger the bug:

```
🔴 CRITICAL: Division by zero possible

Location: src/stats.py:23
Function: calculate_average(numbers: list[int]) -> float

Counterexample:
  numbers = []
  
Execution trace:
  1. calculate_average([]) called
  2. total = sum([]) = 0
  3. count = len([]) = 0
  4. return 0 / 0  ← ZeroDivisionError
```

Counterexamples are invaluable because they:
- **Prove** the bug exists (not a guess)
- Show **exactly** how to reproduce it
- Help you understand the edge case
- Can be converted to test cases

## Fix Suggestions

CodeVerify generates actionable fix suggestions:

```
Suggested Fix:

- return total / count
+ if count == 0:
+     raise ValueError("cannot calculate average of empty list")
+ return total / count

Or alternatively:
+ if not numbers:
+     return 0.0  # Or raise, depending on requirements
+ return sum(numbers) / len(numbers)
```

Fix quality depends on:
- Verification method (formal fixes are more precise)
- Code complexity
- Available context

## Filtering Findings

### By Severity

```bash
# Only critical and high
codeverify analyze src/ --min-severity high

# Exclude low severity
codeverify analyze src/ --min-severity medium
```

### By Type

```bash
# Only specific checks
codeverify analyze src/ --checks null_safety,division_by_zero

# Exclude specific checks
codeverify analyze src/ --exclude-checks integer_overflow
```

### By Path

```yaml
# .codeverify.yml
exclude:
  - "tests/**"
  - "**/*.test.py"
  - "vendor/**"
  - "generated/**"
```

## Suppressing Findings

### Inline Suppression

```python
# Suppress single line
result = a / b  # codeverify: ignore division_by_zero

# Suppress with reason (recommended)
result = a / b  # codeverify: ignore division_by_zero -- b validated in caller

# Suppress multiple checks
data = items[idx].value  # codeverify: ignore array_bounds,null_safety
```

### Function-Level Suppression

```python
# codeverify: ignore-function null_safety
def internal_helper(data):
    """Only called with validated data from process()."""
    return data.value.upper()
```

### File-Level Suppression

```python
# codeverify: ignore-file security
"""
This file contains intentionally vulnerable code for security testing.
"""
```

### Configuration Suppression

```yaml
# .codeverify.yml
ignore:
  # Ignore all findings in test files
  - pattern: "tests/**"
    reason: "Test code has different standards"
    
  # Ignore specific category in migrations
  - pattern: "migrations/**"
    categories: [null_safety, type_safety]
    reason: "Auto-generated migration code"
    
  # Ignore specific finding by ID
  - finding_id: "f_abc123"
    reason: "Known false positive, tracked in ISSUE-456"
```

## Reporting False Positives

If CodeVerify reports something that isn't actually a bug:

### 1. Suppress with Reason

```python
# This is safe because X validates Y before calling
result = process(data)  # codeverify: ignore null_safety -- data validated by caller
```

### 2. Report to Improve

Click the 👎 button on findings in the dashboard or PR comment. This helps CodeVerify learn and improve.

### 3. Add to Ignore List

For persistent false positives:

```yaml
# .codeverify.yml
ignore:
  - pattern: "src/legacy/**"
    reason: "Legacy code with known patterns that trigger false positives"
```

## Finding Lifecycle

```
┌─────────┐     ┌──────────┐     ┌─────────┐     ┌────────┐
│ Created │ ──▶ │ Reviewed │ ──▶ │  Fixed  │ ──▶ │ Closed │
└─────────┘     └──────────┘     └─────────┘     └────────┘
                     │
                     ▼
               ┌───────────┐
               │ Suppressed│
               └───────────┘
```

In the dashboard, track findings through their lifecycle and see trends over time.

## Next Steps

- **[Copilot Trust Score](./trust-scores)** — Scoring AI-generated code
- **[Configuration](/docs/configuration/overview)** — Customize thresholds
- **[Dashboard](/docs/integrations/github)** — View findings in GitHub
