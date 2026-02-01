---
sidebar_position: 3
---

# Automated Test Generation

Generate comprehensive tests from counterexamples found during verification.

## Overview

When CodeVerify finds a bug, it has a concrete counterexample — specific inputs that trigger the issue. This counterexample can become a test case:

```
Finding: Division by zero at divide(a, b)
Counterexample: a = 42, b = 0

Generated test:
def test_divide_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(42, 0)
```

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  Code                                                        │
│  def divide(a, b): return a / b                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Z3 Verification                                            │
│  Query: ∃ b: b = 0                                          │
│  Result: SAT (b = 0)                                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Test Generation                                            │
│  - Extract counterexample values                            │
│  - Generate test function                                   │
│  - Add to test file                                         │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Enable Test Generation

```yaml
# .codeverify.yml
test_generation:
  enabled: true
  
  # Test framework
  framework: pytest  # pytest, unittest, jest, mocha
  
  # Output directory
  output_dir: "tests/generated"
  
  # Generate tests for which severities
  severities:
    - critical
    - high
```

### Per-Language Settings

```yaml
test_generation:
  python:
    framework: pytest
    output_pattern: "tests/generated/test_{module}.py"
    
  typescript:
    framework: jest
    output_pattern: "__tests__/generated/{module}.test.ts"
```

## Usage

### CLI

```bash
# Generate tests for all findings
codeverify generate-tests

# Generate for specific file
codeverify generate-tests src/calculator.py

# Preview without writing
codeverify generate-tests --dry-run

# Specify output
codeverify generate-tests --output tests/from-verification/
```

### Interactive Mode

```bash
codeverify generate-tests --interactive

Finding 1: Division by zero at calculator.py:15
  Counterexample: a=42, b=0
  
Generate test? [y/n/edit]: y

Generated: tests/generated/test_calculator.py

def test_divide_handles_zero_divisor():
    """Regression test for division by zero bug found by CodeVerify."""
    with pytest.raises(ZeroDivisionError):
        divide(42, 0)

Accept? [y/n/edit]: edit

# Opens editor for customization...
```

## Generated Test Examples

### Python (pytest)

```python
# tests/generated/test_calculator.py
"""
Auto-generated tests from CodeVerify verification.
Generated: 2024-01-15T10:30:00Z
"""
import pytest
from calculator import divide, get_element, calculate_total


class TestDivideVerification:
    """Tests generated from verification of divide()."""
    
    def test_divide_zero_divisor(self):
        """
        Counterexample from formal verification.
        Finding: Division by zero possible when b=0
        """
        with pytest.raises(ZeroDivisionError):
            divide(42, 0)
    
    def test_divide_negative_divisor(self):
        """Boundary case: negative divisor."""
        result = divide(100, -5)
        assert result == -20


class TestGetElementVerification:
    """Tests generated from verification of get_element()."""
    
    def test_get_element_empty_list(self):
        """
        Counterexample: empty list access
        Finding: Array bounds violation when list is empty
        """
        with pytest.raises(IndexError):
            get_element([], 0)
    
    def test_get_element_negative_index(self):
        """
        Counterexample: negative index without bounds
        Finding: Negative index may exceed bounds
        """
        with pytest.raises(IndexError):
            get_element([1, 2, 3], -10)
```

### TypeScript (Jest)

```typescript
// __tests__/generated/calculator.test.ts
/**
 * Auto-generated tests from CodeVerify verification.
 * Generated: 2024-01-15T10:30:00Z
 */
import { divide, getElement, calculateTotal } from '../src/calculator';

describe('divide - verification tests', () => {
  it('should handle zero divisor', () => {
    /**
     * Counterexample from formal verification.
     * Finding: Division by zero returns Infinity
     */
    const result = divide(42, 0);
    expect(result).toBe(Infinity);
    // Note: JS doesn't throw, but Infinity is likely unintended
  });
});

describe('getElement - verification tests', () => {
  it('should handle empty array', () => {
    /**
     * Counterexample: empty array access
     * Finding: Returns undefined for empty array
     */
    const result = getElement([], 0);
    expect(result).toBeUndefined();
  });
  
  it('should handle out of bounds index', () => {
    /**
     * Counterexample: index exceeds length
     * Finding: Returns undefined for index >= length
     */
    const result = getElement([1, 2, 3], 10);
    expect(result).toBeUndefined();
  });
});
```

## Test Types

### Regression Tests

Capture the bug to prevent reintroduction:

```python
def test_issue_CVE_123_division_by_zero():
    """
    Regression test for CVE-123.
    Original finding: Division by zero in calculate_rate()
    Fixed in: commit abc123
    """
    with pytest.raises(ValueError, match="Rate cannot be zero"):
        calculate_rate(100, 0)
```

### Boundary Tests

Test edge cases from counterexamples:

```python
@pytest.mark.parametrize("index,expected", [
    (0, "first"),           # First element
    (-1, "last"),           # Last element (Python)
    (999, IndexError),      # Out of bounds
])
def test_get_element_boundaries(index, expected):
    items = ["first", "middle", "last"]
    if expected is IndexError:
        with pytest.raises(IndexError):
            get_element(items, index)
    else:
        assert get_element(items, index) == expected
```

### Property Tests

Generate hypothesis/fast-check style tests:

```python
from hypothesis import given, strategies as st

@given(st.integers(), st.integers().filter(lambda x: x != 0))
def test_divide_never_crashes_with_nonzero_divisor(a, b):
    """Property: divide never crashes when b != 0."""
    result = divide(a, b)
    assert isinstance(result, (int, float))
```

## Customization

### Test Templates

Create custom templates:

```yaml
test_generation:
  templates:
    pytest: ".codeverify/templates/pytest.jinja2"
```

Template example:

```jinja2
# tests/generated/test_{{ module }}.py
"""Generated by CodeVerify on {{ timestamp }}."""
import pytest
from {{ module }} import {{ functions | join(', ') }}

{% for finding in findings %}
class Test{{ finding.function | capitalize }}:
    """Verification tests for {{ finding.function }}()."""
    
    def test_{{ finding.id }}(self):
        """
        {{ finding.description }}
        Severity: {{ finding.severity }}
        """
        {% if finding.should_raise %}
        with pytest.raises({{ finding.exception }}):
            {{ finding.function }}({{ finding.args | join(', ') }})
        {% else %}
        result = {{ finding.function }}({{ finding.args | join(', ') }})
        {{ finding.assertion }}
        {% endif %}
{% endfor %}
```

### Hooks

Run custom logic before/after generation:

```yaml
test_generation:
  hooks:
    before_generate: "scripts/prepare-test-env.sh"
    after_generate: "scripts/format-tests.sh"
```

## Integration with CI

### Generate and Run

```yaml
# .github/workflows/verification-tests.yml
name: Verification Tests

on:
  push:
    branches: [main]

jobs:
  generate-and-run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run CodeVerify
        run: codeverify analyze src/
      
      - name: Generate Tests
        run: codeverify generate-tests --output tests/verification/
      
      - name: Run Generated Tests
        run: pytest tests/verification/ -v
        
      - name: Commit if new tests
        run: |
          git add tests/verification/
          git diff --cached --quiet || git commit -m "Add verification tests"
```

### Test Maintenance

```yaml
test_generation:
  # Don't regenerate existing tests
  skip_existing: true
  
  # Update tests when counterexamples change
  update_on_change: true
  
  # Mark outdated tests
  mark_outdated: true
```

## Best Practices

1. **Review generated tests** — They capture bugs but may need refinement
2. **Keep them separate** — Use `tests/generated/` to distinguish from hand-written
3. **Run in CI** — Ensure bugs stay fixed
4. **Commit meaningful ones** — Not all counterexamples need permanent tests
5. **Update, don't duplicate** — Regenerate rather than accumulate

## Next Steps

- [Verification Debugger](/docs/verification/debugger) — Understand counterexamples
- [CI/CD Integration](/docs/integrations/ci-cd) — Run tests in pipelines
- [Configuration](/docs/configuration/overview) — Full test generation options
