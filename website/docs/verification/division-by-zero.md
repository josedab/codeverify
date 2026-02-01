---
sidebar_position: 5
---

# Division by Zero Verification

Prevent divide-by-zero errors before they crash your application.

## The Problem

Division by zero causes immediate crashes or undefined behavior:

```python
def average(total: int, count: int) -> float:
    return total / count  # 💥 ZeroDivisionError if count is 0
```

```typescript
function average(total: number, count: number): number {
    return total / count;  // Returns Infinity, not an error!
}
```

## How CodeVerify Detects Division by Zero

### Constraint Generation

For every division operation, CodeVerify asks Z3:

```python
def calculate_rate(amount: int, divisor: int) -> float:
    return amount / divisor
```

Z3 query:
```
∃ divisor: divisor = 0
```

Z3 trivially proves: `divisor = 0` is satisfiable → Bug found

### Flow-Sensitive Analysis

CodeVerify tracks guards:

```python
def safe_divide(a: int, b: int) -> float:
    if b == 0:
        return 0.0
    return a / b  # ✅ Safe - guard verified
```

## Common Patterns

### Missing Guard

```python
# ❌ Unsafe
def percentage(part: int, whole: int) -> float:
    return (part / whole) * 100

# ✅ Safe
def percentage_safe(part: int, whole: int) -> float:
    if whole == 0:
        return 0.0
    return (part / whole) * 100
```

### List Length Division

```python
# ❌ Unsafe
def average_score(scores: list[float]) -> float:
    return sum(scores) / len(scores)  # ⚠️ Empty list = div by zero

# ✅ Safe
def average_score_safe(scores: list[float]) -> float | None:
    if not scores:
        return None
    return sum(scores) / len(scores)
```

### Modulo Operations

Modulo by zero has the same problem:

```python
# ❌ Unsafe
def is_even(n: int, divisor: int) -> bool:
    return n % divisor == 0  # ⚠️ ZeroDivisionError if divisor is 0

# ✅ Safe
def is_divisible(n: int, divisor: int) -> bool:
    if divisor == 0:
        raise ValueError("Divisor cannot be zero")
    return n % divisor == 0
```

### Floor Division

```python
# ❌ Unsafe
def pages_needed(items: int, per_page: int) -> int:
    return (items + per_page - 1) // per_page  # ⚠️ per_page could be 0

# ✅ Safe
def pages_needed_safe(items: int, per_page: int) -> int:
    if per_page <= 0:
        raise ValueError("Items per page must be positive")
    return (items + per_page - 1) // per_page
```

### Calculated Divisors

```python
# ❌ Unsafe - divisor comes from calculation
def normalize(values: list[float]) -> list[float]:
    max_val = max(values)
    min_val = min(values)
    range_val = max_val - min_val  # Could be 0 if all values equal
    return [(v - min_val) / range_val for v in values]  # ⚠️

# ✅ Safe
def normalize_safe(values: list[float]) -> list[float]:
    if not values:
        return []
    max_val = max(values)
    min_val = min(values)
    range_val = max_val - min_val
    if range_val == 0:
        return [0.5] * len(values)  # All values are the same
    return [(v - min_val) / range_val for v in values]
```

## TypeScript/JavaScript Behavior

JavaScript doesn't throw on division by zero:

```typescript
1 / 0          // Infinity
-1 / 0         // -Infinity
0 / 0          // NaN

1 % 0          // NaN
```

This makes bugs silent and harder to detect:

```typescript
// ❌ Dangerous - silent NaN propagation
function average(nums: number[]): number {
    return nums.reduce((a, b) => a + b, 0) / nums.length;
}

const result = average([]);  // NaN - silently propagates
```

CodeVerify flags these as potential issues because NaN propagation often indicates a bug.

### Guarding in TypeScript

```typescript
// ✅ Explicit handling
function average(nums: number[]): number | null {
    if (nums.length === 0) {
        return null;
    }
    return nums.reduce((a, b) => a + b, 0) / nums.length;
}
```

## Python Behavior

Python raises `ZeroDivisionError`:

```python
1 / 0   # ZeroDivisionError: division by zero
1 // 0  # ZeroDivisionError: integer division or modulo by zero
1 % 0   # ZeroDivisionError: integer division or modulo by zero
```

### Float Division

```python
import math

1.0 / 0.0  # ZeroDivisionError (unlike JavaScript)
float('inf') / float('inf')  # nan
```

## Configuration

### Enable/Disable

```yaml
verification:
  checks:
    division_by_zero:
      enabled: true
```

### Check Modulo

```yaml
verification:
  checks:
    division_by_zero:
      check_modulo: true  # Also check % operations
```

### Check Floor Division

```yaml
verification:
  checks:
    division_by_zero:
      check_floor_division: true  # Also check // operations
```

## Fixing Division by Zero

### Option 1: Guard Check

```python
def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```

### Option 2: Return Optional

```python
def divide(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return a / b
```

### Option 3: Default Value

```python
def divide(a: float, b: float, default: float = 0.0) -> float:
    if b == 0:
        return default
    return a / b
```

### Option 4: Require Positive

When the divisor should never be zero:

```python
def calculate_rate(total: float, count: int) -> float:
    if count <= 0:
        raise ValueError("Count must be positive")
    return total / count
```

## Examples

### Statistics

```python
def calculate_statistics(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "variance": None}
    
    n = len(values)
    mean = sum(values) / n
    
    if n < 2:
        return {"mean": mean, "variance": None}
    
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    
    return {"mean": mean, "variance": variance}
```

### Financial Calculations

```python
from decimal import Decimal

def calculate_exchange(
    amount: Decimal,
    from_rate: Decimal,
    to_rate: Decimal
) -> Decimal:
    if from_rate == 0 or to_rate == 0:
        raise ValueError("Exchange rates must be non-zero")
    
    # Convert to base currency, then to target
    base_amount = amount / from_rate
    return base_amount * to_rate
```

### Image Processing

```python
def resize_aspect_ratio(
    width: int,
    height: int,
    max_width: int,
    max_height: int
) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    
    width_ratio = max_width / width
    height_ratio = max_height / height
    ratio = min(width_ratio, height_ratio)
    
    return (int(width * ratio), int(height * ratio))
```

## Next Steps

- [Verification Debugger](/docs/verification/debugger) — Debug verification results
- [Null Safety](/docs/verification/null-safety) — Prevent null dereferences
- [Integer Overflow](/docs/verification/integer-overflow) — Prevent arithmetic overflow
