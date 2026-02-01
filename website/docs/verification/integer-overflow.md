---
sidebar_position: 4
---

# Integer Overflow Verification

Detect arithmetic operations that exceed integer bounds.

## The Problem

Integer overflow occurs when arithmetic results exceed the maximum (or minimum) value a type can hold:

```c
// In C/C++
int32_t a = 2147483647;  // INT_MAX
int32_t b = a + 1;       // 💥 Undefined behavior, likely -2147483648
```

While Python handles big integers automatically, TypeScript/JavaScript and lower-level languages don't:

```typescript
const a = Number.MAX_SAFE_INTEGER;  // 9007199254740991
const b = a + 2;                    // 9007199254740992 (wrong!)
```

## Integer Overflow Consequences

- **Incorrect calculations**: Financial, scientific, or business logic errors
- **Security vulnerabilities**: Buffer overflows, privilege escalation
- **Crashes**: In languages with overflow checks
- **Silent corruption**: Wrapping to unexpected values

## How CodeVerify Detects Overflow

### Constraint Generation

CodeVerify models arithmetic with bit-width constraints:

```python
def multiply(a: int, b: int) -> int:
    return a * b
```

For 64-bit integers, Z3 query:

```
∃ a, b: 
  -2^63 ≤ a ≤ 2^63 - 1 ∧
  -2^63 ≤ b ≤ 2^63 - 1 ∧
  a × b > 2^63 - 1
```

Z3 finds: `a = 2^32, b = 2^32` → Overflow possible

### Type-Aware Analysis

CodeVerify uses type information to determine bounds:

```typescript
// Treated as safe 64-bit integer
const bigInt: bigint = 9007199254740991n * 2n;

// Treated as potentially unsafe double
const num: number = 9007199254740991 * 2;  // ⚠️ Precision loss
```

## Common Patterns

### Multiplication Overflow

```python
# ❌ Unsafe
def calculate_area(width: int, height: int) -> int:
    return width * height  # ⚠️ Could overflow with large dimensions

# ✅ Safe with validation
def calculate_area_safe(width: int, height: int) -> int:
    if width <= 0 or height <= 0:
        raise ValueError("Dimensions must be positive")
    if width > 1_000_000 or height > 1_000_000:
        raise ValueError("Dimensions too large")
    return width * height
```

### Addition in Loops

```python
# ❌ Unsafe - unbounded accumulation
def sum_values(values: list[int]) -> int:
    total = 0
    for v in values:
        total += v  # ⚠️ Could overflow with many large values
    return total

# ✅ Safe with bounds checking
def sum_values_safe(values: list[int]) -> int:
    MAX_SAFE = 2**62
    total = 0
    for v in values:
        if total > MAX_SAFE - abs(v):
            raise OverflowError("Sum would overflow")
        total += v
    return total
```

### Exponentiation

```python
# ❌ Unsafe
def power(base: int, exp: int) -> int:
    return base ** exp  # ⚠️ Grows extremely fast

# ✅ Safe
def power_safe(base: int, exp: int, max_result: int = 2**63) -> int:
    if exp < 0:
        raise ValueError("Negative exponent not supported")
    if exp == 0:
        return 1
    
    result = 1
    for _ in range(exp):
        if result > max_result // abs(base):
            raise OverflowError("Result would overflow")
        result *= base
    return result
```

### Array Size Calculations

```python
# ❌ Unsafe - common in image processing
def calculate_buffer_size(width: int, height: int, channels: int) -> int:
    return width * height * channels  # ⚠️ Triple multiplication overflow risk

# ✅ Safe
def calculate_buffer_size_safe(width: int, height: int, channels: int) -> int:
    MAX_SIZE = 2**30  # 1GB max
    
    if width <= 0 or height <= 0 or channels <= 0:
        raise ValueError("All dimensions must be positive")
    
    # Check each multiplication separately
    if width > MAX_SIZE // height:
        raise OverflowError("Width * height would overflow")
    
    area = width * height
    if area > MAX_SIZE // channels:
        raise OverflowError("Total size would overflow")
    
    return area * channels
```

## TypeScript/JavaScript Specifics

### Safe Integer Range

JavaScript numbers are IEEE 754 doubles with 53-bit precision:

```typescript
const MAX_SAFE = Number.MAX_SAFE_INTEGER;  // 9007199254740991
const MIN_SAFE = Number.MIN_SAFE_INTEGER;  // -9007199254740991

// Beyond this, precision is lost
const unsafe = MAX_SAFE + 2;  // 9007199254740992 (should be 9007199254740993)
```

### Using BigInt

```typescript
// ✅ Safe - arbitrary precision
const bigA = BigInt(Number.MAX_SAFE_INTEGER);
const bigB = bigA * bigA;  // Works correctly

// But watch out for mixing
const mixed = bigA + 1;  // ❌ TypeError: Cannot mix BigInt and other types
const correct = bigA + 1n;  // ✅ Correct
```

### Bitwise Operations

Bitwise operations convert to 32-bit integers:

```typescript
const large = 2 ** 40;
const bitwise = large | 0;  // ⚠️ Truncated to 32 bits: 0
```

## Python Specifics

Python integers have arbitrary precision, but:

### NumPy Arrays

NumPy uses fixed-width integers:

```python
import numpy as np

# ⚠️ NumPy has fixed-width integers
arr = np.array([2**62], dtype=np.int64)
result = arr * 4  # Wraps around, no error!

# Check for overflow
np.seterr(over='raise')  # Raises on overflow
```

### Pandas

```python
import pandas as pd

# Similar issues with pandas
df['value'] = df['value'] * 1000000  # ⚠️ May overflow int64
```

### C Extensions

When interfacing with C code, Python integers get truncated:

```python
import ctypes

big_value = 2**100
c_value = ctypes.c_int64(big_value)  # ⚠️ Truncated silently
```

## Configuration

### Enable/Disable

```yaml
verification:
  checks:
    integer_overflow:
      enabled: true
```

### Bit Width

```yaml
verification:
  checks:
    integer_overflow:
      bit_width: 64  # 32 or 64
```

### Operations to Check

```yaml
verification:
  checks:
    integer_overflow:
      operations:
        - addition
        - subtraction
        - multiplication
        # - exponentiation  # Very expensive to verify
```

## Fixing Overflow Issues

### Option 1: Input Validation

```python
MAX_VALUE = 10_000_000

def multiply(a: int, b: int) -> int:
    if not (0 <= a <= MAX_VALUE and 0 <= b <= MAX_VALUE):
        raise ValueError(f"Values must be between 0 and {MAX_VALUE}")
    return a * b
```

### Option 2: Checked Arithmetic

```python
def checked_multiply(a: int, b: int) -> int:
    MAX = 2**63 - 1
    MIN = -(2**63)
    
    result = a * b
    if result > MAX or result < MIN:
        raise OverflowError(f"Multiplication overflow: {a} * {b}")
    return result
```

### Option 3: Use BigInt (TypeScript)

```typescript
function safeMul(a: number, b: number): bigint {
    return BigInt(a) * BigInt(b);
}
```

### Option 4: Saturating Arithmetic

```python
def saturating_add(a: int, b: int, max_val: int = 2**63 - 1) -> int:
    result = a + b
    return min(result, max_val)
```

## Examples

### Financial Calculations

```python
# ❌ Unsafe - currency in cents can overflow
def calculate_total(unit_price_cents: int, quantity: int) -> int:
    return unit_price_cents * quantity

# ✅ Safe
from decimal import Decimal

def calculate_total_safe(unit_price_cents: int, quantity: int) -> Decimal:
    # Use Decimal for financial calculations
    price = Decimal(unit_price_cents) / 100
    return price * quantity
```

### Time Calculations

```python
# ❌ Unsafe - milliseconds can overflow for long durations
def duration_ms(start: int, end: int) -> int:
    return (end - start) * 1000

# ✅ Safe
def duration_ms_safe(start: int, end: int) -> int:
    diff = end - start
    if diff > 2**53 // 1000:  # Check before multiplication
        raise OverflowError("Duration too large for milliseconds")
    return diff * 1000
```

## Next Steps

- [Division by Zero](/docs/verification/division-by-zero) — Prevent divide-by-zero
- [Verification Debugger](/docs/verification/debugger) — Debug verification results
- [Custom Rules](/docs/configuration/custom-rules) — Create arithmetic rules
