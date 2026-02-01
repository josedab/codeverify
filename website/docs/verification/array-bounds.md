---
sidebar_position: 3
---

# Array Bounds Verification

Prevent out-of-bounds array access errors.

## The Problem

Array bounds errors are a major source of crashes and security vulnerabilities:

```python
def get_item(items: list[str], index: int) -> str:
    return items[index]  # 💥 IndexError: list index out of range
```

```typescript
function getItem(items: string[], index: number): string {
    return items[index];  // 💥 undefined (or crash in strict mode)
}
```

These errors can cause:
- Application crashes
- Security vulnerabilities (buffer overflows)
- Data corruption
- Undefined behavior

## How CodeVerify Detects Bounds Issues

### Constraint Generation

CodeVerify generates Z3 constraints for array access:

```python
def get_element(arr: list[int], i: int) -> int:
    return arr[i]
```

Z3 query: "Is there an `i` where `i < 0` or `i >= len(arr)`?"

```
∃ arr, i: i < 0 ∨ i ≥ len(arr)
```

Z3 finds: `arr = [], i = 0` → Bug confirmed

### Flow Analysis

CodeVerify tracks bounds through conditionals:

```python
def safe_get(arr: list[int], i: int) -> int | None:
    if i < 0 or i >= len(arr):
        return None
    return arr[i]  # ✅ Safe - bounds verified
```

## Common Patterns

### Empty Array Access

```python
# ❌ Unsafe
def get_first(items: list[str]) -> str:
    return items[0]  # ⚠️ List may be empty

# ✅ Safe
def get_first_safe(items: list[str]) -> str | None:
    if not items:
        return None
    return items[0]
```

### Loop Index Errors

```python
# ❌ Unsafe - off-by-one error
def print_pairs(items: list[str]) -> None:
    for i in range(len(items)):
        print(f"{items[i]} - {items[i + 1]}")  # ⚠️ Last iteration overflows

# ✅ Safe
def print_pairs_safe(items: list[str]) -> None:
    for i in range(len(items) - 1):
        print(f"{items[i]} - {items[i + 1]}")
```

### Negative Indices

```python
# ❌ Potentially unsafe with dynamic index
def get_from_end(items: list[str], offset: int) -> str:
    return items[-offset]  # ⚠️ offset could be 0 or > len

# ✅ Safe
def get_from_end_safe(items: list[str], offset: int) -> str | None:
    if offset <= 0 or offset > len(items):
        return None
    return items[-offset]
```

### Slice Boundaries

```python
# Python slices are safe (no error, just empty/truncated)
items[10:20]  # ✅ Returns [] if items is shorter

# But using slice results unsafely is still dangerous
def process_chunk(items: list[str]) -> str:
    chunk = items[10:20]
    return chunk[0]  # ⚠️ chunk may be empty
```

### Multi-dimensional Arrays

```python
# ❌ Unsafe
def get_cell(matrix: list[list[int]], row: int, col: int) -> int:
    return matrix[row][col]  # ⚠️ Both indices could be out of bounds

# ✅ Safe
def get_cell_safe(matrix: list[list[int]], row: int, col: int) -> int | None:
    if row < 0 or row >= len(matrix):
        return None
    if col < 0 or col >= len(matrix[row]):
        return None
    return matrix[row][col]
```

## TypeScript/JavaScript Specifics

### Array vs Tuple

```typescript
// Array - length unknown
const items: string[] = getData();
items[0];  // ⚠️ May be undefined

// Tuple - length known
const pair: [string, string] = ["a", "b"];
pair[0];  // ✅ Safe - guaranteed to have 2 elements
```

### TypeScript Strictness

Enable `noUncheckedIndexedAccess` for better safety:

```json
{
  "compilerOptions": {
    "noUncheckedIndexedAccess": true
  }
}
```

With this, `arr[i]` returns `T | undefined` instead of `T`.

### Array Methods vs Direct Access

```typescript
// ❌ Direct access - unsafe
function getFirst(arr: number[]): number {
    return arr[0];  // ⚠️ May be undefined
}

// ✅ Array methods - explicitly handle undefined
function getFirstSafe(arr: number[]): number | undefined {
    return arr.at(0);  // Returns undefined if empty
}
```

## Python Specifics

### List Comprehension Safety

```python
# List comprehensions with conditions are generally safe
[items[i] for i in range(len(items))]  # ✅ Safe

# But watch out for dependent indices
[items[i] + items[i+1] for i in range(len(items))]  # ⚠️ Unsafe
```

### Dictionary as Safe Alternative

```python
# Instead of sparse array access
def get_value(data: list[int], index: int) -> int | None:
    if index < 0 or index >= len(data):
        return None
    return data[index]

# Consider a dict
def get_value_dict(data: dict[int, int], index: int) -> int | None:
    return data.get(index)  # ✅ Always safe
```

## Configuration

### Enable/Disable

```yaml
verification:
  checks:
    array_bounds:
      enabled: true
```

### Dynamic Bounds

For arrays with unknown size at analysis time:

```yaml
verification:
  checks:
    array_bounds:
      assume_dynamic_bounds: true  # Assume could be empty
```

### Negative Index Handling

```yaml
verification:
  checks:
    array_bounds:
      check_negative: true  # Check negative indices
```

## Fixing Bounds Issues

### Option 1: Guard Check

```python
def get_item(arr: list[int], i: int) -> int:
    if i < 0 or i >= len(arr):
        raise IndexError(f"Index {i} out of bounds for array of length {len(arr)}")
    return arr[i]
```

### Option 2: Return Optional

```python
def get_item(arr: list[int], i: int) -> int | None:
    if i < 0 or i >= len(arr):
        return None
    return arr[i]
```

### Option 3: Default Value

```python
def get_item(arr: list[int], i: int, default: int = 0) -> int:
    if i < 0 or i >= len(arr):
        return default
    return arr[i]
```

### Option 4: Clamp Index

```python
def get_item_clamped(arr: list[int], i: int) -> int:
    if not arr:
        raise ValueError("Array cannot be empty")
    clamped_i = max(0, min(i, len(arr) - 1))
    return arr[clamped_i]
```

## Suppressing False Positives

When external constraints guarantee bounds:

```python
# Config file always has at least one entry (external invariant)
config = load_config()  # Guaranteed non-empty by schema

# Option 1: Assert
assert len(config) > 0, "Config must have entries"
first_entry = config[0]  # ✅ Safe after assert

# Option 2: Suppress
first_entry = config[0]  # codeverify-disable-line array_bounds
```

## Examples

### Pagination

```python
def get_page(items: list[Item], page: int, page_size: int) -> list[Item]:
    if page < 1 or page_size < 1:
        raise ValueError("Page and page_size must be positive")
    
    start = (page - 1) * page_size
    end = start + page_size
    
    # Slicing is safe - returns empty if out of bounds
    return items[start:end]
    
def get_page_item(items: list[Item], page: int, index: int) -> Item | None:
    page_items = get_page(items, page, 10)
    
    # But accessing the result needs a check
    if index < 0 or index >= len(page_items):
        return None
    return page_items[index]
```

### Binary Search

```python
def binary_search(arr: list[int], target: int) -> int:
    if not arr:
        return -1
        
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        # mid is always valid: left <= mid <= right < len(arr)
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

## Next Steps

- [Integer Overflow](/docs/verification/integer-overflow) — Prevent arithmetic overflow
- [Division by Zero](/docs/verification/division-by-zero) — Prevent divide-by-zero
- [Verification Debugger](/docs/verification/debugger) — Debug verification results
