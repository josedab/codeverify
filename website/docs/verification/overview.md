---
sidebar_position: 1
---

# Verification Overview

How CodeVerify uses Z3 to mathematically prove code correctness.

## What is Formal Verification?

Formal verification uses mathematical logic to prove properties about code. Unlike testing, which checks specific inputs, verification checks **all possible inputs**.

```python
def divide(a: int, b: int) -> int:
    return a // b
```

- **Testing**: Checks `divide(10, 2)` returns `5` ✓
- **Verification**: Proves `b != 0` for all possible calls — or finds counterexamples

## The Z3 SMT Solver

CodeVerify uses [Z3](https://github.com/Z3Prover/z3), a Satisfiability Modulo Theories (SMT) solver developed by Microsoft Research.

SMT solvers determine if mathematical formulas can be satisfied. CodeVerify translates your code into formulas and asks Z3:

> "Is there any input that causes this error?"

If Z3 finds one, it's a bug. If Z3 proves none exist, the code is verified.

## How It Works

### 1. Code to Constraints

CodeVerify translates code into logical constraints:

```python
def get_element(items: list, index: int) -> int:
    return items[index]
```

Becomes:

```
∃ items, index:
  index < 0 ∨ index ≥ len(items)
```

### 2. Constraint Solving

Z3 searches for values satisfying the constraints. If found:

```
SAT: items = [], index = 0
```

This proves a bug exists.

### 3. Result Reporting

CodeVerify reports the finding with the counterexample:

```
⚠️ Array bounds violation at line 2
   Counterexample: items = [], index = 0
   When 'items' is empty, accessing index 0 fails.
```

## Verification Checks

### Null Safety

Detects null/undefined dereferences:

```python
def process(user: User | None):
    print(user.name)  # ⚠️ 'user' may be None
```

Z3 proves: `∃ user: user = None` → Bug found

### Array Bounds

Catches out-of-bounds access:

```typescript
function getFirst(arr: number[]): number {
    return arr[0];  // ⚠️ 'arr' may be empty
}
```

Z3 proves: `∃ arr: len(arr) = 0` → Bug found

### Integer Overflow

Identifies arithmetic overflow:

```python
def multiply(a: int, b: int) -> int:
    return a * b  # ⚠️ May overflow
```

Z3 proves: `∃ a, b: a × b > MAX_INT` → Bug found

### Division by Zero

Prevents divide-by-zero:

```typescript
function average(total: number, count: number): number {
    return total / count;  // ⚠️ 'count' may be zero
}
```

Z3 proves: `∃ count: count = 0` → Bug found

## Verification Strength

### Soundness

CodeVerify is **sound** — if it says code is safe, it's safe. No false negatives for the properties checked.

### Completeness

CodeVerify is **incomplete** — it may report issues that aren't actually bugs (false positives). This happens when:

- Code has implicit invariants CodeVerify can't infer
- External functions have unknown behavior
- Complex control flow exceeds analysis depth

False positive rate is typically below 10% and can be suppressed.

## Limitations

### What It Can Verify

- Local function properties
- Simple interprocedural analysis
- Statically known types
- Finite execution paths

### What It Cannot Verify

- Distributed system properties
- Concurrency (race conditions)
- External API behavior
- Infinite loops

For these, CodeVerify uses AI analysis as a complement.

## Verification vs Testing

| Aspect | Testing | Verification |
|--------|---------|--------------|
| Coverage | Selected inputs | All possible inputs |
| Guarantees | No bugs found | No bugs exist |
| Speed | Fast | Slower |
| False positives | None | Possible |
| Setup | Write tests | Configuration only |

Use both: verification catches what tests miss, tests catch what verification can't express.

## Next Steps

- [Null Safety](/docs/verification/null-safety) — Deep dive into null checking
- [Array Bounds](/docs/verification/array-bounds) — Understanding bounds verification
- [Verification Debugger](/docs/verification/debugger) — Debug verification results
