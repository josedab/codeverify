---
slug: z3-smt-solver-explained
title: "Z3 SMT Solver: The Engine Behind Formal Verification"
authors: [codeverify]
tags: [z3, smt, formal-verification, technical, deep-dive]
---

Ever wondered how CodeVerify can mathematically *prove* that your code has a bug? The answer is Z3, a powerful SMT solver from Microsoft Research. This post explains how Z3 works and why it's perfect for code verification.

<!-- truncate -->

## What is an SMT Solver?

SMT stands for **Satisfiability Modulo Theories**. In plain English, an SMT solver answers questions like:

> "Given these constraints, is there any combination of values that makes them all true?"

For example:
- Constraint 1: `x > 0`
- Constraint 2: `x < 10`
- Constraint 3: `x * 2 == 14`

Z3 can determine that `x = 7` satisfies all constraints. Or if you asked for `x * 2 == 15`, it would prove no solution exists.

## Why SMT Solvers for Code?

Consider this function:

```python
def divide(a: int, b: int) -> float:
    return a / b
```

We want to prove: **"For all possible values of `a` and `b`, division never fails."**

Division fails when `b = 0`. So we're really asking:

> "Is there any value of `b` where `b == 0`?"

If Z3 finds such a value, we have a bug. If Z3 proves no such value exists (with the given constraints), the code is safe.

## How CodeVerify Uses Z3

### Step 1: Parse Code to AST

```python
def get_element(items: list[int], idx: int) -> int:
    return items[idx]
```

The AST tells us:
- `items` is a list with some length
- `idx` is an integer
- We access `items[idx]`

### Step 2: Generate Constraints

We translate to Z3 constraints:

```python
from z3 import *

# Variables
items_len = Int('items_len')
idx = Int('idx')

# Constraints from type system
s = Solver()
s.add(items_len >= 0)  # Lists have non-negative length

# The safety property we want to DISPROVE
# If we can find idx outside bounds, there's a bug
s.add(Or(idx < 0, idx >= items_len))
```

### Step 3: Solve

```python
if s.check() == sat:
    model = s.model()
    print(f"Bug found! Counterexample: items_len={model[items_len]}, idx={model[idx]}")
else:
    print("Code is safe for all inputs")
```

Result:
```
Bug found! Counterexample: items_len=0, idx=0
```

Z3 found that when the list is empty (`items_len=0`) and we access index 0, we're out of bounds.

## Real-World Example: Null Safety

```python
def get_user_email(user_id: str) -> str:
    user = find_user(user_id)
    return user.email
```

**Question:** Can `user` ever be `None` when we access `.email`?

```python
from z3 import *

# Model the function return as potentially None
user_is_none = Bool('user_is_none')

# We know find_user can return None
# (inferred from return type or documentation)
s = Solver()

# Try to find a case where user is None AND we access .email
s.add(user_is_none == True)

if s.check() == sat:
    print("Bug: user could be None when accessing .email")
```

## Z3 Theories We Use

Z3 supports multiple "theories" for different data types:

### Integer Arithmetic

```python
from z3 import *

x = Int('x')
y = Int('y')

s = Solver()
s.add(x + y == 10)
s.add(x - y == 2)

s.check()  # sat
s.model()  # x = 6, y = 4
```

### Bitvectors (for overflow detection)

```python
from z3 import *

# 32-bit integers
a = BitVec('a', 32)
b = BitVec('b', 32)

s = Solver()
# Check if a + b can overflow
s.add(a > 0)
s.add(b > 0)
s.add(a + b < a)  # Overflow condition

s.check()  # sat - overflow is possible!
s.model()  # a = 2147483647, b = 1
```

### Arrays

```python
from z3 import *

# Model array access
arr = Array('arr', IntSort(), IntSort())
idx = Int('idx')
length = Int('length')

s = Solver()
s.add(length > 0)
s.add(Or(idx < 0, idx >= length))  # Out of bounds?

s.check()  # sat
# Counterexample: length=1, idx=-1 or idx=1
```

## Performance Considerations

### Timeout Handling

Some constraints are too complex to solve quickly. We set timeouts:

```python
s = Solver()
s.set("timeout", 30000)  # 30 seconds

result = s.check()
if result == unknown:
    print("Couldn't determine in time limit")
```

### Constraint Complexity

The number of constraints affects solve time exponentially:

| Constraints | Typical Time |
|-------------|--------------|
| < 50 | < 100ms |
| 50-200 | 100ms - 1s |
| 200-500 | 1s - 10s |
| 500+ | May timeout |

CodeVerify uses heuristics to simplify constraints and analyzes functions independently.

### Incremental Solving

For efficiency, we reuse solver state:

```python
s = Solver()
s.add(base_constraints)

s.push()  # Save state
s.add(check_null_safety)
s.check()
s.pop()   # Restore state

s.push()
s.add(check_array_bounds)
s.check()
s.pop()
```

## Limitations

Z3 is powerful but not magic:

### 1. Undecidability

Some properties are mathematically undecidable (halting problem). Z3 may return `unknown`.

### 2. Loop Invariants

Reasoning about loops requires inferring loop invariants, which is hard:

```python
def sum_to_n(n: int) -> int:
    total = 0
    for i in range(n):
        total += i
    return total
```

We use bounded model checking (unroll loops a fixed number of times).

### 3. External Functions

Z3 can't model external API calls:

```python
response = requests.get(url)  # What constraints does this have?
```

We use AI analysis to infer likely constraints from documentation and usage patterns.

## Why Z3 vs Other Solvers?

| Solver | Strengths | Limitations |
|--------|-----------|-------------|
| **Z3** | Fast, well-maintained, great APIs | Memory-heavy |
| CVC5 | Strong on bitvectors | Slower on integers |
| Yices | Very fast | Fewer features |
| dReal | Real arithmetic | Overkill for code |

We chose Z3 for its balance of speed, features, and excellent Python bindings.

## Try It Yourself

Install Z3:

```bash
pip install z3-solver
```

Basic example:

```python
from z3 import *

# "Is there an x where x > 5 and x < 3?"
x = Int('x')
s = Solver()
s.add(x > 5)
s.add(x < 3)

print(s.check())  # unsat - no such x exists
```

## Conclusion

Z3 transforms code verification from "works on these tests" to "provably correct for all inputs." When combined with AI analysis for semantic understanding, you get the best of both worlds: mathematical rigor and practical usability.

That's the power behind CodeVerify.

---

*Want to dive deeper? Check out our [Verification Debugger](/docs/verification/debugger) to visualize Z3 in action, or read the [Z3 documentation](https://z3prover.github.io/api/html/z3.html).*
