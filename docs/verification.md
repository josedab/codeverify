# Formal Verification with Z3

CodeVerify uses the [Z3 SMT Solver](https://github.com/Z3Prover/z3) to provide mathematical proofs of code correctness. Unlike pattern-based static analysis, formal verification can prove the absence of certain bugs.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Verification Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Source Code  ──▶  AST Parser  ──▶  Constraint Generator       │
│                                              │                   │
│                                              ▼                   │
│                                       ┌─────────────┐            │
│                                       │  Z3 Solver  │            │
│                                       └──────┬──────┘            │
│                                              │                   │
│                          ┌───────────────────┼───────────────┐   │
│                          ▼                   ▼               ▼   │
│                       [SAT]             [UNSAT]          [UNKNOWN]│
│                    Bug Found         Verified Safe      Timeout  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Supported Checks

### Null Safety

Detects potential null/undefined dereferences.

```python
# Example: Potential null dereference
def get_username(user: User | None) -> str:
    return user.name  # ⚠️ user could be None
```

**Z3 Constraint:**
```python
# Simplified representation
user = Symbol('user')
constraint = And(
    user == None,  # Assume user could be None
    Access(user, 'name')  # Attempt to access .name
)
# If SAT → bug exists
```

### Array Bounds

Catches out-of-bounds array access.

```python
# Example: Array bounds violation
def get_item(items: list, index: int) -> Any:
    return items[index]  # ⚠️ index could be out of bounds
```

**Z3 Constraint:**
```python
index = Int('index')
length = Int('length')
constraint = Or(index < 0, index >= length)
# If SAT → potential out-of-bounds access
```

### Integer Overflow

Identifies arithmetic overflow risks.

```python
# Example: Integer overflow
def multiply(a: int, b: int) -> int:
    return a * b  # ⚠️ Could overflow on large values
```

**Z3 Constraint:**
```python
a, b = Ints('a b')
result = a * b
MAX_INT = 2**63 - 1
MIN_INT = -(2**63)
constraint = Or(result > MAX_INT, result < MIN_INT)
# If SAT → overflow possible
```

### Division by Zero

Prevents divide-by-zero errors.

```python
# Example: Division by zero
def divide(a: int, b: int) -> float:
    return a / b  # ⚠️ b could be zero
```

**Z3 Constraint:**
```python
b = Int('b')
constraint = b == 0
# If SAT → division by zero possible
```

## How It Works

### 1. Code Parsing

CodeVerify parses source code into an Abstract Syntax Tree (AST):

```python
from codeverify_verifier.parsers import PythonParser

parser = PythonParser()
ast = parser.parse(source_code)
```

### 2. Path Extraction

The verifier extracts execution paths and conditions:

```python
from codeverify_verifier.conditions import extract_conditions

conditions = extract_conditions(ast, function_name="divide")
# Returns: [PathCondition(variables=['b'], constraint='b != 0')]
```

### 3. Constraint Generation

Conditions are converted to Z3 constraints:

```python
from z3 import Int, Solver

b = Int('b')
solver = Solver()
solver.add(b == 0)  # Check if b can be zero
```

### 4. Solving

Z3 attempts to satisfy the constraints:

```python
result = solver.check()
if result == sat:
    # Found a counterexample - bug exists
    model = solver.model()
    print(f"Bug with b = {model[b]}")
elif result == unsat:
    # No counterexample - code is safe
    print("Verified safe")
else:
    # Timeout or unknown
    print("Could not verify")
```

## API Reference

### Z3Verifier

The main verification interface:

```python
from codeverify_verifier import Z3Verifier

verifier = Z3Verifier(timeout_seconds=30)

# Verify a single function
result = verifier.verify_function(
    code="def divide(a, b): return a / b",
    function_name="divide",
    checks=["division_by_zero", "null_safety"],
)

# Result structure
print(result.status)        # "vulnerable" | "safe" | "unknown"
print(result.findings)      # List of findings
print(result.counterexample)  # Input that triggers bug
```

### VerificationResult

```python
@dataclass
class VerificationResult:
    status: Literal["safe", "vulnerable", "unknown"]
    check_type: str
    findings: list[Finding]
    counterexample: dict | None
    proof_time_ms: int
    z3_stats: dict
```

### Finding

```python
@dataclass
class Finding:
    check_type: str
    severity: Literal["critical", "high", "medium", "low"]
    title: str
    description: str
    line_start: int
    line_end: int
    confidence: float
    counterexample: dict | None
    fix_suggestion: str | None
```

## Configuration

### In `.codeverify.yml`

```yaml
verification:
  enabled: true
  timeout: 30  # seconds per check
  checks:
    - null_safety
    - array_bounds
    - integer_overflow
    - division_by_zero
  
  # Advanced options
  advanced:
    # Max path depth for exploration
    max_path_depth: 10
    
    # Use incremental solving
    incremental: true
    
    # Proof logging for debugging
    log_proofs: false
```

### Check-Specific Configuration

```yaml
verification:
  checks:
    null_safety:
      enabled: true
      # Consider Optional types as potentially null
      strict_optionals: true
      
    array_bounds:
      enabled: true
      # Assume unknown sizes for dynamic arrays
      assume_dynamic_bounds: true
      
    integer_overflow:
      enabled: true
      # Bit width for integer constraints
      bit_width: 64
      
    division_by_zero:
      enabled: true
```

## Inline Annotations

Suppress specific checks with inline comments:

```python
# Suppress a single line
result = a / b  # codeverify: ignore division_by_zero

# Suppress with reason
items[index]  # codeverify: ignore array_bounds -- index validated above

# Suppress for entire function
# codeverify: ignore-function null_safety
def process(user):
    return user.name
```

## Debugging Verification

### Verification Debugger

Use the interactive debugger to understand verification:

```bash
codeverify debug src/math.py --function divide
```

The debugger shows:
- Extracted path conditions
- Z3 constraints generated
- Solving steps
- Counterexample construction

### Proof Logging

Enable proof logging for detailed output:

```yaml
verification:
  advanced:
    log_proofs: true
```

Logs are written to `.codeverify/proofs/`.

### VS Code Integration

The VS Code extension provides inline visualization of verification:
- Green: Verified safe
- Yellow: Unknown/timeout
- Red: Vulnerability found with counterexample

## Limitations

### What Z3 Can Verify

✅ Arithmetic properties (overflow, division by zero)
✅ Array bounds with known sizes
✅ Null safety with explicit types
✅ Simple loop invariants
✅ Linear constraints

### What Z3 Cannot Verify

❌ Complex string operations
❌ Network/IO behavior
❌ Timing/concurrency issues
❌ Unbounded loops
❌ Dynamic code execution (eval, exec)

### Timeouts

Some code is too complex for verification within the timeout. In these cases:
- Result is "unknown"
- AI analysis provides fallback assessment
- Consider splitting complex functions

## Performance

### Typical Verification Times

| Check | Simple Function | Complex Function |
|-------|-----------------|------------------|
| Null Safety | 50-100ms | 500ms-2s |
| Array Bounds | 100-200ms | 1-5s |
| Integer Overflow | 200-500ms | 2-10s |
| Division by Zero | 20-50ms | 100-500ms |

### Optimization Tips

1. **Break up large functions** - Smaller functions verify faster
2. **Add type hints** - Helps constraint generation
3. **Use assertions** - Provides additional constraints
4. **Tune timeout** - Balance thoroughness vs speed

## Further Reading

- [ADR-0002: Formal Verification with Z3](adr/0002-formal-verification-with-z3.md)
- [Z3 Tutorial](https://ericpony.github.io/z3py-tutorial/guide-examples.htm)
- [SMT-LIB Standard](https://smtlib.cs.uiowa.edu/)
