# @codeverify/verifier

Z3 SMT solver integration for formal verification of code.

## Installation

```bash
pip install codeverify-verifier

# Or from source
pip install -e packages/verifier
```

**Requirements:**
- Python 3.11+
- Z3 solver (installed automatically via `z3-solver` package)

## Overview

This package provides formal verification capabilities using the Z3 SMT solver:

- **Null Safety**: Detect potential null/undefined dereferences
- **Array Bounds**: Catch out-of-bounds array access
- **Integer Overflow**: Identify arithmetic overflow risks
- **Division by Zero**: Prevent divide-by-zero errors

## Quick Start

```python
from codeverify_verifier import Z3Verifier

verifier = Z3Verifier()

code = """
def divide(a: int, b: int) -> float:
    return a / b
"""

result = verifier.verify_function(
    code=code,
    function_name="divide",
    checks=["division_by_zero"],
)

print(result.status)  # "vulnerable"
print(result.findings[0].title)  # "Division by zero possible"
print(result.counterexample)  # {"b": 0}
```

## Modules

### `z3_verifier`

Main verification engine:

```python
from codeverify_verifier.z3_verifier import Z3Verifier, VerificationConfig

config = VerificationConfig(
    timeout_seconds=30,
    checks=["null_safety", "array_bounds", "integer_overflow", "division_by_zero"],
    max_path_depth=10,
)

verifier = Z3Verifier(config)

# Verify entire file
results = verifier.verify_file("src/math.py")

# Verify specific function
result = verifier.verify_function(code, "my_function")

# Verify code snippet
result = verifier.verify_snippet(code_snippet, language="python")
```

### `parsers`

Language-specific AST parsers:

```python
from codeverify_verifier.parsers import (
    PythonParser,
    TypeScriptParser,
    get_parser,
)

# Auto-detect parser
parser = get_parser("python")
ast = parser.parse(source_code)

# Extract function info
functions = parser.extract_functions(ast)
for func in functions:
    print(f"{func.name}: {func.parameters} -> {func.return_type}")
```

### `conditions`

Path condition extraction:

```python
from codeverify_verifier.conditions import (
    extract_conditions,
    PathCondition,
    ConditionType,
)

conditions = extract_conditions(ast, function_name="process")
for cond in conditions:
    print(f"{cond.type}: {cond.expression}")
    print(f"  Variables: {cond.variables}")
    print(f"  Line: {cond.line}")
```

### `debugger`

Interactive verification debugger:

```python
from codeverify_verifier.debugger import VerificationDebugger

debugger = VerificationDebugger()

# Start debug session
session = debugger.create_session(code, "my_function")

# Step through verification
while session.has_next():
    step = session.step()
    print(f"Step: {step.description}")
    print(f"Constraints: {step.constraints}")
    print(f"Result: {step.result}")

# Get full trace
trace = session.get_trace()
```

## Verification Results

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

### Status Values

| Status | Meaning |
|--------|---------|
| `safe` | Z3 proved no bug exists (UNSAT) |
| `vulnerable` | Z3 found a counterexample (SAT) |
| `unknown` | Timeout or undecidable (UNKNOWN) |

## Configuration

```python
@dataclass
class VerificationConfig:
    # Timeout per check in seconds
    timeout_seconds: int = 30
    
    # Which checks to run
    checks: list[str] = field(default_factory=lambda: [
        "null_safety",
        "array_bounds", 
        "integer_overflow",
        "division_by_zero",
    ])
    
    # Max path depth for exploration
    max_path_depth: int = 10
    
    # Bit width for integer constraints
    bit_width: int = 64
    
    # Use incremental solving
    incremental: bool = True
    
    # Log Z3 proofs for debugging
    log_proofs: bool = False
```

## Supported Languages

| Language | Parser | Status |
|----------|--------|--------|
| Python | `PythonParser` | ✅ Full support |
| TypeScript | `TypeScriptParser` | ✅ Full support |
| JavaScript | `TypeScriptParser` | ✅ Full support |
| Go | `GoParser` | 🔜 Coming soon |
| Java | `JavaParser` | 🔜 Coming soon |

## Development

```bash
# Install with dev dependencies
pip install -e "packages/verifier[dev]"

# Run tests
pytest packages/verifier/tests -v

# Run with coverage
pytest packages/verifier/tests --cov=codeverify_verifier

# Type checking
mypy packages/verifier/src
```

## Further Reading

- [Verification Deep Dive](../../docs/verification.md)
- [ADR-0002: Formal Verification with Z3](../../docs/adr/0002-formal-verification-with-z3.md)
- [Z3 Python Tutorial](https://ericpony.github.io/z3py-tutorial/guide-examples.htm)
