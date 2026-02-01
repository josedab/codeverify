---
sidebar_position: 2
---

# Verification Types

CodeVerify uses multiple verification methods, each with different strengths and use cases.

## Overview

| Method | Engine | Confidence | Speed | Best For |
|--------|--------|------------|-------|----------|
| **Formal** | Z3 SMT | 95-99% | Medium | Provable properties |
| **AI Semantic** | GPT-4/Claude | 80-95% | Slow | Intent, logic errors |
| **AI Security** | GPT-4/Claude | 85-95% | Slow | Vulnerabilities |
| **Pattern** | Regex | 60-80% | Fast | Custom rules |

## Formal Verification (Z3)

Mathematical proofs using the Z3 Satisfiability Modulo Theories (SMT) solver.

### How It Works

1. **Code Translation**: Your code is translated to mathematical constraints
2. **Constraint Solving**: Z3 searches for inputs that violate safety properties
3. **Result**:
   - **SAT** (satisfiable): Bug found with counterexample
   - **UNSAT** (unsatisfiable): Property is proven safe
   - **UNKNOWN**: Timeout or undecidable

### Supported Checks

#### Null Safety (`null_safety`)

Detects potential null/None dereferences:

```python
def greet(user: User | None) -> str:
    return f"Hello, {user.name}"  # Bug: user could be None
```

**Z3 Analysis:**
```
Query: Can user be None when .name is accessed?
Result: SAT
Counterexample: user = None
```

#### Array Bounds (`array_bounds`)

Catches out-of-bounds array access:

```python
def get_first(items: list[int]) -> int:
    return items[0]  # Bug: list could be empty
```

**Z3 Analysis:**
```
Query: Can len(items) be 0 when items[0] is accessed?
Result: SAT
Counterexample: items = []
```

#### Integer Overflow (`integer_overflow`)

Identifies arithmetic overflow risks:

```typescript
function multiply(a: number, b: number): number {
    return a * b;  // In typed languages, could overflow
}
```

**Z3 Analysis:**
```
Query: Can a * b exceed MAX_INT?
Result: SAT
Counterexample: a = 2147483647, b = 2
```

#### Division by Zero (`division_by_zero`)

Prevents divide-by-zero errors:

```python
def average(total: int, count: int) -> float:
    return total / count  # Bug: count could be zero
```

**Z3 Analysis:**
```
Query: Can count be 0?
Result: SAT
Counterexample: count = 0
```

### Confidence Level

Formal verification has the **highest confidence** (95-99%):
- When Z3 finds a bug, it provides a **proof** (counterexample)
- When Z3 proves safety, it's mathematically guaranteed
- False positives are extremely rare

### Limitations

Z3 cannot verify:
- Complex string operations
- Network/IO behavior
- Unbounded loops
- Dynamic code (eval, reflection)

## AI Semantic Analysis

LLM-powered understanding of code intent and logic.

### How It Works

The Semantic Agent (GPT-4 or Claude) analyzes code in context:

1. Reads function signature, body, and docstring
2. Understands the intended behavior
3. Identifies logical errors and code smells
4. Suggests improvements

### What It Catches

```python
def calculate_discount(price: float, tier: str) -> float:
    """Calculate discount based on customer tier."""
    if tier == "gold":
        return price * 0.8
    elif tier == "silver":
        return price * 0.9
    # Bug: No handling for invalid tier or default case
```

**Semantic Analysis:**
```
⚠️ MEDIUM: Missing default case for tier parameter

The function handles "gold" and "silver" tiers but doesn't
handle other values or provide a default. This could lead to
unexpected behavior or return None implicitly.

Suggestion: Add a default case or raise ValueError for unknown tiers.
```

### Confidence Level

AI semantic analysis has **high confidence** (80-95%):
- Excellent at understanding intent
- May have false positives for unusual patterns
- Confidence varies by code complexity

## AI Security Analysis

Specialized vulnerability scanning powered by LLMs.

### Coverage

| Category | Examples |
|----------|----------|
| **Injection** | SQL, XSS, command, LDAP, path traversal |
| **Auth/AuthZ** | Broken authentication, missing access control |
| **Crypto** | Weak algorithms, hardcoded secrets, insecure random |
| **Data** | Sensitive data exposure, insecure logging |
| **Config** | Security misconfiguration, debug enabled |

### Example: SQL Injection

```python
def find_user(username: str):
    query = f"SELECT * FROM users WHERE name = '{username}'"
    return db.execute(query)
```

**Security Analysis:**
```
🔴 CRITICAL: SQL Injection vulnerability

User input 'username' is directly interpolated into SQL query
without sanitization or parameterization.

Attack vector:
  username = "'; DROP TABLE users; --"

Impact: Database compromise, data theft, data destruction

Fix:
  query = "SELECT * FROM users WHERE name = ?"
  return db.execute(query, [username])
```

### Confidence Level

Security analysis has **high confidence** (85-95%):
- Trained on extensive vulnerability patterns
- May flag defensive code as vulnerable
- Better to over-report than miss real vulnerabilities

## Pattern Matching

Fast regex-based detection for custom rules.

### Built-in Patterns

CodeVerify includes patterns for common issues:
- Hardcoded credentials
- TODO/FIXME comments in production
- Debug print statements
- Deprecated function usage

### Custom Patterns

Define your own rules in `.codeverify.yml`:

```yaml
rules:
  - id: no-print
    name: No print statements
    pattern: "\\bprint\\s*\\("
    message: "Use logging instead of print"
    severity: low
    languages: [python]
    
  - id: no-console-log
    name: No console.log
    pattern: "console\\.log\\s*\\("
    message: "Remove console.log before committing"
    severity: low
    languages: [typescript, javascript]
    
  - id: no-hardcoded-urls
    name: No hardcoded URLs
    pattern: "https?://[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
    message: "Use configuration for URLs"
    severity: medium
    exclude:
      - "**/*.test.*"
      - "**/tests/**"
```

### Confidence Level

Pattern matching has **medium confidence** (60-80%):
- Fast and deterministic
- High false positive rate for complex patterns
- Best for simple, unambiguous rules

## Choosing Verification Methods

### Recommended Configuration

```yaml
verification:
  enabled: true
  checks:
    - null_safety      # Always enable
    - division_by_zero # Always enable
    - array_bounds     # Enable for array-heavy code
    - integer_overflow # Enable for numeric code

ai:
  enabled: true
  semantic: true       # Enable for logic analysis
  security: true       # Always enable

rules:
  - id: custom-rules   # Add project-specific patterns
```

### When to Disable

- **Formal verification**: Disable for performance-critical CI on non-critical code
- **AI analysis**: Disable if no API key or for private/sensitive code
- **Pattern matching**: Keep enabled (fast and free)

## Next Steps

- **[Understanding Findings](./findings)** — Interpret results and confidence
- **[Verification Deep-Dive](/docs/verification/overview)** — Z3 internals
- **[Custom Rules](/docs/configuration/custom-rules)** — Create your own patterns
