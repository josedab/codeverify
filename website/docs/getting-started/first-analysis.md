---
sidebar_position: 3
---

# Your First Analysis

A complete walkthrough of analyzing real code and understanding CodeVerify's findings.

## Sample Project

Let's create a small Python project with common issues that CodeVerify catches:

```python title="src/calculator.py"
"""A simple calculator module with some intentional issues."""

def divide(a: int, b: int) -> float:
    """Divide two numbers."""
    return a / b  # Issue: b could be zero

def get_element(items: list, index: int):
    """Get element at index."""
    return items[index]  # Issue: index could be out of bounds

def process_user(user) -> str:
    """Process user data."""
    return user.name.upper()  # Issue: user could be None

def calculate_average(numbers: list[int]) -> float:
    """Calculate average of numbers."""
    total = sum(numbers)
    return total / len(numbers)  # Issue: list could be empty

def multiply_large(a: int, b: int) -> int:
    """Multiply two large numbers."""
    return a * b  # Issue: could overflow (in languages with fixed-size ints)
```

## Running the Analysis

Run CodeVerify with verbose output:

```bash
codeverify analyze src/calculator.py --verbose
```

## Understanding Each Finding

### Finding 1: Division by Zero

```
🔴 CRITICAL: Division by zero possible

Location: src/calculator.py:5:12
Function: divide(a: int, b: int) -> float

Verification Method: FORMAL (Z3 SMT Solver)
Confidence: 99%

Code:
    def divide(a: int, b: int) -> float:
        """Divide two numbers."""
        return a / b
               ^^^^^

Analysis:
The Z3 solver found that when b = 0, this operation raises
ZeroDivisionError. The function signature allows any integer
for b, including zero.

Counterexample:
  a = 1
  b = 0
  Result: ZeroDivisionError

Suggested Fix:
    def divide(a: int, b: int) -> float:
        """Divide two numbers."""
  +     if b == 0:
  +         raise ValueError("divisor cannot be zero")
        return a / b
```

**Why this matters:** Division by zero crashes your program. The counterexample `b=0` proves this bug exists—it's not a guess.

### Finding 2: Array Bounds Violation

```
🟠 HIGH: Array index may be out of bounds

Location: src/calculator.py:9:12
Function: get_element(items: list, index: int)

Verification Method: FORMAL (Z3 SMT Solver)
Confidence: 99%

Code:
    def get_element(items: list, index: int):
        """Get element at index."""
        return items[index]
               ^^^^^^^^^^^^

Analysis:
The function accepts any integer for index, but valid indices
are in range [0, len(items)-1]. Negative indices or indices
>= len(items) raise IndexError.

Counterexamples:
  Case 1: items = [], index = 0    → IndexError (empty list)
  Case 2: items = [1], index = -2  → IndexError (negative)
  Case 3: items = [1], index = 5   → IndexError (too large)

Suggested Fix:
    def get_element(items: list, index: int):
        """Get element at index."""
  +     if not items:
  +         raise ValueError("items list is empty")
  +     if not (0 <= index < len(items)):
  +         raise IndexError(f"index {index} out of range [0, {len(items)-1}]")
        return items[index]
```

### Finding 3: Null Dereference

```
🟠 HIGH: Potential null dereference

Location: src/calculator.py:13:12
Function: process_user(user) -> str

Verification Method: AI Semantic Analysis
Confidence: 92%

Code:
    def process_user(user) -> str:
        """Process user data."""
        return user.name.upper()
               ^^^^^^^^^

Analysis:
The parameter 'user' has no type annotation, and there's no
null check before accessing user.name. If user is None, this
raises AttributeError.

Additionally, even if user is not None, user.name could be
None, causing another AttributeError on .upper().

Suggested Fix:
    def process_user(user: User | None) -> str:
        """Process user data."""
  +     if user is None:
  +         raise ValueError("user cannot be None")
  +     if user.name is None:
  +         return ""
        return user.name.upper()
```

### Finding 4: Empty Collection Division

```
🔴 CRITICAL: Division by zero possible (empty list)

Location: src/calculator.py:18:12
Function: calculate_average(numbers: list[int]) -> float

Verification Method: FORMAL (Z3 SMT Solver)
Confidence: 99%

Code:
    def calculate_average(numbers: list[int]) -> float:
        """Calculate average of numbers."""
        total = sum(numbers)
        return total / len(numbers)
                       ^^^^^^^^^^^^

Analysis:
When numbers is an empty list, len(numbers) = 0, causing
division by zero.

Counterexample:
  numbers = []
  total = 0
  len(numbers) = 0
  Result: ZeroDivisionError

Suggested Fix:
    def calculate_average(numbers: list[int]) -> float:
        """Calculate average of numbers."""
  +     if not numbers:
  +         raise ValueError("cannot calculate average of empty list")
        total = sum(numbers)
        return total / len(numbers)
```

## Fixing the Code

Here's the corrected version:

```python title="src/calculator.py"
"""A simple calculator module - now verified safe."""

def divide(a: int, b: int) -> float:
    """Divide two numbers.
    
    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("divisor cannot be zero")
    return a / b

def get_element(items: list, index: int):
    """Get element at index.
    
    Raises:
        ValueError: If items is empty.
        IndexError: If index is out of bounds.
    """
    if not items:
        raise ValueError("items list is empty")
    if not (0 <= index < len(items)):
        raise IndexError(f"index {index} out of range")
    return items[index]

def process_user(user) -> str:
    """Process user data.
    
    Raises:
        ValueError: If user is None.
    """
    if user is None:
        raise ValueError("user cannot be None")
    return (user.name or "").upper()

def calculate_average(numbers: list[int]) -> float:
    """Calculate average of numbers.
    
    Raises:
        ValueError: If numbers is empty.
    """
    if not numbers:
        raise ValueError("cannot calculate average of empty list")
    return sum(numbers) / len(numbers)

def multiply_large(a: int, b: int) -> int:
    """Multiply two large numbers."""
    return a * b  # Safe in Python (arbitrary precision integers)
```

## Re-run Analysis

```bash
codeverify analyze src/calculator.py
```

Output:

```
CodeVerify Analysis Results
═══════════════════════════

📁 Analyzed: src/calculator.py (5 functions, 52 lines)
⏱️  Duration: 1.8s

Summary: 0 critical, 0 high, 0 medium, 0 low
Status: ✅ PASSED
```

## Using Watch Mode

For continuous feedback during development:

```bash
codeverify analyze src/ --watch
```

CodeVerify re-analyzes on every file save:

```
[10:23:45] Watching src/ for changes...
[10:24:12] Changed: src/calculator.py
[10:24:14] ✅ All checks passed
[10:25:01] Changed: src/utils.py
[10:25:03] 🔴 1 critical finding in src/utils.py:23
```

## Suppressing False Positives

If CodeVerify flags something you've intentionally handled:

```python
# Inline suppression with reason
result = data / count  # codeverify: ignore division_by_zero -- count validated in caller

# Function-level suppression
# codeverify: ignore-function null_safety
def internal_process(data):
    # This function is only called with validated data
    return data.value
```

## Next Steps

Now that you understand how CodeVerify works:

- **[Configuration Guide](/docs/configuration/overview)** — Customize for your project
- **[Verification Deep-Dive](/docs/verification/overview)** — How Z3 verification works
- **[GitHub Integration](/docs/integrations/github)** — Automatic PR analysis
- **[VS Code Extension](/docs/integrations/vscode)** — Real-time verification
