---
sidebar_position: 6
---

# Verification Debugger

Debug and understand CodeVerify's verification process.

## Overview

The verification debugger helps you:
- Understand why a finding was reported
- See the Z3 constraints generated
- View counterexamples step by step
- Debug false positives
- Tune verification settings

## Starting the Debugger

### Interactive Mode

```bash
codeverify debug src/calculator.py
```

Opens an interactive session for exploring verification results.

### Specific Function

```bash
codeverify debug src/calculator.py --function divide
```

### With Configuration

```bash
codeverify debug src/calculator.py --config .codeverify.yml --verbose
```

## Debugger Commands

### `analyze` - Run Analysis

```
(cv-debug) analyze
Analyzing src/calculator.py...

Found 2 findings:
  1. [HIGH] Division by zero at line 15 (divide)
  2. [MEDIUM] Array bounds at line 23 (get_element)
```

### `show` - View Finding Details

```
(cv-debug) show 1

Finding: Division by zero
Location: src/calculator.py:15
Severity: HIGH
Category: division_by_zero

Code:
    14 │ def divide(a: int, b: int) -> float:
    15 │     return a / b  # ← Finding here
    16 │

Counterexample:
    a = 42
    b = 0

Explanation:
    The parameter 'b' is not checked before division.
    When b = 0, this causes ZeroDivisionError.
```

### `constraints` - View Z3 Constraints

```
(cv-debug) constraints 1

Z3 Constraints for 'divide':

Variables:
    a: Int
    b: Int

Preconditions:
    (none)

Query: Is there an input where b = 0?
    (= b 0)

Result: SAT
    Model: a = 42, b = 0
```

### `trace` - Execution Trace

```
(cv-debug) trace 1

Execution trace for counterexample:

    Line 14: Enter function 'divide'
             a = 42, b = 0
    Line 15: Evaluate 'a / b'
             a / b = 42 / 0
             💥 Division by zero
```

### `paths` - View All Paths

```
(cv-debug) paths divide

Execution paths for 'divide':

Path 1 (verified safe):
    b != 0 → return a / b ✅

Path 2 (counterexample found):
    b = 0 → return a / b 💥
```

### `why` - Explain Finding

```
(cv-debug) why 1

Why this finding was reported:

1. CodeVerify found that 'b' could be 0
2. There is no guard checking 'b != 0' before division
3. Z3 confirmed a counterexample exists: b = 0
4. The finding is not suppressed by any ignore rule

To fix:
    Add a check: if b == 0: raise ValueError("...")
    Or return a default: if b == 0: return 0.0
```

### `suppress` - Mark as False Positive

```
(cv-debug) suppress 1 "External validation ensures b is never 0"

Finding 1 suppressed. Added to .codeverify.yml:

ignore:
  - finding_id: "f_abc123"
    reason: "External validation ensures b is never 0"
```

### `fix` - Apply Suggested Fix

```
(cv-debug) fix 1

Suggested fix for Finding 1:

--- a/src/calculator.py
+++ b/src/calculator.py
@@ -14,6 +14,8 @@
 def divide(a: int, b: int) -> float:
+    if b == 0:
+        raise ValueError("Cannot divide by zero")
     return a / b

Apply this fix? [y/n]: y
Fix applied.
```

### `reanalyze` - Re-run After Changes

```
(cv-debug) reanalyze

Re-analyzing src/calculator.py...

Found 1 finding (was 2):
  1. [MEDIUM] Array bounds at line 23 (get_element)

✅ Finding 1 (division by zero) is now resolved
```

## Verbose Output

### Constraint Generation Details

```bash
codeverify debug src/file.py --verbose=constraints
```

Shows:
- How each line translates to constraints
- Variable bindings
- Type information used

### Z3 Solving Details

```bash
codeverify debug src/file.py --verbose=solver
```

Shows:
- Solving strategy used
- Intermediate results
- Time spent per query

### All Details

```bash
codeverify debug src/file.py --verbose=all
```

## Proof Export

Export Z3 proofs for external analysis:

```bash
codeverify debug src/file.py --export-proofs proofs/
```

Creates:
```
proofs/
├── divide.smt2          # SMT-LIB format
├── divide.proof.json    # Proof details
└── get_element.smt2
```

### SMT-LIB Format

```smt2
; proofs/divide.smt2
(declare-const a Int)
(declare-const b Int)

; Check if b can be 0
(assert (= b 0))

(check-sat)
; Result: sat
; Model: a = 42, b = 0
```

You can run this with Z3 directly:

```bash
z3 proofs/divide.smt2
```

## Comparing Analyses

Compare results between configurations:

```bash
codeverify debug src/file.py --compare strict.yml lenient.yml
```

Output:
```
Configuration comparison:

                     strict.yml    lenient.yml
Findings total:            5              2
  Critical:                1              1
  High:                    2              0
  Medium:                  2              1
  
Differences:
  + Finding 2 (HIGH) only in strict.yml
  + Finding 3 (HIGH) only in strict.yml
  + Finding 5 (MEDIUM) only in strict.yml
```

## Interactive Expression Evaluation

Test expressions against constraints:

```
(cv-debug) eval divide b > 0

With constraint 'b > 0':
  Division is safe ✅
  No counterexample possible
```

```
(cv-debug) eval divide b >= -10 and b <= 10

With constraint '-10 <= b <= 10':
  Division is NOT safe ❌
  Counterexample: b = 0
```

## Batch Debugging

Analyze multiple files:

```bash
codeverify debug src/ --output debug-report.json
```

Creates a JSON report with all findings and counterexamples.

## Configuration for Debugging

### Enable Proof Logging

```yaml
verification:
  advanced:
    log_proofs: true
    proof_dir: ".codeverify/proofs"
```

### Increase Timeout for Debugging

```yaml
verification:
  timeout: 120  # More time for complex queries
  advanced:
    max_path_depth: 20  # Explore more paths
```

## Troubleshooting

### "Unknown" Result

When Z3 returns "unknown":

```
(cv-debug) show 1

Result: UNKNOWN (timeout after 30s)

Try:
  1. Increase timeout: --timeout 120
  2. Simplify function
  3. Add type annotations
```

### Too Many Findings

```
(cv-debug) filter --severity high,critical

Filtered to 2 findings (was 15):
  1. [CRITICAL] SQL injection at line 45
  2. [HIGH] Null dereference at line 78
```

### Reproducibility

Export and replay:

```bash
# Export
codeverify debug src/file.py --export-state debug.state

# Replay on another machine
codeverify debug --replay debug.state
```

## Next Steps

- [Configuration Reference](/docs/configuration/overview) — Tune verification settings
- [Custom Rules](/docs/configuration/custom-rules) — Create custom checks
- [Troubleshooting](/docs/resources/troubleshooting) — Common issues and solutions
