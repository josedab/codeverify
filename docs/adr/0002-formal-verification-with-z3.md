# ADR-0002: Formal Verification with Z3 SMT Solver

## Status

Accepted

## Context

CodeVerify aims to catch bugs in code, particularly AI-generated code, with higher confidence than traditional static analysis. Traditional approaches include:

1. **Pattern matching**: Fast but limited to known patterns
2. **Abstract interpretation**: Good for certain properties but can have false positives
3. **Formal verification**: Mathematical proofs but traditionally complex and slow

We needed to decide which verification approach would provide the best balance of accuracy, performance, and maintainability.

## Decision

We chose **Z3 SMT solver** for formal verification of specific properties:

- **Null safety violations**: Prove variables cannot be null when dereferenced
- **Array bounds checking**: Prove array indices are within bounds
- **Integer overflow detection**: Prove arithmetic operations don't overflow
- **Division by zero prevention**: Prove divisors are never zero

The Z3 verifier is implemented in `packages/verifier/` with:
- Language-specific parsers (Python, TypeScript, Java, Go, etc.)
- Verification condition generators
- Z3 constraint encoding
- Counterexample extraction for failures

## Consequences

### Positive
- **Mathematical guarantees**: When Z3 proves safety, it's certain (no false negatives for encoded properties)
- **Counterexamples**: Failures include concrete inputs that trigger the bug
- **Complementary to AI**: Provides ground truth to validate AI suggestions
- **Extensible**: New properties can be added by encoding verification conditions
- **Open source**: Z3 is MIT-licensed and well-maintained by Microsoft Research

### Negative
- **Limited scope**: Can only verify properties we explicitly encode
- **Performance overhead**: SMT solving can be slow for complex code paths
- **Encoding complexity**: Translating language semantics to SMT is non-trivial
- **Undecidability**: Some problems are undecidable; must use timeouts

### Neutral
- Requires maintaining language-specific parsers
- Learning curve for developers unfamiliar with formal methods

## Alternatives Considered

### Symbolic Execution (KLEE, Angr)
- **Rejected because**: More suited to binary analysis; harder to integrate with source-level analysis

### Abstract Interpretation (Infer, Polyspace)
- **Partially used**: AI agents use abstract reasoning; Z3 provides precision where needed

### Dependent Types (Idris, Agda)
- **Rejected because**: Requires changing source language; not practical for analyzing existing code

### Property-Based Testing (Hypothesis, QuickCheck)
- **Complementary**: Good for finding bugs but doesn't provide proofs

## References

- [Z3 Theorem Prover](https://github.com/Z3Prover/z3)
- [Z3 Python Bindings](https://z3prover.github.io/api/html/namespacez3py.html)
- [SAT/SMT by Example](https://sat-smt.codes/)
