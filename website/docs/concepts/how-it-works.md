---
sidebar_position: 1
---

# How CodeVerify Works

CodeVerify is a multi-stage analysis pipeline that combines AI intelligence with mathematical rigor.

## The Problem

Traditional code analysis has limitations:

| Approach | Strength | Weakness |
|----------|----------|----------|
| **Linters** (ESLint, Pylint) | Fast, deterministic | Pattern-only, miss context |
| **Type checkers** (TypeScript, mypy) | Catch type errors | Don't prove runtime safety |
| **AI review** (Copilot, Claude) | Understands intent | Can hallucinate, no proofs |

CodeVerify combines all three approaches to catch bugs that slip through.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CodeVerify Analysis Pipeline                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Source Code                                                        │
│       │                                                              │
│       ▼                                                              │
│   ┌─────────┐                                                        │
│   │ Parser  │  Language-specific AST extraction                      │
│   └────┬────┘                                                        │
│        │                                                             │
│        ▼                                                             │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │              Parallel Analysis Engines                       │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │   │
│   │  │  Semantic   │  │   Formal    │  │  Security   │         │   │
│   │  │   Agent     │  │  Verifier   │  │   Agent     │         │   │
│   │  │   (LLM)     │  │    (Z3)     │  │   (LLM)     │         │   │
│   │  │             │  │             │  │             │         │   │
│   │  │ • Intent    │  │ • Null safe │  │ • OWASP     │         │   │
│   │  │ • Logic     │  │ • Bounds    │  │ • Injection │         │   │
│   │  │ • Patterns  │  │ • Overflow  │  │ • Auth      │         │   │
│   │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │   │
│   │         │                │                │                 │   │
│   └─────────┼────────────────┼────────────────┼─────────────────┘   │
│             │                │                │                      │
│             └────────────────┼────────────────┘                      │
│                              ▼                                       │
│                      ┌─────────────┐                                 │
│                      │  Synthesis  │  Combine, dedupe, prioritize    │
│                      │    Agent    │                                 │
│                      └──────┬──────┘                                 │
│                             │                                        │
│                             ▼                                        │
│                      ┌─────────────┐                                 │
│                      │  Findings   │  With fixes and counterexamples │
│                      └─────────────┘                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Stage 1: Parsing

CodeVerify parses source code into an Abstract Syntax Tree (AST) enriched with type information.

**Python:**
- Uses Python's `ast` module
- Resolves type hints from annotations and stubs
- Follows imports to build call graph

**TypeScript:**
- Uses TypeScript compiler API
- Full type inference from `tsconfig.json`
- Handles JSX/TSX

## Stage 2: Semantic Analysis (AI)

The Semantic Agent uses GPT-4 or Claude to understand code intent:

- Extracts implicit contracts
- Identifies business logic constraints
- Detects code smells and anti-patterns
- Understands naming conventions and context

## Stage 3: Formal Verification (Z3)

The Z3 SMT solver provides **mathematical proofs** of correctness.

**How it works:**

1. Translate code to mathematical constraints
2. Solve the constraints (SAT/UNSAT/UNKNOWN)
3. Generate counterexample if bug found

**What Z3 can prove:**

| Property | Description |
|----------|-------------|
| Null safety | Variables are not None when dereferenced |
| Array bounds | Indices are within valid range |
| No overflow | Arithmetic stays within type limits |
| No div-by-zero | Divisors are never zero |

## Stage 4: Security Analysis (AI)

The Security Agent scans for vulnerabilities:

- **Injection:** SQL, XSS, command injection
- **Authentication:** Weak patterns, session issues
- **Cryptography:** Weak algorithms, hardcoded keys
- **Data exposure:** Logging secrets, insecure transmission

## Stage 5: Synthesis

The Synthesis Agent combines all findings:

1. Deduplicates overlapping findings
2. Ranks by severity and confidence
3. Generates fix suggestions
4. Produces structured output

## Why This Approach Works

| Scenario | Linter | AI Only | CodeVerify |
|----------|--------|---------|------------|
| `a / b` (b could be 0) | Miss | Maybe | Proof + counterexample |
| SQL injection | Pattern | Detect | Detect + fix |
| Complex business logic | Miss | Detect | Detect |
| Off-by-one in loop | Miss | Maybe | Proof |

## Next Steps

- **[Verification Types](./verification-types)** — Deep dive into each verification method
- **[Understanding Findings](./findings)** — Interpret severity and confidence
- **[Copilot Trust Score](./trust-scores)** — How AI code scoring works
