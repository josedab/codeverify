---
sidebar_position: 3
---

# Comparison with Alternatives

How CodeVerify compares to other code analysis tools.

## Quick Comparison

| Feature | CodeVerify | ESLint/Pylint | SonarQube | Semgrep | GitHub CodeQL |
|---------|------------|---------------|-----------|---------|---------------|
| Formal Verification | ✅ Z3 SMT | ❌ | ❌ | ❌ | ❌ |
| AI Analysis | ✅ GPT-4/Claude | ❌ | Limited | ❌ | ❌ |
| Null Safety | ✅ Proven | Pattern-based | Pattern-based | Pattern-based | Dataflow |
| Array Bounds | ✅ Proven | ❌ | ❌ | ❌ | Limited |
| Copilot Integration | ✅ Trust Score | ❌ | ❌ | ❌ | ❌ |
| Real-time IDE | ✅ | ✅ | ❌ | ❌ | ❌ |
| Self-hosted | ✅ | ✅ | ✅ | ✅ | ✅ |

## Detailed Comparisons

### vs ESLint / Pylint (Linters)

**What linters do well:**
- Fast, instant feedback
- Style and convention checking
- Large plugin ecosystem
- Free and open source

**Where CodeVerify excels:**
- Finds bugs linters can't detect
- Mathematical proofs, not just patterns
- Understands code semantics
- AI-powered suggestions

**When to use both:**
Use linters for style and CodeVerify for correctness. They're complementary.

```yaml
# Use both in CI
steps:
  - run: eslint src/
  - run: codeverify analyze src/
```

### vs SonarQube

**What SonarQube does well:**
- Comprehensive code quality metrics
- Technical debt tracking
- Large language support
- Mature enterprise features

**Where CodeVerify excels:**
- Formal verification (mathematical proofs)
- AI-powered analysis
- Copilot integration
- Faster analysis for verification checks

**When to use both:**
SonarQube for broad code quality metrics, CodeVerify for deep verification.

| Aspect | SonarQube | CodeVerify |
|--------|-----------|------------|
| Focus | Quality metrics | Bug prevention |
| Method | Pattern matching | Formal verification + AI |
| Depth | Broad coverage | Deep on specific bugs |
| Speed | Minutes | Seconds to minutes |

### vs Semgrep

**What Semgrep does well:**
- Fast pattern matching
- Easy custom rules
- Great for security patterns
- Lightweight

**Where CodeVerify excels:**
- Understands code meaning, not just patterns
- Proves absence of bugs
- AI explains findings
- Fewer false positives

**Example:**

Semgrep can find patterns like:
```yaml
# Semgrep rule
pattern: $X / $Y
message: "Potential division by zero"
```

But it flags every division. CodeVerify proves whether `$Y` can actually be zero.

### vs GitHub CodeQL

**What CodeQL does well:**
- Powerful query language
- Deep dataflow analysis
- Security-focused
- GitHub integration

**Where CodeVerify excels:**
- Mathematical proofs via Z3
- AI-powered explanations
- Copilot Trust Score
- Simpler setup (no query writing)

**Analysis comparison:**

```python
def get_user_name(user_id):
    user = find_user(user_id)
    return user.name  # Both detect this
```

- **CodeQL:** Finds dataflow from nullable source to dereference
- **CodeVerify:** Proves `user = None` is satisfiable, provides counterexample

### vs TypeScript's Strict Mode

**What strict mode does well:**
- Compile-time null checks
- Type inference
- Zero runtime cost
- No external tools

**Where CodeVerify excels:**
- Catches issues strict mode misses
- Works with any codebase (not just strict)
- AI suggests fixes
- Verifies runtime behavior

**Issues CodeVerify catches that strict mode doesn't:**

```typescript
// strict mode says this is fine
function getFirst(arr: string[]): string {
    return arr[0];  // ✅ TypeScript: OK
}                   // ⚠️ CodeVerify: arr may be empty

// TypeScript trusts your type assertions
const user = getUser(id) as User;  // ✅ TypeScript: trusts you
user.name;                          // ⚠️ CodeVerify: may still be null
```

### vs Formal Verification Tools (Dafny, SPARK, etc.)

**What specialized tools do well:**
- Full functional correctness proofs
- Can prove complex properties
- Academic rigor

**Where CodeVerify excels:**
- Works with existing code (no rewrite)
- No specification language to learn
- Practical, targeted checks
- Developer-friendly output

**Trade-offs:**

| Aspect | Dafny/SPARK | CodeVerify |
|--------|-------------|------------|
| Properties | Arbitrary specs | Specific checks |
| Learning curve | Steep | Gentle |
| Code changes | Required | None |
| Integration | Specialized | Standard CI |

### vs AI-Only Tools (GPT-based reviewers)

**What AI-only tools do well:**
- Natural language feedback
- Broad suggestions
- Easy to use

**Where CodeVerify excels:**
- Proofs, not just suggestions
- Reproducible results
- Low false positives
- Explainable findings

**Key difference:**
AI alone can miss or hallucinate issues. CodeVerify combines AI with formal verification for the best of both:

```
AI: "This might have a null dereference"
Z3: "Proof: user=None is satisfiable"
Result: Verified finding with mathematical proof
```

## Summary

| If you need... | Use... |
|----------------|--------|
| Style enforcement | ESLint/Pylint + CodeVerify |
| Code quality metrics | SonarQube + CodeVerify |
| Security patterns | Semgrep + CodeVerify |
| Proven null safety | CodeVerify |
| Copilot verification | CodeVerify |
| Full formal proofs | Dafny/SPARK |

## Still Deciding?

Try CodeVerify free for 14 days. It takes 5 minutes to set up and works alongside your existing tools.

```bash
pip install codeverify
codeverify analyze src/
```
