---
sidebar_position: 4
---

# Copilot Trust Score

An ML-powered assessment of AI-generated code quality and risk, specifically designed for code from GitHub Copilot and similar AI assistants.

## Why Trust Scores?

AI coding assistants are incredibly productive but can introduce subtle bugs:

- **Context blindness**: Copilot doesn't see your full codebase
- **Hallucination**: May invent APIs or patterns that don't exist
- **Security gaps**: Trained on code with vulnerabilities
- **Style drift**: Suggestions may not match your conventions

Trust Score gives you instant feedback on whether to accept or review a suggestion.

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    Trust Score Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Copilot Suggestion                                         │
│         │                                                    │
│         ▼                                                    │
│   ┌───────────────┐                                          │
│   │ Context Check │  Does it fit the surrounding code?       │
│   └───────┬───────┘                                          │
│           │                                                  │
│           ▼                                                  │
│   ┌───────────────┐                                          │
│   │ Safety Check  │  Any formal verification issues?         │
│   └───────┬───────┘                                          │
│           │                                                  │
│           ▼                                                  │
│   ┌───────────────┐                                          │
│   │Security Check │  Any vulnerability patterns?             │
│   └───────┬───────┘                                          │
│           │                                                  │
│           ▼                                                  │
│   ┌───────────────┐                                          │
│   │ Style Check   │  Matches project conventions?            │
│   └───────┬───────┘                                          │
│           │                                                  │
│           ▼                                                  │
│   ┌─────────────────────────────────────────────┐           │
│   │  Trust Score: 72/100  ⚠️ Review Recommended │           │
│   └─────────────────────────────────────────────┘           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Score Ranges

| Score | Badge | Meaning | Action |
|-------|-------|---------|--------|
| **90-100** | ✅ Safe | High quality, no issues detected | Accept confidently |
| **70-89** | ⚠️ Review | Minor concerns, worth checking | Review before accepting |
| **50-69** | 🟡 Caution | Significant issues found | Careful review required |
| **Below 50** | 🔴 Risky | Major problems detected | Consider rejecting |

## Scoring Factors

### Positive Factors (+)

| Factor | Weight | Description |
|--------|--------|-------------|
| Type correctness | 20% | Types match context and are consistent |
| Null safety | 15% | Proper null/undefined handling |
| Error handling | 15% | Exceptions and edge cases handled |
| Style match | 15% | Follows project conventions |
| API correctness | 15% | Uses existing APIs correctly |
| Documentation | 10% | Has appropriate comments |
| Test coverage | 10% | Testable and/or includes tests |

### Negative Factors (-)

| Factor | Penalty | Description |
|--------|---------|-------------|
| Division by zero risk | -20 | Unchecked division operations |
| Null dereference risk | -15 | Missing null checks |
| Security vulnerability | -30 | SQL injection, XSS, etc. |
| Type mismatch | -10 | Incompatible with context |
| Hardcoded values | -5 | Magic numbers/strings |
| Missing validation | -10 | No input validation |

## VS Code Integration

### Enable Trust Score

Add to your VS Code settings:

```json
{
  "codeverify.copilotInterception.enabled": true,
  "codeverify.copilotInterception.showTrustScore": true,
  "codeverify.copilotInterception.minimumScore": 70
}
```

### Visual Feedback

When Copilot suggests code, you'll see:

```
┌────────────────────────────────────────────────────┐
│ def calculate_average(numbers):                    │
│     return sum(numbers) / len(numbers)             │
│                                                    │
│ ────────────────────────────────────────────────── │
│ Trust Score: 65/100  🟡 Caution                   │
│                                                    │
│ Issues:                                            │
│ • Division by zero if list is empty (-20)          │
│ • Missing type hints (-10)                         │
│ • No docstring (-5)                                │
│                                                    │
│ [Accept] [Accept with Fix] [Reject]                │
└────────────────────────────────────────────────────┘
```

### Auto-Fix Option

"Accept with Fix" automatically patches issues:

```python
# Original suggestion (score: 65)
def calculate_average(numbers):
    return sum(numbers) / len(numbers)

# After auto-fix (score: 95)
def calculate_average(numbers: list[float]) -> float:
    """Calculate the average of a list of numbers."""
    if not numbers:
        raise ValueError("cannot calculate average of empty list")
    return sum(numbers) / len(numbers)
```

## Configuration

### Basic Setup

```yaml
# .codeverify.yml
trust_score:
  enabled: true
  minimum_score: 70           # Block suggestions below this
  show_inline: true           # Show score in editor
  auto_fix_threshold: 80      # Offer auto-fix below this
```

### Custom Weights

Adjust scoring weights for your project:

```yaml
trust_score:
  enabled: true
  weights:
    type_safety: 0.25         # Increase for typed codebases
    null_checks: 0.20
    security: 0.25            # Increase for security-sensitive code
    style: 0.10               # Decrease if style is flexible
    documentation: 0.10
    error_handling: 0.10
```

### Strict Mode

Block all low-scoring suggestions automatically:

```yaml
trust_score:
  strict_mode: true
  minimum_score: 75
  block_on_security: true     # Always block security issues
```

## API Usage

### Python SDK

```python
from codeverify_agents.trust_score import TrustScoreAgent

agent = TrustScoreAgent()

result = await agent.analyze(
    suggestion="def divide(a, b): return a / b",
    context="""
    # Calculator module
    # All inputs are validated by the caller
    """,
    language="python",
    file_path="src/calculator.py"
)

print(f"Trust Score: {result.score}/100")
print(f"Recommendation: {result.recommendation}")

for issue in result.issues:
    print(f"  - {issue.description} ({issue.penalty:+d})")
    
if result.auto_fix:
    print(f"\nSuggested fix:\n{result.auto_fix.code}")
```

### Response Structure

```python
@dataclass
class TrustScoreResult:
    score: int                    # 0-100
    recommendation: str           # "accept", "review", "caution", "reject"
    issues: list[Issue]           # Found problems
    positives: list[Positive]     # Good qualities
    auto_fix: AutoFix | None      # Suggested fix if applicable
    confidence: float             # How confident is this score
    
@dataclass
class Issue:
    category: str                 # "safety", "security", "style", etc.
    description: str
    penalty: int                  # Negative score impact
    location: Location | None
    fix_suggestion: str | None
```

## Interpreting Scores

### High Score (90+)

```python
# Suggestion: 
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

# Score: 95/100 ✅
# + Correct types
# + Has docstring
# + Simple and safe
# + Matches project style
```

### Medium Score (70-89)

```python
# Suggestion:
def get_user(user_id):
    return db.query(f"SELECT * FROM users WHERE id = {user_id}")

# Score: 72/100 ⚠️
# - Missing type hints (-10)
# - Potential SQL injection (-15)
# + Uses existing db object
# + Logical structure correct
```

### Low Score (Below 70)

```python
# Suggestion:
def process(data):
    return data.items[0].value.strip().lower()

# Score: 45/100 🔴
# - No type hints (-10)
# - data could be None (-15)
# - data.items could be empty (-15)
# - data.items[0].value could be None (-10)
# - No error handling (-10)
```

## Best Practices

### 1. Set Appropriate Thresholds

- **New projects**: Start with `minimum_score: 60`, increase over time
- **Production code**: Use `minimum_score: 75` or higher
- **Security-sensitive**: Use `minimum_score: 85` with `block_on_security: true`

### 2. Review Medium Scores

Scores in the 70-89 range often have minor issues worth fixing:
- Missing type hints
- Incomplete error handling
- Style inconsistencies

### 3. Train Your Team

Share Trust Score reports to help team members:
- Understand common AI assistant pitfalls
- Learn what makes code trustworthy
- Develop intuition for reviewing AI code

### 4. Track Trends

Monitor team-wide Trust Score metrics:
- Average score over time
- Most common issues
- Which team members have highest/lowest averages

## Next Steps

- **[VS Code Extension](/docs/integrations/vscode)** — Full IDE integration
- **[Copilot Interceptor](/docs/advanced/copilot-interceptor)** — Advanced interception
- **[Team Learning](/docs/advanced/team-learning)** — Organization-wide insights
