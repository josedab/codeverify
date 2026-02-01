---
sidebar_position: 1
---

# Copilot Interceptor

Verify AI-generated code suggestions before they enter your codebase.

## Overview

The Copilot Interceptor analyzes GitHub Copilot suggestions in real-time, providing a **Trust Score** before you accept code. This helps catch potential bugs, security issues, and code quality problems in AI-generated code.

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                         VS Code                              │
├─────────────────────────────────────────────────────────────┤
│  1. You type code                                           │
│  2. Copilot generates suggestion                            │
│  3. CodeVerify intercepts & analyzes        ←── Trust Score │
│  4. You see suggestion + verification                       │
│  5. Accept with confidence (or reject)                      │
└─────────────────────────────────────────────────────────────┘
```

## Installation

The interceptor is included in the VS Code extension:

1. Install the CodeVerify VS Code extension
2. Enable Copilot Interceptor in settings:

```json
{
  "codeverify.copilot.interceptor": true
}
```

3. Restart VS Code

## Usage

### Trust Score Badge

When Copilot suggests code, you'll see a trust score badge:

```
function divide(a, b) {          [Trust: 45/100 ⚠️]
    return a / b;
}
```

Scores:
- **80-100** 🟢 High confidence — Safe to accept
- **50-79** 🟡 Medium confidence — Review recommended
- **0-49** 🔴 Low confidence — Likely issues

### Inline Details

Hover over the badge for details:

```
Trust Score: 45/100

Issues Found:
⚠️ Division by zero possible (b not checked)
⚠️ No input validation
⚠️ Missing type annotations

Formal Verification: Failed
AI Analysis: 2 concerns
Pattern Match: 1 warning
```

### Quick Actions

From the suggestion:
- **Accept** — Insert the code as-is
- **Accept & Fix** — Insert with automatic fixes applied
- **Reject** — Dismiss the suggestion
- **View Details** — Open full analysis panel

## Configuration

### Enable/Disable

```json
{
  "codeverify.copilot.interceptor": true,
  "codeverify.copilot.showTrustScore": true
}
```

### Minimum Score Threshold

Auto-warn when score is below threshold:

```json
{
  "codeverify.copilot.minimumScore": 70,
  "codeverify.copilot.warnBelow": 50
}
```

### Analysis Depth

Balance speed vs. thoroughness:

```json
{
  "codeverify.copilot.analysisDepth": "standard"
}
```

Options:
- `"fast"` — Quick pattern matching (100ms)
- `"standard"` — Patterns + basic verification (300ms)
- `"thorough"` — Full verification + AI analysis (1-2s)

### Checks to Run

```json
{
  "codeverify.copilot.checks": [
    "null_safety",
    "array_bounds",
    "division_by_zero",
    "security_patterns"
  ]
}
```

## Trust Score Calculation

The Trust Score combines multiple signals:

| Component | Weight | Description |
|-----------|--------|-------------|
| Formal Verification | 40% | Z3 proof results |
| AI Confidence | 30% | LLM analysis of code quality |
| Pattern Coverage | 20% | Known vulnerability patterns |
| Historical Accuracy | 10% | Past Copilot suggestion quality |

### Score Breakdown

```
Trust Score: 72/100

Components:
├─ Formal Verification: 80/100 (32 points)
│  └─ All null safety checks pass
├─ AI Confidence: 70/100 (21 points)
│  └─ Good structure, minor concerns
├─ Pattern Coverage: 60/100 (12 points)
│  └─ No security patterns detected
└─ Historical: 70/100 (7 points)
   └─ Based on 50 accepted suggestions
```

## Automatic Fixes

CodeVerify can suggest fixes for common issues:

### Before (Copilot Suggestion)

```typescript
function getUser(id: string) {
    const user = users.find(u => u.id === id);
    return user.name;  // Trust: 35 ⚠️
}
```

### After (Accept & Fix)

```typescript
function getUser(id: string): string | undefined {
    const user = users.find(u => u.id === id);
    return user?.name;  // Trust: 92 ✓
}
```

## Team Policies

Enforce team-wide quality standards:

### Repository Configuration

```yaml
# .codeverify.yml
copilot:
  interceptor:
    enabled: true
    
    # Block suggestions below this score
    minimum_score: 60
    
    # Require review for medium scores
    review_threshold: 80
    
    # Auto-apply safe fixes
    auto_fix: true
```

### Organizational Policies

Set defaults for all repositories:

```yaml
# Organization-level config
copilot:
  interceptor:
    minimum_score: 70
    required_checks:
      - null_safety
      - security_patterns
```

## Learning from Feedback

CodeVerify learns from your acceptance/rejection patterns:

```
📊 Suggestion Analytics (last 30 days)

Accepted: 234 suggestions
  Average score: 78
  
Rejected: 45 suggestions
  Average score: 52
  Common issues:
  - Missing null checks (18)
  - Type mismatches (12)
  - Security patterns (8)
  
Your rejection threshold appears to be ~55
Recommend setting minimumScore to 60
```

## Privacy

### What's Analyzed

- The code suggestion itself
- Surrounding context (configurable)
- File type and project context

### What's NOT Sent

- Your private code (unless using cloud AI)
- Git history
- Personal information

### Local-Only Mode

Run all analysis locally:

```json
{
  "codeverify.copilot.mode": "local",
  "codeverify.copilot.ai": false
}
```

This uses only pattern matching and Z3 verification.

## Troubleshooting

### Interceptor Not Working

1. Check VS Code output panel: `View > Output > CodeVerify`
2. Verify Copilot extension is installed and active
3. Ensure interceptor is enabled in settings

### Slow Analysis

1. Reduce analysis depth to `"fast"`
2. Disable AI analysis for inline scoring
3. Limit checks to essential ones

### False Positives

If suggestions are flagged incorrectly:

1. Lower the `minimumScore` threshold
2. Disable specific checks causing false positives
3. Report issues for pattern improvement

## Best Practices

1. **Start lenient** — Begin with `minimumScore: 50`, increase over time
2. **Review medium scores** — Don't auto-accept 50-79 range
3. **Learn the patterns** — Understand why suggestions score low
4. **Use team policies** — Consistent standards across the org
5. **Provide feedback** — Report false positives to improve accuracy

## Next Steps

- [Trust Scores](/docs/concepts/trust-scores) — Understanding the scoring system
- [VS Code Extension](/docs/integrations/vscode) — Full extension features
- [Team Learning](/docs/advanced/team-learning) — Organization-wide insights
