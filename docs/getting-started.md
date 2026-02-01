# Getting Started with CodeVerify

Welcome to CodeVerify! This guide will help you set up CodeVerify for your GitHub repositories and start getting AI-powered code reviews with formal verification.

## Quick Start (5 minutes)

### 1. Install the GitHub App

Visit our [GitHub App page](https://github.com/apps/codeverify) and click **Install**.

Select the repositories you want CodeVerify to analyze:
- **All repositories** - Analyze all current and future repos
- **Select repositories** - Choose specific repos

### 2. Create Your First Pull Request

Once installed, CodeVerify automatically analyzes every pull request. Create or update a PR to see it in action:

```bash
git checkout -b feature/test-codeverify
echo "def divide(a, b): return a / b" > test.py
git add test.py
git commit -m "Add divide function"
git push origin feature/test-codeverify
```

Open a pull request and watch CodeVerify analyze your code!

### 3. Review the Analysis

Within a few minutes, you'll see:
- ✅ A **GitHub Check** showing pass/fail status
- 💬 A **PR comment** with detailed findings
- 📝 **Inline annotations** on specific lines

## What CodeVerify Analyzes

### Formal Verification (Z3 SMT Solver)
- **Null safety**: Detects potential null/undefined access
- **Array bounds**: Catches out-of-bounds array access
- **Integer overflow**: Identifies arithmetic overflow risks
- **Division by zero**: Warns about potential divide-by-zero

### AI-Powered Analysis
- **Semantic understanding**: Understands code intent and logic
- **Security vulnerabilities**: OWASP Top 10 and more
- **Code quality**: Best practices and anti-patterns

## Configuration

### Basic Configuration

Create a `.codeverify.yml` file in your repository root:

```yaml
# .codeverify.yml
version: "1"

# Languages to analyze
languages:
  - python
  - typescript

# Files to exclude
exclude:
  - "node_modules/**"
  - "tests/**"
  - "*.test.ts"

# Severity thresholds (max allowed to pass)
thresholds:
  critical: 0
  high: 0
  medium: 5
  low: 10
```

### Full Configuration Reference

```yaml
version: "1"

# Languages (python, typescript, javascript supported)
languages:
  - python
  - typescript

# File patterns to include/exclude
include:
  - "src/**/*.py"
  - "src/**/*.ts"

exclude:
  - "node_modules/**"
  - "venv/**"
  - "dist/**"
  - "*.min.js"
  - "*.generated.*"

# Formal verification settings
verification:
  enabled: true
  timeout: 30  # seconds per check
  checks:
    - null_safety
    - array_bounds
    - integer_overflow
    - division_by_zero

# AI analysis settings
ai:
  enabled: true
  semantic: true   # Code understanding
  security: true   # Vulnerability detection
  model: gpt-4     # or claude-3-sonnet

# Pass/fail thresholds
thresholds:
  critical: 0  # Any critical = fail
  high: 0      # Any high = fail
  medium: 5    # More than 5 medium = fail
  low: 10      # More than 10 low = fail

# Ignore specific findings
ignore:
  - pattern: "migrations/**"
    reason: "Auto-generated migration files"
  - pattern: "tests/**"
    categories:
      - security
    reason: "Test code has different standards"

# Custom rules
rules:
  - id: no-print
    name: "No print statements"
    description: "Use logging instead"
    severity: low
    pattern: "\\bprint\\s*\\("
    enabled: true

# PR behavior
auto_approve: false      # Never auto-approve
comment_on_pass: true    # Comment even when passing
collapse_findings: true  # Collapse details by default
max_inline_comments: 10  # Limit inline comments
```

## Understanding Results

### Severity Levels

| Level | Meaning | Default Threshold |
|-------|---------|-------------------|
| 🔴 **Critical** | Immediate security risk or crash | 0 (any = fail) |
| 🟠 **High** | Significant bug or vulnerability | 0 (any = fail) |
| 🟡 **Medium** | Potential issue worth reviewing | 5 |
| 🔵 **Low** | Minor improvement suggestion | 10 |

### Verification Types

- **`formal`** - Mathematically proven by Z3 solver (highest confidence)
- **`ai`** - Detected by AI analysis (high confidence)
- **`pattern`** - Matched by regex pattern (medium confidence)

### Confidence Scores

Each finding includes a confidence score (0-100%):
- **90-100%**: Very high confidence, likely a real issue
- **70-89%**: High confidence, worth investigating
- **50-69%**: Medium confidence, may be false positive
- **Below 50%**: Lower confidence, review context

## Dashboard

Access your team dashboard at [dashboard.codeverify.io](https://dashboard.codeverify.io):

- **Overview**: Analysis trends and pass rates
- **Repositories**: Configure per-repo settings
- **Analyses**: Browse past PR analyses
- **Findings**: Search and filter all findings
- **Settings**: Team and organization settings

## Integrations

### Slack Notifications

Get notified in Slack when analyses complete:

1. Go to **Settings > Integrations > Slack**
2. Click **Connect to Slack**
3. Select your channel
4. Configure notification preferences

### CI/CD Integration

Use CodeVerify as a required status check:

1. Go to your repository **Settings > Branches**
2. Edit your branch protection rule
3. Enable **Require status checks**
4. Search for and select **CodeVerify**

Now PRs can't merge until CodeVerify passes!

## API Access

For programmatic access, generate an API key:

1. Go to **Settings > API Keys**
2. Click **Generate New Key**
3. Copy and store securely

```bash
# Example: Get analysis results
curl -H "Authorization: Bearer YOUR_API_KEY" \
  https://api.codeverify.io/v1/analyses/ANALYSIS_ID
```

See our [API Documentation](/api-docs) for full reference.

## FAQ

### How long does analysis take?

Most PRs complete in 2-5 minutes. Larger PRs (500+ lines) may take up to 10 minutes.

### Does CodeVerify store my code?

**No.** Code is processed in memory and never persisted. We only store metadata about findings.

### How do I reduce false positives?

1. Add ignore rules for generated code
2. Tune severity thresholds
3. Use inline comments to suppress specific findings:

```python
user = get_user()  # codeverify: ignore null_safety
print(user.name)
```

### What languages are supported?

Currently:
- ✅ Python
- ✅ TypeScript/JavaScript

Coming soon:
- 🔜 Go
- 🔜 Java
- 🔜 Rust

### How do I report a false positive?

Click the 👎 button on any finding to report it. This helps us improve!

## Support

- 📖 [Documentation](https://docs.codeverify.io)
- 💬 [Discord Community](https://discord.gg/codeverify)
- 📧 [Email Support](mailto:support@codeverify.io)
- 🐛 [Report Issues](https://github.com/codeverify/codeverify/issues)

---

Ready to get started? [Install CodeVerify](https://github.com/apps/codeverify) now!
