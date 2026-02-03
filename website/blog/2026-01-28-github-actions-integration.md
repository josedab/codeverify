---
slug: integrate-codeverify-github-actions
title: "How to Integrate CodeVerify with GitHub Actions in 5 Minutes"
authors: [codeverify]
tags: [tutorial, github, ci-cd, integration, getting-started]
---

Want automated code verification on every pull request? This tutorial shows you how to set up CodeVerify with GitHub Actions. No complex configuration required—just copy, paste, and push.

<!-- truncate -->

## Prerequisites

- A GitHub repository with Python or TypeScript code
- GitHub Actions enabled (it is by default)
- 5 minutes ⏱️

## Option 1: Use Our Official Action (Recommended)

The fastest way to get started:

```yaml
# .github/workflows/codeverify.yml
name: CodeVerify

on:
  pull_request:
    branches: [main]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run CodeVerify
        uses: codeverify/action@v1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
```

That's it! Push this file and CodeVerify will analyze every PR.

## Option 2: Install from PyPI

More control over the process:

```yaml
# .github/workflows/codeverify.yml
name: CodeVerify

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install CodeVerify
        run: pip install codeverify
      
      - name: Run Analysis
        run: codeverify analyze src/ --format github
```

## Adding AI Analysis

To enable AI-powered semantic and security analysis, add an API key:

```yaml
jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run CodeVerify
        uses: codeverify/action@v1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          openai_api_key: ${{ secrets.OPENAI_API_KEY }}
```

Add `OPENAI_API_KEY` to your repository secrets:
1. Go to Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `OPENAI_API_KEY`, Value: your API key

## Customizing the Analysis

### Analyze Only Changed Files

For faster CI on large repos:

```yaml
- uses: codeverify/action@v1
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    changed_only: true
```

### Specific Checks

Run only certain verification types:

```yaml
- uses: codeverify/action@v1
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    checks: 'null_safety,division_by_zero'
```

### Custom Thresholds

Control when the check fails:

```yaml
- uses: codeverify/action@v1
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    fail_on: 'critical'  # Only fail on critical issues
```

Options: `critical`, `critical,high`, `critical,high,medium`, `none`

## Repository Configuration

Create `.codeverify.yml` in your repo root for persistent settings:

```yaml
version: "1"

languages:
  - python
  - typescript

verification:
  enabled: true
  checks:
    - null_safety
    - array_bounds
    - integer_overflow
    - division_by_zero

ai:
  enabled: true
  semantic: true
  security: true

thresholds:
  critical: 0
  high: 0
  medium: 10
  low: 50

exclude:
  - "tests/**"
  - "docs/**"
  - "**/*.test.ts"
```

## PR Comments and Annotations

CodeVerify automatically adds:

### Summary Comment

On every PR, you'll see:

```markdown
## CodeVerify Analysis

✅ **Passed** | Trust Score: 92/100

### Findings Summary
| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 0 |
| Medium | 2 |
| Low | 5 |

### Details
<details>
<summary>Show 7 findings</summary>
...
</details>
```

### Inline Annotations

Findings appear directly in the PR diff:

![Inline annotation showing array bounds issue](/img/blog/annotation-example.svg)

## Upload to GitHub Code Scanning

For a unified security view:

```yaml
- name: Run CodeVerify
  uses: codeverify/action@v1
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
    output_format: sarif
    output_file: codeverify.sarif

- name: Upload to Code Scanning
  uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: codeverify.sarif
```

Results appear in the Security tab alongside CodeQL findings.

## Branch Protection

Require CodeVerify to pass before merging:

1. Go to Settings → Branches
2. Edit or create a branch protection rule for `main`
3. Enable "Require status checks to pass"
4. Search for "CodeVerify" and select it
5. Save

Now PRs can't be merged until CodeVerify passes!

## Monorepo Setup

For monorepos with multiple packages:

```yaml
name: CodeVerify

on:
  pull_request:
    paths:
      - 'packages/**'
      - 'apps/**'

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Get changed packages
        id: changes
        run: |
          CHANGED=$(git diff --name-only ${{ github.event.pull_request.base.sha }} ${{ github.sha }} | grep -E '^(packages|apps)/' | cut -d'/' -f1,2 | sort -u | tr '\n' ' ')
          echo "packages=$CHANGED" >> $GITHUB_OUTPUT
      
      - name: Run CodeVerify
        uses: codeverify/action@v1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          working_directory: ${{ steps.changes.outputs.packages }}
```

## Caching for Speed

Speed up repeated runs:

```yaml
- name: Cache CodeVerify
  uses: actions/cache@v4
  with:
    path: ~/.cache/codeverify
    key: codeverify-${{ hashFiles('**/*.py', '**/*.ts') }}
    restore-keys: |
      codeverify-

- name: Run CodeVerify
  uses: codeverify/action@v1
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
```

## Troubleshooting

### "Check not appearing"

1. Verify the workflow file is in `.github/workflows/`
2. Check Actions tab for workflow runs
3. Ensure branch protection isn't blocking

### "Permission denied"

Add permissions to your workflow:

```yaml
permissions:
  contents: read
  checks: write
  pull-requests: write
```

### "Analysis taking too long"

- Use `changed_only: true`
- Exclude test files
- Reduce Z3 timeout in config

## What's Next?

Now that you have CI set up:

1. **[Install the VS Code extension](/docs/integrations/vscode)** for real-time verification
2. **[Configure custom rules](/docs/configuration/custom-rules)** for your team
3. **[Set up Slack notifications](/docs/integrations/slack-teams)** for findings

---

*Questions? Join our [Discord](https://discord.gg/codeverify) or open a [GitHub Discussion](https://github.com/codeverify/codeverify/discussions).*
