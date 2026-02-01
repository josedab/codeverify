---
sidebar_position: 1
---

# GitHub Integration

CodeVerify integrates deeply with GitHub for automated code review.

## GitHub App (Recommended)

The easiest way to use CodeVerify with GitHub.

### Installation

1. Visit [github.com/apps/codeverify](https://github.com/apps/codeverify)
2. Click **Install**
3. Select repositories to enable
4. Authorize the permissions

### Permissions Required

| Permission | Reason |
|------------|--------|
| Read code | Analyze your source files |
| Read/Write checks | Report verification results |
| Read/Write pull requests | Comment on PRs |
| Read metadata | Repository information |

### What It Does

Once installed, CodeVerify automatically:

- Analyzes every push to your repository
- Comments on pull requests with findings
- Reports results to GitHub Checks
- Blocks merges when thresholds exceeded (optional)

### PR Comments

CodeVerify adds a summary comment to each PR:

````markdown
## CodeVerify Analysis

✅ **Passed** | Trust Score: 87/100

### Summary
| Check | Status | Findings |
|-------|--------|----------|
| Null Safety | ✅ Pass | 0 |
| Array Bounds | ✅ Pass | 0 |
| Integer Overflow | ⚠️ Warning | 2 |
| Division by Zero | ✅ Pass | 0 |
| AI Security | ✅ Pass | 0 |

### Findings (2)
Expandable section showing:
- ⚠️ Medium: Potential integer overflow in calculate_total()
- File: src/billing.py:45
- Code snippet with the issue
- Recommendation: Use checked arithmetic or validate input bounds
````

### Check Runs

CodeVerify creates GitHub Check Runs that:
- Show inline annotations in the diff
- Appear in the PR's checks section
- Can be required for merge

### Configuration

Add `.codeverify.yml` to your repository:

```yaml
version: "1"

github:
  # Comment on PRs
  comment: true
  
  # Create check runs
  checks: true
  
  # Require checks to pass for merge
  required: true
  
  # Show inline annotations
  annotations: true
  
  # Maximum inline annotations (reduce noise)
  max_annotations: 50
  
  # Comment only when there are findings
  comment_on_pass: false
```

## GitHub Actions

For more control, use the GitHub Action.

### Basic Setup

```yaml
# .github/workflows/codeverify.yml
name: CodeVerify

on:
  push:
    branches: [main]
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

### Full Configuration

```yaml
name: CodeVerify

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for better analysis
      
      - name: Run CodeVerify
        uses: codeverify/action@v1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          
          # API key for AI features
          api_key: ${{ secrets.CODEVERIFY_API_KEY }}
          
          # OpenAI key for AI analysis
          openai_api_key: ${{ secrets.OPENAI_API_KEY }}
          
          # Analysis options
          languages: python,typescript
          checks: null_safety,array_bounds,division_by_zero
          
          # Thresholds
          fail_on: critical,high
          
          # Output
          output_format: sarif
          output_file: codeverify-results.sarif
          
          # PR options
          comment_on_pr: true
          annotations: true
      
      # Upload SARIF for GitHub Code Scanning
      - name: Upload SARIF
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: codeverify-results.sarif
```

### Action Inputs

| Input | Description | Default |
|-------|-------------|---------|
| `github_token` | GitHub token for API access | Required |
| `api_key` | CodeVerify API key | Optional |
| `openai_api_key` | OpenAI API key for AI features | Optional |
| `config_file` | Path to config file | `.codeverify.yml` |
| `languages` | Languages to analyze | Auto-detect |
| `checks` | Verification checks to run | All |
| `fail_on` | Severities that fail the check | `critical,high` |
| `output_format` | Output format | `github` |
| `output_file` | Output file path | None |
| `comment_on_pr` | Comment on pull requests | `true` |
| `annotations` | Show inline annotations | `true` |
| `working_directory` | Directory to analyze | `.` |

### Action Outputs

| Output | Description |
|--------|-------------|
| `exit_code` | 0 for pass, 1 for fail |
| `findings_count` | Total findings found |
| `critical_count` | Critical findings count |
| `high_count` | High findings count |
| `trust_score` | Copilot Trust Score (0-100) |
| `sarif_file` | Path to SARIF output |

### Use in Workflow Conditions

```yaml
jobs:
  verify:
    runs-on: ubuntu-latest
    outputs:
      trust_score: ${{ steps.codeverify.outputs.trust_score }}
    steps:
      - uses: actions/checkout@v4
      - id: codeverify
        uses: codeverify/action@v1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
  
  deploy:
    needs: verify
    if: needs.verify.outputs.trust_score >= 80
    runs-on: ubuntu-latest
    steps:
      - run: echo "Deploying with trust score ${{ needs.verify.outputs.trust_score }}"
```

## GitHub Code Scanning

CodeVerify can upload results to GitHub Code Scanning for a unified security view.

### SARIF Integration

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
    category: codeverify
```

Results appear in the Security tab alongside CodeQL findings.

## Branch Protection

Configure CodeVerify as a required check:

1. Go to **Settings > Branches > Branch protection rules**
2. Click **Add rule** or edit existing
3. Enable **Require status checks to pass**
4. Search for "CodeVerify" and select it
5. Save changes

Now PRs cannot be merged until CodeVerify passes.

## Monorepo Support

For monorepos, configure path-based analysis:

```yaml
# .github/workflows/codeverify.yml
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
      
      - name: Get changed files
        id: changed
        uses: tj-actions/changed-files@v42
        
      - name: Run CodeVerify
        uses: codeverify/action@v1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          files: ${{ steps.changed.outputs.all_changed_files }}
```

## Troubleshooting

### Check Not Appearing

1. Verify the GitHub App is installed
2. Check repository permissions
3. Review Actions workflow logs

### Permission Denied

Ensure your workflow has the required permissions:

```yaml
permissions:
  contents: read
  checks: write
  pull-requests: write
```

### Rate Limits

For large repositories, you may hit GitHub API limits. Solutions:

1. Use a GitHub App installation token (higher limits)
2. Analyze only changed files
3. Cache results between runs

### Debug Mode

Enable verbose logging:

```yaml
- uses: codeverify/action@v1
  with:
    github_token: ${{ secrets.GITHUB_TOKEN }}
  env:
    CODEVERIFY_DEBUG: "true"
```

## Next Steps

- Configure [CI/CD Pipelines](/docs/integrations/ci-cd) for other providers
- Set up [Slack Notifications](/docs/integrations/slack-teams)
- Learn about [VS Code Integration](/docs/integrations/vscode)
