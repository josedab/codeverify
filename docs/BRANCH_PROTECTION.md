# Branch Protection Rules

This document describes the recommended branch protection rules for the CodeVerify repository.

## Main Branch (`main`)

The `main` branch is the production-ready branch. All changes must go through pull requests.

### Recommended Settings

```yaml
# GitHub Branch Protection Settings for 'main'

# Require a pull request before merging
require_pull_request:
  required_approving_review_count: 1
  dismiss_stale_reviews: true
  require_code_owner_reviews: true
  require_last_push_approval: true

# Require status checks to pass before merging
require_status_checks:
  strict: true  # Require branches to be up to date
  contexts:
    - "Lint"
    - "Test Python"
    - "Test Node.js"
    - "Security Scanning"
    - "Build Docker Images"
    - "Analyze Python"
    - "Analyze JavaScript/TypeScript"

# Require conversation resolution before merging
require_conversation_resolution: true

# Require signed commits
require_signed_commits: false  # Optional, enable if your team uses GPG signing

# Require linear history
require_linear_history: true  # Enforces squash or rebase merges

# Do not allow bypassing the above settings
enforce_admins: true

# Restrict who can push to matching branches
restrict_pushes:
  users: []
  teams:
    - core-team
  apps: []

# Do not allow force pushes
allow_force_pushes: false

# Do not allow deletions
allow_deletions: false
```

## Setting Up Branch Protection

### Via GitHub UI

1. Go to **Settings** → **Branches**
2. Click **Add branch protection rule**
3. Enter `main` as the branch name pattern
4. Configure the following:

#### Pull Request Requirements
- [x] Require a pull request before merging
  - [x] Require approvals: **1**
  - [x] Dismiss stale pull request approvals when new commits are pushed
  - [x] Require review from Code Owners
  - [x] Require approval of the most recent reviewable push

#### Status Checks
- [x] Require status checks to pass before merging
  - [x] Require branches to be up to date before merging
  - Add these required checks:
    - `Lint`
    - `Test Python`
    - `Test Node.js`
    - `Security Scanning`
    - `Build Docker Images`
    - `Analyze Python` (CodeQL)
    - `Analyze JavaScript/TypeScript` (CodeQL)

#### Additional Settings
- [x] Require conversation resolution before merging
- [x] Require linear history
- [x] Do not allow bypassing the above settings
- [ ] Restrict who can push to matching branches (optional)
- [x] Do not allow force pushes
- [x] Do not allow deletions

### Via GitHub CLI

```bash
gh api repos/{owner}/{repo}/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["Lint","Test Python","Test Node.js","Security Scanning","Build Docker Images"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"dismiss_stale_reviews":true,"require_code_owner_reviews":true,"required_approving_review_count":1}' \
  --field restrictions=null \
  --field required_linear_history=true \
  --field allow_force_pushes=false \
  --field allow_deletions=false \
  --field required_conversation_resolution=true
```

## Development Branches

For feature branches (`feature/*`, `fix/*`, etc.), no protection rules are required. These branches should be:

- Short-lived
- Deleted after merging
- Named according to CONTRIBUTING.md guidelines

## Release Branches (Optional)

If using release branches (`release/*`), consider:

```yaml
require_pull_request:
  required_approving_review_count: 2
  require_code_owner_reviews: true

require_status_checks:
  strict: true
  contexts:
    - "Lint"
    - "Test Python"
    - "Test Node.js"
    - "Security Scanning"
    - "Build Docker Images"
```

## Rulesets (GitHub Enterprise)

For GitHub Enterprise users, consider using [Repository Rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets) for more granular control across multiple branches.

## Bypass Permissions

In exceptional circumstances, repository administrators may need to bypass protections. This should be:

1. Documented in the PR description
2. Used only for critical hotfixes
3. Followed by a post-incident review

## Verifying Protection

Check current protection status:

```bash
gh api repos/{owner}/{repo}/branches/main/protection
```

## References

- [GitHub Branch Protection Documentation](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [GitHub CLI Branch Protection](https://cli.github.com/manual/gh_api)
