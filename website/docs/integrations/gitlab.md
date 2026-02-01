---
sidebar_position: 2
---

# GitLab Integration

Integrate CodeVerify with GitLab CI/CD pipelines.

## GitLab CI/CD

### Basic Setup

Add to your `.gitlab-ci.yml`:

```yaml
codeverify:
  image: codeverify/cli:latest
  stage: test
  script:
    - codeverify analyze .
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
```

### Full Configuration

```yaml
stages:
  - test
  - verify
  - deploy

variables:
  CODEVERIFY_API_KEY: $CODEVERIFY_API_KEY
  OPENAI_API_KEY: $OPENAI_API_KEY

codeverify:
  image: codeverify/cli:latest
  stage: verify
  script:
    - codeverify analyze . --format gitlab --output gl-codeverify-report.json
  artifacts:
    reports:
      codequality: gl-codeverify-report.json
    paths:
      - gl-codeverify-report.json
    expire_in: 1 week
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
  allow_failure: false
```

### Merge Request Comments

Enable MR comments by adding the GitLab integration:

```yaml
codeverify:
  image: codeverify/cli:latest
  stage: verify
  variables:
    GITLAB_TOKEN: $GITLAB_TOKEN
  script:
    - |
      codeverify analyze . \
        --gitlab-mr-comment \
        --gitlab-project-id $CI_PROJECT_ID \
        --gitlab-mr-iid $CI_MERGE_REQUEST_IID
```

Required variables:
- `GITLAB_TOKEN`: Personal access token with `api` scope
- CI variables are automatically available

### Code Quality Report

GitLab displays CodeVerify findings in the Code Quality widget:

```yaml
codeverify:
  image: codeverify/cli:latest
  stage: verify
  script:
    - codeverify analyze . --format gitlab-codequality --output gl-code-quality-report.json
  artifacts:
    reports:
      codequality: gl-code-quality-report.json
```

Findings appear inline in the MR diff view.

## GitLab Self-Hosted

For GitLab self-managed instances:

```yaml
variables:
  GITLAB_URL: "https://gitlab.company.com"
  
codeverify:
  image: codeverify/cli:latest
  script:
    - |
      codeverify analyze . \
        --gitlab-url $GITLAB_URL \
        --gitlab-token $GITLAB_TOKEN
```

## Pipeline Conditions

### Only on Changes

```yaml
codeverify:
  image: codeverify/cli:latest
  script:
    - codeverify analyze . --changed-since $CI_MERGE_REQUEST_DIFF_BASE_SHA
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
      changes:
        - "**/*.py"
        - "**/*.ts"
        - "**/*.js"
```

### Branch-Specific Thresholds

```yaml
codeverify:main:
  image: codeverify/cli:latest
  script:
    - codeverify analyze . --strict
  rules:
    - if: $CI_COMMIT_BRANCH == "main"

codeverify:develop:
  image: codeverify/cli:latest
  script:
    - codeverify analyze . --warn-only
  rules:
    - if: $CI_COMMIT_BRANCH == "develop"
```

## Caching

Speed up repeated analyses:

```yaml
codeverify:
  image: codeverify/cli:latest
  cache:
    key: codeverify-$CI_COMMIT_REF_SLUG
    paths:
      - .codeverify/cache/
  script:
    - codeverify analyze . --cache-dir .codeverify/cache
```

## Required Configuration

Set these variables in **Settings > CI/CD > Variables**:

| Variable | Required | Description |
|----------|----------|-------------|
| `CODEVERIFY_API_KEY` | No | CodeVerify cloud API key |
| `OPENAI_API_KEY` | For AI | OpenAI API key |
| `GITLAB_TOKEN` | For comments | GitLab API token |

## SAST Integration

CodeVerify can complement GitLab SAST:

```yaml
include:
  - template: Security/SAST.gitlab-ci.yml

codeverify:
  stage: test
  image: codeverify/cli:latest
  script:
    - codeverify analyze . --format sarif --output gl-codeverify.sarif
  artifacts:
    reports:
      sast: gl-codeverify.sarif
```

## Troubleshooting

### Report Not Showing

1. Verify artifact is uploaded correctly
2. Check report format matches expected schema
3. Ensure job doesn't fail before artifact upload

### Token Permissions

GitLab token needs:
- `api` scope for MR comments
- `read_repository` for code access

### Debug Mode

```yaml
codeverify:
  script:
    - codeverify analyze . --verbose
  variables:
    CODEVERIFY_DEBUG: "true"
```
