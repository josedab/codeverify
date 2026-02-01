---
sidebar_position: 3
---

# Bitbucket Integration

Integrate CodeVerify with Bitbucket Pipelines.

## Bitbucket Pipelines

### Basic Setup

Add to your `bitbucket-pipelines.yml`:

```yaml
image: codeverify/cli:latest

pipelines:
  pull-requests:
    '**':
      - step:
          name: CodeVerify Analysis
          script:
            - codeverify analyze .
            
  branches:
    main:
      - step:
          name: CodeVerify Analysis
          script:
            - codeverify analyze .
```

### Full Configuration

```yaml
image: codeverify/cli:latest

definitions:
  steps:
    - step: &codeverify
        name: CodeVerify Analysis
        script:
          - |
            codeverify analyze . \
              --format bitbucket \
              --bitbucket-report \
              --output-dir reports
        artifacts:
          - reports/**

pipelines:
  pull-requests:
    '**':
      - step: *codeverify
      
  branches:
    main:
      - step:
          <<: *codeverify
          script:
            - codeverify analyze . --strict
```

### PR Comments

Enable pull request comments:

```yaml
pipelines:
  pull-requests:
    '**':
      - step:
          name: CodeVerify
          script:
            - |
              codeverify analyze . \
                --bitbucket-comment \
                --bitbucket-workspace $BITBUCKET_WORKSPACE \
                --bitbucket-repo $BITBUCKET_REPO_SLUG \
                --bitbucket-pr $BITBUCKET_PR_ID
```

Required variables (set in Repository Settings > Pipelines > Variables):
- `BITBUCKET_TOKEN`: App password with PR write access

### Code Insights

CodeVerify can create Bitbucket Code Insights reports:

```yaml
- step:
    name: CodeVerify
    script:
      - |
        codeverify analyze . \
          --bitbucket-insights \
          --bitbucket-commit $BITBUCKET_COMMIT
```

This creates annotations visible in the PR diff view.

## Environment Variables

Set these in **Repository Settings > Pipelines > Variables**:

| Variable | Required | Description |
|----------|----------|-------------|
| `CODEVERIFY_API_KEY` | No | CodeVerify cloud API key |
| `OPENAI_API_KEY` | For AI | OpenAI API key |
| `BITBUCKET_TOKEN` | For comments | Bitbucket app password |

## Bitbucket Cloud vs Server

### Bitbucket Cloud

Uses the standard configuration above.

### Bitbucket Server (Data Center)

```yaml
- step:
    name: CodeVerify
    script:
      - |
        codeverify analyze . \
          --bitbucket-server \
          --bitbucket-url https://bitbucket.company.com \
          --bitbucket-token $BITBUCKET_TOKEN
```

## Caching

```yaml
definitions:
  caches:
    codeverify: .codeverify/cache

pipelines:
  pull-requests:
    '**':
      - step:
          name: CodeVerify
          caches:
            - codeverify
          script:
            - codeverify analyze . --cache-dir .codeverify/cache
```

## Build Status

CodeVerify sets the build status based on findings:

```yaml
- step:
    name: CodeVerify
    script:
      - codeverify analyze . --fail-on critical,high
```

- Exit code 0: Pass (no findings above threshold)
- Exit code 1: Fail (findings exceed threshold)

## Parallel Analysis

For large repositories:

```yaml
pipelines:
  pull-requests:
    '**':
      - parallel:
          - step:
              name: CodeVerify - Python
              script:
                - codeverify analyze . --languages python
          - step:
              name: CodeVerify - TypeScript  
              script:
                - codeverify analyze . --languages typescript
```

## Troubleshooting

### No PR Comments

1. Verify `BITBUCKET_TOKEN` is set
2. Check token has PR write permissions
3. Ensure PR ID is available (`$BITBUCKET_PR_ID`)

### Pipeline Variables

Built-in variables available:
- `$BITBUCKET_WORKSPACE` — Workspace name
- `$BITBUCKET_REPO_SLUG` — Repository slug
- `$BITBUCKET_PR_ID` — Pull request ID
- `$BITBUCKET_COMMIT` — Commit SHA
- `$BITBUCKET_BRANCH` — Branch name
