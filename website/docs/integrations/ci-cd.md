---
sidebar_position: 5
---

# CI/CD Pipelines

Integrate CodeVerify with any CI/CD platform.

## Generic Setup

CodeVerify works with any CI system that can run Docker or install Python packages.

### Docker

```yaml
# Use the official image
image: codeverify/cli:latest

steps:
  - codeverify analyze .
```

### pip Install

```bash
pip install codeverify
codeverify analyze .
```

## Common Patterns

### Exit Codes

CodeVerify returns appropriate exit codes:

| Code | Meaning |
|------|---------|
| 0 | Success, no findings above threshold |
| 1 | Findings exceed configured threshold |
| 2 | Configuration or setup error |

### Output Formats

```bash
# Human-readable (default)
codeverify analyze . --format text

# JSON for parsing
codeverify analyze . --format json --output results.json

# SARIF for security tools
codeverify analyze . --format sarif --output results.sarif

# JUnit for test reporters
codeverify analyze . --format junit --output results.xml
```

### Environment Variables

```bash
export CODEVERIFY_API_KEY="cv_..."
export OPENAI_API_KEY="sk-..."
export CODEVERIFY_CONFIG=".codeverify.yml"
```

## Jenkins

### Jenkinsfile (Declarative)

```groovy
pipeline {
    agent {
        docker { image 'codeverify/cli:latest' }
    }
    
    environment {
        CODEVERIFY_API_KEY = credentials('codeverify-api-key')
        OPENAI_API_KEY = credentials('openai-api-key')
    }
    
    stages {
        stage('Verify') {
            steps {
                sh 'codeverify analyze . --format junit --output codeverify-results.xml'
            }
            post {
                always {
                    junit 'codeverify-results.xml'
                }
            }
        }
    }
}
```

### Jenkinsfile (Scripted)

```groovy
node {
    docker.image('codeverify/cli:latest').inside {
        stage('Checkout') {
            checkout scm
        }
        
        stage('Verify') {
            withCredentials([
                string(credentialsId: 'codeverify-api-key', variable: 'CODEVERIFY_API_KEY')
            ]) {
                sh 'codeverify analyze . --format junit --output results.xml'
            }
            junit 'results.xml'
        }
    }
}
```

## CircleCI

### .circleci/config.yml

```yaml
version: 2.1

orbs:
  codeverify: codeverify/verify@1.0

jobs:
  verify:
    docker:
      - image: codeverify/cli:latest
    steps:
      - checkout
      - run:
          name: Run CodeVerify
          command: |
            codeverify analyze . \
              --format junit \
              --output test-results/codeverify.xml
      - store_test_results:
          path: test-results

workflows:
  main:
    jobs:
      - verify
```

### With Caching

```yaml
jobs:
  verify:
    docker:
      - image: codeverify/cli:latest
    steps:
      - checkout
      - restore_cache:
          keys:
            - codeverify-cache-{{ checksum ".codeverify.yml" }}
      - run:
          name: Run CodeVerify
          command: codeverify analyze . --cache-dir .cache
      - save_cache:
          key: codeverify-cache-{{ checksum ".codeverify.yml" }}
          paths:
            - .cache
```

## Travis CI

### .travis.yml

```yaml
language: python
python: "3.11"

install:
  - pip install codeverify

script:
  - codeverify analyze .

env:
  global:
    - secure: "ENCRYPTED_CODEVERIFY_API_KEY"
    - secure: "ENCRYPTED_OPENAI_API_KEY"
```

### With Matrix

```yaml
jobs:
  include:
    - stage: verify
      name: "CodeVerify - Python"
      script: codeverify analyze . --languages python
      
    - stage: verify
      name: "CodeVerify - TypeScript"
      script: codeverify analyze . --languages typescript
```

## Azure Pipelines

### azure-pipelines.yml

```yaml
trigger:
  - main

pool:
  vmImage: 'ubuntu-latest'

steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: '3.11'
      
  - script: |
      pip install codeverify
      codeverify analyze . --format junit --output $(Build.ArtifactStagingDirectory)/codeverify.xml
    displayName: 'Run CodeVerify'
    env:
      CODEVERIFY_API_KEY: $(CodeVerifyApiKey)
      OPENAI_API_KEY: $(OpenAiApiKey)
      
  - task: PublishTestResults@2
    inputs:
      testResultsFormat: 'JUnit'
      testResultsFiles: '$(Build.ArtifactStagingDirectory)/codeverify.xml'
```

### With Docker

```yaml
steps:
  - task: Docker@2
    inputs:
      command: 'run'
      arguments: '-v $(Build.SourcesDirectory):/code codeverify/cli:latest analyze /code'
```

## AWS CodeBuild

### buildspec.yml

```yaml
version: 0.2

env:
  secrets-manager:
    CODEVERIFY_API_KEY: codeverify:api_key
    OPENAI_API_KEY: codeverify:openai_key

phases:
  install:
    runtime-versions:
      python: 3.11
    commands:
      - pip install codeverify
      
  build:
    commands:
      - codeverify analyze . --format json --output codeverify-results.json
      
reports:
  codeverify:
    files:
      - codeverify-results.json
    file-format: GENERICJSON
```

## Google Cloud Build

### cloudbuild.yaml

```yaml
steps:
  - name: 'codeverify/cli:latest'
    args: ['analyze', '.', '--format', 'json', '--output', 'results.json']
    secretEnv: ['CODEVERIFY_API_KEY', 'OPENAI_API_KEY']

availableSecrets:
  secretManager:
    - versionName: projects/$PROJECT_ID/secrets/codeverify-api-key/versions/latest
      env: 'CODEVERIFY_API_KEY'
    - versionName: projects/$PROJECT_ID/secrets/openai-api-key/versions/latest
      env: 'OPENAI_API_KEY'
```

## Drone CI

### .drone.yml

```yaml
kind: pipeline
type: docker
name: default

steps:
  - name: codeverify
    image: codeverify/cli:latest
    environment:
      CODEVERIFY_API_KEY:
        from_secret: codeverify_api_key
    commands:
      - codeverify analyze .
```

## TeamCity

### Build Step

```kotlin
// build.gradle.kts
buildType {
    steps {
        script {
            name = "CodeVerify"
            scriptContent = """
                pip install codeverify
                codeverify analyze . --format junit --output codeverify-results.xml
            """.trimIndent()
        }
    }
    
    features {
        xmlReport {
            reportType = XMLReport.XmlReportType.JUNIT
            rules = "+:codeverify-results.xml"
        }
    }
}
```

## Best Practices

### Fail Fast

Configure appropriate thresholds:

```bash
# Fail on critical/high, warn on medium/low
codeverify analyze . --fail-on critical,high
```

### Cache Results

Cache the analysis cache directory:

```bash
codeverify analyze . --cache-dir /cache/codeverify
```

### Analyze Only Changed Files

For faster PR checks:

```bash
# Git diff-based
codeverify analyze . --changed-since origin/main

# Or specify files
codeverify analyze $(git diff --name-only origin/main)
```

### Parallel Analysis

Split by language or directory:

```yaml
parallel:
  - codeverify analyze src/python --languages python
  - codeverify analyze src/typescript --languages typescript
```

### Artifacts

Always save results as artifacts:

```bash
codeverify analyze . \
  --format json --output results.json \
  --format sarif --output results.sarif
```

## Troubleshooting

### Permission Issues

Ensure the CI user has read access to source files:

```bash
chmod -R +r .
```

### Network Issues

For self-hosted runners without internet:

```bash
# Use offline mode (no AI features)
codeverify analyze . --offline
```

### Memory Limits

For large codebases:

```yaml
# Increase container memory
resources:
  limits:
    memory: 4Gi
```

Or limit concurrency:

```bash
codeverify analyze . --max-workers 2
```
