---
sidebar_position: 4
---

# AI Settings

Configure CodeVerify's AI-powered analysis.

## Basic Configuration

```yaml
ai:
  enabled: true
  provider: "openai"
  model: "gpt-4"
```

## Providers

### OpenAI

```yaml
ai:
  provider: "openai"
  model: "gpt-4"         # or gpt-4-turbo, gpt-3.5-turbo
```

Models:
- **gpt-4** — Best accuracy, recommended for production
- **gpt-4-turbo** — Faster, similar quality
- **gpt-3.5-turbo** — Fastest, lower cost, reduced accuracy

### Anthropic

```yaml
ai:
  provider: "anthropic"
  model: "claude-3-opus"  # or claude-3-sonnet, claude-3-haiku
```

Models:
- **claude-3-opus** — Highest quality reasoning
- **claude-3-sonnet** — Good balance of speed and quality
- **claude-3-haiku** — Fastest, good for high volume

### Azure OpenAI

```yaml
ai:
  provider: "azure"
  deployment: "my-gpt4-deployment"
  api_base: "https://my-resource.openai.azure.com/"
  api_version: "2024-02-01"
```

### Local Models

```yaml
ai:
  provider: "ollama"
  model: "codellama:34b"
  endpoint: "http://localhost:11434"
```

Or custom OpenAI-compatible endpoints:

```yaml
ai:
  provider: "custom"
  endpoint: "http://localhost:8080/v1"
  model: "my-model"
```

## API Keys

### Environment Variables (Recommended)

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export AZURE_OPENAI_API_KEY="..."
```

### Configuration File

Create `~/.codeverify/credentials.yml`:

```yaml
providers:
  openai:
    api_key: "sk-..."
  anthropic:
    api_key: "sk-ant-..."
```

:::warning
Never commit API keys to version control. Use environment variables in CI.
:::

## Analysis Features

### Semantic Analysis

Understands code meaning, not just syntax:

```yaml
ai:
  semantic:
    enabled: true
    
    # Detect code smell patterns
    code_smells: true
    
    # Analyze complexity and suggest simplifications
    complexity_analysis: true
    
    # Identify dead code
    dead_code_detection: true
    
    # Check for proper resource cleanup
    resource_management: true
```

### Security Analysis

AI-powered vulnerability detection:

```yaml
ai:
  security:
    enabled: true
    
    # Check for injection vulnerabilities
    injection_detection: true
    
    # Identify authentication issues
    auth_analysis: true
    
    # Detect sensitive data exposure
    data_exposure: true
    
    # Check for insecure dependencies
    dependency_audit: true
```

### Pattern Matching

Match known vulnerability patterns:

```yaml
ai:
  patterns:
    enabled: true
    
    # Use built-in patterns
    builtin: true
    
    # Custom pattern files
    custom:
      - ".codeverify/patterns/*.yml"
```

## Trust Score Configuration

Configure how Copilot Trust Score is calculated:

```yaml
ai:
  trust_score:
    enabled: true
    
    # Minimum score to pass (0-100)
    threshold: 70
    
    # Weight factors
    weights:
      formal_verification: 0.4
      ai_confidence: 0.3
      pattern_coverage: 0.2
      historical_accuracy: 0.1
```

## Response Configuration

```yaml
ai:
  response:
    # Maximum tokens in AI response
    max_tokens: 2000
    
    # Temperature (0 = deterministic, 1 = creative)
    temperature: 0.1
    
    # Request timeout (seconds)
    timeout: 60
    
    # Retry failed requests
    retries: 3
```

### Temperature Guidelines

| Value | Use Case |
|-------|----------|
| 0.0 | Most deterministic, for CI |
| 0.1-0.2 | Recommended for analysis |
| 0.3-0.5 | More varied suggestions |
| >0.5 | Not recommended for code analysis |

## Rate Limiting

Control API usage:

```yaml
ai:
  rate_limits:
    # Maximum requests per minute
    requests_per_minute: 60
    
    # Maximum tokens per minute
    tokens_per_minute: 90000
    
    # Concurrent requests
    max_concurrent: 5
    
    # Backoff strategy: exponential or linear
    backoff: exponential
```

## Caching

Cache AI responses to reduce costs and improve speed:

```yaml
ai:
  cache:
    enabled: true
    
    # Cache location
    path: ".codeverify/cache"
    
    # Cache TTL (hours)
    ttl: 24
    
    # Maximum cache size (MB)
    max_size: 500
```

Cache hits occur when:
- Same code content
- Same configuration
- Cache not expired

## Context Configuration

Control what context is sent to the AI:

```yaml
ai:
  context:
    # Include surrounding code
    window_size: 50  # lines before/after
    
    # Include imports/dependencies
    include_imports: true
    
    # Include type information
    include_types: true
    
    # Include docstrings
    include_docs: true
    
    # Include git history context
    include_git_context: false
```

## Privacy Settings

```yaml
ai:
  privacy:
    # Strip comments before sending
    strip_comments: false
    
    # Anonymize variable names
    anonymize: false
    
    # Files to never send to AI
    exclude:
      - "**/*.env"
      - "**/*credentials*"
      - "**/secrets/**"
```

## Provider-Specific Settings

### OpenAI

```yaml
ai:
  openai:
    organization: "org-..."
    project: "proj-..."
    timeout: 120
    max_retries: 3
```

### Anthropic

```yaml
ai:
  anthropic:
    max_tokens: 4096
    timeout: 60
```

### Azure

```yaml
ai:
  azure:
    deployment: "gpt-4-deployment"
    api_base: "https://resource.openai.azure.com/"
    api_version: "2024-02-01"
```

## Cost Management

### Estimate Costs Before Analysis

```bash
codeverify analyze src/ --dry-run --estimate-cost
```

Output:
```
Estimated API calls: 150
Estimated tokens: 450,000
Estimated cost: $6.75 (GPT-4)
Estimated cost: $0.45 (GPT-3.5-turbo)
```

### Cost Limits

```yaml
ai:
  cost:
    # Stop if estimated cost exceeds
    max_cost_per_run: 10.00
    
    # Warning threshold
    warn_cost: 5.00
    
    # Currency for display
    currency: "USD"
```

## Fallback Configuration

Handle provider failures gracefully:

```yaml
ai:
  fallback:
    enabled: true
    
    # Fallback order
    providers:
      - openai
      - anthropic
      - ollama
    
    # Only fallback for specific errors
    on_errors:
      - rate_limit
      - timeout
      - unavailable
```

## Examples

### Budget-Conscious Setup

```yaml
ai:
  provider: "openai"
  model: "gpt-3.5-turbo"
  
  cache:
    enabled: true
    ttl: 48
    
  rate_limits:
    requests_per_minute: 30
    
  cost:
    max_cost_per_run: 1.00
```

### High-Security Project

```yaml
ai:
  provider: "anthropic"
  model: "claude-3-opus"
  
  security:
    enabled: true
    injection_detection: true
    auth_analysis: true
    data_exposure: true
    
  privacy:
    exclude:
      - "**/*.env*"
      - "**/config/secrets/**"
      
  trust_score:
    threshold: 85
```

### Enterprise Setup

```yaml
ai:
  provider: "azure"
  deployment: "gpt-4-enterprise"
  api_base: "https://company.openai.azure.com/"
  
  cache:
    enabled: true
    
  rate_limits:
    requests_per_minute: 100
    max_concurrent: 10
    
  fallback:
    enabled: true
    providers:
      - azure
      - openai
```
