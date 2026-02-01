# Custom Rules Guide

CodeVerify supports custom rules for organization-specific coding standards.

## Overview

Custom rules allow you to:
- Enforce team conventions
- Catch domain-specific anti-patterns
- Extend beyond built-in checks
- Create no-code rules without writing plugins

## Rule Types

### Pattern Rules (Regex)

Simple text pattern matching:

```yaml
rules:
  - id: no-console-log
    type: pattern
    name: "No console.log"
    description: "Use structured logging instead"
    severity: low
    languages: [javascript, typescript]
    pattern: "console\\.log\\s*\\("
```

### AST Rules

Match specific code structures:

```yaml
rules:
  - id: no-any-type
    type: ast
    name: "No any type"
    description: "Avoid using 'any' type in TypeScript"
    severity: medium
    languages: [typescript]
    ast_pattern:
      type: TSAnyKeyword
```

### Semantic Rules (AI-powered)

Use AI to understand code meaning:

```yaml
rules:
  - id: hardcoded-secrets
    type: semantic
    name: "Hardcoded secrets"
    description: "Detect hardcoded API keys, passwords, etc."
    severity: critical
    prompt: |
      Analyze this code for hardcoded secrets, API keys, passwords, 
      or sensitive credentials. Flag any string literals that appear 
      to be secrets.
```

### Composite Rules

Combine multiple rules:

```yaml
rules:
  - id: security-bundle
    type: composite
    name: "Security checks"
    operator: OR  # OR = any match, AND = all must match
    rules:
      - hardcoded-secrets
      - sql-injection
      - xss-vulnerability
```

## Configuration

### In `.codeverify.yml`

```yaml
version: "1"

# Custom rules inline
rules:
  - id: no-print
    type: pattern
    name: "No print statements"
    pattern: "\\bprint\\s*\\("
    severity: low
    languages: [python]

# Or reference external file
custom_rules_file: .codeverify-rules.yml
```

### External Rules File

```yaml
# .codeverify-rules.yml
rules:
  - id: team-naming
    type: pattern
    name: "Team naming convention"
    description: "Functions must use snake_case"
    pattern: "def [A-Z]"
    severity: low
    languages: [python]
    
  - id: no-magic-numbers
    type: ast
    name: "No magic numbers"
    ast_pattern:
      type: NumericLiteral
      exclude:
        - value: 0
        - value: 1
        - value: -1
```

## Rule Schema

### Required Fields

```yaml
- id: unique-rule-id       # Unique identifier
  type: pattern|ast|semantic|composite
  name: "Human readable name"
  severity: critical|high|medium|low
```

### Optional Fields

```yaml
  description: "Detailed explanation"
  languages: [python, typescript]  # Limit to languages
  include:                          # File patterns to check
    - "src/**/*"
  exclude:                          # File patterns to skip
    - "tests/**"
  enabled: true                     # Enable/disable
  fix_suggestion: "Use X instead"   # Suggested fix text
  references:                       # Documentation links
    - https://example.com/best-practices
```

## Pattern Rules

### Basic Patterns

```yaml
# Match function calls
pattern: "eval\\s*\\("

# Match imports
pattern: "from\\s+os\\s+import"

# Match TODO comments
pattern: "(TODO|FIXME|HACK):"
```

### Advanced Patterns

```yaml
# Multi-line patterns (use DOTALL flag)
pattern: "try:\\s*[^:]+:\\s*pass"
flags: DOTALL

# Word boundaries
pattern: "\\bpassword\\b"

# Capture groups for context
pattern: "(password|secret|api_key)\\s*=\\s*['\"][^'\"]+['\"]"
```

## AST Rules

### Node Types

Common AST node types:

| Language | Node Type | Description |
|----------|-----------|-------------|
| Python | `FunctionDef` | Function definition |
| Python | `Call` | Function call |
| Python | `Import` | Import statement |
| TypeScript | `FunctionDeclaration` | Function |
| TypeScript | `CallExpression` | Function call |
| TypeScript | `TSAnyKeyword` | `any` type |

### Pattern Matching

```yaml
# Match function with specific name
ast_pattern:
  type: FunctionDef
  name: "^test_"  # Regex for name

# Match call with arguments
ast_pattern:
  type: Call
  function: "execute"
  arguments:
    min: 1
    max: 1
    
# Match with nested structure
ast_pattern:
  type: FunctionDef
  body:
    contains:
      type: Return
      value: null
```

## Semantic Rules

### Writing Prompts

```yaml
- id: business-logic-check
  type: semantic
  prompt: |
    You are reviewing code for a financial application.
    Check for:
    1. Missing input validation on monetary amounts
    2. Incorrect rounding operations
    3. Race conditions in balance updates
    
    Respond with findings in JSON format:
    {
      "findings": [
        {
          "title": "...",
          "description": "...",
          "line": N,
          "severity": "high"
        }
      ]
    }
```

### Configuration

```yaml
- id: architecture-check
  type: semantic
  model: gpt-4  # Model to use
  temperature: 0.1  # Lower = more deterministic
  max_tokens: 1000
  prompt: |
    Check if this code follows hexagonal architecture...
```

## CLI Usage

### Test Rules

```bash
# Test rule against file
codeverify test-rule my-rule.yml -t src/example.py

# Test all rules
codeverify test-rule .codeverify-rules.yml
```

### List Rules

```bash
# List all rules (built-in + custom)
codeverify list-rules

# Filter by type
codeverify list-rules --type pattern
```

### Run Specific Rules

```bash
# Run only custom rules
codeverify analyze --rules-only custom

# Run specific rule
codeverify analyze --rule no-print
```

## Built-in Templates

CodeVerify includes rule templates:

```bash
# List templates
codeverify rules templates

# Create rule from template
codeverify rules create --template no-eval
```

### Available Templates

| Template | Description |
|----------|-------------|
| `no-eval` | Block eval() usage |
| `no-exec` | Block exec() usage |
| `no-any` | Block TypeScript any |
| `sql-string` | Detect SQL string concatenation |
| `hardcoded-ip` | Detect hardcoded IP addresses |
| `todo-comment` | Flag TODO comments |

## Sharing Rules

### Organization Rules

Share rules across repositories:

1. Create a rules repository
2. Reference in `.codeverify.yml`:
   ```yaml
   extends:
     - https://github.com/org/codeverify-rules/main/.codeverify-rules.yml
   ```

### Publishing Rules

Publish rules to the community:

1. Create rule file
2. Add tests
3. Submit to [CodeVerify Rules Registry](https://github.com/codeverify/rules)

## Best Practices

1. **Start simple** - Begin with pattern rules, add complexity as needed
2. **Test thoroughly** - Include positive and negative test cases
3. **Document why** - Explain the reasoning in descriptions
4. **Set appropriate severity** - Don't mark everything as critical
5. **Use exclusions wisely** - Exclude generated/test code where appropriate
6. **Version your rules** - Track changes to custom rules

## Examples

### Python Best Practices

```yaml
rules:
  - id: no-bare-except
    type: ast
    name: "No bare except"
    severity: medium
    languages: [python]
    ast_pattern:
      type: ExceptHandler
      exception_type: null
      
  - id: use-pathlib
    type: pattern
    name: "Use pathlib"
    description: "Use pathlib instead of os.path"
    severity: low
    languages: [python]
    pattern: "import os\\.path|from os import path"
```

### TypeScript Best Practices

```yaml
rules:
  - id: prefer-const
    type: ast
    name: "Prefer const"
    severity: low
    languages: [typescript]
    ast_pattern:
      type: VariableDeclaration
      kind: let
      reassigned: false

  - id: no-non-null-assertion
    type: pattern
    name: "No non-null assertion"
    severity: medium
    languages: [typescript]
    pattern: "!\\."
```

### Security Rules

```yaml
rules:
  - id: no-http
    type: pattern
    name: "No HTTP URLs"
    severity: high
    pattern: "http://"
    exclude:
      - "localhost"
      - "127.0.0.1"
      
  - id: validate-input
    type: semantic
    name: "Validate user input"
    severity: critical
    prompt: |
      Check if user input from request parameters, query strings, 
      or form data is properly validated before use...
```
