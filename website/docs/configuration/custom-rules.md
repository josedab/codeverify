---
sidebar_position: 5
---

# Custom Rules

Extend CodeVerify with your own verification rules.

## Overview

Custom rules let you:
- Enforce project-specific conventions
- Detect domain-specific antipatterns
- Add organization security requirements
- Create team coding standards

## Rule File Structure

Create rules in `.codeverify/rules/`:

```
.codeverify/
└── rules/
    ├── security.yml
    ├── conventions.yml
    └── domain-rules.yml
```

## Basic Rule Syntax

```yaml
# .codeverify/rules/security.yml
rules:
  - id: no-hardcoded-secrets
    name: "No Hardcoded Secrets"
    description: "Detect hardcoded API keys, passwords, and tokens"
    severity: critical
    category: security
    languages:
      - python
      - typescript
    pattern: |
      (api_key|api_secret|password|token|secret)\s*=\s*["'][^"']+["']
    message: "Hardcoded secret detected. Use environment variables instead."
    fix_suggestion: "Use os.environ.get('SECRET_NAME') or process.env.SECRET_NAME"
```

## Rule Properties

| Property | Required | Description |
|----------|----------|-------------|
| `id` | Yes | Unique identifier (snake_case) |
| `name` | Yes | Human-readable name |
| `description` | No | Detailed description |
| `severity` | Yes | critical, high, medium, low |
| `category` | Yes | security, correctness, performance, style |
| `languages` | No | Limit to specific languages |
| `pattern` | Yes* | Regex pattern to match |
| `ast` | Yes* | AST pattern to match |
| `message` | Yes | Error message shown to user |
| `fix_suggestion` | No | How to fix the issue |

*Either `pattern` or `ast` is required.

## Pattern Rules

### Simple Regex

```yaml
rules:
  - id: no-console-log
    name: "No Console.log"
    severity: low
    category: style
    languages:
      - typescript
      - javascript
    pattern: console\.log\(
    message: "Remove console.log before committing"
```

### Multi-line Patterns

```yaml
rules:
  - id: missing-error-handling
    name: "Missing Error Handling"
    severity: high
    category: correctness
    pattern: |
      (?s)await\s+\w+\([^)]*\)(?![\s\S]*?\.catch|[\s\S]*?try)
    message: "Async call without error handling"
```

### Capture Groups

```yaml
rules:
  - id: deprecated-function
    name: "Deprecated Function Usage"
    severity: medium
    category: correctness
    pattern: (\w+)\.deprecated_method\(
    message: "'{capture_1}' uses deprecated method. Use new_method() instead."
```

## AST Rules

Match code structure, not just text:

### Python AST

```yaml
rules:
  - id: mutable-default-argument
    name: "Mutable Default Argument"
    severity: high
    category: correctness
    languages:
      - python
    ast:
      type: FunctionDef
      defaults:
        - type: [List, Dict, Set]
    message: "Mutable default argument. Use None and create inside function."
    fix_suggestion: |
      def func(items=None):
          if items is None:
              items = []
```

### TypeScript AST

```yaml
rules:
  - id: any-type-usage
    name: "Avoid 'any' Type"
    severity: medium
    category: style
    languages:
      - typescript
    ast:
      type: TypeAnnotation
      typeAnnotation:
        type: TSAnyKeyword
    message: "Avoid using 'any'. Use a specific type or 'unknown'."
```

## Semantic Rules

Rules that understand code meaning:

```yaml
rules:
  - id: sql-string-concat
    name: "SQL String Concatenation"
    severity: critical
    category: security
    semantic:
      function_calls:
        - execute
        - query
        - raw
      argument_contains: "+"
      argument_type: string
    message: "Potential SQL injection. Use parameterized queries."
    fix_suggestion: |
      # Instead of:
      cursor.execute("SELECT * FROM users WHERE id = " + user_id)
      
      # Use:
      cursor.execute("SELECT * FROM users WHERE id = ?", [user_id])
```

## Composite Rules

Combine multiple conditions:

```yaml
rules:
  - id: unvalidated-user-input
    name: "Unvalidated User Input"
    severity: critical
    category: security
    composite:
      all:
        - source:
            type: parameter
            annotation_not: [Validated, Sanitized]
        - sink:
            function: [execute, eval, subprocess.run, os.system]
    message: "User input reaches dangerous function without validation"
```

## Context-Aware Rules

```yaml
rules:
  - id: missing-auth-check
    name: "Missing Authentication Check"
    severity: high
    category: security
    context:
      # Only apply to API handlers
      file_pattern: "**/api/**/*.py"
      function_decorator: ["route", "api_view", "endpoint"]
      must_contain: ["current_user", "require_auth", "authenticate"]
    message: "API endpoint missing authentication check"
```

## Rule Sets

Group related rules:

```yaml
# .codeverify/rules/rule-sets.yml
rule_sets:
  strict-security:
    description: "Maximum security enforcement"
    rules:
      - no-hardcoded-secrets
      - sql-string-concat
      - unvalidated-user-input
      - missing-auth-check
    
  code-quality:
    description: "Code quality standards"
    rules:
      - no-console-log
      - mutable-default-argument
      - any-type-usage
```

Use in configuration:

```yaml
# .codeverify.yml
rules:
  enable_sets:
    - strict-security
    - code-quality
```

## Importing External Rules

### From Package

```yaml
rules:
  import:
    - "@codeverify/owasp-rules"
    - "@codeverify/react-rules"
```

### From URL

```yaml
rules:
  import:
    - "https://example.com/company-rules.yml"
```

### From Local Path

```yaml
rules:
  import:
    - "./shared-rules/security.yml"
    - "../common/conventions.yml"
```

## Testing Rules

### Test Rule Matches

Create test files:

```yaml
# .codeverify/rules/tests/security-tests.yml
tests:
  - rule: no-hardcoded-secrets
    should_match:
      - |
        api_key = "sk_live_abc123"
      - |
        password = "hunter2"
    should_not_match:
      - |
        api_key = os.environ.get("API_KEY")
      - |
        password = get_password_from_vault()
```

Run tests:

```bash
codeverify rules test
```

### Validate Rules

```bash
codeverify rules validate
```

## Rule Examples

### Security Rules

```yaml
# SQL Injection
- id: sql-injection-risk
  severity: critical
  pattern: |
    (?:execute|query)\s*\(\s*f["']|
    (?:execute|query)\s*\([^)]*\+[^)]*\)|
    (?:execute|query)\s*\([^)]*%s[^)]*%
  message: "Potential SQL injection vulnerability"

# XSS Risk
- id: xss-risk
  severity: high
  pattern: innerHTML\s*=(?!\s*["']<[^>]+>[^<]*</[^>]+>["'])
  message: "Setting innerHTML with dynamic content. Use textContent or sanitize."

# Command Injection
- id: command-injection
  severity: critical
  pattern: |
    subprocess\.(?:run|call|Popen)\([^)]*shell\s*=\s*True
  message: "Shell command with shell=True. Use list arguments instead."
```

### Python Conventions

```yaml
# Type Hints Required
- id: missing-type-hints
  severity: low
  languages: [python]
  ast:
    type: FunctionDef
    returns: null
  message: "Function missing return type hint"

# Docstrings Required
- id: missing-docstring
  severity: low
  languages: [python]
  ast:
    type: FunctionDef
    body:
      - type: {not: Expr}
  message: "Public function missing docstring"
```

### TypeScript Conventions

```yaml
# Prefer const
- id: prefer-const
  severity: low
  languages: [typescript, javascript]
  pattern: let\s+\w+\s*=(?![\s\S]*\1\s*=)
  message: "Use 'const' for variables that are never reassigned"

# No Non-null Assertion
- id: no-non-null-assertion
  severity: medium
  languages: [typescript]
  pattern: \w+!\.
  message: "Avoid non-null assertion. Use optional chaining or null checks."
```

## Disabling Rules

### In Configuration

```yaml
rules:
  disable:
    - no-console-log  # Allow during development
```

### Inline

```python
# codeverify-disable-next-line no-hardcoded-secrets
TEST_API_KEY = "test-key-not-real"  # Used in tests only
```

```typescript
// codeverify-disable-line xss-risk
element.innerHTML = sanitizeHTML(userInput);
```

### For File

```python
# codeverify-disable-file mutable-default-argument
# This file uses mutable defaults intentionally for caching
```

## Rule Metadata

Add metadata for documentation:

```yaml
rules:
  - id: sql-injection-risk
    metadata:
      author: "security-team"
      created: "2024-01-15"
      references:
        - "https://owasp.org/www-community/attacks/SQL_Injection"
        - "https://cwe.mitre.org/data/definitions/89.html"
      cwe: "CWE-89"
      owasp: "A03:2021"
      tags:
        - injection
        - database
        - critical
```

## Next Steps

- See [Verification Settings](/docs/configuration/verification-settings) for formal verification rules
- Learn about [AI Settings](/docs/configuration/ai-settings) for AI-powered analysis
- Check [Troubleshooting](/docs/resources/troubleshooting) for common rule issues
