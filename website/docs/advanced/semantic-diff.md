---
sidebar_position: 6
---

# Semantic Diff

Understand the real impact of code changes, beyond line-by-line diffs.

## Overview

Traditional diffs show what lines changed. Semantic diff shows what the changes **mean**:

- Does this change affect behavior?
- What functions are impacted?
- Are there new risks introduced?
- What's the blast radius?

## How It Works

```
Traditional Diff                    Semantic Diff
─────────────────                   ──────────────────────────────
- old_value = x                     Behavior Change:
+ new_value = x                       Variable renamed (safe)
                                      No functional change
                                      
- if user.is_admin:                 Behavior Change:
+ if user.is_admin or user.is_mod:    Access control expanded
                                      Moderators gain admin access
                                      Risk: HIGH - review required
```

## Usage

### CLI

```bash
# Compare branches
codeverify diff main feature-branch

# Compare commits
codeverify diff abc123 def456

# Compare files
codeverify diff src/auth.py:main src/auth.py:feature
```

### Output

```
Semantic Diff: main → feature-branch

Files changed: 5
Semantic changes: 3

┌─────────────────────────────────────────────────────────────┐
│ src/auth.py                                                 │
├─────────────────────────────────────────────────────────────┤
│ Function: check_permissions()                               │
│ Change Type: BEHAVIOR MODIFICATION                          │
│ Risk: HIGH                                                  │
│                                                             │
│ Before:                                                     │
│   Returns True only if user.role == 'admin'                 │
│                                                             │
│ After:                                                      │
│   Returns True if user.role in ['admin', 'moderator']       │
│                                                             │
│ Impact:                                                     │
│   - Callers: 12 functions across 4 files                    │
│   - Moderators will gain admin-level access                 │
│   - Affects: delete_user(), modify_settings(), export_data()│
│                                                             │
│ Recommendation: Verify moderator access is intentional      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ src/calculator.py                                           │
├─────────────────────────────────────────────────────────────┤
│ Function: calculate_total()                                 │
│ Change Type: REFACTORING                                    │
│ Risk: LOW                                                   │
│                                                             │
│ Changes:                                                    │
│   - Variable renamed: sum → total                           │
│   - Extracted helper: apply_discount()                      │
│   - No behavioral change                                    │
│                                                             │
│ Verification: Equivalent for all inputs ✓                   │
└─────────────────────────────────────────────────────────────┘
```

## Change Types

### No Change (Formatting)

```python
# Before
def add(a,b): return a+b

# After  
def add(a, b):
    return a + b

# Semantic Diff: NO CHANGE (formatting only)
```

### Refactoring

```python
# Before
def process(items):
    result = []
    for item in items:
        result.append(item * 2)
    return result

# After
def process(items):
    return [item * 2 for item in items]

# Semantic Diff: REFACTORING (equivalent behavior)
```

### Behavior Modification

```python
# Before
def get_users(active_only=False):
    if active_only:
        return [u for u in users if u.active]
    return users

# After
def get_users(active_only=True):  # Default changed!
    if active_only:
        return [u for u in users if u.active]
    return users

# Semantic Diff: BEHAVIOR MODIFICATION
# Callers without explicit argument now get different results
```

### New Functionality

```python
# Before
def calculate(a, b):
    return a + b

# After
def calculate(a, b, operation='add'):
    if operation == 'add':
        return a + b
    elif operation == 'subtract':
        return a - b
    raise ValueError(f"Unknown operation: {operation}")

# Semantic Diff: NEW FUNCTIONALITY
# - New parameter with default (backward compatible)
# - New code paths
# - New error condition
```

### Breaking Change

```python
# Before
def create_user(name, email):
    ...

# After
def create_user(name, email, role):  # Required parameter added!
    ...

# Semantic Diff: BREAKING CHANGE
# - All callers must be updated
# - Affected callers: 15 locations
```

## Impact Analysis

### Call Graph

```bash
codeverify diff main feature --show-impact
```

```
Impact Analysis for: check_permissions()

Direct callers (3):
├─ delete_user() in admin.py:45
├─ modify_settings() in settings.py:23
└─ export_data() in export.py:89

Indirect callers (9):
├─ admin.py
│  └─ handle_admin_request() → delete_user()
├─ api/handlers.py
│  ├─ delete_handler() → delete_user()
│  └─ settings_handler() → modify_settings()
└─ ...

Total affected: 12 functions in 5 files
```

### Data Flow

```
Data Flow Impact

Changed: validate_input() now allows special characters

Data sources affected:
├─ API input: /api/users (user-provided data)
├─ Form input: registration form
└─ File import: CSV upload

Data sinks at risk:
├─ Database: users.name column (potential injection)
├─ HTML output: user profile page (potential XSS)
└─ Logging: audit log (potential log injection)
```

## PR Integration

### GitHub Comment

```markdown
## Semantic Diff Analysis

### Summary
| Change Type | Count | Risk |
|-------------|-------|------|
| Breaking | 0 | - |
| Behavior Mod | 2 | 🔴 High |
| New Features | 1 | 🟡 Medium |
| Refactoring | 3 | 🟢 Low |
| Formatting | 5 | ⚪ None |

### High-Risk Changes

<details>
<summary>🔴 check_permissions() - Access control expanded</summary>

**Before:** Only admins have access
**After:** Admins and moderators have access

**Impact:** 12 functions across 4 files

**Review:** Verify moderator access is intentional
</details>

### Breaking Changes
None detected ✓
```

## Configuration

```yaml
# .codeverify.yml
semantic_diff:
  enabled: true
  
  # Include in PR comments
  pr_comment: true
  
  # Fail on breaking changes
  fail_on_breaking: true
  
  # Require review for behavior changes
  require_review:
    - behavior_modification
    - new_functionality
  
  # Ignore formatting-only changes
  ignore_formatting: true
```

## API

### Programmatic Access

```python
from codeverify import SemanticDiff

diff = SemanticDiff.compare(
    base="main",
    head="feature-branch",
    repo_path="."
)

for change in diff.changes:
    print(f"{change.function}: {change.type} ({change.risk})")
    
    if change.type == "breaking":
        print(f"  Affected callers: {change.affected_callers}")
```

### JSON Output

```bash
codeverify diff main feature --format json
```

```json
{
  "base": "main",
  "head": "feature-branch",
  "changes": [
    {
      "file": "src/auth.py",
      "function": "check_permissions",
      "type": "behavior_modification",
      "risk": "high",
      "description": "Access control expanded to include moderators",
      "before": "Returns True only if user.role == 'admin'",
      "after": "Returns True if user.role in ['admin', 'moderator']",
      "affected_callers": [
        {"function": "delete_user", "file": "admin.py", "line": 45}
      ]
    }
  ]
}
```

## Best Practices

1. **Review high-risk changes** — Don't merge without understanding impact
2. **Block breaking changes** — Require explicit approval
3. **Document intentional changes** — Add comments explaining why
4. **Use in code review** — Semantic diff provides better context
5. **Track over time** — Monitor types of changes per team

## Limitations

- Complex refactoring may be flagged as behavior change
- External dependencies aren't fully analyzed
- Dynamic code (eval, reflection) limits accuracy
- Very large diffs may time out

## Next Steps

- [CI/CD Integration](/docs/integrations/ci-cd) — Add to PR workflow
- [API Reference](/docs/api/overview) — Programmatic access
- [Team Learning](/docs/advanced/team-learning) — Track change patterns
