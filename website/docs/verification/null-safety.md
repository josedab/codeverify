---
sidebar_position: 2
---

# Null Safety Verification

Prevent null pointer exceptions and undefined reference errors.

## The Problem

Null reference errors are one of the most common bugs:

```python
def get_user_email(user_id: str) -> str:
    user = find_user(user_id)
    return user.email  # 💥 NoneType has no attribute 'email'
```

```typescript
function getUserEmail(userId: string): string {
    const user = findUser(userId);
    return user.email;  // 💥 Cannot read property 'email' of undefined
}
```

Tony Hoare called null references his "billion dollar mistake."

## How CodeVerify Detects Null Issues

### Static Analysis

CodeVerify tracks nullability through code flow:

```python
def process(user: User | None) -> str:
    # At this point: user may be None
    
    if user is None:
        return "anonymous"
    
    # After the check: user is definitely not None
    return user.name  # ✅ Safe
```

### Z3 Constraint Generation

For complex cases, Z3 proves whether null access is possible:

```python
def get_active_user(users: list[User]) -> str:
    for user in users:
        if user.is_active:
            return user.name
    return None

def greet_active(users: list[User]) -> str:
    user_name = get_active_user(users)
    return f"Hello, {user_name.upper()}"  # ⚠️ May be None
```

Z3 query:
```
∃ users: ∀ u ∈ users: ¬u.is_active
```

Z3 finds: `users = []` or `users = [User(is_active=False)]`

## Common Patterns

### Optional Return Values

```python
# ❌ Unsafe
def find_user(id: str) -> User | None:
    ...

def get_email(id: str) -> str:
    user = find_user(id)
    return user.email  # ⚠️ 'user' may be None

# ✅ Safe
def get_email_safe(id: str) -> str | None:
    user = find_user(id)
    if user is None:
        return None
    return user.email
```

### Chained Access

```typescript
// ❌ Unsafe
function getCity(user: User | undefined): string {
    return user.address.city;  // ⚠️ Multiple potential nulls
}

// ✅ Safe
function getCitySafe(user: User | undefined): string | undefined {
    return user?.address?.city;
}
```

### Collection Operations

```python
# ❌ Unsafe
def get_first_name(users: list[User]) -> str:
    return users[0].name  # ⚠️ List may be empty

# ✅ Safe
def get_first_name_safe(users: list[User]) -> str | None:
    if not users:
        return None
    return users[0].name
```

### Dictionary Access

```python
# ❌ Unsafe
def get_config_value(config: dict, key: str) -> str:
    return config[key].strip()  # ⚠️ Key may not exist, value may be None

# ✅ Safe
def get_config_value_safe(config: dict, key: str) -> str | None:
    value = config.get(key)
    if value is None:
        return None
    return value.strip()
```

## TypeScript/JavaScript Specifics

### undefined vs null

CodeVerify tracks both:

```typescript
function process(value: string | null | undefined) {
    // value could be: string, null, or undefined
    
    if (value === null) {
        // value is: string | undefined
    }
    
    if (value === undefined) {
        // value is: string | null
    }
    
    if (value == null) {
        // Catches both null and undefined
        // value is: string
    }
}
```

### Strict Null Checks

Enable `strictNullChecks` in `tsconfig.json` for best results:

```json
{
  "compilerOptions": {
    "strictNullChecks": true
  }
}
```

CodeVerify's analysis is more precise with this enabled.

## Python Specifics

### Type Annotations

Use `Optional` or union types:

```python
from typing import Optional

# These are equivalent
def find_user(id: str) -> Optional[User]: ...
def find_user(id: str) -> User | None: ...  # Python 3.10+
```

### None Checks

CodeVerify understands various None check patterns:

```python
# All recognized as None guards
if user is None: ...
if user is not None: ...
if user: ...
if not user: ...
if user == None: ...  # Works but `is` preferred
```

### Assertions

```python
def process(user: User | None) -> str:
    assert user is not None, "User required"
    return user.name  # ✅ Safe after assert
```

Note: Assertions can be disabled with `-O`. Use explicit checks in production.

## Configuration

### Enable/Disable

```yaml
verification:
  checks:
    null_safety:
      enabled: true
```

### Strict Mode

Treat all untyped values as potentially null:

```yaml
verification:
  checks:
    null_safety:
      strict_optionals: true
```

### Ignore Patterns

```yaml
ignore:
  - pattern: "tests/**"
    categories:
      - null_safety
    reason: "Test assertions handle nulls"
```

## Fixing Null Safety Issues

### Option 1: Guard Clause

```python
def process(user: User | None) -> str:
    if user is None:
        raise ValueError("User is required")
    return user.name
```

### Option 2: Default Value

```python
def get_name(user: User | None) -> str:
    if user is None:
        return "Unknown"
    return user.name
```

### Option 3: Propagate Nullability

```python
def get_name(user: User | None) -> str | None:
    if user is None:
        return None
    return user.name
```

### Option 4: Assertion (Development)

```python
def process(user: User | None) -> str:
    assert user is not None
    return user.name
```

## Suppressing False Positives

When CodeVerify reports a false positive:

```python
# External API guarantees non-null, but CodeVerify can't know
result = external_api_call()  # Returns User, never None

# Option 1: Type assertion
result: User = external_api_call()  # type: ignore[assignment]

# Option 2: Inline suppression
result.name  # codeverify-disable-line null_safety

# Option 3: Config ignore
```

## Examples

### API Handler

```python
from fastapi import HTTPException

def get_user_handler(user_id: str) -> UserResponse:
    user = db.find_user(user_id)
    
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    # After the check, 'user' is definitely not None
    return UserResponse(
        id=user.id,
        name=user.name,
        email=user.email
    )
```

### React Component

```typescript
interface Props {
    user?: User;
}

function UserCard({ user }: Props) {
    if (!user) {
        return <div>No user</div>;
    }
    
    // After the check, 'user' is defined
    return (
        <div>
            <h2>{user.name}</h2>
            <p>{user.email}</p>
        </div>
    );
}
```

## Next Steps

- [Array Bounds](/docs/verification/array-bounds) — Prevent out-of-bounds access
- [Division by Zero](/docs/verification/division-by-zero) — Prevent divide-by-zero
- [Verification Debugger](/docs/verification/debugger) — Debug verification results
