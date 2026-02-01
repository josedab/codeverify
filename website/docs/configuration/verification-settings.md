---
sidebar_position: 3
---

# Verification Settings

Configure the Z3 formal verification engine.

## Basic Settings

```yaml
verification:
  enabled: true
  timeout: 30
  checks:
    - null_safety
    - array_bounds
    - integer_overflow
    - division_by_zero
```

## Available Checks

| Check | Description | Default |
|-------|-------------|---------|
| `null_safety` | Detect null/undefined dereferences | Enabled |
| `array_bounds` | Catch out-of-bounds array access | Enabled |
| `integer_overflow` | Identify arithmetic overflow | Enabled |
| `division_by_zero` | Prevent divide-by-zero | Enabled |

### Enabling/Disabling Checks

```yaml
verification:
  checks:
    # Enable specific checks
    - null_safety
    - division_by_zero
    
    # Disable by omitting from list
    # (array_bounds and integer_overflow disabled)
```

Or use check-specific configuration:

```yaml
verification:
  checks:
    null_safety:
      enabled: true
      strict_optionals: true
      
    array_bounds:
      enabled: true
      assume_dynamic_bounds: true
      
    integer_overflow:
      enabled: false    # Disable this check
      
    division_by_zero:
      enabled: true
```

## Timeout Configuration

```yaml
verification:
  # Global timeout for all checks (seconds)
  timeout: 30
  
  # Or per-check timeouts
  timeouts:
    null_safety: 20
    array_bounds: 30
    integer_overflow: 45
    division_by_zero: 10
```

Timeout guidelines:
- **10-20s**: Fast feedback, may miss complex issues
- **30-60s**: Good balance for most code
- **60-120s**: Thorough, for critical code paths
- **>120s**: Very thorough, may slow down CI

## Advanced Settings

```yaml
verification:
  advanced:
    # Maximum execution paths to explore
    max_path_depth: 10
    
    # Use incremental solving (faster for related queries)
    incremental: true
    
    # Log Z3 proofs for debugging
    log_proofs: false
    
    # Proof output directory
    proof_dir: ".codeverify/proofs"
    
    # Memory limit (MB)
    memory_limit: 2048
    
    # Use parallel solving
    parallel: true
    max_workers: 4
```

### Path Depth

Controls how many execution paths Z3 explores:

```yaml
verification:
  advanced:
    max_path_depth: 10   # Default
```

- Lower values: Faster but may miss deeply nested bugs
- Higher values: More thorough but slower

For code like:
```python
def process(a, b, c, d):
    if a:
        if b:
            if c:
                if d:
                    # Path depth 4 to reach here
```

### Incremental Solving

```yaml
verification:
  advanced:
    incremental: true   # Default
```

When enabled, Z3 reuses learned information between related queries, making verification faster.

### Proof Logging

Enable for debugging verification issues:

```yaml
verification:
  advanced:
    log_proofs: true
    proof_dir: ".codeverify/proofs"
```

Logs contain:
- Z3 constraints generated
- Solving steps
- Counterexample construction

## Check-Specific Configuration

### Null Safety

```yaml
verification:
  checks:
    null_safety:
      enabled: true
      
      # Treat Optional[T] as potentially null (Python)
      strict_optionals: true
      
      # Track nullability through function calls
      interprocedural: true
      
      # Consider these types as nullable
      nullable_types:
        - "Optional"
        - "None"
        - "null"
        - "undefined"
```

### Array Bounds

```yaml
verification:
  checks:
    array_bounds:
      enabled: true
      
      # For arrays without known size, assume they could be empty
      assume_dynamic_bounds: true
      
      # Check negative indices
      check_negative: true
      
      # Track bounds through slicing
      track_slices: true
```

### Integer Overflow

```yaml
verification:
  checks:
    integer_overflow:
      enabled: true
      
      # Bit width for integer constraints
      bit_width: 64
      
      # Operations to check
      operations:
        - addition
        - subtraction
        - multiplication
        # - division    # Usually caught by div-by-zero
      
      # Assume unsigned integers (no negative overflow)
      assume_unsigned: false
```

### Division by Zero

```yaml
verification:
  checks:
    division_by_zero:
      enabled: true
      
      # Also check modulo operations
      check_modulo: true
      
      # Check floor division
      check_floor_division: true
```

## Performance Tuning

### For Fast CI

```yaml
verification:
  timeout: 15
  advanced:
    max_path_depth: 5
    parallel: true
    max_workers: 2
```

### For Thorough Analysis

```yaml
verification:
  timeout: 120
  advanced:
    max_path_depth: 20
    incremental: true
    memory_limit: 4096
```

### For Critical Code

```yaml
verification:
  timeout: 300
  checks:
    - null_safety
    - array_bounds
    - integer_overflow
    - division_by_zero
  advanced:
    max_path_depth: 50
    log_proofs: true
```

## Disabling Verification

To use only AI analysis:

```yaml
verification:
  enabled: false
  
ai:
  enabled: true
  semantic: true
  security: true
```

Or disable for specific paths:

```yaml
verification:
  enabled: true
  
ignore:
  - pattern: "legacy/**"
    categories:
      - null_safety
      - array_bounds
    reason: "Legacy code, formal verification too slow"
```

## Debugging Verification

If verification produces unexpected results:

1. **Enable proof logging:**
   ```yaml
   verification:
     advanced:
       log_proofs: true
   ```

2. **Use the debugger:**
   ```bash
   codeverify debug src/file.py --function my_function
   ```

3. **Check timeout settings:**
   ```bash
   codeverify analyze src/ --verbose 2>&1 | grep timeout
   ```

See [Verification Debugger](/docs/verification/debugger) for detailed debugging.
