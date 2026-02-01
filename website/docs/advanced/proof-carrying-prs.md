---
sidebar_position: 4
---

# Proof-Carrying PRs

Attach mathematical proofs to pull requests, ensuring verified code stays verified.

## Overview

Proof-Carrying PRs extend the concept of [proof-carrying code](https://en.wikipedia.org/wiki/Proof-carrying_code) to the code review process. When CodeVerify verifies your code, it generates a cryptographic proof that can be:

- Attached to the PR
- Verified by reviewers instantly
- Checked in CI without re-running analysis
- Stored for audit trails

## How It Works

```
Developer                        Reviewer/CI
    │                                │
    │  1. Write code                 │
    │  2. Run CodeVerify             │
    │     → Generate proof           │
    │  3. Commit code + proof        │
    │                                │
    │  ───── Open PR ─────────────>  │
    │                                │
    │                           4. Verify proof (fast)
    │                              → No findings
    │                           5. Approve PR
    │                                │
    └────────────────────────────────┘
```

## Benefits

- **Faster CI** — Verify proofs in seconds, not minutes
- **Reproducibility** — Same result every time
- **Auditability** — Proof of verification at a point in time
- **Trust** — Reviewers know code was verified

## Configuration

### Enable Proof Generation

```yaml
# .codeverify.yml
proofs:
  enabled: true
  
  # Where to store proofs
  output_dir: ".codeverify/proofs"
  
  # Include in git commits
  commit_proofs: true
  
  # Sign proofs cryptographically
  sign: true
```

### CI Configuration

```yaml
# .github/workflows/verify-proofs.yml
name: Verify Proofs

on:
  pull_request:
    branches: [main]

jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Verify Proofs
        uses: codeverify/action@v1
        with:
          mode: verify-proofs
          proofs_dir: .codeverify/proofs
```

## Generating Proofs

### CLI

```bash
# Generate proofs during analysis
codeverify analyze src/ --generate-proofs

# Proofs are written to .codeverify/proofs/
ls .codeverify/proofs/
# calculator.py.proof
# utils.py.proof
# manifest.json
```

### Proof Format

```json
// .codeverify/proofs/manifest.json
{
  "version": "1.0",
  "generated_at": "2024-01-15T10:30:00Z",
  "codeverify_version": "1.2.3",
  "files": [
    {
      "path": "src/calculator.py",
      "hash": "sha256:abc123...",
      "proof": "calculator.py.proof",
      "result": "verified",
      "findings": []
    }
  ],
  "signature": "..."
}
```

### Proof Contents

```
// .codeverify/proofs/calculator.py.proof
{
  "file": "src/calculator.py",
  "file_hash": "sha256:abc123...",
  "functions": [
    {
      "name": "divide",
      "line": 10,
      "checks": [
        {
          "type": "division_by_zero",
          "result": "verified",
          "constraints": "...",
          "z3_proof": "..."
        }
      ]
    }
  ]
}
```

## Verifying Proofs

### In CI

```bash
# Fast verification (checks proofs match current code)
codeverify verify-proofs

# Output:
# Verifying proofs for 15 files...
# ✓ src/calculator.py - proof valid
# ✓ src/utils.py - proof valid
# ✗ src/api.py - file changed since proof generated
# 
# 14/15 proofs valid
# 1 file needs re-verification
```

### What Gets Verified

1. **File hash matches** — Code hasn't changed since proof
2. **Proof is valid** — Z3 can verify the proof
3. **Signature is valid** — Proof wasn't tampered with
4. **Version compatible** — CodeVerify version can verify

## Incremental Proofs

Only re-verify changed files:

```yaml
proofs:
  incremental: true
  
  # Reuse proofs for unchanged files
  cache_valid_proofs: true
```

```bash
# First run: generates all proofs
codeverify analyze src/ --generate-proofs
# Generated 50 proofs in 120s

# After small change: only regenerate affected
codeverify analyze src/ --generate-proofs
# Reusing 48 proofs, regenerating 2
# Completed in 5s
```

## PR Workflow

### 1. Developer Generates Proofs

```bash
# Before committing
codeverify analyze src/ --generate-proofs
git add .codeverify/proofs/
git commit -m "feat: add divide function with verification proof"
```

### 2. CI Verifies Proofs

```yaml
jobs:
  verify-proofs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Verify Proofs
        run: codeverify verify-proofs --strict
```

### 3. Handle Proof Failures

When code changes invalidate proofs:

```
PR Check: CodeVerify Proofs
Status: ❌ Failed

3 files have invalid or missing proofs:
  - src/calculator.py (file changed)
  - src/api.py (new file, no proof)
  - src/utils.py (proof expired)

Run `codeverify analyze --generate-proofs` and commit.
```

## Proof Signing

### Generate Signing Keys

```bash
codeverify keys generate
# Created: ~/.codeverify/keys/private.pem
# Created: ~/.codeverify/keys/public.pem
```

### Configure Signing

```yaml
proofs:
  sign: true
  signing_key: ${CODEVERIFY_SIGNING_KEY}
  
  # Or use file path
  signing_key_file: "~/.codeverify/keys/private.pem"
```

### Verify Signatures

```yaml
proofs:
  verify_signatures: true
  trusted_keys:
    - "~/.codeverify/keys/public.pem"
    - "https://keys.company.com/codeverify.pub"
```

## GitHub Integration

### PR Status Check

CodeVerify adds a status check showing proof status:

```
✓ CodeVerify Proofs
  All 23 proofs verified
  Last verified: 2 minutes ago
```

### PR Comment

```markdown
## CodeVerify Proof Status

| File | Status | Details |
|------|--------|---------|
| `src/calculator.py` | ✅ Verified | Generated 2h ago |
| `src/api.py` | ⚠️ Outdated | File modified, re-run needed |
| `src/new.py` | ❌ Missing | No proof generated |

**Action Required:** Run `codeverify analyze --generate-proofs` for 2 files.
```

## Proof Expiration

Proofs can expire to ensure regular re-verification:

```yaml
proofs:
  # Proofs expire after 30 days
  expiration_days: 30
  
  # Require re-verification on major version updates
  expire_on_version_change: true
```

## Audit Trail

Maintain a history of proofs:

```yaml
proofs:
  audit:
    enabled: true
    storage: "s3://company-audits/codeverify/"
    
    # What to store
    include:
      - proofs
      - findings
      - timestamps
      - signatures
```

Query historical proofs:

```bash
codeverify audit query \
  --file src/calculator.py \
  --from 2024-01-01 \
  --to 2024-06-01

# Shows proof history for the file
```

## Best Practices

1. **Commit proofs with code** — They're part of the verification record
2. **Use signing in production** — Prevents proof tampering
3. **Set reasonable expiration** — Balance freshness vs. re-verification cost
4. **Fail CI on missing proofs** — Ensure all code is verified
5. **Review proof failures** — Understand why proofs invalidated

## Limitations

- Proofs are invalidated by any file change (even whitespace)
- Large files may have large proofs
- Proof verification requires CodeVerify installed
- Proofs don't cover AI analysis (only formal verification)

## Next Steps

- [CI/CD Integration](/docs/integrations/ci-cd) — Full CI setup
- [Verification Overview](/docs/verification/overview) — How verification works

