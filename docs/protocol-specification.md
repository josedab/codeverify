# LLM Output Verification Protocol — Specification v1.0.0

## Abstract

The LLM Output Verification Protocol (LOVP) is a standardized protocol
enabling any AI coding assistant to submit generated code for formal
verification and receive proof certificates. It defines message types,
proof formats, capability negotiation, and session management.

## Status

- **Version**: 1.0.0
- **Status**: Draft
- **Authors**: CodeVerify Team
- **License**: MIT

## 1. Introduction

As AI coding assistants (GitHub Copilot, Cursor, Cody, Continue) become
ubiquitous, there is no standardized way to verify their output. Each
tool has its own quality checks, if any. LOVP provides a common protocol
so that any AI assistant can delegate verification to a specialized
verification server.

### 1.1 Goals

1. **Standardization**: One protocol for all AI coding tools
2. **Proof Certificates**: Cryptographically signed verification results
3. **Language Agnostic**: Support any programming language
4. **Extensible**: Custom check types via capability negotiation
5. **Performance**: Sub-second verification for typical code snippets

### 1.2 Non-Goals

1. Replacing existing linters or type checkers
2. Defining how AI assistants generate code
3. Mandating specific verification algorithms

## 2. Protocol Overview

```
┌─────────────┐                    ┌──────────────────┐
│ AI Assistant │ ── VerifyRequest → │ Verification     │
│ (Client)     │                    │ Server           │
│              │ ← VerifyResponse ─ │ (CodeVerify)     │
│              │                    │                  │
│              │ ← ProofCert ────── │   Z3 + AI Agents │
└─────────────┘                    └──────────────────┘
```

### 2.1 Transport

- **Primary**: JSON-RPC 2.0 over HTTP/HTTPS
- **Alternative**: gRPC with Protocol Buffers
- **WebSocket**: For streaming verification results

### 2.2 Authentication

- Bearer token (API key) in `Authorization` header
- HMAC-SHA256 signature for proof certificates

## 3. Messages

### 3.1 CapabilityQuery / CapabilityResponse

```json
// Request
{
  "jsonrpc": "2.0",
  "method": "capabilities",
  "id": 1
}

// Response
{
  "jsonrpc": "2.0",
  "result": {
    "protocol_version": "1.0.0",
    "supported_languages": ["python", "typescript", "go", "java", "rust", "c", "cpp"],
    "supported_checks": ["null_safety", "bounds_check", "division_zero", "overflow",
                         "memory_safety", "type_safety", "security", "all"],
    "max_file_size_bytes": 500000,
    "max_files_per_request": 20,
    "supports_streaming": true,
    "supports_proofs": true,
    "supports_fixes": true
  },
  "id": 1
}
```

### 3.2 VerifyRequest

```json
{
  "jsonrpc": "2.0",
  "method": "verify",
  "params": {
    "id": "req_abc123",
    "protocol_version": "1.0.0",
    "client_id": "cursor-1.0",
    "client_name": "Cursor IDE",
    "files": [
      {
        "path": "app.py",
        "content": "def divide(a, b):\n    return a / b\n"
      }
    ],
    "language": "python",
    "checks": ["all"],
    "include_proofs": true,
    "include_fixes": false,
    "timeout_ms": 30000
  },
  "id": 2
}
```

### 3.3 VerifyResponse

```json
{
  "jsonrpc": "2.0",
  "result": {
    "request_id": "req_abc123",
    "status": "failed",
    "findings": [
      {
        "id": "f_001",
        "file_path": "app.py",
        "line": 2,
        "check_type": "division_zero",
        "severity": "critical",
        "message": "Potential division by zero when b=0",
        "fix_suggestion": "if b != 0:\n    return a / b\nelse:\n    return 0"
      }
    ],
    "proofs": [
      {
        "id": "p_001",
        "check_type": "division_zero",
        "status": "failed",
        "constraints_checked": 1,
        "content_hash": "a1b2c3d4e5f6",
        "signature": "hmac_sha256_sig",
        "issued_at": "2026-02-22T20:00:00Z"
      }
    ],
    "verification_time_ms": 45,
    "server_id": "codeverify"
  },
  "id": 2
}
```

### 3.4 Error Response

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32001,
    "message": "Rate limit exceeded",
    "data": {
      "retry_after_seconds": 60
    }
  },
  "id": 2
}
```

## 4. Proof Certificates

### 4.1 Format

```json
{
  "id": "p_001",
  "check_type": "null_safety",
  "status": "verified",
  "constraints_checked": 5,
  "content_hash": "sha256_of_verified_code_first_12_chars",
  "signature": "hmac_sha256_of_id:check_type:status:content_hash",
  "issued_at": "2026-02-22T20:00:00Z"
}
```

### 4.2 Signature Verification

```python
import hmac, hashlib

def verify_proof(proof, secret):
    payload = f"{proof['id']}:{proof['check_type']}:{proof['status']}:{proof['content_hash']}"
    expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()[:16]
    return hmac.compare_digest(expected, proof['signature'])
```

### 4.3 Status Values

| Status | Meaning |
|--------|---------|
| `verified` | All constraints satisfied — code is mathematically proven correct for this check |
| `failed` | At least one constraint violated — counterexample exists |
| `partial` | Some constraints checked, some skipped (timeout or complexity limit) |
| `timeout` | Verification exceeded time limit |
| `error` | Server error during verification |

## 5. Check Types

| Check Type | Description |
|-----------|-------------|
| `null_safety` | Variables are not null/None before use |
| `bounds_check` | Array/list indices are within bounds |
| `division_zero` | Divisors are not zero |
| `overflow` | Integer arithmetic does not overflow |
| `memory_safety` | No use-after-free, double-free, buffer overflow (C/C++/Rust) |
| `type_safety` | Values match expected types |
| `security` | No injection, eval(), hardcoded secrets |
| `all` | Run all applicable checks |

## 6. Error Codes

| Code | Meaning |
|------|---------|
| -32001 | Rate limit exceeded |
| -32002 | File too large |
| -32003 | Unsupported language |
| -32004 | Authentication failed |
| -32005 | Verification timeout |
| -32600 | Invalid request |
| -32601 | Method not found |
| -32700 | Parse error |

## 7. Reference Implementations

### 7.1 Python Server

```python
from codeverify_core.verification_protocol import VerificationProtocolServer, VerifyRequest

server = VerificationProtocolServer()
result = server.verify(VerifyRequest(
    files=[{"path": "app.py", "content": code}],
    language="python",
))
```

### 7.2 Python Client

```python
from codeverify_core.protocol_client import VerificationClient

client = VerificationClient()  # local mode
result = client.verify("def f(x): return x / 0", language="python")
print(result.status)    # "failed"
print(result.findings)  # [Finding(severity="critical", ...)]
```

## 8. Versioning

- Protocol versions use Semantic Versioning
- Clients and servers negotiate version via `protocol_version` field
- Servers MUST support the latest minor version within a major version
- Breaking changes require a new major version

## 9. Security Considerations

1. **TLS Required**: All production traffic MUST use HTTPS
2. **API Key Rotation**: Keys should be rotated every 90 days
3. **Proof Integrity**: Proof signatures prevent tampering
4. **Code Privacy**: Source code is processed in memory, not persisted by default
5. **Rate Limiting**: Servers MUST implement per-client rate limits

## 10. References

- [CodeVerify Repository](https://github.com/codeverify/codeverify)
- [Z3 SMT Solver](https://github.com/Z3Prover/z3)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
- [Model Context Protocol](https://modelcontextprotocol.io)
