# Next-Gen Features API Reference (v0.3.0)

This document describes the new API endpoints introduced in CodeVerify v0.3.0.

## Table of Contents

1. [Threat Modeling](#threat-modeling)
2. [Risk Prediction](#risk-prediction)
3. [Consensus Verification](#consensus-verification)
4. [Compliance Attestation](#compliance-attestation)
5. [Cost Optimization](#cost-optimization)
6. [Cross-Language Verification](#cross-language-verification)

---

## Threat Modeling

Generate security threat models using STRIDE methodology and OWASP Top 10 mapping.

### POST /api/v1/threat-model

Generate a threat model from code.

**Request:**
```json
{
  "code": "string (required)",
  "system_name": "string (default: 'Unknown System')",
  "architecture_description": "string (optional)",
  "language": "string (default: 'python')",
  "framework": "string (optional, e.g., 'fastapi')",
  "deployment_context": "string (optional)"
}
```

**Response:**
```json
{
  "system_name": "User API",
  "description": "...",
  "attack_surfaces": [
    {
      "name": "POST /users",
      "type": "api_endpoint",
      "entry_points": ["username", "password"],
      "trust_level": "untrusted"
    }
  ],
  "threats": [
    {
      "id": "T1",
      "title": "SQL Injection",
      "stride_category": "tampering",
      "owasp_category": "A03:2021",
      "risk_score": 9.5,
      "mitigations": ["Use parameterized queries"]
    }
  ],
  "overall_risk_score": 7.5,
  "recommendations": ["Implement input validation"]
}
```

### GET /api/v1/threat-model/categories/stride

Get STRIDE threat categories.

### GET /api/v1/threat-model/categories/owasp

Get OWASP Top 10 2021 categories.

---

## Risk Prediction

Predict bug risk for code changes using the Regression Oracle.

### POST /api/v1/risk-prediction

Predict risk for a single code change.

**Request:**
```json
{
  "diff": "string (required, the diff or changed code)",
  "change_id": "string (optional)",
  "file_paths": ["src/auth.py"],
  "author": "developer@example.com",
  "commit_message": "Add token validation"
}
```

**Response:**
```json
{
  "change_id": "abc123",
  "risk_level": "medium",
  "risk_score": 45.5,
  "confidence": 0.85,
  "verification_priority": 3,
  "risk_factors": [
    {
      "factor": "Large change size",
      "details": "250 lines changed",
      "contribution": 15
    }
  ],
  "recommended_actions": ["Add unit tests for edge cases"],
  "similar_past_bugs": []
}
```

### POST /api/v1/risk-prediction/batch

Predict risk for multiple changes, sorted by priority.

### POST /api/v1/risk-prediction/bugs

Record a bug for training the oracle.

### POST /api/v1/risk-prediction/feedback

Submit feedback on a prediction to improve accuracy.

---

## Consensus Verification

Verify code using multiple LLM models to reduce false positives.

### POST /api/v1/consensus

Run multi-model consensus verification.

**Request:**
```json
{
  "code": "string (required)",
  "file_path": "payment.py",
  "language": "python",
  "consensus_strategy": "majority"
}
```

**Consensus Strategies:**
- `unanimous` - All models must agree
- `majority` - More than 50% must agree
- `weighted` - Weighted by model confidence
- `any` - Any model finding is reported

**Response:**
```json
{
  "code_hash": "abc123",
  "consensus_strategy": "majority",
  "models_queried": ["openai_gpt5", "anthropic_claude"],
  "overall_confidence": 0.92,
  "consensus_findings": [
    {
      "id": "consensus_f1",
      "severity": "high",
      "title": "Missing input validation",
      "agreeing_models": ["openai_gpt5", "anthropic_claude"],
      "dissenting_models": []
    }
  ],
  "model_only_findings": {}
}
```

### POST /api/v1/consensus/escalate

Start with fast verification, escalate to consensus if needed.

### GET /api/v1/consensus/strategies

List available consensus strategies.

### GET /api/v1/consensus/models

List available models for consensus.

---

## Compliance Attestation

Generate compliance reports from verification results.

### POST /api/v1/compliance

Generate a compliance report.

**Request:**
```json
{
  "framework": "soc2",
  "verification_results": [
    {
      "file_path": "src/auth.py",
      "status": "verified",
      "findings": [],
      "verified_properties": ["authentication", "authorization"]
    }
  ],
  "scope": "User Authentication System",
  "organization": "Acme Corp"
}
```

**Supported Frameworks:**
- `soc2` - SOC 2 Trust Services Criteria
- `hipaa` - HIPAA Security Rule
- `pci_dss` - PCI DSS
- `gdpr` - GDPR
- `iso_27001` - ISO 27001

**Response:**
```json
{
  "report_id": "abc123",
  "framework": "soc2",
  "compliance_score": 85.0,
  "overall_status": "partial",
  "controls": [
    {
      "id": "CC6.1",
      "name": "Logical and Physical Access Controls",
      "status": "compliant",
      "verification_coverage": 90.0,
      "gaps": [],
      "recommendations": []
    }
  ]
}
```

### POST /api/v1/compliance/multi-framework

Generate reports for multiple frameworks at once.

### POST /api/v1/compliance/certificate

Generate an attestation certificate.

### GET /api/v1/compliance/frameworks

List supported compliance frameworks.

### GET /api/v1/compliance/frameworks/{framework}/controls

Get controls for a specific framework.

---

## Cost Optimization

Plan verification depth based on risk and budget.

### POST /api/v1/cost/plan

Plan verification for a code change.

**Request:**
```json
{
  "code": "def risky(): eval(input())",
  "risk_profile": {
    "risk_score": 75,
    "is_security_sensitive": true
  },
  "budget": {
    "max_cost_usd": 0.05,
    "min_accuracy": 0.9
  }
}
```

**Verification Depths:**
- `pattern` - Fast pattern matching (~$0.0001)
- `static` - Static analysis (~$0.001)
- `ai` - LLM-based analysis (~$0.02)
- `formal` - Full Z3 verification (~$0.05)
- `consensus` - Multi-model consensus (~$0.15)

**Response:**
```json
{
  "code_hash": "abc123",
  "selected_depth": "ai",
  "estimated_cost_usd": 0.02,
  "estimated_time_ms": 3000,
  "estimated_accuracy": 0.85,
  "rationale": ["High risk score (75)", "Security-sensitive code detected"],
  "fallback_depth": "static"
}
```

### POST /api/v1/cost/plan/batch

Optimize verification for multiple items within a budget.

### POST /api/v1/cost/suggest-budget

Suggest a budget for verifying code items.

### GET /api/v1/cost/usage

Get budget usage statistics.

### GET /api/v1/cost/cost-model

Get current cost model.

### GET /api/v1/cost/depths

List verification depths with descriptions.

---

## Cross-Language Verification

Verify polyglot codebases with unified contracts.

### POST /api/v1/cross-language/infer

Infer a contract from code.

**Request:**
```json
{
  "code": "def calculate(x: int, y: int) -> int:\n    return x + y",
  "language": "python",
  "symbol_name": "calculate",
  "symbol_type": "function"
}
```

**Supported Languages:**
- `python` - Full support
- `typescript` - Full support
- `go` - Full support
- `rust` - Full support
- `java` - Full support
- `javascript`, `csharp`, `cpp` - Type mapping only

**Response:**
```json
{
  "contract_id": "py_calculate",
  "contract_type": "function",
  "contract": {
    "name": "calculate",
    "parameters": [
      {"name": "x", "type": {"base_type": "int"}},
      {"name": "y", "type": {"base_type": "int"}}
    ],
    "return_type": {"base_type": "int"}
  },
  "inferred_from": "python"
}
```

### POST /api/v1/cross-language/verify

Verify code against an existing contract.

### POST /api/v1/cross-language/stub

Generate a stub in the target language.

**Request:**
```json
{
  "contract_id": "py_calculate",
  "target_language": "go"
}
```

**Response:**
```json
{
  "contract_id": "py_calculate",
  "target_language": "go",
  "stub": "// Generated from cross-language contract.\nfunc calculate(x int, y int) int {\n    panic(\"Not implemented\")\n}"
}
```

### POST /api/v1/cross-language/type-mapping

Get type mapping between languages.

### GET /api/v1/cross-language/contracts

List registered contracts.

### GET /api/v1/cross-language/languages

List supported languages.

### GET /api/v1/cross-language/type-mappings

Get all type mappings.

---

## Error Responses

All endpoints may return these error codes:

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Missing or invalid API key |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource doesn't exist |
| 429 | Too Many Requests - Rate limited |
| 500 | Internal Server Error |

Error response format:
```json
{
  "error": {
    "code": "invalid_request",
    "message": "Description of the error"
  }
}
```
