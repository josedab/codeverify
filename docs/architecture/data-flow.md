# CodeVerify Data Flow

This document describes how data flows through the CodeVerify system during code analysis.

## Overview

CodeVerify processes code through a multi-stage pipeline that combines AI semantic analysis with formal verification using the Z3 SMT solver.

```mermaid
flowchart TD
    subgraph Input
        GH[GitHub PR/Push]
        CLI[CLI Analysis]
        API[API Request]
        IDE[VS Code Extension]
    end

    subgraph Ingestion
        WH[Webhook Handler]
        Q[Redis Queue]
    end

    subgraph Analysis Pipeline
        FT[File Fetcher]
        PS[Code Parser]
        
        subgraph Agents
            SA[Semantic Agent]
            SC[Security Agent]
            TS[Trust Score Agent]
        end
        
        subgraph Verification
            Z3[Z3 Verifier]
            DB[Debugger]
        end
        
        SY[Synthesis Agent]
    end

    subgraph Output
        PG[(PostgreSQL)]
        PR[PR Comment]
        CH[Check Status]
        WB[Webhooks]
        NT[Notifications]
    end

    GH --> WH
    CLI --> Q
    API --> Q
    IDE --> Q
    
    WH --> Q
    Q --> FT
    
    FT --> PS
    PS --> SA
    PS --> SC
    PS --> TS
    PS --> Z3
    
    SA --> SY
    SC --> SY
    TS --> SY
    Z3 --> SY
    Z3 --> DB
    
    SY --> PG
    PG --> PR
    PG --> CH
    PG --> WB
    PG --> NT
```

## Stage Details

### 1. Ingestion Layer

#### GitHub Webhook Handler
When a PR is created or updated:

```mermaid
sequenceDiagram
    participant GH as GitHub
    participant WH as Webhook Handler
    participant Q as Redis Queue
    participant DB as PostgreSQL

    GH->>WH: POST /webhooks (signed payload)
    WH->>WH: Verify HMAC signature
    WH->>WH: Parse event type
    WH->>DB: Create Analysis record (status: queued)
    WH->>Q: Enqueue analysis job
    WH-->>GH: 200 OK
```

**Job Payload Structure:**
```json
{
  "job_id": "uuid",
  "type": "pr_analysis",
  "repo": "owner/repo",
  "pr_number": 42,
  "base_sha": "abc123",
  "head_sha": "def456",
  "installation_id": 12345,
  "config": {
    "languages": ["python", "typescript"],
    "checks": ["null_safety", "array_bounds"]
  }
}
```

### 2. Analysis Pipeline

The worker processes jobs through multiple stages:

```mermaid
stateDiagram-v2
    [*] --> Queued
    Queued --> FetchingCode
    FetchingCode --> Parsing
    Parsing --> SemanticAnalysis
    Parsing --> FormalVerification
    Parsing --> SecurityAnalysis
    SemanticAnalysis --> Synthesis
    FormalVerification --> Synthesis
    SecurityAnalysis --> Synthesis
    Synthesis --> GeneratingFixes
    GeneratingFixes --> PostingResults
    PostingResults --> Completed
    
    FetchingCode --> Failed: Error
    Parsing --> Failed: Error
    SemanticAnalysis --> Failed: Error
    FormalVerification --> Failed: Timeout/Error
```

#### 2.1 Code Fetching

```python
# Fetch changed files from GitHub API
files = await github_client.get_pr_files(pr_number)
for file in files:
    content = await github_client.get_file_content(file.path, head_sha)
    diff = await github_client.get_file_diff(file.path, base_sha, head_sha)
```

#### 2.2 Parsing

The parser extracts AST information for each supported language:

```mermaid
flowchart LR
    subgraph Input
        PY[Python Code]
        TS[TypeScript Code]
    end
    
    subgraph Parsers
        PP[Python Parser]
        TP[TypeScript Parser]
    end
    
    subgraph Output
        AST[Unified AST]
        FN[Functions]
        CL[Classes]
        VC[Variables]
    end
    
    PY --> PP --> AST
    TS --> TP --> AST
    AST --> FN
    AST --> CL
    AST --> VC
```

**Parsed Function Structure:**
```python
@dataclass
class ParsedFunction:
    name: str
    parameters: list[Parameter]
    return_type: str | None
    body_ast: ast.AST
    docstring: str | None
    decorators: list[str]
    line_start: int
    line_end: int
```

#### 2.3 Parallel Analysis

Three analysis streams run concurrently:

```mermaid
flowchart TB
    AST[Parsed AST]
    
    subgraph Semantic["Semantic Agent (LLM)"]
        S1[Extract Intent]
        S2[Infer Contracts]
        S3[Find Logic Issues]
    end
    
    subgraph Formal["Z3 Verifier"]
        F1[Generate Constraints]
        F2[Check SAT/UNSAT]
        F3[Extract Counterexamples]
    end
    
    subgraph Security["Security Agent (LLM)"]
        X1[OWASP Scan]
        X2[CWE Detection]
        X3[AI Vulnerability Check]
    end
    
    AST --> Semantic
    AST --> Formal
    AST --> Security
    
    Semantic --> SR[Semantic Findings]
    Formal --> FR[Verification Results]
    Security --> XR[Security Findings]
```

### 3. Z3 Verification Flow

```mermaid
flowchart TD
    Code[Function Code]
    
    subgraph Extraction
        PC[Extract Path Conditions]
        VC[Generate Verification Conditions]
    end
    
    subgraph Z3["Z3 Solver"]
        SMT[SMT-LIB Formula]
        Check{check-sat}
    end
    
    subgraph Results
        SAT[SAT: Bug Found]
        UNSAT[UNSAT: Proven Safe]
        UNK[UNKNOWN: Timeout]
    end
    
    Code --> PC --> VC --> SMT --> Check
    Check -->|satisfiable| SAT
    Check -->|unsatisfiable| UNSAT
    Check -->|timeout| UNK
    
    SAT --> CE[Counterexample]
    CE --> Finding
```

**Verification Condition Example (Division by Zero):**
```smt2
; SMT-LIB formula for: def divide(a, b): return a / b
(declare-const a Int)
(declare-const b Int)
(assert (= b 0))  ; Check if b can be zero
(check-sat)
; Result: sat → Division by zero possible
; Model: b = 0
```

### 4. Synthesis and Output

```mermaid
flowchart LR
    subgraph Input
        SF[Semantic Findings]
        VF[Verification Findings]
        XF[Security Findings]
    end
    
    subgraph Synthesis["Synthesis Agent"]
        DD[Deduplicate]
        PR[Prioritize]
        GF[Generate Fixes]
    end
    
    subgraph Output
        CF[Consolidated Findings]
        FX[Fix Suggestions]
        SM[Summary]
    end
    
    SF --> DD
    VF --> DD
    XF --> DD
    DD --> PR --> GF
    GF --> CF
    GF --> FX
    GF --> SM
```

**Final Finding Structure:**
```python
@dataclass
class Finding:
    id: str
    category: FindingCategory  # security, verification, quality
    severity: Severity  # critical, high, medium, low
    title: str
    description: str
    file_path: str
    line_start: int
    line_end: int
    confidence: float  # 0.0-1.0
    source: str  # semantic, z3, security
    fix_suggestion: str | None
    verification_proof: str | None  # Z3 proof for verified findings
```

### 5. Result Delivery

```mermaid
sequenceDiagram
    participant W as Worker
    participant DB as PostgreSQL
    participant GH as GitHub API
    participant WH as Webhooks
    participant SL as Slack/Teams

    W->>DB: Store findings
    W->>DB: Update analysis status
    
    par GitHub
        W->>GH: Create check run
        W->>GH: Post PR comment
        W->>GH: Add inline annotations
    and Notifications
        W->>WH: Emit analysis.completed
        W->>SL: Send notification
    end
```

## Caching Strategy

```mermaid
flowchart TD
    subgraph Request
        R[Analysis Request]
    end
    
    subgraph Cache["Redis Cache"]
        FC[File Content Cache]
        TC[Trust Score Cache]
        DC[Diff Summary Cache]
    end
    
    subgraph Computation
        AN[Full Analysis]
    end
    
    R --> FC
    FC -->|hit| TC
    FC -->|miss| AN
    AN --> TC
    TC --> DC
```

**Cache Keys:**
- `file:{repo}:{sha}:{path}` - File content (TTL: 1 hour)
- `trust:{repo}:{sha}:{path}` - Trust scores (TTL: 24 hours)
- `diff:{repo}:{base}:{head}` - Diff summaries (TTL: 1 hour)

## Error Handling

```mermaid
flowchart TD
    E[Error Occurs]
    
    E --> T{Error Type}
    
    T -->|Transient| RT[Retry with backoff]
    T -->|Rate Limit| RL[Wait + Retry]
    T -->|Validation| VE[Return 400]
    T -->|Internal| IE[Log + Alert + Return 500]
    T -->|Timeout| TO[Mark as UNKNOWN]
    
    RT -->|Max retries| DLQ[Dead Letter Queue]
    RL -->|Max wait| DLQ
```

## Performance Characteristics

| Stage | Typical Duration | Parallelism |
|-------|-----------------|-------------|
| Code Fetch | 100-500ms | Per file |
| Parsing | 10-50ms/file | Per file |
| Semantic Agent | 2-5s | Per function |
| Z3 Verification | 100ms-30s | Per check |
| Security Agent | 1-3s | Per file |
| Synthesis | 500ms-2s | Single |
| Result Posting | 200-500ms | Single |

## Scalability

```mermaid
flowchart TB
    subgraph Load Balancer
        LB[NGINX]
    end
    
    subgraph API Tier
        A1[API 1]
        A2[API 2]
        A3[API N]
    end
    
    subgraph Worker Tier
        W1[Worker 1]
        W2[Worker 2]
        W3[Worker N]
    end
    
    subgraph Data Tier
        PG[(PostgreSQL Primary)]
        PGR[(PostgreSQL Replica)]
        RD[(Redis Cluster)]
    end
    
    LB --> A1 & A2 & A3
    A1 & A2 & A3 --> RD
    W1 & W2 & W3 --> RD
    A1 & A2 & A3 --> PG
    W1 & W2 & W3 --> PG
    PG --> PGR
```

Workers scale horizontally based on queue depth. Each worker can process multiple jobs concurrently using Celery's prefetch settings.
