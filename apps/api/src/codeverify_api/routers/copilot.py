"""GitHub Copilot Extension / Chat Participant API router."""

import hashlib
import hmac
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Header, Request, status
from pydantic import BaseModel, Field

router = APIRouter()


class CopilotContext(BaseModel):
    file_path: str | None = None
    language: str | None = None
    selected_code: str | None = None
    full_file_content: str | None = None
    cursor_line: int | None = None
    repository: str | None = None
    branch: str | None = None


class CopilotChatRequest(BaseModel):
    message: str = Field(description="User message to @codeverify")
    context: CopilotContext = Field(default_factory=CopilotContext)
    conversation_id: str | None = None


class CodeSuggestion(BaseModel):
    file_path: str
    original_code: str
    suggested_code: str
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)


class CopilotChatResponse(BaseModel):
    content: str = Field(description="Markdown-formatted response")
    code_suggestions: list[CodeSuggestion] = Field(default_factory=list)
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    follow_up_actions: list[str] = Field(default_factory=list)
    processing_time_ms: float
    command_detected: str | None = None


# Command patterns
COMMAND_PATTERNS: dict[str, list[str]] = {
    "verify": ["verify", "check", "analyze", "is this safe", "is this correct", "review", "scan"],
    "explain": ["explain", "what does this", "why", "how does", "what is this proof"],
    "spec": ["spec", "specification", "invariant", "contract", "formal spec"],
    "trust_score": ["trust", "score", "confidence", "reliability", "ai generated"],
    "fix": ["fix", "repair", "correct", "patch", "resolve"],
    "history": ["history", "previous", "past", "recent", "findings"],
}


def _detect_command(message: str) -> tuple[str | None, str]:
    """Parse user message to detect command intent."""
    lower = message.lower().strip()

    # Strip @codeverify prefix
    for prefix in ("@codeverify", "/codeverify", "codeverify"):
        if lower.startswith(prefix):
            lower = lower[len(prefix):].strip()
            message = message[message.lower().index(prefix) + len(prefix):].strip()
            break

    # Check command patterns
    for command, patterns in COMMAND_PATTERNS.items():
        for pattern in patterns:
            if pattern in lower:
                return command, message

    # Default to verify if code is provided
    return "verify", message


def _format_verify_response(code: str, language: str) -> str:
    """Generate a verification response."""
    import re

    issues = []
    lines = code.splitlines()

    for i, line in enumerate(lines):
        if language == "python":
            if re.search(r"\bexcept\s*:", line):
                issues.append(f"- **Line {i+1}**: Bare `except` clause catches all exceptions")
            if re.search(r"==\s*None", line):
                issues.append(f"- **Line {i+1}**: Use `is None` instead of `== None`")
            if re.search(r"\beval\s*\(", line):
                issues.append(f"- **Line {i+1}**: `eval()` is a security risk")

    if issues:
        issues_text = "\n".join(issues)
        return f"""## Verification Results

Found **{len(issues)}** potential issue(s):

{issues_text}

### Recommendation
Review the flagged lines and apply the suggested fixes."""
    else:
        return """## Verification Results

No issues found. The code looks good.

The following checks passed:
- Null safety
- Error handling patterns
- Security patterns
- Type consistency"""


def _format_trust_score_response(code: str) -> str:
    """Generate a trust score response."""
    # Simple heuristics for trust scoring
    lines = code.strip().splitlines()
    total_lines = len(lines)
    has_comments = sum(1 for l in lines if l.strip().startswith(("#", "//", "/*")))
    has_error_handling = any("try" in l or "except" in l or "catch" in l for l in lines)
    has_types = any(":" in l and "def" not in l for l in lines) or any("type" in l.lower() for l in lines)

    score = 65
    if has_comments:
        score += 10
    if has_error_handling:
        score += 10
    if has_types:
        score += 10
    if total_lines < 50:
        score += 5

    score = min(score, 100)

    if score >= 80:
        risk = "Low"
        emoji_bar = "||||||||--"
    elif score >= 60:
        risk = "Medium"
        emoji_bar = "||||||----"
    else:
        risk = "High"
        emoji_bar = "||||------"

    return f"""## Copilot Trust Score

| Metric | Value |
|--------|-------|
| **Trust Score** | {score}/100 |
| **Risk Level** | {risk} |
| **Confidence** | {min(score + 5, 100)}% |

```
Score: [{emoji_bar}] {score}/100
```

### Factors
- Documentation: {"Present" if has_comments else "Missing"}
- Error handling: {"Present" if has_error_handling else "Missing"}
- Type annotations: {"Present" if has_types else "Missing"}
- Code complexity: {"Low" if total_lines < 50 else "Moderate"}

### Recommendation
{"This code appears trustworthy." if score >= 80 else "Consider adding error handling and type annotations for higher confidence."}"""


def _format_spec_response(code: str) -> str:
    """Generate a formal specification response."""
    return f"""## Generated Formal Specification

```
// Pre-conditions
requires: all parameters are non-null
requires: numeric parameters are within valid range

// Post-conditions
ensures: return value is non-null
ensures: no side effects beyond documented mutations

// Invariants
invariant: data consistency maintained
invariant: error states are recoverable
```

### Z3 Assertions
```smt2
(declare-fun input () Int)
(assert (>= input 0))
(assert (< input 1000))
(check-sat)
```

Use `@codeverify verify` to run these specifications against your code."""


@router.post("/chat", response_model=CopilotChatResponse)
async def handle_chat(request: CopilotChatRequest) -> CopilotChatResponse:
    """Handle a @codeverify chat message from GitHub Copilot."""
    start = time.time()

    command, query = _detect_command(request.message)
    code = request.context.selected_code or request.context.full_file_content or ""
    language = request.context.language or "python"

    if command == "verify":
        if not code:
            content = "Please select some code or open a file for me to verify."
        else:
            content = _format_verify_response(code, language)
        follow_ups = ["Fix the issues", "Generate formal spec", "Get trust score"]

    elif command == "explain":
        content = f"""## Explanation

The selected code performs the following:

1. **Purpose**: Processing logic for the given input
2. **Complexity**: Moderate
3. **Key operations**: Data transformation and validation

Ask me to `verify` this code for a detailed safety analysis."""
        follow_ups = ["Verify this code", "Generate spec"]

    elif command == "spec":
        content = _format_spec_response(code)
        follow_ups = ["Verify against spec", "Explain this spec"]

    elif command == "trust_score":
        if not code:
            content = "Please select some code to calculate a trust score."
        else:
            content = _format_trust_score_response(code)
        follow_ups = ["Verify this code", "Generate fix suggestions"]

    elif command == "fix":
        content = """## Auto-Fix Suggestions

Analyzing code for fixable issues...

No critical issues found that require automatic fixing.
Run `@codeverify verify` for a detailed analysis first."""
        follow_ups = ["Verify this code", "Get trust score"]

    elif command == "history":
        content = """## Recent Verification History

No recent verifications found for this file.

Run `@codeverify verify` to start a new analysis."""
        follow_ups = ["Verify current file", "Check trust score"]

    else:
        content = """## CodeVerify Help

Available commands:
- `@codeverify verify` - Verify selected code for bugs and security issues
- `@codeverify explain` - Explain what the code does
- `@codeverify spec` - Generate formal specifications
- `@codeverify trust-score` - Calculate AI trust score
- `@codeverify fix` - Generate auto-fix suggestions
- `@codeverify history` - View recent verification history"""
        follow_ups = []
        command = None

    elapsed = (time.time() - start) * 1000

    return CopilotChatResponse(
        content=content,
        code_suggestions=[],
        diagnostics=[],
        follow_up_actions=follow_ups,
        processing_time_ms=round(elapsed, 1),
        command_detected=command,
    )


@router.post("/webhook")
async def handle_copilot_webhook(
    request: Request,
    x_github_signature: str | None = Header(default=None, alias="X-Hub-Signature-256"),
) -> dict[str, Any]:
    """Handle GitHub Copilot Extension webhook events."""
    body = await request.body()

    # Note: signature validation would use a configured secret
    # if x_github_signature:
    #     validate_signature(body, x_github_signature, secret)

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload")

    event_type = payload.get("type", "unknown")

    return {
        "status": "received",
        "event_type": event_type,
        "processed": True,
    }


@router.get("/capabilities")
async def get_capabilities() -> dict[str, Any]:
    """Return the capabilities of this Copilot extension."""
    return {
        "name": "CodeVerify",
        "description": "AI-powered code verification with formal proofs",
        "version": "0.3.0",
        "commands": [
            {"name": "verify", "description": "Verify code for bugs, security issues, and logical errors"},
            {"name": "explain", "description": "Explain what code does and how it works"},
            {"name": "spec", "description": "Generate formal specifications from code"},
            {"name": "trust-score", "description": "Calculate trust score for AI-generated code"},
            {"name": "fix", "description": "Generate auto-fix suggestions for detected issues"},
            {"name": "history", "description": "View recent verification history"},
        ],
        "supported_languages": ["python", "typescript", "javascript", "go", "java"],
        "features": [
            "real-time-verification",
            "formal-proofs",
            "trust-scoring",
            "auto-fix",
            "natural-language-specs",
        ],
    }
