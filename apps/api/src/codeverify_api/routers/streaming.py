"""Real-time streaming verification API router (SSE-based)."""

import asyncio
import json
import re
import time
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter()


class StreamingVerificationRequest(BaseModel):
    code: str = Field(description="Source code to verify")
    language: str = Field(default="python")
    file_path: str = Field(default="untitled")
    stages: list[str] = Field(
        default=["pattern", "ai", "formal"],
        description="Verification stages to run: pattern, ai, formal",
    )


class DiagnosticModel(BaseModel):
    line: int
    character: int
    end_line: int
    end_character: int
    severity: int = Field(description="1=error, 2=warning, 3=info, 4=hint")
    message: str
    source: str = "codeverify"
    code: str | None = None
    stage: str


# Pattern-based checks for quick analysis
PATTERNS: dict[str, list[dict[str, Any]]] = {
    "python": [
        {"pattern": r"\bexcept\s*:", "message": "Bare except clause catches all exceptions including SystemExit", "severity": 2, "code": "W001"},
        {"pattern": r"def\s+\w+\(.*?=\s*\[\]", "message": "Mutable default argument (list)", "severity": 2, "code": "W002"},
        {"pattern": r"def\s+\w+\(.*?=\s*\{\}", "message": "Mutable default argument (dict)", "severity": 2, "code": "W003"},
        {"pattern": r"==\s*None\b", "message": "Use 'is None' instead of '== None'", "severity": 3, "code": "W004"},
        {"pattern": r"!=\s*None\b", "message": "Use 'is not None' instead of '!= None'", "severity": 3, "code": "W005"},
        {"pattern": r"\beval\s*\(", "message": "Use of eval() is a security risk", "severity": 1, "code": "S001"},
        {"pattern": r"\bexec\s*\(", "message": "Use of exec() is a security risk", "severity": 1, "code": "S002"},
        {"pattern": r"password\s*=\s*['\"]", "message": "Hardcoded password detected", "severity": 1, "code": "S003"},
        {"pattern": r"import\s+pickle", "message": "pickle can execute arbitrary code during deserialization", "severity": 2, "code": "S004"},
        {"pattern": r"\.format\(.*\)", "message": "Consider using f-strings for better readability", "severity": 4, "code": "I001"},
    ],
    "typescript": [
        {"pattern": r":\s*any\b", "message": "Avoid using 'any' type", "severity": 2, "code": "TS001"},
        {"pattern": r"!\.", "message": "Non-null assertion operator (!) may hide null errors", "severity": 2, "code": "TS002"},
        {"pattern": r"==\s", "message": "Use === instead of == for strict equality", "severity": 2, "code": "TS003"},
        {"pattern": r"!=\s", "message": "Use !== instead of != for strict inequality", "severity": 2, "code": "TS004"},
        {"pattern": r"console\.(log|debug|warn)\(", "message": "Remove console statement before production", "severity": 3, "code": "TS005"},
        {"pattern": r"@ts-ignore", "message": "Avoid @ts-ignore - fix the type error instead", "severity": 2, "code": "TS006"},
        {"pattern": r"var\s+\w+", "message": "Use 'let' or 'const' instead of 'var'", "severity": 2, "code": "TS007"},
    ],
    "go": [
        {"pattern": r"\b_\s*=\s*\w+\(", "message": "Error return value ignored", "severity": 2, "code": "GO001"},
        {"pattern": r"fmt\.Print(ln|f)?\(", "message": "Use structured logging instead of fmt.Print", "severity": 3, "code": "GO002"},
        {"pattern": r"panic\(", "message": "Avoid panic() in library code - return errors instead", "severity": 2, "code": "GO003"},
        {"pattern": r"os\.Exit\(", "message": "os.Exit() prevents defer from running", "severity": 2, "code": "GO004"},
        {"pattern": r"\.\(\*?\w+\)\s*$", "message": "Unchecked type assertion - use comma-ok pattern", "severity": 2, "code": "GO005"},
    ],
    "java": [
        {"pattern": r"catch\s*\(\s*Exception\s+\w+\s*\)\s*\{\s*\}", "message": "Empty catch block swallows exceptions", "severity": 1, "code": "J001"},
        {"pattern": r"System\.out\.print", "message": "Use a logging framework instead of System.out", "severity": 3, "code": "J002"},
        {"pattern": r"\bnew\s+Date\(\)", "message": "Use java.time API instead of legacy Date", "severity": 3, "code": "J003"},
        {"pattern": r"catch\s*\(\s*Throwable\b", "message": "Catching Throwable catches errors that shouldn't be caught", "severity": 1, "code": "J004"},
        {"pattern": r"@SuppressWarnings", "message": "Review suppressed warnings", "severity": 3, "code": "J005"},
    ],
}


def _run_pattern_stage(code: str, language: str) -> list[dict[str, Any]]:
    """Run fast pattern matching (<100ms)."""
    diagnostics = []
    patterns = PATTERNS.get(language, [])
    lines = code.splitlines()

    for i, line in enumerate(lines):
        for p in patterns:
            if re.search(p["pattern"], line):
                diagnostics.append({
                    "line": i + 1,
                    "character": 0,
                    "end_line": i + 1,
                    "end_character": len(line),
                    "severity": p["severity"],
                    "message": p["message"],
                    "source": "codeverify",
                    "code": p["code"],
                    "stage": "pattern",
                })

    return diagnostics


def _run_ai_stage(code: str, language: str) -> list[dict[str, Any]]:
    """Simulate AI analysis stage (~2s)."""
    diagnostics = []
    lines = code.splitlines()

    # Detect functions without docstrings (for Python)
    if language == "python":
        for i, line in enumerate(lines):
            if re.match(r"^\s*def\s+\w+", line):
                # Check next non-empty line for docstring
                has_docstring = False
                for j in range(i + 1, min(i + 3, len(lines))):
                    stripped = lines[j].strip()
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        has_docstring = True
                        break
                    if stripped and not stripped.startswith("#"):
                        break
                if not has_docstring:
                    diagnostics.append({
                        "line": i + 1,
                        "character": 0,
                        "end_line": i + 1,
                        "end_character": len(line),
                        "severity": 3,
                        "message": f"Function missing docstring",
                        "source": "codeverify-ai",
                        "code": "AI001",
                        "stage": "ai",
                    })

    return diagnostics


def _run_formal_stage(code: str, language: str) -> list[dict[str, Any]]:
    """Simulate formal verification stage (~5s)."""
    diagnostics = []
    lines = code.splitlines()

    # Detect potential division by zero
    for i, line in enumerate(lines):
        if "/" in line and "import" not in line and "#" not in line.split("/")[0]:
            if re.search(r"\b\w+\s*/\s*\w+", line):
                diagnostics.append({
                    "line": i + 1,
                    "character": 0,
                    "end_line": i + 1,
                    "end_character": len(line),
                    "severity": 2,
                    "message": "Potential division by zero (formal verification pending)",
                    "source": "codeverify-z3",
                    "code": "Z3001",
                    "stage": "formal",
                })

    return diagnostics


async def _stream_verification(
    code: str,
    language: str,
    file_path: str,
    stages: list[str],
) -> AsyncGenerator[str, None]:
    """Stream verification results as SSE events."""
    total_stages = len(stages)

    # Stage 1: Pattern matching (fast)
    if "pattern" in stages:
        yield f"data: {json.dumps({'type': 'stage_start', 'stage': 'pattern', 'progress': 0})}\n\n"
        start = time.time()
        diagnostics = _run_pattern_stage(code, language)
        elapsed = (time.time() - start) * 1000
        yield f"data: {json.dumps({'type': 'stage_complete', 'stage': 'pattern', 'diagnostics': diagnostics, 'elapsed_ms': round(elapsed, 1), 'progress': 1 / total_stages})}\n\n"

    # Stage 2: AI analysis
    if "ai" in stages:
        yield f"data: {json.dumps({'type': 'stage_start', 'stage': 'ai', 'progress': 1 / total_stages})}\n\n"
        await asyncio.sleep(0.1)  # Simulate LLM latency
        start = time.time()
        diagnostics = _run_ai_stage(code, language)
        elapsed = (time.time() - start) * 1000
        yield f"data: {json.dumps({'type': 'stage_complete', 'stage': 'ai', 'diagnostics': diagnostics, 'elapsed_ms': round(elapsed, 1), 'progress': 2 / total_stages})}\n\n"

    # Stage 3: Formal verification
    if "formal" in stages:
        yield f"data: {json.dumps({'type': 'stage_start', 'stage': 'formal', 'progress': 2 / total_stages})}\n\n"
        await asyncio.sleep(0.05)  # Simulate Z3 latency
        start = time.time()
        diagnostics = _run_formal_stage(code, language)
        elapsed = (time.time() - start) * 1000
        yield f"data: {json.dumps({'type': 'stage_complete', 'stage': 'formal', 'diagnostics': diagnostics, 'elapsed_ms': round(elapsed, 1), 'progress': 1.0})}\n\n"

    yield f"data: {json.dumps({'type': 'complete', 'file_path': file_path, 'progress': 1.0})}\n\n"


@router.post("/verify-stream")
async def stream_verification(request: StreamingVerificationRequest) -> StreamingResponse:
    """Stream verification results using Server-Sent Events (SSE).

    Returns progressive results as each verification stage completes:
    - Pattern matching (~100ms)
    - AI analysis (~2s)
    - Formal verification (~5s)
    """
    return StreamingResponse(
        _stream_verification(request.code, request.language, request.file_path, request.stages),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/verify-sync", response_model=list[DiagnosticModel])
async def sync_verification(request: StreamingVerificationRequest) -> list[DiagnosticModel]:
    """Run all verification stages synchronously and return combined results."""
    all_diagnostics: list[dict[str, Any]] = []

    if "pattern" in request.stages:
        all_diagnostics.extend(_run_pattern_stage(request.code, request.language))
    if "ai" in request.stages:
        all_diagnostics.extend(_run_ai_stage(request.code, request.language))
    if "formal" in request.stages:
        all_diagnostics.extend(_run_formal_stage(request.code, request.language))

    return [DiagnosticModel(**d) for d in all_diagnostics]


@router.get("/patterns/{language}")
async def get_patterns(language: str) -> dict[str, Any]:
    """Get available pattern checks for a language."""
    patterns = PATTERNS.get(language)
    if patterns is None:
        return {"language": language, "patterns": [], "supported": False}

    return {
        "language": language,
        "patterns": [
            {"code": p["code"], "message": p["message"], "severity": p["severity"]}
            for p in patterns
        ],
        "total": len(patterns),
        "supported": True,
    }
