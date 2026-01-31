"""
AI-to-AI Collaboration API Router

Provides REST API endpoints for AI assistant collaboration:
- Session management
- Constraint streaming
- Suggestion verification
- Real-time feedback
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

# In production, would import from installed package
# from codeverify_core.ai_collaboration import (
#     CopilotCollaborator,
#     CollaborationSession,
#     VerificationConstraint,
# )


router = APIRouter(prefix="/api/v1/collaboration", tags=["ai-collaboration"])


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateSessionRequest(BaseModel):
    """Request to create a collaboration session."""
    file_path: str = Field(..., description="Path to the file being edited")
    language: str = Field(..., description="Programming language")
    ai_assistant: str = Field(
        "github_copilot", description="AI assistant identifier"
    )
    project_context: Optional[Dict[str, Any]] = Field(
        None, description="Additional project context"
    )


class SessionResponse(BaseModel):
    """Response with session details."""
    session_id: str
    ai_assistant: str
    file_path: str
    language: str
    active_constraints: List[Dict[str, Any]]
    context_prompt: str


class AddConstraintRequest(BaseModel):
    """Request to add a constraint to a session."""
    constraint_type: str = Field(..., description="Type of constraint")
    rule: str = Field(..., description="Constraint rule")
    severity: str = Field("medium", description="Severity level")
    natural_language: Optional[str] = Field(
        None, description="Human-readable description"
    )
    formal_spec: Optional[str] = Field(
        None, description="Formal specification (Z3/SMT-LIB)"
    )
    example_violation: Optional[str] = Field(None, description="Example of violation")
    example_correct: Optional[str] = Field(None, description="Correct example")


class VerifySuggestionRequest(BaseModel):
    """Request to verify an AI suggestion."""
    session_id: str = Field(..., description="Session ID")
    suggestion: str = Field(..., description="Code suggestion to verify")
    context: Optional[Dict[str, Any]] = Field(
        None, description="Additional context"
    )


class VerificationResult(BaseModel):
    """Result of suggestion verification."""
    verified: bool
    proposal_id: str
    issues: List[Dict[str, Any]]
    suggestions: List[str]
    should_block: bool


class StreamCodeRequest(BaseModel):
    """Request to stream code for real-time verification."""
    session_id: str
    code_chunk: str


class SessionStatsResponse(BaseModel):
    """Session statistics response."""
    session_id: str
    duration_seconds: float
    constraints_sent: int
    proposals_received: int
    proposals_accepted: int
    proposals_rejected: int
    acceptance_rate: float


# =============================================================================
# In-Memory State (would use database in production)
# =============================================================================

# Store active sessions
_sessions: Dict[str, Dict[str, Any]] = {}

# Store constraint messages
_message_history: Dict[str, List[Dict[str, Any]]] = {}

# WebSocket connections
_ws_connections: Dict[str, List[WebSocket]] = {}

# Standard constraints
STANDARD_CONSTRAINTS = {
    "null_safety": {
        "constraint_id": "null_safety",
        "constraint_type": "type_safety",
        "rule": "Never return null/None without explicit documentation",
        "severity": "high",
        "natural_language": "Functions should not return null/None unless the return type explicitly allows it",
        "example_violation": "def get_user(id): return None  # Missing Optional type",
        "example_correct": "def get_user(id) -> Optional[User]: return None  # OK",
    },
    "bounds_check": {
        "constraint_id": "bounds_check",
        "constraint_type": "memory_safety",
        "rule": "Array/list access must be bounds-checked",
        "severity": "high",
        "natural_language": "Before accessing array elements, verify the index is within bounds",
        "example_violation": "items[i]  # i could be out of bounds",
        "example_correct": "if 0 <= i < len(items): items[i]",
    },
    "division_safety": {
        "constraint_id": "division_safety",
        "constraint_type": "arithmetic_safety",
        "rule": "Division must check for zero divisor",
        "severity": "critical",
        "natural_language": "Before dividing, ensure the divisor is not zero",
        "example_violation": "result = a / b  # b could be zero",
        "example_correct": "result = a / b if b != 0 else 0",
    },
    "sql_injection": {
        "constraint_id": "sql_injection",
        "constraint_type": "security",
        "rule": "Use parameterized queries for SQL",
        "severity": "critical",
        "natural_language": "Never concatenate user input into SQL queries",
        "example_violation": 'query = f"SELECT * FROM users WHERE id={user_id}"',
        "example_correct": 'cursor.execute("SELECT * FROM users WHERE id=?", (user_id,))',
    },
    "resource_cleanup": {
        "constraint_id": "resource_cleanup",
        "constraint_type": "resource_safety",
        "rule": "Resources must be properly closed",
        "severity": "medium",
        "natural_language": "File handles, connections, and other resources must be closed",
        "example_violation": "f = open('file.txt'); data = f.read()  # Never closed",
        "example_correct": "with open('file.txt') as f: data = f.read()",
    },
}


# =============================================================================
# API Endpoints
# =============================================================================

@router.post(
    "/sessions",
    response_model=SessionResponse,
    summary="Create Collaboration Session",
    description="Create a new AI-to-AI collaboration session"
)
async def create_session(request: CreateSessionRequest) -> SessionResponse:
    """
    Create a new collaboration session.
    
    Returns session ID and initial context to inject into AI assistant.
    """
    session_id = hashlib.sha256(
        f"{request.file_path}-{time.time()}".encode()
    ).hexdigest()[:16]
    
    # Initialize session
    session = {
        "session_id": session_id,
        "ai_assistant": request.ai_assistant,
        "file_path": request.file_path,
        "language": request.language,
        "project_context": request.project_context or {},
        "created_at": time.time(),
        "active_constraints": {},
        "stats": {
            "constraints_sent": 0,
            "proposals_received": 0,
            "proposals_accepted": 0,
            "proposals_rejected": 0,
        },
    }
    
    # Add default constraints based on language
    default_constraints = ["null_safety", "bounds_check", "division_safety"]
    if request.language in ("python", "javascript", "typescript"):
        default_constraints.append("sql_injection")
        default_constraints.append("resource_cleanup")
    
    for cid in default_constraints:
        if cid in STANDARD_CONSTRAINTS:
            session["active_constraints"][cid] = STANDARD_CONSTRAINTS[cid]
            session["stats"]["constraints_sent"] += 1
    
    _sessions[session_id] = session
    _message_history[session_id] = []
    
    # Generate context prompt for AI
    context_prompt = _generate_context_prompt(session)
    
    return SessionResponse(
        session_id=session_id,
        ai_assistant=request.ai_assistant,
        file_path=request.file_path,
        language=request.language,
        active_constraints=list(session["active_constraints"].values()),
        context_prompt=context_prompt,
    )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    summary="Get Session",
    description="Get collaboration session details"
)
async def get_session(session_id: str) -> SessionResponse:
    """Get session details."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(
        session_id=session_id,
        ai_assistant=session["ai_assistant"],
        file_path=session["file_path"],
        language=session["language"],
        active_constraints=list(session["active_constraints"].values()),
        context_prompt=_generate_context_prompt(session),
    )


@router.delete(
    "/sessions/{session_id}",
    response_model=SessionStatsResponse,
    summary="End Session",
    description="End a collaboration session and get statistics"
)
async def end_session(session_id: str) -> SessionStatsResponse:
    """End session and return statistics."""
    session = _sessions.pop(session_id, None)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    _message_history.pop(session_id, None)
    
    stats = session["stats"]
    duration = time.time() - session["created_at"]
    
    return SessionStatsResponse(
        session_id=session_id,
        duration_seconds=duration,
        constraints_sent=stats["constraints_sent"],
        proposals_received=stats["proposals_received"],
        proposals_accepted=stats["proposals_accepted"],
        proposals_rejected=stats["proposals_rejected"],
        acceptance_rate=(
            stats["proposals_accepted"] / stats["proposals_received"]
            if stats["proposals_received"] > 0
            else 0.0
        ),
    )


@router.post(
    "/sessions/{session_id}/constraints",
    summary="Add Constraint",
    description="Add a verification constraint to the session"
)
async def add_constraint(
    session_id: str,
    request: AddConstraintRequest,
) -> Dict[str, Any]:
    """Add a constraint to the session."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    constraint_id = hashlib.sha256(
        f"{request.constraint_type}-{request.rule}".encode()
    ).hexdigest()[:12]
    
    constraint = {
        "constraint_id": constraint_id,
        "constraint_type": request.constraint_type,
        "rule": request.rule,
        "severity": request.severity,
        "natural_language": request.natural_language,
        "formal_spec": request.formal_spec,
        "example_violation": request.example_violation,
        "example_correct": request.example_correct,
    }
    
    session["active_constraints"][constraint_id] = constraint
    session["stats"]["constraints_sent"] += 1
    
    # Broadcast to WebSocket connections
    await _broadcast_to_session(session_id, {
        "type": "constraint_added",
        "constraint": constraint,
    })
    
    return {
        "constraint_id": constraint_id,
        "added": True,
    }


@router.delete(
    "/sessions/{session_id}/constraints/{constraint_id}",
    summary="Remove Constraint",
    description="Remove a constraint from the session"
)
async def remove_constraint(
    session_id: str,
    constraint_id: str,
) -> Dict[str, Any]:
    """Remove a constraint from the session."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if constraint_id not in session["active_constraints"]:
        raise HTTPException(status_code=404, detail="Constraint not found")
    
    del session["active_constraints"][constraint_id]
    
    await _broadcast_to_session(session_id, {
        "type": "constraint_removed",
        "constraint_id": constraint_id,
    })
    
    return {"removed": True, "constraint_id": constraint_id}


@router.post(
    "/verify",
    response_model=VerificationResult,
    summary="Verify Suggestion",
    description="Verify an AI-generated code suggestion"
)
async def verify_suggestion(request: VerifySuggestionRequest) -> VerificationResult:
    """
    Verify a code suggestion against active constraints.
    
    Returns verification result with issues and improvement suggestions.
    """
    session = _sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session["stats"]["proposals_received"] += 1
    
    # Generate proposal ID
    proposal_id = hashlib.sha256(
        f"{request.session_id}-{time.time()}-{request.suggestion[:50]}".encode()
    ).hexdigest()[:16]
    
    # Analyze code against constraints
    issues = _analyze_code(request.suggestion, session["language"])
    
    # Filter issues based on active constraints
    active_types = {
        c["constraint_type"] for c in session["active_constraints"].values()
    }
    issues = [i for i in issues if i.get("constraint_type") in active_types or True]
    
    # Generate suggestions
    suggestions = []
    for issue in issues:
        constraint_id = issue.get("constraint_id")
        if constraint_id in STANDARD_CONSTRAINTS:
            constraint = STANDARD_CONSTRAINTS[constraint_id]
            if constraint.get("example_correct"):
                suggestions.append(f"Consider: {constraint['example_correct']}")
    
    verified = len(issues) == 0
    should_block = any(i.get("severity") == "critical" for i in issues)
    
    if verified:
        session["stats"]["proposals_accepted"] += 1
    else:
        session["stats"]["proposals_rejected"] += 1
    
    # Store in message history
    _message_history[request.session_id].append({
        "type": "verification",
        "proposal_id": proposal_id,
        "verified": verified,
        "issues": issues,
        "timestamp": time.time(),
    })
    
    # Broadcast result
    await _broadcast_to_session(request.session_id, {
        "type": "verification_result",
        "proposal_id": proposal_id,
        "verified": verified,
        "issues": issues,
    })
    
    return VerificationResult(
        verified=verified,
        proposal_id=proposal_id,
        issues=issues,
        suggestions=suggestions,
        should_block=should_block,
    )


@router.post(
    "/stream",
    summary="Stream Code Chunk",
    description="Stream a code chunk for real-time verification"
)
async def stream_code(request: StreamCodeRequest) -> Dict[str, Any]:
    """
    Stream code for real-time verification.
    
    Analyzes code chunk and returns any immediate issues.
    """
    session = _sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Analyze the chunk
    issues = _analyze_code(request.code_chunk, session["language"])
    
    # Filter to only critical/high for streaming
    urgent_issues = [
        i for i in issues
        if i.get("severity") in ("critical", "high")
    ]
    
    if urgent_issues:
        await _broadcast_to_session(request.session_id, {
            "type": "stream_warning",
            "issues": urgent_issues,
        })
    
    return {
        "analyzed": True,
        "issues": urgent_issues,
    }


@router.get(
    "/sessions/{session_id}/context",
    summary="Get AI Context",
    description="Get context string to inject into AI assistant prompt"
)
async def get_ai_context(session_id: str) -> Dict[str, Any]:
    """Get the context string for AI assistant injection."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "context_prompt": _generate_context_prompt(session),
        "constraint_count": len(session["active_constraints"]),
    }


@router.get(
    "/sessions/{session_id}/history",
    summary="Get Message History",
    description="Get collaboration message history for the session"
)
async def get_history(
    session_id: str,
    limit: int = 50,
) -> Dict[str, Any]:
    """Get message history for the session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    history = _message_history.get(session_id, [])
    
    return {
        "session_id": session_id,
        "messages": history[-limit:],
        "total": len(history),
    }


@router.get(
    "/constraints/standard",
    summary="List Standard Constraints",
    description="Get list of standard verification constraints"
)
async def list_standard_constraints() -> Dict[str, Any]:
    """List all standard constraints."""
    return {
        "constraints": list(STANDARD_CONSTRAINTS.values()),
        "count": len(STANDARD_CONSTRAINTS),
    }


# =============================================================================
# WebSocket for Real-Time Collaboration
# =============================================================================

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket for real-time collaboration.
    
    Streams verification updates to connected AI assistants.
    """
    if session_id not in _sessions:
        await websocket.close(code=4004, reason="Session not found")
        return
    
    await websocket.accept()
    
    # Register connection
    if session_id not in _ws_connections:
        _ws_connections[session_id] = []
    _ws_connections[session_id].append(websocket)
    
    try:
        # Send initial context
        session = _sessions[session_id]
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "active_constraints": list(session["active_constraints"].values()),
        })
        
        # Handle incoming messages
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "code_chunk":
                # Analyze code chunk
                issues = _analyze_code(
                    data.get("code", ""),
                    session["language"],
                )
                urgent = [i for i in issues if i.get("severity") in ("critical", "high")]
                
                if urgent:
                    await websocket.send_json({
                        "type": "warning",
                        "issues": urgent,
                    })
            
            elif data.get("type") == "verify":
                # Full verification
                code = data.get("code", "")
                issues = _analyze_code(code, session["language"])
                
                await websocket.send_json({
                    "type": "verification_result",
                    "verified": len(issues) == 0,
                    "issues": issues,
                })
            
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        pass
    finally:
        # Remove connection
        if session_id in _ws_connections:
            _ws_connections[session_id] = [
                ws for ws in _ws_connections[session_id]
                if ws != websocket
            ]


# =============================================================================
# Helper Functions
# =============================================================================

def _generate_context_prompt(session: Dict[str, Any]) -> str:
    """Generate context prompt for AI assistant."""
    lines = [
        "=== CodeVerify Collaboration Context ===",
        f"Session: {session['session_id']}",
        f"File: {session['file_path']}",
        f"Language: {session['language']}",
        "",
        "Active Verification Constraints:",
    ]
    
    for constraint in session["active_constraints"].values():
        lines.append(f"  - [{constraint['severity'].upper()}] {constraint['rule']}")
        if constraint.get("natural_language"):
            lines.append(f"    Description: {constraint['natural_language']}")
    
    lines.extend([
        "",
        "IMPORTANT: Generated code MUST satisfy all constraints above.",
        "=== End CodeVerify Context ===",
    ])
    
    return "\n".join(lines)


async def _broadcast_to_session(session_id: str, message: Dict[str, Any]) -> None:
    """Broadcast a message to all WebSocket connections for a session."""
    connections = _ws_connections.get(session_id, [])
    for ws in connections:
        try:
            await ws.send_json(message)
        except Exception:
            pass


def _analyze_code(code: str, language: str) -> List[Dict[str, Any]]:
    """Analyze code for constraint violations."""
    import re
    
    issues = []
    
    # Null safety check
    if language == "python":
        if "return None" in code and "Optional" not in code and "| None" not in code:
            issues.append({
                "constraint_id": "null_safety",
                "constraint_type": "type_safety",
                "severity": "high",
                "message": "Function may return None without Optional type annotation",
                "suggestion": "Add Optional[] to return type or use | None",
            })
    
    # Division safety
    divisions = re.findall(r"(\w+)\s*/\s*(\w+)", code)
    for _, divisor in divisions:
        if divisor not in ("2", "10", "100", "1000", "2.0", "10.0"):
            if f"if {divisor}" not in code and f"{divisor} != 0" not in code:
                issues.append({
                    "constraint_id": "division_safety",
                    "constraint_type": "arithmetic_safety",
                    "severity": "critical",
                    "message": f"Division by '{divisor}' without zero check",
                    "suggestion": f"Add check: 'if {divisor} != 0'",
                })
    
    # SQL injection
    sql_patterns = [
        r'f"[^"]*(?:SELECT|INSERT|UPDATE|DELETE)[^"]*\{',
        r"f'[^']*(?:SELECT|INSERT|UPDATE|DELETE)[^']*\{",
        r'"[^"]*(?:SELECT|INSERT|UPDATE|DELETE)[^"]*"\s*%',
    ]
    for pattern in sql_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            issues.append({
                "constraint_id": "sql_injection",
                "constraint_type": "security",
                "severity": "critical",
                "message": "Potential SQL injection vulnerability",
                "suggestion": "Use parameterized queries instead of string formatting",
            })
            break
    
    # Resource cleanup
    if "open(" in code and "with " not in code:
        issues.append({
            "constraint_id": "resource_cleanup",
            "constraint_type": "resource_safety",
            "severity": "medium",
            "message": "File opened without context manager",
            "suggestion": "Use 'with open(...) as f:' pattern",
        })
    
    # Bounds check
    array_access = re.findall(r"(\w+)\[(\w+)\]", code)
    checked_indices: set = set()
    for array, index in array_access:
        if index.isdigit():
            continue
        if index in checked_indices:
            continue
        if f"len({array})" in code or f"range(len({array}))" in code:
            continue
        if f"if {index} <" in code or f"0 <= {index}" in code:
            continue
        
        issues.append({
            "constraint_id": "bounds_check",
            "constraint_type": "memory_safety",
            "severity": "high",
            "message": f"Array access '{array}[{index}]' may be out of bounds",
            "suggestion": f"Add bounds check: 'if 0 <= {index} < len({array})'",
        })
        checked_indices.add(index)
    
    return issues
