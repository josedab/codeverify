"""
Verification Replay & Time-Travel API Router

Provides REST API endpoints for verification replay:
- Session recording
- Replay with modified parameters
- State comparison
- Audit logs
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter(prefix="/api/v1/replay", tags=["verification-replay"])


# =============================================================================
# Request/Response Models
# =============================================================================

class StartSessionRequest(BaseModel):
    """Request to start recording session."""
    user_id: Optional[str] = Field(None, description="User ID")
    repository: Optional[str] = Field(None, description="Repository")
    file_path: Optional[str] = Field(None, description="File path")


class RecordEventRequest(BaseModel):
    """Request to record an event."""
    event_type: str = Field(..., description="Event type")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    parent_event_id: Optional[str] = Field(None, description="Parent event ID")


class CreateCheckpointRequest(BaseModel):
    """Request to create a checkpoint."""
    code: str = Field("", description="Current code")
    language: str = Field("python", description="Language")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters")
    constraints: List[Dict[str, Any]] = Field(
        default_factory=list, description="Active constraints"
    )
    issues: List[Dict[str, Any]] = Field(default_factory=list, description="Issues found")
    proofs: List[Dict[str, Any]] = Field(default_factory=list, description="Proofs generated")


class ReplayRequest(BaseModel):
    """Request to replay a session."""
    session_id: str = Field(..., description="Session to replay")
    modified_parameters: Optional[Dict[str, Any]] = Field(
        None, description="Modified parameters for replay"
    )
    start_from_event: Optional[int] = Field(None, description="Start from event sequence")
    end_at_event: Optional[int] = Field(None, description="End at event sequence")


class CompareStatesRequest(BaseModel):
    """Request to compare states."""
    state_a: Dict[str, Any] = Field(..., description="First state")
    state_b: Dict[str, Any] = Field(..., description="Second state")


class SessionResponse(BaseModel):
    """Session response."""
    session_id: str
    started_at: float
    ended_at: Optional[float]
    user_id: Optional[str]
    repository: Optional[str]
    event_count: int
    checkpoint_count: int


class EventResponse(BaseModel):
    """Event response."""
    event_id: str
    event_type: str
    timestamp: float
    sequence_number: int


class ComparisonResponse(BaseModel):
    """State comparison response."""
    code_diff: Optional[str]
    parameter_changes: Dict[str, Dict[str, Any]]
    constraint_changes: Dict[str, Any]
    issues_added: List[Dict[str, Any]]
    issues_removed: List[Dict[str, Any]]
    proofs_added: List[Dict[str, Any]]
    proofs_removed: List[Dict[str, Any]]


# =============================================================================
# In-Memory State
# =============================================================================

# Active recording sessions
_active_sessions: Dict[str, Dict[str, Any]] = {}

# Stored sessions (completed)
_stored_sessions: Dict[str, Dict[str, Any]] = {}

# Event counters
_event_counters: Dict[str, int] = {}

# Event types
EVENT_TYPES = [
    "session_start", "session_end", "code_submitted", "constraint_added",
    "constraint_removed", "verification_start", "verification_complete",
    "issue_found", "proof_generated", "parameter_changed", "state_checkpoint"
]


# =============================================================================
# API Endpoints
# =============================================================================

@router.post(
    "/sessions",
    response_model=SessionResponse,
    summary="Start Recording Session",
    description="Start a new verification recording session"
)
async def start_session(request: StartSessionRequest) -> SessionResponse:
    """Start a new recording session."""
    session_id = hashlib.sha256(
        f"{time.time()}-{request.user_id or 'anon'}".encode()
    ).hexdigest()[:16]
    
    session = {
        "session_id": session_id,
        "started_at": time.time(),
        "ended_at": None,
        "user_id": request.user_id,
        "repository": request.repository,
        "file_path": request.file_path,
        "events": [],
        "checkpoints": [],
        "total_issues_found": 0,
        "total_proofs_generated": 0,
    }
    
    _active_sessions[session_id] = session
    _event_counters[session_id] = 0
    
    # Record session start event
    await record_event(
        session_id,
        RecordEventRequest(
            event_type="session_start",
            data={
                "user_id": request.user_id,
                "repository": request.repository,
                "file_path": request.file_path,
            },
        ),
    )
    
    return SessionResponse(
        session_id=session_id,
        started_at=session["started_at"],
        ended_at=None,
        user_id=request.user_id,
        repository=request.repository,
        event_count=len(session["events"]),
        checkpoint_count=len(session["checkpoints"]),
    )


@router.post(
    "/sessions/{session_id}/end",
    response_model=SessionResponse,
    summary="End Recording Session",
    description="End a recording session"
)
async def end_session(session_id: str) -> SessionResponse:
    """End a recording session."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = _active_sessions[session_id]
    session["ended_at"] = time.time()
    
    # Record session end event
    await record_event(
        session_id,
        RecordEventRequest(
            event_type="session_end",
            data={
                "duration_seconds": session["ended_at"] - session["started_at"],
                "total_events": len(session["events"]),
            },
        ),
    )
    
    # Move to stored sessions
    _stored_sessions[session_id] = session
    del _active_sessions[session_id]
    del _event_counters[session_id]
    
    return SessionResponse(
        session_id=session_id,
        started_at=session["started_at"],
        ended_at=session["ended_at"],
        user_id=session.get("user_id"),
        repository=session.get("repository"),
        event_count=len(session["events"]),
        checkpoint_count=len(session["checkpoints"]),
    )


@router.get(
    "/sessions/{session_id}",
    summary="Get Session",
    description="Get session details"
)
async def get_session(session_id: str) -> Dict[str, Any]:
    """Get session details."""
    session = _active_sessions.get(session_id) or _stored_sessions.get(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session["session_id"],
        "started_at": session["started_at"],
        "ended_at": session.get("ended_at"),
        "user_id": session.get("user_id"),
        "repository": session.get("repository"),
        "file_path": session.get("file_path"),
        "event_count": len(session["events"]),
        "checkpoint_count": len(session["checkpoints"]),
        "total_issues_found": session.get("total_issues_found", 0),
        "total_proofs_generated": session.get("total_proofs_generated", 0),
        "is_active": session_id in _active_sessions,
    }


@router.get(
    "/sessions",
    summary="List Sessions",
    description="List recording sessions"
)
async def list_sessions(
    user_id: Optional[str] = None,
    repository: Optional[str] = None,
    include_active: bool = True,
    include_stored: bool = True,
    limit: int = 50,
) -> Dict[str, Any]:
    """List recording sessions."""
    sessions = []
    
    if include_active:
        sessions.extend(_active_sessions.values())
    
    if include_stored:
        sessions.extend(_stored_sessions.values())
    
    # Filter
    if user_id:
        sessions = [s for s in sessions if s.get("user_id") == user_id]
    
    if repository:
        sessions = [s for s in sessions if s.get("repository") == repository]
    
    # Sort by start time descending
    sessions.sort(key=lambda s: s["started_at"], reverse=True)
    
    return {
        "sessions": [
            {
                "session_id": s["session_id"],
                "started_at": s["started_at"],
                "ended_at": s.get("ended_at"),
                "user_id": s.get("user_id"),
                "repository": s.get("repository"),
                "event_count": len(s["events"]),
                "is_active": s["session_id"] in _active_sessions,
            }
            for s in sessions[:limit]
        ],
        "total": len(sessions),
    }


@router.post(
    "/sessions/{session_id}/events",
    response_model=EventResponse,
    summary="Record Event",
    description="Record an event in a session"
)
async def record_event(
    session_id: str,
    request: RecordEventRequest,
) -> EventResponse:
    """Record an event in a session."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail="Active session not found")
    
    if request.event_type not in EVENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid event type. Must be one of: {EVENT_TYPES}"
        )
    
    session = _active_sessions[session_id]
    
    _event_counters[session_id] += 1
    seq = _event_counters[session_id]
    
    event = {
        "event_id": f"{session_id}-{seq:06d}",
        "event_type": request.event_type,
        "timestamp": time.time(),
        "data": request.data,
        "sequence_number": seq,
        "parent_event_id": request.parent_event_id,
    }
    
    session["events"].append(event)
    
    # Update statistics
    if request.event_type == "issue_found":
        session["total_issues_found"] += 1
    elif request.event_type == "proof_generated":
        session["total_proofs_generated"] += 1
    
    return EventResponse(
        event_id=event["event_id"],
        event_type=event["event_type"],
        timestamp=event["timestamp"],
        sequence_number=event["sequence_number"],
    )


@router.get(
    "/sessions/{session_id}/events",
    summary="Get Events",
    description="Get events from a session"
)
async def get_events(
    session_id: str,
    event_type: Optional[str] = None,
    start_seq: Optional[int] = None,
    end_seq: Optional[int] = None,
) -> Dict[str, Any]:
    """Get events from a session."""
    session = _active_sessions.get(session_id) or _stored_sessions.get(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    events = session["events"]
    
    if event_type:
        events = [e for e in events if e["event_type"] == event_type]
    
    if start_seq is not None:
        events = [e for e in events if e["sequence_number"] >= start_seq]
    
    if end_seq is not None:
        events = [e for e in events if e["sequence_number"] <= end_seq]
    
    return {
        "events": events,
        "total": len(events),
    }


@router.post(
    "/sessions/{session_id}/checkpoints",
    summary="Create Checkpoint",
    description="Create a state checkpoint"
)
async def create_checkpoint(
    session_id: str,
    request: CreateCheckpointRequest,
) -> Dict[str, Any]:
    """Create a state checkpoint."""
    if session_id not in _active_sessions:
        raise HTTPException(status_code=404, detail="Active session not found")
    
    session = _active_sessions[session_id]
    
    checkpoint = {
        "state_id": f"{session_id}-cp-{len(session['checkpoints']):04d}",
        "timestamp": time.time(),
        "code": request.code,
        "language": request.language,
        "parameters": request.parameters,
        "constraints": request.constraints,
        "issues": request.issues,
        "proofs": request.proofs,
    }
    
    session["checkpoints"].append(checkpoint)
    
    # Record checkpoint event
    await record_event(
        session_id,
        RecordEventRequest(
            event_type="state_checkpoint",
            data={"state_id": checkpoint["state_id"]},
        ),
    )
    
    return {
        "created": True,
        "state_id": checkpoint["state_id"],
        "checkpoint_index": len(session["checkpoints"]) - 1,
    }


@router.get(
    "/sessions/{session_id}/checkpoints",
    summary="Get Checkpoints",
    description="Get checkpoints from a session"
)
async def get_checkpoints(session_id: str) -> Dict[str, Any]:
    """Get checkpoints from a session."""
    session = _active_sessions.get(session_id) or _stored_sessions.get(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "checkpoints": [
            {
                "state_id": c["state_id"],
                "timestamp": c["timestamp"],
                "language": c["language"],
                "issue_count": len(c.get("issues", [])),
                "proof_count": len(c.get("proofs", [])),
            }
            for c in session["checkpoints"]
        ],
        "total": len(session["checkpoints"]),
    }


@router.get(
    "/sessions/{session_id}/checkpoints/{checkpoint_index}",
    summary="Get Checkpoint",
    description="Get a specific checkpoint"
)
async def get_checkpoint(session_id: str, checkpoint_index: int) -> Dict[str, Any]:
    """Get a specific checkpoint."""
    session = _active_sessions.get(session_id) or _stored_sessions.get(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if checkpoint_index < 0 or checkpoint_index >= len(session["checkpoints"]):
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    
    return {"checkpoint": session["checkpoints"][checkpoint_index]}


@router.post(
    "/replay",
    summary="Replay Session",
    description="Replay a session with optional parameter modifications"
)
async def replay_session(request: ReplayRequest) -> Dict[str, Any]:
    """Replay a session."""
    session = _stored_sessions.get(request.session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Create replay session
    replay_session = {
        "session_id": f"replay-{request.session_id}",
        "original_session_id": request.session_id,
        "started_at": time.time(),
        "modified_parameters": request.modified_parameters,
        "events": [],
        "checkpoints": [],
    }
    
    # Filter events by range
    events = session["events"]
    if request.start_from_event is not None:
        events = [e for e in events if e["sequence_number"] >= request.start_from_event]
    if request.end_at_event is not None:
        events = [e for e in events if e["sequence_number"] <= request.end_at_event]
    
    # Replay events
    for event in events:
        replay_event = _replay_event(event, request.modified_parameters)
        replay_session["events"].append(replay_event)
    
    replay_session["ended_at"] = time.time()
    
    return {
        "replay_session": {
            "session_id": replay_session["session_id"],
            "original_session_id": replay_session["original_session_id"],
            "events_replayed": len(replay_session["events"]),
            "modified_parameters": request.modified_parameters,
            "duration_ms": (replay_session["ended_at"] - replay_session["started_at"]) * 1000,
        },
        "events": replay_session["events"],
    }


@router.post(
    "/compare",
    response_model=ComparisonResponse,
    summary="Compare States",
    description="Compare two verification states"
)
async def compare_states(request: CompareStatesRequest) -> ComparisonResponse:
    """Compare two verification states."""
    state_a = request.state_a
    state_b = request.state_b
    
    # Compare code
    code_diff = None
    if state_a.get("code", "") != state_b.get("code", ""):
        code_diff = _generate_diff(state_a.get("code", ""), state_b.get("code", ""))
    
    # Compare parameters
    param_changes: Dict[str, Dict[str, Any]] = {}
    all_params = set(state_a.get("parameters", {}).keys()) | set(state_b.get("parameters", {}).keys())
    for param in all_params:
        val_a = state_a.get("parameters", {}).get(param)
        val_b = state_b.get("parameters", {}).get(param)
        if val_a != val_b:
            param_changes[param] = {"old": val_a, "new": val_b}
    
    # Compare constraints
    constraint_changes = _compare_lists(
        state_a.get("constraints", []),
        state_b.get("constraints", []),
        key="constraint_id",
    )
    
    # Compare issues
    issues_a = {_item_key(i): i for i in state_a.get("issues", [])}
    issues_b = {_item_key(i): i for i in state_b.get("issues", [])}
    
    issues_added = [i for k, i in issues_b.items() if k not in issues_a]
    issues_removed = [i for k, i in issues_a.items() if k not in issues_b]
    
    # Compare proofs
    proofs_a = {_item_key(p): p for p in state_a.get("proofs", [])}
    proofs_b = {_item_key(p): p for p in state_b.get("proofs", [])}
    
    proofs_added = [p for k, p in proofs_b.items() if k not in proofs_a]
    proofs_removed = [p for k, p in proofs_a.items() if k not in proofs_b]
    
    return ComparisonResponse(
        code_diff=code_diff,
        parameter_changes=param_changes,
        constraint_changes=constraint_changes,
        issues_added=issues_added,
        issues_removed=issues_removed,
        proofs_added=proofs_added,
        proofs_removed=proofs_removed,
    )


@router.delete(
    "/sessions/{session_id}",
    summary="Delete Session",
    description="Delete a stored session"
)
async def delete_session(session_id: str) -> Dict[str, Any]:
    """Delete a stored session."""
    if session_id in _active_sessions:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete active session. End it first."
        )
    
    if session_id not in _stored_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del _stored_sessions[session_id]
    
    return {"deleted": True, "session_id": session_id}


@router.get(
    "/event-types",
    summary="List Event Types",
    description="Get list of valid event types"
)
async def list_event_types() -> Dict[str, Any]:
    """List valid event types."""
    return {"event_types": EVENT_TYPES}


# =============================================================================
# Helper Functions
# =============================================================================

def _replay_event(
    event: Dict[str, Any],
    modified_params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Replay an event with modified parameters."""
    import copy
    
    replay_event = copy.deepcopy(event)
    replay_event["event_id"] = f"replay-{event['event_id']}"
    replay_event["timestamp"] = time.time()
    
    # Modify verification events if params changed
    if modified_params:
        if event["event_type"] == "verification_start":
            replay_event["data"]["parameters"] = {
                **event["data"].get("parameters", {}),
                **modified_params,
            }
    
    return replay_event


def _generate_diff(text_a: str, text_b: str) -> str:
    """Generate simple diff."""
    lines_a = text_a.split('\n')
    lines_b = text_b.split('\n')
    
    diff_lines = []
    max_lines = max(len(lines_a), len(lines_b))
    
    for i in range(max_lines):
        line_a = lines_a[i] if i < len(lines_a) else ""
        line_b = lines_b[i] if i < len(lines_b) else ""
        
        if line_a != line_b:
            if line_a:
                diff_lines.append(f"-{i+1}: {line_a}")
            if line_b:
                diff_lines.append(f"+{i+1}: {line_b}")
    
    return "\n".join(diff_lines) if diff_lines else ""


def _compare_lists(
    list_a: List[Dict[str, Any]],
    list_b: List[Dict[str, Any]],
    key: str,
) -> Dict[str, Any]:
    """Compare two lists of dicts."""
    ids_a = {item.get(key): item for item in list_a}
    ids_b = {item.get(key): item for item in list_b}
    
    return {
        "added": [item for k, item in ids_b.items() if k not in ids_a],
        "removed": [item for k, item in ids_a.items() if k not in ids_b],
        "unchanged": len(ids_a.keys() & ids_b.keys()),
    }


def _item_key(item: Dict[str, Any]) -> str:
    """Generate key for item deduplication."""
    return f"{item.get('type', '')}:{item.get('message', '')}:{item.get('line', 0)}"
