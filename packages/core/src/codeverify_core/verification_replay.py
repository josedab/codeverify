"""
Verification Replay & Time-Travel

Record verification sessions and replay with different parameters:
- Session recording
- Replay engine
- State comparison
- Audit compliance and debugging aid

Allows users to understand how verification results were reached and
experiment with different parameters.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


# =============================================================================
# Data Models
# =============================================================================

class EventType(str, Enum):
    """Types of verification events."""
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    CODE_SUBMITTED = "code_submitted"
    CONSTRAINT_ADDED = "constraint_added"
    CONSTRAINT_REMOVED = "constraint_removed"
    VERIFICATION_START = "verification_start"
    VERIFICATION_COMPLETE = "verification_complete"
    ISSUE_FOUND = "issue_found"
    PROOF_GENERATED = "proof_generated"
    PARAMETER_CHANGED = "parameter_changed"
    STATE_CHECKPOINT = "state_checkpoint"


@dataclass
class VerificationEvent:
    """A single event in a verification session."""
    
    event_id: str
    event_type: EventType
    timestamp: float
    
    # Event data
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Context
    sequence_number: int = 0
    parent_event_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "sequence_number": self.sequence_number,
            "parent_event_id": self.parent_event_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VerificationEvent:
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            timestamp=data["timestamp"],
            data=data.get("data", {}),
            sequence_number=data.get("sequence_number", 0),
            parent_event_id=data.get("parent_event_id"),
        )


@dataclass
class VerificationState:
    """Snapshot of verification state at a point in time."""
    
    state_id: str
    timestamp: float
    
    # Code state
    code: str = ""
    language: str = "python"
    file_path: Optional[str] = None
    
    # Verification parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Active constraints
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    
    # Current results
    issues: List[Dict[str, Any]] = field(default_factory=list)
    proofs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    verification_mode: str = "standard"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state_id": self.state_id,
            "timestamp": self.timestamp,
            "code": self.code,
            "language": self.language,
            "file_path": self.file_path,
            "parameters": self.parameters,
            "constraints": self.constraints,
            "issues": self.issues,
            "proofs": self.proofs,
            "verification_mode": self.verification_mode,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VerificationState:
        """Create from dictionary."""
        return cls(
            state_id=data["state_id"],
            timestamp=data["timestamp"],
            code=data.get("code", ""),
            language=data.get("language", "python"),
            file_path=data.get("file_path"),
            parameters=data.get("parameters", {}),
            constraints=data.get("constraints", []),
            issues=data.get("issues", []),
            proofs=data.get("proofs", []),
            verification_mode=data.get("verification_mode", "standard"),
        )


@dataclass
class VerificationSession:
    """A recorded verification session."""
    
    session_id: str
    started_at: float
    ended_at: Optional[float] = None
    
    # Session metadata
    user_id: Optional[str] = None
    repository: Optional[str] = None
    file_path: Optional[str] = None
    
    # Events
    events: List[VerificationEvent] = field(default_factory=list)
    
    # State checkpoints
    checkpoints: List[VerificationState] = field(default_factory=list)
    
    # Summary
    total_issues_found: int = 0
    total_proofs_generated: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "user_id": self.user_id,
            "repository": self.repository,
            "file_path": self.file_path,
            "events": [e.to_dict() for e in self.events],
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "total_issues_found": self.total_issues_found,
            "total_proofs_generated": self.total_proofs_generated,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VerificationSession:
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            started_at=data["started_at"],
            ended_at=data.get("ended_at"),
            user_id=data.get("user_id"),
            repository=data.get("repository"),
            file_path=data.get("file_path"),
            events=[VerificationEvent.from_dict(e) for e in data.get("events", [])],
            checkpoints=[VerificationState.from_dict(c) for c in data.get("checkpoints", [])],
            total_issues_found=data.get("total_issues_found", 0),
            total_proofs_generated=data.get("total_proofs_generated", 0),
        )


@dataclass
class StateComparison:
    """Comparison between two verification states."""
    
    state_a_id: str
    state_b_id: str
    
    # Differences
    code_diff: Optional[str] = None
    parameter_changes: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    constraint_changes: Dict[str, Any] = field(default_factory=dict)
    
    # Result differences
    issues_added: List[Dict[str, Any]] = field(default_factory=list)
    issues_removed: List[Dict[str, Any]] = field(default_factory=list)
    proofs_added: List[Dict[str, Any]] = field(default_factory=list)
    proofs_removed: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state_a_id": self.state_a_id,
            "state_b_id": self.state_b_id,
            "code_diff": self.code_diff,
            "parameter_changes": {
                k: {"old": v[0], "new": v[1]}
                for k, v in self.parameter_changes.items()
            },
            "constraint_changes": self.constraint_changes,
            "issues_added": self.issues_added,
            "issues_removed": self.issues_removed,
            "proofs_added": self.proofs_added,
            "proofs_removed": self.proofs_removed,
        }


# =============================================================================
# Session Recorder
# =============================================================================

class SessionRecorder:
    """Records verification sessions."""
    
    def __init__(self):
        self._active_sessions: Dict[str, VerificationSession] = {}
        self._event_counter: Dict[str, int] = {}
    
    def start_session(
        self,
        user_id: Optional[str] = None,
        repository: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> VerificationSession:
        """Start a new recording session."""
        session_id = hashlib.sha256(
            f"{time.time()}-{user_id or 'anon'}".encode()
        ).hexdigest()[:16]
        
        session = VerificationSession(
            session_id=session_id,
            started_at=time.time(),
            user_id=user_id,
            repository=repository,
            file_path=file_path,
        )
        
        self._active_sessions[session_id] = session
        self._event_counter[session_id] = 0
        
        # Record session start event
        self.record_event(
            session_id,
            EventType.SESSION_START,
            {
                "user_id": user_id,
                "repository": repository,
                "file_path": file_path,
            },
        )
        
        return session
    
    def end_session(self, session_id: str) -> Optional[VerificationSession]:
        """End a recording session."""
        session = self._active_sessions.get(session_id)
        if not session:
            return None
        
        session.ended_at = time.time()
        
        # Record session end event
        self.record_event(
            session_id,
            EventType.SESSION_END,
            {
                "duration_seconds": session.ended_at - session.started_at,
                "total_events": len(session.events),
            },
        )
        
        # Remove from active
        del self._active_sessions[session_id]
        del self._event_counter[session_id]
        
        return session
    
    def record_event(
        self,
        session_id: str,
        event_type: EventType,
        data: Dict[str, Any],
        parent_event_id: Optional[str] = None,
    ) -> Optional[VerificationEvent]:
        """Record an event in a session."""
        session = self._active_sessions.get(session_id)
        if not session:
            return None
        
        self._event_counter[session_id] += 1
        seq = self._event_counter[session_id]
        
        event = VerificationEvent(
            event_id=f"{session_id}-{seq:06d}",
            event_type=event_type,
            timestamp=time.time(),
            data=data,
            sequence_number=seq,
            parent_event_id=parent_event_id,
        )
        
        session.events.append(event)
        
        # Update session statistics
        if event_type == EventType.ISSUE_FOUND:
            session.total_issues_found += 1
        elif event_type == EventType.PROOF_GENERATED:
            session.total_proofs_generated += 1
        
        return event
    
    def create_checkpoint(
        self,
        session_id: str,
        state: VerificationState,
    ) -> Optional[VerificationState]:
        """Create a state checkpoint."""
        session = self._active_sessions.get(session_id)
        if not session:
            return None
        
        state.state_id = f"{session_id}-cp-{len(session.checkpoints):04d}"
        state.timestamp = time.time()
        
        session.checkpoints.append(state)
        
        # Record checkpoint event
        self.record_event(
            session_id,
            EventType.STATE_CHECKPOINT,
            {"state_id": state.state_id},
        )
        
        return state
    
    def get_session(self, session_id: str) -> Optional[VerificationSession]:
        """Get a session (active or not)."""
        return self._active_sessions.get(session_id)


# =============================================================================
# Replay Engine
# =============================================================================

class ReplayEngine:
    """Replays recorded verification sessions."""
    
    def __init__(self, verification_func: Optional[Callable] = None):
        self._verification_func = verification_func or self._default_verify
    
    def replay_session(
        self,
        session: VerificationSession,
        modified_parameters: Optional[Dict[str, Any]] = None,
        start_from_event: Optional[int] = None,
        end_at_event: Optional[int] = None,
    ) -> VerificationSession:
        """
        Replay a recorded session.
        
        Optionally modify parameters to see different results.
        """
        # Create new session for replay results
        replay_session = VerificationSession(
            session_id=f"replay-{session.session_id}",
            started_at=time.time(),
            user_id=session.user_id,
            repository=session.repository,
            file_path=session.file_path,
        )
        
        # Filter events by range
        events = session.events
        if start_from_event is not None:
            events = [e for e in events if e.sequence_number >= start_from_event]
        if end_at_event is not None:
            events = [e for e in events if e.sequence_number <= end_at_event]
        
        # Current state during replay
        current_state = VerificationState(
            state_id="replay-state",
            timestamp=time.time(),
            parameters=modified_parameters or {},
        )
        
        # Replay events
        for event in events:
            replay_event = self._replay_event(event, current_state, modified_parameters)
            if replay_event:
                replay_session.events.append(replay_event)
                
                # Update state based on event
                self._update_state(current_state, replay_event)
        
        replay_session.ended_at = time.time()
        replay_session.checkpoints.append(current_state)
        
        return replay_session
    
    def replay_to_checkpoint(
        self,
        session: VerificationSession,
        checkpoint_index: int,
    ) -> Optional[VerificationState]:
        """Replay session to a specific checkpoint."""
        if checkpoint_index < 0 or checkpoint_index >= len(session.checkpoints):
            return None
        
        return copy.deepcopy(session.checkpoints[checkpoint_index])
    
    def _replay_event(
        self,
        event: VerificationEvent,
        state: VerificationState,
        modified_params: Optional[Dict[str, Any]],
    ) -> Optional[VerificationEvent]:
        """Replay a single event with potentially modified parameters."""
        # Create copy of event
        replay_event = VerificationEvent(
            event_id=f"replay-{event.event_id}",
            event_type=event.event_type,
            timestamp=time.time(),
            data=copy.deepcopy(event.data),
            sequence_number=event.sequence_number,
            parent_event_id=event.parent_event_id,
        )
        
        # Handle verification events specially
        if event.event_type == EventType.VERIFICATION_START:
            # Apply modified parameters
            if modified_params:
                replay_event.data["parameters"] = {
                    **event.data.get("parameters", {}),
                    **modified_params,
                }
        
        elif event.event_type == EventType.VERIFICATION_COMPLETE:
            # Re-run verification with modified parameters
            if modified_params and state.code:
                result = self._verification_func(
                    state.code,
                    state.language,
                    modified_params,
                )
                replay_event.data["result"] = result
        
        return replay_event
    
    def _update_state(
        self,
        state: VerificationState,
        event: VerificationEvent,
    ) -> None:
        """Update state based on event."""
        if event.event_type == EventType.CODE_SUBMITTED:
            state.code = event.data.get("code", state.code)
            state.language = event.data.get("language", state.language)
        
        elif event.event_type == EventType.CONSTRAINT_ADDED:
            state.constraints.append(event.data.get("constraint", {}))
        
        elif event.event_type == EventType.CONSTRAINT_REMOVED:
            cid = event.data.get("constraint_id")
            state.constraints = [
                c for c in state.constraints if c.get("constraint_id") != cid
            ]
        
        elif event.event_type == EventType.ISSUE_FOUND:
            state.issues.append(event.data.get("issue", {}))
        
        elif event.event_type == EventType.PROOF_GENERATED:
            state.proofs.append(event.data.get("proof", {}))
        
        elif event.event_type == EventType.PARAMETER_CHANGED:
            param = event.data.get("parameter")
            value = event.data.get("value")
            if param:
                state.parameters[param] = value
    
    def _default_verify(
        self,
        code: str,
        language: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Default verification function for replay."""
        # Simplified verification
        issues = []
        
        if "return None" in code and parameters.get("check_null_safety", True):
            issues.append({
                "type": "null_safety",
                "message": "Potential null return",
            })
        
        if "eval(" in code and parameters.get("check_security", True):
            issues.append({
                "type": "security",
                "message": "Unsafe eval usage",
            })
        
        return {
            "verified": len(issues) == 0,
            "issues": issues,
        }


# =============================================================================
# State Comparator
# =============================================================================

class StateComparator:
    """Compares verification states."""
    
    def compare(
        self,
        state_a: VerificationState,
        state_b: VerificationState,
    ) -> StateComparison:
        """Compare two verification states."""
        comparison = StateComparison(
            state_a_id=state_a.state_id,
            state_b_id=state_b.state_id,
        )
        
        # Compare code
        if state_a.code != state_b.code:
            comparison.code_diff = self._generate_diff(state_a.code, state_b.code)
        
        # Compare parameters
        all_params = set(state_a.parameters.keys()) | set(state_b.parameters.keys())
        for param in all_params:
            val_a = state_a.parameters.get(param)
            val_b = state_b.parameters.get(param)
            if val_a != val_b:
                comparison.parameter_changes[param] = (val_a, val_b)
        
        # Compare constraints
        comparison.constraint_changes = self._compare_constraints(
            state_a.constraints,
            state_b.constraints,
        )
        
        # Compare issues
        issues_a = {self._issue_key(i): i for i in state_a.issues}
        issues_b = {self._issue_key(i): i for i in state_b.issues}
        
        for key, issue in issues_b.items():
            if key not in issues_a:
                comparison.issues_added.append(issue)
        
        for key, issue in issues_a.items():
            if key not in issues_b:
                comparison.issues_removed.append(issue)
        
        # Compare proofs
        proofs_a = {self._proof_key(p): p for p in state_a.proofs}
        proofs_b = {self._proof_key(p): p for p in state_b.proofs}
        
        for key, proof in proofs_b.items():
            if key not in proofs_a:
                comparison.proofs_added.append(proof)
        
        for key, proof in proofs_a.items():
            if key not in proofs_b:
                comparison.proofs_removed.append(proof)
        
        return comparison
    
    def _generate_diff(self, text_a: str, text_b: str) -> str:
        """Generate simple diff between two texts."""
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
        
        return "\n".join(diff_lines)
    
    def _compare_constraints(
        self,
        constraints_a: List[Dict[str, Any]],
        constraints_b: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare constraint lists."""
        ids_a = {c.get("constraint_id"): c for c in constraints_a}
        ids_b = {c.get("constraint_id"): c for c in constraints_b}
        
        added = [c for cid, c in ids_b.items() if cid not in ids_a]
        removed = [c for cid, c in ids_a.items() if cid not in ids_b]
        
        return {
            "added": added,
            "removed": removed,
            "unchanged": len(ids_a.keys() & ids_b.keys()),
        }
    
    def _issue_key(self, issue: Dict[str, Any]) -> str:
        """Generate key for issue deduplication."""
        return f"{issue.get('type', '')}:{issue.get('message', '')}:{issue.get('line', 0)}"
    
    def _proof_key(self, proof: Dict[str, Any]) -> str:
        """Generate key for proof deduplication."""
        return f"{proof.get('property', '')}:{proof.get('result', '')}"


# =============================================================================
# Session Storage
# =============================================================================

class SessionStorage:
    """Stores and retrieves verification sessions."""
    
    def __init__(self):
        self._sessions: Dict[str, VerificationSession] = {}
    
    def save(self, session: VerificationSession) -> None:
        """Save a session."""
        self._sessions[session.session_id] = session
    
    def load(self, session_id: str) -> Optional[VerificationSession]:
        """Load a session by ID."""
        return self._sessions.get(session_id)
    
    def list_sessions(
        self,
        user_id: Optional[str] = None,
        repository: Optional[str] = None,
        limit: int = 100,
    ) -> List[VerificationSession]:
        """List sessions with optional filters."""
        sessions = list(self._sessions.values())
        
        if user_id:
            sessions = [s for s in sessions if s.user_id == user_id]
        
        if repository:
            sessions = [s for s in sessions if s.repository == repository]
        
        # Sort by start time descending
        sessions.sort(key=lambda s: s.started_at, reverse=True)
        
        return sessions[:limit]
    
    def delete(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export session as JSON-serializable dict."""
        session = self._sessions.get(session_id)
        if session:
            return session.to_dict()
        return None
    
    def import_session(self, data: Dict[str, Any]) -> VerificationSession:
        """Import session from dict."""
        session = VerificationSession.from_dict(data)
        self._sessions[session.session_id] = session
        return session
