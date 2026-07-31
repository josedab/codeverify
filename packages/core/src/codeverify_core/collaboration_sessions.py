"""Collaboration Sessions — Real-time multi-developer verification with live trust scores.

Enables collaborative coding sessions with shared verification state,
live trust score broadcasting, and Live Share–style session management.
"""

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class SessionRole(str, Enum):
    """Participant role in a collaboration session."""

    HOST = "host"
    PARTICIPANT = "participant"
    OBSERVER = "observer"


class SessionState(str, Enum):
    """State of a collaboration session."""

    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


class VerificationMode(str, Enum):
    """Verification behavior during collaboration."""

    CONTINUOUS = "continuous"
    ON_PAUSE = "on_pause"
    MANUAL = "manual"
    BRAINSTORMING = "brainstorming"  # No verification, free-form coding


@dataclass
class Participant:
    """A developer in a collaboration session."""

    id: str
    name: str
    role: SessionRole
    active_file: str | None = None
    cursor_position: tuple[int, int] | None = None  # (line, col)
    joined_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_active: datetime = field(default_factory=lambda: datetime.now(UTC))
    edits_count: int = 0
    findings_resolved: int = 0

    @property
    def idle_seconds(self) -> float:
        return (datetime.now(UTC) - self.last_active).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "active_file": self.active_file,
            "cursor_position": self.cursor_position,
            "idle_seconds": self.idle_seconds,
            "edits_count": self.edits_count,
            "findings_resolved": self.findings_resolved,
        }


@dataclass
class LiveTrustScore:
    """Real-time trust score for a session or file."""

    score: float  # 0-100
    trend: str  # "improving", "stable", "declining"
    factors: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 1),
            "trend": self.trend,
            "factors": self.factors,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FileState:
    """Verification state for a single file in a session."""

    file_path: str
    trust_score: float = 100.0
    findings_count: int = 0
    last_verified: datetime | None = None
    editors: list[str] = field(default_factory=list)
    pending_changes: bool = False
    score_history: list[tuple[float, float]] = field(default_factory=list)  # (timestamp, score)

    def record_score(self, score: float) -> None:
        self.trust_score = score
        self.score_history.append((time.time(), score))
        # Keep last 100 entries
        if len(self.score_history) > 100:
            self.score_history = self.score_history[-100:]

    @property
    def trend(self) -> str:
        if len(self.score_history) < 2:
            return "stable"
        recent = self.score_history[-5:]
        if len(recent) < 2:
            return "stable"
        avg_recent = sum(s for _, s in recent) / len(recent)
        avg_older = sum(s for _, s in self.score_history[-10:-5]) / max(
            1, len(self.score_history[-10:-5])
        )
        if avg_older == 0:
            return "stable"
        diff = avg_recent - avg_older
        if diff > 2:
            return "improving"
        elif diff < -2:
            return "declining"
        return "stable"


@dataclass
class ConflictAlert:
    """Alert when multiple participants edit overlapping code."""

    file_path: str
    participant_ids: list[str]
    line_ranges: list[tuple[int, int]]
    severity: str  # "warning", "conflict"
    message: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "participants": self.participant_ids,
            "line_ranges": self.line_ranges,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass
class SessionEvent:
    """An event broadcast to all session participants."""

    type: str  # "trust_update", "finding", "conflict", "participant_joined", etc.
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class CollaborationSession:
    """Manages a real-time collaborative verification session.

    Tracks participants, file states, trust scores, and edit conflicts.

    Example:
        >>> session = CollaborationSession(session_id="sess-1", host_name="alice")
        >>> session.add_participant("bob-id", "bob", SessionRole.PARTICIPANT)
        >>> session.record_edit("bob-id", "main.py", 10, 25)
        >>> trust = session.get_live_trust_score()
    """

    def __init__(
        self,
        session_id: str | None = None,
        host_name: str = "host",
        verification_mode: VerificationMode = VerificationMode.ON_PAUSE,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.state = SessionState.ACTIVE
        self.verification_mode = verification_mode
        self.created_at = datetime.now(UTC)
        self._participants: dict[str, Participant] = {}
        self._files: dict[str, FileState] = {}
        self._events: list[SessionEvent] = []
        self._event_callbacks: list[Callable[[SessionEvent], None]] = []
        self._edit_ranges: dict[
            str, dict[str, tuple[int, int]]
        ] = {}  # file -> {participant_id -> (start, end)}

        # Add host
        host_id = str(uuid.uuid4())
        self._participants[host_id] = Participant(id=host_id, name=host_name, role=SessionRole.HOST)

    def add_participant(
        self,
        participant_id: str,
        name: str,
        role: SessionRole = SessionRole.PARTICIPANT,
    ) -> Participant:
        """Add a participant to the session."""
        participant = Participant(id=participant_id, name=name, role=role)
        self._participants[participant_id] = participant
        self._emit(
            SessionEvent(
                type="participant_joined",
                data={"participant": participant.to_dict()},
            )
        )
        return participant

    def remove_participant(self, participant_id: str) -> None:
        """Remove a participant from the session."""
        participant = self._participants.pop(participant_id, None)
        if participant:
            self._emit(
                SessionEvent(
                    type="participant_left",
                    data={"participant_id": participant_id, "name": participant.name},
                )
            )

    def record_edit(
        self,
        participant_id: str,
        file_path: str,
        line_start: int,
        line_end: int,
    ) -> ConflictAlert | None:
        """Record an edit from a participant and check for conflicts."""
        participant = self._participants.get(participant_id)
        if not participant:
            return None

        participant.last_active = datetime.now(UTC)
        participant.active_file = file_path
        participant.edits_count += 1

        # Initialize file state
        if file_path not in self._files:
            self._files[file_path] = FileState(file_path=file_path)
        file_state = self._files[file_path]
        file_state.pending_changes = True
        if participant_id not in file_state.editors:
            file_state.editors.append(participant_id)

        # Track edit ranges for conflict detection
        if file_path not in self._edit_ranges:
            self._edit_ranges[file_path] = {}
        self._edit_ranges[file_path][participant_id] = (line_start, line_end)

        # Check for overlapping edits
        return self._check_conflicts(file_path, participant_id, line_start, line_end)

    def record_verification(
        self,
        file_path: str,
        trust_score: float,
        findings_count: int,
    ) -> LiveTrustScore:
        """Record verification results for a file and update trust score."""
        if file_path not in self._files:
            self._files[file_path] = FileState(file_path=file_path)

        file_state = self._files[file_path]
        file_state.record_score(trust_score)
        file_state.findings_count = findings_count
        file_state.last_verified = datetime.now(UTC)
        file_state.pending_changes = False

        live_score = LiveTrustScore(
            score=trust_score,
            trend=file_state.trend,
            factors={
                "findings": max(0, 100 - findings_count * 10),
                "verification_coverage": trust_score,
            },
        )

        self._emit(
            SessionEvent(
                type="trust_update",
                data={
                    "file_path": file_path,
                    "trust_score": live_score.to_dict(),
                },
            )
        )

        return live_score

    def get_live_trust_score(self) -> LiveTrustScore:
        """Calculate aggregate trust score across all session files."""
        if not self._files:
            return LiveTrustScore(score=100.0, trend="stable")

        scores = [f.trust_score for f in self._files.values()]
        avg_score = sum(scores) / len(scores)

        trends = [f.trend for f in self._files.values()]
        if trends.count("declining") > len(trends) / 2:
            trend = "declining"
        elif trends.count("improving") > len(trends) / 2:
            trend = "improving"
        else:
            trend = "stable"

        return LiveTrustScore(
            score=avg_score,
            trend=trend,
            factors={
                "files_verified": sum(
                    1 for f in self._files.values() if f.last_verified is not None
                ),
                "pending_changes": sum(1 for f in self._files.values() if f.pending_changes),
                "active_participants": sum(
                    1 for p in self._participants.values() if p.idle_seconds < 300
                ),
            },
        )

    def get_session_state(self) -> dict[str, Any]:
        """Get full session state for broadcasting."""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "verification_mode": self.verification_mode.value,
            "trust_score": self.get_live_trust_score().to_dict(),
            "participants": [p.to_dict() for p in self._participants.values()],
            "files": {
                path: {
                    "trust_score": f.trust_score,
                    "findings_count": f.findings_count,
                    "trend": f.trend,
                    "editors": f.editors,
                    "pending_changes": f.pending_changes,
                }
                for path, f in self._files.items()
            },
            "created_at": self.created_at.isoformat(),
        }

    def set_mode(self, mode: VerificationMode) -> None:
        """Change the verification mode for the session."""
        old_mode = self.verification_mode
        self.verification_mode = mode
        self._emit(
            SessionEvent(
                type="mode_changed",
                data={"old_mode": old_mode.value, "new_mode": mode.value},
            )
        )

    def pause(self) -> None:
        self.state = SessionState.PAUSED
        self._emit(SessionEvent(type="session_paused"))

    def resume(self) -> None:
        self.state = SessionState.ACTIVE
        self._emit(SessionEvent(type="session_resumed"))

    def end(self) -> dict[str, Any]:
        """End the session and return summary."""
        self.state = SessionState.ENDED
        self._emit(SessionEvent(type="session_ended"))
        return {
            "session_id": self.session_id,
            "duration_seconds": (datetime.now(UTC) - self.created_at).total_seconds(),
            "participants": len(self._participants),
            "files_edited": len(self._files),
            "total_edits": sum(p.edits_count for p in self._participants.values()),
            "final_trust_score": self.get_live_trust_score().to_dict(),
        }

    def on_event(self, callback: Callable[[SessionEvent], None]) -> None:
        """Register a callback for session events."""
        self._event_callbacks.append(callback)

    def _emit(self, event: SessionEvent) -> None:
        self._events.append(event)
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error("Event callback error", error=str(e))

    def _check_conflicts(
        self,
        file_path: str,
        editor_id: str,
        line_start: int,
        line_end: int,
    ) -> ConflictAlert | None:
        """Check if this edit overlaps with another participant's edits."""
        ranges = self._edit_ranges.get(file_path, {})
        conflicting = []
        for pid, (start, end) in ranges.items():
            if pid == editor_id:
                continue
            # Check overlap
            if line_start <= end and line_end >= start:
                conflicting.append(pid)

        if conflicting:
            alert = ConflictAlert(
                file_path=file_path,
                participant_ids=[editor_id] + conflicting,
                line_ranges=[(line_start, line_end)],
                severity="conflict" if len(conflicting) > 1 else "warning",
                message=f"Overlapping edits in {file_path} by {len(conflicting) + 1} participants",
            )
            self._emit(
                SessionEvent(
                    type="conflict_detected",
                    data=alert.to_dict(),
                )
            )
            return alert
        return None


class SessionManager:
    """Manages multiple collaboration sessions.

    Example:
        >>> manager = SessionManager()
        >>> session = manager.create_session(host_name="alice")
        >>> manager.join_session(session.session_id, "bob-id", "bob")
    """

    def __init__(self, max_sessions: int = 100) -> None:
        self._sessions: dict[str, CollaborationSession] = {}
        self._max_sessions = max_sessions

    def create_session(
        self,
        host_name: str = "host",
        verification_mode: VerificationMode = VerificationMode.ON_PAUSE,
        session_id: str | None = None,
    ) -> CollaborationSession:
        """Create a new collaboration session."""
        if len(self._sessions) >= self._max_sessions:
            self._cleanup_ended()

        session = CollaborationSession(
            session_id=session_id,
            host_name=host_name,
            verification_mode=verification_mode,
        )
        self._sessions[session.session_id] = session
        logger.info("Collaboration session created", session_id=session.session_id)
        return session

    def get_session(self, session_id: str) -> CollaborationSession | None:
        return self._sessions.get(session_id)

    def join_session(
        self,
        session_id: str,
        participant_id: str,
        name: str,
        role: SessionRole = SessionRole.PARTICIPANT,
    ) -> Participant | None:
        """Join an existing session."""
        session = self._sessions.get(session_id)
        if not session or session.state == SessionState.ENDED:
            return None
        return session.add_participant(participant_id, name, role)

    def end_session(self, session_id: str) -> dict[str, Any] | None:
        """End a session and return summary."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        return session.end()

    def list_active_sessions(self) -> list[dict[str, Any]]:
        """List all active sessions."""
        return [
            {
                "session_id": s.session_id,
                "state": s.state.value,
                "participants": len(s._participants),
                "trust_score": s.get_live_trust_score().score,
            }
            for s in self._sessions.values()
            if s.state != SessionState.ENDED
        ]

    def _cleanup_ended(self) -> None:
        ended = [sid for sid, s in self._sessions.items() if s.state == SessionState.ENDED]
        for sid in ended:
            del self._sessions[sid]
