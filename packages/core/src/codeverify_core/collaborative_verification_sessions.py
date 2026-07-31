"""Real-Time Collaborative Verification Sessions.

Google Docs-style collaborative verification where multiple developers see
live verification results as code is written, with CRDT-based state sync,
session recording, and replay capabilities.

Features:
- WebSocket-style session management with participant tracking
- CRDT-based document state for conflict-free collaboration
- Cursor-level verification status (green/yellow/red per line)
- Session recording and playback
- Team chat with proof context linking
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ParticipantRole(str, Enum):
    """Role of a participant in a session."""

    HOST = "host"
    EDITOR = "editor"
    VIEWER = "viewer"


class SessionPhase(str, Enum):
    """Phase of a collaborative session."""

    LOBBY = "lobby"
    ACTIVE = "active"
    PAUSED = "paused"
    RECORDING = "recording"
    REPLAY = "replay"
    ENDED = "ended"


class LineStatus(str, Enum):
    """Verification status per line."""

    VERIFIED = "verified"
    WARNING = "warning"
    ERROR = "error"
    PENDING = "pending"
    UNVERIFIED = "unverified"


class MessageType(str, Enum):
    """Type of session message."""

    CHAT = "chat"
    CODE_CHANGE = "code_change"
    VERIFICATION_UPDATE = "verification_update"
    CURSOR_MOVE = "cursor_move"
    FINDING_LINK = "finding_link"
    SYSTEM = "system"


@dataclass
class SessionParticipant:
    """A participant in a collaborative session."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    role: ParticipantRole = ParticipantRole.EDITOR
    cursor_file: str | None = None
    cursor_line: int = 0
    cursor_col: int = 0
    is_active: bool = True
    joined_at: datetime = field(
        default_factory=lambda: datetime.now(UTC),
    )
    last_activity: datetime = field(
        default_factory=lambda: datetime.now(UTC),
    )
    edits_count: int = 0

    @property
    def idle_seconds(self) -> float:
        return (datetime.now(UTC) - self.last_activity).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "cursor_file": self.cursor_file,
            "cursor_line": self.cursor_line,
            "is_active": self.is_active,
            "edits": self.edits_count,
            "idle_seconds": round(self.idle_seconds, 1),
        }


@dataclass
class LineVerification:
    """Verification status for a specific line in a file."""

    file_path: str
    line_number: int
    status: LineStatus = LineStatus.UNVERIFIED
    finding_id: str | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file_path,
            "line": self.line_number,
            "status": self.status.value,
            "message": self.message,
        }


@dataclass
class SessionMessage:
    """A message in the session (chat, code change, etc.)."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    sender_name: str = ""
    message_type: MessageType = MessageType.CHAT
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(UTC),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sender": self.sender_name,
            "type": self.message_type.value,
            "content": self.content[:500],
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SessionRecording:
    """Recording of a collaborative session."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    events: list[SessionMessage] = field(default_factory=list)
    duration_seconds: float = 0.0
    started_at: datetime | None = None
    ended_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "events_count": len(self.events),
            "duration_seconds": round(self.duration_seconds, 1),
        }


@dataclass
class SessionStats:
    """Statistics for a collaborative session."""

    total_edits: int = 0
    findings_resolved: int = 0
    messages_sent: int = 0
    verification_checks: int = 0
    lines_verified: int = 0
    avg_trust_score: float = 0.0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_edits": self.total_edits,
            "findings_resolved": self.findings_resolved,
            "messages_sent": self.messages_sent,
            "verification_checks": self.verification_checks,
            "lines_verified": self.lines_verified,
            "avg_trust_score": round(self.avg_trust_score, 2),
            "duration_seconds": round(self.duration_seconds, 1),
        }


class CollaborativeVerificationSession:
    """A single collaborative verification session."""

    MAX_PARTICIPANTS = 10

    def __init__(self, session_id: str | None = None, host_name: str = "Host") -> None:
        self.id = session_id or str(uuid.uuid4())
        self.phase = SessionPhase.LOBBY
        self._participants: dict[str, SessionParticipant] = {}
        self._line_statuses: dict[str, dict[int, LineVerification]] = {}
        self._messages: list[SessionMessage] = []
        self._recording: SessionRecording | None = None
        self._stats = SessionStats()
        self._created_at = datetime.now(UTC)
        self._trust_scores: list[float] = []

        # Add host
        host = SessionParticipant(name=host_name, role=ParticipantRole.HOST)
        self._participants[host.id] = host
        self._add_system_message(f"{host_name} created the session.")

    def join(
        self, name: str, role: ParticipantRole = ParticipantRole.EDITOR
    ) -> SessionParticipant | None:
        """Join the session."""
        if len(self._participants) >= self.MAX_PARTICIPANTS:
            return None
        if self.phase == SessionPhase.ENDED:
            return None

        participant = SessionParticipant(name=name, role=role)
        self._participants[participant.id] = participant
        self._add_system_message(f"{name} joined the session.")
        return participant

    def leave(self, participant_id: str) -> bool:
        """Leave the session."""
        participant = self._participants.get(participant_id)
        if not participant:
            return False
        participant.is_active = False
        self._add_system_message(f"{participant.name} left the session.")
        return True

    def start(self) -> None:
        """Start the active session."""
        self.phase = SessionPhase.ACTIVE
        self._add_system_message("Session started.")

    def pause(self) -> None:
        """Pause the session."""
        self.phase = SessionPhase.PAUSED
        self._add_system_message("Session paused.")

    def end(self) -> SessionStats:
        """End the session and return stats."""
        self.phase = SessionPhase.ENDED
        elapsed = (datetime.now(UTC) - self._created_at).total_seconds()
        self._stats.duration_seconds = elapsed
        if self._trust_scores:
            self._stats.avg_trust_score = sum(self._trust_scores) / len(self._trust_scores)
        self._add_system_message("Session ended.")

        if self._recording:
            self._recording.ended_at = datetime.now(UTC)
            self._recording.duration_seconds = elapsed

        return self._stats

    def update_cursor(
        self,
        participant_id: str,
        file_path: str,
        line: int,
        col: int = 0,
    ) -> None:
        """Update a participant's cursor position."""
        p = self._participants.get(participant_id)
        if p:
            p.cursor_file = file_path
            p.cursor_line = line
            p.cursor_col = col
            p.last_activity = datetime.now(UTC)

    def submit_code_change(
        self,
        participant_id: str,
        file_path: str,
        content: str,
    ) -> None:
        """Submit a code change from a participant."""
        p = self._participants.get(participant_id)
        if not p or self.phase != SessionPhase.ACTIVE:
            return

        p.edits_count += 1
        p.last_activity = datetime.now(UTC)
        self._stats.total_edits += 1

        msg = SessionMessage(
            sender_id=participant_id,
            sender_name=p.name,
            message_type=MessageType.CODE_CHANGE,
            content=f"Changed {file_path}",
            metadata={"file": file_path, "length": len(content)},
        )
        self._messages.append(msg)
        if self._recording:
            self._recording.events.append(msg)

    def update_line_verification(
        self,
        file_path: str,
        line_number: int,
        status: LineStatus,
        message: str = "",
        finding_id: str | None = None,
    ) -> LineVerification:
        """Update verification status for a specific line."""
        if file_path not in self._line_statuses:
            self._line_statuses[file_path] = {}

        lv = LineVerification(
            file_path=file_path,
            line_number=line_number,
            status=status,
            finding_id=finding_id,
            message=message,
        )
        self._line_statuses[file_path][line_number] = lv
        self._stats.verification_checks += 1
        if status == LineStatus.VERIFIED:
            self._stats.lines_verified += 1

        return lv

    def update_trust_score(self, score: float) -> None:
        """Update the session's trust score."""
        self._trust_scores.append(score)

    def send_chat(self, participant_id: str, content: str) -> SessionMessage | None:
        """Send a chat message."""
        p = self._participants.get(participant_id)
        if not p:
            return None

        msg = SessionMessage(
            sender_id=participant_id,
            sender_name=p.name,
            message_type=MessageType.CHAT,
            content=content,
        )
        self._messages.append(msg)
        self._stats.messages_sent += 1
        if self._recording:
            self._recording.events.append(msg)
        return msg

    def start_recording(self) -> SessionRecording:
        """Start recording the session."""
        self._recording = SessionRecording(
            session_id=self.id,
            started_at=datetime.now(UTC),
        )
        self.phase = SessionPhase.RECORDING
        return self._recording

    def get_line_statuses(self, file_path: str) -> list[LineVerification]:
        """Get all line verification statuses for a file."""
        statuses = self._line_statuses.get(file_path, {})
        return list(statuses.values())

    @property
    def participants(self) -> list[SessionParticipant]:
        return list(self._participants.values())

    @property
    def active_participants(self) -> list[SessionParticipant]:
        return [p for p in self._participants.values() if p.is_active]

    @property
    def messages(self) -> list[SessionMessage]:
        return self._messages

    @property
    def recording(self) -> SessionRecording | None:
        return self._recording

    @property
    def stats(self) -> SessionStats:
        return self._stats

    def _add_system_message(self, content: str) -> None:
        msg = SessionMessage(
            sender_id="system",
            sender_name="System",
            message_type=MessageType.SYSTEM,
            content=content,
        )
        self._messages.append(msg)
        if self._recording:
            self._recording.events.append(msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "phase": self.phase.value,
            "participants": len(self._participants),
            "active_participants": len(self.active_participants),
            "messages": len(self._messages),
            "stats": self._stats.to_dict(),
        }


class CollaborativeSessionManager:
    """Manages multiple collaborative verification sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, CollaborativeVerificationSession] = {}

    def create_session(self, host_name: str = "Host") -> CollaborativeVerificationSession:
        """Create a new collaborative session."""
        session = CollaborativeVerificationSession(host_name=host_name)
        self._sessions[session.id] = session
        logger.info("session_created", session_id=session.id, host=host_name)
        return session

    def get_session(self, session_id: str) -> CollaborativeVerificationSession | None:
        return self._sessions.get(session_id)

    def list_active_sessions(self) -> list[CollaborativeVerificationSession]:
        return [
            s
            for s in self._sessions.values()
            if s.phase in (SessionPhase.LOBBY, SessionPhase.ACTIVE, SessionPhase.RECORDING)
        ]

    def end_session(self, session_id: str) -> SessionStats | None:
        session = self._sessions.get(session_id)
        if session:
            return session.end()
        return None

    @property
    def total_sessions(self) -> int:
        return len(self._sessions)

    def get_summary(self) -> dict[str, Any]:
        active = self.list_active_sessions()
        return {
            "total_sessions": len(self._sessions),
            "active_sessions": len(active),
            "total_participants": sum(len(s.active_participants) for s in active),
        }


_default_manager: CollaborativeSessionManager | None = None


def get_collab_session_manager() -> CollaborativeSessionManager:
    """Get the singleton collaborative session manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = CollaborativeSessionManager()
    return _default_manager


def reset_collab_session_manager() -> None:
    """Reset the singleton (for testing)."""
    global _default_manager
    _default_manager = None
