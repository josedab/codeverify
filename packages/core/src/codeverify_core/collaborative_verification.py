"""Collaborative Verification IDE - Real-time multiplayer verification sessions.

This module enables real-time collaboration where team members can annotate,
discuss, and resolve verification findings together in a shared workspace.

Key features:
1. Session Infrastructure: Create/join sessions with shareable links
2. Presence & Cursors: Show collaborators' positions and selections
3. Annotation System: Comments, approvals, dismissals on findings
4. Resolution Workflow: Finding state machine (open→discussing→resolved)
"""

import asyncio
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

import structlog

logger = structlog.get_logger()


class FindingStatus(str, Enum):
    """Status of a finding in collaborative review."""

    OPEN = "open"
    DISCUSSING = "discussing"
    NEEDS_INFO = "needs_info"
    ACKNOWLEDGED = "acknowledged"
    FIX_IN_PROGRESS = "fix_in_progress"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"
    WONT_FIX = "wont_fix"


class ParticipantRole(str, Enum):
    """Role of a participant in a session."""

    OWNER = "owner"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


class AnnotationType(str, Enum):
    """Type of annotation."""

    COMMENT = "comment"
    QUESTION = "question"
    SUGGESTION = "suggestion"
    APPROVAL = "approval"
    REQUEST_CHANGES = "request_changes"


class EventType(str, Enum):
    """Types of collaborative events."""

    PARTICIPANT_JOINED = "participant_joined"
    PARTICIPANT_LEFT = "participant_left"
    CURSOR_MOVED = "cursor_moved"
    SELECTION_CHANGED = "selection_changed"
    FINDING_STATUS_CHANGED = "finding_status_changed"
    ANNOTATION_ADDED = "annotation_added"
    ANNOTATION_RESOLVED = "annotation_resolved"
    SESSION_ENDED = "session_ended"


@dataclass
class CursorPosition:
    """Position of a collaborator's cursor."""

    file_path: str
    line: int
    column: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line": self.line,
            "column": self.column,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TextSelection:
    """A text selection by a collaborator."""

    file_path: str
    start_line: int
    start_column: int
    end_line: int
    end_column: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "start_column": self.start_column,
            "end_line": self.end_line,
            "end_column": self.end_column,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Participant:
    """A participant in a collaborative session."""

    id: str
    name: str
    email: str
    role: ParticipantRole
    avatar_url: str | None = None
    color: str = "#3498db"  # Unique color for cursor/selection
    cursor: CursorPosition | None = None
    selection: TextSelection | None = None
    joined_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)
    is_online: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "role": self.role.value,
            "avatar_url": self.avatar_url,
            "color": self.color,
            "cursor": self.cursor.to_dict() if self.cursor else None,
            "selection": self.selection.to_dict() if self.selection else None,
            "joined_at": self.joined_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "is_online": self.is_online,
        }


@dataclass
class Annotation:
    """An annotation on a finding or code."""

    id: str
    author_id: str
    author_name: str
    type: AnnotationType
    content: str
    finding_id: str | None = None  # If attached to a finding
    file_path: str | None = None  # If attached to code
    line_start: int | None = None
    line_end: int | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    resolved_by: str | None = None
    resolved_at: datetime | None = None
    replies: list["Annotation"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "type": self.type.value,
            "content": self.content,
            "finding_id": self.finding_id,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "created_at": self.created_at.isoformat(),
            "resolved": self.resolved,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "replies": [r.to_dict() for r in self.replies],
        }


@dataclass
class CollaborativeFinding:
    """A finding with collaborative review state."""

    id: str
    title: str
    description: str
    severity: str
    category: str
    file_path: str
    line_start: int
    line_end: int
    status: FindingStatus = FindingStatus.OPEN
    assignee_id: str | None = None
    annotations: list[Annotation] = field(default_factory=list)
    status_history: list[dict[str, Any]] = field(default_factory=list)
    votes: dict[str, str] = field(default_factory=dict)  # participant_id -> vote

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "category": self.category,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "status": self.status.value,
            "assignee_id": self.assignee_id,
            "annotations": [a.to_dict() for a in self.annotations],
            "status_history": self.status_history,
            "votes": self.votes,
        }


@dataclass
class CollaborativeEvent:
    """An event in a collaborative session."""

    id: str
    type: EventType
    participant_id: str
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "participant_id": self.participant_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CollaborativeSession:
    """A collaborative verification review session."""

    id: str
    name: str
    repository: str
    analysis_id: str
    created_by: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    participants: dict[str, Participant] = field(default_factory=dict)
    findings: dict[str, CollaborativeFinding] = field(default_factory=dict)
    annotations: dict[str, Annotation] = field(default_factory=dict)
    events: list[CollaborativeEvent] = field(default_factory=list)
    is_active: bool = True
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "repository": self.repository,
            "analysis_id": self.analysis_id,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "participants": {k: v.to_dict() for k, v in self.participants.items()},
            "findings": {k: v.to_dict() for k, v in self.findings.items()},
            "is_active": self.is_active,
            "settings": self.settings,
        }


class EventBroadcaster:
    """Broadcasts events to session participants."""

    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable[[CollaborativeEvent], None]]] = {}

    def subscribe(
        self,
        session_id: str,
        callback: Callable[[CollaborativeEvent], None],
    ) -> str:
        """Subscribe to session events. Returns subscription ID."""
        if session_id not in self._listeners:
            self._listeners[session_id] = []

        subscription_id = str(uuid.uuid4())
        self._listeners[session_id].append(callback)
        return subscription_id

    def unsubscribe(self, session_id: str, callback: Callable) -> None:
        """Unsubscribe from session events."""
        if session_id in self._listeners:
            self._listeners[session_id] = [
                cb for cb in self._listeners[session_id] if cb != callback
            ]

    def broadcast(self, session_id: str, event: CollaborativeEvent) -> None:
        """Broadcast event to all session subscribers."""
        listeners = self._listeners.get(session_id, [])
        for callback in listeners:
            try:
                callback(event)
            except Exception as e:
                logger.error("Event broadcast failed", error=str(e))


class CollaborativeSessionManager:
    """Manages collaborative verification sessions."""

    PARTICIPANT_COLORS = [
        "#3498db", "#e74c3c", "#2ecc71", "#f39c12",
        "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
    ]

    def __init__(self) -> None:
        self._sessions: dict[str, CollaborativeSession] = {}
        self._broadcaster = EventBroadcaster()
        self._color_index: dict[str, int] = {}

    def create_session(
        self,
        name: str,
        repository: str,
        analysis_id: str,
        created_by: str,
        findings: list[dict[str, Any]],
        settings: dict[str, Any] | None = None,
    ) -> CollaborativeSession:
        """Create a new collaborative session."""
        session_id = str(uuid.uuid4())

        # Convert findings to collaborative findings
        collab_findings = {}
        for finding in findings:
            finding_id = finding.get("id", str(uuid.uuid4()))
            collab_findings[finding_id] = CollaborativeFinding(
                id=finding_id,
                title=finding.get("title", "Untitled"),
                description=finding.get("description", ""),
                severity=finding.get("severity", "medium"),
                category=finding.get("category", "unknown"),
                file_path=finding.get("file_path", ""),
                line_start=finding.get("line_start", 0),
                line_end=finding.get("line_end", 0),
            )

        session = CollaborativeSession(
            id=session_id,
            name=name,
            repository=repository,
            analysis_id=analysis_id,
            created_by=created_by,
            findings=collab_findings,
            settings=settings or {},
        )

        self._sessions[session_id] = session
        self._color_index[session_id] = 0

        logger.info("Created collaborative session", session_id=session_id, name=name)

        return session

    def get_session(self, session_id: str) -> CollaborativeSession | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def get_shareable_link(self, session_id: str, base_url: str = "") -> str:
        """Generate a shareable link for a session."""
        return f"{base_url}/collaborate/{session_id}"

    def join_session(
        self,
        session_id: str,
        user_id: str,
        user_name: str,
        user_email: str,
        role: ParticipantRole = ParticipantRole.REVIEWER,
    ) -> Participant | None:
        """Join an existing session."""
        session = self._sessions.get(session_id)
        if not session or not session.is_active:
            return None

        # Assign color
        color_idx = self._color_index.get(session_id, 0)
        color = self.PARTICIPANT_COLORS[color_idx % len(self.PARTICIPANT_COLORS)]
        self._color_index[session_id] = color_idx + 1

        participant = Participant(
            id=user_id,
            name=user_name,
            email=user_email,
            role=role,
            color=color,
        )

        session.participants[user_id] = participant

        # Broadcast join event
        event = CollaborativeEvent(
            id=str(uuid.uuid4()),
            type=EventType.PARTICIPANT_JOINED,
            participant_id=user_id,
            data={"participant": participant.to_dict()},
        )
        session.events.append(event)
        self._broadcaster.broadcast(session_id, event)

        logger.info(
            "Participant joined session",
            session_id=session_id,
            user_id=user_id,
        )

        return participant

    def leave_session(self, session_id: str, user_id: str) -> None:
        """Leave a session."""
        session = self._sessions.get(session_id)
        if not session:
            return

        if user_id in session.participants:
            session.participants[user_id].is_online = False

            event = CollaborativeEvent(
                id=str(uuid.uuid4()),
                type=EventType.PARTICIPANT_LEFT,
                participant_id=user_id,
                data={},
            )
            session.events.append(event)
            self._broadcaster.broadcast(session_id, event)

    def end_session(self, session_id: str, ended_by: str) -> None:
        """End a session."""
        session = self._sessions.get(session_id)
        if not session:
            return

        session.is_active = False

        event = CollaborativeEvent(
            id=str(uuid.uuid4()),
            type=EventType.SESSION_ENDED,
            participant_id=ended_by,
            data={},
        )
        session.events.append(event)
        self._broadcaster.broadcast(session_id, event)

        logger.info("Session ended", session_id=session_id)

    def update_cursor(
        self,
        session_id: str,
        user_id: str,
        position: CursorPosition,
    ) -> None:
        """Update a participant's cursor position."""
        session = self._sessions.get(session_id)
        if not session or user_id not in session.participants:
            return

        participant = session.participants[user_id]
        participant.cursor = position
        participant.last_active = datetime.utcnow()

        event = CollaborativeEvent(
            id=str(uuid.uuid4()),
            type=EventType.CURSOR_MOVED,
            participant_id=user_id,
            data={"cursor": position.to_dict()},
        )
        self._broadcaster.broadcast(session_id, event)

    def update_selection(
        self,
        session_id: str,
        user_id: str,
        selection: TextSelection | None,
    ) -> None:
        """Update a participant's text selection."""
        session = self._sessions.get(session_id)
        if not session or user_id not in session.participants:
            return

        participant = session.participants[user_id]
        participant.selection = selection
        participant.last_active = datetime.utcnow()

        event = CollaborativeEvent(
            id=str(uuid.uuid4()),
            type=EventType.SELECTION_CHANGED,
            participant_id=user_id,
            data={"selection": selection.to_dict() if selection else None},
        )
        self._broadcaster.broadcast(session_id, event)

    def change_finding_status(
        self,
        session_id: str,
        finding_id: str,
        user_id: str,
        new_status: FindingStatus,
        comment: str | None = None,
    ) -> bool:
        """Change the status of a finding."""
        session = self._sessions.get(session_id)
        if not session or finding_id not in session.findings:
            return False

        finding = session.findings[finding_id]
        old_status = finding.status
        finding.status = new_status

        # Record history
        history_entry = {
            "from": old_status.value,
            "to": new_status.value,
            "changed_by": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "comment": comment,
        }
        finding.status_history.append(history_entry)

        event = CollaborativeEvent(
            id=str(uuid.uuid4()),
            type=EventType.FINDING_STATUS_CHANGED,
            participant_id=user_id,
            data={
                "finding_id": finding_id,
                "old_status": old_status.value,
                "new_status": new_status.value,
                "comment": comment,
            },
        )
        session.events.append(event)
        self._broadcaster.broadcast(session_id, event)

        return True

    def add_annotation(
        self,
        session_id: str,
        user_id: str,
        annotation_type: AnnotationType,
        content: str,
        finding_id: str | None = None,
        file_path: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
        parent_id: str | None = None,
    ) -> Annotation | None:
        """Add an annotation to a finding or code."""
        session = self._sessions.get(session_id)
        if not session or user_id not in session.participants:
            return None

        participant = session.participants[user_id]

        annotation = Annotation(
            id=str(uuid.uuid4()),
            author_id=user_id,
            author_name=participant.name,
            type=annotation_type,
            content=content,
            finding_id=finding_id,
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
        )

        # Add to session annotations
        session.annotations[annotation.id] = annotation

        # Add to finding if specified
        if finding_id and finding_id in session.findings:
            session.findings[finding_id].annotations.append(annotation)

        # Handle reply
        if parent_id and parent_id in session.annotations:
            session.annotations[parent_id].replies.append(annotation)

        event = CollaborativeEvent(
            id=str(uuid.uuid4()),
            type=EventType.ANNOTATION_ADDED,
            participant_id=user_id,
            data={"annotation": annotation.to_dict()},
        )
        session.events.append(event)
        self._broadcaster.broadcast(session_id, event)

        return annotation

    def resolve_annotation(
        self,
        session_id: str,
        annotation_id: str,
        user_id: str,
    ) -> bool:
        """Resolve an annotation."""
        session = self._sessions.get(session_id)
        if not session or annotation_id not in session.annotations:
            return False

        annotation = session.annotations[annotation_id]
        annotation.resolved = True
        annotation.resolved_by = user_id
        annotation.resolved_at = datetime.utcnow()

        event = CollaborativeEvent(
            id=str(uuid.uuid4()),
            type=EventType.ANNOTATION_RESOLVED,
            participant_id=user_id,
            data={"annotation_id": annotation_id},
        )
        session.events.append(event)
        self._broadcaster.broadcast(session_id, event)

        return True

    def vote_on_finding(
        self,
        session_id: str,
        finding_id: str,
        user_id: str,
        vote: str,  # "approve", "reject", "needs_discussion"
    ) -> bool:
        """Vote on a finding."""
        session = self._sessions.get(session_id)
        if not session or finding_id not in session.findings:
            return False

        session.findings[finding_id].votes[user_id] = vote
        return True

    def assign_finding(
        self,
        session_id: str,
        finding_id: str,
        assignee_id: str,
    ) -> bool:
        """Assign a finding to a participant."""
        session = self._sessions.get(session_id)
        if not session or finding_id not in session.findings:
            return False

        if assignee_id and assignee_id not in session.participants:
            return False

        session.findings[finding_id].assignee_id = assignee_id
        return True

    def subscribe_to_events(
        self,
        session_id: str,
        callback: Callable[[CollaborativeEvent], None],
    ) -> str:
        """Subscribe to session events."""
        return self._broadcaster.subscribe(session_id, callback)

    def unsubscribe_from_events(
        self,
        session_id: str,
        callback: Callable,
    ) -> None:
        """Unsubscribe from session events."""
        self._broadcaster.unsubscribe(session_id, callback)

    def get_session_summary(self, session_id: str) -> dict[str, Any] | None:
        """Get a summary of the session state."""
        session = self._sessions.get(session_id)
        if not session:
            return None

        # Count findings by status
        status_counts = {}
        for finding in session.findings.values():
            status = finding.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        # Count annotations
        total_annotations = len(session.annotations)
        resolved_annotations = sum(
            1 for a in session.annotations.values() if a.resolved
        )

        return {
            "session_id": session_id,
            "name": session.name,
            "is_active": session.is_active,
            "participants": {
                "total": len(session.participants),
                "online": sum(1 for p in session.participants.values() if p.is_online),
            },
            "findings": {
                "total": len(session.findings),
                "by_status": status_counts,
            },
            "annotations": {
                "total": total_annotations,
                "resolved": resolved_annotations,
                "unresolved": total_annotations - resolved_annotations,
            },
            "events_count": len(session.events),
        }


# Singleton instance
_session_manager: CollaborativeSessionManager | None = None


def get_session_manager() -> CollaborativeSessionManager:
    """Get the global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = CollaborativeSessionManager()
    return _session_manager


def reset_session_manager() -> None:
    """Reset the global session manager (for testing)."""
    global _session_manager
    _session_manager = None
