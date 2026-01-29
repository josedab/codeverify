"""
AI-to-AI Collaboration Protocol

Enables real-time communication between CodeVerify and AI code assistants
(GitHub Copilot, Claude, etc.) to provide verification constraints during
code generation, preventing bugs before they're written.

Features:
- Collaboration message format for AI-to-AI communication
- Real-time constraint streaming during code generation
- Verification feedback loop for iterative refinement
- Context sharing for verification-aware code suggestions
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    TypeVar,
)


# =============================================================================
# Collaboration Message Protocol
# =============================================================================

class MessageType(str, Enum):
    """Types of collaboration messages."""
    
    # From CodeVerify to AI Assistant
    CONSTRAINT = "constraint"           # Verification constraint to respect
    WARNING = "warning"                  # Potential issue detected
    SUGGESTION = "suggestion"            # Code modification suggestion
    CONTEXT = "context"                  # Contextual information
    PROOF_REQUEST = "proof_request"      # Request for provable code
    
    # From AI Assistant to CodeVerify
    CODE_PROPOSAL = "code_proposal"      # Proposed code for verification
    QUERY = "query"                      # Question about constraints
    ACKNOWLEDGMENT = "ack"               # Acknowledgment of constraint
    
    # Bidirectional
    SYNC = "sync"                        # State synchronization
    HEARTBEAT = "heartbeat"              # Keep-alive


class Severity(str, Enum):
    """Severity levels for messages."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class CollaborationMessage:
    """
    Standard message format for AI-to-AI collaboration.
    
    Designed to be:
    - Machine-parseable for AI consumption
    - Human-readable for debugging
    - Extensible for future message types
    """
    
    message_id: str
    message_type: MessageType
    timestamp: float
    
    # Content
    content: Dict[str, Any]
    
    # Metadata
    source: str = "codeverify"
    target: str = "ai_assistant"
    severity: Severity = Severity.INFO
    
    # Context references
    file_path: Optional[str] = None
    line_range: Optional[tuple[int, int]] = None
    code_context: Optional[str] = None
    
    # Threading
    reply_to: Optional[str] = None
    conversation_id: Optional[str] = None
    
    @classmethod
    def create(
        cls,
        message_type: MessageType,
        content: Dict[str, Any],
        **kwargs: Any,
    ) -> CollaborationMessage:
        """Create a new collaboration message."""
        return cls(
            message_id=hashlib.sha256(
                f"{time.time()}-{message_type}-{json.dumps(content)}".encode()
            ).hexdigest()[:16],
            message_type=message_type,
            timestamp=time.time(),
            content=content,
            **kwargs,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp,
            "content": self.content,
            "source": self.source,
            "target": self.target,
            "severity": self.severity.value,
            "file_path": self.file_path,
            "line_range": self.line_range,
            "code_context": self.code_context,
            "reply_to": self.reply_to,
            "conversation_id": self.conversation_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CollaborationMessage:
        """Create from dictionary."""
        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data["message_type"]),
            timestamp=data["timestamp"],
            content=data["content"],
            source=data.get("source", "unknown"),
            target=data.get("target", "unknown"),
            severity=Severity(data.get("severity", "info")),
            file_path=data.get("file_path"),
            line_range=tuple(data["line_range"]) if data.get("line_range") else None,
            code_context=data.get("code_context"),
            reply_to=data.get("reply_to"),
            conversation_id=data.get("conversation_id"),
        )
    
    def to_prompt_injection(self) -> str:
        """
        Convert to a format suitable for prompt injection.
        
        This allows verification constraints to be injected into
        the AI assistant's context/system prompt.
        """
        if self.message_type == MessageType.CONSTRAINT:
            return self._format_constraint_prompt()
        elif self.message_type == MessageType.WARNING:
            return self._format_warning_prompt()
        elif self.message_type == MessageType.CONTEXT:
            return self._format_context_prompt()
        else:
            return json.dumps(self.content)
    
    def _format_constraint_prompt(self) -> str:
        """Format constraint as prompt text."""
        constraint = self.content
        lines = [
            f"[VERIFICATION CONSTRAINT - {self.severity.value.upper()}]",
            f"Type: {constraint.get('constraint_type', 'general')}",
            f"Rule: {constraint.get('rule', 'No rule specified')}",
        ]
        if constraint.get("reason"):
            lines.append(f"Reason: {constraint['reason']}")
        if constraint.get("example_violation"):
            lines.append(f"Example violation: {constraint['example_violation']}")
        if constraint.get("example_correct"):
            lines.append(f"Correct example: {constraint['example_correct']}")
        return "\n".join(lines)
    
    def _format_warning_prompt(self) -> str:
        """Format warning as prompt text."""
        return f"[WARNING - {self.severity.value.upper()}] {self.content.get('message', '')}"
    
    def _format_context_prompt(self) -> str:
        """Format context as prompt text."""
        return f"[CONTEXT] {json.dumps(self.content)}"


# =============================================================================
# Constraint Types
# =============================================================================

@dataclass
class VerificationConstraint:
    """A constraint that generated code must satisfy."""
    
    constraint_id: str
    constraint_type: str
    rule: str
    
    # Optional details
    formal_spec: Optional[str] = None  # Z3/SMT-LIB specification
    natural_language: Optional[str] = None
    example_violation: Optional[str] = None
    example_correct: Optional[str] = None
    
    # Scope
    applies_to: Optional[List[str]] = None  # function names, patterns
    language: Optional[str] = None
    
    # Enforcement
    severity: Severity = Severity.MEDIUM
    auto_fix_available: bool = False
    
    def to_message(self, file_path: Optional[str] = None) -> CollaborationMessage:
        """Convert to a collaboration message."""
        return CollaborationMessage.create(
            message_type=MessageType.CONSTRAINT,
            content={
                "constraint_id": self.constraint_id,
                "constraint_type": self.constraint_type,
                "rule": self.rule,
                "formal_spec": self.formal_spec,
                "natural_language": self.natural_language,
                "example_violation": self.example_violation,
                "example_correct": self.example_correct,
                "applies_to": self.applies_to,
                "language": self.language,
                "auto_fix_available": self.auto_fix_available,
            },
            severity=self.severity,
            file_path=file_path,
        )


# Standard constraints library
STANDARD_CONSTRAINTS = {
    "null_safety": VerificationConstraint(
        constraint_id="null_safety",
        constraint_type="type_safety",
        rule="Never return null/None without explicit documentation",
        natural_language="Functions should not return null/None unless the return type explicitly allows it",
        example_violation="def get_user(id): return None  # Missing Optional type",
        example_correct="def get_user(id) -> Optional[User]: return None  # OK",
        severity=Severity.HIGH,
    ),
    "bounds_check": VerificationConstraint(
        constraint_id="bounds_check",
        constraint_type="memory_safety",
        rule="Array/list access must be bounds-checked",
        natural_language="Before accessing array elements, verify the index is within bounds",
        example_violation="items[i]  # i could be out of bounds",
        example_correct="if 0 <= i < len(items): items[i]",
        severity=Severity.HIGH,
    ),
    "division_safety": VerificationConstraint(
        constraint_id="division_safety",
        constraint_type="arithmetic_safety",
        rule="Division must check for zero divisor",
        natural_language="Before dividing, ensure the divisor is not zero",
        example_violation="result = a / b  # b could be zero",
        example_correct="result = a / b if b != 0 else 0",
        severity=Severity.CRITICAL,
    ),
    "sql_injection": VerificationConstraint(
        constraint_id="sql_injection",
        constraint_type="security",
        rule="Use parameterized queries for SQL",
        natural_language="Never concatenate user input into SQL queries",
        example_violation='query = f"SELECT * FROM users WHERE id={user_id}"',
        example_correct='query = "SELECT * FROM users WHERE id=?"; params = (user_id,)',
        severity=Severity.CRITICAL,
    ),
    "resource_cleanup": VerificationConstraint(
        constraint_id="resource_cleanup",
        constraint_type="resource_safety",
        rule="Resources must be properly closed",
        natural_language="File handles, connections, and other resources must be closed in all code paths",
        example_violation="f = open('file.txt'); data = f.read()  # Never closed",
        example_correct="with open('file.txt') as f: data = f.read()",
        severity=Severity.MEDIUM,
    ),
}


# =============================================================================
# Collaboration Session
# =============================================================================

@dataclass
class CollaborationSession:
    """
    Manages a collaboration session between CodeVerify and an AI assistant.
    
    Tracks:
    - Active constraints
    - Message history
    - Verification state
    - Code proposals and feedback
    """
    
    session_id: str
    ai_assistant: str  # e.g., "github_copilot", "claude"
    started_at: float = field(default_factory=time.time)
    
    # State
    active_constraints: Dict[str, VerificationConstraint] = field(default_factory=dict)
    message_history: List[CollaborationMessage] = field(default_factory=list)
    pending_proposals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Context
    file_path: Optional[str] = None
    language: Optional[str] = None
    project_context: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    constraints_sent: int = 0
    proposals_received: int = 0
    proposals_accepted: int = 0
    proposals_rejected: int = 0
    
    def add_constraint(self, constraint: VerificationConstraint) -> CollaborationMessage:
        """Add a constraint and create a message to send."""
        self.active_constraints[constraint.constraint_id] = constraint
        self.constraints_sent += 1
        
        message = constraint.to_message(self.file_path)
        message.conversation_id = self.session_id
        self.message_history.append(message)
        
        return message
    
    def receive_proposal(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, CollaborationMessage]:
        """
        Receive a code proposal from the AI assistant.
        
        Returns proposal_id and acknowledgment message.
        """
        proposal_id = hashlib.sha256(
            f"{time.time()}-{code[:100]}".encode()
        ).hexdigest()[:16]
        
        self.pending_proposals[proposal_id] = {
            "code": code,
            "context": context or {},
            "received_at": time.time(),
            "status": "pending",
        }
        self.proposals_received += 1
        
        ack = CollaborationMessage.create(
            message_type=MessageType.ACKNOWLEDGMENT,
            content={
                "proposal_id": proposal_id,
                "status": "received",
                "active_constraints": list(self.active_constraints.keys()),
            },
            target=self.ai_assistant,
        )
        ack.conversation_id = self.session_id
        self.message_history.append(ack)
        
        return proposal_id, ack
    
    def provide_feedback(
        self,
        proposal_id: str,
        verified: bool,
        issues: List[Dict[str, Any]],
        suggestions: List[str],
    ) -> CollaborationMessage:
        """Provide verification feedback on a proposal."""
        if proposal_id not in self.pending_proposals:
            raise ValueError(f"Unknown proposal: {proposal_id}")
        
        proposal = self.pending_proposals[proposal_id]
        proposal["status"] = "verified" if verified else "rejected"
        proposal["verified"] = verified
        proposal["issues"] = issues
        
        if verified:
            self.proposals_accepted += 1
        else:
            self.proposals_rejected += 1
        
        message = CollaborationMessage.create(
            message_type=MessageType.WARNING if issues else MessageType.SUGGESTION,
            content={
                "proposal_id": proposal_id,
                "verified": verified,
                "issues": issues,
                "suggestions": suggestions,
            },
            severity=Severity.HIGH if not verified else Severity.INFO,
            target=self.ai_assistant,
        )
        message.conversation_id = self.session_id
        self.message_history.append(message)
        
        return message
    
    def get_context_for_ai(self) -> str:
        """
        Generate context string to inject into AI assistant's prompt.
        
        This provides the AI with all active constraints and recent
        verification feedback.
        """
        lines = [
            "=== CodeVerify Collaboration Context ===",
            f"Session: {self.session_id}",
            f"File: {self.file_path or 'unknown'}",
            f"Language: {self.language or 'auto-detect'}",
            "",
            "Active Verification Constraints:",
        ]
        
        for constraint in self.active_constraints.values():
            lines.append(f"  - [{constraint.severity.value}] {constraint.rule}")
        
        # Add recent feedback
        recent_warnings = [
            m for m in self.message_history[-10:]
            if m.message_type == MessageType.WARNING
        ]
        
        if recent_warnings:
            lines.append("")
            lines.append("Recent Verification Warnings:")
            for warning in recent_warnings[-3:]:
                lines.append(f"  - {warning.content.get('issues', [])}")
        
        lines.append("")
        lines.append("Please ensure generated code satisfies all constraints.")
        lines.append("=== End CodeVerify Context ===")
        
        return "\n".join(lines)


# =============================================================================
# Real-Time Constraint Streaming
# =============================================================================

class ConstraintStreamHandler(Protocol):
    """Protocol for handling streamed constraints."""
    
    async def on_constraint(self, constraint: VerificationConstraint) -> None:
        """Called when a new constraint is detected."""
        ...
    
    async def on_warning(self, message: CollaborationMessage) -> None:
        """Called when a warning is generated."""
        ...
    
    async def on_suggestion(self, message: CollaborationMessage) -> None:
        """Called when a suggestion is generated."""
        ...


class ConstraintStreamer:
    """
    Streams verification constraints in real-time as code is being generated.
    
    Monitors code changes and emits constraints that the AI assistant
    should respect for the remaining code generation.
    """
    
    def __init__(
        self,
        session: CollaborationSession,
        handler: Optional[ConstraintStreamHandler] = None,
    ):
        self.session = session
        self.handler = handler
        self._running = False
        self._buffer: List[str] = []
        self._last_analysis_time = 0.0
        self._analysis_interval = 0.1  # 100ms
    
    async def start(self) -> None:
        """Start the constraint streamer."""
        self._running = True
    
    async def stop(self) -> None:
        """Stop the constraint streamer."""
        self._running = False
    
    async def feed_code(self, code_chunk: str) -> AsyncGenerator[CollaborationMessage, None]:
        """
        Feed a chunk of code being generated.
        
        Analyzes incrementally and yields relevant constraints/warnings.
        """
        self._buffer.append(code_chunk)
        current_code = "".join(self._buffer)
        
        # Rate limit analysis
        now = time.time()
        if now - self._last_analysis_time < self._analysis_interval:
            return
        
        self._last_analysis_time = now
        
        # Analyze current code state
        async for message in self._analyze_code(current_code):
            yield message
    
    async def _analyze_code(
        self,
        code: str,
    ) -> AsyncGenerator[CollaborationMessage, None]:
        """Analyze code and yield relevant messages."""
        # Check for common patterns that violate constraints
        checks = [
            self._check_null_safety,
            self._check_bounds_safety,
            self._check_division_safety,
            self._check_sql_injection,
            self._check_resource_cleanup,
        ]
        
        for check in checks:
            issues = check(code)
            for issue in issues:
                message = CollaborationMessage.create(
                    message_type=MessageType.WARNING,
                    content=issue,
                    severity=Severity(issue.get("severity", "medium")),
                    file_path=self.session.file_path,
                )
                message.conversation_id = self.session.session_id
                
                if self.handler:
                    await self.handler.on_warning(message)
                
                yield message
    
    def _check_null_safety(self, code: str) -> List[Dict[str, Any]]:
        """Check for null safety issues."""
        issues = []
        
        # Simple pattern detection
        if "return None" in code and "Optional" not in code and "| None" not in code:
            issues.append({
                "type": "null_safety",
                "message": "Function may return None without Optional type annotation",
                "severity": "high",
                "constraint_id": "null_safety",
            })
        
        return issues
    
    def _check_bounds_safety(self, code: str) -> List[Dict[str, Any]]:
        """Check for bounds safety issues."""
        issues = []
        import re
        
        # Check for array access without bounds check
        array_access = re.findall(r"(\w+)\[(\w+)\]", code)
        for array, index in array_access:
            if f"len({array})" not in code and f"range(len({array}))" not in code:
                if index not in ("0", "1", "-1"):
                    issues.append({
                        "type": "bounds_check",
                        "message": f"Array access {array}[{index}] may be out of bounds",
                        "severity": "high",
                        "constraint_id": "bounds_check",
                    })
        
        return issues
    
    def _check_division_safety(self, code: str) -> List[Dict[str, Any]]:
        """Check for division safety issues."""
        issues = []
        import re
        
        # Check for division without zero check
        divisions = re.findall(r"(\w+)\s*/\s*(\w+)", code)
        for _, divisor in divisions:
            if divisor not in ("2", "10", "100", "1000"):
                if f"if {divisor}" not in code and f"{divisor} != 0" not in code:
                    issues.append({
                        "type": "division_safety",
                        "message": f"Division by {divisor} without zero check",
                        "severity": "critical",
                        "constraint_id": "division_safety",
                    })
        
        return issues
    
    def _check_sql_injection(self, code: str) -> List[Dict[str, Any]]:
        """Check for SQL injection vulnerabilities."""
        issues = []
        import re
        
        # Check for string formatting in SQL
        sql_patterns = [
            r'f"[^"]*SELECT[^"]*\{',
            r'f"[^"]*INSERT[^"]*\{',
            r'f"[^"]*UPDATE[^"]*\{',
            r'f"[^"]*DELETE[^"]*\{',
            r'"[^"]*SELECT[^"]*"\s*%',
            r'"[^"]*INSERT[^"]*"\s*%',
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append({
                    "type": "sql_injection",
                    "message": "Potential SQL injection: use parameterized queries",
                    "severity": "critical",
                    "constraint_id": "sql_injection",
                })
                break
        
        return issues
    
    def _check_resource_cleanup(self, code: str) -> List[Dict[str, Any]]:
        """Check for resource cleanup issues."""
        issues = []
        
        # Check for file open without context manager
        if "open(" in code and "with " not in code:
            issues.append({
                "type": "resource_cleanup",
                "message": "File opened without context manager (with statement)",
                "severity": "medium",
                "constraint_id": "resource_cleanup",
            })
        
        return issues


# =============================================================================
# Copilot Integration
# =============================================================================

@dataclass
class CopilotIntegrationConfig:
    """Configuration for GitHub Copilot integration."""
    
    enabled: bool = True
    inject_constraints: bool = True
    stream_warnings: bool = True
    block_on_critical: bool = False
    
    # Constraint categories to enforce
    enforce_null_safety: bool = True
    enforce_bounds_check: bool = True
    enforce_division_safety: bool = True
    enforce_sql_injection: bool = True
    enforce_resource_cleanup: bool = True


class CopilotCollaborator:
    """
    Integrates CodeVerify with GitHub Copilot.
    
    Provides:
    - Constraint injection into Copilot context
    - Real-time verification of Copilot suggestions
    - Feedback loop for iterative improvement
    """
    
    def __init__(self, config: Optional[CopilotIntegrationConfig] = None):
        self.config = config or CopilotIntegrationConfig()
        self.active_sessions: Dict[str, CollaborationSession] = {}
        self._streamers: Dict[str, ConstraintStreamer] = {}
    
    def create_session(
        self,
        file_path: str,
        language: str,
        project_context: Optional[Dict[str, Any]] = None,
    ) -> CollaborationSession:
        """Create a new collaboration session for a file."""
        session_id = hashlib.sha256(
            f"{file_path}-{time.time()}".encode()
        ).hexdigest()[:16]
        
        session = CollaborationSession(
            session_id=session_id,
            ai_assistant="github_copilot",
            file_path=file_path,
            language=language,
            project_context=project_context or {},
        )
        
        # Add relevant constraints based on config
        if self.config.enforce_null_safety:
            session.add_constraint(STANDARD_CONSTRAINTS["null_safety"])
        if self.config.enforce_bounds_check:
            session.add_constraint(STANDARD_CONSTRAINTS["bounds_check"])
        if self.config.enforce_division_safety:
            session.add_constraint(STANDARD_CONSTRAINTS["division_safety"])
        if self.config.enforce_sql_injection:
            session.add_constraint(STANDARD_CONSTRAINTS["sql_injection"])
        if self.config.enforce_resource_cleanup:
            session.add_constraint(STANDARD_CONSTRAINTS["resource_cleanup"])
        
        self.active_sessions[session_id] = session
        self._streamers[session_id] = ConstraintStreamer(session)
        
        return session
    
    def get_copilot_system_prompt_addition(
        self,
        session_id: str,
    ) -> str:
        """
        Get text to add to Copilot's system prompt.
        
        This injects verification constraints into Copilot's context.
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return ""
        
        return session.get_context_for_ai()
    
    async def verify_suggestion(
        self,
        session_id: str,
        suggestion: str,
    ) -> Dict[str, Any]:
        """
        Verify a Copilot suggestion against active constraints.
        
        Returns verification result with issues and suggestions.
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        # Receive the proposal
        proposal_id, _ = session.receive_proposal(suggestion)
        
        # Analyze the suggestion
        issues = []
        streamer = self._streamers.get(session_id)
        
        if streamer:
            async for message in streamer._analyze_code(suggestion):
                if message.message_type == MessageType.WARNING:
                    issues.append(message.content)
        
        # Generate suggestions for fixing issues
        suggestions = []
        for issue in issues:
            constraint_id = issue.get("constraint_id")
            if constraint_id in STANDARD_CONSTRAINTS:
                constraint = STANDARD_CONSTRAINTS[constraint_id]
                if constraint.example_correct:
                    suggestions.append(
                        f"Consider: {constraint.example_correct}"
                    )
        
        # Provide feedback
        verified = len(issues) == 0
        session.provide_feedback(proposal_id, verified, issues, suggestions)
        
        return {
            "verified": verified,
            "proposal_id": proposal_id,
            "issues": issues,
            "suggestions": suggestions,
            "should_block": (
                self.config.block_on_critical and
                any(i.get("severity") == "critical" for i in issues)
            ),
        }
    
    async def stream_verification(
        self,
        session_id: str,
        code_stream: AsyncGenerator[str, None],
    ) -> AsyncGenerator[CollaborationMessage, None]:
        """
        Stream verification as code is being generated.
        
        Yields warnings and suggestions in real-time.
        """
        streamer = self._streamers.get(session_id)
        if not streamer:
            return
        
        await streamer.start()
        
        try:
            async for chunk in code_stream:
                async for message in streamer.feed_code(chunk):
                    yield message
        finally:
            await streamer.stop()
    
    def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """End a collaboration session and return statistics."""
        session = self.active_sessions.pop(session_id, None)
        self._streamers.pop(session_id, None)
        
        if not session:
            return None
        
        return {
            "session_id": session_id,
            "duration_seconds": time.time() - session.started_at,
            "constraints_sent": session.constraints_sent,
            "proposals_received": session.proposals_received,
            "proposals_accepted": session.proposals_accepted,
            "proposals_rejected": session.proposals_rejected,
            "acceptance_rate": (
                session.proposals_accepted / session.proposals_received
                if session.proposals_received > 0
                else 0.0
            ),
        }


# =============================================================================
# MCP Tool for AI Collaboration
# =============================================================================

def create_collaboration_tools() -> List[Dict[str, Any]]:
    """
    Create MCP-compatible tool definitions for AI collaboration.
    
    These tools can be used by AI assistants to interact with CodeVerify.
    """
    return [
        {
            "name": "codeverify_get_constraints",
            "description": "Get active verification constraints for the current file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file being edited",
                    },
                    "language": {
                        "type": "string",
                        "description": "Programming language",
                    },
                },
                "required": ["file_path"],
            },
        },
        {
            "name": "codeverify_verify_code",
            "description": "Verify code against active constraints before suggesting",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Code to verify",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "File context",
                    },
                },
                "required": ["code"],
            },
        },
        {
            "name": "codeverify_report_issue",
            "description": "Report a verification issue found in generated code",
            "parameters": {
                "type": "object",
                "properties": {
                    "issue_type": {
                        "type": "string",
                        "description": "Type of issue",
                    },
                    "description": {
                        "type": "string",
                        "description": "Issue description",
                    },
                    "code_snippet": {
                        "type": "string",
                        "description": "Affected code",
                    },
                },
                "required": ["issue_type", "description"],
            },
        },
    ]
