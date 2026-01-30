"""Real-Time Copilot Session Integration.

This module provides:
- Streaming Copilot SDK sessions for real-time review
- Context-aware prompting using surrounding code
- Smart session management with pooling
- Feedback loop for user corrections
- Integration with pair reviewer agent
"""

import asyncio
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable
from uuid import uuid4

import structlog

logger = structlog.get_logger()


class SessionState(str, Enum):
    """State of a Copilot session."""
    
    IDLE = "idle"
    PREPARING = "preparing"
    STREAMING = "streaming"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class StreamingFinding:
    """A finding streamed from the Copilot session."""
    
    id: str
    partial: bool  # True if still being streamed
    category: str
    priority: str
    message: str
    line_start: int
    line_end: int
    suggestion: str | None = None
    fix_code: str | None = None
    confidence: float = 0.8
    tokens_so_far: int = 0


@dataclass
class SessionContext:
    """Context for a Copilot review session."""
    
    session_id: str
    file_path: str
    language: str
    code: str
    surrounding_context: str = ""
    imports: list[str] = field(default_factory=list)
    function_signature: str | None = None
    class_context: str | None = None
    previous_corrections: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class SessionResult:
    """Result from a completed session."""
    
    session_id: str
    state: SessionState
    findings: list[StreamingFinding] = field(default_factory=list)
    tokens_used: int = 0
    latency_ms: float = 0
    error: str | None = None


class CopilotSessionPool:
    """Pool of reusable Copilot sessions for efficiency.
    
    Maintains a pool of warm sessions to reduce cold start latency.
    Sessions are language-specific for better context retention.
    """
    
    def __init__(
        self,
        max_sessions_per_language: int = 3,
        session_ttl_seconds: float = 300,
    ):
        self.max_sessions_per_language = max_sessions_per_language
        self.session_ttl_seconds = session_ttl_seconds
        
        self._pools: dict[str, list[tuple[str, float]]] = defaultdict(list)  # language -> [(session_id, created_at)]
        self._active_sessions: dict[str, SessionContext] = {}
        self._lock = asyncio.Lock()
    
    async def acquire(self, language: str) -> str:
        """Acquire a session for the given language."""
        async with self._lock:
            pool = self._pools[language]
            
            # Clean up expired sessions
            now = time.time()
            pool[:] = [
                (sid, created) for sid, created in pool
                if now - created < self.session_ttl_seconds
            ]
            
            # Try to reuse an existing session
            if pool:
                session_id, _ = pool.pop(0)
                return session_id
            
            # Create a new session
            session_id = f"copilot_session_{language}_{uuid4().hex[:8]}"
            return session_id
    
    async def release(self, session_id: str, language: str) -> None:
        """Release a session back to the pool."""
        async with self._lock:
            pool = self._pools[language]
            
            # Only keep up to max sessions
            if len(pool) < self.max_sessions_per_language:
                pool.append((session_id, time.time()))
            
            # Remove from active
            self._active_sessions.pop(session_id, None)
    
    async def invalidate(self, session_id: str) -> None:
        """Invalidate a session (on error)."""
        async with self._lock:
            self._active_sessions.pop(session_id, None)
            # Remove from all pools
            for pool in self._pools.values():
                pool[:] = [(sid, t) for sid, t in pool if sid != session_id]
    
    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            "pools": {lang: len(sessions) for lang, sessions in self._pools.items()},
            "active_sessions": len(self._active_sessions),
            "max_per_language": self.max_sessions_per_language,
        }


class CopilotReviewSession:
    """A streaming Copilot review session.
    
    Provides real-time code review feedback by streaming from Copilot SDK.
    Designed for sub-3-second latency feedback.
    """
    
    SYSTEM_PROMPT = """You are an expert code reviewer performing REAL-TIME pair programming review.
Your job is to catch issues BEFORE code is committed - be fast and focused.

RESPOND ONLY WITH VALID JSON in this exact format:
{
    "findings": [
        {
            "category": "bug|security|logic_error|type_error|null_safety|performance",
            "priority": "critical|high|medium|low",
            "message": "Clear, brief description",
            "line_start": <number>,
            "line_end": <number>,
            "suggestion": "How to fix",
            "fix_code": "Corrected code (if simple)",
            "confidence": <0.0-1.0>
        }
    ],
    "summary": "One-sentence assessment"
}

RULES:
1. Focus on BUGS and SECURITY - ignore style issues
2. Only report HIGH CONFIDENCE findings (>0.7)
3. Be BRIEF - developer is actively typing
4. Line numbers are relative to the provided code
5. Skip issues that were previously dismissed by user
"""
    
    def __init__(
        self,
        context: SessionContext,
        on_finding: Callable[[StreamingFinding], None] | None = None,
    ):
        self.context = context
        self.on_finding = on_finding
        self.state = SessionState.IDLE
        self._findings: list[StreamingFinding] = []
        self._tokens_used = 0
        self._start_time: float = 0
        self._cancel_event = asyncio.Event()
        self._client: Any = None  # Copilot SDK client
    
    async def start(self) -> AsyncIterator[StreamingFinding]:
        """Start streaming review and yield findings as they arrive."""
        self.state = SessionState.PREPARING
        self._start_time = time.time()
        self._findings.clear()
        
        try:
            # Build the prompt
            prompt = self._build_prompt()
            
            self.state = SessionState.STREAMING
            
            # Simulate streaming response (replace with actual Copilot SDK call)
            # In production, this would use:
            # async for chunk in self._client.stream_completion(prompt):
            #     ...
            
            response_text = ""
            async for chunk in self._mock_stream_response(prompt):
                if self._cancel_event.is_set():
                    self.state = SessionState.CANCELLED
                    return
                
                response_text += chunk
                self._tokens_used += 1
                
                # Try to parse partial JSON for early findings
                findings = self._parse_partial_response(response_text)
                for finding in findings:
                    if finding.id not in [f.id for f in self._findings]:
                        self._findings.append(finding)
                        if self.on_finding:
                            self.on_finding(finding)
                        yield finding
            
            # Parse final response
            final_findings = self._parse_final_response(response_text)
            for finding in final_findings:
                if finding.id not in [f.id for f in self._findings]:
                    finding.partial = False
                    self._findings.append(finding)
                    if self.on_finding:
                        self.on_finding(finding)
                    yield finding
            
            self.state = SessionState.COMPLETED
            
        except Exception as e:
            logger.error("Copilot session failed", error=str(e), session_id=self.context.session_id)
            self.state = SessionState.ERROR
            raise
    
    def cancel(self) -> None:
        """Cancel the streaming session."""
        self._cancel_event.set()
    
    def get_result(self) -> SessionResult:
        """Get the session result."""
        return SessionResult(
            session_id=self.context.session_id,
            state=self.state,
            findings=self._findings,
            tokens_used=self._tokens_used,
            latency_ms=(time.time() - self._start_time) * 1000 if self._start_time else 0,
        )
    
    def _build_prompt(self) -> str:
        """Build the review prompt with full context."""
        parts = [
            f"Language: {self.context.language}",
            f"File: {self.context.file_path}",
            "",
        ]
        
        # Imports
        if self.context.imports:
            parts.append("Imports:")
            for imp in self.context.imports[:8]:
                parts.append(f"  {imp}")
            parts.append("")
        
        # Class context
        if self.context.class_context:
            parts.append("Class context:")
            parts.append(self.context.class_context[:400])
            parts.append("")
        
        # Function signature
        if self.context.function_signature:
            parts.append(f"Function: {self.context.function_signature}")
            parts.append("")
        
        # Surrounding context
        if self.context.surrounding_context:
            parts.append("Surrounding code:")
            parts.append("```")
            parts.append(self.context.surrounding_context[:800])
            parts.append("```")
            parts.append("")
        
        # Code to review
        parts.append("Code to review:")
        parts.append("```")
        parts.append(self.context.code)
        parts.append("```")
        
        # Previous corrections
        if self.context.previous_corrections:
            parts.append("")
            parts.append("User dismissed these findings (skip similar):")
            for correction in self.context.previous_corrections[-3:]:
                parts.append(f"  - {correction.get('message', 'Unknown')[:60]}")
        
        return "\n".join(parts)
    
    def _parse_partial_response(self, text: str) -> list[StreamingFinding]:
        """Try to parse partial JSON for early findings."""
        findings = []
        
        # Try to find complete finding objects in partial JSON
        # This is a simplified parser - production would use incremental JSON parsing
        try:
            # Look for complete finding objects
            import re
            finding_pattern = r'\{[^{}]*"category"[^{}]*"message"[^{}]*\}'
            matches = re.findall(finding_pattern, text)
            
            for i, match in enumerate(matches):
                try:
                    data = json.loads(match)
                    finding = StreamingFinding(
                        id=f"{self.context.session_id}:partial:{i}",
                        partial=True,
                        category=data.get("category", "bug"),
                        priority=data.get("priority", "medium"),
                        message=data.get("message", ""),
                        line_start=data.get("line_start", 1),
                        line_end=data.get("line_end", data.get("line_start", 1)),
                        suggestion=data.get("suggestion"),
                        fix_code=data.get("fix_code"),
                        confidence=data.get("confidence", 0.8),
                        tokens_so_far=self._tokens_used,
                    )
                    findings.append(finding)
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
        
        return findings
    
    def _parse_final_response(self, text: str) -> list[StreamingFinding]:
        """Parse the final complete response."""
        findings = []
        
        try:
            # Try to parse full JSON
            data = json.loads(text)
            raw_findings = data.get("findings", [])
            
            for i, raw in enumerate(raw_findings):
                finding = StreamingFinding(
                    id=f"{self.context.session_id}:{i}",
                    partial=False,
                    category=raw.get("category", "bug"),
                    priority=raw.get("priority", "medium"),
                    message=raw.get("message", "Unknown issue"),
                    line_start=raw.get("line_start", 1),
                    line_end=raw.get("line_end", raw.get("line_start", 1)),
                    suggestion=raw.get("suggestion"),
                    fix_code=raw.get("fix_code"),
                    confidence=raw.get("confidence", 0.8),
                    tokens_so_far=self._tokens_used,
                )
                findings.append(finding)
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse Copilot response", response=text[:200])
        
        return findings
    
    async def _mock_stream_response(self, prompt: str) -> AsyncIterator[str]:
        """Mock streaming response for testing.
        
        In production, replace with actual Copilot SDK streaming call.
        """
        # Simulate a realistic response
        mock_response = json.dumps({
            "findings": [],
            "summary": "Code looks good - no critical issues found."
        })
        
        # Simulate streaming with small delays
        chunk_size = 20
        for i in range(0, len(mock_response), chunk_size):
            await asyncio.sleep(0.01)  # Simulate network latency
            yield mock_response[i:i + chunk_size]


class RealTimeCopilotReviewer:
    """Main class for real-time Copilot review integration.
    
    Features:
    - Session pooling for low latency
    - Smart throttling
    - User correction feedback loop
    - Streaming findings
    """
    
    def __init__(self):
        self.session_pool = CopilotSessionPool()
        self._active_sessions: dict[str, CopilotReviewSession] = {}
        self._user_corrections: dict[str, list[dict[str, Any]]] = defaultdict(list)  # file_path -> corrections
        self._finding_history: dict[str, list[str]] = defaultdict(list)  # file_path -> finding_ids
    
    async def review(
        self,
        code: str,
        file_path: str,
        language: str,
        surrounding_context: str = "",
        imports: list[str] | None = None,
        function_signature: str | None = None,
        class_context: str | None = None,
        on_finding: Callable[[StreamingFinding], None] | None = None,
    ) -> SessionResult:
        """Perform real-time review with streaming.
        
        Args:
            code: Code to review
            file_path: Path to the file
            language: Programming language
            surrounding_context: Code around the target
            imports: List of imports
            function_signature: Function being analyzed
            class_context: Class containing the code
            on_finding: Callback for each finding
            
        Returns:
            SessionResult with all findings
        """
        # Cancel any existing session for this file
        existing_key = f"review:{file_path}"
        if existing_key in self._active_sessions:
            self._active_sessions[existing_key].cancel()
            await asyncio.sleep(0.05)  # Brief wait for cancellation
        
        # Acquire session from pool
        session_id = await self.session_pool.acquire(language)
        
        # Create context
        context = SessionContext(
            session_id=session_id,
            file_path=file_path,
            language=language,
            code=code,
            surrounding_context=surrounding_context,
            imports=imports or [],
            function_signature=function_signature,
            class_context=class_context,
            previous_corrections=self._user_corrections.get(file_path, [])[-5:],
        )
        
        # Create session
        session = CopilotReviewSession(context, on_finding)
        self._active_sessions[existing_key] = session
        
        try:
            # Stream findings
            findings = []
            async for finding in session.start():
                findings.append(finding)
                
                # Track finding for history
                self._finding_history[file_path].append(finding.id)
            
            result = session.get_result()
            
            # Release session back to pool if successful
            if result.state == SessionState.COMPLETED:
                await self.session_pool.release(session_id, language)
            else:
                await self.session_pool.invalidate(session_id)
            
            return result
            
        except Exception as e:
            await self.session_pool.invalidate(session_id)
            return SessionResult(
                session_id=session_id,
                state=SessionState.ERROR,
                error=str(e),
            )
        finally:
            self._active_sessions.pop(existing_key, None)
    
    def record_correction(
        self,
        file_path: str,
        finding_id: str,
        action: str,
        message: str,
        reason: str | None = None,
    ) -> None:
        """Record a user correction for learning.
        
        Args:
            file_path: File where finding was
            finding_id: ID of the finding
            action: "accepted", "dismissed", "modified"
            message: Original finding message
            reason: User's reason (optional)
        """
        correction = {
            "finding_id": finding_id,
            "action": action,
            "message": message,
            "reason": reason,
            "timestamp": time.time(),
        }
        self._user_corrections[file_path].append(correction)
        
        # Keep only recent corrections per file
        if len(self._user_corrections[file_path]) > 20:
            self._user_corrections[file_path] = self._user_corrections[file_path][-20:]
        
        logger.info(
            "User correction recorded",
            file_path=file_path,
            action=action,
            finding_id=finding_id,
        )
    
    def cancel_review(self, file_path: str) -> None:
        """Cancel any active review for a file."""
        key = f"review:{file_path}"
        if key in self._active_sessions:
            self._active_sessions[key].cancel()
    
    def get_statistics(self) -> dict[str, Any]:
        """Get reviewer statistics."""
        return {
            "pool_stats": self.session_pool.get_stats(),
            "active_sessions": len(self._active_sessions),
            "files_with_corrections": len(self._user_corrections),
            "total_corrections": sum(len(c) for c in self._user_corrections.values()),
        }


# Global instance for the extension
copilot_reviewer = RealTimeCopilotReviewer()
