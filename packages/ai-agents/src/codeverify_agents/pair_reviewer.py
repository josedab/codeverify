"""AI Pair Reviewer Agent - Real-time code review as developers type.

This agent provides:
- Sub-function granularity analysis with streaming feedback
- Incremental verification that catches issues before commit
- Context-aware prompts using surrounding code
- Smart throttling to verify on pause, not keystroke
- User correction feedback loop for continuous improvement
"""

import asyncio
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import structlog

from .base import AgentConfig, AgentResult, BaseAgent

logger = structlog.get_logger()


class ReviewPriority(str, Enum):
    """Priority level for review findings."""
    
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SUGGESTION = "suggestion"


class ReviewCategory(str, Enum):
    """Category of review finding."""
    
    BUG = "bug"
    SECURITY = "security"
    PERFORMANCE = "performance"
    LOGIC_ERROR = "logic_error"
    TYPE_ERROR = "type_error"
    NULL_SAFETY = "null_safety"
    RESOURCE_LEAK = "resource_leak"
    STYLE = "style"
    BEST_PRACTICE = "best_practice"


@dataclass
class CodeRegion:
    """Represents a region of code being analyzed."""
    
    start_line: int
    end_line: int
    start_col: int = 0
    end_col: int | None = None
    content: str = ""
    content_hash: str = ""
    
    def __post_init__(self) -> None:
        if not self.content_hash and self.content:
            self.content_hash = hashlib.md5(self.content.encode()).hexdigest()


@dataclass
class InlineFinding:
    """A finding to display inline in the editor."""
    
    id: str
    category: ReviewCategory
    priority: ReviewPriority
    message: str
    region: CodeRegion
    suggestion: str | None = None
    fix_code: str | None = None
    confidence: float = 0.8
    explanation: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category.value,
            "priority": self.priority.value,
            "message": self.message,
            "line_start": self.region.start_line,
            "line_end": self.region.end_line,
            "col_start": self.region.start_col,
            "col_end": self.region.end_col,
            "suggestion": self.suggestion,
            "fix_code": self.fix_code,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }


@dataclass
class ReviewContext:
    """Context for pair review analysis."""
    
    file_path: str
    language: str
    full_content: str
    change_region: CodeRegion
    surrounding_context: str = ""
    imports: list[str] = field(default_factory=list)
    function_signature: str | None = None
    class_context: str | None = None
    recent_changes: list[CodeRegion] = field(default_factory=list)
    user_corrections: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ReviewFeedback:
    """User feedback on a review finding."""
    
    finding_id: str
    action: str  # "accepted", "dismissed", "modified"
    reason: str | None = None
    correction: str | None = None
    timestamp: float = field(default_factory=time.time)


class SmartThrottler:
    """Intelligent throttling for real-time verification.
    
    Triggers verification on typing pause rather than every keystroke,
    with adaptive delays based on code complexity and user behavior.
    """
    
    def __init__(
        self,
        base_delay_ms: int = 300,
        max_delay_ms: int = 2000,
        complexity_factor: float = 1.5,
    ):
        self.base_delay_ms = base_delay_ms
        self.max_delay_ms = max_delay_ms
        self.complexity_factor = complexity_factor
        
        self._pending_tasks: dict[str, asyncio.Task] = {}
        self._last_trigger: dict[str, float] = {}
        self._typing_velocity: dict[str, list[float]] = defaultdict(list)
        self._region_complexity: dict[str, float] = {}
    
    def calculate_delay(self, region_id: str, content: str) -> float:
        """Calculate adaptive delay based on context."""
        # Base delay
        delay_ms = self.base_delay_ms
        
        # Adjust for code complexity (longer code = more delay)
        line_count = content.count("\n") + 1
        if line_count > 20:
            delay_ms *= min(2.0, 1 + (line_count - 20) / 50)
        
        # Adjust for typing velocity (fast typing = more delay)
        velocities = self._typing_velocity.get(region_id, [])
        if len(velocities) >= 3:
            avg_velocity = sum(velocities[-3:]) / 3
            if avg_velocity > 5:  # chars per second
                delay_ms *= min(1.5, 1 + (avg_velocity - 5) / 10)
        
        # Adjust for region complexity
        complexity = self._region_complexity.get(region_id, 1.0)
        delay_ms *= complexity
        
        return min(self.max_delay_ms, delay_ms) / 1000  # Convert to seconds
    
    def record_keystroke(self, region_id: str) -> None:
        """Record a keystroke for velocity calculation."""
        now = time.time()
        last = self._last_trigger.get(region_id, now)
        
        if now - last > 0:
            velocity = 1 / (now - last)
            self._typing_velocity[region_id].append(velocity)
            # Keep only recent velocities
            if len(self._typing_velocity[region_id]) > 10:
                self._typing_velocity[region_id] = self._typing_velocity[region_id][-10:]
        
        self._last_trigger[region_id] = now
    
    def set_complexity(self, region_id: str, complexity: float) -> None:
        """Set the complexity factor for a region."""
        self._region_complexity[region_id] = max(0.5, min(3.0, complexity))
    
    async def throttle(
        self,
        region_id: str,
        content: str,
        callback: Callable[[], Any],
    ) -> None:
        """Throttle a callback with adaptive delay."""
        self.record_keystroke(region_id)
        
        # Cancel existing pending task
        if region_id in self._pending_tasks:
            self._pending_tasks[region_id].cancel()
            try:
                await self._pending_tasks[region_id]
            except asyncio.CancelledError:
                pass
        
        delay = self.calculate_delay(region_id, content)
        
        async def delayed_callback():
            await asyncio.sleep(delay)
            if asyncio.iscoroutinefunction(callback):
                await callback()
            else:
                callback()
        
        self._pending_tasks[region_id] = asyncio.create_task(delayed_callback())
    
    def cancel(self, region_id: str) -> None:
        """Cancel pending verification for a region."""
        if region_id in self._pending_tasks:
            self._pending_tasks[region_id].cancel()
            del self._pending_tasks[region_id]
    
    def cancel_all(self) -> None:
        """Cancel all pending verifications."""
        for task in self._pending_tasks.values():
            task.cancel()
        self._pending_tasks.clear()


class FeedbackLearner:
    """Learns from user feedback to improve review quality.
    
    Tracks accepted/dismissed findings to:
    - Adjust confidence thresholds
    - Learn project-specific patterns
    - Reduce false positives over time
    """
    
    def __init__(self, min_samples: int = 10):
        self.min_samples = min_samples
        self._feedback_history: list[ReviewFeedback] = []
        self._pattern_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"accepted": 0, "dismissed": 0, "modified": 0}
        )
        self._category_thresholds: dict[str, float] = {}
    
    def record_feedback(self, feedback: ReviewFeedback, finding: InlineFinding) -> None:
        """Record user feedback on a finding."""
        self._feedback_history.append(feedback)
        
        # Update pattern statistics
        pattern_key = f"{finding.category.value}:{finding.message[:50]}"
        self._pattern_stats[pattern_key][feedback.action] += 1
        
        # Recalculate thresholds if we have enough data
        self._update_thresholds()
    
    def _update_thresholds(self) -> None:
        """Update confidence thresholds based on feedback."""
        for pattern_key, stats in self._pattern_stats.items():
            total = stats["accepted"] + stats["dismissed"] + stats["modified"]
            if total < self.min_samples:
                continue
            
            # Calculate acceptance rate
            acceptance_rate = (stats["accepted"] + 0.5 * stats["modified"]) / total
            
            # Adjust threshold based on acceptance rate
            category = pattern_key.split(":")[0]
            if acceptance_rate < 0.3:
                # High false positive rate - increase threshold
                self._category_thresholds[category] = min(0.95, 0.7 + (0.3 - acceptance_rate))
            elif acceptance_rate > 0.8:
                # Good acceptance rate - lower threshold
                self._category_thresholds[category] = max(0.5, 0.7 - (acceptance_rate - 0.8) * 0.5)
    
    def get_confidence_threshold(self, category: ReviewCategory) -> float:
        """Get the confidence threshold for a category."""
        return self._category_thresholds.get(category.value, 0.7)
    
    def should_show_finding(self, finding: InlineFinding) -> bool:
        """Determine if a finding should be shown based on learned patterns."""
        threshold = self.get_confidence_threshold(finding.category)
        return finding.confidence >= threshold
    
    def get_statistics(self) -> dict[str, Any]:
        """Get learning statistics."""
        total_feedback = len(self._feedback_history)
        if total_feedback == 0:
            return {"total_feedback": 0, "categories": {}}
        
        category_stats = {}
        for pattern_key, stats in self._pattern_stats.items():
            category = pattern_key.split(":")[0]
            if category not in category_stats:
                category_stats[category] = {"accepted": 0, "dismissed": 0, "modified": 0}
            for action, count in stats.items():
                category_stats[category][action] += count
        
        return {
            "total_feedback": total_feedback,
            "categories": category_stats,
            "thresholds": dict(self._category_thresholds),
        }


class PairReviewerAgent(BaseAgent):
    """AI agent for real-time pair review as developers type.
    
    Features:
    - Sub-function granularity analysis
    - Streaming feedback with <3 second latency
    - Context-aware prompts
    - Smart throttling
    - Learning from user corrections
    """
    
    SYSTEM_PROMPT = """You are an expert code reviewer performing real-time pair programming review.
Your role is to catch issues BEFORE they are committed, not after.

CRITICAL: You must respond in valid JSON format with this exact structure:
{
    "findings": [
        {
            "category": "bug|security|performance|logic_error|type_error|null_safety|resource_leak|style|best_practice",
            "priority": "critical|high|medium|low|suggestion",
            "message": "Clear, actionable description",
            "line_start": <number>,
            "line_end": <number>,
            "suggestion": "How to fix it",
            "fix_code": "Corrected code snippet (optional)",
            "confidence": <0.0-1.0>,
            "explanation": "Why this is an issue (optional)"
        }
    ],
    "summary": "Brief overall assessment",
    "verification_hints": ["hints for formal verification"]
}

GUIDELINES:
1. Focus on BUGS, SECURITY, and LOGIC ERRORS - not style nitpicks
2. High confidence findings only (>0.7) to minimize noise
3. Be specific about line numbers relative to the provided code
4. Provide actionable fix suggestions
5. Consider the surrounding context when analyzing
6. Don't repeat issues the user has previously dismissed
"""
    
    def __init__(self, config: AgentConfig | None = None):
        super().__init__(config)
        self.throttler = SmartThrottler()
        self.feedback_learner = FeedbackLearner()
        self._active_reviews: dict[str, asyncio.Task] = {}
        self._finding_cache: dict[str, list[InlineFinding]] = {}
    
    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """Perform standard analysis (non-streaming)."""
        review_context = self._build_context(code, context)
        return await self._perform_review(review_context)
    
    async def analyze_realtime(
        self,
        context: ReviewContext,
        on_finding: Callable[[InlineFinding], None] | None = None,
    ) -> AgentResult:
        """Perform real-time analysis with streaming findings.
        
        Args:
            context: Review context with code and metadata
            on_finding: Callback for each finding (for streaming)
            
        Returns:
            AgentResult with all findings
        """
        region_id = f"{context.file_path}:{context.change_region.start_line}"
        
        # Cancel any existing review for this region
        if region_id in self._active_reviews:
            self._active_reviews[region_id].cancel()
            try:
                await self._active_reviews[region_id]
            except asyncio.CancelledError:
                pass
        
        # Check cache first
        cache_key = context.change_region.content_hash
        if cache_key in self._finding_cache:
            findings = self._finding_cache[cache_key]
            if on_finding:
                for finding in findings:
                    on_finding(finding)
            return AgentResult(
                success=True,
                data={"findings": [f.to_dict() for f in findings], "from_cache": True},
            )
        
        # Perform review
        result = await self._perform_review(context)
        
        if result.success:
            findings = self._parse_findings(result.data.get("raw_response", ""), context)
            
            # Filter findings based on learned patterns
            filtered_findings = [
                f for f in findings
                if self.feedback_learner.should_show_finding(f)
            ]
            
            # Cache results
            self._finding_cache[cache_key] = filtered_findings
            
            # Stream findings via callback
            if on_finding:
                for finding in filtered_findings:
                    on_finding(finding)
            
            result.data["findings"] = [f.to_dict() for f in filtered_findings]
            result.data["filtered_count"] = len(findings) - len(filtered_findings)
        
        return result
    
    async def analyze_with_throttling(
        self,
        context: ReviewContext,
        on_finding: Callable[[InlineFinding], None] | None = None,
    ) -> None:
        """Analyze with smart throttling - triggers on typing pause."""
        region_id = f"{context.file_path}:{context.change_region.start_line}"
        
        # Calculate complexity for throttling
        complexity = self._estimate_complexity(context.change_region.content)
        self.throttler.set_complexity(region_id, complexity)
        
        async def do_analysis():
            await self.analyze_realtime(context, on_finding)
        
        await self.throttler.throttle(
            region_id,
            context.change_region.content,
            do_analysis,
        )
    
    def record_feedback(self, finding_id: str, action: str, reason: str | None = None) -> None:
        """Record user feedback on a finding."""
        # Find the original finding
        for findings in self._finding_cache.values():
            for finding in findings:
                if finding.id == finding_id:
                    feedback = ReviewFeedback(
                        finding_id=finding_id,
                        action=action,
                        reason=reason,
                    )
                    self.feedback_learner.record_feedback(feedback, finding)
                    return
    
    def _build_context(self, code: str, context: dict[str, Any]) -> ReviewContext:
        """Build ReviewContext from raw inputs."""
        return ReviewContext(
            file_path=context.get("file_path", "unknown"),
            language=context.get("language", "python"),
            full_content=context.get("full_content", code),
            change_region=CodeRegion(
                start_line=context.get("line_start", 1),
                end_line=context.get("line_end", code.count("\n") + 1),
                content=code,
            ),
            surrounding_context=context.get("surrounding_context", ""),
            imports=context.get("imports", []),
            function_signature=context.get("function_signature"),
            class_context=context.get("class_context"),
        )
    
    async def _perform_review(self, context: ReviewContext) -> AgentResult:
        """Perform the actual review analysis."""
        start_time = time.time()
        
        # Build the prompt with context
        user_prompt = self._build_prompt(context)
        
        try:
            response = await self._call_llm(
                self.SYSTEM_PROMPT,
                user_prompt,
                json_mode=True,
            )
            
            return AgentResult(
                success=True,
                data={
                    "raw_response": response["content"],
                    "review_context": {
                        "file": context.file_path,
                        "language": context.language,
                        "line_start": context.change_region.start_line,
                        "line_end": context.change_region.end_line,
                    },
                },
                tokens_used=response.get("tokens", 0),
                latency_ms=response.get("latency_ms", (time.time() - start_time) * 1000),
            )
        except Exception as e:
            logger.error("Pair review failed", error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )
    
    def _build_prompt(self, context: ReviewContext) -> str:
        """Build the review prompt with full context."""
        parts = []
        
        # File and language info
        parts.append(f"File: {context.file_path}")
        parts.append(f"Language: {context.language}")
        parts.append("")
        
        # Imports context
        if context.imports:
            parts.append("Imports:")
            for imp in context.imports[:10]:  # Limit imports
                parts.append(f"  {imp}")
            parts.append("")
        
        # Class context
        if context.class_context:
            parts.append("Class context:")
            parts.append(context.class_context[:500])  # Limit context
            parts.append("")
        
        # Function signature
        if context.function_signature:
            parts.append(f"Function signature: {context.function_signature}")
            parts.append("")
        
        # Surrounding context
        if context.surrounding_context:
            parts.append("Surrounding code:")
            parts.append("```")
            parts.append(context.surrounding_context[:1000])
            parts.append("```")
            parts.append("")
        
        # The code to review
        parts.append(f"Code to review (lines {context.change_region.start_line}-{context.change_region.end_line}):")
        parts.append("```")
        parts.append(context.change_region.content)
        parts.append("```")
        
        # User corrections context
        if context.user_corrections:
            parts.append("")
            parts.append("Previously dismissed findings (avoid repeating):")
            for correction in context.user_corrections[-5:]:  # Last 5
                parts.append(f"  - {correction.get('message', 'Unknown')}")
        
        return "\n".join(parts)
    
    def _parse_findings(self, response: str, context: ReviewContext) -> list[InlineFinding]:
        """Parse LLM response into InlineFinding objects."""
        findings = []
        
        try:
            data = json.loads(response)
            raw_findings = data.get("findings", [])
            
            for i, raw in enumerate(raw_findings):
                finding = InlineFinding(
                    id=f"{context.change_region.content_hash}:{i}",
                    category=ReviewCategory(raw.get("category", "bug")),
                    priority=ReviewPriority(raw.get("priority", "medium")),
                    message=raw.get("message", "Unknown issue"),
                    region=CodeRegion(
                        start_line=context.change_region.start_line + raw.get("line_start", 1) - 1,
                        end_line=context.change_region.start_line + raw.get("line_end", raw.get("line_start", 1)) - 1,
                        start_col=raw.get("col_start", 0),
                        end_col=raw.get("col_end"),
                    ),
                    suggestion=raw.get("suggestion"),
                    fix_code=raw.get("fix_code"),
                    confidence=raw.get("confidence", 0.8),
                    explanation=raw.get("explanation"),
                )
                findings.append(finding)
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to parse review response", error=str(e))
        
        return findings
    
    def _estimate_complexity(self, code: str) -> float:
        """Estimate code complexity for throttling decisions."""
        complexity = 1.0
        
        # Line count factor
        lines = code.count("\n") + 1
        if lines > 30:
            complexity += 0.5
        if lines > 50:
            complexity += 0.5
        
        # Nesting factor (count indentation changes)
        indent_changes = 0
        prev_indent = 0
        for line in code.split("\n"):
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                if indent != prev_indent:
                    indent_changes += 1
                prev_indent = indent
        if indent_changes > 10:
            complexity += 0.3
        
        # Control flow factor
        control_keywords = ["if", "else", "elif", "for", "while", "try", "except", "with", "match", "case"]
        for keyword in control_keywords:
            if f" {keyword} " in code or f"\n{keyword} " in code or code.startswith(f"{keyword} "):
                complexity += 0.1
        
        return min(3.0, complexity)
    
    def get_statistics(self) -> dict[str, Any]:
        """Get agent statistics."""
        return {
            "cache_size": len(self._finding_cache),
            "active_reviews": len(self._active_reviews),
            "feedback_stats": self.feedback_learner.get_statistics(),
        }
    
    def clear_cache(self) -> None:
        """Clear the finding cache."""
        self._finding_cache.clear()
    
    def shutdown(self) -> None:
        """Shutdown the agent gracefully."""
        self.throttler.cancel_all()
        for task in self._active_reviews.values():
            task.cancel()
        self._active_reviews.clear()
