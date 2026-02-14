"""Real-Time Pair Programming Agent - Core module for live coding assistance.

This module provides the foundation for real-time pair programming with:
- Incremental analysis engine with sub-function change detection
- Smart debouncing that triggers on pause, not keystroke
- IDE feedback system with inline suggestions and CodeLens annotations
- Personalization engine with user preference learning and feedback loops
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

logger = structlog.get_logger()


# =============================================================================
# Enums
# =============================================================================


class AnalysisScope(str, Enum):
    """Scope of incremental analysis."""

    FUNCTION = "function"
    BLOCK = "block"
    FILE = "file"
    PROJECT = "project"


class SuggestionType(str, Enum):
    """Type of inline suggestion."""

    BUG_FIX = "bug_fix"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"


class SuggestionPriority(str, Enum):
    """Priority level for suggestions."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FeedbackAction(str, Enum):
    """User action on a suggestion."""

    ACCEPTED = "accepted"
    DISMISSED = "dismissed"
    DEFERRED = "deferred"
    NEVER_SHOW = "never_show"


_PRIORITY_ORDER = [
    SuggestionPriority.CRITICAL,
    SuggestionPriority.HIGH,
    SuggestionPriority.MEDIUM,
    SuggestionPriority.LOW,
    SuggestionPriority.INFO,
]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CodeChangeEvent:
    """Represents a code change event from the editor."""

    file_path: str
    changed_lines: list[int]
    content: str
    timestamp: float
    cursor_position: tuple[int, int]
    language: str = "python"

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "changed_lines": self.changed_lines,
            "timestamp": self.timestamp,
            "cursor_position": list(self.cursor_position),
            "language": self.language,
        }


@dataclass
class IncrementalAnalysisResult:
    """Result of an incremental analysis pass."""

    file_path: str
    changed_functions: list[str]
    findings: list[dict[str, Any]]
    analysis_time_ms: float
    cache_hit: bool
    scope: AnalysisScope

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "changed_functions": self.changed_functions,
            "findings": self.findings,
            "analysis_time_ms": self.analysis_time_ms,
            "cache_hit": self.cache_hit,
            "scope": self.scope.value,
        }


@dataclass
class InlineSuggestion:
    """A suggestion to display inline in the editor."""

    id: str
    suggestion_type: SuggestionType
    priority: SuggestionPriority
    message: str
    line: int
    column: int
    file_path: str
    fix_code: str | None = None
    explanation: str | None = None
    confidence: float = 0.8

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "suggestion_type": self.suggestion_type.value,
            "priority": self.priority.value,
            "message": self.message,
            "line": self.line,
            "column": self.column,
            "file_path": self.file_path,
            "fix_code": self.fix_code,
            "explanation": self.explanation,
            "confidence": self.confidence,
        }


@dataclass
class CodeLensAnnotation:
    """A CodeLens annotation rendered above a code line."""

    line: int
    label: str
    tooltip: str
    command: str | None = None
    verification_status: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "line": self.line,
            "label": self.label,
            "tooltip": self.tooltip,
            "command": self.command,
            "verification_status": self.verification_status,
        }


@dataclass
class UserPreferences:
    """User-specific preferences for suggestion filtering."""

    user_id: str
    suppressed_rules: list[str] = field(default_factory=list)
    min_priority: SuggestionPriority = SuggestionPriority.LOW
    preferred_fix_style: str = "minimal"
    feedback_history: list[dict] = field(default_factory=list)
    learning_rate: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "suppressed_rules": self.suppressed_rules,
            "min_priority": self.min_priority.value,
            "preferred_fix_style": self.preferred_fix_style,
            "feedback_history_count": len(self.feedback_history),
            "learning_rate": self.learning_rate,
        }


@dataclass
class DebounceConfig:
    """Configuration for smart debouncing."""

    min_delay_ms: float = 300
    max_delay_ms: float = 2000
    typing_pause_ms: float = 500
    syntax_aware: bool = True
    batch_changes: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_delay_ms": self.min_delay_ms,
            "max_delay_ms": self.max_delay_ms,
            "typing_pause_ms": self.typing_pause_ms,
            "syntax_aware": self.syntax_aware,
            "batch_changes": self.batch_changes,
        }


@dataclass
class SessionMetrics:
    """Metrics for a real-time pair programming session."""

    session_id: str
    suggestions_shown: int = 0
    suggestions_accepted: int = 0
    suggestions_dismissed: int = 0
    avg_response_time_ms: float = 0
    cache_hit_rate: float = 0
    total_analyses: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "suggestions_shown": self.suggestions_shown,
            "suggestions_accepted": self.suggestions_accepted,
            "suggestions_dismissed": self.suggestions_dismissed,
            "acceptance_rate": self._acceptance_rate,
            "avg_response_time_ms": self.avg_response_time_ms,
            "cache_hit_rate": self.cache_hit_rate,
            "total_analyses": self.total_analyses,
        }

    @property
    def _acceptance_rate(self) -> float:
        total = self.suggestions_accepted + self.suggestions_dismissed
        if total == 0:
            return 0.0
        return self.suggestions_accepted / total


# =============================================================================
# Incremental Analysis Engine
# =============================================================================


_FUNCTION_PATTERNS: dict[str, re.Pattern] = {
    "python": re.compile(r"^[ \t]*(async\s+)?def\s+(\w+)\s*\(", re.MULTILINE),
    "javascript": re.compile(
        r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()",
        re.MULTILINE,
    ),
    "typescript": re.compile(
        r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()",
        re.MULTILINE,
    ),
}

_ISSUE_PATTERNS: dict[str, list[dict[str, Any]]] = {
    "python": [
        {"pattern": re.compile(r"\bexcept\s*:"), "type": SuggestionType.BUG_FIX,
         "priority": SuggestionPriority.HIGH,
         "message": "Bare except catches SystemExit/KeyboardInterrupt",
         "fix_hint": "Use 'except Exception:' instead"},
        {"pattern": re.compile(r"==\s*None\b"), "type": SuggestionType.STYLE,
         "priority": SuggestionPriority.LOW,
         "message": "Use 'is None' for identity comparison",
         "fix_hint": "Replace '== None' with 'is None'"},
        {"pattern": re.compile(r"\beval\s*\("), "type": SuggestionType.SECURITY,
         "priority": SuggestionPriority.CRITICAL,
         "message": "Use of eval() is a security risk — arbitrary code execution",
         "fix_hint": "Use ast.literal_eval() for safe evaluation of literals"},
        {"pattern": re.compile(r"\bexec\s*\("), "type": SuggestionType.SECURITY,
         "priority": SuggestionPriority.CRITICAL,
         "message": "Use of exec() is a security risk — arbitrary code execution",
         "fix_hint": "Refactor to avoid dynamic code execution"},
        {"pattern": re.compile(r"def\s+\w+\((?:[^)]*,){7,}"), "type": SuggestionType.REFACTORING,
         "priority": SuggestionPriority.MEDIUM,
         "message": "Function has too many parameters (>7)",
         "fix_hint": "Group related parameters into a dataclass or dict"},
        {"pattern": re.compile(r"#\s*TODO\b", re.IGNORECASE), "type": SuggestionType.DOCUMENTATION,
         "priority": SuggestionPriority.INFO,
         "message": "Unresolved TODO comment found",
         "fix_hint": "Address or file an issue for the TODO item"},
        {"pattern": re.compile(r"time\.sleep\("), "type": SuggestionType.PERFORMANCE,
         "priority": SuggestionPriority.MEDIUM,
         "message": "Blocking sleep in potentially async context",
         "fix_hint": "Use asyncio.sleep() in async code"},
    ],
}


class IncrementalAnalyzer:
    """Analyzes code changes incrementally at sub-function granularity.

    Maintains a cache of previously analyzed function bodies to avoid
    redundant work when only part of a file changes.
    """

    def __init__(self) -> None:
        self._function_cache: dict[str, dict[str, str]] = defaultdict(dict)
        self._finding_cache: dict[str, list[dict[str, Any]]] = {}
        self._analysis_count: int = 0

    def analyze_change(
        self,
        event: CodeChangeEvent,
        previous_content: str | None = None,
    ) -> IncrementalAnalysisResult:
        """Analyze a code change event and return findings for affected functions."""
        start_time = time.monotonic()

        changed_functions = self._detect_changed_functions(
            event.content, event.changed_lines, event.language
        )
        scope = self._determine_scope(changed_functions, event)

        all_findings: list[dict[str, Any]] = []
        cache_hit = True

        for func_name in changed_functions:
            func_body = self._extract_function_body(
                event.content, func_name, event.language
            )
            if func_body is None:
                continue

            body_hash = hashlib.md5(func_body.encode()).hexdigest()
            cache_key = f"{event.file_path}::{func_name}"
            cached_hash = self._function_cache[event.file_path].get(func_name)

            if cached_hash == body_hash and cache_key in self._finding_cache:
                all_findings.extend(self._finding_cache[cache_key])
            else:
                cache_hit = False
                findings = self._analyze_function(func_body, func_name, event.language)
                all_findings.extend(findings)
                self._function_cache[event.file_path][func_name] = body_hash
                self._finding_cache[cache_key] = findings

        if not changed_functions:
            cache_hit = False
            all_findings = self._analyze_changed_region(
                event.content, event.changed_lines, event.language
            )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        self._analysis_count += 1

        logger.info(
            "Incremental analysis complete",
            file=event.file_path,
            functions=changed_functions,
            findings_count=len(all_findings),
            cache_hit=cache_hit,
            elapsed_ms=round(elapsed_ms, 2),
        )

        return IncrementalAnalysisResult(
            file_path=event.file_path,
            changed_functions=changed_functions,
            findings=all_findings,
            analysis_time_ms=round(elapsed_ms, 2),
            cache_hit=cache_hit,
            scope=scope,
        )

    def _detect_changed_functions(
        self, content: str, changed_lines: list[int], language: str,
    ) -> list[str]:
        """Detect which functions overlap with the changed lines."""
        pattern = _FUNCTION_PATTERNS.get(language)
        if pattern is None:
            return []

        lines = content.split("\n")
        functions: list[tuple[str, int, int]] = []

        for match in pattern.finditer(content):
            func_name = next((g for g in match.groups() if g is not None), None)
            if func_name is None or func_name in ("async",):
                continue
            start_line = content[: match.start()].count("\n") + 1
            end_line = self._find_function_end(lines, start_line - 1, language)
            functions.append((func_name, start_line, end_line))

        changed_set = set(changed_lines)
        return [
            name for name, start, end in functions
            if changed_set & set(range(start, end + 1))
        ]

    def _find_function_end(self, lines: list[str], start_idx: int, language: str) -> int:
        """Find the ending line number of a function definition."""
        if language == "python":
            if start_idx >= len(lines):
                return start_idx + 1
            def_line = lines[start_idx]
            base_indent = len(def_line) - len(def_line.lstrip())
            end_idx = start_idx + 1
            while end_idx < len(lines):
                line = lines[end_idx]
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    end_idx += 1
                    continue
                if (len(line) - len(line.lstrip())) <= base_indent:
                    break
                end_idx += 1
            return end_idx

        # Brace-delimited languages (JS/TS)
        depth = 0
        found_open = False
        for idx in range(start_idx, len(lines)):
            for char in lines[idx]:
                if char == "{":
                    depth += 1
                    found_open = True
                elif char == "}":
                    depth -= 1
                    if found_open and depth == 0:
                        return idx + 1
        return len(lines)

    def _extract_function_body(self, content: str, func_name: str, language: str) -> str | None:
        """Extract the full body of a named function."""
        pattern = _FUNCTION_PATTERNS.get(language)
        if pattern is None:
            return None

        lines = content.split("\n")
        for match in pattern.finditer(content):
            name = next((g for g in match.groups() if g is not None), None)
            if name != func_name:
                continue
            start_idx = content[: match.start()].count("\n")
            end_idx = self._find_function_end(lines, start_idx, language)
            return "\n".join(lines[start_idx:end_idx])
        return None

    def _analyze_function(self, func_body: str, func_name: str, language: str) -> list[dict[str, Any]]:
        """Run static checks on a single function body."""
        findings: list[dict[str, Any]] = []
        issue_patterns = _ISSUE_PATTERNS.get(language, [])

        for line_offset, line_text in enumerate(func_body.split("\n"), start=1):
            for rule in issue_patterns:
                if rule["pattern"].search(line_text):
                    findings.append({
                        "function": func_name,
                        "line_offset": line_offset,
                        "type": rule["type"].value,
                        "priority": rule["priority"].value,
                        "message": rule["message"],
                        "fix_hint": rule.get("fix_hint", ""),
                        "matched_text": line_text.strip(),
                    })

        complexity = self._estimate_complexity(func_body)
        if complexity > 10:
            findings.append({
                "function": func_name, "line_offset": 1,
                "type": SuggestionType.REFACTORING.value,
                "priority": SuggestionPriority.MEDIUM.value,
                "message": f"High cyclomatic complexity (~{complexity}) — consider splitting",
                "fix_hint": "Extract helper functions for complex branches",
                "matched_text": "",
            })
        return findings

    def _analyze_changed_region(
        self, content: str, changed_lines: list[int], language: str,
    ) -> list[dict[str, Any]]:
        """Analyze individual changed lines when no function scope is found."""
        findings: list[dict[str, Any]] = []
        lines = content.split("\n")
        issue_patterns = _ISSUE_PATTERNS.get(language, [])

        for line_num in changed_lines:
            if line_num < 1 or line_num > len(lines):
                continue
            line_text = lines[line_num - 1]
            for rule in issue_patterns:
                if rule["pattern"].search(line_text):
                    findings.append({
                        "function": "<file-level>", "line_offset": line_num,
                        "type": rule["type"].value, "priority": rule["priority"].value,
                        "message": rule["message"], "fix_hint": rule.get("fix_hint", ""),
                        "matched_text": line_text.strip(),
                    })
        return findings

    @staticmethod
    def _estimate_complexity(func_body: str) -> int:
        """Estimate cyclomatic complexity via branch-keyword counting."""
        keywords = re.findall(
            r"\b(?:if|elif|else|for|while|except|and|or|case)\b", func_body,
        )
        return 1 + len(keywords)

    def _determine_scope(self, changed_functions: list[str], event: CodeChangeEvent) -> AnalysisScope:
        """Determine the appropriate analysis scope."""
        if not changed_functions:
            total_lines = event.content.count("\n") + 1
            if len(event.changed_lines) / max(total_lines, 1) > 0.3:
                return AnalysisScope.FILE
            return AnalysisScope.BLOCK
        if len(changed_functions) > 5:
            return AnalysisScope.FILE
        return AnalysisScope.FUNCTION

    def invalidate_cache(self, file_path: str) -> None:
        """Invalidate all cached data for a given file."""
        self._function_cache.pop(file_path, None)
        keys_to_remove = [k for k in self._finding_cache if k.startswith(f"{file_path}::")]
        for key in keys_to_remove:
            del self._finding_cache[key]
        logger.debug("Cache invalidated", file=file_path)


# =============================================================================
# Smart Debouncing
# =============================================================================


# Patterns indicating an incomplete syntactic construct
_INCOMPLETE_SYNTAX: dict[str, list[re.Pattern]] = {
    "python": [
        re.compile(r":\s*$"), re.compile(r",\s*$"),
        re.compile(r"\\\s*$"), re.compile(r"\(\s*$"),
    ],
    "javascript": [
        re.compile(r"{\s*$"), re.compile(r"\(\s*$"),
        re.compile(r",\s*$"), re.compile(r"=>\s*$"),
    ],
    "typescript": [
        re.compile(r"{\s*$"), re.compile(r"\(\s*$"),
        re.compile(r",\s*$"), re.compile(r"=>\s*$"),
    ],
}


class SmartDebouncer:
    """Syntax-aware debouncer that triggers analysis on typing pauses.

    Delays analysis when the cursor sits on an incomplete construct
    (e.g. open parenthesis) so the user can finish their thought.
    """

    def __init__(self, config: DebounceConfig | None = None) -> None:
        self.config = config or DebounceConfig()
        self._last_event_time: float = 0.0
        self._pending_events: list[CodeChangeEvent] = []
        self._typing_velocities: list[float] = []

    def should_trigger(self, event: CodeChangeEvent) -> bool:
        """Decide whether to trigger analysis for the given event."""
        now = event.timestamp
        time_since_last = (now - self._last_event_time) * 1000 if self._last_event_time else float("inf")

        if self._last_event_time > 0 and time_since_last > 0:
            self._typing_velocities.append(1000.0 / time_since_last)
            if len(self._typing_velocities) > 20:
                self._typing_velocities = self._typing_velocities[-20:]

        self._last_event_time = now

        if self.config.batch_changes:
            self._pending_events.append(event)

        if time_since_last < self.config.min_delay_ms:
            return False

        required_delay = self._calculate_delay(event)
        if time_since_last < required_delay:
            return False

        if self.config.syntax_aware and not self._is_syntax_complete(
            event.content, event.cursor_position, event.language
        ):
            if time_since_last < self.config.max_delay_ms:
                return False

        self._pending_events.clear()
        return True

    def _is_syntax_complete(self, content: str, cursor_pos: tuple[int, int], language: str) -> bool:
        """Check if the line at the cursor looks syntactically complete."""
        lines = content.split("\n")
        line_idx = cursor_pos[0] - 1
        if line_idx < 0 or line_idx >= len(lines):
            return True

        current_line = lines[line_idx]
        for pat in _INCOMPLETE_SYNTAX.get(language, []):
            if pat.search(current_line):
                return False

        open_count = content.count("(") + content.count("[") + content.count("{")
        close_count = content.count(")") + content.count("]") + content.count("}")
        if open_count > close_count:
            return False
        return True

    def _calculate_delay(self, event: CodeChangeEvent) -> float:
        """Calculate the required pause length before triggering analysis."""
        delay = self.config.typing_pause_ms

        if len(self._typing_velocities) >= 3:
            avg_velocity = sum(self._typing_velocities[-5:]) / len(self._typing_velocities[-5:])
            if avg_velocity > 8:
                delay *= min(2.0, 1.0 + (avg_velocity - 8) / 10.0)

        if len(event.changed_lines) > 10:
            delay *= 1.3

        return min(delay, self.config.max_delay_ms)

    def reset(self) -> None:
        """Reset all debouncer state."""
        self._last_event_time = 0.0
        self._pending_events.clear()
        self._typing_velocities.clear()


# =============================================================================
# Suggestion & CodeLens Engine
# =============================================================================


class SuggestionEngine:
    """Generates and ranks inline suggestions and CodeLens annotations."""

    def __init__(self) -> None:
        self._suggestion_counter: int = 0

    def generate_suggestions(
        self, analysis: IncrementalAnalysisResult, preferences: UserPreferences,
    ) -> list[InlineSuggestion]:
        """Generate inline suggestions from an analysis result."""
        raw: list[InlineSuggestion] = []
        for finding in analysis.findings:
            self._suggestion_counter += 1
            raw.append(InlineSuggestion(
                id=f"sug-{uuid4().hex[:12]}",
                suggestion_type=self._map_suggestion_type(finding.get("type", "")),
                priority=self._map_priority(finding.get("priority", "info")),
                message=finding.get("message", ""),
                line=finding.get("line_offset", 1),
                column=0,
                file_path=analysis.file_path,
                fix_code=finding.get("fix_hint"),
                explanation=finding.get("matched_text"),
                confidence=0.85,
            ))

        filtered = self._filter_by_preferences(raw, preferences)
        ranked = self._rank_suggestions(filtered)

        logger.debug("Suggestions generated", total=len(raw), after_filter=len(filtered), file=analysis.file_path)
        return ranked

    def generate_code_lens(self, file_path: str, content: str) -> list[CodeLensAnnotation]:
        """Generate CodeLens annotations for function and class definitions."""
        annotations: list[CodeLensAnnotation] = []
        for idx, line in enumerate(content.split("\n")):
            stripped = line.strip()
            if stripped.startswith(("def ", "async def ")):
                match = re.match(r"^\s*(?:async\s+)?def\s+(\w+)", line)
                name = match.group(1) if match else "unknown"
                annotations.append(CodeLensAnnotation(
                    line=idx + 1, label=f"▶ Verify {name}",
                    tooltip=f"Run incremental verification on {name}",
                    command=f"codeverify.verifyFunction:{name}",
                ))
            elif stripped.startswith("class "):
                match = re.match(r"^\s*class\s+(\w+)", line)
                name = match.group(1) if match else "unknown"
                annotations.append(CodeLensAnnotation(
                    line=idx + 1, label=f"◆ Verify class {name}",
                    tooltip=f"Run verification on class {name}",
                    command=f"codeverify.verifyClass:{name}",
                ))
        return annotations

    def _filter_by_preferences(
        self, suggestions: list[InlineSuggestion], preferences: UserPreferences,
    ) -> list[InlineSuggestion]:
        """Remove suggestions suppressed or below the user's priority threshold."""
        min_idx = _PRIORITY_ORDER.index(preferences.min_priority)
        result: list[InlineSuggestion] = []
        for s in suggestions:
            if s.suggestion_type.value in preferences.suppressed_rules:
                continue
            try:
                s_idx = _PRIORITY_ORDER.index(s.priority)
            except ValueError:
                s_idx = len(_PRIORITY_ORDER)
            if s_idx > min_idx:
                continue
            result.append(s)
        return result

    def _rank_suggestions(self, suggestions: list[InlineSuggestion]) -> list[InlineSuggestion]:
        """Rank suggestions by priority (critical first) then confidence."""
        def sort_key(s: InlineSuggestion) -> tuple[int, float, int]:
            try:
                idx = _PRIORITY_ORDER.index(s.priority)
            except ValueError:
                idx = len(_PRIORITY_ORDER)
            return (idx, -s.confidence, s.line)
        return sorted(suggestions, key=sort_key)

    @staticmethod
    def _map_suggestion_type(raw_type: str) -> SuggestionType:
        mapping = {v.value: v for v in SuggestionType}
        return mapping.get(raw_type, SuggestionType.BUG_FIX)

    @staticmethod
    def _map_priority(raw_priority: str) -> SuggestionPriority:
        mapping = {v.value: v for v in SuggestionPriority}
        return mapping.get(raw_priority, SuggestionPriority.INFO)


# =============================================================================
# Personalization Engine
# =============================================================================


class PersonalizationEngine:
    """Learns from user feedback to personalize suggestion filtering.

    Tracks acceptance/dismissal rates per suggestion type and adjusts
    the user's preferences via exponential moving-average thresholds.
    """

    def __init__(self) -> None:
        self._user_prefs: dict[str, UserPreferences] = {}
        self._feedback_log: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._type_stats: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"accepted": 0, "dismissed": 0, "deferred": 0, "never_show": 0})
        )

    def record_feedback(self, user_id: str, suggestion_id: str, action: FeedbackAction) -> None:
        """Record user feedback on a suggestion."""
        entry = {"suggestion_id": suggestion_id, "action": action.value, "timestamp": time.time()}
        self._feedback_log[user_id].append(entry)
        prefs = self.get_preferences(user_id)
        prefs.feedback_history.append(entry)
        logger.info("Feedback recorded", user=user_id, suggestion=suggestion_id, action=action.value)

    def get_preferences(self, user_id: str) -> UserPreferences:
        """Get or create preferences for a user."""
        if user_id not in self._user_prefs:
            self._user_prefs[user_id] = UserPreferences(user_id=user_id)
        return self._user_prefs[user_id]

    def update_preferences(
        self, user_id: str, feedback_action: FeedbackAction, suggestion_type: SuggestionType,
    ) -> None:
        """Update user preferences based on a feedback action."""
        stats = self._type_stats[user_id][suggestion_type.value]
        stats[feedback_action.value] = stats.get(feedback_action.value, 0) + 1
        prefs = self.get_preferences(user_id)

        if feedback_action == FeedbackAction.NEVER_SHOW:
            if suggestion_type.value not in prefs.suppressed_rules:
                prefs.suppressed_rules.append(suggestion_type.value)
                logger.info("Rule suppressed", user=user_id, rule=suggestion_type.value)
            return

        prefs.suppressed_rules = self._calculate_rule_suppression(user_id)

        # Raise minimum priority when user dismisses too many suggestions
        total = sum(stats.values())
        if total >= 10:
            dismiss_rate = stats["dismissed"] / total
            if dismiss_rate > 0.7:
                current_idx = _PRIORITY_ORDER.index(prefs.min_priority)
                if current_idx > 0:
                    prefs.min_priority = _PRIORITY_ORDER[current_idx - 1]
                    logger.info("Min priority raised", user=user_id, new_min=prefs.min_priority.value)

    def _calculate_rule_suppression(self, user_id: str) -> list[str]:
        """Calculate which rules should be auto-suppressed for a user."""
        suppressed: list[str] = []
        for rule_type, stats in self._type_stats.get(user_id, {}).items():
            total = sum(stats.values())
            if total < 5:
                continue
            dismiss_rate = (stats["dismissed"] + stats["never_show"]) / total
            if dismiss_rate > 0.8:
                suppressed.append(rule_type)
        return suppressed


# =============================================================================
# Real-Time Pair Programming Session
# =============================================================================


class RealTimePairSession:
    """Orchestrates a real-time pair programming session.

    Ties together the incremental analyzer, smart debouncer, suggestion
    engine, and personalization engine into a single per-session object.
    """

    def __init__(self, session_id: str, user_id: str) -> None:
        self.session_id = session_id
        self.user_id = user_id
        self._analyzer = IncrementalAnalyzer()
        self._debouncer = SmartDebouncer()
        self._suggestion_engine = SuggestionEngine()
        self._personalization = PersonalizationEngine()
        self._metrics = SessionMetrics(session_id=session_id)
        self._response_times: list[float] = []
        self._cache_hits: int = 0
        self._active_suggestions: dict[str, InlineSuggestion] = {}
        self._previous_contents: dict[str, str] = {}
        self._closed: bool = False
        logger.info("Pair session started", session=session_id, user=user_id)

    def on_code_change(self, event: CodeChangeEvent) -> list[InlineSuggestion]:
        """Handle a code change event and return suggestions if triggered."""
        if self._closed:
            return []

        if not self._debouncer.should_trigger(event):
            return []

        result = self._analyzer.analyze_change(event, self._previous_contents.get(event.file_path))
        self._previous_contents[event.file_path] = event.content

        self._metrics.total_analyses += 1
        self._response_times.append(result.analysis_time_ms)
        if result.cache_hit:
            self._cache_hits += 1

        preferences = self._personalization.get_preferences(self.user_id)
        suggestions = self._suggestion_engine.generate_suggestions(result, preferences)

        for s in suggestions:
            self._active_suggestions[s.id] = s
        self._metrics.suggestions_shown += len(suggestions)
        return suggestions

    def on_feedback(self, suggestion_id: str, action: FeedbackAction) -> None:
        """Handle user feedback on a suggestion."""
        if self._closed:
            return

        self._personalization.record_feedback(self.user_id, suggestion_id, action)

        suggestion = self._active_suggestions.get(suggestion_id)
        if suggestion is not None:
            self._personalization.update_preferences(self.user_id, action, suggestion.suggestion_type)

        if action == FeedbackAction.ACCEPTED:
            self._metrics.suggestions_accepted += 1
        elif action in (FeedbackAction.DISMISSED, FeedbackAction.NEVER_SHOW):
            self._metrics.suggestions_dismissed += 1

    def get_metrics(self) -> SessionMetrics:
        """Return current session metrics."""
        if self._response_times:
            self._metrics.avg_response_time_ms = round(
                sum(self._response_times) / len(self._response_times), 2
            )
        if self._metrics.total_analyses > 0:
            self._metrics.cache_hit_rate = round(self._cache_hits / self._metrics.total_analyses, 4)
        return self._metrics

    def get_code_lens(self, file_path: str, content: str) -> list[CodeLensAnnotation]:
        """Generate CodeLens annotations for a file."""
        return self._suggestion_engine.generate_code_lens(file_path, content)

    def close(self) -> SessionMetrics:
        """Close the session and return final metrics."""
        self._closed = True
        self._debouncer.reset()
        metrics = self.get_metrics()
        logger.info(
            "Pair session closed", session=self.session_id,
            total_analyses=metrics.total_analyses,
            suggestions_shown=metrics.suggestions_shown,
            acceptance_rate=metrics._acceptance_rate,
        )
        return metrics
