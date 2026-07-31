"""Code Evolution Timeline.

Visualizes how function behavioral contracts have evolved over time,
correlating with git history, proof status, and spec changes.

Features:
- Function evolution tracking across commits
- Proof status timeline (when proofs first passed/failed)
- Spec change correlation with behavioral drift
- Mermaid timeline diagram generation
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class EventType(str, Enum):
    CREATED = "created"
    MODIFIED = "modified"
    PROOF_PASSED = "proof_passed"
    PROOF_FAILED = "proof_failed"
    SPEC_ADDED = "spec_added"
    SPEC_CHANGED = "spec_changed"
    DRIFT_DETECTED = "drift_detected"
    BUG_FIXED = "bug_fixed"
    REFACTORED = "refactored"


@dataclass
class EvolutionEvent:
    """A single event in a function's evolution."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_type: EventType = EventType.MODIFIED
    commit_sha: str = ""
    author: str = ""
    message: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionTimeline:
    """Complete evolution timeline for a function."""

    function_name: str = ""
    file_path: str = ""
    events: list[EvolutionEvent] = field(default_factory=list)
    current_proof_status: str = "unknown"
    total_modifications: int = 0
    total_proof_passes: int = 0
    total_proof_failures: int = 0
    days_since_last_change: int = 0

    @property
    def stability_score(self) -> float:
        if not self.events:
            return 0.0
        passes = sum(1 for e in self.events if e.event_type == EventType.PROOF_PASSED)
        total = len(self.events)
        return round(passes / total, 3) if total > 0 else 0.0


@dataclass
class RepositoryEvolution:
    """Evolution data for an entire repository."""

    repo: str = ""
    timelines: list[FunctionTimeline] = field(default_factory=list)
    total_functions: int = 0
    most_modified: str = ""
    most_stable: str = ""


class TimelineBuilder:
    """Builds evolution timelines from event data."""

    def build(
        self, function_name: str, file_path: str, events: list[dict[str, Any]]
    ) -> FunctionTimeline:
        parsed_events: list[EvolutionEvent] = []
        for e in events:
            try:
                et = EventType(e.get("type", "modified"))
            except ValueError:
                et = EventType.MODIFIED
            parsed_events.append(
                EvolutionEvent(
                    event_type=et,
                    commit_sha=e.get("commit", ""),
                    author=e.get("author", ""),
                    message=e.get("message", ""),
                    details=e.get("details", {}),
                )
            )

        passes = sum(1 for e in parsed_events if e.event_type == EventType.PROOF_PASSED)
        failures = sum(1 for e in parsed_events if e.event_type == EventType.PROOF_FAILED)
        mods = sum(
            1 for e in parsed_events if e.event_type in (EventType.MODIFIED, EventType.REFACTORED)
        )

        current = (
            "passed"
            if parsed_events and parsed_events[-1].event_type == EventType.PROOF_PASSED
            else "unknown"
        )

        return FunctionTimeline(
            function_name=function_name,
            file_path=file_path,
            events=parsed_events,
            current_proof_status=current,
            total_modifications=mods,
            total_proof_passes=passes,
            total_proof_failures=failures,
        )


class TimelineRenderer:
    """Renders timelines as Mermaid diagrams."""

    def render_mermaid(self, timeline: FunctionTimeline) -> str:
        lines = ["timeline", f"    title {timeline.function_name} Evolution"]
        for event in timeline.events[-10:]:
            icon = {
                "proof_passed": "✅",
                "proof_failed": "❌",
                "drift_detected": "⚠️",
                "spec_added": "📝",
                "bug_fixed": "🔧",
                "created": "🆕",
            }.get(event.event_type.value, "📝")
            lines.append(f"    {event.commit_sha[:7] or 'event'} : {icon} {event.event_type.value}")
        return "\n".join(lines)


class CodeEvolutionService:
    """Main service for code evolution timelines."""

    def __init__(self) -> None:
        self._builder = TimelineBuilder()
        self._renderer = TimelineRenderer()
        self._timelines: dict[str, FunctionTimeline] = {}

    def record_events(
        self, function_name: str, file_path: str, events: list[dict[str, Any]]
    ) -> FunctionTimeline:
        timeline = self._builder.build(function_name, file_path, events)
        key = f"{file_path}:{function_name}"
        self._timelines[key] = timeline
        return timeline

    def get_timeline(self, function_name: str, file_path: str) -> FunctionTimeline | None:
        return self._timelines.get(f"{file_path}:{function_name}")

    def render_mermaid(self, function_name: str, file_path: str) -> str:
        timeline = self.get_timeline(function_name, file_path)
        if not timeline:
            return ""
        return self._renderer.render_mermaid(timeline)

    def get_repo_evolution(self, repo: str) -> RepositoryEvolution:
        timelines = list(self._timelines.values())
        most_mod = max(timelines, key=lambda t: t.total_modifications, default=None)
        most_stable = max(timelines, key=lambda t: t.stability_score, default=None)
        return RepositoryEvolution(
            repo=repo,
            timelines=timelines,
            total_functions=len(timelines),
            most_modified=most_mod.function_name if most_mod else "",
            most_stable=most_stable.function_name if most_stable else "",
        )


_evolution_instance: CodeEvolutionService | None = None


def get_code_evolution_service() -> CodeEvolutionService:
    global _evolution_instance
    if _evolution_instance is None:
        _evolution_instance = CodeEvolutionService()
    return _evolution_instance


def reset_code_evolution_service() -> None:
    global _evolution_instance
    _evolution_instance = None
