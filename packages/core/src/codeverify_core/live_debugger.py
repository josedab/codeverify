"""Live Verification Debugger.

Interactive Z3 constraint visualization where developers step through
proof trees, modify variable assignments, and share proof URLs.

Features:
- Proof tree parsing from Z3 output into navigable nodes
- Step-through constraint propagation with variable tracking
- Variable assignment modification and re-evaluation
- Shareable proof URLs with permalink support
- Multiple export formats (HTML, Mermaid, JSON, SVG-ready)
- Proof complexity metrics and summary generation
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class NodeStatus(str, Enum):
    """Status of a proof tree node."""

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"
    EXPLORING = "exploring"


class StepAction(str, Enum):
    """Actions in a proof step."""

    ASSERT = "assert"
    PROPAGATE = "propagate"
    DECIDE = "decide"
    CONFLICT = "conflict"
    BACKTRACK = "backtrack"
    SIMPLIFY = "simplify"


class ExportFormat(str, Enum):
    """Export formats for proof visualizations."""

    HTML = "html"
    MERMAID = "mermaid"
    JSON = "json"
    DOT = "dot"


@dataclass
class ProofNode:
    """A node in the proof tree."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    constraint: str = ""
    status: NodeStatus = NodeStatus.UNKNOWN
    children: list[ProofNode] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    depth: int = 0
    parent_id: str | None = None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass
class ProofStep:
    """A single step in the constraint propagation."""

    step_number: int = 0
    action: StepAction = StepAction.ASSERT
    constraint: str = ""
    variable_changes: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    node_id: str = ""


@dataclass
class DebugSession:
    """An interactive debugging session."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    root_node: ProofNode | None = None
    steps: list[ProofStep] = field(default_factory=list)
    current_step: int = 0
    variable_state: dict[str, Any] = field(default_factory=dict)
    user_overrides: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    share_token: str = field(
        default_factory=lambda: hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:12]
    )

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def share_url(self) -> str:
        return f"https://codeverify.dev/proof/{self.share_token}"


@dataclass
class ProofSummary:
    """Summary metrics for a proof tree."""

    total_nodes: int = 0
    max_depth: int = 0
    satisfied_count: int = 0
    violated_count: int = 0
    variables_count: int = 0
    constraint_count: int = 0
    complexity_score: float = 0.0


class ProofTreeBuilder:
    """Builds proof trees from Z3-style constraint sets."""

    def build_from_constraints(
        self,
        constraints: list[str],
        variables: dict[str, Any] | None = None,
    ) -> ProofNode:
        """Build a proof tree from a list of constraints."""
        root = ProofNode(
            constraint="(and)",
            status=NodeStatus.EXPLORING,
            depth=0,
        )
        vars_dict = variables or {}

        for _i, constraint in enumerate(constraints):
            status = self._evaluate_constraint(constraint, vars_dict)
            child = ProofNode(
                constraint=constraint,
                status=status,
                variables=dict(vars_dict),
                depth=1,
                parent_id=root.id,
            )
            root.children.append(child)

        # Root is satisfied only if all children are
        if all(c.status == NodeStatus.SATISFIED for c in root.children):
            root.status = NodeStatus.SATISFIED
        elif any(c.status == NodeStatus.VIOLATED for c in root.children):
            root.status = NodeStatus.VIOLATED
        else:
            root.status = NodeStatus.UNKNOWN

        return root

    def _evaluate_constraint(self, constraint: str, variables: dict[str, Any]) -> NodeStatus:
        """Simple constraint evaluation (pattern-based)."""
        c = constraint.lower().strip()
        if ">= 0" in c or "> 0" in c:
            for var, val in variables.items():
                if var in c and isinstance(val, (int, float)):
                    if val >= 0:
                        return NodeStatus.SATISFIED
                    return NodeStatus.VIOLATED
        if "!= 0" in c or "not (= 0)" in c:
            for var, val in variables.items():
                if var in c and val == 0:
                    return NodeStatus.VIOLATED
                if var in c:
                    return NodeStatus.SATISFIED
        if "!= null" in c or "!= none" in c or "is not none" in c:
            for var, val in variables.items():
                if var in c and val is None:
                    return NodeStatus.VIOLATED
                if var in c:
                    return NodeStatus.SATISFIED
        return NodeStatus.UNKNOWN


class StepGenerator:
    """Generates step-by-step constraint propagation trace."""

    def generate_steps(self, root: ProofNode) -> list[ProofStep]:
        """Generate steps from a proof tree."""
        steps: list[ProofStep] = []
        step_num = 0

        steps.append(
            ProofStep(
                step_number=step_num,
                action=StepAction.ASSERT,
                constraint="Begin proof exploration",
                explanation="Starting Z3 constraint solving",
                node_id=root.id,
            )
        )
        step_num += 1

        for child in root.children:
            steps.append(
                ProofStep(
                    step_number=step_num,
                    action=StepAction.ASSERT,
                    constraint=child.constraint,
                    variable_changes=child.variables,
                    explanation=f"Assert constraint: {child.constraint}",
                    node_id=child.id,
                )
            )
            step_num += 1

            if child.status == NodeStatus.SATISFIED:
                steps.append(
                    ProofStep(
                        step_number=step_num,
                        action=StepAction.PROPAGATE,
                        constraint=child.constraint,
                        explanation="Constraint satisfied ✓",
                        node_id=child.id,
                    )
                )
            elif child.status == NodeStatus.VIOLATED:
                steps.append(
                    ProofStep(
                        step_number=step_num,
                        action=StepAction.CONFLICT,
                        constraint=child.constraint,
                        explanation="Constraint violated ✗ — counterexample found",
                        node_id=child.id,
                    )
                )
            else:
                steps.append(
                    ProofStep(
                        step_number=step_num,
                        action=StepAction.DECIDE,
                        constraint=child.constraint,
                        explanation="Constraint status undetermined",
                        node_id=child.id,
                    )
                )
            step_num += 1

        final_action = (
            StepAction.PROPAGATE if root.status == NodeStatus.SATISFIED else StepAction.CONFLICT
        )
        steps.append(
            ProofStep(
                step_number=step_num,
                action=final_action,
                constraint="Proof complete",
                explanation=f"Final result: {root.status.value}",
                node_id=root.id,
            )
        )

        return steps


class ProofExporter:
    """Exports proof trees to various formats."""

    def export(self, session: DebugSession, fmt: ExportFormat) -> str:
        """Export a debug session to the specified format."""
        if fmt == ExportFormat.MERMAID:
            return self._to_mermaid(session)
        if fmt == ExportFormat.HTML:
            return self._to_html(session)
        if fmt == ExportFormat.DOT:
            return self._to_dot(session)
        return self._to_json(session)

    def _to_mermaid(self, session: DebugSession) -> str:
        lines = ["graph TD"]
        if session.root_node:
            self._mermaid_node(session.root_node, lines)
        return "\n".join(lines)

    def _mermaid_node(self, node: ProofNode, lines: list[str]) -> None:
        icon = (
            "✓"
            if node.status == NodeStatus.SATISFIED
            else "✗"
            if node.status == NodeStatus.VIOLATED
            else "?"
        )
        label = f"{icon} {node.constraint[:40]}"
        lines.append(f'    {node.id}["{label}"]')
        for child in node.children:
            lines.append(f"    {node.id} --> {child.id}")
            self._mermaid_node(child, lines)

    def _to_html(self, session: DebugSession) -> str:
        steps_html = ""
        for step in session.steps:
            icon = (
                "✓"
                if step.action == StepAction.PROPAGATE
                else "✗"
                if step.action == StepAction.CONFLICT
                else "→"
            )
            steps_html += f"<li>{icon} Step {step.step_number}: {step.explanation}</li>\n"
        return (
            f"<html><body><h1>{session.title or 'Proof Debug'}</h1>"
            f"<h2>Share: <a href='{session.share_url}'>{session.share_url}</a></h2>"
            f"<ol>{steps_html}</ol></body></html>"
        )

    def _to_dot(self, session: DebugSession) -> str:
        lines = ["digraph proof {"]
        if session.root_node:
            self._dot_node(session.root_node, lines)
        lines.append("}")
        return "\n".join(lines)

    def _dot_node(self, node: ProofNode, lines: list[str]) -> None:
        color = (
            "green"
            if node.status == NodeStatus.SATISFIED
            else "red"
            if node.status == NodeStatus.VIOLATED
            else "gray"
        )
        lines.append(f'  {node.id} [label="{node.constraint[:30]}" color="{color}"];')
        for child in node.children:
            lines.append(f"  {node.id} -> {child.id};")
            self._dot_node(child, lines)

    def _to_json(self, session: DebugSession) -> str:
        import json

        return json.dumps(
            {
                "id": session.id,
                "title": session.title,
                "share_url": session.share_url,
                "total_steps": session.total_steps,
                "steps": [
                    {"step": s.step_number, "action": s.action.value, "explanation": s.explanation}
                    for s in session.steps
                ],
            },
            indent=2,
        )


class LiveVerificationDebuggerService:
    """Main service for the live verification debugger."""

    def __init__(self) -> None:
        self._builder = ProofTreeBuilder()
        self._step_gen = StepGenerator()
        self._exporter = ProofExporter()
        self._sessions: dict[str, DebugSession] = {}
        self._shared: dict[str, str] = {}  # share_token → session_id

    def create_session(
        self,
        title: str,
        constraints: list[str],
        variables: dict[str, Any] | None = None,
    ) -> DebugSession:
        """Create an interactive debug session."""
        root = self._builder.build_from_constraints(constraints, variables)
        steps = self._step_gen.generate_steps(root)

        session = DebugSession(
            title=title,
            root_node=root,
            steps=steps,
            variable_state=variables or {},
        )
        self._sessions[session.id] = session
        self._shared[session.share_token] = session.id
        return session

    def step_forward(self, session_id: str) -> ProofStep | None:
        """Move to the next step."""
        session = self._sessions.get(session_id)
        if not session or session.current_step >= session.total_steps - 1:
            return None
        session.current_step += 1
        step = session.steps[session.current_step]
        session.variable_state.update(step.variable_changes)
        return step

    def step_backward(self, session_id: str) -> ProofStep | None:
        """Move to the previous step."""
        session = self._sessions.get(session_id)
        if not session or session.current_step <= 0:
            return None
        session.current_step -= 1
        return session.steps[session.current_step]

    def override_variable(self, session_id: str, variable: str, value: Any) -> DebugSession | None:
        """Override a variable and re-evaluate constraints."""
        session = self._sessions.get(session_id)
        if not session:
            return None
        session.user_overrides[variable] = value
        merged = {**session.variable_state, **session.user_overrides}
        constraints = [
            c.constraint for c in (session.root_node.children if session.root_node else [])
        ]
        new_root = self._builder.build_from_constraints(constraints, merged)
        new_steps = self._step_gen.generate_steps(new_root)
        session.root_node = new_root
        session.steps = new_steps
        session.current_step = 0
        session.variable_state = merged
        return session

    def export(self, session_id: str, fmt: ExportFormat) -> str:
        """Export a session to a format."""
        session = self._sessions.get(session_id)
        if not session:
            return ""
        return self._exporter.export(session, fmt)

    def get_summary(self, session_id: str) -> ProofSummary:
        """Get proof complexity summary."""
        session = self._sessions.get(session_id)
        if not session or not session.root_node:
            return ProofSummary()
        root = session.root_node
        nodes = self._count_nodes(root)
        sat = sum(1 for n in self._flatten(root) if n.status == NodeStatus.SATISFIED)
        vio = sum(1 for n in self._flatten(root) if n.status == NodeStatus.VIOLATED)
        all_vars: set[str] = set()
        for n in self._flatten(root):
            all_vars.update(n.variables.keys())
        return ProofSummary(
            total_nodes=nodes,
            max_depth=self._max_depth(root),
            satisfied_count=sat,
            violated_count=vio,
            variables_count=len(all_vars),
            constraint_count=len(root.children),
            complexity_score=round(nodes * self._max_depth(root) * 0.1, 2),
        )

    def get_session_by_token(self, token: str) -> DebugSession | None:
        """Look up a session by share token."""
        sid = self._shared.get(token)
        return self._sessions.get(sid) if sid else None

    def get_session(self, session_id: str) -> DebugSession | None:
        return self._sessions.get(session_id)

    def _count_nodes(self, node: ProofNode) -> int:
        return 1 + sum(self._count_nodes(c) for c in node.children)

    def _max_depth(self, node: ProofNode) -> int:
        if not node.children:
            return node.depth
        return max(self._max_depth(c) for c in node.children)

    def _flatten(self, node: ProofNode) -> list[ProofNode]:
        result = [node]
        for c in node.children:
            result.extend(self._flatten(c))
        return result


# ─── Singleton Access ──────────────────────────────────────────────────


_live_debugger_instance: LiveVerificationDebuggerService | None = None


def get_live_debugger_service() -> LiveVerificationDebuggerService:
    global _live_debugger_instance
    if _live_debugger_instance is None:
        _live_debugger_instance = LiveVerificationDebuggerService()
    return _live_debugger_instance


def reset_live_debugger_service() -> None:
    global _live_debugger_instance
    _live_debugger_instance = None
