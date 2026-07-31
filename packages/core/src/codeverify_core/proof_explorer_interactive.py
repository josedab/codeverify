"""Interactive Proof Explorer.

Web-based visualization of Z3 proof trees with animated constraint
propagation, shareable proof links, and embeddable widgets for PR
comments and dashboards.

Features:
- Z3 proof tree parsing and visualization data model
- Animated constraint propagation step-through
- Shareable proof URLs with embedding support
- Multiple output formats (HTML, Mermaid, JSON)
- Tiered detail levels (summary, detailed, full)
- Export to PDF-ready format
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ProofNodeType(str, Enum):
    """Type of node in a proof tree."""

    ROOT = "root"
    ASSERTION = "assertion"
    CONSTRAINT = "constraint"
    IMPLICATION = "implication"
    CONJUNCTION = "conjunction"
    DISJUNCTION = "disjunction"
    NEGATION = "negation"
    QUANTIFIER = "quantifier"
    COUNTEREXAMPLE = "counterexample"
    CONCLUSION = "conclusion"
    LEMMA = "lemma"


class ProofStatus(str, Enum):
    """Status of a proof node."""

    VERIFIED = "verified"
    REFUTED = "refuted"
    UNKNOWN = "unknown"
    IN_PROGRESS = "in_progress"
    SKIPPED = "skipped"


class DetailLevel(str, Enum):
    """Level of detail for proof rendering."""

    SUMMARY = "summary"
    DETAILED = "detailed"
    FULL = "full"


class ExportFormat(str, Enum):
    """Export formats for proof visualization."""

    HTML = "html"
    MERMAID = "mermaid"
    JSON = "json"
    DOT = "dot"
    SVG = "svg"


@dataclass
class ProofNode:
    """A node in the proof tree."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    node_type: ProofNodeType = ProofNodeType.ASSERTION
    label: str = ""
    expression: str = ""
    status: ProofStatus = ProofStatus.UNKNOWN
    children: list[ProofNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    depth: int = 0
    step_number: int = 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def subtree_size(self) -> int:
        return 1 + sum(c.subtree_size for c in self.children)


@dataclass
class ConstraintStep:
    """A step in constraint propagation animation."""

    step_number: int = 0
    node_id: str = ""
    action: str = ""
    before_state: dict[str, Any] = field(default_factory=dict)
    after_state: dict[str, Any] = field(default_factory=dict)
    variables_affected: list[str] = field(default_factory=list)
    description: str = ""
    duration_ms: int = 500


@dataclass
class ProofAnimation:
    """Animation data for constraint propagation visualization."""

    proof_id: str = ""
    total_steps: int = 0
    steps: list[ConstraintStep] = field(default_factory=list)
    initial_state: dict[str, Any] = field(default_factory=dict)
    final_state: dict[str, Any] = field(default_factory=dict)
    total_duration_ms: int = 0

    def get_step(self, step_number: int) -> ConstraintStep | None:
        for step in self.steps:
            if step.step_number == step_number:
                return step
        return None


@dataclass
class ShareableProof:
    """A shareable proof with a unique URL."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    share_token: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    proof_tree: ProofNode | None = None
    animation: ProofAnimation | None = None
    title: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    view_count: int = 0
    is_public: bool = True

    @property
    def share_url(self) -> str:
        return f"https://codeverify.dev/proof/{self.share_token}"

    @property
    def embed_html(self) -> str:
        return (
            f'<iframe src="{self.share_url}/embed" '
            f'width="100%" height="400" frameborder="0"></iframe>'
        )


@dataclass
class ProofExplorerState:
    """Current state of the proof explorer UI."""

    current_proof_id: str = ""
    selected_node_id: str | None = None
    detail_level: DetailLevel = DetailLevel.SUMMARY
    animation_step: int = 0
    is_playing: bool = False
    expanded_nodes: list[str] = field(default_factory=list)
    zoom_level: float = 1.0


class ProofTreeParser:
    """Parses Z3 output into a navigable proof tree."""

    def parse_z3_output(self, z3_output: str, check_type: str = "general") -> ProofNode:
        """Parse Z3 solver output into a proof tree."""
        root = ProofNode(
            node_type=ProofNodeType.ROOT,
            label=f"Verification: {check_type}",
            expression=check_type,
            depth=0,
            step_number=0,
        )

        lines = z3_output.strip().split("\n")
        step = 1

        for line in lines:
            line = line.strip()
            if not line or line.startswith(";"):
                continue

            node = self._parse_line(line, step, depth=1)
            if node:
                root.children.append(node)
                step += 1

        # Determine root status
        if any(c.status == ProofStatus.REFUTED for c in root.children):
            root.status = ProofStatus.REFUTED
        elif all(c.status == ProofStatus.VERIFIED for c in root.children):
            root.status = ProofStatus.VERIFIED
        else:
            root.status = ProofStatus.UNKNOWN

        return root

    def parse_constraints(self, constraints: list[str]) -> ProofNode:
        """Parse a list of constraint strings into a proof tree."""
        root = ProofNode(
            node_type=ProofNodeType.ROOT,
            label="Constraint System",
            status=ProofStatus.IN_PROGRESS,
        )

        for i, constraint in enumerate(constraints, 1):
            node = ProofNode(
                node_type=ProofNodeType.CONSTRAINT,
                label=f"C{i}",
                expression=constraint,
                status=ProofStatus.UNKNOWN,
                depth=1,
                step_number=i,
            )
            root.children.append(node)

        return root

    def _parse_line(self, line: str, step: int, depth: int) -> ProofNode | None:
        """Parse a single Z3 output line."""
        if line.startswith("(assert"):
            return ProofNode(
                node_type=ProofNodeType.ASSERTION,
                label=f"Assert #{step}",
                expression=line,
                status=ProofStatus.VERIFIED,
                depth=depth,
                step_number=step,
            )
        elif line == "sat":
            return ProofNode(
                node_type=ProofNodeType.COUNTEREXAMPLE,
                label="Satisfiable (counterexample exists)",
                expression=line,
                status=ProofStatus.REFUTED,
                depth=depth,
                step_number=step,
            )
        elif line == "unsat":
            return ProofNode(
                node_type=ProofNodeType.CONCLUSION,
                label="Unsatisfiable (property holds)",
                expression=line,
                status=ProofStatus.VERIFIED,
                depth=depth,
                step_number=step,
            )
        elif line.startswith("(define"):
            return ProofNode(
                node_type=ProofNodeType.CONSTRAINT,
                label=f"Definition #{step}",
                expression=line,
                status=ProofStatus.VERIFIED,
                depth=depth,
                step_number=step,
            )
        else:
            return ProofNode(
                node_type=ProofNodeType.LEMMA,
                label=f"Step #{step}",
                expression=line,
                status=ProofStatus.UNKNOWN,
                depth=depth,
                step_number=step,
            )


class ConstraintAnimator:
    """Creates step-by-step animation of constraint propagation."""

    def create_animation(
        self, proof_tree: ProofNode, variables: dict[str, Any] | None = None
    ) -> ProofAnimation:
        """Create an animation from a proof tree."""
        steps: list[ConstraintStep] = []
        current_state = dict(variables or {})
        initial_state = dict(current_state)
        step_num = 0

        self._walk_tree(proof_tree, steps, current_state, step_num)

        total_duration = sum(s.duration_ms for s in steps)

        return ProofAnimation(
            proof_id=proof_tree.id,
            total_steps=len(steps),
            steps=steps,
            initial_state=initial_state,
            final_state=current_state,
            total_duration_ms=total_duration,
        )

    def _walk_tree(
        self,
        node: ProofNode,
        steps: list[ConstraintStep],
        state: dict[str, Any],
        step_offset: int,
    ) -> None:
        """Recursively walk the tree to create animation steps."""
        step_num = step_offset + len(steps)
        before = dict(state)

        action = self._describe_action(node)
        affected_vars = self._extract_variables(node.expression)

        for var in affected_vars:
            if var not in state:
                state[var] = "constrained"

        step = ConstraintStep(
            step_number=step_num,
            node_id=node.id,
            action=action,
            before_state=before,
            after_state=dict(state),
            variables_affected=affected_vars,
            description=f"{node.label}: {node.expression[:80]}",
        )
        steps.append(step)

        for child in node.children:
            self._walk_tree(child, steps, state, step_offset)

    def _describe_action(self, node: ProofNode) -> str:
        actions = {
            ProofNodeType.ASSERTION: "Assert constraint",
            ProofNodeType.CONSTRAINT: "Add constraint",
            ProofNodeType.IMPLICATION: "Apply implication",
            ProofNodeType.CONCLUSION: "Draw conclusion",
            ProofNodeType.COUNTEREXAMPLE: "Found counterexample",
            ProofNodeType.ROOT: "Initialize",
        }
        return actions.get(node.node_type, "Process")

    def _extract_variables(self, expression: str) -> list[str]:
        """Extract variable names from a Z3 expression."""
        import re

        # Simple variable extraction from SMT-LIB style expressions
        tokens = re.findall(r"\b([a-zA-Z_]\w*)\b", expression)
        keywords = {
            "assert",
            "define",
            "declare",
            "Int",
            "Bool",
            "Real",
            "and",
            "or",
            "not",
            "ite",
            "let",
            "forall",
            "exists",
            "sat",
            "unsat",
            "true",
            "false",
        }
        return [t for t in tokens if t not in keywords and len(t) > 1]


class ProofRenderer:
    """Renders proof trees in various formats."""

    def render(
        self,
        proof: ProofNode,
        format: ExportFormat = ExportFormat.MERMAID,
        detail_level: DetailLevel = DetailLevel.SUMMARY,
    ) -> str:
        """Render a proof tree in the specified format."""
        if format == ExportFormat.MERMAID:
            return self._render_mermaid(proof, detail_level)
        elif format == ExportFormat.DOT:
            return self._render_dot(proof, detail_level)
        elif format == ExportFormat.JSON:
            return json.dumps(self._to_dict(proof), indent=2)
        elif format == ExportFormat.HTML:
            return self._render_html(proof, detail_level)
        return self._render_mermaid(proof, detail_level)

    def _render_mermaid(self, node: ProofNode, detail: DetailLevel) -> str:
        """Render as Mermaid diagram."""
        lines = ["graph TD"]
        self._mermaid_node(node, lines, detail)
        return "\n".join(lines)

    def _mermaid_node(self, node: ProofNode, lines: list[str], detail: DetailLevel) -> None:
        status_icon = {
            ProofStatus.VERIFIED: "✅",
            ProofStatus.REFUTED: "❌",
            ProofStatus.UNKNOWN: "❓",
            ProofStatus.IN_PROGRESS: "⏳",
        }.get(node.status, "")

        label = f"{status_icon} {node.label}"
        if detail == DetailLevel.FULL and node.expression:
            label += f"<br/>{node.expression[:40]}"

        lines.append(f'    {node.id}["{label}"]')

        for child in node.children:
            lines.append(f"    {node.id} --> {child.id}")
            if detail != DetailLevel.SUMMARY or child.depth <= 2:
                self._mermaid_node(child, lines, detail)

    def _render_dot(self, node: ProofNode, detail: DetailLevel) -> str:
        """Render as Graphviz DOT format."""
        lines = ["digraph proof {", "    rankdir=TB;", "    node [shape=box];"]
        self._dot_node(node, lines, detail)
        lines.append("}")
        return "\n".join(lines)

    def _dot_node(self, node: ProofNode, lines: list[str], detail: DetailLevel) -> None:
        color = {
            ProofStatus.VERIFIED: "green",
            ProofStatus.REFUTED: "red",
            ProofStatus.UNKNOWN: "gray",
        }.get(node.status, "white")

        label = node.label.replace('"', '\\"')
        lines.append(f'    {node.id} [label="{label}" fillcolor="{color}" style="filled"];')

        for child in node.children:
            lines.append(f"    {node.id} -> {child.id};")
            self._dot_node(child, lines, detail)

    def _render_html(self, node: ProofNode, detail: DetailLevel) -> str:
        """Render as interactive HTML."""
        tree_html = self._html_node(node, detail)
        return f"""<!DOCTYPE html>
<html>
<head>
<style>
.proof-node {{ padding: 8px; margin: 4px; border: 1px solid #ddd; border-radius: 4px; cursor: pointer; }}
.verified {{ border-color: #22c55e; background: #f0fdf4; }}
.refuted {{ border-color: #ef4444; background: #fef2f2; }}
.unknown {{ border-color: #94a3b8; background: #f8fafc; }}
.children {{ margin-left: 24px; border-left: 2px solid #e2e8f0; padding-left: 12px; }}
</style>
</head>
<body>
<div class="proof-explorer">{tree_html}</div>
</body>
</html>"""

    def _html_node(self, node: ProofNode, detail: DetailLevel) -> str:
        status_class = node.status.value
        icon = {"verified": "✅", "refuted": "❌", "unknown": "❓"}.get(node.status.value, "")

        expr_html = ""
        if detail != DetailLevel.SUMMARY and node.expression:
            expr_html = f'<div class="expression"><code>{node.expression[:100]}</code></div>'

        children_html = ""
        if node.children:
            inner = "".join(self._html_node(c, detail) for c in node.children)
            children_html = f'<div class="children">{inner}</div>'

        return (
            f'<div class="proof-node {status_class}">'
            f"<span>{icon} {node.label}</span>"
            f"{expr_html}{children_html}</div>"
        )

    def _to_dict(self, node: ProofNode) -> dict[str, Any]:
        return {
            "id": node.id,
            "type": node.node_type.value,
            "label": node.label,
            "expression": node.expression,
            "status": node.status.value,
            "depth": node.depth,
            "step": node.step_number,
            "children": [self._to_dict(c) for c in node.children],
        }


class InteractiveProofExplorer:
    """Main proof explorer for interactive visualization.

    Manages proof parsing, animation creation, rendering,
    and sharing of proof visualizations.
    """

    def __init__(self) -> None:
        self.parser = ProofTreeParser()
        self.animator = ConstraintAnimator()
        self.renderer = ProofRenderer()
        self.shared_proofs: dict[str, ShareableProof] = {}
        self.explorer_states: dict[str, ProofExplorerState] = {}

    def explore_z3_output(
        self,
        z3_output: str,
        check_type: str = "general",
        variables: dict[str, Any] | None = None,
    ) -> tuple[ProofNode, ProofAnimation]:
        """Parse Z3 output and create explorable proof with animation."""
        tree = self.parser.parse_z3_output(z3_output, check_type)
        animation = self.animator.create_animation(tree, variables)
        return tree, animation

    def explore_constraints(
        self,
        constraints: list[str],
        variables: dict[str, Any] | None = None,
    ) -> tuple[ProofNode, ProofAnimation]:
        """Create explorable proof from constraint list."""
        tree = self.parser.parse_constraints(constraints)
        animation = self.animator.create_animation(tree, variables)
        return tree, animation

    def render_proof(
        self,
        proof: ProofNode,
        format: ExportFormat = ExportFormat.MERMAID,
        detail_level: DetailLevel = DetailLevel.SUMMARY,
    ) -> str:
        """Render a proof tree in the specified format."""
        return self.renderer.render(proof, format, detail_level)

    def share_proof(
        self,
        proof: ProofNode,
        animation: ProofAnimation | None = None,
        title: str = "",
        description: str = "",
        is_public: bool = True,
    ) -> ShareableProof:
        """Create a shareable proof link."""
        shared = ShareableProof(
            proof_tree=proof,
            animation=animation,
            title=title or proof.label,
            description=description,
            is_public=is_public,
        )
        self.shared_proofs[shared.share_token] = shared
        logger.info("proof_shared", token=shared.share_token, title=title)
        return shared

    def get_shared_proof(self, token: str) -> ShareableProof | None:
        """Retrieve a shared proof by token."""
        proof = self.shared_proofs.get(token)
        if proof:
            proof.view_count += 1
        return proof

    def create_state(self, proof_id: str) -> ProofExplorerState:
        """Create a new explorer state for UI interaction."""
        state = ProofExplorerState(current_proof_id=proof_id)
        self.explorer_states[proof_id] = state
        return state


# ─── Singleton Access ──────────────────────────────────────────────────


_explorer_instance: InteractiveProofExplorer | None = None


def get_proof_explorer() -> InteractiveProofExplorer:
    """Get or create the singleton InteractiveProofExplorer."""
    global _explorer_instance
    if _explorer_instance is None:
        _explorer_instance = InteractiveProofExplorer()
    return _explorer_instance


def reset_proof_explorer() -> None:
    """Reset the singleton (for testing)."""
    global _explorer_instance
    _explorer_instance = None
