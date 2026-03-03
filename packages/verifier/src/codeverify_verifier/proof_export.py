"""Proof tree and counterexample export to JSON.

Extends the Z3 verifier to emit structured proof trees, constraint graphs,
and counterexample data suitable for browser-based visualization.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    ROOT = "root"
    ASSERTION = "assertion"
    IMPLICATION = "implication"
    CONJUNCTION = "conjunction"
    DISJUNCTION = "disjunction"
    NEGATION = "negation"
    QUANTIFIER = "quantifier"
    VARIABLE = "variable"
    LITERAL = "literal"
    CONSTRAINT = "constraint"


class ProofStatus(str, Enum):
    PROVED = "proved"
    DISPROVED = "disproved"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"


@dataclass
class ProofNode:
    """A single node in the proof tree."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType = NodeType.ASSERTION
    label: str = ""
    expression: str = ""
    status: ProofStatus = ProofStatus.UNKNOWN
    children: list["ProofNode"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.node_type.value,
            "label": self.label,
            "expression": self.expression,
            "status": self.status.value,
            "children": [c.to_dict() for c in self.children],
            "metadata": self.metadata,
        }


@dataclass
class ConstraintEdge:
    """An edge in the constraint graph linking two variables."""

    source: str
    target: str
    constraint: str
    satisfied: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "constraint": self.constraint,
            "satisfied": self.satisfied,
        }


@dataclass
class ConstraintGraph:
    """A graph of variables connected by constraints."""

    variables: list[dict[str, Any]] = field(default_factory=list)
    edges: list[ConstraintEdge] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variables": self.variables,
            "edges": [e.to_dict() for e in self.edges],
        }


@dataclass
class CounterexampleValue:
    """A single variable assignment in a counterexample."""

    name: str
    value: Any
    type: str = "int"
    is_input: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.type,
            "is_input": self.is_input,
        }


@dataclass
class ProofExport:
    """Complete export of a verification result for web visualization."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    function_name: str = ""
    file_path: str = ""
    language: str = "python"
    status: ProofStatus = ProofStatus.UNKNOWN
    proof_tree: ProofNode | None = None
    constraint_graph: ConstraintGraph | None = None
    counterexamples: list[list[CounterexampleValue]] = field(default_factory=list)
    execution_trace: list[dict[str, Any]] = field(default_factory=list)
    explanation: str = ""
    solver_time_ms: float = 0.0
    exported_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "function_name": self.function_name,
            "file_path": self.file_path,
            "language": self.language,
            "status": self.status.value,
            "proof_tree": self.proof_tree.to_dict() if self.proof_tree else None,
            "constraint_graph": self.constraint_graph.to_dict() if self.constraint_graph else None,
            "counterexamples": [
                [v.to_dict() for v in ce] for ce in self.counterexamples
            ],
            "execution_trace": self.execution_trace,
            "explanation": self.explanation,
            "solver_time_ms": self.solver_time_ms,
            "exported_at": self.exported_at,
        }


class ProofExporter:
    """Builds a ProofExport from Z3 verification results."""

    def export_verification(
        self,
        function_name: str,
        file_path: str,
        language: str,
        properties: list[dict[str, Any]],
        solver_result: dict[str, Any] | None = None,
    ) -> ProofExport:
        """Build a complete proof export from verification properties."""
        tree = self._build_proof_tree(function_name, properties)
        graph = self._build_constraint_graph(properties)
        counterexamples = self._extract_counterexamples(solver_result)
        trace = self._build_execution_trace(properties, solver_result)

        status = ProofStatus.PROVED
        if counterexamples:
            status = ProofStatus.DISPROVED
        elif solver_result and solver_result.get("status") == "timeout":
            status = ProofStatus.TIMEOUT

        return ProofExport(
            function_name=function_name,
            file_path=file_path,
            language=language,
            status=status,
            proof_tree=tree,
            constraint_graph=graph,
            counterexamples=counterexamples,
            execution_trace=trace,
            solver_time_ms=solver_result.get("time_ms", 0.0) if solver_result else 0.0,
        )

    def _build_proof_tree(
        self, function_name: str, properties: list[dict[str, Any]]
    ) -> ProofNode:
        root = ProofNode(
            node_type=NodeType.ROOT,
            label=f"Verify: {function_name}",
            expression=f"verify({function_name})",
            status=ProofStatus.PROVED,
        )
        for prop in properties:
            child = ProofNode(
                node_type=NodeType.ASSERTION,
                label=prop.get("name", "property"),
                expression=prop.get("expression", ""),
                status=ProofStatus.PROVED if prop.get("holds", True) else ProofStatus.DISPROVED,
            )

            # Add sub-constraints as children
            for constraint in prop.get("constraints", []):
                sub = ProofNode(
                    node_type=NodeType.CONSTRAINT,
                    label=constraint.get("label", ""),
                    expression=constraint.get("expression", ""),
                    status=ProofStatus.PROVED if constraint.get("holds", True) else ProofStatus.DISPROVED,
                )
                child.children.append(sub)

            root.children.append(child)

        if any(c.status == ProofStatus.DISPROVED for c in root.children):
            root.status = ProofStatus.DISPROVED

        return root

    def _build_constraint_graph(
        self, properties: list[dict[str, Any]]
    ) -> ConstraintGraph:
        variables: dict[str, dict[str, Any]] = {}
        edges: list[ConstraintEdge] = []

        for prop in properties:
            for var_name in prop.get("variables", []):
                if var_name not in variables:
                    variables[var_name] = {
                        "id": var_name,
                        "name": var_name,
                        "type": prop.get("variable_types", {}).get(var_name, "unknown"),
                    }

            for constraint in prop.get("constraints", []):
                vars_in_constraint = constraint.get("variables", [])
                for i, v1 in enumerate(vars_in_constraint):
                    for v2 in vars_in_constraint[i + 1:]:
                        edges.append(ConstraintEdge(
                            source=v1,
                            target=v2,
                            constraint=constraint.get("expression", ""),
                            satisfied=constraint.get("holds", True),
                        ))

        return ConstraintGraph(
            variables=list(variables.values()),
            edges=edges,
        )

    def _extract_counterexamples(
        self, solver_result: dict[str, Any] | None
    ) -> list[list[CounterexampleValue]]:
        if not solver_result:
            return []

        raw_ces = solver_result.get("counterexamples", [])
        result: list[list[CounterexampleValue]] = []
        for ce in raw_ces:
            values = []
            for var_name, var_data in ce.items():
                if isinstance(var_data, dict):
                    values.append(CounterexampleValue(
                        name=var_name,
                        value=var_data.get("value"),
                        type=var_data.get("type", "int"),
                        is_input=var_data.get("is_input", True),
                    ))
                else:
                    values.append(CounterexampleValue(
                        name=var_name,
                        value=var_data,
                    ))
            result.append(values)
        return result

    def _build_execution_trace(
        self, properties: list[dict[str, Any]], solver_result: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        trace: list[dict[str, Any]] = []
        for i, prop in enumerate(properties):
            trace.append({
                "step": i + 1,
                "type": "check_property",
                "property": prop.get("name", f"property_{i}"),
                "expression": prop.get("expression", ""),
                "result": "holds" if prop.get("holds", True) else "violated",
            })
        return trace
