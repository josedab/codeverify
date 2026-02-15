"""Interactive Proof Explorer.

Serializes Z3 proof traces into structured, explorable data for
web-based visualization. Provides step-through navigation, counterexample
rendering, and natural language explanations of proof steps.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ProofStepType(str, Enum):
    """Type of a proof step."""
    DECLARE = "declare"
    ASSERT = "assert"
    CHECK = "check"
    RESULT = "result"
    SIMPLIFY = "simplify"
    SPLIT = "split"
    COUNTEREXAMPLE = "counterexample"


class ProofOutcome(str, Enum):
    """Final outcome of a proof."""
    VERIFIED = "verified"
    COUNTEREXAMPLE_FOUND = "counterexample_found"
    TIMEOUT = "timeout"
    ERROR = "error"


class VisualizationFormat(str, Enum):
    """Output format for proof visualization."""
    JSON = "json"
    MERMAID = "mermaid"
    DOT = "dot"


@dataclass
class VariableBinding:
    """A variable and its value at a proof step."""
    name: str
    type: str = "Int"
    value: str | None = None
    constraints: list[str] = field(default_factory=list)


@dataclass
class ProofStep:
    """A single step in a proof trace."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    step_number: int = 0
    step_type: ProofStepType = ProofStepType.ASSERT
    description: str = ""
    smt_expression: str = ""
    variables: list[VariableBinding] = field(default_factory=list)
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    source_line: int | None = None
    explanation: str = ""


@dataclass
class Counterexample:
    """A counterexample found during verification."""
    variables: dict[str, str] = field(default_factory=dict)
    description: str = ""
    source_function: str = ""
    source_line: int | None = None
    fix_suggestion: str = ""


@dataclass
class ProofTrace:
    """Complete proof trace for a verification."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    function_name: str = ""
    file_path: str = ""
    verification_type: str = ""
    steps: list[ProofStep] = field(default_factory=list)
    outcome: ProofOutcome = ProofOutcome.VERIFIED
    counterexample: Counterexample | None = None
    total_time_ms: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def step_count(self) -> int:
        return len(self.steps)


class ProofTraceSerializer:
    """Serializes Z3 verification into structured proof traces."""

    def create_trace(
        self,
        function_name: str,
        file_path: str,
        verification_type: str,
        smt_formula: str,
        result: dict[str, Any],
    ) -> ProofTrace:
        """Create a proof trace from a Z3 verification result."""
        trace = ProofTrace(
            function_name=function_name,
            file_path=file_path,
            verification_type=verification_type,
            total_time_ms=result.get("proof_time_ms", 0.0),
        )

        # Parse SMT formula into steps
        steps = self._parse_smt_steps(smt_formula, function_name)
        trace.steps = steps

        # Set outcome and counterexample
        if result.get("satisfiable") is True:
            trace.outcome = ProofOutcome.COUNTEREXAMPLE_FOUND
            trace.counterexample = Counterexample(
                variables=result.get("counterexample", {}),
                description=f"Found values that violate {verification_type} in {function_name}",
                source_function=function_name,
            )
        elif result.get("satisfiable") is False:
            trace.outcome = ProofOutcome.VERIFIED
        else:
            trace.outcome = ProofOutcome.TIMEOUT

        # Add result step
        result_step = ProofStep(
            step_number=len(steps),
            step_type=ProofStepType.RESULT,
            description=f"Outcome: {trace.outcome.value}",
            explanation=self._explain_outcome(trace),
        )
        trace.steps.append(result_step)

        return trace

    def _parse_smt_steps(self, smt_formula: str, function_name: str) -> list[ProofStep]:
        """Parse an SMT-LIB formula into proof steps."""
        steps: list[ProofStep] = []
        step_num = 0

        for line in smt_formula.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith(";"):
                continue

            step = ProofStep(step_number=step_num)

            if line.startswith("(declare-const"):
                step.step_type = ProofStepType.DECLARE
                parts = line.replace("(", "").replace(")", "").split()
                if len(parts) >= 3:
                    step.variables = [VariableBinding(name=parts[1], type=parts[2])]
                    step.description = f"Declare variable {parts[1]} of type {parts[2]}"
                    step.explanation = f"We introduce variable '{parts[1]}' to represent a value in function '{function_name}'."
            elif line.startswith("(assert"):
                step.step_type = ProofStepType.ASSERT
                step.smt_expression = line
                step.description = f"Assert constraint: {line}"
                step.explanation = self._explain_assertion(line)
            elif line.startswith("(check-sat"):
                step.step_type = ProofStepType.CHECK
                step.description = "Check satisfiability"
                step.explanation = "The solver now checks if all constraints can be satisfied simultaneously."
            else:
                continue

            step_num += 1
            steps.append(step)

        return steps

    def _explain_assertion(self, assertion: str) -> str:
        """Generate a natural language explanation for an SMT assertion."""
        if "not" in assertion and "null" in assertion.lower():
            return "This constraint ensures the variable is not null/None."
        if ">=" in assertion and "0" in assertion:
            return "This constraint ensures the value is non-negative (≥ 0)."
        if "<" in assertion and "len" in assertion.lower():
            return "This constraint ensures the index stays within array bounds."
        if ">=" in assertion and "<=" in assertion:
            return "This constraint bounds the value within a valid integer range to prevent overflow."
        return f"Constraint: {assertion[:80]}"

    def _explain_outcome(self, trace: ProofTrace) -> str:
        """Generate a natural language explanation of the proof outcome."""
        if trace.outcome == ProofOutcome.VERIFIED:
            return f"✅ The solver proved that {trace.verification_type} holds for function '{trace.function_name}'. No violations are possible."
        elif trace.outcome == ProofOutcome.COUNTEREXAMPLE_FOUND:
            vals = trace.counterexample.variables if trace.counterexample else {}
            return f"❌ The solver found values that violate {trace.verification_type}: {vals}. These inputs could trigger a bug."
        elif trace.outcome == ProofOutcome.TIMEOUT:
            return f"⏱️ The solver timed out while checking {trace.verification_type}. The property could not be proven or disproven."
        return "Unknown outcome."


class ProofExplorer:
    """Interactive proof exploration engine."""

    def __init__(self) -> None:
        self._serializer = ProofTraceSerializer()
        self._traces: dict[str, ProofTrace] = {}

    def add_trace(self, trace: ProofTrace) -> None:
        self._traces[trace.id] = trace

    def create_and_store_trace(
        self,
        function_name: str,
        file_path: str,
        verification_type: str,
        smt_formula: str,
        result: dict[str, Any],
    ) -> ProofTrace:
        """Create a proof trace and store it for later exploration."""
        trace = self._serializer.create_trace(
            function_name, file_path, verification_type, smt_formula, result
        )
        self._traces[trace.id] = trace
        return trace

    def get_trace(self, trace_id: str) -> ProofTrace | None:
        return self._traces.get(trace_id)

    def get_step(self, trace_id: str, step_number: int) -> ProofStep | None:
        trace = self._traces.get(trace_id)
        if trace is None:
            return None
        for step in trace.steps:
            if step.step_number == step_number:
                return step
        return None

    def list_traces(self, file_path: str | None = None) -> list[ProofTrace]:
        traces = list(self._traces.values())
        if file_path:
            traces = [t for t in traces if t.file_path == file_path]
        return traces

    def export_mermaid(self, trace_id: str) -> str | None:
        """Export a proof trace as a Mermaid diagram."""
        trace = self._traces.get(trace_id)
        if trace is None:
            return None

        lines = ["flowchart TD"]
        for step in trace.steps:
            label = step.description[:50].replace('"', "'")
            shape_l, shape_r = ("[", "]")
            if step.step_type == ProofStepType.RESULT:
                if trace.outcome == ProofOutcome.VERIFIED:
                    shape_l, shape_r = ("([", "])")
                else:
                    shape_l, shape_r = ("{{", "}}")
            elif step.step_type == ProofStepType.CHECK:
                shape_l, shape_r = ("{", "}")
            lines.append(f"    S{step.step_number}{shape_l}\"{label}\"{shape_r}")

        for i in range(len(trace.steps) - 1):
            lines.append(f"    S{i} --> S{i+1}")

        return "\n".join(lines)

    def export_json(self, trace_id: str) -> dict[str, Any] | None:
        """Export a proof trace as a JSON-serializable dict."""
        trace = self._traces.get(trace_id)
        if trace is None:
            return None
        return {
            "id": trace.id,
            "function": trace.function_name,
            "file": trace.file_path,
            "type": trace.verification_type,
            "outcome": trace.outcome.value,
            "time_ms": trace.total_time_ms,
            "step_count": trace.step_count,
            "steps": [
                {
                    "number": s.step_number,
                    "type": s.step_type.value,
                    "description": s.description,
                    "explanation": s.explanation,
                    "smt": s.smt_expression,
                    "variables": [
                        {"name": v.name, "type": v.type, "value": v.value}
                        for v in s.variables
                    ],
                }
                for s in trace.steps
            ],
            "counterexample": (
                {
                    "variables": trace.counterexample.variables,
                    "description": trace.counterexample.description,
                    "fix_suggestion": trace.counterexample.fix_suggestion,
                }
                if trace.counterexample
                else None
            ),
        }


# Singleton
_proof_explorer: ProofExplorer | None = None


def get_proof_explorer() -> ProofExplorer:
    global _proof_explorer
    if _proof_explorer is None:
        _proof_explorer = ProofExplorer()
    return _proof_explorer


def reset_proof_explorer() -> None:
    global _proof_explorer
    _proof_explorer = None
