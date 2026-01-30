"""Verification Debugger - Interactive Z3 proof visualization and playback."""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog
from z3 import (
    And,
    ArithRef,
    BoolRef,
    Int,
    Not,
    Or,
    Solver,
    is_and,
    is_not,
    is_or,
    sat,
    unsat,
)

logger = structlog.get_logger()


class StepType(str, Enum):
    """Type of verification step."""

    PARSE = "parse"
    ADD_CONSTRAINT = "add_constraint"
    SIMPLIFY = "simplify"
    CHECK = "check"
    GET_MODEL = "get_model"
    GET_PROOF = "get_proof"


class StepStatus(str, Enum):
    """Status of a verification step."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class VerificationStep:
    """A single step in the verification process."""

    id: int
    step_type: StepType
    description: str
    formula: str | None = None
    result: str | None = None
    status: StepStatus = StepStatus.PENDING
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: dict[str, Any] = field(default_factory=dict)
    children: list["VerificationStep"] = field(default_factory=list)


@dataclass
class VerificationTrace:
    """Complete trace of a verification run."""

    id: str
    name: str
    description: str | None = None
    steps: list[VerificationStep] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    variables: dict[str, str] = field(default_factory=dict)
    result: str | None = None  # "sat", "unsat", "unknown"
    counterexample: dict[str, Any] | None = None
    total_duration_ms: float = 0.0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [self._step_to_dict(s) for s in self.steps],
            "constraints": self.constraints,
            "variables": self.variables,
            "result": self.result,
            "counterexample": self.counterexample,
            "total_duration_ms": self.total_duration_ms,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }

    def _step_to_dict(self, step: VerificationStep) -> dict[str, Any]:
        """Convert a step to dictionary."""
        return {
            "id": step.id,
            "step_type": step.step_type.value,
            "description": step.description,
            "formula": step.formula,
            "result": step.result,
            "status": step.status.value,
            "duration_ms": step.duration_ms,
            "timestamp": step.timestamp.isoformat(),
            "details": step.details,
            "children": [self._step_to_dict(c) for c in step.children],
        }


class VerificationDebugger:
    """
    Interactive debugger for Z3 verification.

    Provides step-by-step tracing and visualization of the SMT solving process.
    """

    def __init__(self, timeout_ms: int = 60000) -> None:
        """Initialize the debugger."""
        self.timeout_ms = timeout_ms
        self._step_counter = 0
        self._current_trace: VerificationTrace | None = None

    def _next_step_id(self) -> int:
        """Get next step ID."""
        self._step_counter += 1
        return self._step_counter

    def _add_step(
        self,
        step_type: StepType,
        description: str,
        formula: str | None = None,
    ) -> VerificationStep:
        """Add a step to the current trace."""
        step = VerificationStep(
            id=self._next_step_id(),
            step_type=step_type,
            description=description,
            formula=formula,
        )
        if self._current_trace:
            self._current_trace.steps.append(step)
        return step

    def _complete_step(
        self,
        step: VerificationStep,
        status: StepStatus,
        result: str | None = None,
        duration_ms: float = 0.0,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Mark a step as complete."""
        step.status = status
        step.result = result
        step.duration_ms = duration_ms
        if details:
            step.details.update(details)

    def verify_with_trace(
        self,
        formula: str,
        name: str = "Verification",
        description: str | None = None,
    ) -> VerificationTrace:
        """
        Verify a formula with full step-by-step tracing.

        Args:
            formula: SMT-LIB formatted formula
            name: Name for this verification
            description: Optional description

        Returns:
            VerificationTrace with all steps and results
        """
        import uuid

        self._step_counter = 0
        self._current_trace = VerificationTrace(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            started_at=datetime.utcnow(),
        )

        total_start = time.time()

        try:
            # Step 1: Parse the formula
            parse_step = self._add_step(
                StepType.PARSE,
                "Parsing SMT-LIB formula",
                formula[:500] + "..." if len(formula) > 500 else formula,
            )
            parse_start = time.time()

            solver = Solver()
            solver.set("timeout", self.timeout_ms)

            try:
                solver.from_string(formula)
                self._complete_step(
                    parse_step,
                    StepStatus.SUCCESS,
                    "Formula parsed successfully",
                    (time.time() - parse_start) * 1000,
                )
            except Exception as e:
                self._complete_step(
                    parse_step,
                    StepStatus.FAILED,
                    f"Parse error: {str(e)}",
                    (time.time() - parse_start) * 1000,
                )
                self._current_trace.result = "error"
                self._current_trace.completed_at = datetime.utcnow()
                self._current_trace.total_duration_ms = (time.time() - total_start) * 1000
                return self._current_trace

            # Step 2: Extract and record constraints
            constraints_step = self._add_step(
                StepType.ADD_CONSTRAINT,
                "Extracting constraints from formula",
            )
            constraints_start = time.time()

            assertions = solver.assertions()
            self._current_trace.constraints = [str(a) for a in assertions]

            self._complete_step(
                constraints_step,
                StepStatus.SUCCESS,
                f"Found {len(assertions)} constraints",
                (time.time() - constraints_start) * 1000,
                {"constraint_count": len(assertions)},
            )

            # Add individual constraint steps
            for i, assertion in enumerate(assertions):
                constraint_step = self._add_step(
                    StepType.ADD_CONSTRAINT,
                    f"Constraint {i + 1}",
                    str(assertion),
                )
                self._complete_step(
                    constraint_step,
                    StepStatus.SUCCESS,
                    "Added to solver",
                    0.1,
                )

            # Step 3: Check satisfiability
            check_step = self._add_step(
                StepType.CHECK,
                "Checking satisfiability",
            )
            check_start = time.time()

            result = solver.check()
            check_duration = (time.time() - check_start) * 1000

            if result == sat:
                self._complete_step(
                    check_step,
                    StepStatus.SUCCESS,
                    "SATISFIABLE - counterexample exists",
                    check_duration,
                    {"solver_result": "sat"},
                )
                self._current_trace.result = "sat"

                # Step 4: Get model (counterexample)
                model_step = self._add_step(
                    StepType.GET_MODEL,
                    "Extracting counterexample model",
                )
                model_start = time.time()

                model = solver.model()
                counterexample = {}
                for decl in model.decls():
                    var_name = str(decl)
                    var_value = str(model[decl])
                    counterexample[var_name] = var_value
                    self._current_trace.variables[var_name] = var_value

                self._current_trace.counterexample = counterexample
                self._complete_step(
                    model_step,
                    StepStatus.SUCCESS,
                    f"Found {len(counterexample)} variable assignments",
                    (time.time() - model_start) * 1000,
                    {"variables": counterexample},
                )

            elif result == unsat:
                self._complete_step(
                    check_step,
                    StepStatus.SUCCESS,
                    "UNSATISFIABLE - property verified",
                    check_duration,
                    {"solver_result": "unsat"},
                )
                self._current_trace.result = "unsat"

                # Step 4: Get proof info
                proof_step = self._add_step(
                    StepType.GET_PROOF,
                    "Verification complete - no counterexample exists",
                )
                self._complete_step(
                    proof_step,
                    StepStatus.SUCCESS,
                    "Property holds for all inputs",
                    0.1,
                )

            else:
                self._complete_step(
                    check_step,
                    StepStatus.FAILED,
                    "UNKNOWN - solver timeout or resource limit",
                    check_duration,
                    {"solver_result": "unknown"},
                )
                self._current_trace.result = "unknown"

        except Exception as e:
            logger.error("Verification error", error=str(e))
            self._current_trace.result = "error"
            self._current_trace.metadata["error"] = str(e)

        self._current_trace.completed_at = datetime.utcnow()
        self._current_trace.total_duration_ms = (time.time() - total_start) * 1000

        return self._current_trace

    def explain_result(self, trace: VerificationTrace) -> dict[str, Any]:
        """
        Generate a human-readable explanation of the verification result.

        Args:
            trace: The verification trace to explain

        Returns:
            Dictionary with explanation components
        """
        explanation = {
            "summary": "",
            "meaning": "",
            "evidence": [],
            "recommendations": [],
        }

        if trace.result == "sat":
            explanation["summary"] = "⚠️ Potential bug found"
            explanation["meaning"] = (
                "The solver found values that violate the property being checked. "
                "This means there exist inputs that can cause the issue."
            )

            if trace.counterexample:
                explanation["evidence"].append(
                    "Counterexample values that trigger the issue:"
                )
                for var, value in trace.counterexample.items():
                    explanation["evidence"].append(f"  • {var} = {value}")

            explanation["recommendations"] = [
                "Review the code path with these input values",
                "Add input validation or bounds checking",
                "Consider the edge cases revealed by the counterexample",
            ]

        elif trace.result == "unsat":
            explanation["summary"] = "✅ Property verified"
            explanation["meaning"] = (
                "The solver mathematically proved that no inputs can violate "
                "the property being checked. The code is correct with respect to this property."
            )
            explanation["evidence"].append(
                f"Verified in {trace.total_duration_ms:.2f}ms"
            )
            explanation["evidence"].append(
                f"Checked {len(trace.constraints)} constraints"
            )
            explanation["recommendations"] = [
                "No action required for this property",
                "Consider adding more verification checks",
            ]

        else:
            explanation["summary"] = "❓ Verification inconclusive"
            explanation["meaning"] = (
                "The solver could not determine if the property holds or not, "
                "possibly due to timeout or complexity."
            )
            explanation["recommendations"] = [
                "Try increasing the timeout",
                "Simplify the verification condition",
                "Break down into smaller checks",
            ]

        return explanation

    def generate_visualization_data(self, trace: VerificationTrace) -> dict[str, Any]:
        """
        Generate data structure for UI visualization.

        Returns data suitable for rendering as a flowchart or tree diagram.
        """
        nodes = []
        edges = []

        # Create nodes for each step
        for step in trace.steps:
            node_type = "default"
            if step.status == StepStatus.SUCCESS:
                node_type = "success"
            elif step.status == StepStatus.FAILED:
                node_type = "error"

            nodes.append({
                "id": f"step_{step.id}",
                "label": step.description,
                "type": node_type,
                "data": {
                    "step_type": step.step_type.value,
                    "formula": step.formula,
                    "result": step.result,
                    "duration_ms": step.duration_ms,
                },
            })

        # Create edges between sequential steps
        for i in range(len(trace.steps) - 1):
            edges.append({
                "source": f"step_{trace.steps[i].id}",
                "target": f"step_{trace.steps[i + 1].id}",
            })

        # Add result node
        result_type = "success" if trace.result == "unsat" else "error" if trace.result == "sat" else "warning"
        nodes.append({
            "id": "result",
            "label": f"Result: {trace.result}",
            "type": result_type,
            "data": {
                "counterexample": trace.counterexample,
            },
        })

        if trace.steps:
            edges.append({
                "source": f"step_{trace.steps[-1].id}",
                "target": "result",
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_steps": len(trace.steps),
                "total_duration_ms": trace.total_duration_ms,
                "constraints_count": len(trace.constraints),
            },
        }


def create_interactive_session(timeout_ms: int = 60000) -> "InteractiveVerificationSession":
    """Create an interactive verification session."""
    return InteractiveVerificationSession(timeout_ms)


class InteractiveVerificationSession:
    """
    Interactive session for step-by-step verification.

    Allows users to add constraints incrementally and check at each step.
    """

    def __init__(self, timeout_ms: int = 60000) -> None:
        """Initialize the session."""
        self.timeout_ms = timeout_ms
        self.solver = Solver()
        self.solver.set("timeout", timeout_ms)
        self.constraints: list[str] = []
        self.variables: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []

    def add_variable(self, name: str, var_type: str = "int") -> Any:
        """Add a variable to the session."""
        if var_type == "int":
            var = Int(name)
        else:
            raise ValueError(f"Unsupported variable type: {var_type}")

        self.variables[name] = var
        self.history.append({
            "action": "add_variable",
            "name": name,
            "type": var_type,
        })
        return var

    def add_constraint(self, constraint: str, description: str = "") -> bool:
        """
        Add a constraint to the solver.

        Args:
            constraint: SMT-LIB formatted constraint or Python expression
            description: Human-readable description

        Returns:
            True if constraint was added successfully
        """
        try:
            # Try to parse as SMT-LIB first
            temp_solver = Solver()
            temp_solver.from_string(f"(assert {constraint})")

            # If successful, add to main solver
            self.solver.from_string(f"(assert {constraint})")
            self.constraints.append(constraint)

            self.history.append({
                "action": "add_constraint",
                "constraint": constraint,
                "description": description,
                "success": True,
            })
            return True

        except Exception as e:
            self.history.append({
                "action": "add_constraint",
                "constraint": constraint,
                "description": description,
                "success": False,
                "error": str(e),
            })
            return False

    def check(self) -> dict[str, Any]:
        """
        Check current constraints.

        Returns:
            Dictionary with result and optional model
        """
        start_time = time.time()
        result = self.solver.check()
        duration_ms = (time.time() - start_time) * 1000

        response = {
            "result": str(result),
            "duration_ms": duration_ms,
            "constraints_count": len(self.constraints),
        }

        if result == sat:
            model = self.solver.model()
            response["model"] = {
                str(d): str(model[d]) for d in model.decls()
            }

        self.history.append({
            "action": "check",
            "result": str(result),
            "duration_ms": duration_ms,
        })

        return response

    def push(self) -> None:
        """Create a backtracking point."""
        self.solver.push()
        self.history.append({"action": "push"})

    def pop(self) -> None:
        """Backtrack to previous state."""
        self.solver.pop()
        self.history.append({"action": "pop"})

    def reset(self) -> None:
        """Reset the solver to initial state."""
        self.solver.reset()
        self.constraints.clear()
        self.variables.clear()
        self.history.append({"action": "reset"})

    def get_state(self) -> dict[str, Any]:
        """Get current session state."""
        return {
            "constraints": self.constraints,
            "variables": list(self.variables.keys()),
            "history_length": len(self.history),
        }
