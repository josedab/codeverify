"""Verification Debugger - Interactive stepping through Z3 proofs.

This module provides an interactive debugging experience for formal verification,
allowing developers to understand why verification succeeds or fails, step through
proofs, and visualize constraint solving.

Features:
- Step-by-step proof exploration
- Counterexample visualization
- Constraint dependency graphs
- Interactive what-if analysis
- Proof explanation in natural language
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from .base import AgentConfig, AgentResult, BaseAgent

logger = structlog.get_logger()


# =============================================================================
# Enums and Data Classes
# =============================================================================


class VerificationStatus(str, Enum):
    """Status of verification."""

    SAT = "satisfiable"
    UNSAT = "unsatisfiable"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    ERROR = "error"


class ConstraintType(str, Enum):
    """Types of constraints."""

    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    INVARIANT = "invariant"
    ASSERTION = "assertion"
    ASSUMPTION = "assumption"
    TYPE_CONSTRAINT = "type_constraint"


class ProofStepType(str, Enum):
    """Types of proof steps."""

    ASSUMPTION = "assumption"
    DERIVATION = "derivation"
    CASE_SPLIT = "case_split"
    SIMPLIFICATION = "simplification"
    CONTRADICTION = "contradiction"
    QED = "qed"


@dataclass
class Constraint:
    """A single constraint in the verification."""

    id: str
    constraint_type: ConstraintType
    expression: str
    z3_expr: str
    description: str

    # Source information
    source_file: str | None = None
    source_line: int | None = None

    # Dependencies
    depends_on: list[str] = field(default_factory=list)

    # Evaluation status
    is_active: bool = True
    current_value: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "constraint_type": self.constraint_type.value,
            "expression": self.expression,
            "z3_expr": self.z3_expr,
            "description": self.description,
            "source_file": self.source_file,
            "source_line": self.source_line,
            "depends_on": self.depends_on,
            "is_active": self.is_active,
            "current_value": str(self.current_value) if self.current_value is not None else None,
        }


@dataclass
class Variable:
    """A variable in the verification."""

    name: str
    var_type: str
    z3_sort: str

    # Current state
    current_value: Any = None
    is_free: bool = True

    # Constraints involving this variable
    involved_constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "var_type": self.var_type,
            "z3_sort": self.z3_sort,
            "current_value": str(self.current_value) if self.current_value is not None else None,
            "is_free": self.is_free,
            "involved_constraints": self.involved_constraints,
        }


@dataclass
class ProofStep:
    """A single step in the proof."""

    step_id: str
    step_type: ProofStepType
    description: str

    # What this step does
    applied_rule: str | None = None
    input_facts: list[str] = field(default_factory=list)
    output_fact: str | None = None

    # State after this step
    active_constraints: list[str] = field(default_factory=list)
    variable_bindings: dict[str, Any] = field(default_factory=dict)

    # For case splits
    case_condition: str | None = None
    sub_proofs: list["ProofStep"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "description": self.description,
            "applied_rule": self.applied_rule,
            "input_facts": self.input_facts,
            "output_fact": self.output_fact,
            "active_constraints": self.active_constraints,
            "variable_bindings": {k: str(v) for k, v in self.variable_bindings.items()},
            "case_condition": self.case_condition,
            "sub_proofs": [sp.to_dict() for sp in self.sub_proofs],
        }


@dataclass
class Counterexample:
    """A counterexample that violates constraints."""

    id: str
    variable_assignments: dict[str, Any]
    violated_constraints: list[str]
    explanation: str

    # Minimized version
    is_minimal: bool = False
    essential_variables: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "variable_assignments": {k: str(v) for k, v in self.variable_assignments.items()},
            "violated_constraints": self.violated_constraints,
            "explanation": self.explanation,
            "is_minimal": self.is_minimal,
            "essential_variables": self.essential_variables,
        }


@dataclass
class DebugSession:
    """An interactive debugging session."""

    session_id: str
    created_at: float

    # Problem definition
    constraints: dict[str, Constraint] = field(default_factory=dict)
    variables: dict[str, Variable] = field(default_factory=dict)

    # Verification state
    status: VerificationStatus = VerificationStatus.UNKNOWN
    proof_steps: list[ProofStep] = field(default_factory=list)
    counterexamples: list[Counterexample] = field(default_factory=list)

    # Current position in proof
    current_step: int = 0

    # User modifications
    disabled_constraints: list[str] = field(default_factory=list)
    user_assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "constraints": {k: v.to_dict() for k, v in self.constraints.items()},
            "variables": {k: v.to_dict() for k, v in self.variables.items()},
            "status": self.status.value,
            "proof_steps": [ps.to_dict() for ps in self.proof_steps],
            "counterexamples": [ce.to_dict() for ce in self.counterexamples],
            "current_step": self.current_step,
            "disabled_constraints": self.disabled_constraints,
            "user_assumptions": self.user_assumptions,
        }


@dataclass
class DebugResult:
    """Result of a debug operation."""

    success: bool
    session: DebugSession
    message: str
    execution_time_ms: float

    # Analysis results
    unsatisfiable_core: list[str] | None = None
    suggested_fixes: list[str] = field(default_factory=list)
    natural_language_explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "session": self.session.to_dict(),
            "message": self.message,
            "execution_time_ms": self.execution_time_ms,
            "unsatisfiable_core": self.unsatisfiable_core,
            "suggested_fixes": self.suggested_fixes,
            "natural_language_explanation": self.natural_language_explanation,
        }


# =============================================================================
# Verification Debugger Agent
# =============================================================================


class VerificationDebugger(BaseAgent):
    """Interactive debugger for formal verification.

    Provides step-by-step proof exploration, counterexample analysis,
    and natural language explanations of verification results.

    Example usage:
        debugger = VerificationDebugger()
        session = debugger.create_session(constraints, variables)
        result = await debugger.verify(session)
        debugger.step_forward(session)
        explanation = debugger.explain_current_state(session)
    """

    EXPLANATION_PROMPT = """You are an expert in formal verification explaining proof results to developers.

Given a verification result, explain:
1. What was being verified
2. Why it succeeded or failed
3. What the counterexample (if any) means in practical terms
4. How to fix the issue

Be concise and use everyday programming terms, not formal logic jargon.

Verification Context:
{context}

Provide a clear, helpful explanation."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize the verification debugger."""
        super().__init__(config)
        self._sessions: dict[str, DebugSession] = {}
        self._z3_available = self._check_z3()

    def _check_z3(self) -> bool:
        """Check if Z3 is available."""
        try:
            import z3
            return True
        except ImportError:
            logger.warning("Z3 not available, some features will be limited")
            return False

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """Analyze verification for debugging."""
        start_time = time.time()

        try:
            # Extract constraints from context
            constraints = context.get("constraints", [])
            variables = context.get("variables", {})

            # Create session
            session = self.create_session(constraints, variables)

            # Run verification
            result = await self.verify(session)

            return AgentResult(
                success=True,
                data=result.to_dict(),
                latency_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            logger.error("Verification debug failed", error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    def create_session(
        self,
        constraints: list[dict[str, Any]],
        variables: dict[str, str],
    ) -> DebugSession:
        """Create a new debugging session."""
        session_id = self._generate_id()

        session = DebugSession(
            session_id=session_id,
            created_at=time.time(),
        )

        # Add variables
        for var_name, var_type in variables.items():
            session.variables[var_name] = Variable(
                name=var_name,
                var_type=var_type,
                z3_sort=self._type_to_z3_sort(var_type),
            )

        # Add constraints
        for i, c in enumerate(constraints):
            constraint_id = f"c_{i}"
            constraint = Constraint(
                id=constraint_id,
                constraint_type=ConstraintType(c.get("type", "assertion")),
                expression=c.get("expression", ""),
                z3_expr=c.get("z3_expr", c.get("expression", "")),
                description=c.get("description", ""),
                source_file=c.get("source_file"),
                source_line=c.get("source_line"),
                depends_on=c.get("depends_on", []),
            )
            session.constraints[constraint_id] = constraint

            # Update variable constraint references
            for var_name in session.variables:
                if var_name in constraint.expression:
                    session.variables[var_name].involved_constraints.append(constraint_id)

        self._sessions[session_id] = session

        logger.info(
            "Created debug session",
            session_id=session_id,
            constraints=len(session.constraints),
            variables=len(session.variables),
        )

        return session

    async def verify(self, session: DebugSession) -> DebugResult:
        """Run verification and populate debug information."""
        start_time = time.time()

        if not self._z3_available:
            return DebugResult(
                success=False,
                session=session,
                message="Z3 not available",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        try:
            from z3 import (
                And, Bool, ForAll, If, Implies, Int, Not, Or, Real, Solver,
                String, is_false, is_true, sat, unknown, unsat
            )

            solver = Solver()
            solver.set("timeout", 10000)  # 10 second timeout

            # Create Z3 variables
            z3_vars = {}
            for var_name, var in session.variables.items():
                if var.z3_sort == "Int":
                    z3_vars[var_name] = Int(var_name)
                elif var.z3_sort == "Real":
                    z3_vars[var_name] = Real(var_name)
                elif var.z3_sort == "Bool":
                    z3_vars[var_name] = Bool(var_name)
                else:
                    z3_vars[var_name] = Int(var_name)  # Default

            # Build Z3 context
            z3_context = {
                **z3_vars,
                "And": And, "Or": Or, "Not": Not, "Implies": Implies,
                "ForAll": ForAll, "If": If,
                "Int": Int, "Real": Real, "Bool": Bool,
            }

            # Add constraints
            constraint_exprs = {}
            for c_id, constraint in session.constraints.items():
                if c_id in session.disabled_constraints:
                    continue

                try:
                    expr = eval(constraint.z3_expr, {"__builtins__": {}}, z3_context)
                    constraint_exprs[c_id] = expr
                    solver.add(expr)
                except Exception as e:
                    logger.warning(f"Failed to add constraint {c_id}: {e}")

            # Add user assumptions
            for assumption in session.user_assumptions:
                try:
                    expr = eval(assumption, {"__builtins__": {}}, z3_context)
                    solver.add(expr)
                except Exception as e:
                    logger.warning(f"Failed to add assumption: {e}")

            # Check satisfiability
            result = solver.check()

            if result == sat:
                session.status = VerificationStatus.SAT
                model = solver.model()

                # Extract variable values
                for var_name, z3_var in z3_vars.items():
                    val = model.eval(z3_var)
                    session.variables[var_name].current_value = val
                    session.variables[var_name].is_free = False

                # Create proof steps for satisfiable case
                self._create_sat_proof_steps(session, model, z3_vars)

            elif result == unsat:
                session.status = VerificationStatus.UNSAT

                # Try to get unsat core
                unsat_core = self._get_unsat_core(session, z3_vars, z3_context)

                # Create counterexample
                counterexample = Counterexample(
                    id=self._generate_id(),
                    variable_assignments={},
                    violated_constraints=unsat_core,
                    explanation="The constraints are mutually contradictory.",
                )
                session.counterexamples.append(counterexample)

                # Create proof steps for unsatisfiable case
                self._create_unsat_proof_steps(session, unsat_core)

            else:
                session.status = VerificationStatus.UNKNOWN

            # Generate natural language explanation
            explanation = await self._generate_explanation(session)

            # Generate fix suggestions
            fixes = self._generate_fix_suggestions(session)

            exec_time = (time.time() - start_time) * 1000

            return DebugResult(
                success=True,
                session=session,
                message=f"Verification completed: {session.status.value}",
                execution_time_ms=exec_time,
                unsatisfiable_core=unsat_core if session.status == VerificationStatus.UNSAT else None,
                suggested_fixes=fixes,
                natural_language_explanation=explanation,
            )

        except Exception as e:
            session.status = VerificationStatus.ERROR
            return DebugResult(
                success=False,
                session=session,
                message=f"Verification error: {str(e)}",
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def _get_unsat_core(
        self,
        session: DebugSession,
        z3_vars: dict,
        z3_context: dict,
    ) -> list[str]:
        """Get the unsatisfiable core of constraints."""
        try:
            from z3 import Solver, unsat

            solver = Solver()
            solver.set("unsat_core", True)

            # Add constraints with tracking
            tracking_vars = {}
            for c_id, constraint in session.constraints.items():
                if c_id in session.disabled_constraints:
                    continue

                try:
                    from z3 import Bool
                    tracker = Bool(f"track_{c_id}")
                    tracking_vars[c_id] = tracker

                    expr = eval(constraint.z3_expr, {"__builtins__": {}}, z3_context)
                    from z3 import Implies
                    solver.add(Implies(tracker, expr))
                except Exception:
                    pass

            # Check with all trackers
            assumptions = list(tracking_vars.values())
            result = solver.check(*assumptions)

            if result == unsat:
                core = solver.unsat_core()
                # Map back to constraint IDs
                core_ids = []
                for c_id, tracker in tracking_vars.items():
                    if tracker in core:
                        core_ids.append(c_id)
                return core_ids

        except Exception as e:
            logger.warning(f"Failed to get unsat core: {e}")

        return []

    def _create_sat_proof_steps(
        self,
        session: DebugSession,
        model: Any,
        z3_vars: dict,
    ) -> None:
        """Create proof steps for satisfiable result."""
        # Step 1: State the goal
        session.proof_steps.append(ProofStep(
            step_id=self._generate_id(),
            step_type=ProofStepType.ASSUMPTION,
            description="Goal: Find values satisfying all constraints",
            active_constraints=list(session.constraints.keys()),
        ))

        # Step 2: Show the solution
        bindings = {}
        for var_name, z3_var in z3_vars.items():
            val = model.eval(z3_var)
            bindings[var_name] = str(val)

        session.proof_steps.append(ProofStep(
            step_id=self._generate_id(),
            step_type=ProofStepType.QED,
            description=f"Found satisfying assignment",
            variable_bindings=bindings,
        ))

    def _create_unsat_proof_steps(
        self,
        session: DebugSession,
        unsat_core: list[str],
    ) -> None:
        """Create proof steps for unsatisfiable result."""
        # Step 1: State the goal
        session.proof_steps.append(ProofStep(
            step_id=self._generate_id(),
            step_type=ProofStepType.ASSUMPTION,
            description="Goal: Find values satisfying all constraints",
            active_constraints=list(session.constraints.keys()),
        ))

        # Step 2: Identify conflicting constraints
        if unsat_core:
            conflict_desc = "Conflicting constraints: " + ", ".join(unsat_core)
            session.proof_steps.append(ProofStep(
                step_id=self._generate_id(),
                step_type=ProofStepType.DERIVATION,
                description=conflict_desc,
                input_facts=unsat_core,
                output_fact="contradiction",
            ))

        # Step 3: Contradiction
        session.proof_steps.append(ProofStep(
            step_id=self._generate_id(),
            step_type=ProofStepType.CONTRADICTION,
            description="No satisfying assignment exists",
        ))

    async def _generate_explanation(self, session: DebugSession) -> str:
        """Generate natural language explanation of the verification result."""
        # Build context
        context_parts = []

        context_parts.append(f"Status: {session.status.value}")
        context_parts.append(f"Variables: {list(session.variables.keys())}")
        context_parts.append(f"Number of constraints: {len(session.constraints)}")

        if session.status == VerificationStatus.SAT:
            context_parts.append("Result: The constraints are satisfiable.")
            context_parts.append("Found values:")
            for var_name, var in session.variables.items():
                if var.current_value is not None:
                    context_parts.append(f"  {var_name} = {var.current_value}")

        elif session.status == VerificationStatus.UNSAT:
            context_parts.append("Result: The constraints are contradictory.")
            if session.counterexamples:
                ce = session.counterexamples[0]
                context_parts.append(f"Conflicting constraints: {ce.violated_constraints}")

        context = "\n".join(context_parts)

        # Try LLM explanation
        try:
            response = await self._call_llm(
                self.EXPLANATION_PROMPT.format(context=context),
                "Explain this verification result simply.",
            )
            return response.get("content", "")
        except Exception:
            # Fallback to simple explanation
            return self._simple_explanation(session)

    def _simple_explanation(self, session: DebugSession) -> str:
        """Generate simple explanation without LLM."""
        if session.status == VerificationStatus.SAT:
            parts = ["The verification succeeded. Values were found that satisfy all constraints:"]
            for var_name, var in session.variables.items():
                if var.current_value is not None:
                    parts.append(f"  - {var_name} = {var.current_value}")
            return "\n".join(parts)

        elif session.status == VerificationStatus.UNSAT:
            parts = ["The verification failed. The constraints cannot all be satisfied simultaneously."]
            if session.counterexamples:
                ce = session.counterexamples[0]
                if ce.violated_constraints:
                    parts.append(f"The conflicting constraints are: {', '.join(ce.violated_constraints)}")
            return "\n".join(parts)

        else:
            return f"Verification status: {session.status.value}"

    def _generate_fix_suggestions(self, session: DebugSession) -> list[str]:
        """Generate suggestions for fixing verification failures."""
        suggestions = []

        if session.status == VerificationStatus.UNSAT:
            suggestions.append("Review the conflicting constraints for logical errors")

            if session.counterexamples:
                ce = session.counterexamples[0]
                for c_id in ce.violated_constraints:
                    if c_id in session.constraints:
                        constraint = session.constraints[c_id]
                        if constraint.constraint_type == ConstraintType.PRECONDITION:
                            suggestions.append(f"Check if precondition '{c_id}' is too restrictive")
                        elif constraint.constraint_type == ConstraintType.POSTCONDITION:
                            suggestions.append(f"Check if postcondition '{c_id}' is achievable")

            suggestions.append("Try disabling constraints one by one to find the conflict source")
            suggestions.append("Add intermediate assertions to narrow down the issue")

        elif session.status == VerificationStatus.TIMEOUT:
            suggestions.append("Simplify complex constraints")
            suggestions.append("Add bounds to unbounded variables")
            suggestions.append("Break the problem into smaller sub-problems")

        return suggestions

    def step_forward(self, session: DebugSession) -> ProofStep | None:
        """Step forward in the proof."""
        if session.current_step < len(session.proof_steps) - 1:
            session.current_step += 1
            return session.proof_steps[session.current_step]
        return None

    def step_backward(self, session: DebugSession) -> ProofStep | None:
        """Step backward in the proof."""
        if session.current_step > 0:
            session.current_step -= 1
            return session.proof_steps[session.current_step]
        return None

    def get_current_step(self, session: DebugSession) -> ProofStep | None:
        """Get the current proof step."""
        if 0 <= session.current_step < len(session.proof_steps):
            return session.proof_steps[session.current_step]
        return None

    def disable_constraint(self, session: DebugSession, constraint_id: str) -> bool:
        """Disable a constraint for what-if analysis."""
        if constraint_id in session.constraints:
            session.disabled_constraints.append(constraint_id)
            session.constraints[constraint_id].is_active = False
            return True
        return False

    def enable_constraint(self, session: DebugSession, constraint_id: str) -> bool:
        """Re-enable a disabled constraint."""
        if constraint_id in session.disabled_constraints:
            session.disabled_constraints.remove(constraint_id)
            session.constraints[constraint_id].is_active = True
            return True
        return False

    def add_assumption(self, session: DebugSession, assumption: str) -> None:
        """Add a user assumption."""
        session.user_assumptions.append(assumption)

    def get_constraint_graph(self, session: DebugSession) -> dict[str, Any]:
        """Get the constraint dependency graph."""
        nodes = []
        edges = []

        # Add variable nodes
        for var_name, var in session.variables.items():
            nodes.append({
                "id": f"var_{var_name}",
                "type": "variable",
                "label": var_name,
                "data": var.to_dict(),
            })

        # Add constraint nodes
        for c_id, constraint in session.constraints.items():
            nodes.append({
                "id": f"con_{c_id}",
                "type": "constraint",
                "label": c_id,
                "data": constraint.to_dict(),
                "is_active": constraint.is_active,
            })

            # Add edges to involved variables
            for var_name in session.variables:
                if var_name in constraint.expression:
                    edges.append({
                        "source": f"var_{var_name}",
                        "target": f"con_{c_id}",
                        "type": "involves",
                    })

            # Add dependency edges
            for dep_id in constraint.depends_on:
                edges.append({
                    "source": f"con_{dep_id}",
                    "target": f"con_{c_id}",
                    "type": "depends_on",
                })

        return {
            "nodes": nodes,
            "edges": edges,
        }

    def get_session(self, session_id: str) -> DebugSession | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def _type_to_z3_sort(self, type_str: str) -> str:
        """Convert type string to Z3 sort."""
        type_lower = type_str.lower()
        if type_lower in ("int", "integer"):
            return "Int"
        elif type_lower in ("real", "float", "double"):
            return "Real"
        elif type_lower in ("bool", "boolean"):
            return "Bool"
        elif type_lower in ("str", "string"):
            return "String"
        else:
            return "Int"  # Default

    def _generate_id(self) -> str:
        """Generate a unique ID."""
        return hashlib.sha256(
            f"{time.time()}{len(self._sessions)}".encode()
        ).hexdigest()[:12]

    def get_statistics(self) -> dict[str, Any]:
        """Get debugger statistics."""
        return {
            "active_sessions": len(self._sessions),
            "z3_available": self._z3_available,
        }
