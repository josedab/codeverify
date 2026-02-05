"""
Verification Debugger API Router

Provides REST API endpoints for interactive verification debugging:
- Create debug sessions
- Step through proofs
- Analyze counterexamples
- What-if analysis
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Import Verification Debugger
try:
    from codeverify_agents.verification_debugger import (
        VerificationDebugger,
        ConstraintType,
        VerificationStatus,
    )
    VERIFICATION_DEBUGGER_AVAILABLE = True
except ImportError:
    VERIFICATION_DEBUGGER_AVAILABLE = False
    VerificationDebugger = None
    ConstraintType = None
    VerificationStatus = None


router = APIRouter(prefix="/api/v1/debug", tags=["verification-debugger"])

# Singleton debugger instance
_debugger: Optional[VerificationDebugger] = None


def get_debugger() -> VerificationDebugger:
    """Get or create the debugger singleton."""
    global _debugger
    if _debugger is None and VERIFICATION_DEBUGGER_AVAILABLE:
        _debugger = VerificationDebugger()
    return _debugger


# =============================================================================
# Request/Response Models
# =============================================================================


class ConstraintInput(BaseModel):
    """Input for a constraint."""
    type: str = Field("assertion", description="Constraint type")
    expression: str = Field(..., description="Human-readable expression")
    z3_expr: Optional[str] = Field(None, description="Z3 expression (defaults to expression)")
    description: Optional[str] = Field("", description="Description of the constraint")
    source_file: Optional[str] = Field(None, description="Source file")
    source_line: Optional[int] = Field(None, description="Source line number")
    depends_on: Optional[List[str]] = Field(default_factory=list, description="Dependency IDs")


class CreateSessionRequest(BaseModel):
    """Request to create a debug session."""
    constraints: List[ConstraintInput] = Field(..., description="Constraints to verify")
    variables: Dict[str, str] = Field(..., description="Variable name to type mapping")


class SessionResponse(BaseModel):
    """Response with session information."""
    session_id: str
    status: str
    num_constraints: int
    num_variables: int
    num_proof_steps: int


class VerifyRequest(BaseModel):
    """Request to run verification."""
    session_id: str = Field(..., description="Session ID")


class StepRequest(BaseModel):
    """Request to step in the proof."""
    session_id: str = Field(..., description="Session ID")


class DisableConstraintRequest(BaseModel):
    """Request to disable a constraint."""
    session_id: str = Field(..., description="Session ID")
    constraint_id: str = Field(..., description="Constraint ID to disable")


class AddAssumptionRequest(BaseModel):
    """Request to add an assumption."""
    session_id: str = Field(..., description="Session ID")
    assumption: str = Field(..., description="Z3 expression for assumption")


# =============================================================================
# API Endpoints
# =============================================================================


@router.post(
    "/session",
    response_model=SessionResponse,
    summary="Create Debug Session",
    description="Create a new verification debugging session"
)
async def create_session(request: CreateSessionRequest) -> SessionResponse:
    """
    Create a new debugging session.

    Provide constraints and variables to set up the verification problem.
    """
    if not VERIFICATION_DEBUGGER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Verification Debugger is not available"
        )

    debugger = get_debugger()
    if not debugger:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Verification Debugger"
        )

    # Convert constraints
    constraints = []
    for c in request.constraints:
        constraints.append({
            "type": c.type,
            "expression": c.expression,
            "z3_expr": c.z3_expr or c.expression,
            "description": c.description or "",
            "source_file": c.source_file,
            "source_line": c.source_line,
            "depends_on": c.depends_on or [],
        })

    session = debugger.create_session(constraints, request.variables)

    return SessionResponse(
        session_id=session.session_id,
        status=session.status.value,
        num_constraints=len(session.constraints),
        num_variables=len(session.variables),
        num_proof_steps=len(session.proof_steps),
    )


@router.post(
    "/verify",
    summary="Run Verification",
    description="Run verification on a debug session"
)
async def run_verification(request: VerifyRequest) -> Dict[str, Any]:
    """
    Run verification and populate proof steps.

    Returns the verification result with counterexamples if unsatisfiable.
    """
    if not VERIFICATION_DEBUGGER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Verification Debugger is not available"
        )

    debugger = get_debugger()
    if not debugger:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Verification Debugger"
        )

    session = debugger.get_session(request.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {request.session_id}"
        )

    result = await debugger.verify(session)

    return result.to_dict()


@router.get(
    "/session/{session_id}",
    summary="Get Session",
    description="Get full session details"
)
async def get_session(session_id: str) -> Dict[str, Any]:
    """Get session details including all constraints and proof steps."""
    if not VERIFICATION_DEBUGGER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Verification Debugger is not available"
        )

    debugger = get_debugger()
    if not debugger:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Verification Debugger"
        )

    session = debugger.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )

    return session.to_dict()


@router.post(
    "/step/forward",
    summary="Step Forward",
    description="Step forward in the proof"
)
async def step_forward(request: StepRequest) -> Dict[str, Any]:
    """Step forward in the proof."""
    if not VERIFICATION_DEBUGGER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Verification Debugger is not available"
        )

    debugger = get_debugger()
    if not debugger:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Verification Debugger"
        )

    session = debugger.get_session(request.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {request.session_id}"
        )

    step = debugger.step_forward(session)

    return {
        "current_step": session.current_step,
        "total_steps": len(session.proof_steps),
        "step": step.to_dict() if step else None,
        "has_next": session.current_step < len(session.proof_steps) - 1,
        "has_prev": session.current_step > 0,
    }


@router.post(
    "/step/backward",
    summary="Step Backward",
    description="Step backward in the proof"
)
async def step_backward(request: StepRequest) -> Dict[str, Any]:
    """Step backward in the proof."""
    if not VERIFICATION_DEBUGGER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Verification Debugger is not available"
        )

    debugger = get_debugger()
    if not debugger:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Verification Debugger"
        )

    session = debugger.get_session(request.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {request.session_id}"
        )

    step = debugger.step_backward(session)

    return {
        "current_step": session.current_step,
        "total_steps": len(session.proof_steps),
        "step": step.to_dict() if step else None,
        "has_next": session.current_step < len(session.proof_steps) - 1,
        "has_prev": session.current_step > 0,
    }


@router.get(
    "/step/current/{session_id}",
    summary="Get Current Step",
    description="Get the current proof step"
)
async def get_current_step(session_id: str) -> Dict[str, Any]:
    """Get the current proof step."""
    if not VERIFICATION_DEBUGGER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Verification Debugger is not available"
        )

    debugger = get_debugger()
    if not debugger:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Verification Debugger"
        )

    session = debugger.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )

    step = debugger.get_current_step(session)

    return {
        "current_step": session.current_step,
        "total_steps": len(session.proof_steps),
        "step": step.to_dict() if step else None,
    }


@router.post(
    "/constraint/disable",
    summary="Disable Constraint",
    description="Disable a constraint for what-if analysis"
)
async def disable_constraint(request: DisableConstraintRequest) -> Dict[str, Any]:
    """
    Disable a constraint.

    Useful for what-if analysis to see which constraint is causing issues.
    """
    if not VERIFICATION_DEBUGGER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Verification Debugger is not available"
        )

    debugger = get_debugger()
    if not debugger:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Verification Debugger"
        )

    session = debugger.get_session(request.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {request.session_id}"
        )

    success = debugger.disable_constraint(session, request.constraint_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Constraint not found: {request.constraint_id}"
        )

    return {
        "disabled": True,
        "constraint_id": request.constraint_id,
        "disabled_constraints": session.disabled_constraints,
    }


@router.post(
    "/constraint/enable",
    summary="Enable Constraint",
    description="Re-enable a disabled constraint"
)
async def enable_constraint(request: DisableConstraintRequest) -> Dict[str, Any]:
    """Re-enable a disabled constraint."""
    if not VERIFICATION_DEBUGGER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Verification Debugger is not available"
        )

    debugger = get_debugger()
    if not debugger:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Verification Debugger"
        )

    session = debugger.get_session(request.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {request.session_id}"
        )

    success = debugger.enable_constraint(session, request.constraint_id)

    return {
        "enabled": success,
        "constraint_id": request.constraint_id,
        "disabled_constraints": session.disabled_constraints,
    }


@router.post(
    "/assumption",
    summary="Add Assumption",
    description="Add a user assumption to the verification"
)
async def add_assumption(request: AddAssumptionRequest) -> Dict[str, Any]:
    """
    Add a user assumption.

    Useful for exploring what happens under specific conditions.
    """
    if not VERIFICATION_DEBUGGER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Verification Debugger is not available"
        )

    debugger = get_debugger()
    if not debugger:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Verification Debugger"
        )

    session = debugger.get_session(request.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {request.session_id}"
        )

    debugger.add_assumption(session, request.assumption)

    return {
        "added": True,
        "assumption": request.assumption,
        "all_assumptions": session.user_assumptions,
    }


@router.get(
    "/graph/{session_id}",
    summary="Get Constraint Graph",
    description="Get the constraint dependency graph"
)
async def get_constraint_graph(session_id: str) -> Dict[str, Any]:
    """
    Get the constraint dependency graph.

    Returns nodes and edges for visualization.
    """
    if not VERIFICATION_DEBUGGER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Verification Debugger is not available"
        )

    debugger = get_debugger()
    if not debugger:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Verification Debugger"
        )

    session = debugger.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )

    graph = debugger.get_constraint_graph(session)

    return graph


@router.get(
    "/stats",
    summary="Get Debugger Statistics",
    description="Get debugger statistics"
)
async def get_stats() -> Dict[str, Any]:
    """Get debugger statistics."""
    if not VERIFICATION_DEBUGGER_AVAILABLE:
        return {
            "available": False,
            "message": "Verification Debugger is not available",
        }

    debugger = get_debugger()
    if not debugger:
        return {
            "available": False,
            "message": "Failed to initialize debugger",
        }

    stats = debugger.get_statistics()
    stats["available"] = True

    return stats
