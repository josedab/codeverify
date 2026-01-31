"""Verification Debugger API endpoints."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter()


class VerifyWithTraceRequest(BaseModel):
    """Request to verify with full tracing."""

    formula: str = Field(..., description="SMT-LIB formatted formula")
    name: str = Field(default="Verification", description="Name for this verification")
    description: str | None = Field(default=None, description="Optional description")
    timeout_ms: int = Field(default=60000, ge=1000, le=300000, description="Timeout in milliseconds")


class VerificationStepResponse(BaseModel):
    """A verification step in the trace."""

    id: int
    step_type: str
    description: str
    formula: str | None
    result: str | None
    status: str
    duration_ms: float


class VerificationTraceResponse(BaseModel):
    """Complete verification trace response."""

    id: str
    name: str
    description: str | None
    steps: list[VerificationStepResponse]
    constraints: list[str]
    variables: dict[str, str]
    result: str | None
    counterexample: dict[str, Any] | None
    total_duration_ms: float
    explanation: dict[str, Any] | None = None
    visualization: dict[str, Any] | None = None


class InteractiveSessionRequest(BaseModel):
    """Request for interactive session operations."""

    action: str = Field(..., description="Action: add_variable, add_constraint, check, push, pop, reset")
    data: dict[str, Any] = Field(default_factory=dict, description="Action-specific data")


class InteractiveSessionResponse(BaseModel):
    """Response from interactive session."""

    success: bool
    result: dict[str, Any] | None = None
    state: dict[str, Any]
    error: str | None = None


# In-memory session storage (should use Redis in production)
_sessions: dict[str, Any] = {}


@router.post("/trace", response_model=VerificationTraceResponse)
async def verify_with_trace(request: VerifyWithTraceRequest) -> VerificationTraceResponse:
    """
    Verify a formula with full step-by-step tracing.

    Returns a complete trace of the verification process that can be
    used for debugging and visualization.
    """
    from codeverify_verifier.debugger import VerificationDebugger

    debugger = VerificationDebugger(timeout_ms=request.timeout_ms)
    trace = debugger.verify_with_trace(
        formula=request.formula,
        name=request.name,
        description=request.description,
    )

    # Generate explanation and visualization
    explanation = debugger.explain_result(trace)
    visualization = debugger.generate_visualization_data(trace)

    return VerificationTraceResponse(
        id=trace.id,
        name=trace.name,
        description=trace.description,
        steps=[
            VerificationStepResponse(
                id=s.id,
                step_type=s.step_type.value,
                description=s.description,
                formula=s.formula,
                result=s.result,
                status=s.status.value,
                duration_ms=s.duration_ms,
            )
            for s in trace.steps
        ],
        constraints=trace.constraints,
        variables=trace.variables,
        result=trace.result,
        counterexample=trace.counterexample,
        total_duration_ms=trace.total_duration_ms,
        explanation=explanation,
        visualization=visualization,
    )


@router.post("/session/create")
async def create_interactive_session(
    timeout_ms: int = 60000,
) -> dict[str, Any]:
    """
    Create a new interactive verification session.

    Returns a session ID that can be used for subsequent operations.
    """
    import uuid
    from codeverify_verifier.debugger import create_interactive_session

    session_id = str(uuid.uuid4())
    _sessions[session_id] = create_interactive_session(timeout_ms)

    return {
        "session_id": session_id,
        "status": "created",
        "timeout_ms": timeout_ms,
    }


@router.post("/session/{session_id}/execute", response_model=InteractiveSessionResponse)
async def execute_session_action(
    session_id: str,
    request: InteractiveSessionRequest,
) -> InteractiveSessionResponse:
    """
    Execute an action in an interactive session.

    Actions:
    - add_variable: {"name": "x", "type": "int"}
    - add_constraint: {"constraint": "(> x 0)", "description": "x is positive"}
    - check: {}
    - push: {}
    - pop: {}
    - reset: {}
    """
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    session = _sessions[session_id]
    result = None
    error = None

    try:
        if request.action == "add_variable":
            var = session.add_variable(
                name=request.data.get("name", "x"),
                var_type=request.data.get("type", "int"),
            )
            result = {"variable": str(var)}

        elif request.action == "add_constraint":
            success = session.add_constraint(
                constraint=request.data.get("constraint", ""),
                description=request.data.get("description", ""),
            )
            result = {"added": success}

        elif request.action == "check":
            result = session.check()

        elif request.action == "push":
            session.push()
            result = {"pushed": True}

        elif request.action == "pop":
            session.pop()
            result = {"popped": True}

        elif request.action == "reset":
            session.reset()
            result = {"reset": True}

        else:
            raise ValueError(f"Unknown action: {request.action}")

    except Exception as e:
        error = str(e)

    return InteractiveSessionResponse(
        success=error is None,
        result=result,
        state=session.get_state(),
        error=error,
    )


@router.get("/session/{session_id}")
async def get_session_state(session_id: str) -> dict[str, Any]:
    """Get the current state of an interactive session."""
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    session = _sessions[session_id]
    return {
        "session_id": session_id,
        "state": session.get_state(),
        "history": session.history[-10:],  # Last 10 actions
    }


@router.delete("/session/{session_id}")
async def delete_session(session_id: str) -> dict[str, Any]:
    """Delete an interactive session."""
    if session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )

    del _sessions[session_id]
    return {"deleted": True, "session_id": session_id}


@router.post("/explain")
async def explain_verification(
    formula: str,
    result: str,
    counterexample: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate a human-readable explanation for a verification result.

    This endpoint can be used to explain results from external verification.
    """
    explanation = {
        "summary": "",
        "meaning": "",
        "evidence": [],
        "recommendations": [],
    }

    if result == "sat":
        explanation["summary"] = "⚠️ Potential issue found"
        explanation["meaning"] = (
            "The verification found inputs that violate the checked property."
        )
        if counterexample:
            explanation["evidence"].append("Counterexample values:")
            for var, value in counterexample.items():
                explanation["evidence"].append(f"  • {var} = {value}")
        explanation["recommendations"] = [
            "Review the code with these input values",
            "Add appropriate validation or bounds checking",
        ]

    elif result == "unsat":
        explanation["summary"] = "✅ Property verified correct"
        explanation["meaning"] = (
            "No inputs can violate the property - the code is correct."
        )
        explanation["recommendations"] = [
            "No action required for this property",
        ]

    else:
        explanation["summary"] = "❓ Result inconclusive"
        explanation["meaning"] = "The solver could not determine the result."
        explanation["recommendations"] = [
            "Try increasing the timeout",
            "Simplify the verification condition",
        ]

    return explanation
