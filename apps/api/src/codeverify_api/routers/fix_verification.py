"""
Fix Verification API Router

Provides REST API endpoints for automated fix verification:
- Register issues
- Submit fixes for verification
- Verify fixes automatically
- Get verification results
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Import Fix Verification Engine
try:
    from codeverify_agents.fix_verification import (
        FixVerificationEngine,
        FixStatus,
        IssueType,
        SafetyLevel,
    )
    FIX_VERIFICATION_AVAILABLE = True
except ImportError:
    FIX_VERIFICATION_AVAILABLE = False
    FixVerificationEngine = None
    FixStatus = None
    IssueType = None
    SafetyLevel = None


router = APIRouter(prefix="/api/v1/fix-verify", tags=["fix-verification"])

# Singleton engine instance
_engine: Optional[FixVerificationEngine] = None


def get_engine() -> FixVerificationEngine:
    """Get or create the engine singleton."""
    global _engine
    if _engine is None and FIX_VERIFICATION_AVAILABLE:
        _engine = FixVerificationEngine()
    return _engine


# =============================================================================
# Request/Response Models
# =============================================================================


class RegisterIssueRequest(BaseModel):
    """Request to register an issue."""
    issue_type: str = Field(..., description="Type: security, bug, performance, etc.")
    description: str = Field(..., description="Description of the issue")
    file_path: str = Field(..., description="Path to the affected file")
    line_start: int = Field(..., ge=1, description="Start line number")
    line_end: int = Field(..., ge=1, description="End line number")
    severity: str = Field("medium", description="Severity level")
    code_snippet: str = Field(..., description="Affected code snippet")
    suggested_fix: Optional[str] = Field(None, description="Suggested fix if available")


class IssueResponse(BaseModel):
    """Response with issue information."""
    id: str
    issue_type: str
    description: str
    file_path: str
    severity: str


class SubmitFixRequest(BaseModel):
    """Request to submit a fix."""
    issue_id: str = Field(..., description="ID of the issue being fixed")
    original_code: str = Field(..., description="Original code before fix")
    fixed_code: str = Field(..., description="Code after fix")
    description: str = Field(..., description="Description of the fix")
    file_path: str = Field(..., description="Path to the file")
    line_start: int = Field(..., ge=1, description="Start line number")
    line_end: int = Field(..., ge=1, description="End line number")
    author: Optional[str] = Field(None, description="Author of the fix")


class FixResponse(BaseModel):
    """Response with fix information."""
    id: str
    issue_id: str
    description: str
    file_path: str


class VerifyRequest(BaseModel):
    """Request to verify a fix."""
    fix_id: str = Field(..., description="ID of the fix to verify")
    context: Optional[Dict[str, str]] = Field(None, description="Additional context")


# =============================================================================
# API Endpoints
# =============================================================================


@router.post(
    "/issue",
    response_model=IssueResponse,
    summary="Register Issue",
    description="Register an issue to be fixed"
)
async def register_issue(request: RegisterIssueRequest) -> IssueResponse:
    """
    Register an issue that needs to be fixed.

    Issues are tracked and linked to fixes for verification.
    """
    if not FIX_VERIFICATION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Fix Verification Engine is not available"
        )

    engine = get_engine()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Fix Verification Engine"
        )

    issue = engine.register_issue(
        issue_type=request.issue_type,
        description=request.description,
        file_path=request.file_path,
        line_start=request.line_start,
        line_end=request.line_end,
        severity=request.severity,
        code_snippet=request.code_snippet,
        suggested_fix=request.suggested_fix,
    )

    return IssueResponse(
        id=issue.id,
        issue_type=issue.type.value,
        description=issue.description,
        file_path=issue.file_path,
        severity=issue.severity,
    )


@router.get(
    "/issue/{issue_id}",
    summary="Get Issue",
    description="Get issue details"
)
async def get_issue(issue_id: str) -> Dict[str, Any]:
    """Get issue details."""
    if not FIX_VERIFICATION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Fix Verification Engine is not available"
        )

    engine = get_engine()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Fix Verification Engine"
        )

    issue = engine.get_issue(issue_id)
    if not issue:
        raise HTTPException(
            status_code=404,
            detail=f"Issue not found: {issue_id}"
        )

    return issue.to_dict()


@router.post(
    "/fix",
    response_model=FixResponse,
    summary="Submit Fix",
    description="Submit a fix for verification"
)
async def submit_fix(request: SubmitFixRequest) -> FixResponse:
    """
    Submit a fix for an issue.

    The fix will be verified to ensure it resolves the issue safely.
    """
    if not FIX_VERIFICATION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Fix Verification Engine is not available"
        )

    engine = get_engine()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Fix Verification Engine"
        )

    # Verify issue exists
    issue = engine.get_issue(request.issue_id)
    if not issue:
        raise HTTPException(
            status_code=404,
            detail=f"Issue not found: {request.issue_id}"
        )

    fix = engine.submit_fix(
        issue_id=request.issue_id,
        original_code=request.original_code,
        fixed_code=request.fixed_code,
        description=request.description,
        file_path=request.file_path,
        line_start=request.line_start,
        line_end=request.line_end,
        author=request.author,
    )

    return FixResponse(
        id=fix.id,
        issue_id=fix.issue_id,
        description=fix.description,
        file_path=fix.file_path,
    )


@router.get(
    "/fix/{fix_id}",
    summary="Get Fix",
    description="Get fix details"
)
async def get_fix(fix_id: str) -> Dict[str, Any]:
    """Get fix details."""
    if not FIX_VERIFICATION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Fix Verification Engine is not available"
        )

    engine = get_engine()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Fix Verification Engine"
        )

    fix = engine.get_fix(fix_id)
    if not fix:
        raise HTTPException(
            status_code=404,
            detail=f"Fix not found: {fix_id}"
        )

    return fix.to_dict()


@router.post(
    "/verify",
    summary="Verify Fix",
    description="Verify a submitted fix"
)
async def verify_fix(request: VerifyRequest) -> Dict[str, Any]:
    """
    Verify a submitted fix.

    Checks if the fix resolves the issue, doesn't introduce regressions,
    and is safe to apply.
    """
    if not FIX_VERIFICATION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Fix Verification Engine is not available"
        )

    engine = get_engine()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Fix Verification Engine"
        )

    try:
        result = await engine.verify_fix(
            fix_id=request.fix_id,
            context=request.context,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    return result.to_dict()


@router.get(
    "/result/{result_id}",
    summary="Get Result",
    description="Get verification result details"
)
async def get_result(result_id: str) -> Dict[str, Any]:
    """Get verification result details."""
    if not FIX_VERIFICATION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Fix Verification Engine is not available"
        )

    engine = get_engine()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Fix Verification Engine"
        )

    result = engine.get_result(result_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Result not found: {result_id}"
        )

    return result.to_dict()


@router.get(
    "/results/{fix_id}",
    summary="Get Results for Fix",
    description="Get all verification results for a fix"
)
async def get_results_for_fix(fix_id: str) -> Dict[str, Any]:
    """Get all verification results for a fix."""
    if not FIX_VERIFICATION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Fix Verification Engine is not available"
        )

    engine = get_engine()
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Fix Verification Engine"
        )

    results = engine.get_results_for_fix(fix_id)

    return {
        "results": [r.to_dict() for r in results],
        "count": len(results),
        "fix_id": fix_id,
    }


@router.get(
    "/issue-types",
    summary="Get Issue Types",
    description="Get available issue types"
)
async def get_issue_types() -> Dict[str, Any]:
    """Get available issue types."""
    if not FIX_VERIFICATION_AVAILABLE:
        return {
            "available": False,
            "types": [],
        }

    types = [
        {"value": "security", "name": "Security", "description": "Security vulnerability"},
        {"value": "bug", "name": "Bug", "description": "Functional bug"},
        {"value": "performance", "name": "Performance", "description": "Performance issue"},
        {"value": "style", "name": "Style", "description": "Code style issue"},
        {"value": "logic", "name": "Logic", "description": "Logic error"},
        {"value": "type_error", "name": "Type Error", "description": "Type-related error"},
        {"value": "resource_leak", "name": "Resource Leak", "description": "Resource not properly released"},
        {"value": "null_check", "name": "Null Check", "description": "Missing null/none check"},
        {"value": "bounds_check", "name": "Bounds Check", "description": "Missing bounds check"},
        {"value": "concurrency", "name": "Concurrency", "description": "Concurrency issue"},
    ]

    return {
        "available": True,
        "types": types,
    }


@router.get(
    "/safety-levels",
    summary="Get Safety Levels",
    description="Get safety level descriptions"
)
async def get_safety_levels() -> Dict[str, Any]:
    """Get safety level descriptions."""
    if not FIX_VERIFICATION_AVAILABLE:
        return {
            "available": False,
            "levels": [],
        }

    levels = [
        {"value": "safe", "name": "Safe", "description": "Fix is safe to apply"},
        {"value": "mostly_safe", "name": "Mostly Safe", "description": "Minor concerns, review recommended"},
        {"value": "needs_review", "name": "Needs Review", "description": "Manual review required"},
        {"value": "potentially_unsafe", "name": "Potentially Unsafe", "description": "Significant concerns"},
        {"value": "unsafe", "name": "Unsafe", "description": "Should not be applied"},
    ]

    return {
        "available": True,
        "levels": levels,
    }


@router.get(
    "/stats",
    summary="Get Statistics",
    description="Get verification statistics"
)
async def get_stats() -> Dict[str, Any]:
    """Get verification statistics."""
    if not FIX_VERIFICATION_AVAILABLE:
        return {
            "available": False,
            "message": "Fix Verification Engine is not available",
        }

    engine = get_engine()
    if not engine:
        return {
            "available": False,
            "message": "Failed to initialize engine",
        }

    stats = engine.get_statistics()
    stats["available"] = True

    return stats
