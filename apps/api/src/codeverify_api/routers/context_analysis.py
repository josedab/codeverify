"""
Context-Aware Analysis API Router

Provides REST API endpoints for context-aware code analysis:
- Analyze project context
- Adjust finding severity based on context
- Get project patterns and conventions
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Import Context-Aware Analysis
try:
    from codeverify_agents.context_aware_analysis import (
        ContextAwareAnalyzer,
        ArchitectureType,
        ProjectType,
        Severity,
    )
    CONTEXT_ANALYSIS_AVAILABLE = True
except ImportError:
    CONTEXT_ANALYSIS_AVAILABLE = False
    ContextAwareAnalyzer = None
    ArchitectureType = None
    ProjectType = None
    Severity = None


router = APIRouter(prefix="/api/v1/context", tags=["context-analysis"])

# Singleton analyzer instance
_analyzer: Optional[ContextAwareAnalyzer] = None


def get_analyzer() -> ContextAwareAnalyzer:
    """Get or create the analyzer singleton."""
    global _analyzer
    if _analyzer is None and CONTEXT_ANALYSIS_AVAILABLE:
        _analyzer = ContextAwareAnalyzer()
    return _analyzer


# =============================================================================
# Request/Response Models
# =============================================================================


class AnalyzeProjectRequest(BaseModel):
    """Request to analyze a project."""
    project_name: str = Field(..., description="Name of the project")
    file_paths: List[str] = Field(..., description="List of file paths")
    file_contents: Dict[str, str] = Field(..., description="Map of path to content")
    dependencies: Optional[List[str]] = Field(None, description="List of dependencies")


class AnalyzeProjectResponse(BaseModel):
    """Response with project context."""
    context_id: str
    project_name: str
    project_type: str
    architecture_type: str
    architecture_confidence: float
    patterns_count: int
    conventions_count: int
    frameworks: List[str]


class AdjustSeverityRequest(BaseModel):
    """Request to adjust finding severity."""
    finding_id: str = Field(..., description="ID of the finding")
    finding_type: str = Field(..., description="Type of the finding")
    original_severity: str = Field(..., description="Original severity level")
    context_id: str = Field(..., description="Project context ID")
    code_snippet: Optional[str] = Field(None, description="Related code snippet")


class BatchAdjustRequest(BaseModel):
    """Request to adjust multiple findings."""
    context_id: str = Field(..., description="Project context ID")
    findings: List[Dict[str, str]] = Field(..., description="List of findings to adjust")


# =============================================================================
# API Endpoints
# =============================================================================


@router.post(
    "/analyze",
    response_model=AnalyzeProjectResponse,
    summary="Analyze Project",
    description="Analyze a project to build context"
)
async def analyze_project(request: AnalyzeProjectRequest) -> AnalyzeProjectResponse:
    """
    Analyze a project to build context.

    This builds a context model including architecture, patterns, and conventions
    that can be used to adjust finding severity.
    """
    if not CONTEXT_ANALYSIS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Context-Aware Analysis is not available"
        )

    analyzer = get_analyzer()
    if not analyzer:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Context-Aware Analysis"
        )

    context = analyzer.analyze_project(
        project_name=request.project_name,
        file_paths=request.file_paths,
        file_contents=request.file_contents,
        dependencies=request.dependencies,
    )

    return AnalyzeProjectResponse(
        context_id=context.id,
        project_name=context.name,
        project_type=context.project_type.value,
        architecture_type=context.architecture.type.value,
        architecture_confidence=context.architecture.confidence,
        patterns_count=len(context.patterns),
        conventions_count=len(context.conventions),
        frameworks=context.frameworks,
    )


@router.get(
    "/context/{context_id}",
    summary="Get Context",
    description="Get full project context details"
)
async def get_context(context_id: str) -> Dict[str, Any]:
    """Get full project context details."""
    if not CONTEXT_ANALYSIS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Context-Aware Analysis is not available"
        )

    analyzer = get_analyzer()
    if not analyzer:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Context-Aware Analysis"
        )

    context = analyzer.get_context(context_id)
    if not context:
        raise HTTPException(
            status_code=404,
            detail=f"Context not found: {context_id}"
        )

    return context.to_dict()


@router.get(
    "/contexts",
    summary="List Contexts",
    description="List all project contexts"
)
async def list_contexts() -> Dict[str, Any]:
    """List all project contexts."""
    if not CONTEXT_ANALYSIS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Context-Aware Analysis is not available"
        )

    analyzer = get_analyzer()
    if not analyzer:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Context-Aware Analysis"
        )

    contexts = analyzer.list_contexts()

    return {
        "contexts": [
            {
                "id": c.id,
                "name": c.name,
                "project_type": c.project_type.value,
                "architecture": c.architecture.type.value,
                "created_at": c.created_at.isoformat(),
            }
            for c in contexts
        ],
        "count": len(contexts),
    }


@router.get(
    "/context/{context_id}/architecture",
    summary="Get Architecture",
    description="Get detected architecture details"
)
async def get_architecture(context_id: str) -> Dict[str, Any]:
    """Get detected architecture details."""
    if not CONTEXT_ANALYSIS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Context-Aware Analysis is not available"
        )

    analyzer = get_analyzer()
    if not analyzer:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Context-Aware Analysis"
        )

    context = analyzer.get_context(context_id)
    if not context:
        raise HTTPException(
            status_code=404,
            detail=f"Context not found: {context_id}"
        )

    return context.architecture.to_dict()


@router.get(
    "/context/{context_id}/patterns",
    summary="Get Patterns",
    description="Get detected coding patterns"
)
async def get_patterns(context_id: str) -> Dict[str, Any]:
    """Get detected coding patterns."""
    if not CONTEXT_ANALYSIS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Context-Aware Analysis is not available"
        )

    analyzer = get_analyzer()
    if not analyzer:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Context-Aware Analysis"
        )

    context = analyzer.get_context(context_id)
    if not context:
        raise HTTPException(
            status_code=404,
            detail=f"Context not found: {context_id}"
        )

    return {
        "patterns": [p.to_dict() for p in context.patterns],
        "count": len(context.patterns),
    }


@router.get(
    "/context/{context_id}/conventions",
    summary="Get Conventions",
    description="Get detected coding conventions"
)
async def get_conventions(context_id: str) -> Dict[str, Any]:
    """Get detected coding conventions."""
    if not CONTEXT_ANALYSIS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Context-Aware Analysis is not available"
        )

    analyzer = get_analyzer()
    if not analyzer:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Context-Aware Analysis"
        )

    context = analyzer.get_context(context_id)
    if not context:
        raise HTTPException(
            status_code=404,
            detail=f"Context not found: {context_id}"
        )

    return {
        "conventions": [c.to_dict() for c in context.conventions],
        "count": len(context.conventions),
    }


@router.post(
    "/adjust",
    summary="Adjust Severity",
    description="Adjust finding severity based on context"
)
async def adjust_severity(request: AdjustSeverityRequest) -> Dict[str, Any]:
    """
    Adjust finding severity based on project context.

    Returns the original and adjusted severity with explanation.
    """
    if not CONTEXT_ANALYSIS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Context-Aware Analysis is not available"
        )

    analyzer = get_analyzer()
    if not analyzer:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Context-Aware Analysis"
        )

    result = analyzer.adjust_finding_severity(
        finding_id=request.finding_id,
        finding_type=request.finding_type,
        original_severity=request.original_severity,
        context_id=request.context_id,
        code_snippet=request.code_snippet,
    )

    return result.to_dict()


@router.post(
    "/adjust/batch",
    summary="Batch Adjust Severity",
    description="Adjust severity for multiple findings"
)
async def batch_adjust_severity(request: BatchAdjustRequest) -> Dict[str, Any]:
    """
    Adjust severity for multiple findings.

    Efficiently processes multiple findings against the same context.
    """
    if not CONTEXT_ANALYSIS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Context-Aware Analysis is not available"
        )

    analyzer = get_analyzer()
    if not analyzer:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Context-Aware Analysis"
        )

    results = []
    for finding in request.findings:
        result = analyzer.adjust_finding_severity(
            finding_id=finding.get("finding_id", ""),
            finding_type=finding.get("finding_type", ""),
            original_severity=finding.get("original_severity", "medium"),
            context_id=request.context_id,
            code_snippet=finding.get("code_snippet"),
        )
        results.append(result.to_dict())

    return {
        "results": results,
        "count": len(results),
        "adjustments_made": sum(1 for r in results if r["adjustment"] is not None),
    }


@router.get(
    "/architecture-types",
    summary="Get Architecture Types",
    description="Get available architecture types"
)
async def get_architecture_types() -> Dict[str, Any]:
    """Get available architecture types."""
    if not CONTEXT_ANALYSIS_AVAILABLE:
        return {
            "available": False,
            "types": [],
        }

    types = [
        {"value": "monolith", "name": "Monolith", "description": "Single deployable unit"},
        {"value": "microservices", "name": "Microservices", "description": "Distributed services"},
        {"value": "modular_monolith", "name": "Modular Monolith", "description": "Well-structured monolith"},
        {"value": "serverless", "name": "Serverless", "description": "Function-as-a-Service"},
        {"value": "event_driven", "name": "Event Driven", "description": "Event-based architecture"},
        {"value": "layered", "name": "Layered", "description": "Traditional layered architecture"},
        {"value": "hexagonal", "name": "Hexagonal", "description": "Ports and adapters"},
    ]

    return {
        "available": True,
        "types": types,
    }


@router.get(
    "/project-types",
    summary="Get Project Types",
    description="Get available project types"
)
async def get_project_types() -> Dict[str, Any]:
    """Get available project types."""
    if not CONTEXT_ANALYSIS_AVAILABLE:
        return {
            "available": False,
            "types": [],
        }

    types = [
        {"value": "web_frontend", "name": "Web Frontend", "description": "Browser-based UI"},
        {"value": "web_backend", "name": "Web Backend", "description": "Server-side application"},
        {"value": "cli_tool", "name": "CLI Tool", "description": "Command-line interface"},
        {"value": "library", "name": "Library", "description": "Reusable library/package"},
        {"value": "api_service", "name": "API Service", "description": "REST/GraphQL API"},
        {"value": "mobile_app", "name": "Mobile App", "description": "Mobile application"},
        {"value": "data_pipeline", "name": "Data Pipeline", "description": "ETL/data processing"},
        {"value": "ml_project", "name": "ML Project", "description": "Machine learning"},
    ]

    return {
        "available": True,
        "types": types,
    }


@router.get(
    "/stats",
    summary="Get Statistics",
    description="Get context analysis statistics"
)
async def get_stats() -> Dict[str, Any]:
    """Get context analysis statistics."""
    if not CONTEXT_ANALYSIS_AVAILABLE:
        return {
            "available": False,
            "message": "Context-Aware Analysis is not available",
        }

    analyzer = get_analyzer()
    if not analyzer:
        return {
            "available": False,
            "message": "Failed to initialize analyzer",
        }

    stats = analyzer.get_statistics()
    stats["available"] = True

    return stats
