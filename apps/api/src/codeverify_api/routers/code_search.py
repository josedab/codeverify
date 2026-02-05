"""
Smart Code Search API Router

Provides REST API endpoints for semantic code search:
- Index code for searching
- Semantic and structural search
- Find duplicates
- Cluster related code
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Import Smart Code Search
try:
    from codeverify_agents.smart_code_search import (
        SmartCodeSearch,
        CodeUnit,
        CodeType,
        SearchQuery,
        SearchMode,
    )
    SMART_CODE_SEARCH_AVAILABLE = True
except ImportError:
    SMART_CODE_SEARCH_AVAILABLE = False
    SmartCodeSearch = None
    CodeUnit = None
    CodeType = None
    SearchQuery = None
    SearchMode = None


router = APIRouter(prefix="/api/v1/search", tags=["code-search"])

# Singleton search engine instance
_search_engine: Optional[SmartCodeSearch] = None


def get_search_engine() -> SmartCodeSearch:
    """Get or create the search engine singleton."""
    global _search_engine
    if _search_engine is None and SMART_CODE_SEARCH_AVAILABLE:
        _search_engine = SmartCodeSearch()
    return _search_engine


# =============================================================================
# Request/Response Models
# =============================================================================


class CodeUnitInput(BaseModel):
    """Input for a code unit to index."""
    id: Optional[str] = Field(None, description="Unique ID (auto-generated if not provided)")
    code: str = Field(..., description="The code content")
    code_type: str = Field("snippet", description="Type: function, class, method, module, snippet")
    name: Optional[str] = Field(None, description="Name of the code unit")
    file_path: Optional[str] = Field(None, description="Source file path")
    line_start: Optional[int] = Field(None, description="Start line number")
    line_end: Optional[int] = Field(None, description="End line number")
    language: str = Field("python", description="Programming language")
    docstring: Optional[str] = Field(None, description="Docstring/documentation")
    signature: Optional[str] = Field(None, description="Function/method signature")


class IndexRequest(BaseModel):
    """Request to index code units."""
    code_units: List[CodeUnitInput] = Field(..., description="Code units to index")
    generate_embeddings: bool = Field(False, description="Generate semantic embeddings")


class SearchRequest(BaseModel):
    """Request to search code."""
    query: str = Field(..., description="Search query")
    mode: str = Field("hybrid", description="Search mode: semantic, structural, hybrid")
    language: Optional[str] = Field(None, description="Filter by language")
    code_type: Optional[str] = Field(None, description="Filter by code type")
    file_pattern: Optional[str] = Field(None, description="Filter by file pattern regex")
    max_results: int = Field(10, ge=1, le=100, description="Maximum results")
    min_similarity: float = Field(0.5, ge=0, le=1, description="Minimum similarity score")


class FindDuplicatesRequest(BaseModel):
    """Request to find duplicate code."""
    threshold: float = Field(0.9, ge=0.5, le=1.0, description="Similarity threshold")


class FindRelatedRequest(BaseModel):
    """Request to find related code."""
    code: str = Field(..., description="Code to find related items for")
    code_type: str = Field("snippet", description="Type of the code")
    max_results: int = Field(5, ge=1, le=20, description="Maximum results")


class ClusterRequest(BaseModel):
    """Request to cluster code."""
    num_clusters: int = Field(10, ge=2, le=50, description="Number of clusters")


# =============================================================================
# API Endpoints
# =============================================================================


@router.post(
    "/index",
    summary="Index Code",
    description="Index code units for searching"
)
async def index_code(request: IndexRequest) -> Dict[str, Any]:
    """
    Index code units for searching.

    Provide code snippets to be indexed for semantic and structural search.
    """
    if not SMART_CODE_SEARCH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Smart Code Search is not available"
        )

    search_engine = get_search_engine()
    if not search_engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Smart Code Search"
        )

    # Convert to CodeUnit objects
    code_units = []
    for i, unit_input in enumerate(request.code_units):
        try:
            code_type = CodeType(unit_input.code_type)
        except ValueError:
            code_type = CodeType.SNIPPET

        unit = CodeUnit(
            id=unit_input.id or f"unit_{i}_{hash(unit_input.code) % 10000}",
            code=unit_input.code,
            code_type=code_type,
            name=unit_input.name,
            file_path=unit_input.file_path,
            line_start=unit_input.line_start,
            line_end=unit_input.line_end,
            language=unit_input.language,
            docstring=unit_input.docstring,
            signature=unit_input.signature,
        )
        code_units.append(unit)

    # Index
    if request.generate_embeddings:
        indexed = await search_engine.index_with_embeddings(code_units)
    else:
        indexed = search_engine.index_code(code_units)

    stats = search_engine.get_statistics()

    return {
        "indexed": indexed,
        "total_indexed": stats["indexed_units"],
        "with_embeddings": stats["units_with_embeddings"],
    }


@router.post(
    "/query",
    summary="Search Code",
    description="Search for code matching a query"
)
async def search_code(request: SearchRequest) -> Dict[str, Any]:
    """
    Search for code matching a query.

    Supports semantic (embedding-based), structural (AST-based),
    and hybrid search modes.
    """
    if not SMART_CODE_SEARCH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Smart Code Search is not available"
        )

    search_engine = get_search_engine()
    if not search_engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Smart Code Search"
        )

    # Build query
    try:
        mode = SearchMode(request.mode)
    except ValueError:
        mode = SearchMode.HYBRID

    code_type = None
    if request.code_type:
        try:
            code_type = CodeType(request.code_type)
        except ValueError:
            pass

    query = SearchQuery(
        query=request.query,
        mode=mode,
        language=request.language,
        code_type=code_type,
        file_pattern=request.file_pattern,
        max_results=request.max_results,
        min_similarity=request.min_similarity,
    )

    # Search
    results = await search_engine.search(query)

    return {
        "results": [r.to_dict() for r in results],
        "count": len(results),
        "query": query.to_dict(),
    }


@router.post(
    "/duplicates",
    summary="Find Duplicates",
    description="Find duplicate or near-duplicate code"
)
async def find_duplicates(request: FindDuplicatesRequest) -> Dict[str, Any]:
    """
    Find duplicate or near-duplicate code.

    Returns groups of code that are identical or semantically similar.
    """
    if not SMART_CODE_SEARCH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Smart Code Search is not available"
        )

    search_engine = get_search_engine()
    if not search_engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Smart Code Search"
        )

    groups = search_engine.find_duplicates(request.threshold)

    return {
        "groups": [g.to_dict() for g in groups],
        "count": len(groups),
        "total_duplicates": sum(len(g.code_units) for g in groups),
    }


@router.post(
    "/related",
    summary="Find Related Code",
    description="Find code related to a given snippet"
)
async def find_related(request: FindRelatedRequest) -> Dict[str, Any]:
    """
    Find code related to a given snippet.

    Useful for understanding how similar patterns are used elsewhere.
    """
    if not SMART_CODE_SEARCH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Smart Code Search is not available"
        )

    search_engine = get_search_engine()
    if not search_engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Smart Code Search"
        )

    # Create temporary code unit
    try:
        code_type = CodeType(request.code_type)
    except ValueError:
        code_type = CodeType.SNIPPET

    temp_unit = CodeUnit(
        id=f"temp_{hash(request.code) % 10000}",
        code=request.code,
        code_type=code_type,
    )

    # Index temporarily
    search_engine.index_code([temp_unit])

    # Generate embedding if needed
    await search_engine.index_with_embeddings([temp_unit])

    # Find related
    results = search_engine.find_related(temp_unit, request.max_results)

    return {
        "results": [r.to_dict() for r in results],
        "count": len(results),
    }


@router.post(
    "/cluster",
    summary="Cluster Code",
    description="Cluster code by semantic similarity"
)
async def cluster_code(request: ClusterRequest) -> Dict[str, Any]:
    """
    Cluster code by semantic similarity.

    Groups related code together for analysis and organization.
    """
    if not SMART_CODE_SEARCH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Smart Code Search is not available"
        )

    search_engine = get_search_engine()
    if not search_engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Smart Code Search"
        )

    clusters = search_engine.cluster_code(request.num_clusters)

    return {
        "clusters": [c.to_dict() for c in clusters],
        "count": len(clusters),
    }


@router.get(
    "/stats",
    summary="Get Search Statistics",
    description="Get search engine statistics"
)
async def get_stats() -> Dict[str, Any]:
    """Get search engine statistics."""
    if not SMART_CODE_SEARCH_AVAILABLE:
        return {
            "available": False,
            "message": "Smart Code Search is not available",
        }

    search_engine = get_search_engine()
    if not search_engine:
        return {
            "available": False,
            "message": "Failed to initialize search engine",
        }

    stats = search_engine.get_statistics()
    stats["available"] = True

    return stats


@router.delete(
    "/index",
    summary="Clear Index",
    description="Clear the search index"
)
async def clear_index() -> Dict[str, Any]:
    """Clear the search index."""
    if not SMART_CODE_SEARCH_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Smart Code Search is not available"
        )

    search_engine = get_search_engine()
    if not search_engine:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize Smart Code Search"
        )

    search_engine.clear_index()

    return {
        "cleared": True,
        "message": "Search index cleared",
    }
