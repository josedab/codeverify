"""Verification cache management API router."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from codeverify_core.verification_cache import (
    CacheConfig,
    VerificationCache,
)

router = APIRouter()

_cache_instance: VerificationCache | None = None


def get_cache() -> VerificationCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = VerificationCache(CacheConfig())
    return _cache_instance


class CacheStatsResponse(BaseModel):
    hits: int = Field(description="Number of cache hits")
    misses: int = Field(description="Number of cache misses")
    hit_rate: float = Field(description="Cache hit rate (0-1)")
    total_time_saved_ms: float = Field(description="Total verification time saved")
    evictions: int = Field(description="Number of cache evictions")
    size: int = Field(description="Current cache size")
    backend: str = Field(description="Cache backend type")


class InvalidateRequest(BaseModel):
    file_path: str | None = Field(default=None, description="File path to invalidate")
    fingerprint: str | None = Field(default=None, description="Function fingerprint to invalidate")


class InvalidateResponse(BaseModel):
    invalidated: int = Field(description="Number of entries invalidated")


@router.get("/stats", response_model=CacheStatsResponse)
async def get_cache_stats() -> CacheStatsResponse:
    """Get verification cache performance statistics."""
    cache = get_cache()
    stats = cache.get_stats()
    return CacheStatsResponse(**stats)


@router.post("/invalidate", response_model=InvalidateResponse)
async def invalidate_cache(request: InvalidateRequest) -> InvalidateResponse:
    """Invalidate cached verification results."""
    cache = get_cache()

    if request.file_path:
        count = cache.invalidate_file(request.file_path)
        return InvalidateResponse(invalidated=count)
    elif request.fingerprint:
        count = cache.invalidate_function(request.fingerprint)
        return InvalidateResponse(invalidated=count)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide file_path or fingerprint",
        )


@router.post("/clear")
async def clear_cache() -> dict[str, str]:
    """Clear all cached verification results."""
    cache = get_cache()
    cache.clear()
    return {"status": "cleared"}


@router.get("/lookup")
async def lookup_cache(
    fingerprint: str = Query(description="Function fingerprint"),
    verification_type: str = Query(description="Verification type"),
) -> dict[str, Any]:
    """Look up a specific cached verification result."""
    cache = get_cache()
    entry = cache.get(fingerprint, verification_type)
    if entry is None:
        return {"cached": False}
    return {
        "cached": True,
        "function_name": entry.function_name,
        "file_path": entry.file_path,
        "verification_type": entry.verification_type,
        "result": entry.result,
        "created_at": entry.created_at,
        "original_proof_time_ms": entry.original_proof_time_ms,
    }
