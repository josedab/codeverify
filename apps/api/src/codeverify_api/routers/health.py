"""Health check endpoints."""

import os
from typing import Any

import redis.asyncio as redis
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from codeverify_api.db.database import get_db
from codeverify_api.config import settings

router = APIRouter()


async def check_database(db: AsyncSession) -> dict[str, Any]:
    """Check database connectivity."""
    try:
        await db.execute(text("SELECT 1"))
        return {"status": "healthy", "latency_ms": 0}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def check_redis() -> dict[str, Any]:
    """Check Redis connectivity."""
    try:
        redis_url = settings.REDIS_URL
        client = redis.from_url(redis_url, decode_responses=True)
        await client.ping()
        await client.close()
        return {"status": "healthy", "latency_ms": 0}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "healthy"}


@router.get("/health/ready")
async def readiness_check(
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Readiness check - verifies all dependencies are available."""
    db_status = await check_database(db)
    redis_status = await check_redis()
    
    all_healthy = (
        db_status["status"] == "healthy" and
        redis_status["status"] == "healthy"
    )
    
    return {
        "status": "ready" if all_healthy else "not_ready",
        "checks": {
            "database": db_status,
            "redis": redis_status,
        },
    }


@router.get("/health/live")
async def liveness_check() -> dict[str, str]:
    """Liveness check - verifies the service is running."""
    return {"status": "alive"}


@router.get("/health/detailed")
async def detailed_health_check(
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Detailed health check with all component statuses."""
    db_status = await check_database(db)
    redis_status = await check_redis()
    
    all_healthy = (
        db_status["status"] == "healthy" and
        redis_status["status"] == "healthy"
    )
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "version": "0.1.0",
        "environment": settings.ENVIRONMENT,
        "checks": {
            "database": db_status,
            "redis": redis_status,
        },
    }
