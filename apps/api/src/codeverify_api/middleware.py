"""Rate limiting and API authentication middleware."""

import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Rate limit configuration per plan
RATE_LIMITS = {
    "free": {"requests_per_minute": 60, "requests_per_hour": 500},
    "pro": {"requests_per_minute": 300, "requests_per_hour": 5000},
    "enterprise": {"requests_per_minute": 1000, "requests_per_hour": 50000},
}

# In-memory rate limit tracking (use Redis in production)
_request_counts: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"minute": [], "hour": []})


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting API requests."""

    def __init__(self, app, exempt_paths: list[str] | None = None):
        super().__init__(app)
        self.exempt_paths = exempt_paths or ["/health", "/docs", "/openapi.json"]

    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        # Skip rate limiting for exempt paths
        if any(request.url.path.startswith(p) for p in self.exempt_paths):
            return await call_next(request)

        # Only rate limit API endpoints
        if not request.url.path.startswith("/api/v1"):
            return await call_next(request)

        # Get API key from header
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").replace("Bearer ", "")

        if not api_key:
            # Anonymous requests get very limited rate
            client_id = request.client.host if request.client else "unknown"
            plan = "anonymous"
            limits = {"requests_per_minute": 10, "requests_per_hour": 50}
        else:
            # Authenticated requests - lookup plan from key
            client_id = api_key[:16]  # Use key prefix as identifier
            plan = await self._get_plan_for_key(api_key)
            limits = RATE_LIMITS.get(plan, RATE_LIMITS["free"])

        # Check rate limits
        now = time.time()
        counts = _request_counts[client_id]

        # Clean old entries
        minute_ago = now - 60
        hour_ago = now - 3600
        counts["minute"] = [t for t in counts["minute"] if t > minute_ago]
        counts["hour"] = [t for t in counts["hour"] if t > hour_ago]

        # Check limits
        if len(counts["minute"]) >= limits["requests_per_minute"]:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded: {limits['requests_per_minute']} requests per minute",
                    "retry_after": 60 - (now - counts["minute"][0]),
                },
                headers={
                    "X-RateLimit-Limit": str(limits["requests_per_minute"]),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": str(int(60 - (now - counts["minute"][0]))),
                },
            )

        if len(counts["hour"]) >= limits["requests_per_hour"]:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded: {limits['requests_per_hour']} requests per hour",
                    "retry_after": 3600 - (now - counts["hour"][0]),
                },
                headers={
                    "X-RateLimit-Limit": str(limits["requests_per_hour"]),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": str(int(3600 - (now - counts["hour"][0]))),
                },
            )

        # Record request
        counts["minute"].append(now)
        counts["hour"].append(now)

        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limits["requests_per_minute"])
        response.headers["X-RateLimit-Remaining"] = str(limits["requests_per_minute"] - len(counts["minute"]))
        response.headers["X-RateLimit-Reset"] = str(int(minute_ago + 60))

        return response

    async def _get_plan_for_key(self, api_key: str) -> str:
        """Look up the plan for an API key.

        Note: In production, this should query the database. The current
        implementation returns a default plan for development/testing.
        See: https://github.com/codeverify/codeverify/issues/TBD
        """
        import hashlib

        # Hash the key for lookup (keys are stored hashed in DB)
        if api_key.startswith("cv_"):
            api_key = api_key[3:]
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Production: Query database for plan
        # from codeverify_api.db import get_db
        # async with get_db() as db:
        #     result = await db.execute(
        #         "SELECT plan FROM api_keys WHERE key_hash = :hash AND revoked_at IS NULL",
        #         {"hash": key_hash}
        #     )
        #     row = result.fetchone()
        #     return row.plan if row else "free"

        # Development fallback: assume "pro" plan
        return "pro"


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication."""

    def __init__(self, app, required_paths: list[str] | None = None):
        super().__init__(app)
        self.required_paths = required_paths or ["/api/v1"]

    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        # Check if path requires authentication
        requires_auth = any(request.url.path.startswith(p) for p in self.required_paths)

        if not requires_auth:
            return await call_next(request)

        # Allow webhooks without auth (they have their own verification)
        if "/webhooks" in request.url.path and request.method == "POST":
            return await call_next(request)

        # Get API key
        api_key = request.headers.get("X-API-Key")
        auth_header = request.headers.get("Authorization", "")

        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]

        if not api_key:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "missing_api_key",
                    "message": "API key required. Provide via X-API-Key header or Authorization: Bearer <key>",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Validate API key
        key_data = await self._validate_key(api_key)

        if not key_data:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "invalid_api_key",
                    "message": "Invalid or expired API key",
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check scopes for write operations
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            if "write" not in key_data.get("scopes", []) and "admin" not in key_data.get("scopes", []):
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "error": "insufficient_scope",
                        "message": "API key lacks 'write' scope for this operation",
                        "required_scope": "write",
                    },
                )

        # Attach key data to request state
        request.state.api_key = key_data

        return await call_next(request)

    async def _validate_key(self, api_key: str) -> dict[str, Any] | None:
        """Validate an API key and return its data.

        Note: In production, this should query the database. The current
        implementation accepts any key for development/testing.
        See: https://github.com/codeverify/codeverify/issues/TBD
        """
        import hashlib

        # Remove prefix if present
        original_key = api_key
        if api_key.startswith("cv_"):
            api_key = api_key[3:]

        # Hash for lookup (keys are stored hashed in DB)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Production: Query database for key data
        # from codeverify_api.db import get_db
        # async with get_db() as db:
        #     result = await db.execute(
        #         """SELECT id, scopes, plan, org_id FROM api_keys
        #            WHERE key_hash = :hash AND revoked_at IS NULL""",
        #         {"hash": key_hash}
        #     )
        #     row = result.fetchone()
        #     if row:
        #         return {"id": row.id, "scopes": row.scopes, "plan": row.plan, "org_id": row.org_id}
        #     return None

        # Development fallback: accept any properly prefixed key
        if original_key.startswith("cv_") and len(api_key) >= 16:
            return {
                "id": f"dev_{key_hash[:8]}",
                "scopes": ["read", "write"],
                "plan": "pro",
            }

        return None


def get_api_key(request: Request) -> dict[str, Any]:
    """Dependency to get the current API key."""
    if not hasattr(request.state, "api_key"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return request.state.api_key


def require_scope(scope: str) -> Callable:
    """Dependency factory to require a specific scope."""

    def check_scope(request: Request) -> dict[str, Any]:
        key_data = get_api_key(request)
        if scope not in key_data.get("scopes", []) and "admin" not in key_data.get("scopes", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This operation requires the '{scope}' scope",
            )
        return key_data

    return check_scope
