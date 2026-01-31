"""Rate limiting middleware for CodeVerify API."""
from __future__ import annotations

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from fastapi import FastAPI, Request

from codeverify_api.config import settings


def get_user_identifier(request: Request) -> str:
    """Get rate limit identifier from request.
    
    Uses authenticated user ID if available, otherwise falls back to IP.
    """
    # Try to get user from auth header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        # Use token hash as identifier for authenticated users
        token = auth_header[7:]
        return f"user:{hash(token) % 1000000}"
    
    # Fall back to IP address
    return get_remote_address(request)


# Create limiter instance
limiter = Limiter(
    key_func=get_user_identifier,
    default_limits=["100/minute"],
    storage_uri=getattr(settings, 'REDIS_URL', None),
)


# Rate limit configurations by tier
RATE_LIMITS = {
    "free": {
        "default": "60/minute",
        "analyses": "50/day",
        "webhooks": "100/minute",
    },
    "team": {
        "default": "300/minute",
        "analyses": "500/day",
        "webhooks": "500/minute",
    },
    "enterprise": {
        "default": "1000/minute",
        "analyses": "10000/day",
        "webhooks": "2000/minute",
    },
}


def setup_rate_limiting(app: FastAPI) -> None:
    """Configure rate limiting for the application."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)


# Decorators for specific endpoints
def rate_limit_default(func):
    """Apply default rate limit."""
    return limiter.limit("100/minute")(func)


def rate_limit_analyses(func):
    """Apply analyses rate limit (stricter)."""
    return limiter.limit("10/minute")(func)


def rate_limit_webhooks(func):
    """Apply webhooks rate limit (more lenient)."""
    return limiter.limit("200/minute")(func)


def rate_limit_auth(func):
    """Apply auth rate limit (strict to prevent brute force)."""
    return limiter.limit("10/minute")(func)
