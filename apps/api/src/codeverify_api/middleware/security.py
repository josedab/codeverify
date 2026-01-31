"""Security headers middleware for CodeVerify API."""
from __future__ import annotations

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # XSS protection (legacy browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self' https://api.github.com; "
            "frame-ancestors 'none';"
        )
        
        # Permissions policy
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), "
            "camera=(), "
            "geolocation=(), "
            "gyroscope=(), "
            "magnetometer=(), "
            "microphone=(), "
            "payment=(), "
            "usb=()"
        )
        
        # HSTS (only in production)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )
        
        return response


def setup_security_headers(app: FastAPI) -> None:
    """Configure security headers middleware."""
    app.add_middleware(SecurityHeadersMiddleware)


class InputSanitizationMiddleware(BaseHTTPMiddleware):
    """Sanitize potentially dangerous input."""
    
    # Characters that could be used in injection attacks
    DANGEROUS_PATTERNS = [
        "<script",
        "javascript:",
        "data:text/html",
        "vbscript:",
        "onload=",
        "onerror=",
    ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check query parameters
        for key, value in request.query_params.items():
            if self._is_dangerous(value):
                return Response(
                    content='{"error": "Invalid input detected"}',
                    status_code=400,
                    media_type="application/json"
                )
        
        return await call_next(request)
    
    def _is_dangerous(self, value: str) -> bool:
        """Check if value contains dangerous patterns."""
        value_lower = value.lower()
        return any(pattern in value_lower for pattern in self.DANGEROUS_PATTERNS)


def setup_input_sanitization(app: FastAPI) -> None:
    """Configure input sanitization middleware."""
    app.add_middleware(InputSanitizationMiddleware)
