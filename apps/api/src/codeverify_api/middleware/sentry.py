"""Sentry error tracking configuration for CodeVerify API."""
from __future__ import annotations

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.httpx import HttpxIntegration
from fastapi import FastAPI, Request
from typing import Any

from codeverify_api.config import settings


def setup_sentry(app: FastAPI) -> None:
    """Configure Sentry error tracking."""
    sentry_dsn = getattr(settings, 'SENTRY_DSN', None)
    
    if not sentry_dsn:
        return
    
    sentry_sdk.init(
        dsn=sentry_dsn,
        environment=settings.ENVIRONMENT,
        release=f"codeverify-api@{getattr(settings, 'VERSION', '0.1.0')}",
        
        # Performance monitoring
        traces_sample_rate=0.1 if settings.ENVIRONMENT == "production" else 1.0,
        profiles_sample_rate=0.1 if settings.ENVIRONMENT == "production" else 1.0,
        
        # Integrations
        integrations=[
            FastApiIntegration(transaction_style="endpoint"),
            SqlalchemyIntegration(),
            RedisIntegration(),
            CeleryIntegration(),
            HttpxIntegration(),
        ],
        
        # Filter sensitive data
        before_send=filter_sensitive_data,
        
        # Ignore certain errors
        ignore_errors=[
            "KeyboardInterrupt",
            "asyncio.CancelledError",
        ],
        
        # Additional options
        send_default_pii=False,
        attach_stacktrace=True,
        max_breadcrumbs=50,
    )


def filter_sensitive_data(event: dict, hint: dict) -> dict | None:
    """Filter sensitive data from Sentry events."""
    # Remove sensitive headers
    if "request" in event and "headers" in event["request"]:
        sensitive_headers = ["authorization", "cookie", "x-api-key"]
        event["request"]["headers"] = {
            k: "[FILTERED]" if k.lower() in sensitive_headers else v
            for k, v in event["request"]["headers"].items()
        }
    
    # Remove sensitive data from request body
    if "request" in event and "data" in event["request"]:
        sensitive_fields = ["password", "token", "secret", "api_key", "private_key"]
        data = event["request"]["data"]
        if isinstance(data, dict):
            event["request"]["data"] = {
                k: "[FILTERED]" if any(sf in k.lower() for sf in sensitive_fields) else v
                for k, v in data.items()
            }
    
    return event


def set_user_context(user_id: str, username: str, email: str | None = None) -> None:
    """Set Sentry user context for better error tracking."""
    sentry_sdk.set_user({
        "id": user_id,
        "username": username,
        "email": email,
    })


def set_analysis_context(
    analysis_id: str,
    repository: str,
    pr_number: int | None = None
) -> None:
    """Set analysis context for debugging."""
    sentry_sdk.set_context("analysis", {
        "analysis_id": analysis_id,
        "repository": repository,
        "pr_number": pr_number,
    })


def capture_analysis_error(
    error: Exception,
    analysis_id: str,
    stage: str,
    extra: dict[str, Any] | None = None
) -> str:
    """Capture an analysis-specific error with context."""
    with sentry_sdk.push_scope() as scope:
        scope.set_tag("analysis_stage", stage)
        scope.set_extra("analysis_id", analysis_id)
        if extra:
            for key, value in extra.items():
                scope.set_extra(key, value)
        
        return sentry_sdk.capture_exception(error)


def add_breadcrumb(
    category: str,
    message: str,
    level: str = "info",
    data: dict[str, Any] | None = None
) -> None:
    """Add a breadcrumb for debugging."""
    sentry_sdk.add_breadcrumb(
        category=category,
        message=message,
        level=level,
        data=data or {},
    )


# Context manager for tracking analysis stages
class AnalysisStageSpan:
    """Context manager for tracking analysis stage performance."""
    
    def __init__(self, stage_name: str, analysis_id: str):
        self.stage_name = stage_name
        self.analysis_id = analysis_id
        self.span = None
    
    def __enter__(self):
        self.span = sentry_sdk.start_span(
            op=f"analysis.{self.stage_name}",
            description=f"Analysis stage: {self.stage_name}"
        )
        self.span.set_tag("analysis_id", self.analysis_id)
        add_breadcrumb(
            category="analysis",
            message=f"Starting stage: {self.stage_name}",
            data={"analysis_id": self.analysis_id}
        )
        return self.span
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.span.set_status("error")
            add_breadcrumb(
                category="analysis",
                message=f"Stage failed: {self.stage_name}",
                level="error",
                data={"error": str(exc_val)}
            )
        else:
            self.span.set_status("ok")
            add_breadcrumb(
                category="analysis",
                message=f"Stage completed: {self.stage_name}",
            )
        self.span.finish()
