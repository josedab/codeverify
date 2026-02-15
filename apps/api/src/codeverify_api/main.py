"""CodeVerify API - Main Application"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from codeverify_api.config import settings
from codeverify_api.middleware.metrics import setup_metrics
from codeverify_api.middleware.rate_limit import setup_rate_limiting
from codeverify_api.middleware.security import setup_security_headers
from codeverify_api.middleware.sentry import setup_sentry

# Next-gen feature routers (v0.3.0)
# Next-gen planning features (v0.4.0)
from codeverify_api.routers import (
    ai_drift,
    analyses,
    audit_logs,
    auth,
    autofix,
    badges,
    code_evolution,
    code_search,
    collaboration,
    compliance,
    consensus,
    context_analysis,
    continuous_learning,
    continuous_verification,
    copilot,
    cost_optimization,
    cross_language,
    cross_repo,
    debugger,
    dependency_scanner,
    diff_summarizer,
    export,
    feedback,
    fix_verification,
    formal_specs,
    hallucination,
    health,
    impact_analysis,
    internal,
    language_support,
    marketplace,
    network,
    nl_queries,
    notifications,
    organizations,
    paste_interception,
    policy_engine,
    public_api,
    regression,
    replay,
    repositories,
    risk_prediction,
    rules,
    scanning,
    sso,
    stats,
    streaming,
    telemetry,
    tenancy,
    threat_modeling,
    trust_score,
    usage,
    verification_api,
    verification_cache,
    verification_debugger,
    webhooks,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("Starting CodeVerify API", environment=settings.ENVIRONMENT)
    yield
    logger.info("Shutting down CodeVerify API")


app = FastAPI(
    title="CodeVerify API",
    description="AI-powered code review with formal verification",
    version="0.1.0",
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
    lifespan=lifespan,
)

# Setup middleware
setup_sentry(app)
setup_security_headers(app)
setup_rate_limiting(app)
setup_metrics(app)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(webhooks.router, prefix="/webhooks", tags=["webhooks"])
app.include_router(organizations.router, prefix="/api/v1/organizations", tags=["organizations"])
app.include_router(repositories.router, prefix="/api/v1/repositories", tags=["repositories"])
app.include_router(analyses.router, prefix="/api/v1/analyses", tags=["analyses"])
app.include_router(stats.router, prefix="/api/v1/stats", tags=["stats"])
app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["feedback"])
app.include_router(usage.router, prefix="/api/v1/usage", tags=["usage"])
app.include_router(export.router, prefix="/api/v1/export", tags=["export"])
app.include_router(sso.router, prefix="/api/v1/sso", tags=["sso"])
app.include_router(trust_score.router, prefix="/api/v1/trust-score", tags=["trust-score"])
app.include_router(debugger.router, prefix="/api/v1/debugger", tags=["debugger"])
app.include_router(rules.router, prefix="/api/v1/rules", tags=["rules"])
app.include_router(diff_summarizer.router, prefix="/api/v1/diff", tags=["diff-summarizer"])
app.include_router(scanning.router, prefix="/api/v1/scans", tags=["scanning"])
app.include_router(notifications.router, prefix="/api/v1/notifications", tags=["notifications"])
app.include_router(public_api.router, prefix="/api", tags=["public-api"])
app.include_router(internal.router, prefix="/internal", tags=["internal"])
app.include_router(badges.router, prefix="/api/v1/badges", tags=["badges"])
app.include_router(marketplace.router, prefix="/api/v1/marketplace", tags=["marketplace"])
app.include_router(continuous_verification.router, tags=["continuous-verification"])
app.include_router(collaboration.router, tags=["ai-collaboration"])
app.include_router(formal_specs.router, tags=["formal-specs"])
app.include_router(cross_repo.router, tags=["cross-repo"])
app.include_router(regression.router, tags=["regression-learning"])
app.include_router(replay.router, tags=["verification-replay"])
app.include_router(nl_queries.router, tags=["nl-queries"])
app.include_router(network.router, tags=["distributed-network"])
app.include_router(audit_logs.router, prefix="/api/v1", tags=["audit-logs"])

# Next-gen features (v0.3.0)
app.include_router(threat_modeling.router, prefix="/api/v1/threat-model", tags=["threat-modeling"])
app.include_router(
    risk_prediction.router, prefix="/api/v1/risk-prediction", tags=["risk-prediction"]
)
app.include_router(consensus.router, prefix="/api/v1/consensus", tags=["consensus-verification"])
app.include_router(compliance.router, prefix="/api/v1/compliance", tags=["compliance-attestation"])
app.include_router(cost_optimization.router, prefix="/api/v1/cost", tags=["cost-optimization"])
app.include_router(cross_language.router, prefix="/api/v1/cross-language", tags=["cross-language"])

# Public Verification API Marketplace
app.include_router(verification_api.router, tags=["verification-api"])

# Paste Interception (Real-Time AI Code Detection)
app.include_router(
    paste_interception.router, prefix="/api/v1/analyses", tags=["paste-interception"]
)

# AI Drift Detection
app.include_router(ai_drift.router, tags=["ai-drift"])

# Verification Debugger
app.include_router(verification_debugger.router, tags=["verification-debugger"])

# Smart Code Search
app.include_router(code_search.router, tags=["code-search"])

# Continuous Learning Engine
app.include_router(continuous_learning.router, tags=["continuous-learning"])

# Context-Aware Analysis
app.include_router(context_analysis.router, tags=["context-analysis"])

# Code Evolution Tracker
app.include_router(code_evolution.router, tags=["code-evolution"])

# Fix Verification Engine
app.include_router(fix_verification.router, tags=["fix-verification"])

# Dependency Vulnerability Scanner
app.include_router(dependency_scanner.router, tags=["dependency-scanner"])

# Next-gen planning features (v0.4.0)
app.include_router(verification_cache.router, prefix="/api/v1/cache", tags=["verification-cache"])
app.include_router(tenancy.router, prefix="/api/v1/tenants", tags=["multi-tenancy"])
app.include_router(language_support.router, prefix="/api/v1/languages", tags=["language-support"])
app.include_router(streaming.router, prefix="/api/v1/streaming", tags=["streaming-verification"])
app.include_router(autofix.router, prefix="/api/v1/autofix", tags=["auto-fix"])
app.include_router(
    hallucination.router, prefix="/api/v1/hallucination", tags=["hallucination-detection"]
)
app.include_router(telemetry.router, prefix="/api/v1/telemetry", tags=["telemetry-roi"])
app.include_router(policy_engine.router, prefix="/api/v1/policies", tags=["policy-engine"])
app.include_router(impact_analysis.router, prefix="/api/v1/impact", tags=["impact-analysis"])
app.include_router(copilot.router, prefix="/api/v1/copilot", tags=["copilot-extension"])


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        path=request.url.path,
        method=request.method,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "An internal error occurred",
            }
        },
    )


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "service": "CodeVerify API",
        "version": "0.1.0",
        "status": "running",
    }
