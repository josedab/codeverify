"""CodeVerify API - Main Application"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi import Request
from fastapi.responses import JSONResponse

from codeverify_api.config import settings
from codeverify_api.routers import analyses, auth, health, organizations, repositories, stats, webhooks
from codeverify_api.routers import feedback, usage, export, sso, trust_score, debugger, rules, diff_summarizer, scanning, notifications, public_api, internal, badges, marketplace, continuous_verification
from codeverify_api.routers import collaboration, formal_specs, cross_repo, regression, replay, nl_queries, network
# Next-gen feature routers (v0.3.0)
from codeverify_api.routers import threat_modeling, risk_prediction, consensus, compliance, cost_optimization, cross_language
from codeverify_api.middleware.rate_limit import setup_rate_limiting
from codeverify_api.middleware.security import setup_security_headers
from codeverify_api.middleware.metrics import setup_metrics
from codeverify_api.middleware.sentry import setup_sentry

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

# Next-gen features (v0.3.0)
app.include_router(threat_modeling.router, prefix="/api/v1/threat-model", tags=["threat-modeling"])
app.include_router(risk_prediction.router, prefix="/api/v1/risk-prediction", tags=["risk-prediction"])
app.include_router(consensus.router, prefix="/api/v1/consensus", tags=["consensus-verification"])
app.include_router(compliance.router, prefix="/api/v1/compliance", tags=["compliance-attestation"])
app.include_router(cost_optimization.router, prefix="/api/v1/cost", tags=["cost-optimization"])
app.include_router(cross_language.router, prefix="/api/v1/cross-language", tags=["cross-language"])


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
