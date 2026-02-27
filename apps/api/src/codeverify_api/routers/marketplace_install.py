"""GitHub Marketplace one-click install API router.

Provides zero-config onboarding: auto-detect language, generate .codeverify.yml,
activate on PR checks immediately, and track install-to-activation funnel.
"""

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

router = APIRouter()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MarketplaceInstallEvent(BaseModel):
    """Incoming webhook from GitHub Marketplace installation."""
    action: str = Field(description="created, deleted, suspend, unsuspend")
    installation_id: int
    account_login: str
    account_type: str = Field(description="Organization or User")
    plan: str | None = Field(default=None, description="Marketplace plan name")
    sender_login: str


class OnboardingStatus(BaseModel):
    installation_id: int
    account_login: str
    status: str
    steps: list[dict[str, Any]]
    config_generated: bool
    languages_detected: list[str]
    repos_activated: int
    created_at: str


class LanguageDetectionResult(BaseModel):
    repo_full_name: str
    languages: list[dict[str, Any]]
    primary_language: str
    recommended_config: dict[str, Any]


class GeneratedConfig(BaseModel):
    repo_full_name: str
    config_yaml: str
    language: str
    verification_level: str
    auto_fix_enabled: bool


class FunnelMetrics(BaseModel):
    total_installs: int
    completed_onboarding: int
    first_analysis_run: int
    active_after_7_days: int
    conversion_rates: dict[str, float]


# ---------------------------------------------------------------------------
# Language detection patterns
# ---------------------------------------------------------------------------

LANGUAGE_CONFIGS: dict[str, dict[str, Any]] = {
    "python": {
        "file_patterns": ["*.py", "**/*.py"],
        "verification_level": "standard",
        "checks": ["null_safety", "type_safety", "bounds_check", "exception_handling"],
        "auto_fix": True,
    },
    "typescript": {
        "file_patterns": ["*.ts", "*.tsx", "**/*.ts", "**/*.tsx"],
        "verification_level": "standard",
        "checks": ["null_safety", "type_safety", "bounds_check", "async_safety"],
        "auto_fix": True,
    },
    "javascript": {
        "file_patterns": ["*.js", "*.jsx", "**/*.js", "**/*.jsx"],
        "verification_level": "basic",
        "checks": ["null_safety", "type_coercion", "bounds_check"],
        "auto_fix": True,
    },
    "go": {
        "file_patterns": ["*.go", "**/*.go"],
        "verification_level": "standard",
        "checks": ["null_safety", "error_handling", "bounds_check", "concurrency"],
        "auto_fix": False,
    },
    "java": {
        "file_patterns": ["*.java", "**/*.java"],
        "verification_level": "standard",
        "checks": ["null_safety", "type_safety", "bounds_check", "resource_management"],
        "auto_fix": False,
    },
    "rust": {
        "file_patterns": ["*.rs", "**/*.rs"],
        "verification_level": "advanced",
        "checks": ["bounds_check", "integer_overflow", "unsafe_usage"],
        "auto_fix": False,
    },
}


def _generate_codeverify_yml(language: str, repo_full_name: str) -> str:
    """Generate a .codeverify.yml config file for a detected language."""
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS["python"])

    yml = f"""# CodeVerify Configuration
# Auto-generated for {repo_full_name}
# Docs: https://docs.codeverify.dev/configuration

version: "1"

language: {language}
verification_level: {config["verification_level"]}

checks:
"""
    for check in config["checks"]:
        yml += f"  - {check}\n"

    yml += f"""
auto_fix:
  enabled: {str(config["auto_fix"]).lower()}
  require_verification: true
  create_pr: false

ignore:
  - "**/*.test.*"
  - "**/*.spec.*"
  - "**/node_modules/**"
  - "**/__pycache__/**"
  - "**/vendor/**"

pr_checks:
  enabled: true
  block_on_critical: true
  comment_findings: true
  suggest_fixes: true

notifications:
  slack: false
  email: false
"""
    return yml


# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

_installations: dict[int, dict[str, Any]] = {}
_onboarding: dict[int, dict[str, Any]] = {}
_funnel_events: list[dict[str, Any]] = []


def _record_funnel_event(installation_id: int, event: str, metadata: dict[str, Any] | None = None) -> None:
    _funnel_events.append({
        "id": str(uuid.uuid4()),
        "installation_id": installation_id,
        "event": event,
        "metadata": metadata or {},
        "timestamp": datetime.utcnow().isoformat(),
    })


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/install/webhook")
async def handle_marketplace_webhook(event: MarketplaceInstallEvent) -> dict[str, Any]:
    """Handle GitHub Marketplace install/uninstall webhooks."""
    if event.action == "created":
        _installations[event.installation_id] = {
            "installation_id": event.installation_id,
            "account_login": event.account_login,
            "account_type": event.account_type,
            "plan": event.plan,
            "sender_login": event.sender_login,
            "status": "installed",
            "installed_at": datetime.utcnow().isoformat(),
        }

        # Start onboarding pipeline
        _onboarding[event.installation_id] = {
            "installation_id": event.installation_id,
            "account_login": event.account_login,
            "status": "detecting_languages",
            "steps": [
                {"name": "install", "status": "complete", "completed_at": datetime.utcnow().isoformat()},
                {"name": "detect_languages", "status": "pending"},
                {"name": "generate_config", "status": "pending"},
                {"name": "first_analysis", "status": "pending"},
            ],
            "config_generated": False,
            "languages_detected": [],
            "repos_activated": 0,
            "created_at": datetime.utcnow().isoformat(),
        }

        _record_funnel_event(event.installation_id, "installed", {"plan": event.plan})

        return {"status": "onboarding_started", "installation_id": event.installation_id}

    elif event.action == "deleted":
        _installations.pop(event.installation_id, None)
        _onboarding.pop(event.installation_id, None)
        _record_funnel_event(event.installation_id, "uninstalled")
        return {"status": "uninstalled"}

    elif event.action in ("suspend", "unsuspend"):
        inst = _installations.get(event.installation_id)
        if inst:
            inst["status"] = "suspended" if event.action == "suspend" else "active"
        return {"status": event.action}

    return {"status": "ignored", "action": event.action}


@router.post("/install/{installation_id}/detect-languages", response_model=list[LanguageDetectionResult])
async def detect_languages(installation_id: int) -> list[LanguageDetectionResult]:
    """Auto-detect languages for all repositories in an installation."""
    onboarding = _onboarding.get(installation_id)
    if not onboarding:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Installation not found")

    # Simulate language detection across repos
    detected = [
        LanguageDetectionResult(
            repo_full_name=f"{onboarding['account_login']}/api-service",
            languages=[
                {"name": "python", "percentage": 85.0},
                {"name": "dockerfile", "percentage": 10.0},
                {"name": "yaml", "percentage": 5.0},
            ],
            primary_language="python",
            recommended_config=LANGUAGE_CONFIGS["python"],
        ),
        LanguageDetectionResult(
            repo_full_name=f"{onboarding['account_login']}/web-app",
            languages=[
                {"name": "typescript", "percentage": 78.0},
                {"name": "css", "percentage": 15.0},
                {"name": "html", "percentage": 7.0},
            ],
            primary_language="typescript",
            recommended_config=LANGUAGE_CONFIGS["typescript"],
        ),
    ]

    languages_found = list({d.primary_language for d in detected})
    onboarding["languages_detected"] = languages_found
    onboarding["status"] = "languages_detected"

    # Update step
    for step in onboarding["steps"]:
        if step["name"] == "detect_languages":
            step["status"] = "complete"
            step["completed_at"] = datetime.utcnow().isoformat()
            break

    _record_funnel_event(installation_id, "languages_detected", {"languages": languages_found})

    return detected


@router.post("/install/{installation_id}/generate-config", response_model=list[GeneratedConfig])
async def generate_configs(installation_id: int) -> list[GeneratedConfig]:
    """Generate .codeverify.yml configs for all detected repos."""
    onboarding = _onboarding.get(installation_id)
    if not onboarding:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Installation not found")

    if not onboarding.get("languages_detected"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Run language detection first",
        )

    configs = []
    account = onboarding["account_login"]

    # Generate config for each detected language/repo
    repo_languages = [
        (f"{account}/api-service", "python"),
        (f"{account}/web-app", "typescript"),
    ]

    for repo_name, lang in repo_languages:
        yml = _generate_codeverify_yml(lang, repo_name)
        lang_config = LANGUAGE_CONFIGS.get(lang, LANGUAGE_CONFIGS["python"])

        configs.append(GeneratedConfig(
            repo_full_name=repo_name,
            config_yaml=yml,
            language=lang,
            verification_level=lang_config["verification_level"],
            auto_fix_enabled=lang_config["auto_fix"],
        ))

    onboarding["config_generated"] = True
    onboarding["repos_activated"] = len(configs)
    onboarding["status"] = "config_generated"

    for step in onboarding["steps"]:
        if step["name"] == "generate_config":
            step["status"] = "complete"
            step["completed_at"] = datetime.utcnow().isoformat()
            break

    _record_funnel_event(installation_id, "config_generated", {"repos": len(configs)})

    return configs


@router.get("/install/{installation_id}/status", response_model=OnboardingStatus)
async def get_onboarding_status(installation_id: int) -> OnboardingStatus:
    """Get onboarding progress for an installation."""
    onboarding = _onboarding.get(installation_id)
    if not onboarding:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Installation not found")

    return OnboardingStatus(**onboarding)


@router.post("/install/{installation_id}/activate")
async def activate_installation(installation_id: int) -> dict[str, Any]:
    """Mark installation as fully activated (first PR check run)."""
    onboarding = _onboarding.get(installation_id)
    if not onboarding:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Installation not found")

    onboarding["status"] = "active"
    for step in onboarding["steps"]:
        if step["name"] == "first_analysis":
            step["status"] = "complete"
            step["completed_at"] = datetime.utcnow().isoformat()
            break

    _record_funnel_event(installation_id, "first_analysis_run")

    return {
        "status": "activated",
        "installation_id": installation_id,
        "repos_activated": onboarding["repos_activated"],
    }


@router.get("/install/funnel", response_model=FunnelMetrics)
async def get_funnel_metrics() -> FunnelMetrics:
    """Get install-to-activation funnel analytics."""
    events_by_type: dict[str, set[int]] = {}
    for e in _funnel_events:
        events_by_type.setdefault(e["event"], set()).add(e["installation_id"])

    installs = len(events_by_type.get("installed", set()))
    onboarded = len(events_by_type.get("config_generated", set()))
    first_run = len(events_by_type.get("first_analysis_run", set()))

    return FunnelMetrics(
        total_installs=installs,
        completed_onboarding=onboarded,
        first_analysis_run=first_run,
        active_after_7_days=0,  # Requires time-based tracking
        conversion_rates={
            "install_to_onboard": round(onboarded / max(installs, 1) * 100, 1),
            "onboard_to_first_run": round(first_run / max(onboarded, 1) * 100, 1),
            "install_to_first_run": round(first_run / max(installs, 1) * 100, 1),
        },
    )


@router.get("/install/listings")
async def get_marketplace_listing() -> dict[str, Any]:
    """Return the GitHub Marketplace listing metadata."""
    return {
        "name": "CodeVerify",
        "tagline": "AI-Powered Code Review with Formal Verification",
        "description": (
            "CodeVerify automatically reviews your pull requests using AI analysis "
            "and Z3 formal verification. Zero-config setup — just install and go."
        ),
        "categories": ["code-review", "security", "continuous-integration"],
        "pricing": [
            {"plan": "free", "price": 0, "unit": "month", "description": "100 verifications/month for public repos"},
            {"plan": "pro", "price": 29, "unit": "month", "description": "5,000 verifications/month, private repos"},
            {"plan": "enterprise", "price": None, "unit": "custom", "description": "Unlimited, SSO, compliance"},
        ],
        "features": [
            "Zero-config language detection",
            "AI-powered code analysis",
            "Z3 formal verification",
            "Auto-fix with verified patches",
            "Trust score for AI-generated code",
            "Compliance reporting (SOC 2, HIPAA)",
        ],
        "verified_publisher": True,
        "setup_url": "https://app.codeverify.dev/setup",
        "callback_url": "https://api.codeverify.dev/api/v1/marketplace-install/install/webhook",
    }
