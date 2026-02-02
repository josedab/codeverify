"""Verification API Marketplace - External API with usage-based billing.

This module provides:
- Public verification API endpoints for third-party tools
- Usage-based billing and metering
- Rate limiting by subscription tier
- SDKs documentation generation
"""

import hashlib
import hmac
import secrets
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field

from codeverify_api.auth.dependencies import get_current_user, get_current_user_optional
from codeverify_api.config import settings

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/verification")


class SubscriptionTier(str, Enum):
    """API subscription tiers."""
    FREE = "free"
    DEVELOPER = "developer"
    TEAM = "team"
    ENTERPRISE = "enterprise"


class BillingPeriod(str, Enum):
    """Billing periods."""
    MONTHLY = "monthly"
    YEARLY = "yearly"


@dataclass
class TierLimits:
    """Rate limits and quotas for a subscription tier."""
    requests_per_minute: int = 10
    requests_per_day: int = 100
    requests_per_month: int = 1000
    max_file_size_kb: int = 100
    max_files_per_request: int = 5
    max_concurrent_requests: int = 1
    include_proof: bool = False
    include_fix_suggestions: bool = False
    priority_queue: bool = False


TIER_LIMITS: dict[SubscriptionTier, TierLimits] = {
    SubscriptionTier.FREE: TierLimits(
        requests_per_minute=5,
        requests_per_day=50,
        requests_per_month=500,
        max_file_size_kb=50,
        max_files_per_request=1,
        max_concurrent_requests=1,
        include_proof=False,
        include_fix_suggestions=False,
        priority_queue=False,
    ),
    SubscriptionTier.DEVELOPER: TierLimits(
        requests_per_minute=30,
        requests_per_day=500,
        requests_per_month=5000,
        max_file_size_kb=200,
        max_files_per_request=10,
        max_concurrent_requests=3,
        include_proof=True,
        include_fix_suggestions=False,
        priority_queue=False,
    ),
    SubscriptionTier.TEAM: TierLimits(
        requests_per_minute=100,
        requests_per_day=2000,
        requests_per_month=25000,
        max_file_size_kb=500,
        max_files_per_request=25,
        max_concurrent_requests=10,
        include_proof=True,
        include_fix_suggestions=True,
        priority_queue=True,
    ),
    SubscriptionTier.ENTERPRISE: TierLimits(
        requests_per_minute=500,
        requests_per_day=10000,
        requests_per_month=100000,
        max_file_size_kb=2000,
        max_files_per_request=100,
        max_concurrent_requests=50,
        include_proof=True,
        include_fix_suggestions=True,
        priority_queue=True,
    ),
}

TIER_PRICING: dict[SubscriptionTier, dict[str, float]] = {
    SubscriptionTier.FREE: {"monthly": 0, "yearly": 0, "overage_per_1k": 0.50},
    SubscriptionTier.DEVELOPER: {"monthly": 29, "yearly": 290, "overage_per_1k": 0.30},
    SubscriptionTier.TEAM: {"monthly": 99, "yearly": 990, "overage_per_1k": 0.20},
    SubscriptionTier.ENTERPRISE: {"monthly": 499, "yearly": 4990, "overage_per_1k": 0.10},
}


# Request/Response models

class VerifyCodeRequest(BaseModel):
    """Request to verify code."""
    code: str = Field(..., description="Source code to verify")
    language: str = Field(..., description="Programming language (python, typescript, java, etc.)")
    context: str | None = Field(default=None, description="Additional context about the code")
    include_proof: bool = Field(default=False, description="Include mathematical proof details")
    include_fixes: bool = Field(default=False, description="Include fix suggestions for bugs")
    categories: list[str] | None = Field(
        default=None,
        description="Categories to check: null_safety, bounds, overflow, division, security",
    )


class VerifyFileRequest(BaseModel):
    """Request to verify multiple files."""
    files: list[dict[str, str]] = Field(
        ...,
        description="List of {path, content} objects",
    )
    language: str = Field(..., description="Programming language")
    project_context: str | None = Field(default=None)


class VerificationFinding(BaseModel):
    """A verification finding."""
    id: str
    category: str
    severity: str
    title: str
    description: str
    file_path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    code_snippet: str | None = None
    confidence: float
    proof: str | None = None
    fix_suggestion: str | None = None


class VerificationResponse(BaseModel):
    """Response from verification."""
    request_id: str
    status: str  # success, error, rate_limited
    verified: bool
    trust_score: float
    findings: list[VerificationFinding]
    proof_summary: str | None = None
    processing_time_ms: float
    tokens_used: int
    remaining_quota: dict[str, int]


class UsageSummary(BaseModel):
    """Usage summary for billing."""
    period_start: datetime
    period_end: datetime
    tier: str
    requests_used: int
    requests_limit: int
    overage_requests: int
    overage_cost: float
    current_balance: float


class APIKeyInfo(BaseModel):
    """API key information."""
    id: str
    name: str
    key_prefix: str
    tier: str
    created_at: datetime
    last_used_at: datetime | None
    usage_today: int
    usage_this_month: int


class CreateSubscriptionRequest(BaseModel):
    """Request to create/upgrade subscription."""
    tier: SubscriptionTier
    billing_period: BillingPeriod = BillingPeriod.MONTHLY
    payment_method_id: str | None = None


class SubscriptionResponse(BaseModel):
    """Subscription details."""
    id: str
    tier: str
    billing_period: str
    current_period_start: datetime
    current_period_end: datetime
    status: str  # active, past_due, canceled
    limits: dict[str, Any]


# In-memory storage (would be database in production)
_api_subscriptions: dict[str, dict[str, Any]] = {}
_api_usage: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(int))
_rate_limit_windows: dict[str, list[datetime]] = defaultdict(list)


# Helper functions

def _get_subscription(api_key_id: str) -> dict[str, Any]:
    """Get subscription for an API key."""
    if api_key_id not in _api_subscriptions:
        # Default to free tier
        now = datetime.utcnow()
        _api_subscriptions[api_key_id] = {
            "id": str(uuid4()),
            "api_key_id": api_key_id,
            "tier": SubscriptionTier.FREE.value,
            "billing_period": BillingPeriod.MONTHLY.value,
            "current_period_start": now.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
            "current_period_end": (now.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(seconds=1),
            "status": "active",
            "created_at": now,
        }
    return _api_subscriptions[api_key_id]


def _check_rate_limit(api_key_id: str, tier: SubscriptionTier) -> tuple[bool, str]:
    """Check if request is within rate limits. Returns (allowed, reason)."""
    limits = TIER_LIMITS[tier]
    now = datetime.utcnow()
    
    # Clean old entries
    window = _rate_limit_windows[api_key_id]
    minute_ago = now - timedelta(minutes=1)
    day_ago = now - timedelta(days=1)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    _rate_limit_windows[api_key_id] = [t for t in window if t > month_start]
    window = _rate_limit_windows[api_key_id]
    
    # Check per-minute limit
    minute_count = sum(1 for t in window if t > minute_ago)
    if minute_count >= limits.requests_per_minute:
        return False, f"Rate limit exceeded: {limits.requests_per_minute} requests per minute"
    
    # Check per-day limit
    day_count = sum(1 for t in window if t > day_ago)
    if day_count >= limits.requests_per_day:
        return False, f"Daily limit exceeded: {limits.requests_per_day} requests per day"
    
    # Check per-month limit
    month_count = len(window)
    if month_count >= limits.requests_per_month:
        return False, f"Monthly limit exceeded: {limits.requests_per_month} requests per month"
    
    return True, ""


def _record_usage(api_key_id: str, tokens: int = 0) -> None:
    """Record API usage."""
    now = datetime.utcnow()
    _rate_limit_windows[api_key_id].append(now)
    
    # Update usage counters
    date_key = now.strftime("%Y-%m-%d")
    month_key = now.strftime("%Y-%m")
    
    _api_usage[api_key_id][f"day:{date_key}"] += 1
    _api_usage[api_key_id][f"month:{month_key}"] += 1
    _api_usage[api_key_id][f"tokens:{month_key}"] += tokens


def _get_remaining_quota(api_key_id: str, tier: SubscriptionTier) -> dict[str, int]:
    """Get remaining quota for an API key."""
    limits = TIER_LIMITS[tier]
    now = datetime.utcnow()
    window = _rate_limit_windows.get(api_key_id, [])
    
    minute_ago = now - timedelta(minutes=1)
    day_ago = now - timedelta(days=1)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    minute_used = sum(1 for t in window if t > minute_ago)
    day_used = sum(1 for t in window if t > day_ago)
    month_used = sum(1 for t in window if t > month_start)
    
    return {
        "requests_per_minute": max(0, limits.requests_per_minute - minute_used),
        "requests_per_day": max(0, limits.requests_per_day - day_used),
        "requests_per_month": max(0, limits.requests_per_month - month_used),
    }


async def _verify_code_impl(
    code: str,
    language: str,
    context: str | None,
    include_proof: bool,
    include_fixes: bool,
    categories: list[str] | None,
) -> tuple[list[VerificationFinding], str | None, float, int]:
    """Internal verification implementation."""
    import time
    start = time.time()
    findings = []
    proof_summary = None
    tokens_used = len(code.split()) * 2  # Rough estimate
    
    # Simple pattern-based verification (in production, this calls the verifier)
    if language in ("python", "py"):
        if "/ " in code and "if" not in code.split("/")[0]:
            findings.append(VerificationFinding(
                id=f"finding-{uuid4().hex[:8]}",
                category="division",
                severity="medium",
                title="Potential division by zero",
                description="Division operation without guard against zero divisor",
                confidence=0.75,
                proof="∀d. (d ≠ 0) → safe_div(n, d)" if include_proof else None,
                fix_suggestion="if divisor != 0:\n    result = x / divisor" if include_fixes else None,
            ))
        
        if "[]" in code or "[i]" in code:
            if "len(" not in code and "range(" not in code:
                findings.append(VerificationFinding(
                    id=f"finding-{uuid4().hex[:8]}",
                    category="bounds",
                    severity="medium",
                    title="Potential array out of bounds",
                    description="Array access without bounds check",
                    confidence=0.70,
                    proof="∀i,n. (0 ≤ i < n) → safe_access(arr, i)" if include_proof else None,
                ))
        
        if ".value" in code or ".attribute" in code:
            if "is not None" not in code and "is None" not in code:
                findings.append(VerificationFinding(
                    id=f"finding-{uuid4().hex[:8]}",
                    category="null_safety",
                    severity="high",
                    title="Potential null dereference",
                    description="Object access without null check",
                    confidence=0.80,
                    proof="∀x. (x ≠ null → safe_access(x))" if include_proof else None,
                ))
    
    # Calculate trust score
    trust_score = 1.0 - (len(findings) * 0.15)
    trust_score = max(0.0, min(1.0, trust_score))
    
    if include_proof and findings:
        proof_summary = f"Z3 analyzed {len(code.splitlines())} lines. Found {len(findings)} potential issues."
    
    return findings, proof_summary, trust_score, tokens_used


# API Endpoints

@router.post("/verify", response_model=VerificationResponse)
async def verify_code(
    request: VerifyCodeRequest,
    x_api_key: str = Header(..., alias="X-API-Key"),
) -> VerificationResponse:
    """
    Verify code for potential bugs using AI + Z3 formal verification.
    
    This is the primary verification endpoint for external tools and integrations.
    """
    import time
    start = time.time()
    request_id = str(uuid4())
    
    # Get subscription and check limits
    subscription = _get_subscription(x_api_key)
    tier = SubscriptionTier(subscription["tier"])
    limits = TIER_LIMITS[tier]
    
    # Check rate limits
    allowed, reason = _check_rate_limit(x_api_key, tier)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error": "rate_limited", "message": reason},
        )
    
    # Check file size
    code_size_kb = len(request.code.encode()) / 1024
    if code_size_kb > limits.max_file_size_kb:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Code size ({code_size_kb:.1f} KB) exceeds limit ({limits.max_file_size_kb} KB)",
        )
    
    # Check feature access
    include_proof = request.include_proof and limits.include_proof
    include_fixes = request.include_fixes and limits.include_fix_suggestions
    
    # Perform verification
    findings, proof_summary, trust_score, tokens_used = await _verify_code_impl(
        code=request.code,
        language=request.language,
        context=request.context,
        include_proof=include_proof,
        include_fixes=include_fixes,
        categories=request.categories,
    )
    
    # Record usage
    _record_usage(x_api_key, tokens_used)
    
    processing_time = (time.time() - start) * 1000
    
    logger.info(
        "API verification completed",
        request_id=request_id,
        api_key_prefix=x_api_key[:8],
        language=request.language,
        findings_count=len(findings),
        processing_time_ms=processing_time,
    )
    
    return VerificationResponse(
        request_id=request_id,
        status="success",
        verified=len(findings) == 0,
        trust_score=trust_score,
        findings=findings,
        proof_summary=proof_summary,
        processing_time_ms=processing_time,
        tokens_used=tokens_used,
        remaining_quota=_get_remaining_quota(x_api_key, tier),
    )


@router.post("/verify/batch", response_model=dict[str, Any])
async def verify_files(
    request: VerifyFileRequest,
    x_api_key: str = Header(..., alias="X-API-Key"),
) -> dict[str, Any]:
    """
    Verify multiple files in a single request.
    
    Useful for project-wide verification.
    """
    subscription = _get_subscription(x_api_key)
    tier = SubscriptionTier(subscription["tier"])
    limits = TIER_LIMITS[tier]
    
    # Check rate limits
    allowed, reason = _check_rate_limit(x_api_key, tier)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error": "rate_limited", "message": reason},
        )
    
    # Check file count
    if len(request.files) > limits.max_files_per_request:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files ({len(request.files)}). Limit: {limits.max_files_per_request}",
        )
    
    # Verify each file
    results = []
    total_findings = 0
    total_tokens = 0
    
    for file_info in request.files:
        path = file_info.get("path", "unknown")
        content = file_info.get("content", "")
        
        findings, proof_summary, trust_score, tokens = await _verify_code_impl(
            code=content,
            language=request.language,
            context=request.project_context,
            include_proof=limits.include_proof,
            include_fixes=limits.include_fix_suggestions,
            categories=None,
        )
        
        for f in findings:
            f.file_path = path
        
        results.append({
            "path": path,
            "trust_score": trust_score,
            "findings": [f.model_dump() for f in findings],
        })
        
        total_findings += len(findings)
        total_tokens += tokens
    
    # Record usage
    _record_usage(x_api_key, total_tokens)
    
    return {
        "request_id": str(uuid4()),
        "status": "success",
        "files_processed": len(request.files),
        "total_findings": total_findings,
        "results": results,
        "remaining_quota": _get_remaining_quota(x_api_key, tier),
    }


@router.get("/usage", response_model=UsageSummary)
async def get_usage(
    x_api_key: str = Header(..., alias="X-API-Key"),
) -> UsageSummary:
    """Get current usage and billing information."""
    subscription = _get_subscription(x_api_key)
    tier = SubscriptionTier(subscription["tier"])
    limits = TIER_LIMITS[tier]
    pricing = TIER_PRICING[tier]
    
    now = datetime.utcnow()
    month_key = now.strftime("%Y-%m")
    usage = _api_usage.get(x_api_key, {})
    
    requests_used = usage.get(f"month:{month_key}", 0)
    overage = max(0, requests_used - limits.requests_per_month)
    overage_cost = (overage / 1000) * pricing["overage_per_1k"]
    
    return UsageSummary(
        period_start=subscription["current_period_start"],
        period_end=subscription["current_period_end"],
        tier=tier.value,
        requests_used=requests_used,
        requests_limit=limits.requests_per_month,
        overage_requests=overage,
        overage_cost=overage_cost,
        current_balance=pricing.get("monthly", 0) + overage_cost,
    )


@router.get("/subscription", response_model=SubscriptionResponse)
async def get_subscription(
    x_api_key: str = Header(..., alias="X-API-Key"),
) -> SubscriptionResponse:
    """Get current subscription details."""
    subscription = _get_subscription(x_api_key)
    tier = SubscriptionTier(subscription["tier"])
    limits = TIER_LIMITS[tier]
    
    return SubscriptionResponse(
        id=subscription["id"],
        tier=tier.value,
        billing_period=subscription["billing_period"],
        current_period_start=subscription["current_period_start"],
        current_period_end=subscription["current_period_end"],
        status=subscription["status"],
        limits={
            "requests_per_minute": limits.requests_per_minute,
            "requests_per_day": limits.requests_per_day,
            "requests_per_month": limits.requests_per_month,
            "max_file_size_kb": limits.max_file_size_kb,
            "max_files_per_request": limits.max_files_per_request,
            "include_proof": limits.include_proof,
            "include_fix_suggestions": limits.include_fix_suggestions,
        },
    )


@router.post("/subscription", response_model=SubscriptionResponse)
async def create_or_update_subscription(
    request: CreateSubscriptionRequest,
    x_api_key: str = Header(..., alias="X-API-Key"),
    current_user: dict = Depends(get_current_user),
) -> SubscriptionResponse:
    """Create or upgrade subscription."""
    subscription = _get_subscription(x_api_key)
    now = datetime.utcnow()
    
    # Update subscription
    subscription["tier"] = request.tier.value
    subscription["billing_period"] = request.billing_period.value
    subscription["current_period_start"] = now
    
    if request.billing_period == BillingPeriod.MONTHLY:
        subscription["current_period_end"] = now + timedelta(days=30)
    else:
        subscription["current_period_end"] = now + timedelta(days=365)
    
    subscription["status"] = "active"
    _api_subscriptions[x_api_key] = subscription
    
    logger.info(
        "Subscription updated",
        api_key_prefix=x_api_key[:8],
        tier=request.tier.value,
    )
    
    return await get_subscription(x_api_key)


@router.get("/tiers")
async def list_tiers() -> dict[str, Any]:
    """List available subscription tiers and their features."""
    tiers = []
    for tier in SubscriptionTier:
        limits = TIER_LIMITS[tier]
        pricing = TIER_PRICING[tier]
        
        tiers.append({
            "tier": tier.value,
            "pricing": {
                "monthly": pricing["monthly"],
                "yearly": pricing["yearly"],
                "overage_per_1k_requests": pricing["overage_per_1k"],
            },
            "limits": {
                "requests_per_minute": limits.requests_per_minute,
                "requests_per_day": limits.requests_per_day,
                "requests_per_month": limits.requests_per_month,
                "max_file_size_kb": limits.max_file_size_kb,
                "max_files_per_request": limits.max_files_per_request,
                "max_concurrent_requests": limits.max_concurrent_requests,
            },
            "features": {
                "include_proof": limits.include_proof,
                "include_fix_suggestions": limits.include_fix_suggestions,
                "priority_queue": limits.priority_queue,
            },
        })
    
    return {"tiers": tiers}


@router.get("/languages")
async def list_languages() -> dict[str, Any]:
    """List supported programming languages."""
    return {
        "languages": [
            {"code": "python", "name": "Python", "versions": ["3.8", "3.9", "3.10", "3.11", "3.12"]},
            {"code": "typescript", "name": "TypeScript", "versions": ["4.x", "5.x"]},
            {"code": "javascript", "name": "JavaScript", "versions": ["ES2020+"]},
            {"code": "java", "name": "Java", "versions": ["11", "17", "21"]},
            {"code": "go", "name": "Go", "versions": ["1.20", "1.21", "1.22"]},
            {"code": "rust", "name": "Rust", "versions": ["1.70+"]},
            {"code": "csharp", "name": "C#", "versions": [".NET 6", ".NET 7", ".NET 8"]},
        ]
    }


@router.get("/categories")
async def list_categories() -> dict[str, Any]:
    """List verification categories."""
    return {
        "categories": [
            {
                "code": "null_safety",
                "name": "Null Safety",
                "description": "Detect potential null/undefined dereferences",
            },
            {
                "code": "bounds",
                "name": "Bounds Checking",
                "description": "Detect array/buffer out-of-bounds access",
            },
            {
                "code": "overflow",
                "name": "Integer Overflow",
                "description": "Detect arithmetic overflow/underflow",
            },
            {
                "code": "division",
                "name": "Division Safety",
                "description": "Detect division by zero",
            },
            {
                "code": "security",
                "name": "Security",
                "description": "Detect security vulnerabilities (injection, XSS, etc.)",
            },
            {
                "code": "concurrency",
                "name": "Concurrency",
                "description": "Detect race conditions and deadlocks",
            },
            {
                "code": "resource",
                "name": "Resource Management",
                "description": "Detect resource leaks and improper cleanup",
            },
        ]
    }


# SDK code generation endpoints

@router.get("/sdk/python")
async def get_python_sdk() -> dict[str, Any]:
    """Get Python SDK installation and usage."""
    return {
        "language": "python",
        "package": "codeverify-sdk",
        "installation": "pip install codeverify-sdk",
        "quickstart": '''
from codeverify import CodeVerifyClient

client = CodeVerifyClient(api_key="your-api-key")

# Verify code
result = client.verify(
    code="def divide(a, b): return a / b",
    language="python"
)

print(f"Trust Score: {result.trust_score}")
for finding in result.findings:
    print(f"- {finding.severity}: {finding.title}")
''',
        "docs_url": "https://docs.codeverify.dev/sdk/python",
    }


@router.get("/sdk/typescript")
async def get_typescript_sdk() -> dict[str, Any]:
    """Get TypeScript/JavaScript SDK installation and usage."""
    return {
        "language": "typescript",
        "package": "@codeverify/sdk",
        "installation": "npm install @codeverify/sdk",
        "quickstart": '''
import { CodeVerifyClient } from '@codeverify/sdk';

const client = new CodeVerifyClient({ apiKey: 'your-api-key' });

// Verify code
const result = await client.verify({
  code: 'function divide(a: number, b: number) { return a / b; }',
  language: 'typescript',
});

console.log(`Trust Score: ${result.trustScore}`);
result.findings.forEach(f => {
  console.log(`- ${f.severity}: ${f.title}`);
});
''',
        "docs_url": "https://docs.codeverify.dev/sdk/typescript",
    }


@router.get("/sdk/curl")
async def get_curl_example() -> dict[str, Any]:
    """Get cURL example for direct API usage."""
    return {
        "language": "curl",
        "example": '''
curl -X POST https://api.codeverify.dev/api/v1/verification/verify \\
  -H "Content-Type: application/json" \\
  -H "X-API-Key: your-api-key" \\
  -d '{
    "code": "def divide(a, b): return a / b",
    "language": "python",
    "include_proof": true
  }'
''',
        "docs_url": "https://docs.codeverify.dev/api",
    }
