"""Public API Platform - REST/GraphQL API for programmatic access."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Header, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from codeverify_api.db.database import get_db

router = APIRouter()


# API Key models

class APIKeyCreate(BaseModel):
    """Request to create an API key."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="")
    scopes: list[str] = Field(
        default_factory=lambda: ["read"],
        description="Scopes: read, write, admin",
    )
    expires_in_days: int | None = Field(default=None, description="Days until expiration (None = never)")


class APIKey(BaseModel):
    """API key response."""

    id: str
    name: str
    key_prefix: str
    scopes: list[str]
    created_at: datetime
    expires_at: datetime | None
    last_used_at: datetime | None = None


class APIKeyWithSecret(APIKey):
    """API key with the secret (only shown once at creation)."""

    key: str


class WebhookCreate(BaseModel):
    """Request to create a webhook."""

    url: str = Field(..., description="Webhook endpoint URL")
    events: list[str] = Field(
        ...,
        description="Events to subscribe to: analysis.completed, analysis.failed, scan.completed, finding.created",
    )
    secret: str | None = Field(default=None, description="Secret for signature verification")
    active: bool = Field(default=True)


class Webhook(BaseModel):
    """Webhook configuration."""

    id: str
    url: str
    events: list[str]
    active: bool
    created_at: datetime
    last_triggered_at: datetime | None = None
    failure_count: int = 0


class WebhookDelivery(BaseModel):
    """A webhook delivery attempt."""

    id: str
    webhook_id: str
    event: str
    payload: dict[str, Any]
    response_status: int | None
    response_body: str | None
    success: bool
    delivered_at: datetime


# In-memory storage (would be database in production)
_api_keys: dict[str, dict[str, Any]] = {}
_webhooks: dict[str, dict[str, Any]] = {}
_webhook_deliveries: list[dict[str, Any]] = []


# API Key management

@router.post("/keys", response_model=APIKeyWithSecret, status_code=status.HTTP_201_CREATED)
async def create_api_key(request: APIKeyCreate) -> APIKeyWithSecret:
    """
    Create a new API key.

    The full key is only returned once at creation time. Store it securely.
    """
    import secrets

    key_id = str(uuid4())
    key_secret = secrets.token_urlsafe(32)
    key_prefix = key_secret[:8]

    now = datetime.utcnow()
    expires_at = None
    if request.expires_in_days:
        from datetime import timedelta
        expires_at = now + timedelta(days=request.expires_in_days)

    key_data = {
        "id": key_id,
        "name": request.name,
        "description": request.description,
        "key_hash": _hash_key(key_secret),
        "key_prefix": key_prefix,
        "scopes": request.scopes,
        "created_at": now,
        "expires_at": expires_at,
        "last_used_at": None,
    }

    _api_keys[key_id] = key_data

    return APIKeyWithSecret(
        id=key_id,
        name=request.name,
        key_prefix=key_prefix,
        key=f"cv_{key_secret}",  # Full key with prefix
        scopes=request.scopes,
        created_at=now,
        expires_at=expires_at,
    )


@router.get("/keys", response_model=list[APIKey])
async def list_api_keys() -> list[APIKey]:
    """List all API keys (without secrets)."""
    return [
        APIKey(
            id=k["id"],
            name=k["name"],
            key_prefix=k["key_prefix"],
            scopes=k["scopes"],
            created_at=k["created_at"],
            expires_at=k.get("expires_at"),
            last_used_at=k.get("last_used_at"),
        )
        for k in _api_keys.values()
    ]


@router.delete("/keys/{key_id}")
async def revoke_api_key(key_id: str) -> dict[str, Any]:
    """Revoke an API key."""
    if key_id not in _api_keys:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key not found: {key_id}",
        )

    del _api_keys[key_id]
    return {"revoked": True, "key_id": key_id}


# Webhook management

@router.post("/webhooks", response_model=Webhook, status_code=status.HTTP_201_CREATED)
async def create_webhook(request: WebhookCreate) -> Webhook:
    """Create a new webhook subscription."""
    webhook_id = str(uuid4())
    now = datetime.utcnow()

    webhook_data = {
        "id": webhook_id,
        "url": request.url,
        "events": request.events,
        "secret": request.secret,
        "active": request.active,
        "created_at": now,
        "last_triggered_at": None,
        "failure_count": 0,
    }

    _webhooks[webhook_id] = webhook_data

    return Webhook(
        id=webhook_id,
        url=request.url,
        events=request.events,
        active=request.active,
        created_at=now,
    )


@router.get("/webhooks", response_model=list[Webhook])
async def list_webhooks() -> list[Webhook]:
    """List all webhooks."""
    return [
        Webhook(
            id=w["id"],
            url=w["url"],
            events=w["events"],
            active=w["active"],
            created_at=w["created_at"],
            last_triggered_at=w.get("last_triggered_at"),
            failure_count=w.get("failure_count", 0),
        )
        for w in _webhooks.values()
    ]


@router.get("/webhooks/{webhook_id}", response_model=Webhook)
async def get_webhook(webhook_id: str) -> Webhook:
    """Get webhook details."""
    if webhook_id not in _webhooks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook not found: {webhook_id}",
        )

    w = _webhooks[webhook_id]
    return Webhook(
        id=w["id"],
        url=w["url"],
        events=w["events"],
        active=w["active"],
        created_at=w["created_at"],
        last_triggered_at=w.get("last_triggered_at"),
        failure_count=w.get("failure_count", 0),
    )


@router.patch("/webhooks/{webhook_id}")
async def update_webhook(
    webhook_id: str,
    url: str | None = None,
    events: list[str] | None = None,
    active: bool | None = None,
) -> Webhook:
    """Update webhook configuration."""
    if webhook_id not in _webhooks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook not found: {webhook_id}",
        )

    w = _webhooks[webhook_id]
    if url is not None:
        w["url"] = url
    if events is not None:
        w["events"] = events
    if active is not None:
        w["active"] = active

    return Webhook(
        id=w["id"],
        url=w["url"],
        events=w["events"],
        active=w["active"],
        created_at=w["created_at"],
        last_triggered_at=w.get("last_triggered_at"),
        failure_count=w.get("failure_count", 0),
    )


@router.delete("/webhooks/{webhook_id}")
async def delete_webhook(webhook_id: str) -> dict[str, Any]:
    """Delete a webhook."""
    if webhook_id not in _webhooks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook not found: {webhook_id}",
        )

    del _webhooks[webhook_id]
    return {"deleted": True, "webhook_id": webhook_id}


@router.post("/webhooks/{webhook_id}/test")
async def test_webhook(webhook_id: str) -> dict[str, Any]:
    """Send a test event to a webhook."""
    if webhook_id not in _webhooks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook not found: {webhook_id}",
        )

    w = _webhooks[webhook_id]

    test_payload = {
        "event": "test",
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "message": "This is a test webhook delivery from CodeVerify",
        },
    }

    result = await _deliver_webhook(w, "test", test_payload)

    return {
        "success": result["success"],
        "response_status": result.get("status"),
        "message": "Test webhook delivered" if result["success"] else "Webhook delivery failed",
    }


@router.get("/webhooks/{webhook_id}/deliveries", response_model=list[WebhookDelivery])
async def get_webhook_deliveries(
    webhook_id: str,
    limit: int = Query(default=20, le=100),
) -> list[WebhookDelivery]:
    """Get recent deliveries for a webhook."""
    if webhook_id not in _webhooks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook not found: {webhook_id}",
        )

    deliveries = [
        d for d in _webhook_deliveries
        if d["webhook_id"] == webhook_id
    ]
    deliveries.sort(key=lambda d: d["delivered_at"], reverse=True)

    return [
        WebhookDelivery(
            id=d["id"],
            webhook_id=d["webhook_id"],
            event=d["event"],
            payload=d["payload"],
            response_status=d.get("response_status"),
            response_body=d.get("response_body"),
            success=d["success"],
            delivered_at=d["delivered_at"],
        )
        for d in deliveries[:limit]
    ]


# Public API endpoints

@router.get("/v1/analyses")
async def api_list_analyses(
    repo: str | None = Query(default=None, description="Filter by repository (full_name)"),
    analysis_status: str | None = Query(default=None, alias="status", description="Filter by status"),
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    List analyses via public API.

    Requires API key authentication.
    """
    from sqlalchemy import select, func
    from codeverify_api.db.models import Analysis, Repository

    query = select(Analysis).join(Repository)
    count_query = select(func.count(Analysis.id)).join(Repository)

    if repo:
        query = query.where(Repository.full_name == repo)
        count_query = count_query.where(Repository.full_name == repo)
    if analysis_status:
        query = query.where(Analysis.status == analysis_status)
        count_query = count_query.where(Analysis.status == analysis_status)

    query = query.order_by(Analysis.created_at.desc()).offset(offset).limit(limit)

    result = await db.execute(query)
    analyses = result.scalars().all()

    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    return {
        "data": [
            {
                "id": str(a.id),
                "repo_id": str(a.repo_id),
                "pr_number": a.pr_number,
                "head_sha": a.head_sha,
                "status": a.status,
                "created_at": a.created_at.isoformat(),
            }
            for a in analyses
        ],
        "pagination": {
            "total": total,
            "limit": limit,
            "offset": offset,
        },
    }


@router.get("/v1/analyses/{analysis_id}")
async def api_get_analysis(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Get analysis details via public API.

    Returns analysis with all findings.
    """
    from codeverify_api.db.repositories import AnalysisRepository

    try:
        analysis_uuid = UUID(analysis_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid analysis ID format",
        )

    repo = AnalysisRepository(db)
    analysis = await repo.get_with_findings(analysis_uuid)

    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis not found: {analysis_id}",
        )

    return {
        "id": str(analysis.id),
        "repo_id": str(analysis.repo_id),
        "pr_number": analysis.pr_number,
        "pr_title": analysis.pr_title,
        "head_sha": analysis.head_sha,
        "base_sha": analysis.base_sha,
        "status": analysis.status,
        "created_at": analysis.created_at.isoformat(),
        "completed_at": analysis.completed_at.isoformat() if analysis.completed_at else None,
        "findings": [
            {
                "id": str(f.id),
                "category": f.category,
                "severity": f.severity,
                "title": f.title,
                "description": f.description,
                "file_path": f.file_path,
                "line_start": f.line_start,
                "line_end": f.line_end,
                "confidence": f.confidence,
            }
            for f in analysis.findings
        ],
        "findings_count": len(analysis.findings),
    }


@router.post("/v1/analyses")
async def api_trigger_analysis(
    repo: str,
    ref: str = "HEAD",
    pr_number: int | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """
    Trigger a new analysis via API.

    Can analyze a commit, branch, or PR.
    """
    from codeverify_api.db.models import Analysis
    from codeverify_api.db.repositories import RepositoryRepository

    # Get repository
    repo_repo = RepositoryRepository(db)
    repository = await repo_repo.get_by_full_name(repo)

    if not repository:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository not found: {repo}",
        )

    # Create pending analysis
    analysis = Analysis(
        repo_id=repository.id,
        pr_number=pr_number or 0,
        head_sha=ref,
        status="queued",
    )
    db.add(analysis)
    await db.flush()
    await db.refresh(analysis)

    # Note: In production, this would queue a Celery task
    # celery_app.send_task("analyze_pr", args=[...])

    return {
        "id": str(analysis.id),
        "status": "queued",
        "repo": repo,
        "ref": ref,
        "pr_number": pr_number,
    }


@router.get("/v1/findings")
async def api_list_findings(
    analysis_id: str | None = None,
    repo: str | None = None,
    severity: str | None = None,
    category: str | None = None,
    limit: int = Query(default=50, le=100),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """List findings via public API."""
    from sqlalchemy import select, func
    from codeverify_api.db.models import Finding, Analysis, Repository

    query = select(Finding).join(Analysis)
    count_query = select(func.count(Finding.id)).join(Analysis)

    if analysis_id:
        try:
            analysis_uuid = UUID(analysis_id)
            query = query.where(Finding.analysis_id == analysis_uuid)
            count_query = count_query.where(Finding.analysis_id == analysis_uuid)
        except ValueError:
            pass

    if repo:
        query = query.join(Repository).where(Repository.full_name == repo)
        count_query = count_query.join(Repository).where(Repository.full_name == repo)

    if severity:
        query = query.where(Finding.severity == severity)
        count_query = count_query.where(Finding.severity == severity)

    if category:
        query = query.where(Finding.category == category)
        count_query = count_query.where(Finding.category == category)

    query = query.order_by(Finding.created_at.desc()).offset(offset).limit(limit)

    result = await db.execute(query)
    findings = result.scalars().all()

    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    return {
        "data": [
            {
                "id": str(f.id),
                "analysis_id": str(f.analysis_id),
                "category": f.category,
                "severity": f.severity,
                "title": f.title,
                "description": f.description,
                "file_path": f.file_path,
                "line_start": f.line_start,
                "line_end": f.line_end,
                "confidence": f.confidence,
                "created_at": f.created_at.isoformat(),
            }
            for f in findings
        ],
        "pagination": {
            "total": total,
            "limit": limit,
            "offset": offset,
        },
    }


@router.get("/v1/stats")
async def api_get_stats(
    repo: str | None = None,
    days: int = Query(default=30, le=90),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Get statistics via public API."""
    from datetime import timedelta
    from sqlalchemy import select, func, and_
    from codeverify_api.db.models import Analysis, Finding, Repository

    cutoff = datetime.utcnow() - timedelta(days=days)

    # Base filters
    analysis_filter = Analysis.created_at >= cutoff
    if repo:
        repo_query = select(Repository.id).where(Repository.full_name == repo)
        analysis_filter = and_(analysis_filter, Analysis.repo_id.in_(repo_query))

    # Total analyses
    total_query = select(func.count(Analysis.id)).where(analysis_filter)
    total_result = await db.execute(total_query)
    total_analyses = total_result.scalar() or 0

    # Passed analyses
    passed_query = select(func.count(Analysis.id)).where(
        and_(analysis_filter, Analysis.status == "completed")
    )
    passed_result = await db.execute(passed_query)
    passed_analyses = passed_result.scalar() or 0

    # Findings by severity
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

    for sev in severity_counts.keys():
        sev_query = (
            select(func.count(Finding.id))
            .join(Analysis)
            .where(and_(analysis_filter, Finding.severity == sev))
        )
        sev_result = await db.execute(sev_query)
        severity_counts[sev] = sev_result.scalar() or 0

    pass_rate = (passed_analyses / total_analyses * 100) if total_analyses > 0 else 0.0

    return {
        "period_days": days,
        "total_analyses": total_analyses,
        "pass_rate": round(pass_rate, 1),
        "findings_by_severity": severity_counts,
    }


# Webhook events

WEBHOOK_EVENTS = {
    "analysis.started": "Triggered when an analysis starts",
    "analysis.completed": "Triggered when an analysis completes successfully",
    "analysis.failed": "Triggered when an analysis fails",
    "finding.created": "Triggered for each new finding",
    "scan.started": "Triggered when a codebase scan starts",
    "scan.completed": "Triggered when a codebase scan completes",
    "security.critical": "Triggered for critical security findings",
}


@router.get("/events")
async def list_webhook_events() -> dict[str, Any]:
    """List available webhook events."""
    return {
        "events": [
            {"name": name, "description": desc}
            for name, desc in WEBHOOK_EVENTS.items()
        ]
    }


# Helper functions

def _hash_key(key: str) -> str:
    """Hash an API key for storage."""
    import hashlib
    return hashlib.sha256(key.encode()).hexdigest()


async def _deliver_webhook(
    webhook: dict[str, Any],
    event: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Deliver a webhook."""
    import httpx
    import hmac
    import hashlib

    delivery_id = str(uuid4())
    now = datetime.utcnow()

    headers = {
        "Content-Type": "application/json",
        "X-CodeVerify-Event": event,
        "X-CodeVerify-Delivery": delivery_id,
    }

    # Add signature if secret is configured
    if webhook.get("secret"):
        import json
        payload_bytes = json.dumps(payload).encode()
        signature = hmac.new(
            webhook["secret"].encode(),
            payload_bytes,
            hashlib.sha256,
        ).hexdigest()
        headers["X-CodeVerify-Signature"] = f"sha256={signature}"

    result = {
        "id": delivery_id,
        "webhook_id": webhook["id"],
        "event": event,
        "payload": payload,
        "delivered_at": now,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                webhook["url"],
                json=payload,
                headers=headers,
                timeout=10.0,
            )
            result["success"] = 200 <= response.status_code < 300
            result["response_status"] = response.status_code
            result["response_body"] = response.text[:500]

    except Exception as e:
        result["success"] = False
        result["response_body"] = str(e)

    _webhook_deliveries.append(result)

    # Update webhook stats
    webhook["last_triggered_at"] = now
    if not result["success"]:
        webhook["failure_count"] = webhook.get("failure_count", 0) + 1

    return result


async def trigger_webhook_event(event: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    """Trigger all webhooks subscribed to an event."""
    results = []

    for webhook in _webhooks.values():
        if not webhook["active"]:
            continue
        if event not in webhook["events"] and "*" not in webhook["events"]:
            continue

        payload = {
            "event": event,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }

        result = await _deliver_webhook(webhook, event, payload)
        results.append(result)

    return results
