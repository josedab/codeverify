"""Multi-provider webhook handlers for GitHub, GitLab, and Bitbucket."""

import hashlib
import hmac
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from codeverify_api.config import settings
from codeverify_api.services.analysis_service import AnalysisService
from codeverify_api.db.database import get_db

router = APIRouter()
logger = structlog.get_logger()


def verify_github_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook signature."""
    if not signature or not secret:
        return False
    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def verify_gitlab_token(token: str, secret: str) -> bool:
    """Verify GitLab webhook token."""
    if not token or not secret:
        return False
    return hmac.compare_digest(token, secret)


def verify_bitbucket_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify Bitbucket webhook signature."""
    if not signature or not secret:
        return False
    expected = hmac.new(
        secret.encode("utf-8"), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


@router.post("/github")
async def handle_github_webhook(
    request: Request,
    x_github_event: str = Header(..., alias="X-GitHub-Event"),
    x_hub_signature_256: str = Header(None, alias="X-Hub-Signature-256"),
    x_github_delivery: str = Header(..., alias="X-GitHub-Delivery"),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Handle incoming GitHub webhooks."""
    payload = await request.body()

    # Verify signature in production
    if settings.ENVIRONMENT != "development" and settings.GITHUB_WEBHOOK_SECRET:
        if not verify_github_signature(
            payload, x_hub_signature_256 or "", settings.GITHUB_WEBHOOK_SECRET
        ):
            logger.warning("Invalid webhook signature", delivery_id=x_github_delivery)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid signature",
            )

    data = await request.json()
    logger.info(
        "Received GitHub webhook",
        event=x_github_event,
        delivery_id=x_github_delivery,
        action=data.get("action"),
    )

    # Handle different event types
    if x_github_event == "pull_request":
        return await handle_pull_request_event(data, x_github_delivery, "github")
    elif x_github_event == "installation" or x_github_event == "installation_repositories":
        return await handle_installation_event(data, x_github_delivery, db)
    elif x_github_event == "ping":
        return {"status": "pong", "delivery_id": x_github_delivery}

    return {"status": "ignored", "event": x_github_event}


@router.post("/gitlab")
async def handle_gitlab_webhook(
    request: Request,
    x_gitlab_event: str = Header(None, alias="X-Gitlab-Event"),
    x_gitlab_token: str = Header(None, alias="X-Gitlab-Token"),
) -> dict[str, Any]:
    """Handle incoming GitLab webhooks."""
    payload = await request.body()

    # Verify token in production
    gitlab_secret = getattr(settings, "GITLAB_WEBHOOK_SECRET", None)
    if settings.ENVIRONMENT != "development" and gitlab_secret:
        if not verify_gitlab_token(x_gitlab_token or "", gitlab_secret):
            logger.warning("Invalid GitLab webhook token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )

    data = await request.json()
    event_type = data.get("object_kind", x_gitlab_event or "unknown")

    logger.info(
        "Received GitLab webhook",
        event=event_type,
        action=data.get("object_attributes", {}).get("action"),
    )

    # Handle merge request events
    if event_type == "merge_request":
        return await handle_gitlab_merge_request(data)
    elif event_type == "push":
        return {"status": "ignored", "event": "push"}

    return {"status": "ignored", "event": event_type}


@router.post("/bitbucket")
async def handle_bitbucket_webhook(
    request: Request,
    x_event_key: str = Header(..., alias="X-Event-Key"),
    x_hook_uuid: str = Header(None, alias="X-Hook-UUID"),
    x_request_uuid: str = Header(None, alias="X-Request-UUID"),
) -> dict[str, Any]:
    """Handle incoming Bitbucket webhooks."""
    payload = await request.body()

    # Verify signature in production
    bitbucket_secret = getattr(settings, "BITBUCKET_WEBHOOK_SECRET", None)
    if settings.ENVIRONMENT != "development" and bitbucket_secret:
        # Bitbucket sends signature in X-Hub-Signature header
        signature = request.headers.get("X-Hub-Signature", "")
        if signature.startswith("sha256="):
            signature = signature[7:]
        if not verify_bitbucket_signature(payload, signature, bitbucket_secret):
            logger.warning("Invalid Bitbucket webhook signature")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid signature",
            )

    data = await request.json()
    logger.info(
        "Received Bitbucket webhook",
        event=x_event_key,
        hook_uuid=x_hook_uuid,
    )

    # Handle pull request events
    if x_event_key.startswith("pullrequest:"):
        return await handle_bitbucket_pull_request(data, x_event_key)
    elif x_event_key.startswith("repo:push"):
        return {"status": "ignored", "event": "push"}

    return {"status": "ignored", "event": x_event_key}


async def handle_pull_request_event(
    data: dict[str, Any],
    delivery_id: str,
    provider: str,
) -> dict[str, Any]:
    """Handle GitHub pull request events."""
    action = data.get("action")
    pr = data.get("pull_request", {})
    repo = data.get("repository", {})

    # Only trigger analysis on relevant actions
    if action not in ("opened", "synchronize", "reopened"):
        logger.debug("Ignoring PR action", action=action)
        return {"status": "ignored", "reason": f"action '{action}' not tracked"}

    logger.info(
        "Processing pull request",
        provider=provider,
        action=action,
        pr_number=pr.get("number"),
        repo=repo.get("full_name"),
        head_sha=pr.get("head", {}).get("sha"),
    )

    # Queue analysis job
    analysis_service = AnalysisService()
    job_id = await analysis_service.queue_analysis(
        repo_full_name=repo.get("full_name"),
        repo_id=repo.get("id"),
        pr_number=pr.get("number"),
        pr_title=pr.get("title"),
        head_sha=pr.get("head", {}).get("sha"),
        base_sha=pr.get("base", {}).get("sha"),
        installation_id=data.get("installation", {}).get("id"),
        vcs_provider=provider,
    )

    return {
        "status": "queued",
        "provider": provider,
        "job_id": job_id,
        "pr_number": pr.get("number"),
        "delivery_id": delivery_id,
    }


async def handle_gitlab_merge_request(data: dict[str, Any]) -> dict[str, Any]:
    """Handle GitLab merge request events."""
    mr = data.get("object_attributes", {})
    project = data.get("project", {})
    action = mr.get("action")

    # Map GitLab actions to normalized actions
    action_map = {
        "open": "opened",
        "update": "synchronize",
        "reopen": "reopened",
    }
    normalized_action = action_map.get(action, action)

    if normalized_action not in ("opened", "synchronize", "reopened"):
        logger.debug("Ignoring MR action", action=action)
        return {"status": "ignored", "reason": f"action '{action}' not tracked"}

    logger.info(
        "Processing GitLab merge request",
        action=action,
        mr_iid=mr.get("iid"),
        project=project.get("path_with_namespace"),
    )

    # Queue analysis job
    analysis_service = AnalysisService()
    job_id = await analysis_service.queue_analysis(
        repo_full_name=project.get("path_with_namespace"),
        repo_id=project.get("id"),
        pr_number=mr.get("iid"),
        pr_title=mr.get("title"),
        head_sha=mr.get("last_commit", {}).get("id"),
        base_sha=mr.get("diff_refs", {}).get("base_sha"),
        vcs_provider="gitlab",
    )

    return {
        "status": "queued",
        "provider": "gitlab",
        "job_id": job_id,
        "mr_iid": mr.get("iid"),
    }


async def handle_bitbucket_pull_request(
    data: dict[str, Any],
    event_key: str,
) -> dict[str, Any]:
    """Handle Bitbucket pull request events."""
    pr = data.get("pullrequest", {})
    repo = data.get("repository", {})
    action = event_key.split(":")[1] if ":" in event_key else event_key

    # Map Bitbucket actions to normalized actions
    action_map = {
        "created": "opened",
        "updated": "synchronize",
        "fulfilled": "merged",
        "rejected": "closed",
    }
    normalized_action = action_map.get(action, action)

    if normalized_action not in ("opened", "synchronize"):
        logger.debug("Ignoring PR action", action=action)
        return {"status": "ignored", "reason": f"action '{action}' not tracked"}

    logger.info(
        "Processing Bitbucket pull request",
        action=action,
        pr_id=pr.get("id"),
        repo=repo.get("full_name"),
    )

    # Queue analysis job
    analysis_service = AnalysisService()
    job_id = await analysis_service.queue_analysis(
        repo_full_name=repo.get("full_name"),
        repo_id=repo.get("uuid"),
        pr_number=pr.get("id"),
        pr_title=pr.get("title"),
        head_sha=pr.get("source", {}).get("commit", {}).get("hash"),
        base_sha=pr.get("destination", {}).get("commit", {}).get("hash"),
        vcs_provider="bitbucket",
    )

    return {
        "status": "queued",
        "provider": "bitbucket",
        "job_id": job_id,
        "pr_id": pr.get("id"),
    }


async def handle_installation_event(
    data: dict[str, Any],
    delivery_id: str,
    db: AsyncSession,
) -> dict[str, Any]:
    """Handle GitHub App installation events.
    
    Processes installation created/deleted events to manage
    organizations and repositories in the database.
    """
    from codeverify_api.db.models import Installation, Organization, Repository
    from sqlalchemy import select, delete

    action = data.get("action")
    installation = data.get("installation", {})
    account = installation.get("account", {})
    repositories = data.get("repositories", [])

    installation_id = installation.get("id")
    account_type = account.get("type", "User").lower()
    account_login = account.get("login", "unknown")
    account_id = account.get("id")

    logger.info(
        "Processing installation event",
        action=action,
        installation_id=installation_id,
        account=account_login,
        account_type=account_type,
        repo_count=len(repositories),
    )

    if action == "created":
        # Create or update organization if this is an org installation
        org = None
        if account_type == "organization" and account_id:
            result = await db.execute(
                select(Organization).where(Organization.github_id == account_id)
            )
            org = result.scalar_one_or_none()

            if not org:
                org = Organization(
                    github_id=account_id,
                    name=account.get("name") or account_login,
                    login=account_login,
                    avatar_url=account.get("avatar_url"),
                    settings={},
                )
                db.add(org)
                await db.flush()
                await db.refresh(org)
                logger.info("Created organization", org_id=str(org.id), login=account_login)

        # Create installation record
        result = await db.execute(
            select(Installation).where(Installation.github_installation_id == installation_id)
        )
        existing = result.scalar_one_or_none()

        if not existing:
            new_installation = Installation(
                github_installation_id=installation_id,
                org_id=org.id if org else None,
                account_type=account_type,
                permissions=installation.get("permissions", {}),
                events=installation.get("events", []),
            )
            db.add(new_installation)
            logger.info("Created installation record", installation_id=installation_id)

        # Create repository records
        for repo_data in repositories:
            result = await db.execute(
                select(Repository).where(Repository.github_id == repo_data.get("id"))
            )
            existing_repo = result.scalar_one_or_none()

            if not existing_repo:
                new_repo = Repository(
                    github_id=repo_data.get("id"),
                    org_id=org.id if org else None,
                    name=repo_data.get("name", ""),
                    full_name=repo_data.get("full_name", ""),
                    private=repo_data.get("private", False),
                    enabled=True,
                )
                db.add(new_repo)
                logger.info("Created repository", repo=repo_data.get("full_name"))

        await db.flush()

        return {
            "status": "processed",
            "action": action,
            "installation_id": installation_id,
            "org_created": org is not None,
            "repos_added": len(repositories),
            "delivery_id": delivery_id,
        }

    elif action == "deleted":
        # Mark installation as deleted (soft delete)
        result = await db.execute(
            select(Installation).where(Installation.github_installation_id == installation_id)
        )
        existing = result.scalar_one_or_none()

        if existing:
            # Update suspended_at to mark as deleted
            from datetime import datetime
            existing.suspended_at = datetime.utcnow()
            logger.info("Marked installation as deleted", installation_id=installation_id)

        return {
            "status": "processed",
            "action": action,
            "installation_id": installation_id,
            "delivery_id": delivery_id,
        }

    elif action == "suspend":
        result = await db.execute(
            select(Installation).where(Installation.github_installation_id == installation_id)
        )
        existing = result.scalar_one_or_none()

        if existing:
            from datetime import datetime
            existing.suspended_at = datetime.utcnow()
            logger.info("Suspended installation", installation_id=installation_id)

        return {
            "status": "processed",
            "action": action,
            "installation_id": installation_id,
            "delivery_id": delivery_id,
        }

    elif action == "unsuspend":
        result = await db.execute(
            select(Installation).where(Installation.github_installation_id == installation_id)
        )
        existing = result.scalar_one_or_none()

        if existing:
            existing.suspended_at = None
            logger.info("Unsuspended installation", installation_id=installation_id)

        return {
            "status": "processed",
            "action": action,
            "installation_id": installation_id,
            "delivery_id": delivery_id,
        }

    # Handle repositories added/removed events
    elif action == "added":
        repos_added = data.get("repositories_added", [])
        for repo_data in repos_added:
            result = await db.execute(
                select(Repository).where(Repository.github_id == repo_data.get("id"))
            )
            existing_repo = result.scalar_one_or_none()

            if not existing_repo:
                new_repo = Repository(
                    github_id=repo_data.get("id"),
                    name=repo_data.get("name", ""),
                    full_name=repo_data.get("full_name", ""),
                    private=repo_data.get("private", False),
                    enabled=True,
                )
                db.add(new_repo)
                logger.info("Added repository", repo=repo_data.get("full_name"))

        await db.flush()

        return {
            "status": "processed",
            "action": action,
            "installation_id": installation_id,
            "repos_added": len(repos_added),
            "delivery_id": delivery_id,
        }

    elif action == "removed":
        repos_removed = data.get("repositories_removed", [])
        for repo_data in repos_removed:
            result = await db.execute(
                select(Repository).where(Repository.github_id == repo_data.get("id"))
            )
            existing_repo = result.scalar_one_or_none()

            if existing_repo:
                existing_repo.enabled = False
                logger.info("Disabled repository", repo=repo_data.get("full_name"))

        await db.flush()

        return {
            "status": "processed",
            "action": action,
            "installation_id": installation_id,
            "repos_removed": len(repos_removed),
            "delivery_id": delivery_id,
        }

    return {
        "status": "processed",
        "action": action,
        "installation_id": installation_id,
        "delivery_id": delivery_id,
    }
