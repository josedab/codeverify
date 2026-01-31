"""Notifications API endpoints for Slack and Teams integration."""

from typing import Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

router = APIRouter()


class NotificationConfigRequest(BaseModel):
    """Request to configure notifications."""

    channel: str = Field(..., description="Channel type: slack, teams, email, webhook")
    webhook_url: str = Field(..., description="Webhook URL for the channel")
    enabled: bool = Field(default=True)
    notification_types: list[str] = Field(
        default_factory=list,
        description="Types: analysis_complete, analysis_failed, critical_finding, scan_complete, daily_digest, weekly_report",
    )
    mention_on_critical: bool = Field(default=True)
    mention_users: list[str] = Field(default_factory=list, description="User IDs to mention")
    channel_name: str | None = Field(default=None, description="Channel name override")


class TestNotificationRequest(BaseModel):
    """Request to send a test notification."""

    webhook_url: str
    channel: str = "slack"


class SendAnalysisNotificationRequest(BaseModel):
    """Request to send an analysis notification."""

    repo_full_name: str
    pr_number: int
    pr_title: str
    pr_url: str
    status: str
    total_findings: int
    critical_findings: int = 0
    high_findings: int = 0
    findings_url: str
    author: str


class SendScanNotificationRequest(BaseModel):
    """Request to send a scan notification."""

    repo_full_name: str
    scan_type: str
    status: str
    security_score: float | None = None
    quality_score: float | None = None
    total_findings: int = 0
    critical_findings: int = 0
    scan_url: str


@router.post("/config")
async def create_notification_config(
    repo_full_name: str,
    request: NotificationConfigRequest,
) -> dict[str, Any]:
    """Create a notification configuration for a repository."""
    from codeverify_core.notifications import (
        NotificationConfig,
        NotificationChannel,
        NotificationType,
        save_config_for_repo,
    )

    config = NotificationConfig(
        channel=NotificationChannel(request.channel),
        webhook_url=request.webhook_url,
        enabled=request.enabled,
        notification_types=[NotificationType(t) for t in request.notification_types],
        mention_on_critical=request.mention_on_critical,
        mention_users=request.mention_users,
        channel_name=request.channel_name,
    )

    save_config_for_repo(repo_full_name, config)

    return {
        "status": "created",
        "repo_full_name": repo_full_name,
        "channel": request.channel,
    }


@router.get("/config/{repo_full_name:path}")
async def get_notification_configs(repo_full_name: str) -> dict[str, Any]:
    """Get notification configurations for a repository."""
    from codeverify_core.notifications import get_configs_for_repo

    configs = get_configs_for_repo(repo_full_name)

    return {
        "repo_full_name": repo_full_name,
        "configs": [
            {
                "channel": c.channel.value,
                "enabled": c.enabled,
                "notification_types": [t.value for t in c.notification_types],
                "mention_on_critical": c.mention_on_critical,
            }
            for c in configs
        ],
    }


@router.post("/test")
async def test_notification(request: TestNotificationRequest) -> dict[str, Any]:
    """Send a test notification to verify webhook configuration."""
    from codeverify_core.notifications import (
        NotificationSender,
        NotificationConfig,
        NotificationChannel,
        NotificationType,
        AnalysisNotification,
    )

    sender = NotificationSender()

    # Create a test notification
    test_notification = AnalysisNotification(
        repo_full_name="codeverify/test-repo",
        pr_number=42,
        pr_title="Test PR - CodeVerify Integration",
        pr_url="https://github.com/codeverify/test-repo/pull/42",
        status="passed",
        total_findings=3,
        critical_findings=0,
        high_findings=1,
        findings_url="https://codeverify.io/test",
        author="test-user",
        analyzed_at=datetime.utcnow(),
    )

    # Create a temporary config
    config = NotificationConfig(
        channel=NotificationChannel(request.channel),
        webhook_url=request.webhook_url,
        enabled=True,
        notification_types=[NotificationType.ANALYSIS_COMPLETE],
    )

    results = await sender.send_analysis_notification(test_notification, [config])

    if results and results[0].get("success"):
        return {"status": "success", "message": "Test notification sent successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to send test notification. Check your webhook URL.",
        )


@router.post("/send/analysis")
async def send_analysis_notification(
    request: SendAnalysisNotificationRequest,
) -> dict[str, Any]:
    """Send an analysis notification to configured channels."""
    from codeverify_core.notifications import (
        NotificationSender,
        AnalysisNotification,
        get_configs_for_repo,
    )

    configs = get_configs_for_repo(request.repo_full_name)

    if not configs:
        return {"status": "skipped", "reason": "No notification configs for repository"}

    notification = AnalysisNotification(
        repo_full_name=request.repo_full_name,
        pr_number=request.pr_number,
        pr_title=request.pr_title,
        pr_url=request.pr_url,
        status=request.status,
        total_findings=request.total_findings,
        critical_findings=request.critical_findings,
        high_findings=request.high_findings,
        findings_url=request.findings_url,
        author=request.author,
        analyzed_at=datetime.utcnow(),
    )

    sender = NotificationSender()
    results = await sender.send_analysis_notification(notification, configs)

    return {
        "status": "sent",
        "channels": len(results),
        "results": results,
    }


@router.post("/send/scan")
async def send_scan_notification(
    request: SendScanNotificationRequest,
) -> dict[str, Any]:
    """Send a scan notification to configured channels."""
    from codeverify_core.notifications import (
        NotificationSender,
        ScanNotification,
        get_configs_for_repo,
    )

    configs = get_configs_for_repo(request.repo_full_name)

    if not configs:
        return {"status": "skipped", "reason": "No notification configs for repository"}

    notification = ScanNotification(
        repo_full_name=request.repo_full_name,
        scan_type=request.scan_type,
        status=request.status,
        security_score=request.security_score,
        quality_score=request.quality_score,
        total_findings=request.total_findings,
        critical_findings=request.critical_findings,
        scan_url=request.scan_url,
        scanned_at=datetime.utcnow(),
    )

    sender = NotificationSender()
    results = await sender.send_scan_notification(notification, configs)

    return {
        "status": "sent",
        "channels": len(results),
        "results": results,
    }


@router.get("/templates")
async def get_notification_templates() -> dict[str, Any]:
    """Get example notification payloads for Slack and Teams."""
    return {
        "slack": {
            "analysis": {
                "description": "Example Slack notification for analysis completion",
                "payload": {
                    "attachments": [
                        {
                            "color": "#36a64f",
                            "blocks": [
                                {
                                    "type": "header",
                                    "text": {
                                        "type": "plain_text",
                                        "text": "✅ CodeVerify Analysis Complete",
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
        },
        "teams": {
            "analysis": {
                "description": "Example Teams notification for analysis completion",
                "payload": {
                    "@type": "MessageCard",
                    "@context": "http://schema.org/extensions",
                    "themeColor": "00FF00",
                    "summary": "CodeVerify Analysis Complete",
                }
            }
        }
    }


@router.get("/channels")
async def list_supported_channels() -> dict[str, Any]:
    """List supported notification channels and their configuration requirements."""
    return {
        "channels": [
            {
                "id": "slack",
                "name": "Slack",
                "description": "Send notifications to Slack channels via Incoming Webhooks",
                "setup_url": "https://api.slack.com/messaging/webhooks",
                "required_fields": ["webhook_url"],
                "optional_fields": ["channel_name", "mention_users"],
            },
            {
                "id": "teams",
                "name": "Microsoft Teams",
                "description": "Send notifications to Teams channels via Incoming Webhooks",
                "setup_url": "https://docs.microsoft.com/en-us/microsoftteams/platform/webhooks-and-connectors/how-to/add-incoming-webhook",
                "required_fields": ["webhook_url"],
                "optional_fields": ["mention_users"],
            },
            {
                "id": "email",
                "name": "Email",
                "description": "Send email notifications (requires SMTP configuration)",
                "required_fields": ["email_addresses"],
                "status": "coming_soon",
            },
            {
                "id": "webhook",
                "name": "Custom Webhook",
                "description": "Send raw JSON to any webhook endpoint",
                "required_fields": ["webhook_url"],
                "optional_fields": ["headers", "payload_template"],
            },
        ]
    }
