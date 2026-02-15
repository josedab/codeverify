"""Fix Approval Workflow — Human-in-loop approval for autonomous fixes.

Provides multi-channel notification (Slack, Teams, GitHub) for fix approval,
enabling human oversight of AI-generated code fixes before they are merged.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class NotificationChannel(str, Enum):
    """Supported notification channels."""

    SLACK = "slack"
    TEAMS = "teams"
    GITHUB = "github"
    EMAIL = "email"
    WEBHOOK = "webhook"


class ApprovalStatus(str, Enum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ApprovalUrgency(str, Enum):
    """Urgency level for approval requests."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ApprovalPolicy:
    """Defines who can approve and under what conditions."""

    required_approvers: int = 1
    allowed_approver_roles: list[str] = field(
        default_factory=lambda: ["maintainer", "admin"]
    )
    auto_approve_confidence_threshold: float = 0.95
    expiry_hours: int = 72
    channels: list[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.GITHUB]
    )
    escalation_hours: int = 24
    escalation_channels: list[NotificationChannel] = field(
        default_factory=lambda: [NotificationChannel.SLACK]
    )


@dataclass
class ApprovalRequest:
    """A request for human approval of an autonomous fix."""

    id: str
    fix_id: str
    pr_url: str | None
    repository: str
    title: str
    description: str
    diff_summary: str
    confidence: float
    severity: str
    urgency: ApprovalUrgency
    policy: ApprovalPolicy
    status: ApprovalStatus = ApprovalStatus.PENDING
    approvers: list[dict[str, Any]] = field(default_factory=list)
    notifications_sent: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    resolved_at: datetime | None = None
    resolution_comment: str | None = None

    @property
    def is_resolved(self) -> bool:
        return self.status in (
            ApprovalStatus.APPROVED,
            ApprovalStatus.REJECTED,
            ApprovalStatus.EXPIRED,
        )

    @property
    def approval_count(self) -> int:
        return sum(1 for a in self.approvers if a.get("decision") == "approved")

    @property
    def has_sufficient_approvals(self) -> bool:
        return self.approval_count >= self.policy.required_approvers

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "fix_id": self.fix_id,
            "pr_url": self.pr_url,
            "repository": self.repository,
            "title": self.title,
            "confidence": self.confidence,
            "severity": self.severity,
            "urgency": self.urgency.value,
            "status": self.status.value,
            "approvers": self.approvers,
            "approval_count": self.approval_count,
            "required_approvals": self.policy.required_approvers,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class NotificationPayload:
    """Payload for sending notifications across channels."""

    channel: NotificationChannel
    recipient: str
    subject: str
    body: str
    metadata: dict[str, Any] = field(default_factory=dict)
    action_url: str | None = None
    sent_at: datetime | None = None
    delivery_status: str = "pending"


class ChannelAdapter:
    """Base adapter for notification channels."""

    async def send(self, payload: NotificationPayload) -> bool:
        raise NotImplementedError


class SlackAdapter(ChannelAdapter):
    """Sends approval notifications to Slack via webhook."""

    def __init__(self, webhook_url: str = "", token: str = "") -> None:
        self.webhook_url = webhook_url
        self.token = token

    async def send(self, payload: NotificationPayload) -> bool:
        import httpx

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"🔧 {payload.subject}"},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": payload.body},
            },
        ]
        if payload.action_url:
            blocks.append(
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Review Fix"},
                            "url": payload.action_url,
                            "style": "primary",
                        }
                    ],
                }
            )

        message = {"channel": payload.recipient, "blocks": blocks}
        try:
            async with httpx.AsyncClient() as client:
                if self.webhook_url:
                    resp = await client.post(self.webhook_url, json=message)
                else:
                    resp = await client.post(
                        "https://slack.com/api/chat.postMessage",
                        json=message,
                        headers={"Authorization": f"Bearer {self.token}"},
                    )
                payload.delivery_status = (
                    "delivered" if resp.status_code == 200 else "failed"
                )
                return resp.status_code == 200
        except Exception as e:
            logger.error("Slack notification failed", error=str(e))
            payload.delivery_status = "failed"
            return False


class TeamsAdapter(ChannelAdapter):
    """Sends approval notifications to Microsoft Teams via webhook."""

    def __init__(self, webhook_url: str = "") -> None:
        self.webhook_url = webhook_url

    async def send(self, payload: NotificationPayload) -> bool:
        import httpx

        card = {
            "@type": "MessageCard",
            "summary": payload.subject,
            "themeColor": "0076D7",
            "title": f"🔧 {payload.subject}",
            "sections": [{"text": payload.body}],
        }
        if payload.action_url:
            card["potentialAction"] = [
                {
                    "@type": "OpenUri",
                    "name": "Review Fix",
                    "targets": [{"os": "default", "uri": payload.action_url}],
                }
            ]
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(self.webhook_url, json=card)
                payload.delivery_status = (
                    "delivered" if resp.status_code == 200 else "failed"
                )
                return resp.status_code == 200
        except Exception as e:
            logger.error("Teams notification failed", error=str(e))
            payload.delivery_status = "failed"
            return False


class GitHubAdapter(ChannelAdapter):
    """Creates GitHub PR review requests and issue comments."""

    def __init__(self, token: str = "") -> None:
        self.token = token

    async def send(self, payload: NotificationPayload) -> bool:
        import httpx

        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }
        comment_body = f"## 🔧 Fix Approval Required\n\n{payload.body}"
        if payload.action_url:
            comment_body += f"\n\n[Review this fix]({payload.action_url})"

        try:
            async with httpx.AsyncClient() as client:
                url = payload.metadata.get("comment_url", "")
                if url:
                    resp = await client.post(
                        url,
                        json={"body": comment_body},
                        headers=headers,
                    )
                    payload.delivery_status = (
                        "delivered" if resp.status_code in (200, 201) else "failed"
                    )
                    return resp.status_code in (200, 201)
                payload.delivery_status = "skipped"
                return True
        except Exception as e:
            logger.error("GitHub notification failed", error=str(e))
            payload.delivery_status = "failed"
            return False


class WebhookAdapter(ChannelAdapter):
    """Sends approval notifications to a generic webhook URL."""

    def __init__(self, url: str = "") -> None:
        self.url = url

    async def send(self, payload: NotificationPayload) -> bool:
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.url,
                    json={
                        "subject": payload.subject,
                        "body": payload.body,
                        "action_url": payload.action_url,
                        "metadata": payload.metadata,
                    },
                )
                payload.delivery_status = (
                    "delivered" if resp.status_code < 300 else "failed"
                )
                return resp.status_code < 300
        except Exception as e:
            logger.error("Webhook notification failed", error=str(e))
            payload.delivery_status = "failed"
            return False


class ApprovalNotifier:
    """Orchestrates multi-channel notifications for fix approval workflow.

    Manages the lifecycle of approval requests: creation, notification,
    escalation, and resolution tracking.

    Example:
        >>> notifier = ApprovalNotifier(adapters={
        ...     NotificationChannel.SLACK: SlackAdapter(webhook_url="..."),
        ...     NotificationChannel.GITHUB: GitHubAdapter(token="..."),
        ... })
        >>> request = notifier.create_request(fix_id="fix-123", ...)
        >>> await notifier.notify(request)
        >>> notifier.record_decision(request.id, "user@co.com", "approved")
    """

    def __init__(
        self,
        adapters: dict[NotificationChannel, ChannelAdapter] | None = None,
        default_policy: ApprovalPolicy | None = None,
    ) -> None:
        self._adapters = adapters or {}
        self._default_policy = default_policy or ApprovalPolicy()
        self._requests: dict[str, ApprovalRequest] = {}

    def create_request(
        self,
        fix_id: str,
        repository: str,
        title: str,
        description: str,
        diff_summary: str,
        confidence: float,
        severity: str = "medium",
        pr_url: str | None = None,
        policy: ApprovalPolicy | None = None,
    ) -> ApprovalRequest:
        """Create a new approval request for a fix."""
        import uuid

        effective_policy = policy or self._default_policy

        urgency = ApprovalUrgency.LOW
        if severity == "critical":
            urgency = ApprovalUrgency.CRITICAL
        elif severity == "high":
            urgency = ApprovalUrgency.HIGH
        elif severity == "medium":
            urgency = ApprovalUrgency.MEDIUM

        request = ApprovalRequest(
            id=str(uuid.uuid4()),
            fix_id=fix_id,
            pr_url=pr_url,
            repository=repository,
            title=title,
            description=description,
            diff_summary=diff_summary,
            confidence=confidence,
            severity=severity,
            urgency=urgency,
            policy=effective_policy,
        )

        # Auto-approve if confidence exceeds threshold
        if confidence >= effective_policy.auto_approve_confidence_threshold:
            request.status = ApprovalStatus.APPROVED
            request.resolved_at = datetime.now(timezone.utc)
            request.resolution_comment = "Auto-approved: confidence above threshold"
            logger.info(
                "Fix auto-approved",
                fix_id=fix_id,
                confidence=confidence,
                threshold=effective_policy.auto_approve_confidence_threshold,
            )

        self._requests[request.id] = request
        return request

    async def notify(self, request: ApprovalRequest) -> list[NotificationPayload]:
        """Send notifications for an approval request across configured channels."""
        if request.is_resolved:
            return []

        payloads = []
        for channel in request.policy.channels:
            adapter = self._adapters.get(channel)
            if not adapter:
                logger.warning("No adapter for channel", channel=channel.value)
                continue

            payload = NotificationPayload(
                channel=channel,
                recipient=request.repository,
                subject=f"Fix Approval: {request.title}",
                body=self._format_body(request),
                action_url=request.pr_url,
                metadata={"request_id": request.id, "fix_id": request.fix_id},
            )
            success = await adapter.send(payload)
            payload.sent_at = datetime.now(timezone.utc)
            payloads.append(payload)

            request.notifications_sent.append(
                {
                    "channel": channel.value,
                    "sent_at": payload.sent_at.isoformat(),
                    "success": success,
                }
            )

        return payloads

    async def escalate(self, request: ApprovalRequest) -> list[NotificationPayload]:
        """Escalate an unresolved request via escalation channels."""
        if request.is_resolved:
            return []

        payloads = []
        for channel in request.policy.escalation_channels:
            adapter = self._adapters.get(channel)
            if not adapter:
                continue

            payload = NotificationPayload(
                channel=channel,
                recipient=request.repository,
                subject=f"⚠️ ESCALATION: {request.title}",
                body=self._format_escalation_body(request),
                action_url=request.pr_url,
                metadata={"request_id": request.id, "escalation": True},
            )
            await adapter.send(payload)
            payload.sent_at = datetime.now(timezone.utc)
            payloads.append(payload)

        return payloads

    def record_decision(
        self,
        request_id: str,
        approver: str,
        decision: str,
        comment: str = "",
    ) -> ApprovalRequest | None:
        """Record an approver's decision on a request."""
        request = self._requests.get(request_id)
        if not request or request.is_resolved:
            return request

        request.approvers.append(
            {
                "user": approver,
                "decision": decision,
                "comment": comment,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

        if decision == "rejected":
            request.status = ApprovalStatus.REJECTED
            request.resolved_at = datetime.now(timezone.utc)
            request.resolution_comment = f"Rejected by {approver}: {comment}"
        elif decision == "approved" and request.has_sufficient_approvals:
            request.status = ApprovalStatus.APPROVED
            request.resolved_at = datetime.now(timezone.utc)
            request.resolution_comment = "Approved with sufficient approvals"

        logger.info(
            "Approval decision recorded",
            request_id=request_id,
            approver=approver,
            decision=decision,
            status=request.status.value,
        )
        return request

    def get_request(self, request_id: str) -> ApprovalRequest | None:
        return self._requests.get(request_id)

    def get_pending_requests(self) -> list[ApprovalRequest]:
        return [
            r
            for r in self._requests.values()
            if r.status == ApprovalStatus.PENDING
        ]

    def get_stats(self) -> dict[str, Any]:
        """Return approval workflow statistics."""
        all_requests = list(self._requests.values())
        if not all_requests:
            return {"total": 0}

        approved = [r for r in all_requests if r.status == ApprovalStatus.APPROVED]
        rejected = [r for r in all_requests if r.status == ApprovalStatus.REJECTED]
        pending = [r for r in all_requests if r.status == ApprovalStatus.PENDING]
        auto_approved = [
            r for r in approved if "Auto-approved" in (r.resolution_comment or "")
        ]

        return {
            "total": len(all_requests),
            "approved": len(approved),
            "rejected": len(rejected),
            "pending": len(pending),
            "auto_approved": len(auto_approved),
            "approval_rate": len(approved) / len(all_requests) if all_requests else 0,
            "avg_confidence": sum(r.confidence for r in all_requests) / len(all_requests),
        }

    def _format_body(self, request: ApprovalRequest) -> str:
        return (
            f"**Repository:** {request.repository}\n"
            f"**Fix:** {request.title}\n"
            f"**Confidence:** {request.confidence:.0%}\n"
            f"**Severity:** {request.severity}\n\n"
            f"{request.description}\n\n"
            f"**Diff Summary:**\n```\n{request.diff_summary}\n```"
        )

    def _format_escalation_body(self, request: ApprovalRequest) -> str:
        hours = (
            datetime.now(timezone.utc) - request.created_at
        ).total_seconds() / 3600
        return (
            f"⚠️ This fix has been pending approval for {hours:.1f} hours.\n\n"
            + self._format_body(request)
        )
