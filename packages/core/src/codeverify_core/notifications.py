"""Notifications - Slack and Microsoft Teams integrations."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class NotificationChannel(str, Enum):
    """Supported notification channels."""

    SLACK = "slack"
    TEAMS = "teams"
    EMAIL = "email"
    WEBHOOK = "webhook"


class NotificationType(str, Enum):
    """Types of notifications."""

    ANALYSIS_COMPLETE = "analysis_complete"
    ANALYSIS_FAILED = "analysis_failed"
    CRITICAL_FINDING = "critical_finding"
    SCAN_COMPLETE = "scan_complete"
    DAILY_DIGEST = "daily_digest"
    WEEKLY_REPORT = "weekly_report"


@dataclass
class NotificationConfig:
    """Configuration for a notification channel."""

    channel: NotificationChannel
    webhook_url: str
    enabled: bool = True
    notification_types: list[NotificationType] = field(default_factory=list)
    mention_on_critical: bool = True
    mention_users: list[str] = field(default_factory=list)  # User IDs to mention
    channel_name: str | None = None  # For Slack channel override


@dataclass
class AnalysisNotification:
    """Data for an analysis notification."""

    repo_full_name: str
    pr_number: int
    pr_title: str
    pr_url: str
    status: str  # "passed", "failed", "warnings"
    total_findings: int
    critical_findings: int
    high_findings: int
    findings_url: str
    author: str
    analyzed_at: datetime


@dataclass
class ScanNotification:
    """Data for a scan notification."""

    repo_full_name: str
    scan_type: str
    status: str
    security_score: float | None
    quality_score: float | None
    total_findings: int
    critical_findings: int
    scan_url: str
    scanned_at: datetime


@dataclass
class DigestNotification:
    """Data for a daily/weekly digest."""

    organization: str
    period: str  # "daily", "weekly"
    start_date: datetime
    end_date: datetime
    total_prs_analyzed: int
    prs_passed: int
    prs_failed: int
    total_findings: int
    critical_findings: int
    top_issues: list[dict[str, Any]]
    trend: str  # "improving", "stable", "declining"


class NotificationFormatter(ABC):
    """Base class for notification formatters."""

    @abstractmethod
    def format_analysis(self, notification: AnalysisNotification) -> dict[str, Any]:
        """Format an analysis notification."""
        pass

    @abstractmethod
    def format_scan(self, notification: ScanNotification) -> dict[str, Any]:
        """Format a scan notification."""
        pass

    @abstractmethod
    def format_digest(self, notification: DigestNotification) -> dict[str, Any]:
        """Format a digest notification."""
        pass


class SlackFormatter(NotificationFormatter):
    """Formats notifications for Slack."""

    def format_analysis(self, notification: AnalysisNotification) -> dict[str, Any]:
        """Format analysis notification for Slack."""
        # Determine status emoji and color
        if notification.status == "passed":
            emoji = "✅"
            color = "#36a64f"  # Green
        elif notification.critical_findings > 0:
            emoji = "🚨"
            color = "#ff0000"  # Red
        else:
            emoji = "⚠️"
            color = "#ffcc00"  # Yellow

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} CodeVerify Analysis Complete",
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Repository:*\n{notification.repo_full_name}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*PR:*\n<{notification.pr_url}|#{notification.pr_number}>",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Author:*\n{notification.author}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Status:*\n{notification.status.upper()}",
                    },
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{notification.pr_title}*",
                }
            },
        ]

        # Add findings summary
        if notification.total_findings > 0:
            findings_text = []
            if notification.critical_findings > 0:
                findings_text.append(f"🔴 Critical: {notification.critical_findings}")
            if notification.high_findings > 0:
                findings_text.append(f"🟠 High: {notification.high_findings}")
            other = notification.total_findings - notification.critical_findings - notification.high_findings
            if other > 0:
                findings_text.append(f"🟡 Other: {other}")

            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Findings:* " + " | ".join(findings_text),
                }
            })

        # Add action button
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "View Details",
                    },
                    "url": notification.findings_url,
                    "style": "primary",
                }
            ]
        })

        return {
            "attachments": [
                {
                    "color": color,
                    "blocks": blocks,
                }
            ]
        }

    def format_scan(self, notification: ScanNotification) -> dict[str, Any]:
        """Format scan notification for Slack."""
        if notification.status == "completed":
            emoji = "✅"
            color = "#36a64f"
        else:
            emoji = "❌"
            color = "#ff0000"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Codebase Scan Complete",
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Repository:*\n{notification.repo_full_name}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Scan Type:*\n{notification.scan_type}",
                    },
                ]
            },
        ]

        # Add scores
        if notification.security_score is not None:
            score_emoji = "🟢" if notification.security_score >= 80 else "🟡" if notification.security_score >= 60 else "🔴"
            blocks.append({
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Security Score:*\n{score_emoji} {notification.security_score:.1f}/100",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Quality Score:*\n{notification.quality_score:.1f}/100" if notification.quality_score else "*Quality Score:*\nN/A",
                    },
                ]
            })

        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View Report"},
                    "url": notification.scan_url,
                    "style": "primary",
                }
            ]
        })

        return {"attachments": [{"color": color, "blocks": blocks}]}

    def format_digest(self, notification: DigestNotification) -> dict[str, Any]:
        """Format digest notification for Slack."""
        period_label = "Daily" if notification.period == "daily" else "Weekly"
        trend_emoji = "📈" if notification.trend == "improving" else "📉" if notification.trend == "declining" else "➡️"

        pass_rate = (notification.prs_passed / notification.total_prs_analyzed * 100) if notification.total_prs_analyzed > 0 else 0

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"📊 {period_label} CodeVerify Report",
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{notification.organization}* | {notification.start_date.strftime('%b %d')} - {notification.end_date.strftime('%b %d, %Y')}",
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*PRs Analyzed:*\n{notification.total_prs_analyzed}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Pass Rate:*\n{pass_rate:.1f}%",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Critical Issues:*\n{notification.critical_findings}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Trend:*\n{trend_emoji} {notification.trend.capitalize()}",
                    },
                ]
            },
        ]

        if notification.top_issues:
            issues_text = "\n".join(
                f"• {issue['title']} ({issue['count']})"
                for issue in notification.top_issues[:5]
            )
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Top Issues:*\n{issues_text}",
                }
            })

        return {"blocks": blocks}


class TeamsFormatter(NotificationFormatter):
    """Formats notifications for Microsoft Teams."""

    def format_analysis(self, notification: AnalysisNotification) -> dict[str, Any]:
        """Format analysis notification for Teams."""
        if notification.status == "passed":
            theme_color = "00FF00"
        elif notification.critical_findings > 0:
            theme_color = "FF0000"
        else:
            theme_color = "FFCC00"

        facts = [
            {"name": "Repository", "value": notification.repo_full_name},
            {"name": "PR", "value": f"[#{notification.pr_number}]({notification.pr_url})"},
            {"name": "Author", "value": notification.author},
            {"name": "Status", "value": notification.status.upper()},
            {"name": "Total Findings", "value": str(notification.total_findings)},
        ]

        if notification.critical_findings > 0:
            facts.append({"name": "Critical", "value": str(notification.critical_findings)})

        return {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": theme_color,
            "summary": f"CodeVerify: {notification.pr_title}",
            "sections": [
                {
                    "activityTitle": "CodeVerify Analysis Complete",
                    "activitySubtitle": notification.pr_title,
                    "facts": facts,
                    "markdown": True,
                }
            ],
            "potentialAction": [
                {
                    "@type": "OpenUri",
                    "name": "View Details",
                    "targets": [{"os": "default", "uri": notification.findings_url}],
                }
            ],
        }

    def format_scan(self, notification: ScanNotification) -> dict[str, Any]:
        """Format scan notification for Teams."""
        theme_color = "00FF00" if notification.status == "completed" else "FF0000"

        facts = [
            {"name": "Repository", "value": notification.repo_full_name},
            {"name": "Scan Type", "value": notification.scan_type},
            {"name": "Status", "value": notification.status},
        ]

        if notification.security_score is not None:
            facts.append({"name": "Security Score", "value": f"{notification.security_score:.1f}/100"})
        if notification.quality_score is not None:
            facts.append({"name": "Quality Score", "value": f"{notification.quality_score:.1f}/100"})

        return {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": theme_color,
            "summary": f"Codebase Scan: {notification.repo_full_name}",
            "sections": [
                {
                    "activityTitle": "Codebase Scan Complete",
                    "facts": facts,
                    "markdown": True,
                }
            ],
            "potentialAction": [
                {
                    "@type": "OpenUri",
                    "name": "View Report",
                    "targets": [{"os": "default", "uri": notification.scan_url}],
                }
            ],
        }

    def format_digest(self, notification: DigestNotification) -> dict[str, Any]:
        """Format digest notification for Teams."""
        period_label = "Daily" if notification.period == "daily" else "Weekly"
        pass_rate = (notification.prs_passed / notification.total_prs_analyzed * 100) if notification.total_prs_analyzed > 0 else 0

        facts = [
            {"name": "Organization", "value": notification.organization},
            {"name": "Period", "value": f"{notification.start_date.strftime('%b %d')} - {notification.end_date.strftime('%b %d')}"},
            {"name": "PRs Analyzed", "value": str(notification.total_prs_analyzed)},
            {"name": "Pass Rate", "value": f"{pass_rate:.1f}%"},
            {"name": "Critical Issues", "value": str(notification.critical_findings)},
            {"name": "Trend", "value": notification.trend.capitalize()},
        ]

        return {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": "0078D4",
            "summary": f"{period_label} CodeVerify Report",
            "sections": [
                {
                    "activityTitle": f"📊 {period_label} CodeVerify Report",
                    "facts": facts,
                    "markdown": True,
                }
            ],
        }


class NotificationSender:
    """Sends notifications to configured channels."""

    def __init__(self) -> None:
        """Initialize notification sender."""
        self.formatters = {
            NotificationChannel.SLACK: SlackFormatter(),
            NotificationChannel.TEAMS: TeamsFormatter(),
        }

    async def send_analysis_notification(
        self,
        notification: AnalysisNotification,
        configs: list[NotificationConfig],
    ) -> list[dict[str, Any]]:
        """Send analysis notification to all configured channels."""
        results = []

        for config in configs:
            if not config.enabled:
                continue
            if NotificationType.ANALYSIS_COMPLETE not in config.notification_types:
                continue

            formatter = self.formatters.get(config.channel)
            if not formatter:
                continue

            payload = formatter.format_analysis(notification)
            result = await self._send_webhook(config.webhook_url, payload)
            results.append({
                "channel": config.channel.value,
                "success": result.get("success", False),
            })

        return results

    async def send_scan_notification(
        self,
        notification: ScanNotification,
        configs: list[NotificationConfig],
    ) -> list[dict[str, Any]]:
        """Send scan notification to all configured channels."""
        results = []

        for config in configs:
            if not config.enabled:
                continue
            if NotificationType.SCAN_COMPLETE not in config.notification_types:
                continue

            formatter = self.formatters.get(config.channel)
            if not formatter:
                continue

            payload = formatter.format_scan(notification)
            result = await self._send_webhook(config.webhook_url, payload)
            results.append({
                "channel": config.channel.value,
                "success": result.get("success", False),
            })

        return results

    async def send_digest(
        self,
        notification: DigestNotification,
        configs: list[NotificationConfig],
    ) -> list[dict[str, Any]]:
        """Send digest notification to all configured channels."""
        notification_type = (
            NotificationType.DAILY_DIGEST
            if notification.period == "daily"
            else NotificationType.WEEKLY_REPORT
        )

        results = []

        for config in configs:
            if not config.enabled:
                continue
            if notification_type not in config.notification_types:
                continue

            formatter = self.formatters.get(config.channel)
            if not formatter:
                continue

            payload = formatter.format_digest(notification)
            result = await self._send_webhook(config.webhook_url, payload)
            results.append({
                "channel": config.channel.value,
                "success": result.get("success", False),
            })

        return results

    async def _send_webhook(
        self,
        url: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a webhook request."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10.0,
                )
                response.raise_for_status()

            logger.info("Notification sent successfully", url=url[:50])
            return {"success": True}

        except Exception as e:
            logger.error("Failed to send notification", error=str(e), url=url[:50])
            return {"success": False, "error": str(e)}


# In-memory storage for notification configs (should use database)
_notification_configs: dict[str, list[NotificationConfig]] = {}


def get_configs_for_repo(repo_full_name: str) -> list[NotificationConfig]:
    """Get notification configs for a repository."""
    return _notification_configs.get(repo_full_name, [])


def save_config_for_repo(repo_full_name: str, config: NotificationConfig) -> None:
    """Save a notification config for a repository."""
    if repo_full_name not in _notification_configs:
        _notification_configs[repo_full_name] = []
    _notification_configs[repo_full_name].append(config)
