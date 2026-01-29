"""Event handlers for notifications.

This module connects the event bus to the notification system,
automatically sending notifications when relevant events occur.

Usage:
    # Initialize handlers at application startup
    from codeverify_core.notification_handlers import setup_notification_handlers
    
    configs = [NotificationConfig(...), ...]
    setup_notification_handlers(configs)
    
    # Events will now automatically trigger notifications
    await event_bus.publish(AnalysisCompleteEvent(...))
"""

from typing import Any

import structlog

from codeverify_core.events import (
    AnalysisCompleteEvent,
    AnalysisFailedEvent,
    CriticalFindingEvent,
    DigestReadyEvent,
    Event,
    EventPriority,
    ScanCompleteEvent,
    get_event_bus,
)
from codeverify_core.notifications import (
    AnalysisNotification,
    DigestNotification,
    NotificationChannel,
    NotificationConfig,
    NotificationSender,
    NotificationType,
    ScanNotification,
)

logger = structlog.get_logger()


class NotificationEventHandler:
    """
    Handles events and dispatches notifications to configured channels.
    
    This class bridges the event system with the notification system,
    converting events to notification payloads and sending them.
    """
    
    def __init__(
        self,
        sender: NotificationSender | None = None,
    ) -> None:
        """
        Initialize the notification event handler.
        
        Args:
            sender: NotificationSender instance (created if not provided)
        """
        self.sender = sender or NotificationSender()
        self._config_provider: ConfigProvider | None = None
    
    def set_config_provider(self, provider: "ConfigProvider") -> None:
        """Set the configuration provider for looking up notification configs."""
        self._config_provider = provider
    
    async def _get_configs(self, repo_full_name: str) -> list[NotificationConfig]:
        """Get notification configs for a repository."""
        if self._config_provider is None:
            logger.warning("No config provider set, using empty configs")
            return []
        return await self._config_provider.get_configs(repo_full_name)
    
    async def handle_analysis_complete(self, event: AnalysisCompleteEvent) -> None:
        """Handle analysis completion events."""
        configs = await self._get_configs(event.repo_full_name)
        if not configs:
            return
        
        notification = AnalysisNotification(
            repo_full_name=event.repo_full_name,
            pr_number=event.pr_number,
            pr_title=event.pr_title,
            pr_url=event.pr_url,
            status=event.status,
            total_findings=event.total_findings,
            critical_findings=event.critical_findings,
            high_findings=event.high_findings,
            findings_url=event.findings_url,
            author=event.author,
            analyzed_at=event.timestamp,
        )
        
        results = await self.sender.send_analysis_notification(notification, configs)
        logger.info(
            "Analysis notification sent",
            event_id=event.event_id,
            results=results,
        )
    
    async def handle_analysis_failed(self, event: AnalysisFailedEvent) -> None:
        """Handle analysis failure events."""
        configs = await self._get_configs(event.repo_full_name)
        if not configs:
            return
        
        # Filter to configs that want failure notifications
        failure_configs = [
            c for c in configs
            if NotificationType.ANALYSIS_FAILED in c.notification_types
        ]
        if not failure_configs:
            return
        
        notification = AnalysisNotification(
            repo_full_name=event.repo_full_name,
            pr_number=event.pr_number,
            pr_title=event.pr_title,
            pr_url=event.pr_url,
            status="failed",
            total_findings=0,
            critical_findings=0,
            high_findings=0,
            findings_url="",
            author="",
            analyzed_at=event.timestamp,
        )
        
        # Note: We're using send_analysis_notification but the status="failed"
        # will be handled by the formatter
        results = await self.sender.send_analysis_notification(notification, failure_configs)
        logger.info(
            "Analysis failure notification sent",
            event_id=event.event_id,
            error_type=event.error_type,
            results=results,
        )
    
    async def handle_critical_finding(self, event: CriticalFindingEvent) -> None:
        """Handle critical finding events with immediate notification."""
        configs = await self._get_configs(event.repo_full_name)
        if not configs:
            return
        
        # Filter to configs that want critical finding notifications
        critical_configs = [
            c for c in configs
            if NotificationType.CRITICAL_FINDING in c.notification_types
        ]
        if not critical_configs:
            return
        
        # For critical findings, we create a special notification
        # that goes out immediately (not batched with other findings)
        notification = AnalysisNotification(
            repo_full_name=event.repo_full_name,
            pr_number=event.pr_number,
            pr_title=f"🚨 Critical: {event.finding_title}",
            pr_url=event.pr_url,
            status="critical",
            total_findings=1,
            critical_findings=1,
            high_findings=0,
            findings_url=event.pr_url,
            author=event.author,
            analyzed_at=event.timestamp,
        )
        
        results = await self.sender.send_analysis_notification(notification, critical_configs)
        logger.info(
            "Critical finding notification sent",
            event_id=event.event_id,
            finding_id=event.finding_id,
            cwe_id=event.cwe_id,
            results=results,
        )
    
    async def handle_scan_complete(self, event: ScanCompleteEvent) -> None:
        """Handle scan completion events."""
        configs = await self._get_configs(event.repo_full_name)
        if not configs:
            return
        
        notification = ScanNotification(
            repo_full_name=event.repo_full_name,
            scan_type=event.scan_type,
            status=event.status,
            security_score=event.security_score,
            quality_score=event.quality_score,
            total_findings=event.total_findings,
            critical_findings=event.critical_findings,
            scan_url=event.scan_url,
            scanned_at=event.timestamp,
        )
        
        results = await self.sender.send_scan_notification(notification, configs)
        logger.info(
            "Scan notification sent",
            event_id=event.event_id,
            scan_id=event.scan_id,
            results=results,
        )
    
    async def handle_digest_ready(self, event: DigestReadyEvent) -> None:
        """Handle digest ready events."""
        # For digests, we need org-level configs, not repo-level
        # Use the config provider's org method if available
        if self._config_provider is None:
            return
        
        configs = await self._config_provider.get_org_configs(event.organization)
        if not configs:
            return
        
        notification = DigestNotification(
            organization=event.organization,
            period=event.period,
            start_date=event.start_date,
            end_date=event.end_date,
            total_prs_analyzed=event.total_prs_analyzed,
            prs_passed=event.prs_passed,
            prs_failed=event.prs_failed,
            total_findings=event.total_findings,
            critical_findings=event.critical_findings,
            top_issues=event.top_issues,
            trend=event.trend,
        )
        
        results = await self.sender.send_digest(notification, configs)
        logger.info(
            "Digest notification sent",
            event_id=event.event_id,
            organization=event.organization,
            period=event.period,
            results=results,
        )


class ConfigProvider:
    """
    Abstract interface for providing notification configurations.
    
    Implement this to integrate with your configuration storage
    (database, API, etc.).
    """
    
    async def get_configs(self, repo_full_name: str) -> list[NotificationConfig]:
        """Get notification configs for a repository."""
        raise NotImplementedError
    
    async def get_org_configs(self, organization: str) -> list[NotificationConfig]:
        """Get notification configs for an organization."""
        raise NotImplementedError


class InMemoryConfigProvider(ConfigProvider):
    """In-memory config provider for testing and simple deployments."""
    
    def __init__(self) -> None:
        self._repo_configs: dict[str, list[NotificationConfig]] = {}
        self._org_configs: dict[str, list[NotificationConfig]] = {}
    
    def add_repo_config(self, repo_full_name: str, config: NotificationConfig) -> None:
        """Add a config for a repository."""
        if repo_full_name not in self._repo_configs:
            self._repo_configs[repo_full_name] = []
        self._repo_configs[repo_full_name].append(config)
    
    def add_org_config(self, organization: str, config: NotificationConfig) -> None:
        """Add a config for an organization."""
        if organization not in self._org_configs:
            self._org_configs[organization] = []
        self._org_configs[organization].append(config)
    
    async def get_configs(self, repo_full_name: str) -> list[NotificationConfig]:
        """Get notification configs for a repository."""
        return self._repo_configs.get(repo_full_name, [])
    
    async def get_org_configs(self, organization: str) -> list[NotificationConfig]:
        """Get notification configs for an organization."""
        return self._org_configs.get(organization, [])


def setup_notification_handlers(
    config_provider: ConfigProvider | None = None,
    sender: NotificationSender | None = None,
) -> NotificationEventHandler:
    """
    Set up notification handlers on the global event bus.
    
    Call this at application startup to enable event-driven notifications.
    
    Args:
        config_provider: Provider for notification configurations
        sender: NotificationSender instance
        
    Returns:
        The configured NotificationEventHandler
    """
    handler = NotificationEventHandler(sender=sender)
    if config_provider:
        handler.set_config_provider(config_provider)
    
    bus = get_event_bus()
    
    # Subscribe to all notification-relevant events
    bus.subscribe(
        AnalysisCompleteEvent,
        handler.handle_analysis_complete,
        priority=EventPriority.NORMAL,
    )
    
    bus.subscribe(
        AnalysisFailedEvent,
        handler.handle_analysis_failed,
        priority=EventPriority.NORMAL,
    )
    
    bus.subscribe(
        CriticalFindingEvent,
        handler.handle_critical_finding,
        priority=EventPriority.HIGH,  # Critical findings get priority
    )
    
    bus.subscribe(
        ScanCompleteEvent,
        handler.handle_scan_complete,
        priority=EventPriority.NORMAL,
    )
    
    bus.subscribe(
        DigestReadyEvent,
        handler.handle_digest_ready,
        priority=EventPriority.LOW,  # Digests are less urgent
    )
    
    logger.info("Notification event handlers configured")
    return handler


# Middleware for logging all events
async def logging_middleware(event: Event) -> Event:
    """Middleware that logs all events passing through the bus."""
    logger.debug(
        "Event passing through bus",
        event_type=event.event_type,
        event_id=event.event_id,
    )
    return event


# Middleware for filtering events based on repo configuration
class RepoFilterMiddleware:
    """Middleware that filters events for disabled repositories."""
    
    def __init__(self, disabled_repos: set[str] | None = None) -> None:
        self.disabled_repos = disabled_repos or set()
    
    async def __call__(self, event: Event) -> Event | None:
        """Filter events from disabled repositories."""
        repo = getattr(event, "repo_full_name", None)
        if repo and repo in self.disabled_repos:
            logger.debug(
                "Event filtered (repo disabled)",
                event_id=event.event_id,
                repo=repo,
            )
            return None
        return event
