"""Tests for the event-driven notification system."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codeverify_core.events import (
    AnalysisCompleteEvent,
    AnalysisFailedEvent,
    CriticalFindingEvent,
    DigestReadyEvent,
    Event,
    EventBus,
    EventPriority,
    ScanCompleteEvent,
    get_event_bus,
    on_event,
    reset_event_bus,
)
from codeverify_core.notification_handlers import (
    ConfigProvider,
    InMemoryConfigProvider,
    NotificationEventHandler,
    setup_notification_handlers,
)
from codeverify_core.notifications import (
    NotificationChannel,
    NotificationConfig,
    NotificationSender,
    NotificationType,
)


class TestEventBus:
    """Tests for EventBus functionality."""
    
    def setup_method(self):
        """Reset event bus before each test."""
        reset_event_bus()
    
    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self):
        """Test basic subscription and publishing."""
        bus = EventBus()
        received_events = []
        
        async def handler(event: AnalysisCompleteEvent):
            received_events.append(event)
        
        bus.subscribe(AnalysisCompleteEvent, handler)
        
        event = AnalysisCompleteEvent(
            repo_full_name="org/repo",
            pr_number=123,
            status="passed",
        )
        await bus.publish(event)
        
        assert len(received_events) == 1
        assert received_events[0].repo_full_name == "org/repo"
    
    @pytest.mark.asyncio
    async def test_multiple_handlers(self):
        """Test multiple handlers for same event type."""
        bus = EventBus()
        handler1_calls = []
        handler2_calls = []
        
        async def handler1(event: AnalysisCompleteEvent):
            handler1_calls.append(event)
        
        async def handler2(event: AnalysisCompleteEvent):
            handler2_calls.append(event)
        
        bus.subscribe(AnalysisCompleteEvent, handler1)
        bus.subscribe(AnalysisCompleteEvent, handler2)
        
        await bus.publish(AnalysisCompleteEvent(repo_full_name="test/repo"))
        
        assert len(handler1_calls) == 1
        assert len(handler2_calls) == 1
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Test that handlers are called in priority order."""
        bus = EventBus()
        call_order = []
        
        async def low_handler(event):
            call_order.append("low")
        
        async def high_handler(event):
            call_order.append("high")
        
        async def normal_handler(event):
            call_order.append("normal")
        
        # Subscribe in non-priority order
        bus.subscribe(AnalysisCompleteEvent, low_handler, EventPriority.LOW)
        bus.subscribe(AnalysisCompleteEvent, high_handler, EventPriority.HIGH)
        bus.subscribe(AnalysisCompleteEvent, normal_handler, EventPriority.NORMAL)
        
        await bus.publish(AnalysisCompleteEvent())
        
        # High should be called first, then normal, then low
        assert call_order == ["high", "normal", "low"]
    
    @pytest.mark.asyncio
    async def test_event_filter(self):
        """Test filtering events per handler."""
        bus = EventBus()
        received = []
        
        async def handler(event: AnalysisCompleteEvent):
            received.append(event)
        
        # Only handle events for specific repo
        bus.subscribe(
            AnalysisCompleteEvent,
            handler,
            filter_fn=lambda e: e.repo_full_name == "target/repo",
        )
        
        await bus.publish(AnalysisCompleteEvent(repo_full_name="other/repo"))
        await bus.publish(AnalysisCompleteEvent(repo_full_name="target/repo"))
        
        assert len(received) == 1
        assert received[0].repo_full_name == "target/repo"
    
    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test unsubscribing from events."""
        bus = EventBus()
        received = []
        
        async def handler(event):
            received.append(event)
        
        unsubscribe = bus.subscribe(AnalysisCompleteEvent, handler)
        
        await bus.publish(AnalysisCompleteEvent())
        assert len(received) == 1
        
        unsubscribe()
        
        await bus.publish(AnalysisCompleteEvent())
        assert len(received) == 1  # No new events
    
    @pytest.mark.asyncio
    async def test_handler_error_isolation(self):
        """Test that one handler error doesn't affect others."""
        bus = EventBus()
        successful_calls = []
        
        async def failing_handler(event):
            raise ValueError("Handler error")
        
        async def successful_handler(event):
            successful_calls.append(event)
        
        bus.subscribe(AnalysisCompleteEvent, failing_handler, EventPriority.HIGH)
        bus.subscribe(AnalysisCompleteEvent, successful_handler, EventPriority.LOW)
        
        errors = await bus.publish(AnalysisCompleteEvent())
        
        # Successful handler still ran
        assert len(successful_calls) == 1
        # Error was captured
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
    
    @pytest.mark.asyncio
    async def test_middleware(self):
        """Test event middleware."""
        bus = EventBus()
        received = []
        
        async def handler(event):
            received.append(event)
        
        async def modify_middleware(event):
            # Modify the event
            event.repo_full_name = "modified/repo"
            return event
        
        bus.add_middleware(modify_middleware)
        bus.subscribe(AnalysisCompleteEvent, handler)
        
        await bus.publish(AnalysisCompleteEvent(repo_full_name="original/repo"))
        
        assert received[0].repo_full_name == "modified/repo"
    
    @pytest.mark.asyncio
    async def test_middleware_filter(self):
        """Test middleware can filter events."""
        bus = EventBus()
        received = []
        
        async def handler(event):
            received.append(event)
        
        async def filter_middleware(event):
            # Filter out events with pr_number == 0
            if getattr(event, "pr_number", 0) == 0:
                return None
            return event
        
        bus.add_middleware(filter_middleware)
        bus.subscribe(AnalysisCompleteEvent, handler)
        
        await bus.publish(AnalysisCompleteEvent(pr_number=0))
        await bus.publish(AnalysisCompleteEvent(pr_number=123))
        
        assert len(received) == 1
        assert received[0].pr_number == 123


class TestGlobalEventBus:
    """Tests for global event bus singleton."""
    
    def setup_method(self):
        reset_event_bus()
    
    def test_get_event_bus_singleton(self):
        """Test that get_event_bus returns the same instance."""
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2
    
    def test_reset_event_bus(self):
        """Test that reset_event_bus creates new instance."""
        bus1 = get_event_bus()
        reset_event_bus()
        bus2 = get_event_bus()
        assert bus1 is not bus2
    
    @pytest.mark.asyncio
    async def test_on_event_decorator(self):
        """Test the @on_event decorator."""
        received = []
        
        @on_event(AnalysisCompleteEvent)
        async def handler(event: AnalysisCompleteEvent):
            received.append(event)
        
        await get_event_bus().publish(AnalysisCompleteEvent(repo_full_name="test"))
        
        assert len(received) == 1


class TestEventTypes:
    """Tests for event type definitions."""
    
    def test_analysis_complete_event(self):
        """Test AnalysisCompleteEvent structure."""
        event = AnalysisCompleteEvent(
            repo_full_name="org/repo",
            pr_number=42,
            pr_title="Fix bug",
            status="passed",
            total_findings=5,
            critical_findings=0,
        )
        
        assert event.event_type == "analysis.complete"
        assert event.repo_full_name == "org/repo"
        assert event.pr_number == 42
        assert event.event_id  # Auto-generated
        assert event.timestamp  # Auto-generated
    
    def test_analysis_failed_event(self):
        """Test AnalysisFailedEvent structure."""
        event = AnalysisFailedEvent(
            repo_full_name="org/repo",
            pr_number=42,
            error_message="Timeout",
            error_type="TimeoutError",
        )
        
        assert event.event_type == "analysis.failed"
    
    def test_critical_finding_event(self):
        """Test CriticalFindingEvent structure."""
        event = CriticalFindingEvent(
            repo_full_name="org/repo",
            pr_number=42,
            finding_title="SQL Injection",
            cwe_id="CWE-89",
        )
        
        assert event.event_type == "finding.critical"
        assert event.severity == "critical"
    
    def test_scan_complete_event(self):
        """Test ScanCompleteEvent structure."""
        event = ScanCompleteEvent(
            repo_full_name="org/repo",
            scan_type="full",
            status="completed",
            security_score=85.5,
        )
        
        assert event.event_type == "scan.complete"
    
    def test_digest_ready_event(self):
        """Test DigestReadyEvent structure."""
        event = DigestReadyEvent(
            organization="my-org",
            period="daily",
            total_prs_analyzed=10,
        )
        
        assert event.event_type == "digest.daily"
        
        weekly = DigestReadyEvent(period="weekly")
        assert weekly.event_type == "digest.weekly"


class TestInMemoryConfigProvider:
    """Tests for InMemoryConfigProvider."""
    
    @pytest.mark.asyncio
    async def test_repo_configs(self):
        """Test repository config storage."""
        provider = InMemoryConfigProvider()
        
        config = NotificationConfig(
            channel=NotificationChannel.SLACK,
            webhook_url="https://hooks.slack.com/test",
            notification_types=[NotificationType.ANALYSIS_COMPLETE],
        )
        
        provider.add_repo_config("org/repo", config)
        
        configs = await provider.get_configs("org/repo")
        assert len(configs) == 1
        assert configs[0].channel == NotificationChannel.SLACK
        
        # Non-existent repo returns empty
        empty = await provider.get_configs("other/repo")
        assert len(empty) == 0
    
    @pytest.mark.asyncio
    async def test_org_configs(self):
        """Test organization config storage."""
        provider = InMemoryConfigProvider()
        
        config = NotificationConfig(
            channel=NotificationChannel.TEAMS,
            webhook_url="https://teams.webhook/test",
            notification_types=[NotificationType.DAILY_DIGEST],
        )
        
        provider.add_org_config("my-org", config)
        
        configs = await provider.get_org_configs("my-org")
        assert len(configs) == 1
        assert configs[0].channel == NotificationChannel.TEAMS


class TestNotificationEventHandler:
    """Tests for NotificationEventHandler."""
    
    def setup_method(self):
        reset_event_bus()
    
    @pytest.mark.asyncio
    async def test_handle_analysis_complete(self):
        """Test handling analysis complete events."""
        sender = MagicMock(spec=NotificationSender)
        sender.send_analysis_notification = AsyncMock(return_value=[{"success": True}])
        
        provider = InMemoryConfigProvider()
        provider.add_repo_config(
            "org/repo",
            NotificationConfig(
                channel=NotificationChannel.SLACK,
                webhook_url="https://test",
                notification_types=[NotificationType.ANALYSIS_COMPLETE],
            ),
        )
        
        handler = NotificationEventHandler(sender=sender)
        handler.set_config_provider(provider)
        
        event = AnalysisCompleteEvent(
            repo_full_name="org/repo",
            pr_number=123,
            status="passed",
        )
        
        await handler.handle_analysis_complete(event)
        
        sender.send_analysis_notification.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_critical_finding(self):
        """Test handling critical finding events."""
        sender = MagicMock(spec=NotificationSender)
        sender.send_analysis_notification = AsyncMock(return_value=[{"success": True}])
        
        provider = InMemoryConfigProvider()
        provider.add_repo_config(
            "org/repo",
            NotificationConfig(
                channel=NotificationChannel.SLACK,
                webhook_url="https://test",
                notification_types=[NotificationType.CRITICAL_FINDING],
            ),
        )
        
        handler = NotificationEventHandler(sender=sender)
        handler.set_config_provider(provider)
        
        event = CriticalFindingEvent(
            repo_full_name="org/repo",
            pr_number=123,
            finding_title="SQL Injection",
        )
        
        await handler.handle_critical_finding(event)
        
        sender.send_analysis_notification.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_no_config_no_notification(self):
        """Test that no notification is sent without config."""
        sender = MagicMock(spec=NotificationSender)
        sender.send_analysis_notification = AsyncMock()
        
        provider = InMemoryConfigProvider()  # Empty
        
        handler = NotificationEventHandler(sender=sender)
        handler.set_config_provider(provider)
        
        event = AnalysisCompleteEvent(repo_full_name="org/repo")
        
        await handler.handle_analysis_complete(event)
        
        sender.send_analysis_notification.assert_not_called()


class TestSetupNotificationHandlers:
    """Tests for setup_notification_handlers integration."""
    
    def setup_method(self):
        reset_event_bus()
    
    @pytest.mark.asyncio
    async def test_setup_subscribes_to_events(self):
        """Test that setup subscribes to all event types."""
        sender = MagicMock(spec=NotificationSender)
        sender.send_analysis_notification = AsyncMock(return_value=[])
        sender.send_scan_notification = AsyncMock(return_value=[])
        sender.send_digest = AsyncMock(return_value=[])
        
        provider = InMemoryConfigProvider()
        provider.add_repo_config(
            "org/repo",
            NotificationConfig(
                channel=NotificationChannel.SLACK,
                webhook_url="https://test",
                notification_types=[
                    NotificationType.ANALYSIS_COMPLETE,
                    NotificationType.SCAN_COMPLETE,
                ],
            ),
        )
        
        setup_notification_handlers(config_provider=provider, sender=sender)
        bus = get_event_bus()
        
        # Publish different event types
        await bus.publish(AnalysisCompleteEvent(repo_full_name="org/repo"))
        await bus.publish(ScanCompleteEvent(repo_full_name="org/repo"))
        
        sender.send_analysis_notification.assert_called()
        sender.send_scan_notification.assert_called()
