"""Event-driven notification system for CodeVerify.

This module provides an event bus pattern for decoupling notification triggers
from notification delivery. Components can publish events, and listeners can
subscribe to handle them asynchronously.

Example usage:
    # Subscribe to events
    event_bus = EventBus()
    event_bus.subscribe(AnalysisCompleteEvent, send_slack_notification)
    event_bus.subscribe(AnalysisCompleteEvent, send_teams_notification)
    
    # Publish events from anywhere
    await event_bus.publish(AnalysisCompleteEvent(
        repo_full_name="org/repo",
        pr_number=123,
        ...
    ))
"""

import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Generic, TypeVar
from uuid import uuid4

import structlog

logger = structlog.get_logger()


# Type variable for event types
E = TypeVar("E", bound="Event")

# Type alias for event handlers
EventHandler = Callable[[E], Coroutine[Any, Any, None]]


class EventPriority(int, Enum):
    """Priority levels for event handlers."""
    
    LOW = 0
    NORMAL = 50
    HIGH = 100
    CRITICAL = 200


@dataclass
class Event(ABC):
    """Base class for all events."""
    
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    @abstractmethod
    def event_type(self) -> str:
        """Return the event type identifier."""
        pass


@dataclass
class AnalysisCompleteEvent(Event):
    """Event fired when a PR analysis completes."""
    
    repo_full_name: str = ""
    pr_number: int = 0
    pr_title: str = ""
    pr_url: str = ""
    status: str = ""  # "passed", "failed", "warnings"
    total_findings: int = 0
    critical_findings: int = 0
    high_findings: int = 0
    findings_url: str = ""
    author: str = ""
    analysis_id: str = ""
    
    @property
    def event_type(self) -> str:
        return "analysis.complete"


@dataclass
class AnalysisFailedEvent(Event):
    """Event fired when a PR analysis fails."""
    
    repo_full_name: str = ""
    pr_number: int = 0
    pr_title: str = ""
    pr_url: str = ""
    error_message: str = ""
    error_type: str = ""
    analysis_id: str = ""
    
    @property
    def event_type(self) -> str:
        return "analysis.failed"


@dataclass
class CriticalFindingEvent(Event):
    """Event fired when a critical security finding is detected."""
    
    repo_full_name: str = ""
    pr_number: int = 0
    pr_url: str = ""
    finding_id: str = ""
    finding_title: str = ""
    finding_description: str = ""
    severity: str = "critical"
    cwe_id: str | None = None
    file_path: str = ""
    line_number: int | None = None
    author: str = ""
    
    @property
    def event_type(self) -> str:
        return "finding.critical"


@dataclass
class ScanCompleteEvent(Event):
    """Event fired when a codebase scan completes."""
    
    repo_full_name: str = ""
    scan_type: str = ""
    status: str = ""
    security_score: float | None = None
    quality_score: float | None = None
    total_findings: int = 0
    critical_findings: int = 0
    scan_url: str = ""
    scan_id: str = ""
    
    @property
    def event_type(self) -> str:
        return "scan.complete"


@dataclass
class DigestReadyEvent(Event):
    """Event fired when a digest report is ready to send."""
    
    organization: str = ""
    period: str = ""  # "daily", "weekly"
    start_date: datetime = field(default_factory=datetime.utcnow)
    end_date: datetime = field(default_factory=datetime.utcnow)
    total_prs_analyzed: int = 0
    prs_passed: int = 0
    prs_failed: int = 0
    total_findings: int = 0
    critical_findings: int = 0
    top_issues: list[dict[str, Any]] = field(default_factory=list)
    trend: str = "stable"  # "improving", "stable", "declining"
    
    @property
    def event_type(self) -> str:
        return f"digest.{self.period}"


@dataclass
class Subscription(Generic[E]):
    """Represents a subscription to an event type."""
    
    handler: EventHandler
    priority: EventPriority = EventPriority.NORMAL
    filter_fn: Callable[[E], bool] | None = None
    
    def matches(self, event: E) -> bool:
        """Check if this subscription should handle the event."""
        if self.filter_fn is None:
            return True
        return self.filter_fn(event)


class EventBus:
    """
    Central event bus for publishing and subscribing to events.
    
    The event bus supports:
    - Multiple handlers per event type
    - Priority-based handler execution
    - Event filtering per handler
    - Async handler execution
    - Error isolation between handlers
    """
    
    def __init__(self) -> None:
        """Initialize the event bus."""
        self._subscriptions: dict[type[Event], list[Subscription]] = defaultdict(list)
        self._middleware: list[Callable[[Event], Coroutine[Any, Any, Event | None]]] = []
    
    def subscribe(
        self,
        event_type: type[E],
        handler: EventHandler,
        priority: EventPriority = EventPriority.NORMAL,
        filter_fn: Callable[[E], bool] | None = None,
    ) -> Callable[[], None]:
        """
        Subscribe a handler to an event type.
        
        Args:
            event_type: The event class to subscribe to
            handler: Async function to handle the event
            priority: Handler priority (higher runs first)
            filter_fn: Optional filter to selectively handle events
            
        Returns:
            Unsubscribe function
        """
        subscription = Subscription(
            handler=handler,
            priority=priority,
            filter_fn=filter_fn,
        )
        self._subscriptions[event_type].append(subscription)
        
        # Sort by priority (descending)
        self._subscriptions[event_type].sort(
            key=lambda s: s.priority.value,
            reverse=True,
        )
        
        logger.debug(
            "Event handler subscribed",
            event_type=event_type.__name__,
            priority=priority.name,
        )
        
        def unsubscribe() -> None:
            self._subscriptions[event_type].remove(subscription)
        
        return unsubscribe
    
    def add_middleware(
        self,
        middleware: Callable[[Event], Coroutine[Any, Any, Event | None]],
    ) -> None:
        """
        Add middleware that processes events before handlers.
        
        Middleware can:
        - Modify events before they reach handlers
        - Filter events by returning None
        - Add logging, metrics, etc.
        """
        self._middleware.append(middleware)
    
    async def publish(self, event: Event) -> list[Exception]:
        """
        Publish an event to all subscribed handlers.
        
        Args:
            event: The event to publish
            
        Returns:
            List of exceptions from failed handlers (empty if all succeeded)
        """
        logger.info(
            "Publishing event",
            event_type=event.event_type,
            event_id=event.event_id,
        )
        
        # Run middleware
        processed_event: Event | None = event
        for middleware in self._middleware:
            try:
                processed_event = await middleware(processed_event)
                if processed_event is None:
                    logger.debug("Event filtered by middleware", event_id=event.event_id)
                    return []
            except Exception as e:
                logger.error(
                    "Middleware error",
                    event_id=event.event_id,
                    error=str(e),
                )
                # Continue with original event if middleware fails
                processed_event = event
        
        # Get subscriptions for this event type
        subscriptions = self._subscriptions.get(type(event), [])
        if not subscriptions:
            logger.debug("No handlers for event", event_type=event.event_type)
            return []
        
        # Execute handlers
        errors: list[Exception] = []
        for subscription in subscriptions:
            if not subscription.matches(processed_event):
                continue
                
            try:
                await subscription.handler(processed_event)
            except Exception as e:
                logger.error(
                    "Event handler failed",
                    event_type=event.event_type,
                    event_id=event.event_id,
                    error=str(e),
                )
                errors.append(e)
        
        return errors
    
    async def publish_all(self, events: list[Event]) -> dict[str, list[Exception]]:
        """
        Publish multiple events, returning errors per event.
        
        Args:
            events: List of events to publish
            
        Returns:
            Dict mapping event_id to list of errors
        """
        results = {}
        for event in events:
            errors = await self.publish(event)
            if errors:
                results[event.event_id] = errors
        return results
    
    def clear(self) -> None:
        """Clear all subscriptions and middleware."""
        self._subscriptions.clear()
        self._middleware.clear()


# Global event bus singleton
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def reset_event_bus() -> None:
    """Reset the global event bus (mainly for testing)."""
    global _event_bus
    if _event_bus is not None:
        _event_bus.clear()
    _event_bus = None


# Decorator for subscribing functions to events
def on_event(
    event_type: type[E],
    priority: EventPriority = EventPriority.NORMAL,
    filter_fn: Callable[[E], bool] | None = None,
) -> Callable[[EventHandler], EventHandler]:
    """
    Decorator to subscribe a function to an event type.
    
    Example:
        @on_event(AnalysisCompleteEvent)
        async def handle_analysis(event: AnalysisCompleteEvent):
            print(f"Analysis complete: {event.repo_full_name}")
    """
    def decorator(handler: EventHandler) -> EventHandler:
        get_event_bus().subscribe(event_type, handler, priority, filter_fn)
        return handler
    return decorator
