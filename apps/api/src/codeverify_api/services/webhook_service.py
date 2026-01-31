"""Webhook notifications service for CodeVerify."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Any, Callable
from uuid import UUID, uuid4
from enum import Enum

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WebhookStatus(str, Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"


class WebhookConfig(BaseModel):
    """Webhook configuration."""
    
    id: UUID = Field(default_factory=uuid4)
    organization_id: UUID
    url: str
    secret: str
    events: list[str] = Field(default_factory=lambda: ["analysis.completed"])
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Retry configuration
    max_retries: int = 5
    timeout_seconds: int = 30
    
    # Events: analysis.started, analysis.completed, analysis.failed, finding.created


class WebhookDelivery(BaseModel):
    """Webhook delivery record."""
    
    id: UUID = Field(default_factory=uuid4)
    webhook_id: UUID
    event: str
    payload: dict[str, Any]
    status: WebhookStatus = WebhookStatus.PENDING
    response_code: int | None = None
    response_body: str | None = None
    error_message: str | None = None
    attempts: int = 0
    max_attempts: int = 5
    next_retry_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    delivered_at: datetime | None = None
    last_attempt_at: datetime | None = None


class RetryStrategy:
    """Exponential backoff retry strategy with jitter."""
    
    def __init__(
        self,
        base_delay: float = 60.0,      # 1 minute
        max_delay: float = 3600.0,     # 1 hour
        exponential_base: float = 2.0,
        jitter: float = 0.1,           # 10% jitter
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        # Exponential backoff: base_delay * (exponential_base ^ attempt)
        delay = self.base_delay * (self.exponential_base ** attempt)
        
        # Cap at max delay
        delay = min(delay, self.max_delay)
        
        # Add jitter (±jitter%)
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)
    
    def get_next_retry_time(self, attempt: int) -> datetime:
        """Get the datetime for next retry."""
        delay = self.get_delay(attempt)
        return datetime.utcnow() + timedelta(seconds=delay)


class DeadLetterQueue:
    """Dead letter queue for failed webhook deliveries."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queue: list[WebhookDelivery] = []
    
    def add(self, delivery: WebhookDelivery) -> None:
        """Add a delivery to the dead letter queue."""
        delivery.status = WebhookStatus.DEAD_LETTER
        self._queue.append(delivery)
        
        # Trim if over max size (remove oldest)
        if len(self._queue) > self.max_size:
            self._queue = self._queue[-self.max_size:]
        
        logger.warning(
            f"Webhook delivery {delivery.id} moved to dead letter queue "
            f"after {delivery.attempts} attempts"
        )
    
    def get_all(self) -> list[WebhookDelivery]:
        """Get all deliveries in the dead letter queue."""
        return self._queue.copy()
    
    def get_for_webhook(self, webhook_id: UUID) -> list[WebhookDelivery]:
        """Get dead letter deliveries for a specific webhook."""
        return [d for d in self._queue if d.webhook_id == webhook_id]
    
    def remove(self, delivery_id: UUID) -> bool:
        """Remove a delivery from the queue (e.g., after manual retry)."""
        for i, d in enumerate(self._queue):
            if d.id == delivery_id:
                self._queue.pop(i)
                return True
        return False
    
    def clear(self) -> int:
        """Clear the queue and return number of items removed."""
        count = len(self._queue)
        self._queue = []
        return count


class WebhookMetrics:
    """Track webhook delivery metrics."""
    
    def __init__(self):
        self.total_deliveries = 0
        self.successful_deliveries = 0
        self.failed_deliveries = 0
        self.retried_deliveries = 0
        self.dead_letter_count = 0
        self.total_attempts = 0
        self.latencies: list[float] = []  # Last 1000 latencies
    
    def record_delivery(
        self,
        success: bool,
        attempts: int,
        latency_ms: float | None = None,
    ) -> None:
        """Record a delivery result."""
        self.total_deliveries += 1
        self.total_attempts += attempts
        
        if success:
            self.successful_deliveries += 1
        else:
            self.failed_deliveries += 1
        
        if attempts > 1:
            self.retried_deliveries += 1
        
        if latency_ms is not None:
            self.latencies.append(latency_ms)
            if len(self.latencies) > 1000:
                self.latencies = self.latencies[-1000:]
    
    def record_dead_letter(self) -> None:
        """Record a dead letter delivery."""
        self.dead_letter_count += 1
    
    def get_stats(self) -> dict[str, Any]:
        """Get current metrics."""
        success_rate = (
            self.successful_deliveries / self.total_deliveries * 100
            if self.total_deliveries > 0 else 0
        )
        
        avg_attempts = (
            self.total_attempts / self.total_deliveries
            if self.total_deliveries > 0 else 0
        )
        
        avg_latency = (
            sum(self.latencies) / len(self.latencies)
            if self.latencies else 0
        )
        
        return {
            "total_deliveries": self.total_deliveries,
            "successful_deliveries": self.successful_deliveries,
            "failed_deliveries": self.failed_deliveries,
            "retried_deliveries": self.retried_deliveries,
            "dead_letter_count": self.dead_letter_count,
            "success_rate_percent": round(success_rate, 2),
            "average_attempts": round(avg_attempts, 2),
            "average_latency_ms": round(avg_latency, 2),
        }


class WebhookService:
    """Service for managing and delivering webhooks with exponential backoff."""
    
    def __init__(
        self,
        retry_strategy: RetryStrategy | None = None,
        dead_letter_queue: DeadLetterQueue | None = None,
    ):
        self._webhooks: dict[UUID, WebhookConfig] = {}
        self._deliveries: list[WebhookDelivery] = []
        self._pending_retries: list[WebhookDelivery] = []
        
        self.retry_strategy = retry_strategy or RetryStrategy()
        self.dead_letter_queue = dead_letter_queue or DeadLetterQueue()
        self.metrics = WebhookMetrics()
        
        # Background retry task
        self._retry_task: asyncio.Task | None = None
    
    def register_webhook(self, config: WebhookConfig) -> WebhookConfig:
        """Register a new webhook."""
        self._webhooks[config.id] = config
        logger.info(f"Registered webhook {config.id} for {config.url}")
        return config
    
    def unregister_webhook(self, webhook_id: UUID) -> bool:
        """Unregister a webhook."""
        if webhook_id in self._webhooks:
            del self._webhooks[webhook_id]
            logger.info(f"Unregistered webhook {webhook_id}")
            return True
        return False
    
    def get_webhooks_for_org(self, organization_id: UUID) -> list[WebhookConfig]:
        """Get all webhooks for an organization."""
        return [
            w for w in self._webhooks.values()
            if w.organization_id == organization_id and w.enabled
        ]
    
    def get_webhooks_for_event(
        self,
        organization_id: UUID,
        event: str
    ) -> list[WebhookConfig]:
        """Get webhooks that subscribe to a specific event."""
        return [
            w for w in self.get_webhooks_for_org(organization_id)
            if event in w.events
        ]
    
    def _generate_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature for webhook payload."""
        return "sha256=" + hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
    
    async def deliver_webhook(
        self,
        webhook: WebhookConfig,
        event: str,
        payload: dict[str, Any],
        delivery: WebhookDelivery | None = None,
    ) -> WebhookDelivery:
        """Deliver a webhook notification with exponential backoff retry."""
        
        # Create new delivery or use existing (for retries)
        if delivery is None:
            delivery = WebhookDelivery(
                webhook_id=webhook.id,
                event=event,
                payload=payload,
                max_attempts=webhook.max_retries,
            )
        
        # Prepare payload
        full_payload = {
            "event": event,
            "timestamp": datetime.utcnow().isoformat(),
            "delivery_id": str(delivery.id),
            "attempt": delivery.attempts + 1,
            "data": payload,
        }
        payload_str = json.dumps(full_payload, default=str)
        
        # Generate signature
        signature = self._generate_signature(payload_str, webhook.secret)
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "CodeVerify-Webhook/1.0",
            "X-CodeVerify-Event": event,
            "X-CodeVerify-Signature": signature,
            "X-CodeVerify-Delivery": str(delivery.id),
            "X-CodeVerify-Attempt": str(delivery.attempts + 1),
        }
        
        start_time = datetime.utcnow()
        delivery.attempts += 1
        delivery.last_attempt_at = start_time
        delivery.status = WebhookStatus.PENDING
        
        try:
            async with httpx.AsyncClient(timeout=webhook.timeout_seconds) as client:
                response = await client.post(
                    webhook.url,
                    content=payload_str,
                    headers=headers,
                )
                
                end_time = datetime.utcnow()
                latency_ms = (end_time - start_time).total_seconds() * 1000
                
                delivery.response_code = response.status_code
                delivery.response_body = response.text[:1000]
                
                if 200 <= response.status_code < 300:
                    # Success
                    delivery.status = WebhookStatus.SUCCESS
                    delivery.delivered_at = end_time
                    
                    self.metrics.record_delivery(
                        success=True,
                        attempts=delivery.attempts,
                        latency_ms=latency_ms,
                    )
                    
                    logger.info(
                        f"Webhook delivered: {delivery.id} to {webhook.url} "
                        f"(attempt {delivery.attempts}, {latency_ms:.0f}ms)"
                    )
                else:
                    # Non-2xx response
                    delivery.error_message = f"HTTP {response.status_code}"
                    await self._handle_failure(webhook, delivery)
                    
        except httpx.TimeoutException:
            delivery.error_message = "Request timed out"
            await self._handle_failure(webhook, delivery)
            
        except httpx.RequestError as e:
            delivery.error_message = f"Request error: {str(e)}"
            await self._handle_failure(webhook, delivery)
            
        except Exception as e:
            delivery.error_message = f"Unexpected error: {str(e)}"
            logger.exception(f"Unexpected webhook error for {delivery.id}")
            await self._handle_failure(webhook, delivery)
        
        self._deliveries.append(delivery)
        return delivery
    
    async def _handle_failure(
        self,
        webhook: WebhookConfig,
        delivery: WebhookDelivery,
    ) -> None:
        """Handle a failed delivery attempt."""
        logger.warning(
            f"Webhook delivery failed: {delivery.id} to {webhook.url}, "
            f"attempt {delivery.attempts}/{delivery.max_attempts}, "
            f"error: {delivery.error_message}"
        )
        
        if delivery.attempts < delivery.max_attempts:
            # Schedule retry
            delivery.status = WebhookStatus.RETRYING
            delivery.next_retry_at = self.retry_strategy.get_next_retry_time(delivery.attempts)
            self._pending_retries.append(delivery)
            
            logger.info(
                f"Webhook {delivery.id} scheduled for retry at {delivery.next_retry_at}"
            )
        else:
            # Max retries exceeded - move to dead letter queue
            delivery.status = WebhookStatus.FAILED
            self.dead_letter_queue.add(delivery)
            self.metrics.record_dead_letter()
            self.metrics.record_delivery(success=False, attempts=delivery.attempts)
    
    async def process_pending_retries(self) -> int:
        """Process any pending retries that are due."""
        now = datetime.utcnow()
        processed = 0
        
        # Find retries that are due
        due_retries = [
            d for d in self._pending_retries
            if d.next_retry_at and d.next_retry_at <= now
        ]
        
        for delivery in due_retries:
            self._pending_retries.remove(delivery)
            
            webhook = self._webhooks.get(delivery.webhook_id)
            if not webhook or not webhook.enabled:
                logger.warning(f"Webhook {delivery.webhook_id} not found or disabled, skipping retry")
                delivery.status = WebhookStatus.FAILED
                continue
            
            # Retry delivery
            await self.deliver_webhook(
                webhook=webhook,
                event=delivery.event,
                payload=delivery.payload,
                delivery=delivery,
            )
            processed += 1
        
        return processed
    
    async def start_retry_worker(self, interval: float = 30.0) -> None:
        """Start background worker to process retries."""
        async def worker():
            while True:
                try:
                    processed = await self.process_pending_retries()
                    if processed > 0:
                        logger.debug(f"Processed {processed} webhook retries")
                except Exception as e:
                    logger.error(f"Retry worker error: {e}")
                
                await asyncio.sleep(interval)
        
        self._retry_task = asyncio.create_task(worker())
        logger.info("Webhook retry worker started")
    
    def stop_retry_worker(self) -> None:
        """Stop the background retry worker."""
        if self._retry_task:
            self._retry_task.cancel()
            self._retry_task = None
            logger.info("Webhook retry worker stopped")
    
    async def emit_event(
        self,
        organization_id: UUID,
        event: str,
        payload: dict[str, Any]
    ) -> list[WebhookDelivery]:
        """Emit an event to all subscribed webhooks."""
        webhooks = self.get_webhooks_for_event(organization_id, event)
        
        if not webhooks:
            logger.debug(f"No webhooks for event {event} in org {organization_id}")
            return []
        
        # Deliver to all webhooks concurrently
        tasks = [
            self.deliver_webhook(webhook, event, payload)
            for webhook in webhooks
        ]
        
        return await asyncio.gather(*tasks)
    
    def get_delivery_status(self, delivery_id: UUID) -> WebhookDelivery | None:
        """Get the status of a specific delivery."""
        for d in self._deliveries:
            if d.id == delivery_id:
                return d
        return None
    
    def get_pending_retries(self) -> list[WebhookDelivery]:
        """Get all pending retries."""
        return self._pending_retries.copy()
    
    def get_metrics(self) -> dict[str, Any]:
        """Get webhook delivery metrics."""
        return {
            **self.metrics.get_stats(),
            "pending_retries": len(self._pending_retries),
            "dead_letter_size": len(self.dead_letter_queue.get_all()),
        }


# Global instance
webhook_service = WebhookService()


# Event emission helpers
async def emit_analysis_started(
    organization_id: UUID,
    analysis_id: UUID,
    repository: str,
    pr_number: int | None
):
    """Emit analysis.started event."""
    await webhook_service.emit_event(
        organization_id,
        "analysis.started",
        {
            "analysis_id": str(analysis_id),
            "repository": repository,
            "pr_number": pr_number,
        }
    )


async def emit_analysis_completed(
    organization_id: UUID,
    analysis_id: UUID,
    repository: str,
    pr_number: int | None,
    conclusion: str,
    summary: dict[str, Any]
):
    """Emit analysis.completed event."""
    await webhook_service.emit_event(
        organization_id,
        "analysis.completed",
        {
            "analysis_id": str(analysis_id),
            "repository": repository,
            "pr_number": pr_number,
            "conclusion": conclusion,
            "summary": summary,
        }
    )


async def emit_analysis_failed(
    organization_id: UUID,
    analysis_id: UUID,
    repository: str,
    error: str
):
    """Emit analysis.failed event."""
    await webhook_service.emit_event(
        organization_id,
        "analysis.failed",
        {
            "analysis_id": str(analysis_id),
            "repository": repository,
            "error": error,
        }
    )


async def emit_finding_created(
    organization_id: UUID,
    analysis_id: UUID,
    finding_id: UUID,
    severity: str,
    category: str,
    title: str
):
    """Emit finding.created event."""
    await webhook_service.emit_event(
        organization_id,
        "finding.created",
        {
            "analysis_id": str(analysis_id),
            "finding_id": str(finding_id),
            "severity": severity,
            "category": category,
            "title": title,
        }
    )
