"""Redis-backed storage for v0.8.0 multi-tenant and plugin features.

Provides persistent Redis implementations for:
- TenantManager: tenant configs, quotas, and usage records
- PluginRegistry: plugin manifests, reviews, and search index
- CostOptimizer: budget tracking and model routing stats

Follows the same pattern as RedisCacheBackend in verification_cache.py.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any

import structlog

logger = structlog.get_logger()


class _RedisBase:
    """Shared Redis connection logic."""

    PREFIX = "codeverify:"

    def __init__(self, redis_url: str = "redis://localhost:6379/2") -> None:
        self._redis_url = redis_url
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import redis

                self._client = redis.from_url(self._redis_url, decode_responses=True)
            except ImportError:
                logger.warning("redis package not installed")
                raise
        return self._client


class RedisTenantStore(_RedisBase):
    """Redis-backed storage for tenant configurations and usage."""

    PREFIX = "codeverify:tenants:"

    def save_tenant(self, tenant_id: str, tenant_data: dict[str, Any]) -> None:
        """Persist a tenant configuration."""
        try:
            client = self._get_client()
            client.set(
                f"{self.PREFIX}{tenant_id}",
                json.dumps(tenant_data, default=str),
            )
        except Exception as e:
            logger.warning("Failed to save tenant", tenant_id=tenant_id, error=str(e))

    def load_tenant(self, tenant_id: str) -> dict[str, Any] | None:
        """Load a tenant configuration."""
        try:
            client = self._get_client()
            data = client.get(f"{self.PREFIX}{tenant_id}")
            return json.loads(data) if data else None
        except Exception as e:
            logger.warning("Failed to load tenant", tenant_id=tenant_id, error=str(e))
            return None

    def delete_tenant(self, tenant_id: str) -> None:
        """Remove a tenant."""
        try:
            client = self._get_client()
            client.delete(f"{self.PREFIX}{tenant_id}")
            # Also clean up usage records
            keys = client.keys(f"{self.PREFIX}{tenant_id}:usage:*")
            if keys:
                client.delete(*keys)
        except Exception as e:
            logger.warning("Failed to delete tenant", tenant_id=tenant_id, error=str(e))

    def record_usage(self, tenant_id: str, record: dict[str, Any]) -> None:
        """Append a usage record with 30-day TTL."""
        try:
            client = self._get_client()
            key = f"{self.PREFIX}{tenant_id}:usage"
            entry = json.dumps({**record, "timestamp": time.time()}, default=str)
            client.rpush(key, entry)
            client.expire(key, 30 * 86400)  # 30 days
        except Exception as e:
            logger.warning("Failed to record usage", error=str(e))

    def get_usage(self, tenant_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent usage records for a tenant."""
        try:
            client = self._get_client()
            key = f"{self.PREFIX}{tenant_id}:usage"
            entries = client.lrange(key, -limit, -1)
            return [json.loads(e) for e in entries]
        except Exception as e:
            logger.warning("Failed to get usage", error=str(e))
            return []

    def list_tenants(self) -> list[str]:
        """List all tenant IDs."""
        try:
            client = self._get_client()
            keys = client.keys(f"{self.PREFIX}*")
            prefix_len = len(self.PREFIX)
            return [
                k[prefix_len:]
                for k in keys
                if ":" not in k[prefix_len:]  # Exclude sub-keys
            ]
        except Exception as e:
            logger.warning("Failed to list tenants", error=str(e))
            return []


class RedisPluginStore(_RedisBase):
    """Redis-backed storage for plugin registry."""

    PREFIX = "codeverify:plugins:"

    def save_plugin(self, plugin_id: str, manifest: dict[str, Any]) -> None:
        """Store a plugin manifest."""
        try:
            client = self._get_client()
            client.set(
                f"{self.PREFIX}{plugin_id}",
                json.dumps(manifest, default=str),
            )
            # Index by name for search
            name = manifest.get("name", plugin_id)
            client.sadd(f"{self.PREFIX}_index", plugin_id)
            client.set(f"{self.PREFIX}name:{name}", plugin_id)
        except Exception as e:
            logger.warning("Failed to save plugin", plugin_id=plugin_id, error=str(e))

    def load_plugin(self, plugin_id: str) -> dict[str, Any] | None:
        """Load a plugin manifest."""
        try:
            client = self._get_client()
            data = client.get(f"{self.PREFIX}{plugin_id}")
            return json.loads(data) if data else None
        except Exception as e:
            logger.warning("Failed to load plugin", plugin_id=plugin_id, error=str(e))
            return None

    def delete_plugin(self, plugin_id: str) -> None:
        """Remove a plugin."""
        try:
            client = self._get_client()
            manifest = self.load_plugin(plugin_id)
            client.delete(f"{self.PREFIX}{plugin_id}")
            client.srem(f"{self.PREFIX}_index", plugin_id)
            if manifest:
                client.delete(f"{self.PREFIX}name:{manifest.get('name', '')}")
        except Exception as e:
            logger.warning("Failed to delete plugin", plugin_id=plugin_id, error=str(e))

    def search_plugins(self, query: str) -> list[dict[str, Any]]:
        """Search plugins by name substring."""
        try:
            client = self._get_client()
            all_ids = client.smembers(f"{self.PREFIX}_index")
            results = []
            for pid in all_ids:
                manifest = self.load_plugin(pid)
                if manifest and query.lower() in manifest.get("name", "").lower():
                    results.append(manifest)
            return results
        except Exception as e:
            logger.warning("Failed to search plugins", error=str(e))
            return []

    def save_review(self, plugin_id: str, review: dict[str, Any]) -> None:
        """Add a review to a plugin."""
        try:
            client = self._get_client()
            key = f"{self.PREFIX}{plugin_id}:reviews"
            client.rpush(key, json.dumps({**review, "timestamp": time.time()}, default=str))
        except Exception as e:
            logger.warning("Failed to save review", error=str(e))

    def get_reviews(self, plugin_id: str) -> list[dict[str, Any]]:
        """Get all reviews for a plugin."""
        try:
            client = self._get_client()
            key = f"{self.PREFIX}{plugin_id}:reviews"
            entries = client.lrange(key, 0, -1)
            return [json.loads(e) for e in entries]
        except Exception as e:
            logger.warning("Failed to get reviews", error=str(e))
            return []

    def list_plugins(self) -> list[str]:
        """List all plugin IDs."""
        try:
            client = self._get_client()
            return list(client.smembers(f"{self.PREFIX}_index"))
        except Exception as e:
            logger.warning("Failed to list plugins", error=str(e))
            return []


class RedisCostStore(_RedisBase):
    """Redis-backed storage for LLM cost optimizer budget tracking."""

    PREFIX = "codeverify:costs:"

    def record_call(self, model: str, tokens: int, cost: float, complexity: str) -> None:
        """Record an LLM call for budget tracking."""
        try:
            client = self._get_client()
            entry = json.dumps({
                "model": model,
                "tokens": tokens,
                "cost": cost,
                "complexity": complexity,
                "timestamp": time.time(),
            })
            client.rpush(f"{self.PREFIX}calls", entry)
            # Roll up daily totals
            day_key = time.strftime("%Y-%m-%d")
            client.incrbyfloat(f"{self.PREFIX}daily:{day_key}", cost)
            client.expire(f"{self.PREFIX}daily:{day_key}", 90 * 86400)  # 90 days
        except Exception as e:
            logger.warning("Failed to record cost", error=str(e))

    def get_daily_spend(self, date: str | None = None) -> float:
        """Get total spend for a given day (defaults to today)."""
        try:
            client = self._get_client()
            day_key = date or time.strftime("%Y-%m-%d")
            val = client.get(f"{self.PREFIX}daily:{day_key}")
            return float(val) if val else 0.0
        except Exception as e:
            logger.warning("Failed to get daily spend", error=str(e))
            return 0.0

    def get_monthly_spend(self) -> float:
        """Get total spend for the current month."""
        try:
            client = self._get_client()
            prefix = time.strftime("%Y-%m")
            keys = client.keys(f"{self.PREFIX}daily:{prefix}-*")
            total = 0.0
            for key in keys:
                val = client.get(key)
                if val:
                    total += float(val)
            return total
        except Exception as e:
            logger.warning("Failed to get monthly spend", error=str(e))
            return 0.0

    def set_budget(self, monthly_budget: float) -> None:
        """Set the monthly budget limit."""
        try:
            client = self._get_client()
            client.set(f"{self.PREFIX}budget", str(monthly_budget))
        except Exception as e:
            logger.warning("Failed to set budget", error=str(e))

    def get_budget(self) -> float | None:
        """Get the monthly budget limit."""
        try:
            client = self._get_client()
            val = client.get(f"{self.PREFIX}budget")
            return float(val) if val else None
        except Exception as e:
            logger.warning("Failed to get budget", error=str(e))
            return None

    def is_over_budget(self) -> bool:
        """Check if current month's spend exceeds budget."""
        budget = self.get_budget()
        if budget is None:
            return False
        return self.get_monthly_spend() >= budget
