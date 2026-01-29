"""Repository pattern abstractions for storage backends.

This module provides repository interfaces and in-memory implementations
for persistence concerns. Production deployments should implement
database-backed versions of these interfaces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar
from uuid import UUID

import structlog

logger = structlog.get_logger()

# Generic type variable for repository entities
T = TypeVar("T")


class Repository(ABC, Generic[T]):
    """Abstract base class for repositories.
    
    Repositories provide a collection-like interface for accessing domain
    entities while abstracting away the underlying storage mechanism.
    """

    @abstractmethod
    async def get(self, id: str | UUID) -> T | None:
        """Get an entity by ID."""
        pass

    @abstractmethod
    async def save(self, entity: T) -> T:
        """Save an entity (create or update)."""
        pass

    @abstractmethod
    async def delete(self, id: str | UUID) -> bool:
        """Delete an entity by ID. Returns True if deleted."""
        pass

    @abstractmethod
    async def list(self, **filters: Any) -> list[T]:
        """List entities with optional filters."""
        pass


class InMemoryRepository(Repository[T]):
    """In-memory implementation of Repository.
    
    Suitable for testing and development. Not suitable for production
    use as data is not persisted across restarts.
    """

    def __init__(self) -> None:
        self._storage: dict[str, T] = {}

    def _get_id(self, entity: T) -> str:
        """Extract ID from an entity. Override if ID field is not 'id'."""
        if hasattr(entity, "id"):
            id_val = getattr(entity, "id")
            return str(id_val) if id_val else ""
        raise ValueError(f"Entity {type(entity)} has no 'id' attribute")

    async def get(self, id: str | UUID) -> T | None:
        return self._storage.get(str(id))

    async def save(self, entity: T) -> T:
        entity_id = self._get_id(entity)
        self._storage[entity_id] = entity
        return entity

    async def delete(self, id: str | UUID) -> bool:
        return self._storage.pop(str(id), None) is not None

    async def list(self, **filters: Any) -> list[T]:
        """List entities, applying filters by attribute matching."""
        results = list(self._storage.values())
        for key, value in filters.items():
            results = [
                e for e in results
                if hasattr(e, key) and getattr(e, key) == value
            ]
        return results


# Type-specific repository interfaces for explicit contracts


class ScanResultRepository(Repository["CodebaseScanResult"]):
    """Repository interface for scan results."""

    @abstractmethod
    async def get_by_repo(
        self, repo_full_name: str, limit: int = 10
    ) -> list["CodebaseScanResult"]:
        """Get scan history for a repository."""
        pass

    @abstractmethod
    async def get_completed_since(
        self, repo_full_name: str, since: datetime
    ) -> list["CodebaseScanResult"]:
        """Get completed scans since a given date."""
        pass


class ScheduledScanRepository(Repository["ScheduledScan"]):
    """Repository interface for scheduled scans."""

    @abstractmethod
    async def get_due_scans(self, before: datetime) -> list["ScheduledScan"]:
        """Get scheduled scans due to run before the given time."""
        pass

    @abstractmethod
    async def get_by_repo(self, repo_full_name: str) -> list["ScheduledScan"]:
        """Get all scheduled scans for a repository."""
        pass


class NotificationConfigRepository(Repository["NotificationConfig"]):
    """Repository interface for notification configurations."""

    @abstractmethod
    async def get_by_repo(self, repo_full_name: str) -> list["NotificationConfig"]:
        """Get all notification configs for a repository."""
        pass

    @abstractmethod
    async def add_for_repo(
        self, repo_full_name: str, config: "NotificationConfig"
    ) -> "NotificationConfig":
        """Add a notification config for a repository."""
        pass


# Forward references for type hints (actual classes are defined elsewhere)
# These are imported at runtime to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeverify_core.notifications import NotificationConfig
    from codeverify_core.scanning import CodebaseScanResult, ScheduledScan


# In-memory implementations


@dataclass
class RepoNotificationConfigs:
    """Wrapper to store configs by repository."""
    id: str  # repo_full_name
    configs: list[Any] = field(default_factory=list)


class InMemoryScanResultRepository(InMemoryRepository["CodebaseScanResult"], ScanResultRepository):
    """In-memory implementation of ScanResultRepository."""

    async def get_by_repo(
        self, repo_full_name: str, limit: int = 10
    ) -> list["CodebaseScanResult"]:
        results = [
            r for r in self._storage.values()
            if r.repo_full_name == repo_full_name
        ]
        results.sort(key=lambda r: r.started_at or datetime.min, reverse=True)
        return results[:limit]

    async def get_completed_since(
        self, repo_full_name: str, since: datetime
    ) -> list["CodebaseScanResult"]:
        from codeverify_core.scanning import ScanStatus
        
        return [
            r for r in self._storage.values()
            if r.repo_full_name == repo_full_name
            and r.started_at
            and r.started_at > since
            and r.status == ScanStatus.COMPLETED
        ]


class InMemoryScheduledScanRepository(InMemoryRepository["ScheduledScan"], ScheduledScanRepository):
    """In-memory implementation of ScheduledScanRepository."""

    async def get_due_scans(self, before: datetime) -> list["ScheduledScan"]:
        return [
            s for s in self._storage.values()
            if s.enabled and s.next_run and s.next_run <= before
        ]

    async def get_by_repo(self, repo_full_name: str) -> list["ScheduledScan"]:
        return [
            s for s in self._storage.values()
            if s.config.repo_full_name == repo_full_name
        ]


class InMemoryNotificationConfigRepository(NotificationConfigRepository):
    """In-memory implementation of NotificationConfigRepository."""

    def __init__(self) -> None:
        self._storage: dict[str, list[Any]] = {}

    async def get(self, id: str | UUID) -> Any | None:
        # Not directly applicable - configs are stored by repo
        return None

    async def save(self, entity: Any) -> Any:
        # Not directly applicable
        raise NotImplementedError("Use add_for_repo instead")

    async def delete(self, id: str | UUID) -> bool:
        # Not directly applicable
        return False

    async def list(self, **filters: Any) -> list[Any]:
        all_configs = []
        for configs in self._storage.values():
            all_configs.extend(configs)
        return all_configs

    async def get_by_repo(self, repo_full_name: str) -> list[Any]:
        return self._storage.get(repo_full_name, [])

    async def add_for_repo(self, repo_full_name: str, config: Any) -> Any:
        if repo_full_name not in self._storage:
            self._storage[repo_full_name] = []
        self._storage[repo_full_name].append(config)
        return config


# Default singleton instances for backward compatibility
# In production, inject proper database-backed implementations

_scan_result_repo: ScanResultRepository | None = None
_scheduled_scan_repo: ScheduledScanRepository | None = None
_notification_config_repo: NotificationConfigRepository | None = None


def get_scan_result_repository() -> ScanResultRepository:
    """Get the scan result repository instance."""
    global _scan_result_repo
    if _scan_result_repo is None:
        _scan_result_repo = InMemoryScanResultRepository()
    return _scan_result_repo


def get_scheduled_scan_repository() -> ScheduledScanRepository:
    """Get the scheduled scan repository instance."""
    global _scheduled_scan_repo
    if _scheduled_scan_repo is None:
        _scheduled_scan_repo = InMemoryScheduledScanRepository()
    return _scheduled_scan_repo


def get_notification_config_repository() -> NotificationConfigRepository:
    """Get the notification config repository instance."""
    global _notification_config_repo
    if _notification_config_repo is None:
        _notification_config_repo = InMemoryNotificationConfigRepository()
    return _notification_config_repo


def set_scan_result_repository(repo: ScanResultRepository) -> None:
    """Set the scan result repository (for dependency injection)."""
    global _scan_result_repo
    _scan_result_repo = repo


def set_scheduled_scan_repository(repo: ScheduledScanRepository) -> None:
    """Set the scheduled scan repository (for dependency injection)."""
    global _scheduled_scan_repo
    _scheduled_scan_repo = repo


def set_notification_config_repository(repo: NotificationConfigRepository) -> None:
    """Set the notification config repository (for dependency injection)."""
    global _notification_config_repo
    _notification_config_repo = repo
