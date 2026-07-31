"""Plugin Marketplace & SDK.

Provides plugin interfaces for rules, language adapters, agents, and
visualizations. Includes a registry API, SDK base classes, and a
CLI-oriented publish/install/search workflow.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class PluginType(str, Enum):
    """Type of plugin."""

    RULE = "rule"
    LANGUAGE_ADAPTER = "language_adapter"
    AGENT = "agent"
    VISUALIZATION = "visualization"
    INTEGRATION = "integration"


class PluginStatus(str, Enum):
    """Publication status of a plugin."""

    DRAFT = "draft"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"
    SUSPENDED = "suspended"


@dataclass
class PluginManifest:
    """Manifest describing a plugin."""

    name: str
    version: str
    description: str
    plugin_type: PluginType
    author: str = ""
    author_email: str = ""
    license: str = "MIT"
    homepage: str = ""
    repository: str = ""
    keywords: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    min_codeverify_version: str = "0.8.0"
    entry_point: str = ""

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "type": self.plugin_type.value,
            "author": self.author,
            "license": self.license,
            "keywords": self.keywords,
            "dependencies": self.dependencies,
            "min_codeverify_version": self.min_codeverify_version,
            "entry_point": self.entry_point,
        }


@dataclass
class PluginEntry:
    """A plugin in the registry."""

    manifest: PluginManifest
    status: PluginStatus = PluginStatus.PUBLISHED
    downloads: int = 0
    rating: float = 0.0
    rating_count: int = 0
    published_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    checksum: str = ""

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def version(self) -> str:
        return self.manifest.version


@dataclass
class PluginReview:
    """A review of a plugin."""

    plugin_name: str
    reviewer: str = ""
    rating: int = 5
    comment: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class PluginSearchResult:
    """Search result from the plugin registry."""

    plugins: list[PluginEntry] = field(default_factory=list)
    total_count: int = 0
    query: str = ""


class PluginSDK:
    """SDK for building CodeVerify plugins."""

    @staticmethod
    def create_manifest(
        name: str,
        version: str,
        description: str,
        plugin_type: PluginType,
        author: str = "",
        **kwargs: Any,
    ) -> PluginManifest:
        """Create a plugin manifest."""
        return PluginManifest(
            name=name,
            version=version,
            description=description,
            plugin_type=plugin_type,
            author=author,
            **kwargs,
        )

    @staticmethod
    def validate_manifest(manifest: PluginManifest) -> list[str]:
        """Validate a plugin manifest. Returns list of errors."""
        errors: list[str] = []
        if not manifest.name:
            errors.append("Plugin name is required")
        if not manifest.version:
            errors.append("Plugin version is required")
        if not manifest.description:
            errors.append("Plugin description is required")
        if len(manifest.name) > 64:
            errors.append("Plugin name must be <= 64 characters")
        if not all(c.isalnum() or c in "-_" for c in manifest.name):
            errors.append("Plugin name must contain only alphanumeric, dash, or underscore")
        return errors

    @staticmethod
    def compute_checksum(content: str) -> str:
        """Compute a SHA-256 checksum for plugin content."""
        return hashlib.sha256(content.encode()).hexdigest()


class PluginRegistry:
    """In-memory plugin registry."""

    def __init__(self) -> None:
        self._plugins: dict[str, PluginEntry] = {}
        self._reviews: dict[str, list[PluginReview]] = {}

    def publish(self, manifest: PluginManifest, content: str = "") -> PluginEntry:
        """Publish a plugin to the registry."""
        errors = PluginSDK.validate_manifest(manifest)
        if errors:
            raise ValueError(f"Invalid manifest: {'; '.join(errors)}")

        entry = PluginEntry(
            manifest=manifest,
            checksum=PluginSDK.compute_checksum(content or json.dumps(manifest.to_dict())),
        )
        self._plugins[manifest.id] = entry
        logger.info("plugin_published", name=manifest.name, version=manifest.version)
        return entry

    def get(self, name: str, version: str | None = None) -> PluginEntry | None:
        """Get a plugin by name and optional version."""
        if version:
            return self._plugins.get(f"{name}@{version}")
        # Return latest version
        matching = [
            e
            for e in self._plugins.values()
            if e.name == name and e.status == PluginStatus.PUBLISHED
        ]
        if not matching:
            return None
        return max(matching, key=lambda e: e.version)

    def search(
        self,
        query: str = "",
        plugin_type: PluginType | None = None,
        limit: int = 20,
    ) -> PluginSearchResult:
        """Search for plugins."""
        results = list(self._plugins.values())
        results = [p for p in results if p.status == PluginStatus.PUBLISHED]

        if query:
            q = query.lower()
            results = [
                p
                for p in results
                if q in p.name.lower()
                or q in p.manifest.description.lower()
                or any(q in kw.lower() for kw in p.manifest.keywords)
            ]

        if plugin_type:
            results = [p for p in results if p.manifest.plugin_type == plugin_type]

        results.sort(key=lambda p: p.downloads, reverse=True)
        total = len(results)
        results = results[:limit]

        return PluginSearchResult(plugins=results, total_count=total, query=query)

    def install(self, name: str, version: str | None = None) -> PluginEntry | None:
        """Simulate installing a plugin (increment download count)."""
        entry = self.get(name, version)
        if entry:
            entry.downloads += 1
            logger.info("plugin_installed", name=name, version=entry.version)
        return entry

    def add_review(self, plugin_name: str, review: PluginReview) -> bool:
        """Add a review for a plugin."""
        entry = self.get(plugin_name)
        if entry is None:
            return False
        self._reviews.setdefault(plugin_name, []).append(review)
        # Update rating
        reviews = self._reviews[plugin_name]
        entry.rating = sum(r.rating for r in reviews) / len(reviews)
        entry.rating_count = len(reviews)
        return True

    def get_reviews(self, plugin_name: str) -> list[PluginReview]:
        return list(self._reviews.get(plugin_name, []))

    def list_by_type(self, plugin_type: PluginType) -> list[PluginEntry]:
        return [
            p
            for p in self._plugins.values()
            if p.manifest.plugin_type == plugin_type and p.status == PluginStatus.PUBLISHED
        ]

    def deprecate(self, name: str, version: str) -> bool:
        key = f"{name}@{version}"
        entry = self._plugins.get(key)
        if entry:
            entry.status = PluginStatus.DEPRECATED
            return True
        return False

    def stats(self) -> dict[str, Any]:
        """Return registry statistics."""
        published = [p for p in self._plugins.values() if p.status == PluginStatus.PUBLISHED]
        return {
            "total_plugins": len(published),
            "total_downloads": sum(p.downloads for p in published),
            "by_type": {
                t.value: sum(1 for p in published if p.manifest.plugin_type == t)
                for t in PluginType
            },
            "avg_rating": (
                sum(p.rating for p in published if p.rating_count > 0)
                / max(1, sum(1 for p in published if p.rating_count > 0))
            ),
        }


# Singleton
_plugin_registry: PluginRegistry | None = None


def get_plugin_registry() -> PluginRegistry:
    global _plugin_registry
    if _plugin_registry is None:
        _plugin_registry = PluginRegistry()
    return _plugin_registry


def reset_plugin_registry() -> None:
    global _plugin_registry
    _plugin_registry = None
