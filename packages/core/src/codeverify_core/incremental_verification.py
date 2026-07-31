"""Incremental Verification Engine.

Content-addressed caching of verification results with dependency-aware
invalidation for fast re-verification of changed code.

Features:
- Content-addressed hashing for code blocks (function/class level)
- Dependency graph tracking between code units
- Cache hit/miss tracking with metrics
- Automatic invalidation on dependency changes
- Fallback to full verification when cache is cold
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class CacheStatus(str, Enum):
    """Status of a cache lookup."""

    HIT = "hit"
    MISS = "miss"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"


class VerificationStatus(str, Enum):
    """Status of a cached verification result."""

    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"


@dataclass
class CodeUnit:
    """A discrete unit of code for verification (function, class, block)."""

    id: str = ""
    file_path: str = ""
    name: str = ""
    start_line: int = 0
    end_line: int = 0
    content: str = ""
    language: str = "python"
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash and self.content:
            self.content_hash = self._compute_hash()
        if not self.id:
            self.id = f"{self.file_path}:{self.name}"

    def _compute_hash(self) -> str:
        normalized = self.content.strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]


@dataclass
class CachedResult:
    """A cached verification result for a code unit."""

    code_unit_id: str = ""
    content_hash: str = ""
    status: VerificationStatus = VerificationStatus.UNKNOWN
    findings: list[dict[str, Any]] = field(default_factory=list)
    cached_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    ttl_seconds: int = 86400  # 24 hours
    verification_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        elapsed = (datetime.now(UTC) - self.cached_at).total_seconds()
        return elapsed > self.ttl_seconds

    @property
    def is_safe(self) -> bool:
        return self.status == VerificationStatus.SAFE


@dataclass
class DependencyEdge:
    """A dependency between two code units."""

    source_id: str = ""
    target_id: str = ""
    dep_type: str = "calls"  # calls, imports, inherits


class DependencyGraph:
    """Tracks dependencies between code units for cache invalidation."""

    def __init__(self) -> None:
        self._edges: list[DependencyEdge] = []
        self._forward: dict[str, set[str]] = defaultdict(set)  # A depends on B
        self._reverse: dict[str, set[str]] = defaultdict(set)  # B is depended on by A

    def add_dependency(self, source_id: str, target_id: str, dep_type: str = "calls") -> None:
        edge = DependencyEdge(source_id=source_id, target_id=target_id, dep_type=dep_type)
        self._edges.append(edge)
        self._forward[source_id].add(target_id)
        self._reverse[target_id].add(source_id)

    def get_dependents(self, unit_id: str) -> set[str]:
        """Get all units that depend on the given unit (transitive)."""
        visited: set[str] = set()
        stack = [unit_id]
        while stack:
            current = stack.pop()
            for dep in self._reverse.get(current, set()):
                if dep not in visited:
                    visited.add(dep)
                    stack.append(dep)
        return visited

    def get_dependencies(self, unit_id: str) -> set[str]:
        """Get all units that the given unit depends on (direct)."""
        return self._forward.get(unit_id, set()).copy()

    def remove_unit(self, unit_id: str) -> None:
        self._forward.pop(unit_id, None)
        self._reverse.pop(unit_id, None)
        for deps in self._forward.values():
            deps.discard(unit_id)
        for deps in self._reverse.values():
            deps.discard(unit_id)

    @property
    def total_edges(self) -> int:
        return sum(len(deps) for deps in self._forward.values())

    @property
    def total_units(self) -> int:
        all_ids = set(self._forward.keys()) | set(self._reverse.keys())
        return len(all_ids)


@dataclass
class CacheMetrics:
    """Metrics for cache performance."""

    hits: int = 0
    misses: int = 0
    invalidations: int = 0
    expirations: int = 0
    total_saved_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_lookups(self) -> int:
        return self.hits + self.misses

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "invalidations": self.invalidations,
            "expirations": self.expirations,
            "hit_rate": round(self.hit_rate, 3),
            "total_saved_ms": round(self.total_saved_ms, 1),
        }


class IncrementalVerificationEngine:
    """Caches verification results and only re-verifies changed code."""

    def __init__(self, ttl_seconds: int = 86400) -> None:
        self._cache: dict[str, CachedResult] = {}
        self._dep_graph = DependencyGraph()
        self._metrics = CacheMetrics()
        self._ttl = ttl_seconds

    def lookup(self, unit: CodeUnit) -> tuple[CacheStatus, CachedResult | None]:
        """Look up a cached result for a code unit."""
        cached = self._cache.get(unit.id)
        if cached is None:
            self._metrics.misses += 1
            return CacheStatus.MISS, None

        if cached.is_expired:
            self._metrics.expirations += 1
            del self._cache[unit.id]
            return CacheStatus.EXPIRED, None

        if cached.content_hash != unit.content_hash:
            self._metrics.invalidations += 1
            del self._cache[unit.id]
            return CacheStatus.INVALIDATED, None

        self._metrics.hits += 1
        self._metrics.total_saved_ms += cached.verification_time_ms
        return CacheStatus.HIT, cached

    def store(
        self,
        unit: CodeUnit,
        status: VerificationStatus,
        findings: list[dict[str, Any]] | None = None,
        verification_time_ms: float = 0.0,
    ) -> CachedResult:
        """Store a verification result in the cache."""
        result = CachedResult(
            code_unit_id=unit.id,
            content_hash=unit.content_hash,
            status=status,
            findings=findings or [],
            ttl_seconds=self._ttl,
            verification_time_ms=verification_time_ms,
        )
        self._cache[unit.id] = result
        return result

    def invalidate(self, unit_id: str, cascade: bool = True) -> list[str]:
        """Invalidate a cache entry, optionally cascading to dependents."""
        invalidated = [unit_id] if unit_id in self._cache else []
        self._cache.pop(unit_id, None)

        if cascade:
            for dep_id in self._dep_graph.get_dependents(unit_id):
                if dep_id in self._cache:
                    del self._cache[dep_id]
                    invalidated.append(dep_id)
                    self._metrics.invalidations += 1

        return invalidated

    def add_dependency(self, source_id: str, target_id: str, dep_type: str = "calls") -> None:
        self._dep_graph.add_dependency(source_id, target_id, dep_type)

    def get_units_to_verify(self, units: list[CodeUnit]) -> list[CodeUnit]:
        """Filter units to only those that need re-verification."""
        to_verify = []
        for unit in units:
            status, _ = self.lookup(unit)
            if status != CacheStatus.HIT:
                to_verify.append(unit)
        return to_verify

    @property
    def metrics(self) -> CacheMetrics:
        return self._metrics

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    @property
    def dependency_graph(self) -> DependencyGraph:
        return self._dep_graph

    def clear(self) -> None:
        self._cache.clear()
        self._dep_graph = DependencyGraph()
        self._metrics = CacheMetrics()


_engine: IncrementalVerificationEngine | None = None


def get_incremental_engine() -> IncrementalVerificationEngine:
    """Get the singleton IncrementalVerificationEngine instance."""
    global _engine
    if _engine is None:
        _engine = IncrementalVerificationEngine()
    return _engine


def reset_incremental_engine() -> None:
    """Reset the singleton (useful for testing)."""
    global _engine
    _engine = None
