"""Continuous Verification Engine - Incremental analysis for real-time verification.

This module provides:
- Incremental AST parsing with change tracking
- Change detection and impact analysis
- Constraint caching for partial Z3 solving
- Debouncing strategy for continuous verification
"""

import asyncio
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ChangeType(str, Enum):
    """Type of code change."""

    INSERT = "insert"
    DELETE = "delete"
    MODIFY = "modify"
    RENAME = "rename"


class VerificationStatus(str, Enum):
    """Status of verification for a code region."""

    PENDING = "pending"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    WARNING = "warning"
    ERROR = "error"
    STALE = "stale"


@dataclass
class TextRange:
    """Range in a text document."""

    start_line: int
    start_col: int
    end_line: int
    end_col: int

    def contains(self, line: int, col: int) -> bool:
        """Check if position is within range."""
        if line < self.start_line or line > self.end_line:
            return False
        if line == self.start_line and col < self.start_col:
            return False
        if line == self.end_line and col > self.end_col:
            return False
        return True

    def overlaps(self, other: "TextRange") -> bool:
        """Check if ranges overlap."""
        if self.end_line < other.start_line:
            return False
        if self.start_line > other.end_line:
            return False
        if self.end_line == other.start_line and self.end_col < other.start_col:
            return False
        if self.start_line == other.end_line and self.start_col > other.end_col:
            return False
        return True


@dataclass
class TextChange:
    """A change to text content."""

    range: TextRange
    new_text: str
    change_type: ChangeType
    timestamp: float = field(default_factory=time.time)


@dataclass
class ASTNode:
    """Simplified AST node for tracking."""

    id: str
    node_type: str  # function, class, statement, expression
    name: str | None
    range: TextRange
    content_hash: str
    children: list["ASTNode"] = field(default_factory=list)
    parent_id: str | None = None
    verification_status: VerificationStatus = VerificationStatus.PENDING
    last_verified: float | None = None
    constraints: list[str] = field(default_factory=list)


class IncrementalASTParser:
    """Incremental AST parser with change tracking.
    
    Maintains an AST representation that can be efficiently updated
    when code changes occur.
    """

    def __init__(self):
        self.nodes: dict[str, ASTNode] = {}
        self.root_id: str | None = None
        self.content: str = ""
        self.content_lines: list[str] = []
        self._node_by_range: dict[tuple[int, int, int, int], str] = {}

    def parse_full(self, content: str, language: str = "python") -> ASTNode:
        """Parse full content and build AST."""
        self.content = content
        self.content_lines = content.split("\n")
        self.nodes.clear()
        self._node_by_range.clear()

        # Simplified parsing - in production use tree-sitter
        root = self._parse_structure(content, language)
        self.root_id = root.id
        return root

    def _parse_structure(self, content: str, language: str) -> ASTNode:
        """Parse code structure into AST nodes."""
        root_id = str(uuid4())
        root = ASTNode(
            id=root_id,
            node_type="module",
            name=None,
            range=TextRange(0, 0, len(self.content_lines) - 1, len(self.content_lines[-1]) if self.content_lines else 0),
            content_hash=self._hash_content(content),
        )
        self.nodes[root_id] = root

        # Parse based on language
        if language == "python":
            self._parse_python(content, root)
        elif language in ("typescript", "javascript"):
            self._parse_typescript(content, root)
        else:
            self._parse_generic(content, root)

        return root

    def _parse_python(self, content: str, root: ASTNode) -> None:
        """Parse Python code structure."""
        import re

        lines = content.split("\n")
        current_indent = 0
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Function definition
            func_match = re.match(r"^(\s*)(async\s+)?def\s+(\w+)\s*\(", line)
            if func_match:
                indent = len(func_match.group(1))
                is_async = func_match.group(2) is not None
                func_name = func_match.group(3)

                # Find function end (next line with same or less indent that's not empty)
                end_line = i + 1
                while end_line < len(lines):
                    next_line = lines[end_line]
                    if next_line.strip() and not next_line.startswith(" " * (indent + 1)):
                        if not next_line.startswith(" " * (indent + 1)) and next_line.strip():
                            break
                    end_line += 1
                end_line = max(i, end_line - 1)

                func_content = "\n".join(lines[i : end_line + 1])
                node = ASTNode(
                    id=str(uuid4()),
                    node_type="async_function" if is_async else "function",
                    name=func_name,
                    range=TextRange(i, 0, end_line, len(lines[end_line]) if end_line < len(lines) else 0),
                    content_hash=self._hash_content(func_content),
                    parent_id=root.id,
                )
                self.nodes[node.id] = node
                root.children.append(node)
                self._node_by_range[(i, 0, end_line, len(lines[end_line]) if end_line < len(lines) else 0)] = node.id
                i = end_line + 1
                continue

            # Class definition
            class_match = re.match(r"^(\s*)class\s+(\w+)", line)
            if class_match:
                indent = len(class_match.group(1))
                class_name = class_match.group(2)

                # Find class end
                end_line = i + 1
                while end_line < len(lines):
                    next_line = lines[end_line]
                    if next_line.strip() and not next_line.startswith(" " * (indent + 1)):
                        break
                    end_line += 1
                end_line = max(i, end_line - 1)

                class_content = "\n".join(lines[i : end_line + 1])
                node = ASTNode(
                    id=str(uuid4()),
                    node_type="class",
                    name=class_name,
                    range=TextRange(i, 0, end_line, len(lines[end_line]) if end_line < len(lines) else 0),
                    content_hash=self._hash_content(class_content),
                    parent_id=root.id,
                )
                self.nodes[node.id] = node
                root.children.append(node)
                self._node_by_range[(i, 0, end_line, len(lines[end_line]) if end_line < len(lines) else 0)] = node.id
                i = end_line + 1
                continue

            i += 1

    def _parse_typescript(self, content: str, root: ASTNode) -> None:
        """Parse TypeScript/JavaScript code structure."""
        import re

        lines = content.split("\n")
        i = 0
        brace_depth = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Function/method definition
            func_match = re.match(
                r"^(\s*)(export\s+)?(async\s+)?function\s+(\w+)|"
                r"^(\s*)(export\s+)?(const|let|var)\s+(\w+)\s*=\s*(async\s+)?(?:\([^)]*\)|[a-zA-Z_]\w*)\s*=>",
                line,
            )
            if func_match:
                groups = func_match.groups()
                func_name = groups[3] or groups[7]

                # Find function end by counting braces
                end_line = i
                depth = 0
                found_start = False
                for j in range(i, len(lines)):
                    for char in lines[j]:
                        if char == "{":
                            depth += 1
                            found_start = True
                        elif char == "}":
                            depth -= 1
                            if found_start and depth == 0:
                                end_line = j
                                break
                    if found_start and depth == 0:
                        break
                else:
                    end_line = min(i + 50, len(lines) - 1)

                func_content = "\n".join(lines[i : end_line + 1])
                node = ASTNode(
                    id=str(uuid4()),
                    node_type="function",
                    name=func_name,
                    range=TextRange(i, 0, end_line, len(lines[end_line]) if end_line < len(lines) else 0),
                    content_hash=self._hash_content(func_content),
                    parent_id=root.id,
                )
                self.nodes[node.id] = node
                root.children.append(node)
                i = end_line + 1
                continue

            # Class definition
            class_match = re.match(r"^(\s*)(export\s+)?(abstract\s+)?class\s+(\w+)", line)
            if class_match:
                class_name = class_match.group(4)

                # Find class end
                end_line = i
                depth = 0
                found_start = False
                for j in range(i, len(lines)):
                    for char in lines[j]:
                        if char == "{":
                            depth += 1
                            found_start = True
                        elif char == "}":
                            depth -= 1
                            if found_start and depth == 0:
                                end_line = j
                                break
                    if found_start and depth == 0:
                        break
                else:
                    end_line = min(i + 100, len(lines) - 1)

                class_content = "\n".join(lines[i : end_line + 1])
                node = ASTNode(
                    id=str(uuid4()),
                    node_type="class",
                    name=class_name,
                    range=TextRange(i, 0, end_line, len(lines[end_line]) if end_line < len(lines) else 0),
                    content_hash=self._hash_content(class_content),
                    parent_id=root.id,
                )
                self.nodes[node.id] = node
                root.children.append(node)
                i = end_line + 1
                continue

            i += 1

    def _parse_generic(self, content: str, root: ASTNode) -> None:
        """Generic parser for unsupported languages."""
        # Just track the whole content as one node
        pass

    def _hash_content(self, content: str) -> str:
        """Hash content for change detection."""
        return hashlib.md5(content.encode()).hexdigest()

    def apply_change(self, change: TextChange) -> list[str]:
        """Apply a change and return affected node IDs."""
        affected_nodes: list[str] = []

        # Find nodes affected by this change
        for node_id, node in self.nodes.items():
            if node.range.overlaps(change.range):
                affected_nodes.append(node_id)
                node.verification_status = VerificationStatus.STALE

        # Update content
        lines = self.content.split("\n")
        start = change.range.start_line
        end = change.range.end_line

        if change.change_type == ChangeType.INSERT:
            new_lines = change.new_text.split("\n")
            lines = lines[:start] + new_lines + lines[start:]
        elif change.change_type == ChangeType.DELETE:
            lines = lines[:start] + lines[end + 1 :]
        elif change.change_type == ChangeType.MODIFY:
            # Replace affected lines
            new_lines = change.new_text.split("\n")
            lines = lines[:start] + new_lines + lines[end + 1 :]

        self.content = "\n".join(lines)
        self.content_lines = lines

        return affected_nodes

    def get_affected_nodes(self, line: int) -> list[ASTNode]:
        """Get nodes affected by change at line."""
        affected = []
        for node in self.nodes.values():
            if node.range.start_line <= line <= node.range.end_line:
                affected.append(node)
        return affected

    def get_stale_nodes(self) -> list[ASTNode]:
        """Get nodes that need re-verification."""
        return [n for n in self.nodes.values() if n.verification_status == VerificationStatus.STALE]


@dataclass
class CachedConstraint:
    """Cached Z3 constraint for reuse."""

    constraint_id: str
    z3_expr: str  # Serialized Z3 expression
    node_id: str
    content_hash: str
    dependencies: list[str]  # Other constraint IDs this depends on
    created_at: float = field(default_factory=time.time)
    hits: int = 0


class ConstraintCache:
    """Cache for Z3 constraints to enable incremental verification."""

    def __init__(self, max_size: int = 10000, ttl_seconds: float = 3600):
        self.cache: dict[str, CachedConstraint] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._node_constraints: dict[str, set[str]] = defaultdict(set)

    def get(self, content_hash: str) -> CachedConstraint | None:
        """Get cached constraint by content hash."""
        constraint = self.cache.get(content_hash)
        if constraint:
            # Check TTL
            if time.time() - constraint.created_at > self.ttl_seconds:
                self.invalidate(content_hash)
                return None
            constraint.hits += 1
            return constraint
        return None

    def put(
        self,
        content_hash: str,
        z3_expr: str,
        node_id: str,
        dependencies: list[str] | None = None,
    ) -> CachedConstraint:
        """Cache a constraint."""
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        constraint = CachedConstraint(
            constraint_id=str(uuid4()),
            z3_expr=z3_expr,
            node_id=node_id,
            content_hash=content_hash,
            dependencies=dependencies or [],
        )
        self.cache[content_hash] = constraint
        self._node_constraints[node_id].add(content_hash)
        return constraint

    def invalidate(self, content_hash: str) -> None:
        """Invalidate a cached constraint and its dependents."""
        if content_hash not in self.cache:
            return

        constraint = self.cache[content_hash]
        node_id = constraint.node_id

        # Remove from node index
        self._node_constraints[node_id].discard(content_hash)

        # Find and invalidate dependents
        dependents = [
            h for h, c in self.cache.items() if content_hash in c.dependencies
        ]

        # Remove this constraint
        del self.cache[content_hash]

        # Recursively invalidate dependents
        for dep_hash in dependents:
            self.invalidate(dep_hash)

    def invalidate_node(self, node_id: str) -> None:
        """Invalidate all constraints for a node."""
        hashes = list(self._node_constraints.get(node_id, set()))
        for content_hash in hashes:
            self.invalidate(content_hash)

    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        if not self.cache:
            return

        # Sort by hits (LFU) and created_at (LRU)
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: (x[1].hits, x[1].created_at),
        )

        # Remove bottom 10%
        evict_count = max(1, len(sorted_entries) // 10)
        for content_hash, _ in sorted_entries[:evict_count]:
            self.invalidate(content_hash)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(c.hits for c in self.cache.values())
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "total_hits": total_hits,
            "unique_nodes": len(self._node_constraints),
        }


class Debouncer:
    """Debouncing strategy for continuous verification."""

    def __init__(
        self,
        delay_ms: int = 300,
        max_delay_ms: int = 2000,
        adaptive: bool = True,
    ):
        self.delay_ms = delay_ms
        self.max_delay_ms = max_delay_ms
        self.adaptive = adaptive
        self._pending_tasks: dict[str, asyncio.Task] = {}
        self._last_trigger: dict[str, float] = {}
        self._trigger_count: dict[str, int] = defaultdict(int)

    def _calculate_delay(self, key: str) -> float:
        """Calculate adaptive delay based on trigger frequency."""
        if not self.adaptive:
            return self.delay_ms / 1000

        count = self._trigger_count[key]
        last_trigger = self._last_trigger.get(key, 0)
        time_since_last = (time.time() - last_trigger) * 1000

        # Increase delay if triggering frequently
        if time_since_last < self.delay_ms:
            multiplier = min(4, 1 + count / 10)
        else:
            multiplier = 1
            self._trigger_count[key] = 0

        delay_ms = min(self.delay_ms * multiplier, self.max_delay_ms)
        return delay_ms / 1000

    async def debounce(
        self,
        key: str,
        callback: Callable[[], Any],
    ) -> None:
        """Debounce a callback."""
        # Cancel existing pending task
        if key in self._pending_tasks:
            self._pending_tasks[key].cancel()
            try:
                await self._pending_tasks[key]
            except asyncio.CancelledError:
                pass

        self._trigger_count[key] += 1
        delay = self._calculate_delay(key)

        async def delayed_callback():
            await asyncio.sleep(delay)
            self._last_trigger[key] = time.time()
            if asyncio.iscoroutinefunction(callback):
                await callback()
            else:
                callback()

        self._pending_tasks[key] = asyncio.create_task(delayed_callback())

    def cancel(self, key: str) -> None:
        """Cancel pending debounced callback."""
        if key in self._pending_tasks:
            self._pending_tasks[key].cancel()
            del self._pending_tasks[key]

    def cancel_all(self) -> None:
        """Cancel all pending callbacks."""
        for task in self._pending_tasks.values():
            task.cancel()
        self._pending_tasks.clear()


class ChangeImpactAnalyzer:
    """Analyze impact of code changes for targeted re-verification."""

    def __init__(self, parser: IncrementalASTParser):
        self.parser = parser
        self._dependencies: dict[str, set[str]] = defaultdict(set)

    def analyze_impact(self, changed_node_ids: list[str]) -> set[str]:
        """Analyze which nodes need re-verification based on changes."""
        affected = set(changed_node_ids)

        # Add nodes that depend on changed nodes
        for node_id in list(affected):
            affected.update(self._get_dependents(node_id))

        # Add parent nodes (class methods affect class)
        for node_id in list(affected):
            node = self.parser.nodes.get(node_id)
            if node and node.parent_id:
                affected.add(node.parent_id)

        return affected

    def _get_dependents(self, node_id: str) -> set[str]:
        """Get nodes that depend on this node."""
        dependents = set()
        for dep_id, deps in self._dependencies.items():
            if node_id in deps:
                dependents.add(dep_id)
        return dependents

    def register_dependency(self, node_id: str, depends_on: str) -> None:
        """Register a dependency between nodes."""
        self._dependencies[node_id].add(depends_on)

    def clear_dependencies(self, node_id: str) -> None:
        """Clear dependencies for a node."""
        self._dependencies.pop(node_id, None)


class VerificationResult(BaseModel):
    """Result of verifying a node."""

    node_id: str
    status: VerificationStatus
    findings: list[dict[str, Any]] = Field(default_factory=list)
    execution_time_ms: int = 0
    from_cache: bool = False


class ContinuousVerificationEngine:
    """Main engine for continuous real-time verification.
    
    Orchestrates incremental parsing, caching, debouncing,
    and verification for real-time code analysis.
    """

    def __init__(
        self,
        debounce_delay_ms: int = 300,
        cache_max_size: int = 10000,
        cache_ttl_seconds: float = 3600,
    ):
        self.parser = IncrementalASTParser()
        self.cache = ConstraintCache(max_size=cache_max_size, ttl_seconds=cache_ttl_seconds)
        self.debouncer = Debouncer(delay_ms=debounce_delay_ms, adaptive=True)
        self.impact_analyzer = ChangeImpactAnalyzer(self.parser)

        self._verification_queue: asyncio.Queue[str] = asyncio.Queue()
        self._results: dict[str, VerificationResult] = {}
        self._callbacks: list[Callable[[VerificationResult], None]] = []
        self._running = False
        self._worker_task: asyncio.Task | None = None

    async def initialize(self, content: str, language: str = "python") -> None:
        """Initialize engine with file content."""
        self.parser.parse_full(content, language)
        self._running = True
        self._worker_task = asyncio.create_task(self._verification_worker())

    async def shutdown(self) -> None:
        """Shutdown the engine."""
        self._running = False
        self.debouncer.cancel_all()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    def on_result(self, callback: Callable[[VerificationResult], None]) -> None:
        """Register callback for verification results."""
        self._callbacks.append(callback)

    async def on_change(self, change: TextChange) -> None:
        """Handle a code change."""
        # Apply change to parser
        affected_ids = self.parser.apply_change(change)

        # Invalidate cached constraints
        for node_id in affected_ids:
            self.cache.invalidate_node(node_id)

        # Analyze impact
        all_affected = self.impact_analyzer.analyze_impact(affected_ids)

        # Debounce verification
        for node_id in all_affected:
            await self.debouncer.debounce(
                f"verify:{node_id}",
                lambda nid=node_id: self._queue_verification(nid),
            )

    def _queue_verification(self, node_id: str) -> None:
        """Queue a node for verification."""
        self._verification_queue.put_nowait(node_id)

    async def _verification_worker(self) -> None:
        """Background worker for processing verification queue."""
        while self._running:
            try:
                node_id = await asyncio.wait_for(
                    self._verification_queue.get(),
                    timeout=1.0,
                )

                result = await self._verify_node(node_id)
                self._results[node_id] = result

                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(result)
                    except Exception:
                        pass

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def _verify_node(self, node_id: str) -> VerificationResult:
        """Verify a single node."""
        start_time = time.time()
        node = self.parser.nodes.get(node_id)

        if not node:
            return VerificationResult(
                node_id=node_id,
                status=VerificationStatus.ERROR,
                findings=[{"error": "Node not found"}],
            )

        # Check cache
        cached = self.cache.get(node.content_hash)
        if cached:
            node.verification_status = VerificationStatus.VERIFIED
            node.last_verified = time.time()
            return VerificationResult(
                node_id=node_id,
                status=VerificationStatus.VERIFIED,
                from_cache=True,
                execution_time_ms=int((time.time() - start_time) * 1000),
            )

        # Perform verification (placeholder - integrate with actual verifier)
        node.verification_status = VerificationStatus.VERIFYING

        # Simulate verification
        await asyncio.sleep(0.05)  # Placeholder for actual verification

        # Cache result
        self.cache.put(
            content_hash=node.content_hash,
            z3_expr="",  # Would contain actual Z3 constraints
            node_id=node_id,
        )

        node.verification_status = VerificationStatus.VERIFIED
        node.last_verified = time.time()

        return VerificationResult(
            node_id=node_id,
            status=VerificationStatus.VERIFIED,
            execution_time_ms=int((time.time() - start_time) * 1000),
        )

    def get_verification_status(self) -> dict[str, VerificationStatus]:
        """Get verification status for all nodes."""
        return {
            node_id: node.verification_status
            for node_id, node in self.parser.nodes.items()
        }

    def get_node_at_position(self, line: int, col: int) -> ASTNode | None:
        """Get the most specific node at a position."""
        candidates = []
        for node in self.parser.nodes.values():
            if node.range.contains(line, col):
                candidates.append(node)

        if not candidates:
            return None

        # Return most specific (smallest) node
        return min(
            candidates,
            key=lambda n: (n.range.end_line - n.range.start_line),
        )

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "nodes_count": len(self.parser.nodes),
            "cache": self.cache.get_stats(),
            "pending_verifications": self._verification_queue.qsize(),
            "verified_nodes": sum(
                1 for n in self.parser.nodes.values()
                if n.verification_status == VerificationStatus.VERIFIED
            ),
            "stale_nodes": len(self.parser.get_stale_nodes()),
        }


# Quick vs Deep verification modes
class VerificationMode(str, Enum):
    """Verification mode."""

    QUICK = "quick"  # Fast, minimal checks
    STANDARD = "standard"  # Normal verification
    DEEP = "deep"  # Comprehensive verification


class VerificationConfig(BaseModel):
    """Configuration for verification."""

    mode: VerificationMode = VerificationMode.STANDARD
    timeout_ms: int = 5000
    enable_formal: bool = True
    enable_ai: bool = True
    max_findings: int = 100
    skip_unchanged: bool = True


async def create_continuous_engine(
    content: str,
    language: str = "python",
    config: VerificationConfig | None = None,
) -> ContinuousVerificationEngine:
    """Create and initialize a continuous verification engine."""
    config = config or VerificationConfig()

    debounce_delay = {
        VerificationMode.QUICK: 100,
        VerificationMode.STANDARD: 300,
        VerificationMode.DEEP: 500,
    }[config.mode]

    engine = ContinuousVerificationEngine(debounce_delay_ms=debounce_delay)
    await engine.initialize(content, language)
    return engine
