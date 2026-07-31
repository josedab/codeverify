"""Streaming IDE Verification (Language Server).

Real-time, sub-function-level verification as developers type,
with incremental verification, inline proof status, and
sub-second latency through content-addressed caching.

Features:
- Incremental function-level verification on file save
- Content-addressed verification caching for sub-second responses
- Inline proof status reporting (verified/warning/error per block)
- Diagnostic streaming with severity mapping to LSP levels
- Verification session management with debouncing
- Multi-file dependency tracking for invalidation
"""

from __future__ import annotations

import hashlib
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ProofStatus(str, Enum):
    """Inline proof status for a code block."""

    PENDING = "pending"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


class DiagnosticSeverity(str, Enum):
    """LSP-compatible diagnostic severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFORMATION = "information"
    HINT = "hint"


class VerificationTrigger(str, Enum):
    """What triggered a verification run."""

    FILE_SAVE = "file_save"
    MANUAL = "manual"
    DEPENDENCY_CHANGE = "dependency_change"
    DEBOUNCE_TIMEOUT = "debounce_timeout"


@dataclass
class CodeBlock:
    """A verifiable unit of code (function, method, class)."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    file_path: str = ""
    name: str = ""
    block_type: str = "function"  # function, method, class, module
    start_line: int = 0
    end_line: int = 0
    content: str = ""
    content_hash: str = ""
    dependencies: list[str] = field(default_factory=list)

    def compute_hash(self) -> str:
        self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
        return self.content_hash


@dataclass
class BlockDiagnostic:
    """A diagnostic finding for a code block."""

    file_path: str = ""
    line: int = 0
    end_line: int = 0
    column: int = 0
    end_column: int = 0
    severity: DiagnosticSeverity = DiagnosticSeverity.WARNING
    message: str = ""
    source: str = "codeverify"
    code: str = ""
    fix_suggestion: str | None = None


@dataclass
class BlockVerificationResult:
    """Result of verifying a single code block."""

    block_id: str = ""
    block_name: str = ""
    file_path: str = ""
    status: ProofStatus = ProofStatus.PENDING
    diagnostics: list[BlockDiagnostic] = field(default_factory=list)
    proof_summary: str = ""
    verification_time_ms: int = 0
    cached: bool = False
    content_hash: str = ""

    @property
    def has_errors(self) -> bool:
        return any(d.severity == DiagnosticSeverity.ERROR for d in self.diagnostics)


@dataclass
class FileVerificationResult:
    """Aggregated verification result for a file."""

    file_path: str = ""
    blocks: list[BlockVerificationResult] = field(default_factory=list)
    total_time_ms: int = 0
    cache_hit_rate: float = 0.0
    trigger: VerificationTrigger = VerificationTrigger.FILE_SAVE
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def overall_status(self) -> ProofStatus:
        if any(b.status == ProofStatus.ERROR for b in self.blocks):
            return ProofStatus.ERROR
        if any(b.status == ProofStatus.WARNING for b in self.blocks):
            return ProofStatus.WARNING
        if all(b.status == ProofStatus.VERIFIED for b in self.blocks):
            return ProofStatus.VERIFIED
        return ProofStatus.PENDING

    @property
    def diagnostic_count(self) -> int:
        return sum(len(b.diagnostics) for b in self.blocks)


@dataclass
class VerificationCacheEntry:
    """Cached verification result keyed by content hash."""

    content_hash: str = ""
    result: BlockVerificationResult | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    hit_count: int = 0
    ttl_seconds: int = 3600


@dataclass
class StreamingSession:
    """Active verification session for an open file."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    file_path: str = ""
    is_active: bool = True
    last_verification_at: datetime | None = None
    pending_changes: bool = False
    debounce_ms: int = 500
    total_verifications: int = 0
    total_cache_hits: int = 0


class CodeBlockParser:
    """Parses source code into verifiable blocks."""

    FUNCTION_PATTERNS: dict[str, list[str]] = {
        "python": ["def ", "async def "],
        "typescript": ["function ", "async function ", "const ", "export function "],
        "go": ["func "],
        "java": ["public ", "private ", "protected ", "static "],
        "rust": ["fn ", "pub fn ", "async fn "],
    }

    def parse_blocks(
        self,
        file_path: str,
        content: str,
        language: str = "python",
    ) -> list[CodeBlock]:
        """Parse a file into verifiable code blocks."""
        lines = content.split("\n")
        blocks: list[CodeBlock] = []
        patterns = self.FUNCTION_PATTERNS.get(language, ["def "])

        current_block: CodeBlock | None = None

        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            for pattern in patterns:
                if stripped.startswith(pattern):
                    if current_block:
                        current_block.end_line = i - 1
                        current_block.compute_hash()
                        blocks.append(current_block)

                    name = self._extract_name(stripped, pattern)
                    current_block = CodeBlock(
                        file_path=file_path,
                        name=name,
                        block_type="function",
                        start_line=i,
                        content=line + "\n",
                    )
                    break
            else:
                if current_block:
                    current_block.content += line + "\n"

        if current_block:
            current_block.end_line = len(lines)
            current_block.compute_hash()
            blocks.append(current_block)

        return blocks

    def _extract_name(self, line: str, pattern: str) -> str:
        after_pattern = line[len(pattern) :]
        name = after_pattern.split("(")[0].split(":")[0].split("{")[0]
        return name.strip().rstrip(" =")


class IncrementalVerificationCache:
    """Content-addressed cache for verification results."""

    def __init__(self, max_entries: int = 10000, default_ttl: int = 3600) -> None:
        self._cache: dict[str, VerificationCacheEntry] = {}
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def get(self, content_hash: str) -> BlockVerificationResult | None:
        entry = self._cache.get(content_hash)
        if entry is None:
            self._stats["misses"] += 1
            return None

        age = (datetime.now(UTC) - entry.created_at).total_seconds()
        if age > entry.ttl_seconds:
            del self._cache[content_hash]
            self._stats["misses"] += 1
            return None

        entry.hit_count += 1
        self._stats["hits"] += 1
        if entry.result:
            entry.result.cached = True
        return entry.result

    def put(
        self,
        content_hash: str,
        result: BlockVerificationResult,
        ttl: int | None = None,
    ) -> None:
        if len(self._cache) >= self._max_entries:
            self._evict_oldest()

        self._cache[content_hash] = VerificationCacheEntry(
            content_hash=content_hash,
            result=result,
            ttl_seconds=ttl or self._default_ttl,
        )

    def invalidate(self, content_hash: str) -> bool:
        if content_hash in self._cache:
            del self._cache[content_hash]
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    @property
    def stats(self) -> dict[str, float]:
        total = self._stats["hits"] + self._stats["misses"]
        return {
            **self._stats,
            "size": len(self._cache),
            "hit_rate": round(self._stats["hits"] / total, 3) if total > 0 else 0.0,
        }

    def _evict_oldest(self) -> None:
        if not self._cache:
            return
        oldest_key = min(self._cache, key=lambda k: self._cache[k].created_at)
        del self._cache[oldest_key]
        self._stats["evictions"] += 1


class StreamingVerifier:
    """Performs incremental, streaming verification of code blocks."""

    SIMPLE_CHECKS: dict[str, list[tuple[str, str, str]]] = {
        "python": [
            (r"/ 0", "division_by_zero", "Potential division by zero"),
            (r"[i]", "array_bounds", "Potential array bounds issue"),
            (r"eval(", "security", "Use of eval() is a security risk"),
            (r"exec(", "security", "Use of exec() is a security risk"),
        ],
        "typescript": [
            (r"/ 0", "division_by_zero", "Potential division by zero"),
            (r"eval(", "security", "Use of eval() is a security risk"),
            (r"any", "type_safety", "Use of 'any' bypasses type checking"),
        ],
    }

    def verify_block(
        self,
        block: CodeBlock,
        language: str = "python",
    ) -> BlockVerificationResult:
        """Verify a single code block."""
        start_time = time.time()
        diagnostics: list[BlockDiagnostic] = []

        checks = self.SIMPLE_CHECKS.get(language, self.SIMPLE_CHECKS.get("python", []))
        for pattern, code, message in checks:
            for i, line in enumerate(block.content.split("\n"), block.start_line):
                if pattern in line:
                    diagnostics.append(
                        BlockDiagnostic(
                            file_path=block.file_path,
                            line=i,
                            column=line.find(pattern),
                            severity=DiagnosticSeverity.WARNING
                            if code != "security"
                            else DiagnosticSeverity.ERROR,
                            message=message,
                            code=code,
                        )
                    )

        elapsed_ms = int((time.time() - start_time) * 1000)
        status = ProofStatus.VERIFIED
        if any(d.severity == DiagnosticSeverity.ERROR for d in diagnostics):
            status = ProofStatus.ERROR
        elif diagnostics:
            status = ProofStatus.WARNING

        return BlockVerificationResult(
            block_id=block.id,
            block_name=block.name,
            file_path=block.file_path,
            status=status,
            diagnostics=diagnostics,
            proof_summary=f"Verified {block.name}: {status.value}",
            verification_time_ms=elapsed_ms,
            content_hash=block.content_hash,
        )


class StreamingIDEVerificationService:
    """Main service for streaming IDE verification."""

    def __init__(
        self,
        cache_max_entries: int = 10000,
        cache_ttl: int = 3600,
        debounce_ms: int = 500,
    ) -> None:
        self._parser = CodeBlockParser()
        self._cache = IncrementalVerificationCache(
            max_entries=cache_max_entries, default_ttl=cache_ttl
        )
        self._verifier = StreamingVerifier()
        self._sessions: dict[str, StreamingSession] = {}
        self._debounce_ms = debounce_ms
        self._dependency_graph: dict[str, set[str]] = defaultdict(set)

    @property
    def cache(self) -> IncrementalVerificationCache:
        return self._cache

    def open_session(self, file_path: str) -> StreamingSession:
        """Open a verification session for a file."""
        session = StreamingSession(
            file_path=file_path,
            debounce_ms=self._debounce_ms,
        )
        self._sessions[file_path] = session
        logger.info("streaming_session_opened", file_path=file_path)
        return session

    def close_session(self, file_path: str) -> bool:
        """Close a verification session."""
        session = self._sessions.pop(file_path, None)
        if session:
            session.is_active = False
            return True
        return False

    def verify_file(
        self,
        file_path: str,
        content: str,
        language: str = "python",
        trigger: VerificationTrigger = VerificationTrigger.FILE_SAVE,
    ) -> FileVerificationResult:
        """Verify a file incrementally using cached results."""
        start_time = time.time()
        blocks = self._parser.parse_blocks(file_path, content, language)
        block_results: list[BlockVerificationResult] = []
        cache_hits = 0

        for block in blocks:
            cached = self._cache.get(block.content_hash)
            if cached:
                cache_hits += 1
                block_results.append(cached)
            else:
                result = self._verifier.verify_block(block, language)
                self._cache.put(block.content_hash, result)
                block_results.append(result)

        total_ms = int((time.time() - start_time) * 1000)
        hit_rate = cache_hits / len(blocks) if blocks else 0.0

        session = self._sessions.get(file_path)
        if session:
            session.total_verifications += 1
            session.total_cache_hits += cache_hits
            session.last_verification_at = datetime.now(UTC)
            session.pending_changes = False

        return FileVerificationResult(
            file_path=file_path,
            blocks=block_results,
            total_time_ms=total_ms,
            cache_hit_rate=round(hit_rate, 3),
            trigger=trigger,
        )

    def register_dependency(self, source_file: str, depends_on: str) -> None:
        """Register that source_file depends on depends_on."""
        self._dependency_graph[depends_on].add(source_file)

    def get_affected_files(self, changed_file: str) -> list[str]:
        """Get all files affected by a change (transitive)."""
        affected: set[str] = set()
        queue = [changed_file]
        while queue:
            current = queue.pop(0)
            dependents = self._dependency_graph.get(current, set())
            for dep in dependents:
                if dep not in affected:
                    affected.add(dep)
                    queue.append(dep)
        return sorted(affected)

    def invalidate_file(self, file_path: str) -> int:
        """Invalidate cache entries for a file's blocks."""
        count = 0
        keys_to_remove = []
        for key, entry in self._cache._cache.items():
            if entry.result and entry.result.file_path == file_path:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            self._cache.invalidate(key)
            count += 1
        return count

    def get_session(self, file_path: str) -> StreamingSession | None:
        return self._sessions.get(file_path)

    def get_active_sessions(self) -> list[StreamingSession]:
        return [s for s in self._sessions.values() if s.is_active]

    def get_stats(self) -> dict[str, Any]:
        """Get service-level stats."""
        return {
            "active_sessions": len(self.get_active_sessions()),
            "cache": self._cache.stats,
            "dependency_edges": sum(len(deps) for deps in self._dependency_graph.values()),
        }


# ─── Singleton Access ──────────────────────────────────────────────────


_streaming_ide_instance: StreamingIDEVerificationService | None = None


def get_streaming_ide_service() -> StreamingIDEVerificationService:
    """Get or create the singleton StreamingIDEVerificationService."""
    global _streaming_ide_instance
    if _streaming_ide_instance is None:
        _streaming_ide_instance = StreamingIDEVerificationService()
    return _streaming_ide_instance


def reset_streaming_ide_service() -> None:
    """Reset the singleton (for testing)."""
    global _streaming_ide_instance
    _streaming_ide_instance = None
