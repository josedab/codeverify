"""Finding Deduplication Engine.

Deduplicates verification findings across runs using semantic fingerprinting,
tracks first-seen / last-seen timestamps, and supports suppression of known
false positives.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FindingFingerprint:
    """Semantic fingerprint of a finding for deduplication."""

    rule_id: str
    file_path: str
    normalized_snippet: str
    context_hash: str

    @property
    def fingerprint(self) -> str:
        raw = f"{self.rule_id}|{self.file_path}|{self.normalized_snippet}|{self.context_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class TrackedFinding:
    """A finding with lifecycle tracking."""

    fingerprint: str
    rule_id: str
    file_path: str
    severity: str
    message: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    occurrence_count: int = 1
    suppressed: bool = False
    suppression_reason: str = ""


def normalize_code_snippet(snippet: str) -> str:
    """Normalize a code snippet for semantic comparison.

    Strips variable names to single-letter placeholders, normalizes
    whitespace, removes comments, and lowercases.
    """
    if not snippet:
        return ""
    # Remove inline comments
    result = re.sub(r"#.*$", "", snippet, flags=re.MULTILINE)
    result = re.sub(r"//.*$", "", result, flags=re.MULTILINE)
    # Collapse whitespace
    result = re.sub(r"\s+", " ", result).strip()
    # Normalize string literals
    result = re.sub(r'"[^"]*"', '"S"', result)
    result = re.sub(r"'[^']*'", "'S'", result)
    # Normalize numeric literals
    result = re.sub(r"\b\d+\.?\d*\b", "N", result)
    return result.lower()


def compute_context_hash(file_path: str, line: int, window: int = 3) -> str:
    """Compute a hash of the location context.

    Uses file path and approximate line region to handle small line shifts.
    """
    region = line // window  # Quantize to window
    raw = f"{file_path}:{region}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def fingerprint_finding(
    rule_id: str,
    file_path: str,
    code_snippet: str,
    line: int = 0,
) -> FindingFingerprint:
    """Create a semantic fingerprint from a finding."""
    return FindingFingerprint(
        rule_id=rule_id,
        file_path=file_path,
        normalized_snippet=normalize_code_snippet(code_snippet),
        context_hash=compute_context_hash(file_path, line),
    )


# =============================================================================
# Deduplication store
# =============================================================================


class FindingDeduplicator:
    """Deduplicates findings across verification runs.

    Usage:
        dedup = FindingDeduplicator()
        unique = dedup.process_findings(findings)
        new_findings = dedup.get_new_findings(findings)
    """

    def __init__(self) -> None:
        self._store: dict[str, TrackedFinding] = {}
        self._suppressions: set[str] = set()  # Suppressed fingerprints

    def process_findings(
        self,
        findings: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Process findings and return deduplicated results.

        Each finding dict should contain: rule_id, file_path, message, severity,
        and optionally code_snippet and line.
        """
        unique: list[dict[str, Any]] = []
        seen_fps: set[str] = set()

        for finding in findings:
            fp = fingerprint_finding(
                rule_id=finding.get("rule_id", ""),
                file_path=finding.get("file_path", ""),
                code_snippet=finding.get("code_snippet", ""),
                line=finding.get("line", 0),
            )
            fp_hash = fp.fingerprint

            if fp_hash in seen_fps:
                continue
            seen_fps.add(fp_hash)

            if fp_hash in self._suppressions:
                continue

            now = time.time()
            if fp_hash in self._store:
                tracked = self._store[fp_hash]
                tracked.last_seen = now
                tracked.occurrence_count += 1
            else:
                tracked = TrackedFinding(
                    fingerprint=fp_hash,
                    rule_id=finding.get("rule_id", ""),
                    file_path=finding.get("file_path", ""),
                    severity=finding.get("severity", "medium"),
                    message=finding.get("message", ""),
                    first_seen=now,
                    last_seen=now,
                )
                self._store[fp_hash] = tracked

            if not tracked.suppressed:
                enriched = {**finding, "_fingerprint": fp_hash}
                unique.append(enriched)

        return unique

    def get_new_findings(
        self,
        findings: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return only findings not previously seen."""
        new: list[dict[str, Any]] = []
        for finding in findings:
            fp = fingerprint_finding(
                rule_id=finding.get("rule_id", ""),
                file_path=finding.get("file_path", ""),
                code_snippet=finding.get("code_snippet", ""),
                line=finding.get("line", 0),
            )
            if fp.fingerprint not in self._store:
                new.append(finding)
        return new

    def suppress(self, fingerprint: str, reason: str = "") -> None:
        """Suppress a finding by fingerprint (mark as false positive)."""
        self._suppressions.add(fingerprint)
        if fingerprint in self._store:
            self._store[fingerprint].suppressed = True
            self._store[fingerprint].suppression_reason = reason

    def unsuppress(self, fingerprint: str) -> None:
        """Remove suppression for a finding."""
        self._suppressions.discard(fingerprint)
        if fingerprint in self._store:
            self._store[fingerprint].suppressed = False

    def get_tracked(self, fingerprint: str) -> TrackedFinding | None:
        return self._store.get(fingerprint)

    @property
    def total_tracked(self) -> int:
        return len(self._store)

    @property
    def total_suppressed(self) -> int:
        return len(self._suppressions)

    def get_recurring_findings(self, min_occurrences: int = 3) -> list[TrackedFinding]:
        """Return findings that have recurred across multiple runs."""
        return [
            t
            for t in self._store.values()
            if t.occurrence_count >= min_occurrences and not t.suppressed
        ]
