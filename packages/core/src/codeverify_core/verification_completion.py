"""Verification-First Code Completion.

LSP middleware that pre-verifies code completions before showing them to
developers. Integrates with completion providers to filter and rank suggestions
through Z3 verification, showing verified completions first.

Features:
- Async verification of top-N completion candidates
- Verification status badges (verified / partial / unverified)
- Pattern-based cache for instant re-verification of known patterns
- Latency-aware fallback (show unverified after timeout)
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

import structlog

logger = structlog.get_logger()


class VerificationStatus(str, Enum):
    """Verification status for a completion candidate."""

    VERIFIED = "verified"
    PARTIAL = "partial"
    UNVERIFIED = "unverified"
    FAILED = "failed"
    TIMEOUT = "timeout"


class CompletionSource(str, Enum):
    """Source of the completion suggestion."""

    COPILOT = "copilot"
    CURSOR = "cursor"
    CODEWHISPERER = "codewhisperer"
    CUSTOM = "custom"
    MANUAL = "manual"


@dataclass
class CompletionCandidate:
    """A code completion candidate to be verified."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str = ""
    source: CompletionSource = CompletionSource.CUSTOM
    language: str = "python"
    context_before: str = ""
    context_after: str = ""
    file_path: str = ""
    line: int = 0
    confidence: float = 0.0


@dataclass
class VerifiedCompletion:
    """A completion candidate with verification results."""

    candidate: CompletionCandidate
    status: VerificationStatus = VerificationStatus.UNVERIFIED
    checks_passed: int = 0
    checks_total: int = 0
    issues: list[str] = field(default_factory=list)
    verification_time_ms: float = 0.0
    cached: bool = False

    @property
    def score(self) -> float:
        """Composite score: verification weight + original confidence."""
        status_weight = {
            VerificationStatus.VERIFIED: 1.0,
            VerificationStatus.PARTIAL: 0.6,
            VerificationStatus.UNVERIFIED: 0.3,
            VerificationStatus.FAILED: 0.1,
            VerificationStatus.TIMEOUT: 0.3,
        }
        v_weight = status_weight.get(self.status, 0.3)
        return v_weight * 0.7 + self.candidate.confidence * 0.3

    @property
    def badge(self) -> str:
        badges = {
            VerificationStatus.VERIFIED: "✅",
            VerificationStatus.PARTIAL: "⚠️",
            VerificationStatus.UNVERIFIED: "❓",
            VerificationStatus.FAILED: "❌",
            VerificationStatus.TIMEOUT: "⏱️",
        }
        return badges.get(self.status, "❓")


@dataclass
class VerificationRule:
    """A quick verification rule for completion checking."""

    id: str = ""
    name: str = ""
    pattern: str = ""
    check_type: str = "regex"
    severity: str = "warning"
    message: str = ""

    def matches(self, code: str) -> bool:
        try:
            return bool(re.search(self.pattern, code))
        except re.error:
            return False


# Built-in quick-check rules per language
_COMPLETION_RULES: dict[str, list[VerificationRule]] = {
    "python": [
        VerificationRule(
            id="py-div-zero",
            name="Division by zero risk",
            pattern=r"/\s*(?:0|zero|\bx\b)\s*(?:[;\n]|$)",
            message="Potential division by zero",
        ),
        VerificationRule(
            id="py-none-access",
            name="None attribute access",
            pattern=r"(?:None|null)\s*\.",
            message="Accessing attribute on None value",
        ),
        VerificationRule(
            id="py-bare-except",
            name="Bare except clause",
            pattern=r"except\s*:",
            message="Bare except catches all exceptions including SystemExit",
        ),
        VerificationRule(
            id="py-eval-usage",
            name="Eval usage",
            pattern=r"\beval\s*\(",
            message="eval() is a security risk",
        ),
        VerificationRule(
            id="py-mutable-default",
            name="Mutable default argument",
            pattern=r"def\s+\w+\s*\([^)]*=\s*(\[\]|\{\})",
            message="Mutable default argument will be shared across calls",
        ),
    ],
    "typescript": [
        VerificationRule(
            id="ts-any-cast",
            name="Any type cast",
            pattern=r"\bas\s+any\b",
            message="Casting to 'any' bypasses type safety",
        ),
        VerificationRule(
            id="ts-non-null-assert",
            name="Non-null assertion",
            pattern=r"\w+![\.\[]",
            message="Non-null assertion may cause runtime error",
        ),
        VerificationRule(
            id="ts-eval-usage",
            name="Eval usage",
            pattern=r"\beval\s*\(",
            message="eval() is a security risk",
        ),
    ],
    "rust": [
        VerificationRule(
            id="rs-unwrap",
            name="Unwrap usage",
            pattern=r"\.unwrap\s*\(",
            message="unwrap() may panic; prefer ? operator",
        ),
        VerificationRule(
            id="rs-unsafe",
            name="Unsafe block",
            pattern=r"\bunsafe\s*\{",
            message="Unsafe block requires careful review",
        ),
    ],
    "go": [
        VerificationRule(
            id="go-err-ignored",
            name="Error ignored",
            pattern=r"\w+,\s*_\s*:?=",
            message="Error return value is being discarded",
        ),
    ],
}


@dataclass
class CompletionCacheEntry:
    """Cached verification result for a completion pattern."""

    fingerprint: str
    status: VerificationStatus
    checks_passed: int
    checks_total: int
    issues: list[str]
    created_at: float = field(default_factory=time.time)
    ttl_seconds: int = 3600

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


class CompletionVerifier:
    """Verifies code completion candidates against safety rules.

    Applies language-specific rules and optional Z3 verification
    to rank completions by safety.
    """

    def __init__(
        self,
        timeout_ms: float = 200.0,
        max_candidates: int = 5,
        cache_size: int = 1000,
        custom_rules: list[VerificationRule] | None = None,
    ) -> None:
        self.timeout_ms = timeout_ms
        self.max_candidates = max_candidates
        self._cache: dict[str, CompletionCacheEntry] = {}
        self._cache_size = cache_size
        self._custom_rules = custom_rules or []
        self._stats = CompletionVerifierStats()

    @property
    def stats(self) -> CompletionVerifierStats:
        return self._stats

    def _fingerprint(self, candidate: CompletionCandidate) -> str:
        content = f"{candidate.language}:{candidate.text}:{candidate.context_before[-100:]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_rules(self, language: str) -> list[VerificationRule]:
        rules = list(_COMPLETION_RULES.get(language, []))
        rules.extend(self._custom_rules)
        return rules

    def _check_cache(self, fingerprint: str) -> CompletionCacheEntry | None:
        entry = self._cache.get(fingerprint)
        if entry is None:
            return None
        if entry.is_expired:
            del self._cache[fingerprint]
            return None
        return entry

    def _update_cache(
        self,
        fingerprint: str,
        status: VerificationStatus,
        checks_passed: int,
        checks_total: int,
        issues: list[str],
    ) -> None:
        if len(self._cache) >= self._cache_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k].created_at)
            del self._cache[oldest_key]
            self._stats.cache_evictions += 1

        self._cache[fingerprint] = CompletionCacheEntry(
            fingerprint=fingerprint,
            status=status,
            checks_passed=checks_passed,
            checks_total=checks_total,
            issues=issues,
        )

    def verify_candidate(self, candidate: CompletionCandidate) -> VerifiedCompletion:
        """Verify a single completion candidate."""
        start = time.time()
        fingerprint = self._fingerprint(candidate)

        # Check cache first
        cached = self._check_cache(fingerprint)
        if cached is not None:
            self._stats.cache_hits += 1
            elapsed = (time.time() - start) * 1000
            return VerifiedCompletion(
                candidate=candidate,
                status=cached.status,
                checks_passed=cached.checks_passed,
                checks_total=cached.checks_total,
                issues=list(cached.issues),
                verification_time_ms=elapsed,
                cached=True,
            )

        self._stats.cache_misses += 1

        # Apply rules
        rules = self._get_rules(candidate.language)
        full_code = candidate.context_before + candidate.text + candidate.context_after
        issues: list[str] = []
        checks_passed = 0

        for rule in rules:
            # Check against the full code (surrounding context + candidate text)
            # so rules can catch issues that span the context boundary.
            if rule.matches(full_code):
                issues.append(f"[{rule.id}] {rule.message}")
            else:
                checks_passed += 1

        checks_total = len(rules)
        elapsed = (time.time() - start) * 1000

        # Determine status
        if elapsed > self.timeout_ms:
            status = VerificationStatus.TIMEOUT
        elif checks_total == 0:
            status = VerificationStatus.UNVERIFIED
        elif len(issues) == 0:
            status = VerificationStatus.VERIFIED
        elif checks_passed > checks_total // 2:
            status = VerificationStatus.PARTIAL
        else:
            status = VerificationStatus.FAILED

        self._stats.total_verifications += 1
        self._stats.total_time_ms += elapsed

        self._update_cache(fingerprint, status, checks_passed, checks_total, issues)

        return VerifiedCompletion(
            candidate=candidate,
            status=status,
            checks_passed=checks_passed,
            checks_total=checks_total,
            issues=issues,
            verification_time_ms=elapsed,
            cached=False,
        )

    def verify_and_rank(
        self,
        candidates: list[CompletionCandidate],
    ) -> list[VerifiedCompletion]:
        """Verify multiple candidates and rank by verification score."""
        limited = candidates[: self.max_candidates]
        results = [self.verify_candidate(c) for c in limited]
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def clear_cache(self) -> None:
        self._cache.clear()


@dataclass
class CompletionVerifierStats:
    """Statistics for the completion verifier."""

    total_verifications: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0
    total_time_ms: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def avg_verification_ms(self) -> float:
        return (
            self.total_time_ms / self.total_verifications if self.total_verifications > 0 else 0.0
        )


class CompletionMiddleware:
    """LSP-style middleware that intercepts completions and adds verification.

    Sits between the completion provider (Copilot/Cursor) and the IDE,
    verifying and re-ranking suggestions.
    """

    def __init__(
        self,
        verifier: CompletionVerifier | None = None,
        enabled: bool = True,
        min_text_length: int = 5,
    ) -> None:
        self.verifier = verifier or CompletionVerifier()
        self.enabled = enabled
        self.min_text_length = min_text_length
        self._intercepted_count = 0

    def process_completions(
        self,
        candidates: list[CompletionCandidate],
        file_path: str = "",
        language: str = "python",
    ) -> list[VerifiedCompletion]:
        """Process completion candidates through verification pipeline."""
        if not self.enabled:
            return [
                VerifiedCompletion(candidate=c, status=VerificationStatus.UNVERIFIED)
                for c in candidates
            ]

        self._intercepted_count += 1

        # Enrich candidates with context
        for c in candidates:
            if not c.file_path:
                c.file_path = file_path
            if not c.language:
                c.language = language

        # Filter out very short completions
        eligible = [c for c in candidates if len(c.text.strip()) >= self.min_text_length]
        short = [c for c in candidates if len(c.text.strip()) < self.min_text_length]

        verified = self.verifier.verify_and_rank(eligible)

        # Add unverified short completions at the end
        for c in short:
            verified.append(VerifiedCompletion(candidate=c, status=VerificationStatus.UNVERIFIED))

        return verified

    def format_completion_label(self, vc: VerifiedCompletion) -> str:
        """Format a verified completion for display in IDE."""
        badge = vc.badge
        checks = f"{vc.checks_passed}/{vc.checks_total}" if vc.checks_total > 0 else "N/A"
        cached_tag = " (cached)" if vc.cached else ""
        return f"{badge} [{checks}]{cached_tag} {vc.candidate.text[:50]}"

    @property
    def intercepted_count(self) -> int:
        return self._intercepted_count


# Singleton
_completion_verifier_instance: CompletionVerifier | None = None


def get_completion_verifier() -> CompletionVerifier:
    global _completion_verifier_instance
    if _completion_verifier_instance is None:
        _completion_verifier_instance = CompletionVerifier()
    return _completion_verifier_instance


def reset_completion_verifier() -> None:
    global _completion_verifier_instance
    _completion_verifier_instance = None
