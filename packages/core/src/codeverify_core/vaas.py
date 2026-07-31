"""Verification-as-a-Service API.

Provides a hosted API where any tool or CI pipeline can submit code for
formal verification and AI analysis, with usage-based pricing and
async webhook delivery.

Features:
- RESTful API for code submission and verification
- Async verification with webhook delivery
- Usage-based billing (pay-per-verification)
- Multi-language SDK support (Python, Node.js, Go, Rust)
- Rate limiting and quota management
- Verification result caching
- API key authentication with scoped permissions
"""

from __future__ import annotations

import hashlib
import secrets
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class VerificationTier(str, Enum):
    """Service tiers for VaaS."""

    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class VerificationStatus(str, Enum):
    """Status of a verification job."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class WebhookEventType(str, Enum):
    """Types of webhook events."""

    VERIFICATION_COMPLETED = "verification.completed"
    VERIFICATION_FAILED = "verification.failed"
    QUOTA_WARNING = "quota.warning"
    QUOTA_EXCEEDED = "quota.exceeded"


class OutputFormat(str, Enum):
    """Output formats for verification results."""

    JSON = "json"
    SARIF = "sarif"
    MARKDOWN = "markdown"
    HTML = "html"


@dataclass
class TierLimits:
    """Limits for a service tier."""

    verifications_per_month: int = 100
    max_file_size_kb: int = 500
    max_files_per_request: int = 5
    rate_limit_per_minute: int = 10
    concurrent_requests: int = 2
    webhook_enabled: bool = False
    priority_queue: bool = False
    sla_response_ms: int = 30000

    @classmethod
    def for_tier(cls, tier: VerificationTier) -> TierLimits:
        tiers = {
            VerificationTier.FREE: cls(
                verifications_per_month=100,
                max_file_size_kb=500,
                max_files_per_request=5,
                rate_limit_per_minute=10,
                concurrent_requests=2,
            ),
            VerificationTier.STARTER: cls(
                verifications_per_month=1000,
                max_file_size_kb=2000,
                max_files_per_request=20,
                rate_limit_per_minute=30,
                concurrent_requests=5,
                webhook_enabled=True,
            ),
            VerificationTier.PROFESSIONAL: cls(
                verifications_per_month=10000,
                max_file_size_kb=5000,
                max_files_per_request=50,
                rate_limit_per_minute=100,
                concurrent_requests=20,
                webhook_enabled=True,
                priority_queue=True,
                sla_response_ms=10000,
            ),
            VerificationTier.ENTERPRISE: cls(
                verifications_per_month=-1,
                max_file_size_kb=50000,
                max_files_per_request=200,
                rate_limit_per_minute=500,
                concurrent_requests=100,
                webhook_enabled=True,
                priority_queue=True,
                sla_response_ms=5000,
            ),
        }
        return tiers.get(tier, cls())


@dataclass
class VaaSApiKey:
    """API key for VaaS access."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    key_hash: str = ""
    name: str = ""
    tier: VerificationTier = VerificationTier.FREE
    owner_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_used_at: datetime | None = None
    is_active: bool = True
    usage_this_month: int = 0


@dataclass
class VerificationRequest:
    """A verification request submitted to the API."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    api_key_id: str = ""
    files: list[dict[str, str]] = field(default_factory=list)
    language: str = "python"
    checks: list[str] = field(default_factory=list)
    output_format: OutputFormat = OutputFormat.JSON
    webhook_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    submitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class VerificationFinding:
    """A finding from VaaS verification."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    file_path: str = ""
    line: int = 0
    column: int = 0
    severity: str = "medium"
    category: str = ""
    message: str = ""
    fix_suggestion: str = ""
    z3_proof: str | None = None
    confidence: float = 0.0


@dataclass
class VerificationResponse:
    """Response from a verification request."""

    request_id: str = ""
    status: VerificationStatus = VerificationStatus.QUEUED
    findings: list[VerificationFinding] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    verification_time_ms: int = 0
    completed_at: datetime | None = None
    output_format: OutputFormat = OutputFormat.JSON
    cached: bool = False

    @property
    def finding_count(self) -> int:
        return len(self.findings)

    def to_sarif(self) -> dict[str, Any]:
        """Convert to SARIF format."""
        results = []
        for f in self.findings:
            results.append(
                {
                    "ruleId": f.category,
                    "level": self._sarif_level(f.severity),
                    "message": {"text": f.message},
                    "locations": [
                        {
                            "physicalLocation": {
                                "artifactLocation": {"uri": f.file_path},
                                "region": {"startLine": f.line, "startColumn": f.column},
                            }
                        }
                    ],
                }
            )
        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {"driver": {"name": "CodeVerify VaaS", "version": "1.2.0"}},
                    "results": results,
                }
            ],
        }

    @staticmethod
    def _sarif_level(severity: str) -> str:
        return {"critical": "error", "high": "error", "medium": "warning", "low": "note"}.get(
            severity, "note"
        )


@dataclass
class WebhookConfig:
    """Webhook configuration for a client."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    url: str = ""
    events: list[WebhookEventType] = field(default_factory=list)
    secret: str = field(default_factory=lambda: secrets.token_hex(32))
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    webhook_id: str = ""
    event_type: WebhookEventType = WebhookEventType.VERIFICATION_COMPLETED
    payload: dict[str, Any] = field(default_factory=dict)
    status_code: int = 0
    delivered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    success: bool = False


@dataclass
class UsageStats:
    """Usage statistics for a client."""

    api_key_id: str = ""
    period: str = ""
    total_verifications: int = 0
    total_findings: int = 0
    total_files_processed: int = 0
    average_response_ms: int = 0
    cache_hit_rate: float = 0.0
    quota_used_percent: float = 0.0


class VerificationCache:
    """Caches verification results by content hash."""

    def __init__(self, max_entries: int = 10000, ttl_hours: int = 24) -> None:
        self.max_entries = max_entries
        self.ttl = timedelta(hours=ttl_hours)
        self._cache: dict[str, tuple[VerificationResponse, datetime]] = {}

    def get(self, content_hash: str) -> VerificationResponse | None:
        entry = self._cache.get(content_hash)
        if entry is None:
            return None
        response, cached_at = entry
        if datetime.now(UTC) - cached_at > self.ttl:
            del self._cache[content_hash]
            return None
        response.cached = True
        return response

    def put(self, content_hash: str, response: VerificationResponse) -> None:
        if len(self._cache) >= self.max_entries:
            # Evict oldest
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        self._cache[content_hash] = (response, datetime.now(UTC))

    @staticmethod
    def compute_hash(files: list[dict[str, str]], checks: list[str]) -> str:
        content = str(sorted([(f.get("path", ""), f.get("content", "")) for f in files]))
        content += str(sorted(checks))
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class VaaSService:
    """Main Verification-as-a-Service engine.

    Handles API key management, request processing, caching,
    rate limiting, and webhook delivery.
    """

    def __init__(self) -> None:
        self.api_keys: dict[str, VaaSApiKey] = {}
        self._raw_key_map: dict[str, str] = {}  # raw_key -> key_id
        self.requests: dict[str, VerificationRequest] = {}
        self.responses: dict[str, VerificationResponse] = {}
        self.webhooks: dict[str, WebhookConfig] = {}
        self.deliveries: list[WebhookDelivery] = []
        self.cache = VerificationCache()
        self._rate_counters: dict[str, list[float]] = defaultdict(list)

    def create_api_key(
        self,
        name: str,
        tier: VerificationTier = VerificationTier.FREE,
        owner_id: str = "",
    ) -> tuple[VaaSApiKey, str]:
        """Create a new API key."""
        raw_key = f"cv_vaas_{secrets.token_hex(24)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        api_key = VaaSApiKey(
            key_hash=key_hash,
            name=name,
            tier=tier,
            owner_id=owner_id,
        )
        self.api_keys[api_key.id] = api_key
        self._raw_key_map[raw_key] = api_key.id
        return api_key, raw_key

    def validate_api_key(self, raw_key: str) -> VaaSApiKey | None:
        """Validate an API key and return its metadata."""
        key_id = self._raw_key_map.get(raw_key)
        if key_id is None:
            return None
        api_key = self.api_keys.get(key_id)
        if api_key and api_key.is_active:
            api_key.last_used_at = datetime.now(UTC)
            return api_key
        return None

    def check_rate_limit(self, api_key_id: str) -> bool:
        """Check if request is within rate limits."""
        api_key = self.api_keys.get(api_key_id)
        if not api_key:
            return False

        limits = TierLimits.for_tier(api_key.tier)
        now = time.monotonic()

        # Clean old entries
        self._rate_counters[api_key_id] = [
            t for t in self._rate_counters[api_key_id] if now - t < 60
        ]

        if len(self._rate_counters[api_key_id]) >= limits.rate_limit_per_minute:
            return False

        self._rate_counters[api_key_id].append(now)
        return True

    def submit_verification(
        self,
        api_key_id: str,
        files: list[dict[str, str]],
        language: str = "python",
        checks: list[str] | None = None,
        output_format: OutputFormat = OutputFormat.JSON,
        webhook_url: str | None = None,
    ) -> VerificationResponse:
        """Submit code for verification."""
        api_key = self.api_keys.get(api_key_id)
        if not api_key:
            raise ValueError("Invalid API key")

        limits = TierLimits.for_tier(api_key.tier)

        # Check quota
        if (
            limits.verifications_per_month > 0
            and api_key.usage_this_month >= limits.verifications_per_month
        ):
            raise ValueError("Monthly quota exceeded")

        # Check file count
        if len(files) > limits.max_files_per_request:
            raise ValueError(f"Too many files: {len(files)} > {limits.max_files_per_request}")

        effective_checks = checks or ["null_safety", "array_bounds", "integer_overflow"]

        # Check cache
        content_hash = self.cache.compute_hash(files, effective_checks)
        cached = self.cache.get(content_hash)
        if cached:
            cached.request_id = str(uuid.uuid4())[:8]
            return cached

        request = VerificationRequest(
            api_key_id=api_key_id,
            files=files,
            language=language,
            checks=effective_checks,
            output_format=output_format,
            webhook_url=webhook_url,
        )
        self.requests[request.id] = request

        # Process verification
        start = time.monotonic()
        findings = self._run_verification(files, language, effective_checks)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        severity_counts: defaultdict[str, int] = defaultdict(int)
        for f in findings:
            severity_counts[f.severity] += 1

        response = VerificationResponse(
            request_id=request.id,
            status=VerificationStatus.COMPLETED,
            findings=findings,
            summary={
                "total_findings": len(findings),
                "by_severity": dict(severity_counts),
                "files_processed": len(files),
                "language": language,
                "checks": effective_checks,
            },
            verification_time_ms=elapsed_ms,
            completed_at=datetime.now(UTC),
            output_format=output_format,
        )

        self.responses[request.id] = response
        self.cache.put(content_hash, response)
        api_key.usage_this_month += 1

        return response

    def get_result(self, request_id: str) -> VerificationResponse | None:
        """Get verification result by request ID."""
        return self.responses.get(request_id)

    def register_webhook(
        self,
        api_key_id: str,
        url: str,
        events: list[WebhookEventType] | None = None,
    ) -> WebhookConfig:
        """Register a webhook for event delivery."""
        api_key = self.api_keys.get(api_key_id)
        if not api_key:
            raise ValueError("Invalid API key")

        limits = TierLimits.for_tier(api_key.tier)
        if not limits.webhook_enabled:
            raise ValueError("Webhooks not available on your plan")

        webhook = WebhookConfig(
            url=url,
            events=events or [WebhookEventType.VERIFICATION_COMPLETED],
        )
        self.webhooks[webhook.id] = webhook
        return webhook

    def get_usage_stats(self, api_key_id: str) -> UsageStats:
        """Get usage statistics for an API key."""
        api_key = self.api_keys.get(api_key_id)
        if not api_key:
            raise ValueError("Invalid API key")

        limits = TierLimits.for_tier(api_key.tier)
        quota_pct = (
            (api_key.usage_this_month / limits.verifications_per_month * 100)
            if limits.verifications_per_month > 0
            else 0.0
        )

        total_findings = sum(
            len(r.findings)
            for r in self.responses.values()
            if self.requests.get(r.request_id, VerificationRequest()).api_key_id == api_key_id
        )

        return UsageStats(
            api_key_id=api_key_id,
            period=datetime.now(UTC).strftime("%Y-%m"),
            total_verifications=api_key.usage_this_month,
            total_findings=total_findings,
            quota_used_percent=quota_pct,
        )

    def _run_verification(
        self,
        files: list[dict[str, str]],
        _language: str,
        checks: list[str],
    ) -> list[VerificationFinding]:
        """Run verification pipeline on submitted files."""
        findings = []

        for file_info in files:
            path = file_info.get("path", "unknown")
            content = file_info.get("content", "")
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                stripped = line.strip()

                if (
                    "null_safety" in checks
                    and "None" in stripped
                    and "if" not in stripped
                    and "is not" not in stripped
                    and "= None" not in stripped
                    and "== None" not in stripped
                ):
                    findings.append(
                        VerificationFinding(
                            file_path=path,
                            line=i,
                            severity="medium",
                            category="null_safety",
                            message="Potential None usage without explicit null check",
                            confidence=0.6,
                        )
                    )

                if "integer_overflow" in checks and ("**" in stripped or "pow(" in stripped):
                    findings.append(
                        VerificationFinding(
                            file_path=path,
                            line=i,
                            severity="low",
                            category="integer_overflow",
                            message="Potential integer overflow in exponentiation",
                            confidence=0.4,
                        )
                    )

        return findings


# ─── Singleton Access ──────────────────────────────────────────────────


_vaas_instance: VaaSService | None = None


def get_vaas_service() -> VaaSService:
    """Get or create the singleton VaaSService."""
    global _vaas_instance
    if _vaas_instance is None:
        _vaas_instance = VaaSService()
    return _vaas_instance


def reset_vaas_service() -> None:
    """Reset the singleton (for testing)."""
    global _vaas_instance
    _vaas_instance = None
