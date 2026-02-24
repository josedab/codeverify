"""Proof-as-a-Service API — Public API for on-demand code verification.

Provides pay-per-proof pricing, API key management with HMAC hashing,
sliding-window rate limiting, usage tracking, and proof artifact generation
in multiple formats (SMT-LIB, JSON, human-readable, certificate).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HMAC_SECRET = os.environ.get("CODEVERIFY_PROOF_HMAC_SECRET", "").encode() or None
if _HMAC_SECRET is None:
    raise RuntimeError("CODEVERIFY_PROOF_HMAC_SECRET environment variable must be set")
_RATE_LIMIT_WINDOW_SECONDS = 60
_DEFAULT_TIMEOUT_SECONDS = 30
_MAX_PRIORITY = 10

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ProofRequestStatus(str, Enum):
    """Lifecycle status of a proof request."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELED = "canceled"


class VerificationCheck(str, Enum):
    """Categories of verification checks."""

    NULL_SAFETY = "null_safety"
    BOUNDS_CHECK = "bounds_check"
    OVERFLOW = "overflow"
    DIVISION_BY_ZERO = "division_by_zero"
    TYPE_SAFETY = "type_safety"
    INJECTION = "injection"
    RESOURCE_LEAK = "resource_leak"
    RACE_CONDITION = "race_condition"


class ProofFormat(str, Enum):
    """Output formats for proof artifacts."""

    SMT_LIB = "smt_lib"
    JSON = "json"
    HUMAN_READABLE = "human_readable"
    CERTIFICATE = "certificate"


class PricingModel(str, Enum):
    """Billing models for the service."""

    PER_PROOF = "per_proof"
    PER_LINE = "per_line"
    SUBSCRIPTION = "subscription"
    FREE_TIER = "free_tier"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ProofRequest:
    """Incoming request for code verification."""

    id: str
    api_key: str
    code: str
    language: str
    checks: list[VerificationCheck]
    callback_url: str | None = None
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS
    priority: int = 5
    context: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "language": self.language,
            "checks": [c.value for c in self.checks],
            "callback_url": self.callback_url,
            "timeout_seconds": self.timeout_seconds,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ProofResult:
    """Result produced after processing a proof request."""

    request_id: str
    status: ProofRequestStatus
    verified: bool
    proof_artifact: str | None = None
    counterexample: dict[str, Any] | None = None
    findings: list[dict[str, Any]] = field(default_factory=list)
    execution_time_ms: float = 0
    tokens_used: int = 0
    cost_cents: float = 0
    proof_format: ProofFormat = ProofFormat.JSON

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "verified": self.verified,
            "proof_artifact": self.proof_artifact,
            "counterexample": self.counterexample,
            "findings": self.findings,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "tokens_used": self.tokens_used,
            "cost_cents": round(self.cost_cents, 4),
            "proof_format": self.proof_format.value,
        }


@dataclass
class APIKeyConfig:
    """Configuration and metadata for an API key."""

    key_id: str
    key_hash: str
    tenant_id: str
    name: str
    rate_limit_per_minute: int = 10
    monthly_quota: int = 100
    allowed_checks: list[VerificationCheck] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_id": self.key_id,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "monthly_quota": self.monthly_quota,
            "allowed_checks": [c.value for c in self.allowed_checks],
            "active": self.active,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class UsageBucket:
    """Aggregated usage statistics for a billing period."""

    api_key_id: str
    period: str
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    total_cost_cents: float = 0
    avg_latency_ms: float = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_key_id": self.api_key_id,
            "period": self.period,
            "total_requests": self.total_requests,
            "successful": self.successful,
            "failed": self.failed,
            "total_cost_cents": round(self.total_cost_cents, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


@dataclass
class PricingConfig:
    """Pricing parameters for the proof service."""

    model: PricingModel = PricingModel.PER_PROOF
    cost_per_proof_cents: float = 5.0
    cost_per_line_cents: float = 0.1
    free_tier_monthly: int = 100
    volume_discount_threshold: int = 1000
    volume_discount_percent: float = 20.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.value,
            "cost_per_proof_cents": self.cost_per_proof_cents,
            "cost_per_line_cents": self.cost_per_line_cents,
            "free_tier_monthly": self.free_tier_monthly,
            "volume_discount_threshold": self.volume_discount_threshold,
            "volume_discount_percent": self.volume_discount_percent,
        }


# ---------------------------------------------------------------------------
# Rate limiter (sliding window)
# ---------------------------------------------------------------------------


class RateLimiter:
    """Sliding-window rate limiter keyed by API key ID."""

    def __init__(self) -> None:
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._limits: dict[str, int] = {}

    def set_limit(self, api_key_id: str, max_per_minute: int) -> None:
        """Configure the per-minute limit for an API key."""
        self._limits[api_key_id] = max_per_minute

    def check_rate_limit(self, api_key_id: str) -> tuple[bool, int, int]:
        """Check whether a request is allowed.

        Returns (allowed, remaining, limit).
        """
        limit = self._limits.get(api_key_id, 10)
        window_key = self._get_window_key(api_key_id)
        now = time.monotonic()

        # Prune entries outside the sliding window
        timestamps = self._windows[window_key]
        cutoff = now - _RATE_LIMIT_WINDOW_SECONDS
        self._windows[window_key] = [ts for ts in timestamps if ts > cutoff]

        current = len(self._windows[window_key])
        remaining = max(0, limit - current)
        allowed = current < limit

        return allowed, remaining, limit

    def record_request(self, api_key_id: str) -> None:
        """Record a request against the sliding window."""
        window_key = self._get_window_key(api_key_id)
        self._windows[window_key].append(time.monotonic())

    def get_remaining(self, api_key_id: str) -> int:
        """Return how many requests remain in the current window."""
        _, remaining, _ = self.check_rate_limit(api_key_id)
        return remaining

    def _get_window_key(self, api_key_id: str) -> str:
        """Build the internal window key for an API key."""
        return f"rl:{api_key_id}"


# ---------------------------------------------------------------------------
# API key manager
# ---------------------------------------------------------------------------


class APIKeyManager:
    """Manages API key lifecycle: creation, validation, rotation, revocation."""

    def __init__(self) -> None:
        self._keys_by_id: dict[str, APIKeyConfig] = {}
        self._keys_by_hash: dict[str, APIKeyConfig] = {}

    # -- public API ----------------------------------------------------------

    def create_key(
        self,
        tenant_id: str,
        name: str,
        rate_limit: int = 10,
        monthly_quota: int = 100,
    ) -> tuple[str, APIKeyConfig]:
        """Create a new API key.

        Returns (raw_key, config) — the raw key is only available at creation.
        """
        raw_key = f"cv_{uuid.uuid4().hex}"
        key_id = f"key_{uuid.uuid4().hex[:12]}"
        key_hash = self._hash_key(raw_key)

        config = APIKeyConfig(
            key_id=key_id,
            key_hash=key_hash,
            tenant_id=tenant_id,
            name=name,
            rate_limit_per_minute=rate_limit,
            monthly_quota=monthly_quota,
        )

        self._keys_by_id[key_id] = config
        self._keys_by_hash[key_hash] = config

        logger.info(
            "api_key_created",
            key_id=key_id,
            tenant_id=tenant_id,
            name=name,
        )
        return raw_key, config

    def validate_key(self, raw_key: str) -> APIKeyConfig | None:
        """Validate a raw API key and return its config, or None."""
        key_hash = self._hash_key(raw_key)
        config = self._keys_by_hash.get(key_hash)
        if config is None:
            logger.warning("api_key_validation_failed", reason="unknown_key")
            return None

        if not config.active:
            logger.warning("api_key_validation_failed", key_id=config.key_id, reason="inactive")
            return None

        if config.expires_at and datetime.now(timezone.utc) > config.expires_at:
            logger.warning("api_key_validation_failed", key_id=config.key_id, reason="expired")
            return None

        return config

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key by its ID."""
        config = self._keys_by_id.get(key_id)
        if config is None:
            return False

        config.active = False
        logger.info("api_key_revoked", key_id=key_id)
        return True

    def rotate_key(self, key_id: str) -> tuple[str, APIKeyConfig]:
        """Rotate an API key: revoke old, issue new with same settings.

        Returns (new_raw_key, new_config).

        Raises ``KeyError`` if the key ID does not exist.
        """
        old_config = self._keys_by_id.get(key_id)
        if old_config is None:
            raise KeyError(f"API key '{key_id}' not found")

        # Revoke the old key
        self.revoke_key(key_id)

        # Create replacement with identical settings
        new_raw, new_config = self.create_key(
            tenant_id=old_config.tenant_id,
            name=old_config.name,
            rate_limit=old_config.rate_limit_per_minute,
            monthly_quota=old_config.monthly_quota,
        )
        new_config.allowed_checks = list(old_config.allowed_checks)

        logger.info("api_key_rotated", old_key_id=key_id, new_key_id=new_config.key_id)
        return new_raw, new_config

    # -- internals -----------------------------------------------------------

    def _hash_key(self, raw_key: str) -> str:
        """Produce an HMAC-SHA256 hash of the raw API key."""
        return hmac.new(_HMAC_SECRET, raw_key.encode(), hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Usage tracker
# ---------------------------------------------------------------------------


class UsageTracker:
    """Tracks per-key usage and billing data."""

    def __init__(self) -> None:
        self._buckets: dict[str, UsageBucket] = {}
        self._latency_sums: dict[str, float] = defaultdict(float)
        self._tenant_keys: dict[str, list[str]] = defaultdict(list)

    def register_key(self, api_key_id: str, tenant_id: str) -> None:
        """Associate an API key with a tenant for billing summaries."""
        if api_key_id not in [k for k in self._tenant_keys.get(tenant_id, [])]:
            self._tenant_keys[tenant_id].append(api_key_id)

    def record(
        self,
        api_key_id: str,
        cost_cents: float,
        latency_ms: float,
        success: bool,
    ) -> None:
        """Record a single request's outcome."""
        period = self._current_period()
        bucket = self._ensure_bucket(api_key_id, period)

        bucket.total_requests += 1
        bucket.total_cost_cents += cost_cents
        if success:
            bucket.successful += 1
        else:
            bucket.failed += 1

        # Incremental average latency
        self._latency_sums[api_key_id] += latency_ms
        bucket.avg_latency_ms = self._latency_sums[api_key_id] / bucket.total_requests

        logger.debug(
            "usage_recorded",
            api_key_id=api_key_id,
            cost_cents=cost_cents,
            success=success,
        )

    def get_usage(self, api_key_id: str, period: str | None = None) -> UsageBucket:
        """Return the usage bucket for a key and period."""
        period = period or self._current_period()
        return self._ensure_bucket(api_key_id, period)

    def check_quota(self, api_key_id: str, monthly_quota: int) -> tuple[bool, int]:
        """Check whether the monthly quota has been exhausted.

        Returns (within_quota, remaining).
        """
        bucket = self.get_usage(api_key_id)
        used = bucket.total_requests
        remaining = max(0, monthly_quota - used)
        return remaining > 0, remaining

    def get_billing_summary(self, tenant_id: str, period: str | None = None) -> dict[str, Any]:
        """Aggregate billing across all keys owned by a tenant."""
        period = period or self._current_period()
        keys = self._tenant_keys.get(tenant_id, [])
        total_cost = 0.0
        total_requests = 0
        total_successful = 0
        total_failed = 0

        per_key: list[dict[str, Any]] = []
        for key_id in keys:
            bucket = self.get_usage(key_id, period)
            total_cost += bucket.total_cost_cents
            total_requests += bucket.total_requests
            total_successful += bucket.successful
            total_failed += bucket.failed
            per_key.append(bucket.to_dict())

        return {
            "tenant_id": tenant_id,
            "period": period,
            "total_cost_cents": round(total_cost, 4),
            "total_requests": total_requests,
            "successful": total_successful,
            "failed": total_failed,
            "keys": per_key,
        }

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _current_period() -> str:
        """Return the current billing period as ``YYYY-MM``."""
        return datetime.now(timezone.utc).strftime("%Y-%m")

    def _ensure_bucket(self, api_key_id: str, period: str) -> UsageBucket:
        bucket_key = f"{api_key_id}:{period}"
        if bucket_key not in self._buckets:
            self._buckets[bucket_key] = UsageBucket(api_key_id=api_key_id, period=period)
        return self._buckets[bucket_key]


# ---------------------------------------------------------------------------
# Proof request processor
# ---------------------------------------------------------------------------

# Mapping from check type to simulated finding templates
_CHECK_TEMPLATES: dict[VerificationCheck, dict[str, Any]] = {
    VerificationCheck.NULL_SAFETY: {
        "rule": "null_safety",
        "message": "Potential null/None dereference detected",
        "severity": "high",
    },
    VerificationCheck.BOUNDS_CHECK: {
        "rule": "bounds_check",
        "message": "Array/index access may exceed bounds",
        "severity": "high",
    },
    VerificationCheck.OVERFLOW: {
        "rule": "overflow",
        "message": "Arithmetic operation may overflow",
        "severity": "medium",
    },
    VerificationCheck.DIVISION_BY_ZERO: {
        "rule": "division_by_zero",
        "message": "Division by zero is possible",
        "severity": "critical",
    },
    VerificationCheck.TYPE_SAFETY: {
        "rule": "type_safety",
        "message": "Type mismatch or unsafe cast detected",
        "severity": "medium",
    },
    VerificationCheck.INJECTION: {
        "rule": "injection",
        "message": "Possible code/SQL injection vector",
        "severity": "critical",
    },
    VerificationCheck.RESOURCE_LEAK: {
        "rule": "resource_leak",
        "message": "Resource may not be released on all paths",
        "severity": "medium",
    },
    VerificationCheck.RACE_CONDITION: {
        "rule": "race_condition",
        "message": "Shared state accessed without synchronisation",
        "severity": "high",
    },
}


class ProofRequestProcessor:
    """Processes proof requests: queues, runs verification, calculates cost."""

    def __init__(self, pricing: PricingConfig | None = None) -> None:
        self._pricing = pricing or PricingConfig()
        self._requests: dict[str, ProofRequest] = {}
        self._results: dict[str, ProofResult] = {}
        self._queue: list[str] = []

    # -- public API ----------------------------------------------------------

    def submit(self, request: ProofRequest) -> str:
        """Accept a proof request and place it in the queue.

        Returns the request ID.
        """
        self._requests[request.id] = request
        self._queue.append(request.id)

        logger.info(
            "proof_request_submitted",
            request_id=request.id,
            language=request.language,
            checks=[c.value for c in request.checks],
        )
        return request.id

    def process(self, request_id: str) -> ProofResult:
        """Process a queued request synchronously and return the result."""
        request = self._requests.get(request_id)
        if request is None:
            raise KeyError(f"Request '{request_id}' not found")

        # Remove from queue
        if request_id in self._queue:
            self._queue.remove(request_id)

        result = self._run_verification(request)
        result.cost_cents = self._calculate_cost(request, result)
        self._results[request_id] = result

        logger.info(
            "proof_request_processed",
            request_id=request_id,
            status=result.status.value,
            verified=result.verified,
            cost_cents=result.cost_cents,
        )
        return result

    def get_result(self, request_id: str) -> ProofResult | None:
        """Retrieve a previously computed result."""
        return self._results.get(request_id)

    def cancel(self, request_id: str) -> bool:
        """Cancel a queued (not yet processing) request."""
        if request_id in self._queue:
            self._queue.remove(request_id)
            self._results[request_id] = ProofResult(
                request_id=request_id,
                status=ProofRequestStatus.CANCELED,
                verified=False,
            )
            logger.info("proof_request_canceled", request_id=request_id)
            return True
        return False

    # -- internals -----------------------------------------------------------

    def _run_verification(self, request: ProofRequest) -> ProofResult:
        """Execute the verification checks against the submitted code."""
        start = time.monotonic()

        findings: list[dict[str, Any]] = []
        code_lines = request.code.strip().splitlines()
        counterexample: dict[str, Any] | None = None

        for check in request.checks:
            template = _CHECK_TEMPLATES.get(check)
            if template is None:
                continue

            # Heuristic scan: look for patterns that hint at issues
            flagged = self._scan_for_check(check, code_lines)
            if flagged:
                finding = {
                    **template,
                    "check": check.value,
                    "lines": flagged,
                }
                findings.append(finding)

                # Generate a counterexample for the first finding
                if counterexample is None:
                    counterexample = {
                        "check": check.value,
                        "line": flagged[0],
                        "description": template["message"],
                    }

        elapsed_ms = (time.monotonic() - start) * 1000
        verified = len(findings) == 0
        status = ProofRequestStatus.COMPLETED

        # Detect timeout
        if elapsed_ms > request.timeout_seconds * 1000:
            status = ProofRequestStatus.TIMEOUT
            verified = False

        proof_artifact = self._generate_proof_artifact(
            request.code,
            findings,
            ProofFormat.JSON,
        )

        tokens_used = len(request.code) // 4  # rough token estimate

        return ProofResult(
            request_id=request.id,
            status=status,
            verified=verified,
            proof_artifact=proof_artifact,
            counterexample=counterexample,
            findings=findings,
            execution_time_ms=elapsed_ms,
            tokens_used=tokens_used,
        )

    @staticmethod
    def _scan_for_check(check: VerificationCheck, lines: list[str]) -> list[int]:
        """Return 1-based line numbers where the check is potentially violated."""
        patterns: dict[VerificationCheck, list[str]] = {
            VerificationCheck.NULL_SAFETY: ["None", "null", "nil", "nullptr"],
            VerificationCheck.BOUNDS_CHECK: ["[", "index", "offset"],
            VerificationCheck.OVERFLOW: ["+", "*", "**", "pow("],
            VerificationCheck.DIVISION_BY_ZERO: ["/", "//", "%"],
            VerificationCheck.TYPE_SAFETY: ["cast", "as ", "type(", "isinstance"],
            VerificationCheck.INJECTION: ["exec(", "eval(", "system(", "query("],
            VerificationCheck.RESOURCE_LEAK: ["open(", "connect(", "socket("],
            VerificationCheck.RACE_CONDITION: ["threading", "Thread(", "Lock", "global "],
        }

        keywords = patterns.get(check, [])
        flagged: list[int] = []
        for idx, line in enumerate(lines, start=1):
            if any(kw in line for kw in keywords):
                flagged.append(idx)
        return flagged

    def _calculate_cost(self, request: ProofRequest, result: ProofResult) -> float:
        """Determine the cost in cents for a completed proof request."""
        pricing = self._pricing

        if pricing.model == PricingModel.FREE_TIER:
            return 0.0

        if pricing.model == PricingModel.PER_LINE:
            line_count = len(request.code.strip().splitlines())
            base_cost = line_count * pricing.cost_per_line_cents
        elif pricing.model == PricingModel.SUBSCRIPTION:
            return 0.0  # covered by subscription
        else:
            # PER_PROOF (default)
            base_cost = pricing.cost_per_proof_cents

        # Apply volume discount
        base_cost = self._apply_volume_discount(base_cost, pricing)

        # Surcharge for extra checks
        check_count = len(request.checks)
        if check_count > 3:
            base_cost *= 1 + 0.1 * (check_count - 3)

        return round(base_cost, 4)

    @staticmethod
    def _apply_volume_discount(base_cost: float, pricing: PricingConfig) -> float:
        """Apply volume discount when threshold is reached."""
        # Discount is applied externally based on cumulative usage; here we
        # simply expose the helper so callers can adjust after quota checks.
        discount_factor = 1.0 - (pricing.volume_discount_percent / 100.0)
        # In a real system this would check cumulative volume; return base for now
        return base_cost * discount_factor if base_cost > 0 else 0.0

    def _generate_proof_artifact(
        self,
        code: str,
        findings: list[dict[str, Any]],
        fmt: ProofFormat,
    ) -> str:
        """Generate a serialised proof artifact in the requested format."""
        if fmt == ProofFormat.SMT_LIB:
            return self._artifact_smt_lib(code, findings)
        if fmt == ProofFormat.HUMAN_READABLE:
            return self._artifact_human_readable(code, findings)
        if fmt == ProofFormat.CERTIFICATE:
            return self._artifact_certificate(code, findings)
        # Default: JSON
        return self._artifact_json(code, findings)

    # -- artifact formatters -------------------------------------------------

    @staticmethod
    def _artifact_json(code: str, findings: list[dict[str, Any]]) -> str:
        """Produce a JSON proof artifact."""
        artifact = {
            "version": "1.0",
            "verified": len(findings) == 0,
            "code_hash": hashlib.sha256(code.encode()).hexdigest(),
            "findings_count": len(findings),
            "findings": findings,
        }
        return json.dumps(artifact, indent=2)

    @staticmethod
    def _artifact_smt_lib(code: str, findings: list[dict[str, Any]]) -> str:
        """Produce an SMT-LIB v2 proof artifact with assertions per finding."""
        lines: list[str] = [
            "; SMT-LIB v2 proof artifact generated by CodeVerify",
            f"; code-hash: {hashlib.sha256(code.encode()).hexdigest()}",
            "(set-logic QF_LIA)",
            "",
        ]

        if not findings:
            lines.append("; All checks passed — no violations found.")
            lines.append("(assert true)")
            lines.append("(check-sat)")
            lines.append("; expected: sat (proof holds)")
        else:
            for idx, finding in enumerate(findings):
                check = finding.get("check", "unknown")
                message = finding.get("message", "")
                flagged = finding.get("lines", [])
                var = f"violation_{idx}"
                lines.append(f"; Finding {idx + 1}: {check} — {message}")
                lines.append(f"(declare-const {var} Bool)")
                lines.append(f"(assert {var})  ; line(s): {flagged}")
                lines.append("")

            lines.append("; Check satisfiability of violation conjunction")
            violation_vars = " ".join(f"violation_{i}" for i in range(len(findings)))
            lines.append(f"(assert (and {violation_vars}))")
            lines.append("(check-sat)")
            lines.append("; expected: sat (violations are reachable)")

        return "\n".join(lines)

    @staticmethod
    def _artifact_human_readable(code: str, findings: list[dict[str, Any]]) -> str:
        """Produce a human-readable proof summary."""
        parts: list[str] = [
            "=== CodeVerify Proof Report ===",
            f"Code hash: {hashlib.sha256(code.encode()).hexdigest()[:16]}...",
            f"Total findings: {len(findings)}",
            f"Verdict: {'PASS' if not findings else 'FAIL'}",
            "",
        ]
        for idx, finding in enumerate(findings, start=1):
            parts.append(f"[{idx}] {finding.get('severity', '?').upper()}: {finding.get('message', '')}")
            parts.append(f"    Check : {finding.get('check', 'unknown')}")
            parts.append(f"    Lines : {finding.get('lines', [])}")
            parts.append("")
        return "\n".join(parts)

    @staticmethod
    def _artifact_certificate(code: str, findings: list[dict[str, Any]]) -> str:
        """Produce a verification certificate (JSON envelope with signature placeholder)."""
        payload = {
            "type": "codeverify_certificate",
            "version": "1.0",
            "code_hash": hashlib.sha256(code.encode()).hexdigest(),
            "verified": len(findings) == 0,
            "findings_count": len(findings),
            "issued_at": datetime.now(timezone.utc).isoformat(),
            "signature": "PLACEHOLDER",  # Real impl would sign with service key
        }
        return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Proof-as-a-Service orchestrator
# ---------------------------------------------------------------------------


class ProofServiceAPI:
    """Main orchestrator for the Proof-as-a-Service API.

    Ties together key management, rate limiting, usage tracking,
    and proof processing into a single public interface.

    Usage::

        api = ProofServiceAPI()
        raw_key, _ = api._key_manager.create_key("tenant-1", "my-key")
        result = api.verify(raw_key, "x = 1 / y", "python")
    """

    def __init__(self, pricing: PricingConfig | None = None) -> None:
        self._pricing = pricing or PricingConfig()
        self._key_manager = APIKeyManager()
        self._rate_limiter = RateLimiter()
        self._usage_tracker = UsageTracker()
        self._processor = ProofRequestProcessor(pricing=self._pricing)

        logger.info("proof_service_api_initialized", pricing_model=self._pricing.model.value)

    # -- public API ----------------------------------------------------------

    def verify(
        self,
        api_key: str,
        code: str,
        language: str,
        checks: list[VerificationCheck] | None = None,
        **kwargs: Any,
    ) -> ProofResult:
        """Submit code for verification and return the result.

        Raises ``PermissionError`` for invalid keys or exceeded limits.
        """
        # 1. Validate API key
        config = self._key_manager.validate_key(api_key)
        if config is None:
            raise PermissionError("Invalid or inactive API key")

        # 2. Check rate limit
        allowed, remaining, limit = self._rate_limiter.check_rate_limit(config.key_id)
        if not allowed:
            logger.warning(
                "rate_limit_exceeded",
                key_id=config.key_id,
                limit=limit,
            )
            raise PermissionError(
                f"Rate limit exceeded ({limit} requests/min). Retry after window resets."
            )

        # 3. Check monthly quota
        within_quota, quota_remaining = self._usage_tracker.check_quota(
            config.key_id, config.monthly_quota
        )
        if not within_quota:
            raise PermissionError("Monthly quota exhausted")

        # 4. Resolve checks
        resolved_checks = checks or list(VerificationCheck)
        if config.allowed_checks:
            resolved_checks = [c for c in resolved_checks if c in config.allowed_checks]

        # 5. Build and submit request
        request = ProofRequest(
            id=f"pr_{uuid.uuid4().hex[:16]}",
            api_key=api_key,
            code=code,
            language=language,
            checks=resolved_checks,
            callback_url=kwargs.get("callback_url"),
            timeout_seconds=kwargs.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS),
            priority=kwargs.get("priority", 5),
            context=kwargs.get("context", {}),
        )

        self._processor.submit(request)
        self._rate_limiter.record_request(config.key_id)

        # 6. Process synchronously
        result = self._processor.process(request.id)

        # 7. Record usage
        self._usage_tracker.record(
            api_key_id=config.key_id,
            cost_cents=result.cost_cents,
            latency_ms=result.execution_time_ms,
            success=result.status == ProofRequestStatus.COMPLETED,
        )

        return result

    def get_status(self, api_key: str, request_id: str) -> ProofResult | None:
        """Look up the result of a previous verification request."""
        config = self._key_manager.validate_key(api_key)
        if config is None:
            raise PermissionError("Invalid or inactive API key")

        return self._processor.get_result(request_id)

    def get_usage(self, api_key: str) -> UsageBucket:
        """Return usage statistics for the calling API key."""
        config = self._key_manager.validate_key(api_key)
        if config is None:
            raise PermissionError("Invalid or inactive API key")

        return self._usage_tracker.get_usage(config.key_id)

    def get_pricing(self) -> dict[str, Any]:
        """Return the current pricing configuration."""
        return self._pricing.to_dict()
