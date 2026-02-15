"""Webhook Signature Verification.

Verifies webhook payload authenticity for GitHub, GitLab, and Bitbucket
using HMAC-SHA256/SHA1 signatures. Provides replay attack prevention
and payload size enforcement.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum


class WebhookProvider(str, Enum):
    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"


@dataclass
class WebhookSecurityConfig:
    """Configuration for webhook security."""

    max_payload_bytes: int = 25 * 1024 * 1024  # 25 MB
    nonce_ttl_seconds: int = 300  # 5 minutes
    max_nonce_cache_size: int = 10_000
    timestamp_tolerance_seconds: int = 300  # 5 minutes


class PayloadTooLargeError(Exception):
    pass


class InvalidSignatureError(Exception):
    pass


class ReplayAttackError(Exception):
    pass


class TimestampValidationError(Exception):
    pass


# =============================================================================
# Nonce tracker for replay prevention
# =============================================================================


class NonceTracker:
    """Tracks seen nonces to prevent replay attacks."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 10_000):
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._seen: OrderedDict[str, float] = OrderedDict()

    def check_and_record(self, nonce: str) -> bool:
        """Return True if nonce is fresh (not seen), False if replay.

        Also records the nonce and evicts expired entries.
        """
        now = time.time()
        self._evict_expired(now)

        if nonce in self._seen:
            return False

        if len(self._seen) >= self._max_size:
            self._seen.popitem(last=False)

        self._seen[nonce] = now
        return True

    def _evict_expired(self, now: float) -> None:
        while self._seen:
            oldest_key = next(iter(self._seen))
            if now - self._seen[oldest_key] > self._ttl:
                self._seen.pop(oldest_key)
            else:
                break

    @property
    def size(self) -> int:
        return len(self._seen)


# =============================================================================
# Signature verification
# =============================================================================


def _compute_hmac(secret: str, payload: bytes, algorithm: str = "sha256") -> str:
    """Compute HMAC for a payload."""
    if algorithm == "sha256":
        digest = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    elif algorithm == "sha1":
        digest = hmac.new(secret.encode(), payload, hashlib.sha1).hexdigest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return digest


def verify_github_signature(
    payload: bytes,
    secret: str,
    signature_header: str,
) -> bool:
    """Verify a GitHub webhook signature.

    GitHub sends `X-Hub-Signature-256: sha256=<hex>`.
    """
    if not signature_header:
        return False

    parts = signature_header.split("=", 1)
    if len(parts) != 2:
        return False

    algorithm, received_hash = parts
    if algorithm == "sha256":
        expected = _compute_hmac(secret, payload, "sha256")
    elif algorithm == "sha1":
        expected = _compute_hmac(secret, payload, "sha1")
    else:
        return False

    return hmac.compare_digest(expected, received_hash)


def verify_gitlab_signature(
    payload: bytes,
    secret: str,
    token_header: str,
) -> bool:
    """Verify a GitLab webhook token.

    GitLab sends `X-Gitlab-Token: <secret>` (plain comparison).
    """
    if not token_header:
        return False
    return hmac.compare_digest(secret, token_header)


def verify_bitbucket_signature(
    payload: bytes,
    secret: str,
    signature_header: str,
) -> bool:
    """Verify a Bitbucket Cloud webhook signature.

    Bitbucket sends `X-Hub-Signature: sha256=<hex>`.
    """
    return verify_github_signature(payload, secret, signature_header)


# =============================================================================
# WebhookVerifier (high-level API)
# =============================================================================


class WebhookVerifier:
    """High-level webhook verification with replay protection.

    Usage:
        verifier = WebhookVerifier(secrets={"github": "whsec_..."})
        verifier.verify(
            provider=WebhookProvider.GITHUB,
            payload=request_body,
            signature_header=request.headers["X-Hub-Signature-256"],
            delivery_id=request.headers.get("X-GitHub-Delivery"),
        )
    """

    def __init__(
        self,
        secrets: dict[str, str] | None = None,
        config: WebhookSecurityConfig | None = None,
    ):
        self._secrets: dict[str, str] = secrets or {}
        self._config = config or WebhookSecurityConfig()
        self._nonce_tracker = NonceTracker(
            ttl_seconds=self._config.nonce_ttl_seconds,
            max_size=self._config.max_nonce_cache_size,
        )

    def set_secret(self, provider: str, secret: str) -> None:
        self._secrets[provider] = secret

    def verify(
        self,
        provider: WebhookProvider,
        payload: bytes,
        signature_header: str = "",
        delivery_id: str | None = None,
        timestamp: float | None = None,
    ) -> VerificationResult:
        """Verify a webhook payload.

        Args:
            provider: The VCS provider.
            payload: Raw request body bytes.
            signature_header: Signature header value.
            delivery_id: Unique delivery ID (for replay prevention).
            timestamp: Request timestamp (for freshness check).

        Returns:
            VerificationResult with success/failure details.

        Raises:
            PayloadTooLargeError, InvalidSignatureError, ReplayAttackError,
            TimestampValidationError.
        """
        # Check payload size
        if len(payload) > self._config.max_payload_bytes:
            raise PayloadTooLargeError(
                f"Payload {len(payload)} bytes exceeds limit {self._config.max_payload_bytes}"
            )

        # Check timestamp freshness
        if timestamp is not None:
            age = abs(time.time() - timestamp)
            if age > self._config.timestamp_tolerance_seconds:
                raise TimestampValidationError(
                    f"Timestamp age {age:.0f}s exceeds tolerance "
                    f"{self._config.timestamp_tolerance_seconds}s"
                )

        # Check replay
        if delivery_id is not None:
            if not self._nonce_tracker.check_and_record(delivery_id):
                raise ReplayAttackError(f"Duplicate delivery ID: {delivery_id}")

        # Verify signature
        secret = self._secrets.get(provider.value, "")
        if not secret:
            return VerificationResult(
                valid=False,
                provider=provider,
                error="No secret configured for provider",
            )

        if provider == WebhookProvider.GITHUB:
            valid = verify_github_signature(payload, secret, signature_header)
        elif provider == WebhookProvider.GITLAB:
            valid = verify_gitlab_signature(payload, secret, signature_header)
        elif provider == WebhookProvider.BITBUCKET:
            valid = verify_bitbucket_signature(payload, secret, signature_header)
        else:
            valid = False

        if not valid:
            raise InvalidSignatureError(f"Invalid signature for {provider.value} webhook")

        return VerificationResult(valid=True, provider=provider)


@dataclass
class VerificationResult:
    """Result of webhook verification."""

    valid: bool
    provider: WebhookProvider = WebhookProvider.GITHUB
    error: str = ""
