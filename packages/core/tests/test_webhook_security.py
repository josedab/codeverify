"""Tests for webhook_security module."""

from __future__ import annotations

import hashlib
import hmac
import time

import pytest

from codeverify_core.webhook_security import (
    InvalidSignatureError,
    NonceTracker,
    PayloadTooLargeError,
    ReplayAttackError,
    TimestampValidationError,
    WebhookProvider,
    WebhookSecurityConfig,
    WebhookVerifier,
    verify_bitbucket_signature,
    verify_github_signature,
    verify_gitlab_signature,
)

SECRET = "test_webhook_secret_123"
PAYLOAD = b'{"action": "opened", "number": 42}'


def _gh_sig(payload: bytes, secret: str, algo: str = "sha256") -> str:
    if algo == "sha256":
        h = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
    else:
        h = hmac.new(secret.encode(), payload, hashlib.sha1).hexdigest()
    return f"{algo}={h}"


# =============================================================================
# Low-level signature verification
# =============================================================================


class TestGitHubSignature:
    def test_valid_sha256(self):
        sig = _gh_sig(PAYLOAD, SECRET, "sha256")
        assert verify_github_signature(PAYLOAD, SECRET, sig) is True

    def test_valid_sha1(self):
        sig = _gh_sig(PAYLOAD, SECRET, "sha1")
        assert verify_github_signature(PAYLOAD, SECRET, sig) is True

    def test_invalid_signature(self):
        assert verify_github_signature(PAYLOAD, SECRET, "sha256=bad") is False

    def test_empty_header(self):
        assert verify_github_signature(PAYLOAD, SECRET, "") is False

    def test_malformed_header(self):
        assert verify_github_signature(PAYLOAD, SECRET, "noseparator") is False

    def test_wrong_secret(self):
        sig = _gh_sig(PAYLOAD, "wrong_secret")
        assert verify_github_signature(PAYLOAD, SECRET, sig) is False

    def test_tampered_payload(self):
        sig = _gh_sig(PAYLOAD, SECRET)
        assert verify_github_signature(b"tampered", SECRET, sig) is False


class TestGitLabSignature:
    def test_valid_token(self):
        assert verify_gitlab_signature(PAYLOAD, SECRET, SECRET) is True

    def test_invalid_token(self):
        assert verify_gitlab_signature(PAYLOAD, SECRET, "wrong") is False

    def test_empty_token(self):
        assert verify_gitlab_signature(PAYLOAD, SECRET, "") is False


class TestBitbucketSignature:
    def test_valid_signature(self):
        sig = _gh_sig(PAYLOAD, SECRET)
        assert verify_bitbucket_signature(PAYLOAD, SECRET, sig) is True


# =============================================================================
# NonceTracker
# =============================================================================


class TestNonceTracker:
    def test_first_nonce_is_fresh(self):
        tracker = NonceTracker()
        assert tracker.check_and_record("abc-123") is True

    def test_duplicate_nonce_is_replay(self):
        tracker = NonceTracker()
        tracker.check_and_record("abc-123")
        assert tracker.check_and_record("abc-123") is False

    def test_different_nonces_are_fresh(self):
        tracker = NonceTracker()
        assert tracker.check_and_record("a") is True
        assert tracker.check_and_record("b") is True

    def test_expired_nonces_are_evicted(self):
        tracker = NonceTracker(ttl_seconds=0)
        tracker.check_and_record("old")
        time.sleep(0.01)
        assert tracker.check_and_record("old") is True  # Evicted, so fresh again

    def test_max_size_eviction(self):
        tracker = NonceTracker(max_size=3)
        for i in range(5):
            tracker.check_and_record(f"n{i}")
        assert tracker.size <= 3

    def test_size_property(self):
        tracker = NonceTracker()
        tracker.check_and_record("a")
        tracker.check_and_record("b")
        assert tracker.size == 2


# =============================================================================
# WebhookVerifier (high-level)
# =============================================================================


class TestWebhookVerifier:
    def test_github_verification_success(self):
        verifier = WebhookVerifier(secrets={"github": SECRET})
        sig = _gh_sig(PAYLOAD, SECRET)
        result = verifier.verify(
            provider=WebhookProvider.GITHUB,
            payload=PAYLOAD,
            signature_header=sig,
        )
        assert result.valid is True
        assert result.provider == WebhookProvider.GITHUB

    def test_gitlab_verification_success(self):
        verifier = WebhookVerifier(secrets={"gitlab": SECRET})
        result = verifier.verify(
            provider=WebhookProvider.GITLAB,
            payload=PAYLOAD,
            signature_header=SECRET,
        )
        assert result.valid is True

    def test_bitbucket_verification_success(self):
        verifier = WebhookVerifier(secrets={"bitbucket": SECRET})
        sig = _gh_sig(PAYLOAD, SECRET)
        result = verifier.verify(
            provider=WebhookProvider.BITBUCKET,
            payload=PAYLOAD,
            signature_header=sig,
        )
        assert result.valid is True

    def test_invalid_signature_raises(self):
        verifier = WebhookVerifier(secrets={"github": SECRET})
        with pytest.raises(InvalidSignatureError):
            verifier.verify(
                provider=WebhookProvider.GITHUB,
                payload=PAYLOAD,
                signature_header="sha256=bad",
            )

    def test_missing_secret_returns_invalid(self):
        verifier = WebhookVerifier()
        result = verifier.verify(
            provider=WebhookProvider.GITHUB,
            payload=PAYLOAD,
            signature_header="sha256=abc",
        )
        assert result.valid is False
        assert "No secret" in result.error

    def test_payload_too_large(self):
        config = WebhookSecurityConfig(max_payload_bytes=10)
        verifier = WebhookVerifier(secrets={"github": SECRET}, config=config)
        with pytest.raises(PayloadTooLargeError):
            verifier.verify(
                provider=WebhookProvider.GITHUB,
                payload=b"x" * 100,
                signature_header="sha256=abc",
            )

    def test_replay_attack_prevention(self):
        verifier = WebhookVerifier(secrets={"github": SECRET})
        sig = _gh_sig(PAYLOAD, SECRET)
        verifier.verify(
            provider=WebhookProvider.GITHUB,
            payload=PAYLOAD,
            signature_header=sig,
            delivery_id="delivery-001",
        )
        with pytest.raises(ReplayAttackError):
            verifier.verify(
                provider=WebhookProvider.GITHUB,
                payload=PAYLOAD,
                signature_header=sig,
                delivery_id="delivery-001",
            )

    def test_timestamp_validation(self):
        config = WebhookSecurityConfig(timestamp_tolerance_seconds=5)
        verifier = WebhookVerifier(secrets={"github": SECRET}, config=config)
        old_ts = time.time() - 600
        with pytest.raises(TimestampValidationError):
            verifier.verify(
                provider=WebhookProvider.GITHUB,
                payload=PAYLOAD,
                signature_header=_gh_sig(PAYLOAD, SECRET),
                timestamp=old_ts,
            )

    def test_fresh_timestamp_passes(self):
        verifier = WebhookVerifier(secrets={"github": SECRET})
        result = verifier.verify(
            provider=WebhookProvider.GITHUB,
            payload=PAYLOAD,
            signature_header=_gh_sig(PAYLOAD, SECRET),
            timestamp=time.time(),
        )
        assert result.valid is True

    def test_set_secret(self):
        verifier = WebhookVerifier()
        verifier.set_secret("github", SECRET)
        result = verifier.verify(
            provider=WebhookProvider.GITHUB,
            payload=PAYLOAD,
            signature_header=_gh_sig(PAYLOAD, SECRET),
        )
        assert result.valid is True
