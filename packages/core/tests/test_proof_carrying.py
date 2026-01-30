"""Tests for Proof-Carrying PRs module."""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from codeverify_core.proof_carrying import (
    ProofAttestation,
    ProofCarryingManager,
    ProofSerializer,
    VerificationProof,
)


class TestVerificationProof:
    """Tests for VerificationProof dataclass."""

    def test_create_proof(self):
        """Can create a verification proof."""
        proof = VerificationProof(
            proof_id="proof-123",
            code_hash="abc123",
            verification_type="formal",
            result="passed",
            verifier_version="1.0.0",
            timestamp=datetime.utcnow(),
        )
        assert proof.proof_id == "proof-123"
        assert proof.result == "passed"

    def test_proof_with_metadata(self):
        """Proof can include metadata."""
        proof = VerificationProof(
            proof_id="proof-456",
            code_hash="def456",
            verification_type="static",
            result="passed",
            verifier_version="1.0.0",
            timestamp=datetime.utcnow(),
            metadata={"checks": ["null_safety", "bounds"]},
        )
        assert proof.metadata["checks"] == ["null_safety", "bounds"]


class TestProofAttestation:
    """Tests for ProofAttestation dataclass."""

    def test_create_attestation(self):
        """Can create an attestation."""
        proof = VerificationProof(
            proof_id="proof-1",
            code_hash="hash1",
            verification_type="formal",
            result="passed",
            verifier_version="1.0.0",
            timestamp=datetime.utcnow(),
        )
        attestation = ProofAttestation(
            attestation_id="att-1",
            proof=proof,
            signature="sig123",
            signed_at=datetime.utcnow(),
        )
        assert attestation.attestation_id == "att-1"
        assert attestation.signature == "sig123"


class TestProofSerializer:
    """Tests for ProofSerializer."""

    def test_serialize_proof(self):
        """Serializes proof to JSON."""
        proof = VerificationProof(
            proof_id="proof-1",
            code_hash="hash1",
            verification_type="formal",
            result="passed",
            verifier_version="1.0.0",
            timestamp=datetime(2024, 1, 15, 12, 0, 0),
        )
        
        serialized = ProofSerializer.serialize(proof)
        assert isinstance(serialized, str)
        
        # Should be valid JSON
        data = json.loads(serialized)
        assert data["proof_id"] == "proof-1"
        assert data["code_hash"] == "hash1"

    def test_deserialize_proof(self):
        """Deserializes JSON to proof."""
        json_str = json.dumps({
            "proof_id": "proof-2",
            "code_hash": "hash2",
            "verification_type": "static",
            "result": "passed",
            "verifier_version": "1.0.0",
            "timestamp": "2024-01-15T12:00:00",
        })
        
        proof = ProofSerializer.deserialize(json_str)
        assert proof.proof_id == "proof-2"
        assert proof.verification_type == "static"

    def test_round_trip(self):
        """Serialize then deserialize preserves data."""
        original = VerificationProof(
            proof_id="proof-rt",
            code_hash="hashrt",
            verification_type="ai",
            result="warning",
            verifier_version="2.0.0",
            timestamp=datetime(2024, 6, 1, 10, 30, 0),
            metadata={"findings": 3},
        )
        
        serialized = ProofSerializer.serialize(original)
        restored = ProofSerializer.deserialize(serialized)
        
        assert restored.proof_id == original.proof_id
        assert restored.code_hash == original.code_hash
        assert restored.result == original.result

    def test_compress_proof(self):
        """Compresses proof for transport."""
        proof = VerificationProof(
            proof_id="proof-compress",
            code_hash="hash" * 100,  # Long hash
            verification_type="formal",
            result="passed",
            verifier_version="1.0.0",
            timestamp=datetime.utcnow(),
            metadata={"large": "data" * 100},
        )
        
        compressed = ProofSerializer.compress(proof)
        assert isinstance(compressed, bytes)
        
        # Compressed should be smaller than original JSON
        original_size = len(ProofSerializer.serialize(proof).encode())
        assert len(compressed) < original_size

    def test_decompress_proof(self):
        """Decompresses proof from bytes."""
        original = VerificationProof(
            proof_id="proof-decompress",
            code_hash="hashdc",
            verification_type="static",
            result="passed",
            verifier_version="1.0.0",
            timestamp=datetime(2024, 1, 1, 0, 0, 0),
        )
        
        compressed = ProofSerializer.compress(original)
        restored = ProofSerializer.decompress(compressed)
        
        assert restored.proof_id == original.proof_id


class TestProofCarryingManager:
    """Tests for ProofCarryingManager."""

    def test_create_manager(self):
        """Can create manager with secret key."""
        manager = ProofCarryingManager(secret_key="test-secret-key-123")
        assert manager is not None

    def test_create_proof(self):
        """Creates a new verification proof."""
        manager = ProofCarryingManager(secret_key="secret")
        
        proof = manager.create_proof(
            code_hash="abc123",
            verification_type="formal",
            result="passed",
        )
        
        assert proof.code_hash == "abc123"
        assert proof.verification_type == "formal"
        assert proof.proof_id is not None

    def test_sign_proof(self):
        """Signs a proof and creates attestation."""
        manager = ProofCarryingManager(secret_key="my-secret-key")
        
        proof = manager.create_proof(
            code_hash="xyz789",
            verification_type="static",
            result="passed",
        )
        
        attestation = manager.sign_proof(proof)
        
        assert attestation.proof == proof
        assert attestation.signature is not None
        assert len(attestation.signature) > 0

    def test_verify_attestation_valid(self):
        """Verifies a valid attestation."""
        secret = "verification-secret"
        manager = ProofCarryingManager(secret_key=secret)
        
        proof = manager.create_proof(
            code_hash="validhash",
            verification_type="formal",
            result="passed",
        )
        attestation = manager.sign_proof(proof)
        
        # Same manager should verify its own attestations
        assert manager.verify_attestation(attestation) is True

    def test_verify_attestation_invalid_signature(self):
        """Rejects attestation with invalid signature."""
        manager = ProofCarryingManager(secret_key="secret1")
        
        proof = manager.create_proof(
            code_hash="hash",
            verification_type="static",
            result="passed",
        )
        attestation = manager.sign_proof(proof)
        
        # Tamper with signature
        attestation.signature = "tampered-signature"
        
        assert manager.verify_attestation(attestation) is False

    def test_verify_attestation_wrong_key(self):
        """Rejects attestation signed with different key."""
        manager1 = ProofCarryingManager(secret_key="key1")
        manager2 = ProofCarryingManager(secret_key="key2")
        
        proof = manager1.create_proof(
            code_hash="hash",
            verification_type="formal",
            result="passed",
        )
        attestation = manager1.sign_proof(proof)
        
        # Different manager should not verify
        assert manager2.verify_attestation(attestation) is False

    def test_attestation_expiry(self):
        """Attestations can expire."""
        manager = ProofCarryingManager(
            secret_key="secret",
            attestation_ttl_hours=1,
        )
        
        proof = manager.create_proof(
            code_hash="hash",
            verification_type="static",
            result="passed",
        )
        attestation = manager.sign_proof(proof)
        
        # Manually set signed_at to past
        attestation.signed_at = datetime.utcnow() - timedelta(hours=2)
        
        assert manager.verify_attestation(attestation, check_expiry=True) is False

    def test_embed_in_commit_message(self):
        """Embeds attestation in commit message."""
        manager = ProofCarryingManager(secret_key="secret")
        
        proof = manager.create_proof(
            code_hash="hash",
            verification_type="formal",
            result="passed",
        )
        attestation = manager.sign_proof(proof)
        
        original_message = "feat: add new feature\n\nSome description"
        embedded = manager.embed_in_commit_message(attestation, original_message)
        
        assert "feat: add new feature" in embedded
        assert "CodeVerify-Attestation:" in embedded

    def test_extract_from_commit_message(self):
        """Extracts attestation from commit message."""
        manager = ProofCarryingManager(secret_key="secret")
        
        proof = manager.create_proof(
            code_hash="extracthash",
            verification_type="formal",
            result="passed",
        )
        attestation = manager.sign_proof(proof)
        
        # Embed then extract
        message = manager.embed_in_commit_message(attestation, "Original message")
        extracted = manager.extract_from_commit_message(message)
        
        assert extracted is not None
        assert extracted.proof.code_hash == "extracthash"

    def test_extract_from_message_without_attestation(self):
        """Returns None for message without attestation."""
        manager = ProofCarryingManager(secret_key="secret")
        
        result = manager.extract_from_commit_message("Regular commit message")
        assert result is None

    def test_create_pr_comment(self):
        """Creates PR comment with attestation."""
        manager = ProofCarryingManager(secret_key="secret")
        
        proof = manager.create_proof(
            code_hash="prhash",
            verification_type="formal",
            result="passed",
        )
        attestation = manager.sign_proof(proof)
        
        comment = manager.create_pr_comment(attestation)
        
        assert "Verification Attestation" in comment
        assert "passed" in comment.lower()


class TestProofCarryingEdgeCases:
    """Edge case tests for proof-carrying functionality."""

    def test_empty_secret_key(self):
        """Handles empty secret key."""
        with pytest.raises(ValueError):
            ProofCarryingManager(secret_key="")

    def test_special_characters_in_metadata(self):
        """Handles special characters in metadata."""
        manager = ProofCarryingManager(secret_key="secret")
        
        proof = manager.create_proof(
            code_hash="hash",
            verification_type="static",
            result="passed",
            metadata={"message": "Test with 'quotes' and \"double quotes\""},
        )
        
        attestation = manager.sign_proof(proof)
        assert manager.verify_attestation(attestation) is True

    def test_unicode_in_proof(self):
        """Handles unicode characters."""
        manager = ProofCarryingManager(secret_key="secret")
        
        proof = manager.create_proof(
            code_hash="hash",
            verification_type="static",
            result="passed",
            metadata={"desc": "Unicode: 你好世界 🔐"},
        )
        
        serialized = ProofSerializer.serialize(proof)
        restored = ProofSerializer.deserialize(serialized)
        
        assert "你好世界" in restored.metadata["desc"]

    def test_large_metadata(self):
        """Handles large metadata."""
        manager = ProofCarryingManager(secret_key="secret")
        
        large_data = {"items": [f"item-{i}" for i in range(1000)]}
        
        proof = manager.create_proof(
            code_hash="hash",
            verification_type="static",
            result="passed",
            metadata=large_data,
        )
        
        attestation = manager.sign_proof(proof)
        assert manager.verify_attestation(attestation) is True
