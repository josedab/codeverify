"""Tests for Proof-Carrying PRs module."""

import hashlib
from datetime import datetime, timedelta

from codeverify_core.proof_carrying import (
    ProofAttestation,
    ProofCarryingPRManager,
    ProofCompressor,
    ProofMetadata,
    ProofSerializer,
    ProofStatus,
    VerificationProof,
    VerificationType,
)


def _make_verification_result(**overrides):
    """Helper to create a verification result dict for create_proof."""
    result = {
        "formula": "(assert (> x 0))",
        "satisfiable": False,
        "solver_version": "z3-4.12",
        "proof_time_ms": 42.0,
    }
    result.update(overrides)
    return result


class TestVerificationProof:
    """Tests for VerificationProof dataclass."""

    def test_create_proof(self):
        """Can create a verification proof."""
        now = datetime.utcnow()
        metadata = ProofMetadata(
            proof_id="proof-123",
            timestamp=now,
            verification_type=VerificationType.FORMAL,
            verifier_version="1.0.0",
        )
        proof = VerificationProof(
            proof_id="proof-123",
            commit_sha="abc123",
            file_path="src/main.py",
            function_name="foo",
            verification_type=VerificationType.FORMAL,
            result="proven",
            formula="(assert true)",
            formula_hash=hashlib.sha256(b"(assert true)").hexdigest(),
            counterexample=None,
            proof_time_ms=10.0,
            metadata=metadata,
        )
        assert proof.proof_id == "proof-123"
        assert proof.result == "proven"

    def test_proof_with_counterexample(self):
        """Proof can include counterexample data."""
        now = datetime.utcnow()
        metadata = ProofMetadata(
            proof_id="proof-456",
            timestamp=now,
            verification_type=VerificationType.SECURITY,
            verifier_version="1.0.0",
        )
        proof = VerificationProof(
            proof_id="proof-456",
            commit_sha="def456",
            file_path="src/lib.py",
            function_name="bar",
            verification_type=VerificationType.SECURITY,
            result="counterexample",
            formula="(assert (< x 0))",
            formula_hash=hashlib.sha256(b"(assert (< x 0))").hexdigest(),
            counterexample={"x": -1},
            proof_time_ms=5.0,
            metadata=metadata,
        )
        assert proof.counterexample == {"x": -1}


class TestProofAttestation:
    """Tests for ProofAttestation dataclass."""

    def test_create_attestation(self):
        """Can create an attestation."""
        now = datetime.utcnow()
        attestation = ProofAttestation(
            attestation_id="att-1",
            pr_number=42,
            repo_full_name="org/repo",
            head_sha="abc123",
            base_sha=None,
            proofs=[],
            summary={"total_proofs": 0},
            created_at=now,
            expires_at=now + timedelta(hours=168),
            signature="sig123",
        )
        assert attestation.attestation_id == "att-1"
        assert attestation.signature == "sig123"


class TestProofSerializer:
    """Tests for ProofSerializer."""

    def _make_proof(self, proof_id="proof-1", result="proven"):
        now = datetime(2024, 1, 15, 12, 0, 0)
        formula = "(assert true)"
        metadata = ProofMetadata(
            proof_id=proof_id,
            timestamp=now,
            verification_type=VerificationType.FORMAL,
            verifier_version="1.0.0",
        )
        return VerificationProof(
            proof_id=proof_id,
            commit_sha="hash1",
            file_path="src/main.py",
            function_name="foo",
            verification_type=VerificationType.FORMAL,
            result=result,
            formula=formula,
            formula_hash=hashlib.sha256(formula.encode()).hexdigest(),
            counterexample=None,
            proof_time_ms=10.0,
            metadata=metadata,
        )

    def test_serialize_proof(self):
        """Serializes proof to dict."""
        proof = self._make_proof()
        serialized = ProofSerializer.serialize_proof(proof)
        assert isinstance(serialized, dict)
        assert serialized["proof_id"] == "proof-1"
        assert serialized["commit_sha"] == "hash1"

    def test_deserialize_proof(self):
        """Deserializes dict to proof."""
        proof = self._make_proof(proof_id="proof-2")
        data = ProofSerializer.serialize_proof(proof)
        restored = ProofSerializer.deserialize_proof(data)
        assert restored.proof_id == "proof-2"
        assert restored.verification_type == VerificationType.FORMAL

    def test_round_trip(self):
        """Serialize then deserialize preserves data."""
        original = self._make_proof(proof_id="proof-rt", result="proven")
        serialized = ProofSerializer.serialize_proof(original)
        restored = ProofSerializer.deserialize_proof(serialized)
        assert restored.proof_id == original.proof_id
        assert restored.commit_sha == original.commit_sha
        assert restored.result == original.result

    def test_compress_decompress(self):
        """Compresses and decompresses proof data."""
        proof = self._make_proof()
        data = ProofSerializer.serialize_proof(proof)
        compressed = ProofCompressor.compress(data)
        assert isinstance(compressed, str)

        restored_data = ProofCompressor.decompress(compressed)
        assert restored_data["proof_id"] == "proof-1"


class TestProofCarryingPRManager:
    """Tests for ProofCarryingPRManager."""

    def test_create_manager(self):
        """Can create manager with signing key."""
        manager = ProofCarryingPRManager(signing_key="test-secret-key-123")
        assert manager is not None

    def test_create_proof(self):
        """Creates a new verification proof."""
        manager = ProofCarryingPRManager(signing_key="secret")

        proof = manager.create_proof(
            commit_sha="abc123",
            file_path="src/main.py",
            verification_result=_make_verification_result(),
        )

        assert proof.commit_sha == "abc123"
        assert proof.verification_type == VerificationType.FORMAL
        assert proof.proof_id is not None
        assert proof.signature is not None

    def test_create_proof_proven(self):
        """Proof result is 'proven' when satisfiable is False."""
        manager = ProofCarryingPRManager(signing_key="secret")
        proof = manager.create_proof(
            commit_sha="abc",
            file_path="f.py",
            verification_result=_make_verification_result(satisfiable=False),
        )
        assert proof.result == "proven"

    def test_create_proof_counterexample(self):
        """Proof result is 'counterexample' when satisfiable is True."""
        manager = ProofCarryingPRManager(signing_key="secret")
        proof = manager.create_proof(
            commit_sha="abc",
            file_path="f.py",
            verification_result=_make_verification_result(
                satisfiable=True,
                counterexample={"x": 5},
            ),
        )
        assert proof.result == "counterexample"

    def test_create_attestation(self):
        """Creates a signed attestation for a PR."""
        manager = ProofCarryingPRManager(signing_key="my-secret-key")

        proof = manager.create_proof(
            commit_sha="xyz789",
            file_path="src/lib.py",
            verification_result=_make_verification_result(),
        )

        attestation = manager.create_attestation(
            pr_number=1,
            repo_full_name="org/repo",
            head_sha="xyz789",
            proofs=[proof],
        )

        assert attestation.proofs == [proof]
        assert attestation.signature is not None
        assert len(attestation.signature) > 0

    def test_verify_attestation_valid(self):
        """Verifies a valid attestation."""
        manager = ProofCarryingPRManager(signing_key="verification-secret")

        proof = manager.create_proof(
            commit_sha="validhash",
            file_path="src/main.py",
            verification_result=_make_verification_result(),
        )
        attestation = manager.create_attestation(
            pr_number=1,
            repo_full_name="org/repo",
            head_sha="validhash",
            proofs=[proof],
        )

        result = manager.verify_attestation(attestation)
        assert result.valid is True
        assert result.status == ProofStatus.VALID

    def test_verify_attestation_invalid_signature(self):
        """Rejects attestation with invalid signature."""
        manager = ProofCarryingPRManager(signing_key="secret1")

        proof = manager.create_proof(
            commit_sha="hash",
            file_path="src/main.py",
            verification_result=_make_verification_result(),
        )
        attestation = manager.create_attestation(
            pr_number=1,
            repo_full_name="org/repo",
            head_sha="hash",
            proofs=[proof],
        )

        # Tamper with signature
        attestation.signature = "tampered-signature"

        result = manager.verify_attestation(attestation)
        assert result.valid is False
        assert result.status == ProofStatus.INVALID

    def test_verify_attestation_wrong_key(self):
        """Rejects attestation signed with different key."""
        manager1 = ProofCarryingPRManager(signing_key="key1")
        manager2 = ProofCarryingPRManager(signing_key="key2")

        proof = manager1.create_proof(
            commit_sha="hash",
            file_path="src/main.py",
            verification_result=_make_verification_result(),
        )
        attestation = manager1.create_attestation(
            pr_number=1,
            repo_full_name="org/repo",
            head_sha="hash",
            proofs=[proof],
        )

        # Different manager should not verify
        result = manager2.verify_attestation(attestation)
        assert result.valid is False

    def test_verify_attestation_commit_mismatch(self):
        """Rejects attestation when expected commit doesn't match."""
        manager = ProofCarryingPRManager(signing_key="secret")

        proof = manager.create_proof(
            commit_sha="commit1",
            file_path="src/main.py",
            verification_result=_make_verification_result(),
        )
        attestation = manager.create_attestation(
            pr_number=1,
            repo_full_name="org/repo",
            head_sha="commit1",
            proofs=[proof],
        )

        result = manager.verify_attestation(attestation, expected_commit="commit2")
        assert result.valid is False
        assert result.status == ProofStatus.INVALID

    def test_serialize_for_github_round_trip(self):
        """Serialize and deserialize for GitHub preserves data."""
        manager = ProofCarryingPRManager(signing_key="secret")

        proof = manager.create_proof(
            commit_sha="hash",
            file_path="src/main.py",
            verification_result=_make_verification_result(),
        )
        attestation = manager.create_attestation(
            pr_number=42,
            repo_full_name="org/repo",
            head_sha="hash",
            proofs=[proof],
        )

        compressed = manager.serialize_for_github(attestation)
        restored = manager.deserialize_from_github(compressed)

        assert restored.attestation_id == attestation.attestation_id
        assert restored.pr_number == 42
        assert len(restored.proofs) == 1

    def test_generate_badge_url_verified(self):
        """Generates badge URL for fully verified PR."""
        manager = ProofCarryingPRManager(signing_key="secret")

        proof = manager.create_proof(
            commit_sha="hash",
            file_path="src/main.py",
            verification_result=_make_verification_result(),
        )
        attestation = manager.create_attestation(
            pr_number=1,
            repo_full_name="org/repo",
            head_sha="hash123",
            proofs=[proof],
        )

        url = manager.generate_badge_url(attestation)
        assert "verified" in url
        assert "org/repo" in url


class TestProofCarryingEdgeCases:
    """Edge case tests for proof-carrying functionality."""

    def test_special_characters_in_formula(self):
        """Handles special characters in formula."""
        manager = ProofCarryingPRManager(signing_key="secret")

        proof = manager.create_proof(
            commit_sha="hash",
            file_path="src/main.py",
            verification_result=_make_verification_result(
                formula="(assert (= x \"hello 'world'\"))",
            ),
        )

        attestation = manager.create_attestation(
            pr_number=1,
            repo_full_name="org/repo",
            head_sha="hash",
            proofs=[proof],
        )
        result = manager.verify_attestation(attestation)
        assert result.valid is True

    def test_multiple_proofs_in_attestation(self):
        """Handles multiple proofs in a single attestation."""
        manager = ProofCarryingPRManager(signing_key="secret")

        proofs = []
        for i in range(5):
            proof = manager.create_proof(
                commit_sha="hash",
                file_path=f"src/file{i}.py",
                verification_result=_make_verification_result(),
            )
            proofs.append(proof)

        attestation = manager.create_attestation(
            pr_number=1,
            repo_full_name="org/repo",
            head_sha="hash",
            proofs=proofs,
        )

        result = manager.verify_attestation(attestation)
        assert result.valid is True
        assert attestation.summary["total_proofs"] == 5
