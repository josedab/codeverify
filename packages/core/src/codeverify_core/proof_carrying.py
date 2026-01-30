"""Proof-Carrying PRs - Cryptographic attestations for verification proofs.

Embeds verification proofs as cryptographic attestations in PR metadata,
enabling downstream verification without re-running the analysis.
"""

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

logger = structlog.get_logger()


class ProofStatus(str, Enum):
    """Status of a verification proof."""
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    UNKNOWN = "unknown"


class VerificationType(str, Enum):
    """Type of verification that produced the proof."""
    FORMAL = "formal"
    SEMANTIC = "semantic"
    SECURITY = "security"
    COMPOSITE = "composite"


@dataclass
class ProofMetadata:
    """Metadata about a proof."""
    proof_id: str
    timestamp: datetime
    verification_type: VerificationType
    verifier_version: str
    solver_version: str | None = None
    timeout_ms: int | None = None
    config_hash: str | None = None


@dataclass
class VerificationProof:
    """A verification proof that can be embedded in PRs."""
    proof_id: str
    commit_sha: str
    file_path: str
    function_name: str | None
    verification_type: VerificationType
    result: str  # "proven", "counterexample", "timeout"
    formula: str  # The SMT-LIB formula verified
    formula_hash: str
    counterexample: dict[str, Any] | None
    proof_time_ms: float
    metadata: ProofMetadata
    signature: str | None = None


@dataclass
class ProofAttestation:
    """A signed attestation of verification results."""
    attestation_id: str
    pr_number: int
    repo_full_name: str
    head_sha: str
    base_sha: str | None
    proofs: list[VerificationProof]
    summary: dict[str, Any]
    created_at: datetime
    expires_at: datetime
    signature: str
    certificate_chain: list[str] = field(default_factory=list)


@dataclass
class AttestationVerificationResult:
    """Result of verifying an attestation."""
    valid: bool
    status: ProofStatus
    message: str
    attestation: ProofAttestation | None = None
    verification_time_ms: float = 0


class ProofSerializer:
    """Serializes proofs to portable format."""

    @staticmethod
    def serialize_proof(proof: VerificationProof) -> dict[str, Any]:
        """Serialize a proof to dictionary."""
        return {
            "proof_id": proof.proof_id,
            "commit_sha": proof.commit_sha,
            "file_path": proof.file_path,
            "function_name": proof.function_name,
            "verification_type": proof.verification_type.value,
            "result": proof.result,
            "formula": proof.formula,
            "formula_hash": proof.formula_hash,
            "counterexample": proof.counterexample,
            "proof_time_ms": proof.proof_time_ms,
            "metadata": {
                "proof_id": proof.metadata.proof_id,
                "timestamp": proof.metadata.timestamp.isoformat(),
                "verification_type": proof.metadata.verification_type.value,
                "verifier_version": proof.metadata.verifier_version,
                "solver_version": proof.metadata.solver_version,
                "timeout_ms": proof.metadata.timeout_ms,
                "config_hash": proof.metadata.config_hash,
            },
            "signature": proof.signature,
        }

    @staticmethod
    def deserialize_proof(data: dict[str, Any]) -> VerificationProof:
        """Deserialize a proof from dictionary."""
        metadata_data = data.get("metadata", {})
        metadata = ProofMetadata(
            proof_id=metadata_data.get("proof_id", ""),
            timestamp=datetime.fromisoformat(metadata_data.get("timestamp", datetime.utcnow().isoformat())),
            verification_type=VerificationType(metadata_data.get("verification_type", "formal")),
            verifier_version=metadata_data.get("verifier_version", "unknown"),
            solver_version=metadata_data.get("solver_version"),
            timeout_ms=metadata_data.get("timeout_ms"),
            config_hash=metadata_data.get("config_hash"),
        )
        
        return VerificationProof(
            proof_id=data.get("proof_id", ""),
            commit_sha=data.get("commit_sha", ""),
            file_path=data.get("file_path", ""),
            function_name=data.get("function_name"),
            verification_type=VerificationType(data.get("verification_type", "formal")),
            result=data.get("result", "unknown"),
            formula=data.get("formula", ""),
            formula_hash=data.get("formula_hash", ""),
            counterexample=data.get("counterexample"),
            proof_time_ms=data.get("proof_time_ms", 0),
            metadata=metadata,
            signature=data.get("signature"),
        )

    @staticmethod
    def serialize_attestation(attestation: ProofAttestation) -> dict[str, Any]:
        """Serialize an attestation to dictionary."""
        return {
            "attestation_id": attestation.attestation_id,
            "pr_number": attestation.pr_number,
            "repo_full_name": attestation.repo_full_name,
            "head_sha": attestation.head_sha,
            "base_sha": attestation.base_sha,
            "proofs": [ProofSerializer.serialize_proof(p) for p in attestation.proofs],
            "summary": attestation.summary,
            "created_at": attestation.created_at.isoformat(),
            "expires_at": attestation.expires_at.isoformat(),
            "signature": attestation.signature,
            "certificate_chain": attestation.certificate_chain,
        }

    @staticmethod
    def deserialize_attestation(data: dict[str, Any]) -> ProofAttestation:
        """Deserialize an attestation from dictionary."""
        proofs = [ProofSerializer.deserialize_proof(p) for p in data.get("proofs", [])]
        
        return ProofAttestation(
            attestation_id=data.get("attestation_id", ""),
            pr_number=data.get("pr_number", 0),
            repo_full_name=data.get("repo_full_name", ""),
            head_sha=data.get("head_sha", ""),
            base_sha=data.get("base_sha"),
            proofs=proofs,
            summary=data.get("summary", {}),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat())),
            expires_at=datetime.fromisoformat(data.get("expires_at", datetime.utcnow().isoformat())),
            signature=data.get("signature", ""),
            certificate_chain=data.get("certificate_chain", []),
        )


class ProofSigner:
    """Signs and verifies proofs using HMAC-SHA256."""

    def __init__(self, signing_key: str | bytes) -> None:
        """Initialize with signing key."""
        if isinstance(signing_key, str):
            signing_key = signing_key.encode('utf-8')
        self._key = signing_key

    def sign_proof(self, proof: VerificationProof) -> str:
        """Sign a verification proof."""
        payload = self._create_proof_payload(proof)
        signature = hmac.new(
            self._key,
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def verify_proof_signature(self, proof: VerificationProof) -> bool:
        """Verify a proof's signature."""
        if not proof.signature:
            return False
        
        expected = self.sign_proof(proof)
        return hmac.compare_digest(expected, proof.signature)

    def sign_attestation(self, attestation: ProofAttestation) -> str:
        """Sign an attestation."""
        payload = self._create_attestation_payload(attestation)
        signature = hmac.new(
            self._key,
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def verify_attestation_signature(self, attestation: ProofAttestation) -> bool:
        """Verify an attestation's signature."""
        if not attestation.signature:
            return False
        
        # Temporarily remove signature for verification
        original_sig = attestation.signature
        attestation.signature = ""
        expected = self.sign_attestation(attestation)
        attestation.signature = original_sig
        
        return hmac.compare_digest(expected, original_sig)

    def _create_proof_payload(self, proof: VerificationProof) -> str:
        """Create canonical payload for signing."""
        data = {
            "proof_id": proof.proof_id,
            "commit_sha": proof.commit_sha,
            "file_path": proof.file_path,
            "formula_hash": proof.formula_hash,
            "result": proof.result,
            "timestamp": proof.metadata.timestamp.isoformat(),
        }
        return json.dumps(data, sort_keys=True)

    def _create_attestation_payload(self, attestation: ProofAttestation) -> str:
        """Create canonical payload for signing."""
        data = {
            "attestation_id": attestation.attestation_id,
            "pr_number": attestation.pr_number,
            "repo_full_name": attestation.repo_full_name,
            "head_sha": attestation.head_sha,
            "created_at": attestation.created_at.isoformat(),
            "proof_count": len(attestation.proofs),
            "proof_hashes": [p.formula_hash for p in attestation.proofs],
        }
        return json.dumps(data, sort_keys=True)


class ProofCompressor:
    """Compresses proofs for storage efficiency."""

    @staticmethod
    def compress(data: dict[str, Any]) -> str:
        """Compress proof data to base64."""
        import gzip
        json_str = json.dumps(data, separators=(',', ':'))
        compressed = gzip.compress(json_str.encode('utf-8'))
        return base64.b64encode(compressed).decode('ascii')

    @staticmethod
    def decompress(compressed: str) -> dict[str, Any]:
        """Decompress base64 proof data."""
        import gzip
        raw = base64.b64decode(compressed)
        decompressed = gzip.decompress(raw)
        return json.loads(decompressed.decode('utf-8'))


class ProofCarryingPRManager:
    """
    Manages proof-carrying PRs.
    
    Creates, stores, and verifies cryptographic attestations
    of verification results for pull requests.
    """

    def __init__(
        self,
        signing_key: str,
        verifier_version: str = "1.0.0",
        attestation_ttl_hours: int = 168,  # 1 week default
    ) -> None:
        """Initialize the manager."""
        self._signer = ProofSigner(signing_key)
        self._verifier_version = verifier_version
        self._attestation_ttl_hours = attestation_ttl_hours

    def create_proof(
        self,
        commit_sha: str,
        file_path: str,
        verification_result: dict[str, Any],
        function_name: str | None = None,
        config_hash: str | None = None,
    ) -> VerificationProof:
        """Create a signed verification proof."""
        proof_id = str(uuid4())
        now = datetime.utcnow()
        
        formula = verification_result.get("formula", "")
        formula_hash = hashlib.sha256(formula.encode('utf-8')).hexdigest()
        
        # Determine result type
        if verification_result.get("satisfiable") is False:
            result = "proven"
        elif verification_result.get("satisfiable") is True:
            result = "counterexample"
        else:
            result = "timeout"
        
        metadata = ProofMetadata(
            proof_id=proof_id,
            timestamp=now,
            verification_type=VerificationType.FORMAL,
            verifier_version=self._verifier_version,
            solver_version=verification_result.get("solver_version", "z3-4.12"),
            timeout_ms=verification_result.get("timeout_ms"),
            config_hash=config_hash,
        )
        
        proof = VerificationProof(
            proof_id=proof_id,
            commit_sha=commit_sha,
            file_path=file_path,
            function_name=function_name,
            verification_type=VerificationType.FORMAL,
            result=result,
            formula=formula,
            formula_hash=formula_hash,
            counterexample=verification_result.get("counterexample"),
            proof_time_ms=verification_result.get("proof_time_ms", 0),
            metadata=metadata,
        )
        
        # Sign the proof
        proof.signature = self._signer.sign_proof(proof)
        
        logger.info(
            "Created verification proof",
            proof_id=proof_id,
            result=result,
            file_path=file_path,
        )
        
        return proof

    def create_attestation(
        self,
        pr_number: int,
        repo_full_name: str,
        head_sha: str,
        proofs: list[VerificationProof],
        base_sha: str | None = None,
    ) -> ProofAttestation:
        """Create a signed attestation for a PR."""
        attestation_id = str(uuid4())
        now = datetime.utcnow()
        expires = datetime.utcnow()
        expires = datetime(
            expires.year,
            expires.month,
            expires.day,
            expires.hour + self._attestation_ttl_hours,
            expires.minute,
            expires.second,
        )
        
        # Generate summary
        summary = self._generate_summary(proofs)
        
        attestation = ProofAttestation(
            attestation_id=attestation_id,
            pr_number=pr_number,
            repo_full_name=repo_full_name,
            head_sha=head_sha,
            base_sha=base_sha,
            proofs=proofs,
            summary=summary,
            created_at=now,
            expires_at=expires,
            signature="",  # Will be set after signing
        )
        
        # Sign the attestation
        attestation.signature = self._signer.sign_attestation(attestation)
        
        logger.info(
            "Created PR attestation",
            attestation_id=attestation_id,
            pr_number=pr_number,
            proof_count=len(proofs),
        )
        
        return attestation

    def _generate_summary(self, proofs: list[VerificationProof]) -> dict[str, Any]:
        """Generate summary statistics from proofs."""
        proven = sum(1 for p in proofs if p.result == "proven")
        counterexamples = sum(1 for p in proofs if p.result == "counterexample")
        timeouts = sum(1 for p in proofs if p.result == "timeout")
        total_time = sum(p.proof_time_ms for p in proofs)
        
        return {
            "total_proofs": len(proofs),
            "proven": proven,
            "counterexamples": counterexamples,
            "timeouts": timeouts,
            "total_proof_time_ms": total_time,
            "files_verified": len(set(p.file_path for p in proofs)),
            "all_verified": counterexamples == 0 and timeouts == 0,
        }

    def verify_attestation(
        self,
        attestation: ProofAttestation,
        expected_commit: str | None = None,
    ) -> AttestationVerificationResult:
        """Verify an attestation's validity."""
        start_time = time.time()
        
        # Check signature
        if not self._signer.verify_attestation_signature(attestation):
            return AttestationVerificationResult(
                valid=False,
                status=ProofStatus.INVALID,
                message="Invalid attestation signature",
                verification_time_ms=(time.time() - start_time) * 1000,
            )
        
        # Check expiration
        if datetime.utcnow() > attestation.expires_at:
            return AttestationVerificationResult(
                valid=False,
                status=ProofStatus.EXPIRED,
                message="Attestation has expired",
                attestation=attestation,
                verification_time_ms=(time.time() - start_time) * 1000,
            )
        
        # Check commit if provided
        if expected_commit and attestation.head_sha != expected_commit:
            return AttestationVerificationResult(
                valid=False,
                status=ProofStatus.INVALID,
                message=f"Commit mismatch: expected {expected_commit}, got {attestation.head_sha}",
                attestation=attestation,
                verification_time_ms=(time.time() - start_time) * 1000,
            )
        
        # Verify individual proof signatures
        for proof in attestation.proofs:
            if not self._signer.verify_proof_signature(proof):
                return AttestationVerificationResult(
                    valid=False,
                    status=ProofStatus.INVALID,
                    message=f"Invalid proof signature for {proof.proof_id}",
                    attestation=attestation,
                    verification_time_ms=(time.time() - start_time) * 1000,
                )
        
        return AttestationVerificationResult(
            valid=True,
            status=ProofStatus.VALID,
            message="Attestation verified successfully",
            attestation=attestation,
            verification_time_ms=(time.time() - start_time) * 1000,
        )

    def serialize_for_github(self, attestation: ProofAttestation) -> str:
        """Serialize attestation for GitHub Check metadata."""
        data = ProofSerializer.serialize_attestation(attestation)
        return ProofCompressor.compress(data)

    def deserialize_from_github(self, compressed: str) -> ProofAttestation:
        """Deserialize attestation from GitHub Check metadata."""
        data = ProofCompressor.decompress(compressed)
        return ProofSerializer.deserialize_attestation(data)

    def generate_badge_url(
        self,
        attestation: ProofAttestation,
        base_url: str = "https://codeverify.dev",
    ) -> str:
        """Generate a verification badge URL."""
        summary = attestation.summary
        
        if summary.get("all_verified"):
            status = "verified"
            color = "brightgreen"
        elif summary.get("counterexamples", 0) > 0:
            status = f"{summary['counterexamples']}_issues"
            color = "red"
        else:
            status = "partial"
            color = "yellow"
        
        return (
            f"{base_url}/badge/{attestation.repo_full_name}"
            f"/{attestation.head_sha[:7]}"
            f"?status={status}&color={color}"
        )


class ProofArtifactStore:
    """
    Storage for proof artifacts.
    
    Provides a searchable library of verification proofs
    that can be reused across projects.
    """

    def __init__(self) -> None:
        """Initialize the store."""
        self._proofs: dict[str, VerificationProof] = {}
        self._by_formula: dict[str, list[str]] = {}  # formula_hash -> proof_ids
        self._by_file: dict[str, list[str]] = {}  # file_path -> proof_ids

    def store(self, proof: VerificationProof) -> None:
        """Store a proof."""
        self._proofs[proof.proof_id] = proof
        
        # Index by formula hash
        if proof.formula_hash not in self._by_formula:
            self._by_formula[proof.formula_hash] = []
        self._by_formula[proof.formula_hash].append(proof.proof_id)
        
        # Index by file path
        if proof.file_path not in self._by_file:
            self._by_file[proof.file_path] = []
        self._by_file[proof.file_path].append(proof.proof_id)

    def get(self, proof_id: str) -> VerificationProof | None:
        """Get a proof by ID."""
        return self._proofs.get(proof_id)

    def find_by_formula(self, formula_hash: str) -> list[VerificationProof]:
        """Find proofs by formula hash."""
        proof_ids = self._by_formula.get(formula_hash, [])
        return [self._proofs[pid] for pid in proof_ids if pid in self._proofs]

    def find_by_file(self, file_path: str) -> list[VerificationProof]:
        """Find proofs for a file."""
        proof_ids = self._by_file.get(file_path, [])
        return [self._proofs[pid] for pid in proof_ids if pid in self._proofs]

    def find_reusable_proof(
        self,
        formula: str,
        commit_sha: str | None = None,
    ) -> VerificationProof | None:
        """
        Find a reusable proof for a formula.
        
        Returns a valid proof if the formula has been verified before
        with the same result.
        """
        formula_hash = hashlib.sha256(formula.encode('utf-8')).hexdigest()
        proofs = self.find_by_formula(formula_hash)
        
        # Return most recent proven proof
        proven = [p for p in proofs if p.result == "proven"]
        if proven:
            return max(proven, key=lambda p: p.metadata.timestamp)
        
        return None

    def get_statistics(self) -> dict[str, Any]:
        """Get storage statistics."""
        total = len(self._proofs)
        proven = sum(1 for p in self._proofs.values() if p.result == "proven")
        counterexamples = sum(1 for p in self._proofs.values() if p.result == "counterexample")
        
        return {
            "total_proofs": total,
            "proven": proven,
            "counterexamples": counterexamples,
            "unique_formulas": len(self._by_formula),
            "files_covered": len(self._by_file),
        }

    def export(self) -> list[dict[str, Any]]:
        """Export all proofs."""
        return [ProofSerializer.serialize_proof(p) for p in self._proofs.values()]

    def import_proofs(self, proofs_data: list[dict[str, Any]]) -> int:
        """Import proofs from exported data."""
        count = 0
        for data in proofs_data:
            try:
                proof = ProofSerializer.deserialize_proof(data)
                self.store(proof)
                count += 1
            except Exception as e:
                logger.warning("Failed to import proof", error=str(e))
        return count
