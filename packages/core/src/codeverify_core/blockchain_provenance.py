"""Blockchain-Verified Code Provenance.

Immutable audit trail of code verification on blockchain with attestation
records, NFT-style verification badges, and tamper-proof supply chain security.

Features:
- Attestation records with cryptographic hashes
- Blockchain transaction simulation (chain-agnostic abstraction)
- NFT-style verification badges for verified repos/releases
- IPFS-like content addressing for proof storage
- Multi-chain support (Ethereum, Polygon, private chains)
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ChainType(str, Enum):
    """Supported blockchain types."""

    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    HYPERLEDGER = "hyperledger"
    LOCAL = "local"


class AttestationStatus(str, Enum):
    """Status of a blockchain attestation."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    REVOKED = "revoked"


class BadgeLevel(str, Enum):
    """Verification badge levels (NFT tiers)."""

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"
    DIAMOND = "diamond"


@dataclass
class ContentAddress:
    """Content-addressed storage reference (IPFS-like)."""

    cid: str = ""
    size_bytes: int = 0
    content_hash: str = ""

    @staticmethod
    def from_content(data: str) -> ContentAddress:
        """Create content address from data."""
        content_hash = hashlib.sha256(data.encode()).hexdigest()
        cid = f"Qm{hashlib.sha256(content_hash.encode()).hexdigest()[:44]}"
        return ContentAddress(
            cid=cid,
            size_bytes=len(data.encode()),
            content_hash=content_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"cid": self.cid, "size_bytes": self.size_bytes, "hash": self.content_hash}


@dataclass
class BlockchainAttestation:
    """An attestation record stored on blockchain."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    repo_id: str = ""
    commit_sha: str = ""
    verification_hash: str = ""
    proof_cid: ContentAddress | None = None
    chain: ChainType = ChainType.LOCAL
    tx_hash: str = ""
    block_number: int = 0
    status: AttestationStatus = AttestationStatus.PENDING
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    gas_used: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "commit_sha": self.commit_sha,
            "verification_hash": self.verification_hash[:16] + "...",
            "chain": self.chain.value,
            "tx_hash": self.tx_hash[:16] + "..." if self.tx_hash else "",
            "block_number": self.block_number,
            "status": self.status.value,
            "gas_used": self.gas_used,
        }


@dataclass
class VerificationBadge:
    """NFT-style verification badge for a repository or release."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    repo_id: str = ""
    release_tag: str = ""
    level: BadgeLevel = BadgeLevel.BRONZE
    attestation_ids: list[str] = field(default_factory=list)
    trust_score: float = 0.0
    verification_coverage: float = 0.0
    issued_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    token_id: int = 0
    metadata_uri: str = ""

    @staticmethod
    def level_from_score(score: float) -> BadgeLevel:
        """Determine badge level from trust score."""
        if score >= 95:
            return BadgeLevel.DIAMOND
        elif score >= 85:
            return BadgeLevel.PLATINUM
        elif score >= 70:
            return BadgeLevel.GOLD
        elif score >= 50:
            return BadgeLevel.SILVER
        return BadgeLevel.BRONZE

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "release_tag": self.release_tag,
            "level": self.level.value,
            "trust_score": round(self.trust_score, 2),
            "verification_coverage": round(self.verification_coverage, 4),
            "attestation_count": len(self.attestation_ids),
            "token_id": self.token_id,
        }


class LocalBlockchain:
    """Simulated local blockchain for development and testing."""

    def __init__(self) -> None:
        self._blocks: list[dict[str, Any]] = []
        self._block_number: int = 0

    def submit_transaction(self, data: dict[str, Any]) -> tuple[str, int]:
        """Submit a transaction to the local chain. Returns (tx_hash, block_number)."""
        self._block_number += 1
        tx_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()
        self._blocks.append({
            "block_number": self._block_number,
            "tx_hash": tx_hash,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        return tx_hash, self._block_number

    def verify_transaction(self, tx_hash: str) -> dict[str, Any] | None:
        """Verify a transaction exists on chain."""
        for block in self._blocks:
            if block["tx_hash"] == tx_hash:
                return block
        return None

    @property
    def block_height(self) -> int:
        return self._block_number


class BlockchainProvenanceEngine:
    """Engine for blockchain-verified code provenance."""

    def __init__(self, chain: ChainType = ChainType.LOCAL) -> None:
        self._chain_type = chain
        self._local_chain = LocalBlockchain()
        self._attestations: dict[str, BlockchainAttestation] = {}
        self._badges: dict[str, VerificationBadge] = {}
        self._token_counter: int = 0

    def create_attestation(
        self,
        repo_id: str,
        commit_sha: str,
        proof_data: str,
        metadata: dict[str, Any] | None = None,
    ) -> BlockchainAttestation:
        """Create and submit a verification attestation to the blockchain."""
        verification_hash = hashlib.sha256(proof_data.encode()).hexdigest()
        proof_cid = ContentAddress.from_content(proof_data)

        attestation = BlockchainAttestation(
            repo_id=repo_id,
            commit_sha=commit_sha,
            verification_hash=verification_hash,
            proof_cid=proof_cid,
            chain=self._chain_type,
            metadata=metadata or {},
        )

        tx_data = {
            "type": "verification_attestation",
            "repo_id": repo_id,
            "commit_sha": commit_sha,
            "verification_hash": verification_hash,
            "proof_cid": proof_cid.cid,
        }
        tx_hash, block_num = self._local_chain.submit_transaction(tx_data)
        attestation.tx_hash = tx_hash
        attestation.block_number = block_num
        attestation.status = AttestationStatus.CONFIRMED
        attestation.gas_used = len(json.dumps(tx_data)) * 21  # Simulated gas

        self._attestations[attestation.id] = attestation
        logger.info(
            "attestation_created",
            attestation_id=attestation.id,
            tx_hash=tx_hash[:16],
            block=block_num,
        )
        return attestation

    def verify_attestation(self, attestation_id: str) -> bool:
        """Verify an attestation exists on chain and is valid."""
        att = self._attestations.get(attestation_id)
        if not att or not att.tx_hash:
            return False
        block = self._local_chain.verify_transaction(att.tx_hash)
        return block is not None

    def issue_badge(
        self,
        repo_id: str,
        release_tag: str,
        trust_score: float,
        verification_coverage: float = 0.0,
        attestation_ids: list[str] | None = None,
    ) -> VerificationBadge:
        """Issue an NFT verification badge for a release."""
        self._token_counter += 1
        badge = VerificationBadge(
            repo_id=repo_id,
            release_tag=release_tag,
            level=VerificationBadge.level_from_score(trust_score),
            attestation_ids=attestation_ids or [],
            trust_score=trust_score,
            verification_coverage=verification_coverage,
            token_id=self._token_counter,
            metadata_uri=f"ipfs://badge/{self._token_counter}",
        )

        tx_data = {
            "type": "badge_mint",
            "token_id": self._token_counter,
            "repo_id": repo_id,
            "level": badge.level.value,
        }
        self._local_chain.submit_transaction(tx_data)
        self._badges[badge.id] = badge

        logger.info(
            "badge_issued",
            badge_id=badge.id,
            level=badge.level.value,
            token_id=badge.token_id,
        )
        return badge

    def revoke_attestation(self, attestation_id: str) -> bool:
        """Revoke an attestation."""
        att = self._attestations.get(attestation_id)
        if not att:
            return False
        att.status = AttestationStatus.REVOKED
        self._local_chain.submit_transaction({
            "type": "revocation",
            "attestation_id": attestation_id,
        })
        return True

    def get_repo_attestations(self, repo_id: str) -> list[BlockchainAttestation]:
        """Get all attestations for a repository."""
        return [
            a for a in self._attestations.values()
            if a.repo_id == repo_id and a.status == AttestationStatus.CONFIRMED
        ]

    def get_provenance_summary(self) -> dict[str, Any]:
        """Get summary of all provenance records."""
        confirmed = [a for a in self._attestations.values() if a.status == AttestationStatus.CONFIRMED]
        return {
            "total_attestations": len(self._attestations),
            "confirmed_attestations": len(confirmed),
            "total_badges": len(self._badges),
            "chain": self._chain_type.value,
            "block_height": self._local_chain.block_height,
            "total_gas_used": sum(a.gas_used for a in confirmed),
        }


_default_engine: BlockchainProvenanceEngine | None = None


def get_blockchain_provenance() -> BlockchainProvenanceEngine:
    """Get the singleton blockchain provenance engine."""
    global _default_engine
    if _default_engine is None:
        _default_engine = BlockchainProvenanceEngine()
    return _default_engine


def reset_blockchain_provenance() -> None:
    """Reset the singleton (for testing)."""
    global _default_engine
    _default_engine = None
