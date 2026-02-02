"""Tests for Compliance Evidence Vault."""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from codeverify_core.evidence_vault import (
    AccessToken,
    AuditAction,
    ComplianceFramework,
    ComplianceReportGenerator,
    ControlStatus,
    EvidenceMetadata,
    EvidenceType,
    EvidenceVault,
    FileSystemStorage,
    InMemoryStorage,
    StoredEvidence,
    get_evidence_vault,
    reset_evidence_vault,
)


class TestStoredEvidence:
    """Tests for StoredEvidence."""

    def test_compute_hash(self) -> None:
        """Test content hash computation."""
        evidence = StoredEvidence()
        content = b"test content"
        hash_value = evidence.compute_hash(content)
        
        assert hash_value is not None
        assert len(hash_value) == 64  # SHA-256 hex digest

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        metadata = EvidenceMetadata(
            evidence_type=EvidenceType.VERIFICATION_PROOF,
            source_system="codeverify",
            source_id="pr-123",
            created_by="user@example.com",
        )
        evidence = StoredEvidence(metadata=metadata)
        
        data = evidence.to_dict()
        
        assert "id" in data
        assert data["metadata"]["evidence_type"] == "verification_proof"
        assert data["metadata"]["source_system"] == "codeverify"


class TestInMemoryStorage:
    """Tests for InMemoryStorage."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self) -> None:
        """Test storing and retrieving evidence."""
        storage = InMemoryStorage()
        content = b"test evidence content"
        
        success = await storage.store("evidence-1", content)
        assert success
        
        retrieved = await storage.retrieve("evidence-1")
        assert retrieved == content

    @pytest.mark.asyncio
    async def test_exists(self) -> None:
        """Test existence check."""
        storage = InMemoryStorage()
        
        assert not await storage.exists("nonexistent")
        
        await storage.store("evidence-1", b"content")
        assert await storage.exists("evidence-1")

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        """Test deletion."""
        storage = InMemoryStorage()
        await storage.store("evidence-1", b"content")
        
        deleted = await storage.delete("evidence-1")
        assert deleted
        assert not await storage.exists("evidence-1")


class TestFileSystemStorage:
    """Tests for FileSystemStorage."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self) -> None:
        """Test file-based storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = FileSystemStorage(Path(tmpdir))
            content = b"test evidence content"
            
            success = await storage.store("evidence-1", content)
            assert success
            
            retrieved = await storage.retrieve("evidence-1")
            assert retrieved == content


class TestEvidenceVault:
    """Tests for EvidenceVault."""

    @pytest.mark.asyncio
    async def test_store_evidence(self) -> None:
        """Test storing evidence with integrity."""
        storage = InMemoryStorage()
        vault = EvidenceVault(storage)
        
        metadata = EvidenceMetadata(
            evidence_type=EvidenceType.VERIFICATION_PROOF,
            source_system="codeverify",
            source_id="pr-123",
            created_by="user@example.com",
            frameworks=[ComplianceFramework.SOC2],
        )
        content = b'{"findings": [], "verified": true}'
        
        evidence = await vault.store_evidence(content, metadata)
        
        assert evidence.id is not None
        assert evidence.content_hash is not None
        assert evidence.signature is not None
        assert evidence.chain_hash is not None
        assert evidence.sequence_number == 1

    @pytest.mark.asyncio
    async def test_retrieve_evidence_with_verification(self) -> None:
        """Test retrieving evidence with integrity verification."""
        storage = InMemoryStorage()
        vault = EvidenceVault(storage)
        
        metadata = EvidenceMetadata(
            evidence_type=EvidenceType.SECURITY_SCAN,
            source_system="scanner",
            source_id="scan-456",
            created_by="system",
        )
        content = b"scan results"
        
        stored = await vault.store_evidence(content, metadata)
        result = await vault.retrieve_evidence(stored.id)
        
        assert result is not None
        evidence, retrieved_content = result
        assert retrieved_content == content
        assert evidence.content_hash == stored.content_hash

    @pytest.mark.asyncio
    async def test_chain_integrity(self) -> None:
        """Test that evidence chain hashes are linked."""
        storage = InMemoryStorage()
        vault = EvidenceVault(storage)
        
        metadata = EvidenceMetadata(
            evidence_type=EvidenceType.AUDIT_LOG,
            source_system="test",
            source_id="1",
            created_by="system",
        )
        
        e1 = await vault.store_evidence(b"evidence 1", metadata)
        e2 = await vault.store_evidence(b"evidence 2", metadata)
        e3 = await vault.store_evidence(b"evidence 3", metadata)
        
        assert e1.sequence_number == 1
        assert e2.sequence_number == 2
        assert e3.sequence_number == 3
        
        # Chain hashes should all be different
        assert e1.chain_hash != e2.chain_hash
        assert e2.chain_hash != e3.chain_hash

    @pytest.mark.asyncio
    async def test_seal_evidence(self) -> None:
        """Test sealing evidence."""
        storage = InMemoryStorage()
        vault = EvidenceVault(storage)
        
        metadata = EvidenceMetadata(
            evidence_type=EvidenceType.ATTESTATION,
            source_system="test",
            source_id="1",
            created_by="auditor",
        )
        
        evidence = await vault.store_evidence(b"attestation", metadata)
        assert not evidence.is_sealed
        
        sealed = await vault.seal_evidence(evidence.id)
        assert sealed
        
        result = await vault.retrieve_evidence(evidence.id)
        assert result is not None
        assert result[0].is_sealed

    def test_search_evidence(self) -> None:
        """Test searching evidence."""
        # Note: This test uses sync search, storage ops tested separately
        storage = InMemoryStorage()
        vault = EvidenceVault(storage)
        
        # Pre-populate index for search test
        e1 = StoredEvidence(
            id="e1",
            metadata=EvidenceMetadata(
                evidence_type=EvidenceType.VERIFICATION_PROOF,
                source_system="test",
                source_id="1",
                created_by="system",
                frameworks=[ComplianceFramework.SOC2],
            ),
        )
        e2 = StoredEvidence(
            id="e2",
            metadata=EvidenceMetadata(
                evidence_type=EvidenceType.SECURITY_SCAN,
                source_system="test",
                source_id="2",
                created_by="system",
                frameworks=[ComplianceFramework.HIPAA],
            ),
        )
        
        vault._evidence_index["e1"] = e1
        vault._evidence_index["e2"] = e2
        
        # Search by type
        results = vault.search_evidence(evidence_type=EvidenceType.VERIFICATION_PROOF)
        assert len(results) == 1
        assert results[0].id == "e1"
        
        # Search by framework
        results = vault.search_evidence(framework=ComplianceFramework.HIPAA)
        assert len(results) == 1
        assert results[0].id == "e2"


class TestAccessTokens:
    """Tests for access token management."""

    def test_create_access_token(self) -> None:
        """Test creating an access token."""
        storage = InMemoryStorage()
        vault = EvidenceVault(storage)
        
        raw_token, token = vault.create_access_token(
            org_id="org-123",
            auditor_email="auditor@example.com",
            auditor_name="John Auditor",
            permissions=["read", "export"],
            frameworks=[ComplianceFramework.SOC2],
            valid_days=14,
        )
        
        assert raw_token is not None
        assert len(raw_token) > 30
        assert token.auditor_email == "auditor@example.com"
        assert token.expires_at > datetime.utcnow()

    def test_validate_token(self) -> None:
        """Test validating an access token."""
        storage = InMemoryStorage()
        vault = EvidenceVault(storage)
        
        raw_token, _ = vault.create_access_token(
            org_id="org-123",
            auditor_email="auditor@example.com",
            auditor_name="John",
            permissions=["read"],
            frameworks=[],
        )
        
        validated = vault.validate_token(raw_token)
        assert validated is not None
        assert validated.auditor_email == "auditor@example.com"
        
        # Invalid token
        invalid = vault.validate_token("invalid-token")
        assert invalid is None

    def test_revoke_token(self) -> None:
        """Test revoking an access token."""
        storage = InMemoryStorage()
        vault = EvidenceVault(storage)
        
        raw_token, token = vault.create_access_token(
            org_id="org-123",
            auditor_email="auditor@example.com",
            auditor_name="John",
            permissions=["read"],
            frameworks=[],
        )
        
        revoked = vault.revoke_token(token.token_id)
        assert revoked
        
        # Token should no longer validate
        validated = vault.validate_token(raw_token)
        assert validated is None


class TestExportPackage:
    """Tests for evidence export."""

    @pytest.mark.asyncio
    async def test_export_evidence_package(self) -> None:
        """Test exporting evidence as a package."""
        storage = InMemoryStorage()
        vault = EvidenceVault(storage)
        
        # Store some evidence
        metadata = EvidenceMetadata(
            evidence_type=EvidenceType.VERIFICATION_PROOF,
            source_system="test",
            source_id="1",
            created_by="system",
        )
        
        e1 = await vault.store_evidence(b"evidence 1", metadata)
        e2 = await vault.store_evidence(b"evidence 2", metadata)
        
        # Export
        package_data, export_id = await vault.export_evidence_package(
            [e1.id, e2.id],
            actor_id="exporter",
        )
        
        assert package_data is not None
        assert len(package_data) > 0
        assert export_id is not None
        
        # Verify it's a valid zip
        import io
        import zipfile
        
        with zipfile.ZipFile(io.BytesIO(package_data), "r") as zf:
            names = zf.namelist()
            assert "manifest.json" in names
            assert "chain_of_custody.json" in names


class TestAuditLog:
    """Tests for audit logging."""

    @pytest.mark.asyncio
    async def test_audit_log_entries(self) -> None:
        """Test that actions are logged."""
        storage = InMemoryStorage()
        vault = EvidenceVault(storage)
        
        metadata = EvidenceMetadata(
            evidence_type=EvidenceType.AUDIT_LOG,
            source_system="test",
            source_id="1",
            created_by="user",
        )
        
        evidence = await vault.store_evidence(b"content", metadata, actor_id="test-user")
        await vault.retrieve_evidence(evidence.id, actor_id="test-user")
        
        log = vault.get_audit_log()
        
        assert len(log) >= 2
        assert any(e.action == AuditAction.CREATE for e in log)
        assert any(e.action == AuditAction.READ for e in log)

    def test_audit_log_filtering(self) -> None:
        """Test filtering audit log."""
        storage = InMemoryStorage()
        vault = EvidenceVault(storage)
        
        # Add some entries manually for filtering test
        vault._log_action(AuditAction.CREATE, "user1", "evidence", "e1")
        vault._log_action(AuditAction.READ, "user2", "evidence", "e1")
        vault._log_action(AuditAction.CREATE, "user1", "evidence", "e2")
        
        # Filter by action
        creates = vault.get_audit_log(action=AuditAction.CREATE)
        assert len(creates) == 2
        
        # Filter by actor
        user1_actions = vault.get_audit_log(actor_id="user1")
        assert len(user1_actions) == 2


class TestComplianceReportGenerator:
    """Tests for compliance report generation."""

    @pytest.mark.asyncio
    async def test_generate_soc2_report(self) -> None:
        """Test generating a SOC2 compliance report."""
        storage = InMemoryStorage()
        vault = EvidenceVault(storage)
        
        # Store some evidence
        for i in range(5):
            metadata = EvidenceMetadata(
                evidence_type=EvidenceType.VERIFICATION_PROOF,
                source_system="codeverify",
                source_id=f"pr-{i}",
                created_by="system",
                frameworks=[ComplianceFramework.SOC2],
                controls=["CC6.1", "CC8.1"],
            )
            await vault.store_evidence(f"proof {i}".encode(), metadata)
        
        generator = ComplianceReportGenerator(vault)
        report = generator.generate_report(
            org_id="org-123",
            framework=ComplianceFramework.SOC2,
            period_start=datetime.utcnow() - timedelta(days=90),
            period_end=datetime.utcnow(),
        )
        
        assert report is not None
        assert report.framework == ComplianceFramework.SOC2
        assert len(report.controls) > 0
        assert report.compliance_score >= 0

    @pytest.mark.asyncio
    async def test_export_report_html(self) -> None:
        """Test exporting report as HTML."""
        storage = InMemoryStorage()
        vault = EvidenceVault(storage)
        
        generator = ComplianceReportGenerator(vault)
        report = generator.generate_report(
            org_id="org-123",
            framework=ComplianceFramework.SOC2,
            period_start=datetime.utcnow() - timedelta(days=30),
            period_end=datetime.utcnow(),
        )
        
        html = generator.export_report_html(report)
        
        assert "<!DOCTYPE html>" in html
        assert report.title in html
        assert "Compliance" in html

    @pytest.mark.asyncio
    async def test_export_report_json(self) -> None:
        """Test exporting report as JSON."""
        storage = InMemoryStorage()
        vault = EvidenceVault(storage)
        
        generator = ComplianceReportGenerator(vault)
        report = generator.generate_report(
            org_id="org-123",
            framework=ComplianceFramework.HIPAA,
            period_start=datetime.utcnow() - timedelta(days=30),
            period_end=datetime.utcnow(),
        )
        
        json_str = generator.export_report_json(report)
        data = json.loads(json_str)
        
        assert data["framework"] == "hipaa"
        assert "controls" in data
        assert "compliance_score" in data


class TestGlobalVault:
    """Tests for global vault functions."""

    def test_get_vault(self) -> None:
        """Test getting global vault."""
        reset_evidence_vault()
        
        vault1 = get_evidence_vault()
        vault2 = get_evidence_vault()
        
        assert vault1 is vault2

    def test_reset_vault(self) -> None:
        """Test resetting global vault."""
        vault1 = get_evidence_vault()
        reset_evidence_vault()
        vault2 = get_evidence_vault()
        
        assert vault1 is not vault2
