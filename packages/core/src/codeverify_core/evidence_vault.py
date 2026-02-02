"""Compliance Evidence Vault - Immutable proof storage and compliance reporting.

This module provides:
1. Immutable storage for verification proofs with cryptographic integrity
2. Compliance report generation (SOC2, HIPAA, PCI-DSS, GDPR, ISO 27001)
3. Auditor portal with time-scoped access tokens
4. Export with chain of custody tracking
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger()


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""
    SOC2 = "soc2"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    GDPR = "gdpr"
    ISO_27001 = "iso_27001"
    NIST_CSF = "nist_csf"
    FedRAMP = "fedramp"
    CCPA = "ccpa"


class EvidenceType(str, Enum):
    """Types of evidence that can be stored."""
    VERIFICATION_PROOF = "verification_proof"
    CODE_REVIEW = "code_review"
    SECURITY_SCAN = "security_scan"
    TRUST_SCORE = "trust_score"
    ATTESTATION = "attestation"
    AUDIT_LOG = "audit_log"
    CONFIGURATION = "configuration"
    POLICY_CHECK = "policy_check"


class ControlStatus(str, Enum):
    """Status of a compliance control."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"
    NOT_ASSESSED = "not_assessed"


class AuditAction(str, Enum):
    """Types of audit actions."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXPORT = "export"
    VERIFY = "verify"
    GRANT_ACCESS = "grant_access"
    REVOKE_ACCESS = "revoke_access"


@dataclass
class EvidenceMetadata:
    """Metadata for stored evidence."""
    evidence_type: EvidenceType
    source_system: str
    source_id: str
    created_by: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: list[str] = field(default_factory=list)
    frameworks: list[ComplianceFramework] = field(default_factory=list)
    controls: list[str] = field(default_factory=list)
    retention_days: int = 2555  # ~7 years default
    custom_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class StoredEvidence:
    """A piece of evidence stored in the vault."""
    id: str = field(default_factory=lambda: str(uuid4()))
    content_hash: str = ""
    content_size: int = 0
    metadata: EvidenceMetadata = field(default_factory=lambda: EvidenceMetadata(
        evidence_type=EvidenceType.VERIFICATION_PROOF,
        source_system="unknown",
        source_id="unknown",
        created_by="system",
    ))
    signature: str = ""
    chain_hash: str = ""
    sequence_number: int = 0
    stored_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None
    is_sealed: bool = False

    def compute_hash(self, content: bytes) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content_hash": self.content_hash,
            "content_size": self.content_size,
            "metadata": {
                "evidence_type": self.metadata.evidence_type.value,
                "source_system": self.metadata.source_system,
                "source_id": self.metadata.source_id,
                "created_by": self.metadata.created_by,
                "created_at": self.metadata.created_at.isoformat(),
                "tags": self.metadata.tags,
                "frameworks": [f.value for f in self.metadata.frameworks],
                "controls": self.metadata.controls,
            },
            "signature": self.signature,
            "chain_hash": self.chain_hash,
            "sequence_number": self.sequence_number,
            "stored_at": self.stored_at.isoformat(),
            "is_sealed": self.is_sealed,
        }


@dataclass
class AuditLogEntry:
    """An entry in the audit log."""
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    action: AuditAction = AuditAction.READ
    actor_id: str = ""
    actor_type: str = "user"  # user, system, api
    resource_type: str = ""
    resource_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    ip_address: str | None = None
    user_agent: str | None = None
    success: bool = True
    error_message: str | None = None


@dataclass
class AccessToken:
    """Time-scoped access token for auditors."""
    token_id: str = field(default_factory=lambda: str(uuid4()))
    token_hash: str = ""
    org_id: str = ""
    auditor_email: str = ""
    auditor_name: str = ""
    permissions: list[str] = field(default_factory=list)  # read, export
    frameworks: list[ComplianceFramework] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=30))
    last_used: datetime | None = None
    is_revoked: bool = False


@dataclass
class ControlMapping:
    """Mapping of evidence to compliance control."""
    framework: ComplianceFramework
    control_id: str
    control_name: str
    control_description: str
    evidence_ids: list[str] = field(default_factory=list)
    status: ControlStatus = ControlStatus.NOT_ASSESSED
    assessment_notes: str = ""
    assessed_at: datetime | None = None
    assessed_by: str | None = None


@dataclass
class ComplianceReport:
    """A compliance report for a specific framework."""
    id: str = field(default_factory=lambda: str(uuid4()))
    org_id: str = ""
    framework: ComplianceFramework = ComplianceFramework.SOC2
    title: str = ""
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: datetime = field(default_factory=datetime.utcnow)
    controls: list[ControlMapping] = field(default_factory=list)
    summary: str = ""
    generated_at: datetime = field(default_factory=datetime.utcnow)
    generated_by: str = ""
    
    @property
    def compliance_score(self) -> float:
        """Calculate overall compliance percentage."""
        if not self.controls:
            return 0.0
        compliant = sum(1 for c in self.controls if c.status == ControlStatus.COMPLIANT)
        applicable = sum(1 for c in self.controls if c.status != ControlStatus.NOT_APPLICABLE)
        return (compliant / applicable * 100) if applicable > 0 else 0.0

    @property
    def status_summary(self) -> dict[str, int]:
        """Get count of controls by status."""
        summary: dict[str, int] = {}
        for control in self.controls:
            status = control.status.value
            summary[status] = summary.get(status, 0) + 1
        return summary


class EvidenceStorageBackend(ABC):
    """Abstract base class for evidence storage backends."""

    @abstractmethod
    async def store(self, evidence_id: str, content: bytes) -> bool:
        """Store evidence content."""
        pass

    @abstractmethod
    async def retrieve(self, evidence_id: str) -> bytes | None:
        """Retrieve evidence content."""
        pass

    @abstractmethod
    async def exists(self, evidence_id: str) -> bool:
        """Check if evidence exists."""
        pass

    @abstractmethod
    async def delete(self, evidence_id: str) -> bool:
        """Delete evidence (if allowed by retention policy)."""
        pass


class FileSystemStorage(EvidenceStorageBackend):
    """File system based evidence storage."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_path(self, evidence_id: str) -> Path:
        """Get storage path for evidence."""
        # Use subdirectories based on ID prefix to avoid too many files in one directory
        prefix = evidence_id[:2]
        subdir = self.base_path / prefix
        subdir.mkdir(exist_ok=True)
        return subdir / f"{evidence_id}.evidence"

    async def store(self, evidence_id: str, content: bytes) -> bool:
        """Store evidence content."""
        try:
            path = self._get_path(evidence_id)
            path.write_bytes(content)
            return True
        except Exception as e:
            logger.error("Failed to store evidence", evidence_id=evidence_id, error=str(e))
            return False

    async def retrieve(self, evidence_id: str) -> bytes | None:
        """Retrieve evidence content."""
        try:
            path = self._get_path(evidence_id)
            if path.exists():
                return path.read_bytes()
            return None
        except Exception as e:
            logger.error("Failed to retrieve evidence", evidence_id=evidence_id, error=str(e))
            return None

    async def exists(self, evidence_id: str) -> bool:
        """Check if evidence exists."""
        return self._get_path(evidence_id).exists()

    async def delete(self, evidence_id: str) -> bool:
        """Delete evidence."""
        try:
            path = self._get_path(evidence_id)
            if path.exists():
                path.unlink()
                return True
            return False
        except Exception as e:
            logger.error("Failed to delete evidence", evidence_id=evidence_id, error=str(e))
            return False


class InMemoryStorage(EvidenceStorageBackend):
    """In-memory evidence storage for testing."""

    def __init__(self) -> None:
        self._storage: dict[str, bytes] = {}

    async def store(self, evidence_id: str, content: bytes) -> bool:
        self._storage[evidence_id] = content
        return True

    async def retrieve(self, evidence_id: str) -> bytes | None:
        return self._storage.get(evidence_id)

    async def exists(self, evidence_id: str) -> bool:
        return evidence_id in self._storage

    async def delete(self, evidence_id: str) -> bool:
        if evidence_id in self._storage:
            del self._storage[evidence_id]
            return True
        return False


class EvidenceVault:
    """Main evidence vault for storing and managing compliance evidence."""

    def __init__(
        self,
        storage: EvidenceStorageBackend,
        signing_key: str | None = None,
    ) -> None:
        self.storage = storage
        self._signing_key = signing_key or os.environ.get("EVIDENCE_SIGNING_KEY", secrets.token_hex(32))
        self._evidence_index: dict[str, StoredEvidence] = {}
        self._audit_log: list[AuditLogEntry] = []
        self._access_tokens: dict[str, AccessToken] = {}
        self._sequence_counter = 0
        self._last_chain_hash = "genesis"

    async def store_evidence(
        self,
        content: bytes,
        metadata: EvidenceMetadata,
        actor_id: str = "system",
    ) -> StoredEvidence:
        """Store evidence with cryptographic integrity."""
        evidence = StoredEvidence(metadata=metadata)
        
        # Compute content hash
        evidence.content_hash = evidence.compute_hash(content)
        evidence.content_size = len(content)
        
        # Compute chain hash (links to previous evidence)
        self._sequence_counter += 1
        evidence.sequence_number = self._sequence_counter
        chain_data = f"{self._last_chain_hash}:{evidence.id}:{evidence.content_hash}:{evidence.sequence_number}"
        evidence.chain_hash = hashlib.sha256(chain_data.encode()).hexdigest()
        self._last_chain_hash = evidence.chain_hash
        
        # Sign the evidence
        evidence.signature = self._sign(evidence.content_hash + evidence.chain_hash)
        
        # Set expiration
        if metadata.retention_days > 0:
            evidence.expires_at = datetime.utcnow() + timedelta(days=metadata.retention_days)
        
        # Store content
        success = await self.storage.store(evidence.id, content)
        if not success:
            raise RuntimeError(f"Failed to store evidence content: {evidence.id}")
        
        # Index the evidence
        self._evidence_index[evidence.id] = evidence
        
        # Audit log
        self._log_action(
            AuditAction.CREATE,
            actor_id,
            "evidence",
            evidence.id,
            {"content_hash": evidence.content_hash, "type": metadata.evidence_type.value},
        )
        
        logger.info(
            "Stored evidence",
            evidence_id=evidence.id,
            type=metadata.evidence_type.value,
            size=evidence.content_size,
        )
        
        return evidence

    async def retrieve_evidence(
        self,
        evidence_id: str,
        actor_id: str = "system",
        verify: bool = True,
    ) -> tuple[StoredEvidence, bytes] | None:
        """Retrieve evidence with optional integrity verification."""
        evidence = self._evidence_index.get(evidence_id)
        if not evidence:
            return None
        
        content = await self.storage.retrieve(evidence_id)
        if content is None:
            return None
        
        # Verify integrity if requested
        if verify:
            computed_hash = evidence.compute_hash(content)
            if computed_hash != evidence.content_hash:
                logger.error(
                    "Evidence integrity check failed",
                    evidence_id=evidence_id,
                    expected=evidence.content_hash,
                    computed=computed_hash,
                )
                self._log_action(
                    AuditAction.VERIFY,
                    actor_id,
                    "evidence",
                    evidence_id,
                    {"result": "failed", "error": "hash_mismatch"},
                    success=False,
                )
                raise ValueError("Evidence integrity check failed")
            
            # Verify signature
            expected_sig = self._sign(evidence.content_hash + evidence.chain_hash)
            if evidence.signature != expected_sig:
                logger.error("Evidence signature verification failed", evidence_id=evidence_id)
                raise ValueError("Evidence signature verification failed")
        
        # Audit log
        self._log_action(AuditAction.READ, actor_id, "evidence", evidence_id)
        
        return evidence, content

    async def seal_evidence(self, evidence_id: str, actor_id: str = "system") -> bool:
        """Seal evidence to prevent modification."""
        evidence = self._evidence_index.get(evidence_id)
        if not evidence:
            return False
        
        evidence.is_sealed = True
        
        self._log_action(
            AuditAction.UPDATE,
            actor_id,
            "evidence",
            evidence_id,
            {"action": "seal"},
        )
        
        return True

    def search_evidence(
        self,
        evidence_type: EvidenceType | None = None,
        framework: ComplianceFramework | None = None,
        tags: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        source_system: str | None = None,
    ) -> list[StoredEvidence]:
        """Search evidence by criteria."""
        results = []
        
        for evidence in self._evidence_index.values():
            # Filter by type
            if evidence_type and evidence.metadata.evidence_type != evidence_type:
                continue
            
            # Filter by framework
            if framework and framework not in evidence.metadata.frameworks:
                continue
            
            # Filter by tags
            if tags and not any(tag in evidence.metadata.tags for tag in tags):
                continue
            
            # Filter by date range
            if start_date and evidence.stored_at < start_date:
                continue
            if end_date and evidence.stored_at > end_date:
                continue
            
            # Filter by source system
            if source_system and evidence.metadata.source_system != source_system:
                continue
            
            results.append(evidence)
        
        return sorted(results, key=lambda e: e.stored_at, reverse=True)

    def create_access_token(
        self,
        org_id: str,
        auditor_email: str,
        auditor_name: str,
        permissions: list[str],
        frameworks: list[ComplianceFramework],
        valid_days: int = 30,
        actor_id: str = "system",
    ) -> tuple[str, AccessToken]:
        """Create a time-scoped access token for an auditor."""
        # Generate secure token
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        
        token = AccessToken(
            token_hash=token_hash,
            org_id=org_id,
            auditor_email=auditor_email,
            auditor_name=auditor_name,
            permissions=permissions,
            frameworks=frameworks,
            expires_at=datetime.utcnow() + timedelta(days=valid_days),
        )
        
        self._access_tokens[token.token_id] = token
        
        self._log_action(
            AuditAction.GRANT_ACCESS,
            actor_id,
            "access_token",
            token.token_id,
            {
                "auditor": auditor_email,
                "permissions": permissions,
                "frameworks": [f.value for f in frameworks],
                "expires_at": token.expires_at.isoformat(),
            },
        )
        
        logger.info(
            "Created auditor access token",
            token_id=token.token_id,
            auditor=auditor_email,
            expires_at=token.expires_at.isoformat(),
        )
        
        return raw_token, token

    def validate_token(self, raw_token: str) -> AccessToken | None:
        """Validate an access token."""
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        
        for token in self._access_tokens.values():
            if token.token_hash == token_hash:
                if token.is_revoked:
                    return None
                if token.expires_at < datetime.utcnow():
                    return None
                token.last_used = datetime.utcnow()
                return token
        
        return None

    def revoke_token(self, token_id: str, actor_id: str = "system") -> bool:
        """Revoke an access token."""
        token = self._access_tokens.get(token_id)
        if not token:
            return False
        
        token.is_revoked = True
        
        self._log_action(
            AuditAction.REVOKE_ACCESS,
            actor_id,
            "access_token",
            token_id,
            {"auditor": token.auditor_email},
        )
        
        return True

    async def export_evidence_package(
        self,
        evidence_ids: list[str],
        format: str = "zip",
        actor_id: str = "system",
    ) -> tuple[bytes, str]:
        """Export evidence package with chain of custody."""
        import io
        import zipfile
        
        buffer = io.BytesIO()
        manifest = {
            "export_id": str(uuid4()),
            "exported_at": datetime.utcnow().isoformat(),
            "exported_by": actor_id,
            "evidence_count": len(evidence_ids),
            "evidence": [],
        }
        
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for evidence_id in evidence_ids:
                result = await self.retrieve_evidence(evidence_id, actor_id, verify=True)
                if not result:
                    continue
                
                evidence, content = result
                
                # Add evidence content
                filename = f"evidence/{evidence_id}/{evidence.metadata.evidence_type.value}.dat"
                zf.writestr(filename, content)
                
                # Add evidence metadata
                meta_filename = f"evidence/{evidence_id}/metadata.json"
                zf.writestr(meta_filename, json.dumps(evidence.to_dict(), indent=2))
                
                manifest["evidence"].append({
                    "id": evidence_id,
                    "hash": evidence.content_hash,
                    "chain_hash": evidence.chain_hash,
                    "sequence": evidence.sequence_number,
                })
            
            # Add manifest
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            
            # Add chain of custody certificate
            custody_cert = self._generate_custody_certificate(manifest, actor_id)
            zf.writestr("chain_of_custody.json", json.dumps(custody_cert, indent=2))
        
        # Log export
        self._log_action(
            AuditAction.EXPORT,
            actor_id,
            "evidence_package",
            manifest["export_id"],
            {"evidence_count": len(evidence_ids)},
        )
        
        buffer.seek(0)
        return buffer.read(), manifest["export_id"]

    def _generate_custody_certificate(
        self,
        manifest: dict[str, Any],
        actor_id: str,
    ) -> dict[str, Any]:
        """Generate chain of custody certificate."""
        cert_data = json.dumps(manifest, sort_keys=True)
        signature = self._sign(cert_data)
        
        return {
            "certificate_id": str(uuid4()),
            "generated_at": datetime.utcnow().isoformat(),
            "generated_by": actor_id,
            "manifest_hash": hashlib.sha256(cert_data.encode()).hexdigest(),
            "signature": signature,
            "signature_algorithm": "HMAC-SHA256",
            "verification_instructions": (
                "To verify: compute HMAC-SHA256 of the manifest JSON (sorted keys) "
                "using the organization's signing key and compare to signature."
            ),
        }

    def get_audit_log(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        action: AuditAction | None = None,
        actor_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditLogEntry]:
        """Get filtered audit log entries."""
        results = []
        
        for entry in reversed(self._audit_log):
            if start_date and entry.timestamp < start_date:
                continue
            if end_date and entry.timestamp > end_date:
                continue
            if action and entry.action != action:
                continue
            if actor_id and entry.actor_id != actor_id:
                continue
            
            results.append(entry)
            if len(results) >= limit:
                break
        
        return results

    def _sign(self, data: str) -> str:
        """Sign data using HMAC-SHA256."""
        return hmac.new(
            self._signing_key.encode(),
            data.encode(),
            hashlib.sha256,
        ).hexdigest()

    def _log_action(
        self,
        action: AuditAction,
        actor_id: str,
        resource_type: str,
        resource_id: str,
        details: dict[str, Any] | None = None,
        success: bool = True,
        error_message: str | None = None,
    ) -> None:
        """Add entry to audit log."""
        entry = AuditLogEntry(
            action=action,
            actor_id=actor_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            success=success,
            error_message=error_message,
        )
        self._audit_log.append(entry)


class ComplianceReportGenerator:
    """Generates compliance reports from stored evidence."""

    # Control definitions by framework
    CONTROL_DEFINITIONS: dict[ComplianceFramework, list[dict[str, str]]] = {
        ComplianceFramework.SOC2: [
            {"id": "CC1.1", "name": "Control Environment", "description": "Management demonstrates commitment to integrity and ethical values"},
            {"id": "CC2.1", "name": "Communication", "description": "Organization communicates with external parties"},
            {"id": "CC3.1", "name": "Risk Assessment", "description": "Organization specifies objectives with sufficient clarity"},
            {"id": "CC4.1", "name": "Monitoring Activities", "description": "Organization selects and develops ongoing evaluations"},
            {"id": "CC5.1", "name": "Control Activities", "description": "Organization selects control activities that mitigate risks"},
            {"id": "CC6.1", "name": "Logical Access", "description": "Organization implements logical access security controls"},
            {"id": "CC6.6", "name": "External Threats", "description": "Organization protects against external threats"},
            {"id": "CC6.7", "name": "Information Transmission", "description": "Organization restricts transmission of confidential information"},
            {"id": "CC7.1", "name": "System Operations", "description": "Organization detects security events"},
            {"id": "CC7.2", "name": "Incident Response", "description": "Organization responds to identified security incidents"},
            {"id": "CC8.1", "name": "Change Management", "description": "Organization authorizes, designs, and implements changes"},
            {"id": "CC9.1", "name": "Risk Mitigation", "description": "Organization identifies and assesses risks from business partners"},
        ],
        ComplianceFramework.HIPAA: [
            {"id": "164.308(a)(1)", "name": "Security Management Process", "description": "Implement policies and procedures to prevent, detect, contain, and correct security violations"},
            {"id": "164.308(a)(3)", "name": "Workforce Security", "description": "Ensure all workforce members have appropriate access"},
            {"id": "164.308(a)(4)", "name": "Information Access Management", "description": "Implement policies for authorizing access to ePHI"},
            {"id": "164.308(a)(5)", "name": "Security Awareness", "description": "Implement security awareness and training program"},
            {"id": "164.308(a)(6)", "name": "Security Incident Procedures", "description": "Implement policies for responding to security incidents"},
            {"id": "164.310(a)(1)", "name": "Facility Access Controls", "description": "Implement policies to limit physical access"},
            {"id": "164.310(d)(1)", "name": "Device and Media Controls", "description": "Implement policies for disposal of ePHI"},
            {"id": "164.312(a)(1)", "name": "Access Control", "description": "Implement technical policies to allow access only to authorized persons"},
            {"id": "164.312(b)", "name": "Audit Controls", "description": "Implement mechanisms to record and examine system activity"},
            {"id": "164.312(c)(1)", "name": "Integrity", "description": "Implement mechanisms to protect ePHI from improper alteration"},
            {"id": "164.312(d)", "name": "Authentication", "description": "Implement procedures to verify person seeking access"},
            {"id": "164.312(e)(1)", "name": "Transmission Security", "description": "Implement technical security measures for ePHI transmission"},
        ],
        ComplianceFramework.PCI_DSS: [
            {"id": "1.1", "name": "Firewall Configuration", "description": "Install and maintain network security controls"},
            {"id": "2.1", "name": "Secure Configurations", "description": "Apply secure configurations to all system components"},
            {"id": "3.1", "name": "Account Data Protection", "description": "Protect stored account data"},
            {"id": "4.1", "name": "Strong Cryptography", "description": "Protect cardholder data with strong cryptography during transmission"},
            {"id": "5.1", "name": "Malware Protection", "description": "Protect all systems against malware"},
            {"id": "6.1", "name": "Secure Development", "description": "Develop and maintain secure systems and software"},
            {"id": "7.1", "name": "Access Restriction", "description": "Restrict access to cardholder data by business need to know"},
            {"id": "8.1", "name": "User Identification", "description": "Identify users and authenticate access"},
            {"id": "9.1", "name": "Physical Access", "description": "Restrict physical access to cardholder data"},
            {"id": "10.1", "name": "Logging and Monitoring", "description": "Log and monitor all access to cardholder data"},
            {"id": "11.1", "name": "Security Testing", "description": "Test security of systems and networks regularly"},
            {"id": "12.1", "name": "Security Policy", "description": "Maintain information security policy"},
        ],
        ComplianceFramework.GDPR: [
            {"id": "Art.5", "name": "Data Processing Principles", "description": "Personal data processing principles"},
            {"id": "Art.6", "name": "Lawful Processing", "description": "Lawfulness of processing"},
            {"id": "Art.7", "name": "Consent", "description": "Conditions for consent"},
            {"id": "Art.25", "name": "Data Protection by Design", "description": "Data protection by design and default"},
            {"id": "Art.30", "name": "Records of Processing", "description": "Records of processing activities"},
            {"id": "Art.32", "name": "Security of Processing", "description": "Security of personal data processing"},
            {"id": "Art.33", "name": "Breach Notification", "description": "Notification of personal data breach to supervisory authority"},
            {"id": "Art.35", "name": "Impact Assessment", "description": "Data protection impact assessment"},
        ],
        ComplianceFramework.ISO_27001: [
            {"id": "A.5", "name": "Information Security Policies", "description": "Management direction for information security"},
            {"id": "A.6", "name": "Organization of Security", "description": "Internal organization and mobile/teleworking"},
            {"id": "A.7", "name": "Human Resource Security", "description": "Prior to, during, and termination of employment"},
            {"id": "A.8", "name": "Asset Management", "description": "Responsibility for assets and information classification"},
            {"id": "A.9", "name": "Access Control", "description": "Business requirements and user access management"},
            {"id": "A.10", "name": "Cryptography", "description": "Cryptographic controls"},
            {"id": "A.11", "name": "Physical Security", "description": "Secure areas and equipment"},
            {"id": "A.12", "name": "Operations Security", "description": "Operational procedures and responsibilities"},
            {"id": "A.13", "name": "Communications Security", "description": "Network security and information transfer"},
            {"id": "A.14", "name": "System Development", "description": "Security requirements and secure development"},
            {"id": "A.15", "name": "Supplier Relationships", "description": "Information security in supplier relationships"},
            {"id": "A.16", "name": "Incident Management", "description": "Management of security incidents"},
            {"id": "A.17", "name": "Business Continuity", "description": "Information security continuity"},
            {"id": "A.18", "name": "Compliance", "description": "Compliance with legal and contractual requirements"},
        ],
    }

    # Mapping of evidence types to controls
    EVIDENCE_TO_CONTROLS: dict[EvidenceType, dict[ComplianceFramework, list[str]]] = {
        EvidenceType.VERIFICATION_PROOF: {
            ComplianceFramework.SOC2: ["CC6.1", "CC8.1"],
            ComplianceFramework.PCI_DSS: ["6.1"],
            ComplianceFramework.ISO_27001: ["A.14"],
        },
        EvidenceType.SECURITY_SCAN: {
            ComplianceFramework.SOC2: ["CC6.6", "CC7.1"],
            ComplianceFramework.HIPAA: ["164.308(a)(1)", "164.312(b)"],
            ComplianceFramework.PCI_DSS: ["5.1", "11.1"],
            ComplianceFramework.ISO_27001: ["A.12"],
        },
        EvidenceType.CODE_REVIEW: {
            ComplianceFramework.SOC2: ["CC8.1"],
            ComplianceFramework.PCI_DSS: ["6.1"],
            ComplianceFramework.ISO_27001: ["A.14"],
        },
        EvidenceType.AUDIT_LOG: {
            ComplianceFramework.SOC2: ["CC4.1", "CC7.1"],
            ComplianceFramework.HIPAA: ["164.312(b)"],
            ComplianceFramework.PCI_DSS: ["10.1"],
            ComplianceFramework.ISO_27001: ["A.12"],
        },
        EvidenceType.CONFIGURATION: {
            ComplianceFramework.SOC2: ["CC6.1"],
            ComplianceFramework.PCI_DSS: ["2.1"],
            ComplianceFramework.ISO_27001: ["A.12"],
        },
    }

    def __init__(self, vault: EvidenceVault) -> None:
        self.vault = vault

    def generate_report(
        self,
        org_id: str,
        framework: ComplianceFramework,
        period_start: datetime,
        period_end: datetime,
        generated_by: str = "system",
    ) -> ComplianceReport:
        """Generate a compliance report for a framework."""
        # Get control definitions
        controls_defs = self.CONTROL_DEFINITIONS.get(framework, [])
        
        # Search for evidence in the period
        evidence_list = self.vault.search_evidence(
            framework=framework,
            start_date=period_start,
            end_date=period_end,
        )
        
        # Build control mappings
        controls: list[ControlMapping] = []
        for control_def in controls_defs:
            # Find evidence for this control
            evidence_ids = self._find_evidence_for_control(
                evidence_list,
                framework,
                control_def["id"],
            )
            
            # Determine status based on evidence
            if evidence_ids:
                status = ControlStatus.COMPLIANT
            else:
                status = ControlStatus.NOT_ASSESSED
            
            mapping = ControlMapping(
                framework=framework,
                control_id=control_def["id"],
                control_name=control_def["name"],
                control_description=control_def["description"],
                evidence_ids=evidence_ids,
                status=status,
                assessed_at=datetime.utcnow() if evidence_ids else None,
            )
            controls.append(mapping)
        
        # Generate report
        report = ComplianceReport(
            org_id=org_id,
            framework=framework,
            title=f"{framework.value.upper()} Compliance Report",
            period_start=period_start,
            period_end=period_end,
            controls=controls,
            generated_by=generated_by,
        )
        
        # Generate summary
        report.summary = self._generate_summary(report)
        
        logger.info(
            "Generated compliance report",
            report_id=report.id,
            framework=framework.value,
            compliance_score=report.compliance_score,
        )
        
        return report

    def _find_evidence_for_control(
        self,
        evidence_list: list[StoredEvidence],
        framework: ComplianceFramework,
        control_id: str,
    ) -> list[str]:
        """Find evidence that supports a specific control."""
        evidence_ids = []
        
        for evidence in evidence_list:
            # Check if evidence explicitly maps to this control
            if control_id in evidence.metadata.controls:
                evidence_ids.append(evidence.id)
                continue
            
            # Check implicit mapping based on evidence type
            type_mappings = self.EVIDENCE_TO_CONTROLS.get(evidence.metadata.evidence_type, {})
            if framework in type_mappings and control_id in type_mappings[framework]:
                evidence_ids.append(evidence.id)
        
        return evidence_ids

    def _generate_summary(self, report: ComplianceReport) -> str:
        """Generate report summary."""
        status_counts = report.status_summary
        compliant = status_counts.get("compliant", 0)
        total = len(report.controls)
        
        return (
            f"Compliance assessment for {report.framework.value.upper()} framework. "
            f"Period: {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}. "
            f"Overall compliance score: {report.compliance_score:.1f}%. "
            f"Controls assessed: {total}. Compliant: {compliant}. "
            f"Non-compliant: {status_counts.get('non_compliant', 0)}. "
            f"Partial: {status_counts.get('partial', 0)}. "
            f"Not assessed: {status_counts.get('not_assessed', 0)}."
        )

    def export_report_html(self, report: ComplianceReport) -> str:
        """Export report as HTML."""
        controls_html = ""
        for control in report.controls:
            status_class = control.status.value.replace("_", "-")
            evidence_links = ", ".join(control.evidence_ids) if control.evidence_ids else "None"
            controls_html += f"""
            <tr class="{status_class}">
                <td>{control.control_id}</td>
                <td>{control.control_name}</td>
                <td>{control.status.value.replace('_', ' ').title()}</td>
                <td>{len(control.evidence_ids)}</td>
                <td>{control.control_description}</td>
            </tr>
            """
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
                .summary {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                .score {{ font-size: 48px; font-weight: bold; color: #007bff; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border: 1px solid #ddd; }}
                th {{ background: #007bff; color: white; }}
                .compliant {{ background: #d4edda; }}
                .non-compliant {{ background: #f8d7da; }}
                .partial {{ background: #fff3cd; }}
                .not-assessed {{ background: #f8f9fa; }}
                .footer {{ margin-top: 40px; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <h1>{report.title}</h1>
            
            <div class="summary">
                <p><strong>Organization:</strong> {report.org_id}</p>
                <p><strong>Period:</strong> {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}</p>
                <p><strong>Generated:</strong> {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                <div class="score">{report.compliance_score:.1f}%</div>
                <p>{report.summary}</p>
            </div>
            
            <h2>Control Assessment</h2>
            <table>
                <thead>
                    <tr>
                        <th>Control ID</th>
                        <th>Control Name</th>
                        <th>Status</th>
                        <th>Evidence Count</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    {controls_html}
                </tbody>
            </table>
            
            <div class="footer">
                <p>Report ID: {report.id}</p>
                <p>Generated by: {report.generated_by}</p>
                <p>This report is for internal assessment purposes only.</p>
            </div>
        </body>
        </html>
        """

    def export_report_json(self, report: ComplianceReport) -> str:
        """Export report as JSON."""
        return json.dumps({
            "report_id": report.id,
            "org_id": report.org_id,
            "framework": report.framework.value,
            "title": report.title,
            "period": {
                "start": report.period_start.isoformat(),
                "end": report.period_end.isoformat(),
            },
            "compliance_score": report.compliance_score,
            "status_summary": report.status_summary,
            "summary": report.summary,
            "controls": [
                {
                    "control_id": c.control_id,
                    "control_name": c.control_name,
                    "description": c.control_description,
                    "status": c.status.value,
                    "evidence_count": len(c.evidence_ids),
                    "evidence_ids": c.evidence_ids,
                    "assessment_notes": c.assessment_notes,
                }
                for c in report.controls
            ],
            "generated_at": report.generated_at.isoformat(),
            "generated_by": report.generated_by,
        }, indent=2)


# Global vault instance
_evidence_vault: EvidenceVault | None = None


def get_evidence_vault(
    storage: EvidenceStorageBackend | None = None,
    signing_key: str | None = None,
) -> EvidenceVault:
    """Get or create the global evidence vault."""
    global _evidence_vault
    if _evidence_vault is None:
        if storage is None:
            storage = InMemoryStorage()
        _evidence_vault = EvidenceVault(storage, signing_key)
    return _evidence_vault


def reset_evidence_vault() -> None:
    """Reset the global evidence vault (for testing)."""
    global _evidence_vault
    _evidence_vault = None
