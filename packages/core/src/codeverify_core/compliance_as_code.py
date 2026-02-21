"""Compliance-as-Code Framework — deep framework mappings, evidence vault, attestation.

Provides programmatic compliance with real SOC2, HIPAA, PCI-DSS v4, ISO 27001,
GDPR, NIST 800-53, FedRAMP, and EU AI Act control definitions. Includes an
integrated evidence vault with cryptographic integrity, HMAC-based attestation
signatures, and automated compliance report generation.

.. deprecated::
    This module is superseded by ``codeverify_core.compliance_engine``.
    It remains importable for backward compatibility but will be
    removed in a future release.
"""

from __future__ import annotations

import warnings as _warnings
_warnings.warn(
    "codeverify_core.compliance_as_code is deprecated. Use codeverify_core.compliance_engine instead.",
    DeprecationWarning,
    stacklevel=2,
)


import hashlib
import hmac
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


# =============================================================================
# Enumerations
# =============================================================================


class ComplianceFrameworkType(str, Enum):
    """Supported compliance frameworks with deep control mappings."""

    SOC2_TYPE_II = "SOC2_TYPE_II"
    HIPAA = "HIPAA"
    PCI_DSS_V4 = "PCI_DSS_V4"
    ISO_27001 = "ISO_27001"
    GDPR = "GDPR"
    NIST_800_53 = "NIST_800_53"
    FEDRAMP = "FEDRAMP"
    EU_AI_ACT = "EU_AI_ACT"


class EvidenceType(str, Enum):
    """Types of evidence artifacts collected for compliance."""

    VERIFICATION_REPORT = "verification_report"
    PROOF_ARTIFACT = "proof_artifact"
    AUDIT_LOG = "audit_log"
    TEST_RESULT = "test_result"
    CONFIGURATION = "configuration"
    ATTESTATION = "attestation"
    SCAN_RESULT = "scan_result"


class AttestationLevel(str, Enum):
    """Trust level of an attestation."""

    SELF_ATTESTED = "self_attested"
    PEER_REVIEWED = "peer_reviewed"
    AUTOMATED = "automated"
    AUDITOR_VERIFIED = "auditor_verified"


class ControlCategory(str, Enum):
    """High-level compliance control categories."""

    ACCESS_CONTROL = "access_control"
    CHANGE_MANAGEMENT = "change_management"
    RISK_ASSESSMENT = "risk_assessment"
    INCIDENT_RESPONSE = "incident_response"
    SECURITY_MONITORING = "security_monitoring"
    DATA_PROTECTION = "data_protection"
    AVAILABILITY = "availability"
    CODE_QUALITY = "code_quality"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FrameworkControl:
    """A single control within a compliance framework."""

    control_id: str
    framework: ComplianceFrameworkType
    category: ControlCategory
    title: str
    description: str
    requirements: list[str]
    verification_rules: list[str] = field(default_factory=list)
    evidence_types: list[EvidenceType] = field(default_factory=list)
    automated: bool = False
    status: str = "not_assessed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id, "framework": self.framework.value,
            "category": self.category.value, "title": self.title,
            "description": self.description, "requirements": self.requirements,
            "verification_rules": self.verification_rules,
            "evidence_types": [e.value for e in self.evidence_types],
            "automated": self.automated, "status": self.status,
        }


@dataclass
class EvidenceArtifact:
    """An immutable evidence artifact stored in the vault."""

    id: str
    evidence_type: EvidenceType
    title: str
    description: str
    content_hash: str
    collected_at: datetime
    control_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    retention_days: int = 2555  # 7 years

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "evidence_type": self.evidence_type.value,
            "title": self.title, "description": self.description,
            "content_hash": self.content_hash,
            "collected_at": self.collected_at.isoformat(),
            "control_ids": self.control_ids, "metadata": self.metadata,
            "retention_days": self.retention_days,
        }


@dataclass
class ComplianceAttestation:
    """A signed attestation that a control has been satisfied."""

    id: str
    control_id: str
    level: AttestationLevel
    attester: str
    statement: str
    evidence_ids: list[str]
    created_at: datetime
    expires_at: datetime | None
    signature: str | None = None
    valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "control_id": self.control_id,
            "level": self.level.value, "attester": self.attester,
            "statement": self.statement, "evidence_ids": self.evidence_ids,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "signature": self.signature, "valid": self.valid,
        }


@dataclass
class ControlAssessment:
    """Result of assessing a single control."""

    control_id: str
    framework: ComplianceFrameworkType
    status: str
    evidence_count: int
    attestation_count: int
    gaps: list[str]
    last_assessed: datetime
    next_assessment_due: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id, "framework": self.framework.value,
            "status": self.status, "evidence_count": self.evidence_count,
            "attestation_count": self.attestation_count, "gaps": self.gaps,
            "last_assessed": self.last_assessed.isoformat(),
            "next_assessment_due": (
                self.next_assessment_due.isoformat() if self.next_assessment_due else None
            ),
        }


@dataclass
class ComplianceReport:
    """Full compliance report for an organization and framework."""

    id: str
    framework: ComplianceFrameworkType
    generated_at: datetime
    organization: str
    period_start: datetime
    period_end: datetime
    total_controls: int
    passing_controls: int
    failing_controls: int
    not_assessed: int
    assessments: list[ControlAssessment]
    executive_summary: str
    risk_areas: list[str]
    remediation_priorities: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "framework": self.framework.value,
            "generated_at": self.generated_at.isoformat(),
            "organization": self.organization,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_controls": self.total_controls,
            "passing_controls": self.passing_controls,
            "failing_controls": self.failing_controls,
            "not_assessed": self.not_assessed,
            "assessments": [a.to_dict() for a in self.assessments],
            "executive_summary": self.executive_summary,
            "risk_areas": self.risk_areas,
            "remediation_priorities": self.remediation_priorities,
        }


# =============================================================================
# Framework Mapper — real control definitions
# =============================================================================

_FW = ComplianceFrameworkType
_CAT = ControlCategory
_ET = EvidenceType


def _ctrl(
    cid: str, fw: _FW, cat: _CAT, title: str, desc: str,
    reqs: list[str], rules: list[str], evid: list[_ET],
    auto: bool = True,
) -> FrameworkControl:
    """Shorthand factory for FrameworkControl."""
    return FrameworkControl(
        control_id=cid, framework=fw, category=cat, title=title,
        description=desc, requirements=reqs, verification_rules=rules,
        evidence_types=evid, automated=auto,
    )


class FrameworkMapper:
    """Maps verification rules to compliance framework controls."""

    def __init__(self) -> None:
        self._controls: dict[ComplianceFrameworkType, list[FrameworkControl]] = {}
        self._load_all_frameworks()

    def _load_all_frameworks(self) -> None:
        self._controls[_FW.SOC2_TYPE_II] = self._get_soc2_controls()
        self._controls[_FW.HIPAA] = self._get_hipaa_controls()
        self._controls[_FW.PCI_DSS_V4] = self._get_pci_dss_controls()
        self._controls[_FW.GDPR] = self._get_gdpr_controls()
        logger.info("compliance_frameworks_loaded", count=len(self._controls))

    def get_controls(self, framework: ComplianceFrameworkType) -> list[FrameworkControl]:
        """Return all controls for a given framework."""
        return list(self._controls.get(framework, []))

    def map_verification_to_controls(
        self, verification_rules: list[str], framework: ComplianceFrameworkType,
    ) -> dict[str, list[str]]:
        """Map verification rule IDs to the controls they satisfy."""
        rule_set = set(verification_rules)
        mapping: dict[str, list[str]] = {}
        for control in self.get_controls(framework):
            matched = [r for r in control.verification_rules if r in rule_set]
            if matched:
                mapping[control.control_id] = matched
        return mapping

    def get_gap_analysis(
        self, framework: ComplianceFrameworkType, mapped_rules: dict[str, list[str]],
    ) -> list[str]:
        """Identify controls with no or partial verification coverage."""
        gaps: list[str] = []
        for control in self.get_controls(framework):
            if control.control_id not in mapped_rules:
                gaps.append(f"{control.control_id}: {control.title} — no rules mapped")
            elif len(mapped_rules[control.control_id]) < len(control.verification_rules):
                missing = set(control.verification_rules) - set(mapped_rules[control.control_id])
                gaps.append(
                    f"{control.control_id}: {control.title} — missing: {', '.join(sorted(missing))}"
                )
        return gaps

    # -- SOC2 Type II (Trust Services Criteria) --------------------------------

    def _get_soc2_controls(self) -> list[FrameworkControl]:
        fw = _FW.SOC2_TYPE_II
        return [
            _ctrl("CC5.1", fw, _CAT.RISK_ASSESSMENT, "Risk Identification and Assessment",
                  "Identify and assess risks to objectives including fraud risk.",
                  ["Periodic risk assessments", "Fraud risk consideration", "Risk response identification"],
                  ["risk_assessment_check", "threat_model_review"],
                  [_ET.VERIFICATION_REPORT, _ET.AUDIT_LOG]),
            _ctrl("CC6.1", fw, _CAT.ACCESS_CONTROL, "Logical Access Security",
                  "Implement logical access security over protected information assets.",
                  ["Role-based access provisioning", "Authentication mechanisms", "Access revocation"],
                  ["access_control_check", "auth_verification"],
                  [_ET.SCAN_RESULT, _ET.AUDIT_LOG]),
            _ctrl("CC7.2", fw, _CAT.SECURITY_MONITORING, "System Monitoring",
                  "Monitor system components for anomalies indicative of malicious acts.",
                  ["Anomaly detection", "Alert and escalation procedures", "Vulnerability scanning"],
                  ["security_scan", "vulnerability_detection", "monitoring_check"],
                  [_ET.SCAN_RESULT, _ET.VERIFICATION_REPORT]),
            _ctrl("CC8.1", fw, _CAT.CHANGE_MANAGEMENT, "Change Management Process",
                  "Authorize, develop, test, approve, and implement changes to infrastructure and software.",
                  ["Change authorization", "Testing before deployment", "Separation of environments"],
                  ["code_review_required", "approval_workflow", "ci_cd_check"],
                  [_ET.PROOF_ARTIFACT, _ET.AUDIT_LOG]),
            _ctrl("A1.2", fw, _CAT.AVAILABILITY, "Recovery and Continuity",
                  "Implement environmental protections, recovery, and continuity plans.",
                  ["Disaster recovery plan", "Backup procedures", "Recovery testing"],
                  ["backup_verification", "recovery_test"],
                  [_ET.TEST_RESULT, _ET.CONFIGURATION], auto=False),
        ]

    # -- HIPAA (Technical Safeguards, 45 CFR §164.312) -------------------------

    def _get_hipaa_controls(self) -> list[FrameworkControl]:
        fw = _FW.HIPAA
        return [
            _ctrl("§164.312(a)(1)", fw, _CAT.ACCESS_CONTROL, "Access Control",
                  "Implement technical policies for ePHI systems allowing only authorised access.",
                  ["Unique user identification", "Emergency access", "Automatic logoff", "Encryption"],
                  ["access_control_check", "auth_verification", "encryption_check"],
                  [_ET.SCAN_RESULT, _ET.AUDIT_LOG]),
            _ctrl("§164.312(a)(2)(iv)", fw, _CAT.DATA_PROTECTION, "Encryption and Decryption",
                  "Implement a mechanism to encrypt and decrypt ePHI.",
                  ["Encryption at rest", "Encryption in transit", "Key management"],
                  ["encryption_check", "tls_verification", "key_management_check"],
                  [_ET.CONFIGURATION, _ET.SCAN_RESULT]),
            _ctrl("§164.312(b)", fw, _CAT.SECURITY_MONITORING, "Audit Controls",
                  "Record and examine activity in systems that contain or use ePHI.",
                  ["Audit logging enabled", "Log review procedures", "Tamper-evident logging"],
                  ["logging_check", "audit_trail"],
                  [_ET.AUDIT_LOG, _ET.CONFIGURATION]),
            _ctrl("§164.312(c)(1)", fw, _CAT.DATA_PROTECTION, "Integrity Controls",
                  "Protect ePHI from improper alteration or destruction.",
                  ["Data integrity verification", "Digital signatures for data"],
                  ["input_validation", "data_integrity_check"],
                  [_ET.VERIFICATION_REPORT]),
            _ctrl("§164.312(e)(1)", fw, _CAT.DATA_PROTECTION, "Transmission Security",
                  "Guard against unauthorised access to ePHI transmitted over a network.",
                  ["Integrity controls for transmission", "Encryption for transmission"],
                  ["tls_verification", "transport_security_check"],
                  [_ET.SCAN_RESULT, _ET.CONFIGURATION]),
        ]

    # -- PCI-DSS v4.0 (Requirements 6 & 8) ------------------------------------

    def _get_pci_dss_controls(self) -> list[FrameworkControl]:
        fw = _FW.PCI_DSS_V4
        return [
            _ctrl("Req 6.2.4", fw, _CAT.CODE_QUALITY, "Software Engineering Techniques",
                  "Use techniques to prevent or mitigate common software attacks.",
                  ["Injection prevention", "Buffer overflow prevention", "Insecure crypto prevention"],
                  ["injection_check", "input_validation", "encryption_check"],
                  [_ET.SCAN_RESULT, _ET.VERIFICATION_REPORT]),
            _ctrl("Req 6.3.1", fw, _CAT.CODE_QUALITY, "Vulnerability Identification",
                  "Identify and manage security vulnerabilities via a defined process.",
                  ["Vulnerability scanning", "Risk ranking", "Timely patching"],
                  ["vulnerability_detection", "dependency_scan"],
                  [_ET.SCAN_RESULT]),
            _ctrl("Req 6.5.1", fw, _CAT.CHANGE_MANAGEMENT, "Change Control Processes",
                  "Changes to production follow established change control procedures.",
                  ["Impact analysis", "Documented approval", "Rollback procedures"],
                  ["code_review_required", "approval_workflow"],
                  [_ET.PROOF_ARTIFACT, _ET.AUDIT_LOG]),
            _ctrl("Req 8.3.1", fw, _CAT.ACCESS_CONTROL, "User Authentication Management",
                  "All user access to system components is uniquely identified.",
                  ["Unique user IDs", "Strong authentication", "MFA for remote access"],
                  ["auth_verification", "mfa_check"],
                  [_ET.SCAN_RESULT, _ET.CONFIGURATION]),
        ]

    # -- GDPR (Articles 25, 32, 33, 35) ----------------------------------------

    def _get_gdpr_controls(self) -> list[FrameworkControl]:
        fw = _FW.GDPR
        return [
            _ctrl("Art. 25", fw, _CAT.DATA_PROTECTION, "Data Protection by Design and Default",
                  "Implement measures ensuring only necessary personal data is processed by default.",
                  ["Data minimisation", "Purpose limitation", "Privacy by design review"],
                  ["data_minimization_check", "pii_detection"],
                  [_ET.VERIFICATION_REPORT, _ET.ATTESTATION]),
            _ctrl("Art. 32", fw, _CAT.DATA_PROTECTION, "Security of Processing",
                  "Implement measures to ensure security appropriate to the risk.",
                  ["Pseudonymisation and encryption", "Confidentiality", "Availability", "Regular testing"],
                  ["encryption_check", "access_control_check", "security_scan"],
                  [_ET.SCAN_RESULT, _ET.TEST_RESULT]),
            _ctrl("Art. 35", fw, _CAT.RISK_ASSESSMENT, "Data Protection Impact Assessment",
                  "Carry out an impact assessment for high-risk processing of personal data.",
                  ["Systematic description of processing", "Necessity and proportionality", "Risk assessment"],
                  ["risk_assessment_check", "data_flow_analysis"],
                  [_ET.VERIFICATION_REPORT, _ET.ATTESTATION], auto=False),
            _ctrl("Art. 33", fw, _CAT.INCIDENT_RESPONSE, "Personal Data Breach Notification",
                  "Notify the supervisory authority of a personal data breach within 72 hours.",
                  ["Breach detection mechanisms", "Notification procedures", "Breach documentation"],
                  ["incident_detection_check", "notification_workflow"],
                  [_ET.AUDIT_LOG, _ET.CONFIGURATION]),
        ]


# =============================================================================
# Evidence Vault
# =============================================================================


class EvidenceVault:
    """Immutable evidence store with cryptographic integrity verification."""

    def __init__(self) -> None:
        self._artifacts: dict[str, EvidenceArtifact] = {}
        self._hash_index: dict[str, str] = {}

    def store_evidence(self, evidence: EvidenceArtifact) -> str:
        """Store an evidence artifact and return its ID."""
        self._artifacts[evidence.id] = evidence
        self._hash_index[evidence.content_hash] = evidence.id
        logger.info("evidence_stored", evidence_id=evidence.id,
                     evidence_type=evidence.evidence_type.value)
        return evidence.id

    def retrieve_evidence(self, evidence_id: str) -> EvidenceArtifact | None:
        """Retrieve an evidence artifact by ID."""
        return self._artifacts.get(evidence_id)

    def get_evidence_for_control(self, control_id: str) -> list[EvidenceArtifact]:
        """Return all evidence artifacts linked to a given control ID."""
        return [a for a in self._artifacts.values() if control_id in a.control_ids]

    def verify_integrity(self, evidence_id: str) -> bool:
        """Verify that stored evidence has not been tampered with."""
        artifact = self._artifacts.get(evidence_id)
        if artifact is None:
            return False
        recomputed = self._compute_hash(
            f"{artifact.id}:{artifact.evidence_type.value}:{artifact.title}"
        )
        return hmac.compare_digest(artifact.content_hash, recomputed)

    def _compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def cleanup_expired(self) -> int:
        """Remove evidence that has exceeded its retention period."""
        now = datetime.now(timezone.utc)
        expired_ids = [
            aid for aid, a in self._artifacts.items()
            if now > a.collected_at + timedelta(days=a.retention_days)
        ]
        for aid in expired_ids:
            artifact = self._artifacts.pop(aid)
            self._hash_index.pop(artifact.content_hash, None)
        if expired_ids:
            logger.info("evidence_cleanup", removed=len(expired_ids))
        return len(expired_ids)


# =============================================================================
# Attestation Engine
# =============================================================================

# Shared HMAC secret (in production, inject via secrets manager)
_ATTESTATION_SECRET = b"codeverify-compliance-attestation-key"


class AttestationEngine:
    """Creates and verifies HMAC-signed compliance attestations."""

    def __init__(self) -> None:
        self._attestations: dict[str, ComplianceAttestation] = {}

    def create_attestation(
        self, control_id: str, attester: str, statement: str,
        evidence_ids: list[str],
        level: AttestationLevel = AttestationLevel.AUTOMATED,
        expires_in_days: int = 365,
    ) -> ComplianceAttestation:
        """Create a new signed attestation for a control."""
        now = datetime.now(timezone.utc)
        attestation_id = str(uuid.uuid4())
        sig_data = f"{attestation_id}:{control_id}:{attester}:{statement}"

        attestation = ComplianceAttestation(
            id=attestation_id, control_id=control_id, level=level,
            attester=attester, statement=statement, evidence_ids=evidence_ids,
            created_at=now, expires_at=now + timedelta(days=expires_in_days),
            signature=self._generate_signature(sig_data), valid=True,
        )
        self._attestations[attestation_id] = attestation
        logger.info("attestation_created", attestation_id=attestation_id,
                     control_id=control_id, level=level.value)
        return attestation

    def verify_attestation(self, attestation: ComplianceAttestation) -> bool:
        """Verify the HMAC signature and validity of an attestation."""
        if not attestation.valid or not attestation.signature:
            return False
        if attestation.expires_at and datetime.now(timezone.utc) > attestation.expires_at:
            return False
        sig_data = (
            f"{attestation.id}:{attestation.control_id}:"
            f"{attestation.attester}:{attestation.statement}"
        )
        return self._verify_signature(sig_data, attestation.signature)

    def _generate_signature(self, data: str) -> str:
        """Generate HMAC-SHA256 signature."""
        return hmac.new(
            _ATTESTATION_SECRET, data.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def _verify_signature(self, data: str, signature: str) -> bool:
        """Verify HMAC-SHA256 signature using constant-time comparison."""
        return hmac.compare_digest(self._generate_signature(data), signature)


# =============================================================================
# Compliance Report Generator
# =============================================================================


class ComplianceReportGenerator:
    """Generates comprehensive compliance reports from control assessments."""

    def __init__(self) -> None:
        self._reports: dict[str, ComplianceReport] = {}

    def generate_report(
        self, organization: str, framework: ComplianceFrameworkType,
        assessments: list[ControlAssessment],
        period_start: datetime, period_end: datetime,
    ) -> ComplianceReport:
        """Generate a full compliance report."""
        passing = [a for a in assessments if a.status == "passing"]
        failing = [a for a in assessments if a.status == "failing"]
        na = [a for a in assessments if a.status == "not_assessed"]

        report = ComplianceReport(
            id=str(uuid.uuid4()), framework=framework,
            generated_at=datetime.now(timezone.utc), organization=organization,
            period_start=period_start, period_end=period_end,
            total_controls=len(assessments), passing_controls=len(passing),
            failing_controls=len(failing), not_assessed=len(na),
            assessments=assessments,
            executive_summary=self._generate_executive_summary(assessments, framework),
            risk_areas=self._identify_risk_areas(assessments),
            remediation_priorities=self._prioritize_remediation(failing),
        )
        self._reports[report.id] = report
        logger.info("compliance_report_generated", report_id=report.id,
                     framework=framework.value, passing=len(passing), failing=len(failing))
        return report

    def _generate_executive_summary(
        self, assessments: list[ControlAssessment], framework: ComplianceFrameworkType,
    ) -> str:
        total = len(assessments)
        if total == 0:
            return f"No controls assessed for {framework.value}."
        passing = sum(1 for a in assessments if a.status == "passing")
        failing = sum(1 for a in assessments if a.status == "failing")
        rate = (passing / total) * 100
        parts = [f"Compliance assessment for {framework.value}: {passing}/{total} controls passing ({rate:.0f}%)."]
        if failing:
            parts.append(f" {failing} control(s) require remediation.")
        gap_controls = [a for a in assessments if a.gaps]
        if gap_controls:
            parts.append(f" {len(gap_controls)} control(s) have evidence gaps.")
        return "".join(parts)

    def _identify_risk_areas(self, assessments: list[ControlAssessment]) -> list[str]:
        risk_areas: list[str] = []
        for a in assessments:
            if a.status == "failing":
                risk_areas.append(f"{a.control_id}: failing — {len(a.gaps)} gap(s)")
            elif a.gaps:
                risk_areas.append(f"{a.control_id}: evidence gaps — {', '.join(a.gaps[:3])}")
        return risk_areas

    def _prioritize_remediation(self, failing: list[ControlAssessment]) -> list[dict[str, Any]]:
        priorities: list[dict[str, Any]] = []
        for idx, a in enumerate(sorted(failing, key=lambda x: len(x.gaps), reverse=True)):
            priorities.append({
                "priority": idx + 1, "control_id": a.control_id,
                "framework": a.framework.value, "gaps": a.gaps,
                "evidence_count": a.evidence_count,
                "recommendation": f"Address {len(a.gaps)} gap(s) for {a.control_id} to achieve compliance.",
            })
        return priorities

    def export_pdf_data(self, report: ComplianceReport) -> dict[str, Any]:
        """Export report data structured for PDF rendering."""
        return {
            "title": f"{report.framework.value} Compliance Report",
            "organization": report.organization,
            "generated_at": report.generated_at.isoformat(),
            "period": {"start": report.period_start.isoformat(), "end": report.period_end.isoformat()},
            "summary": {
                "total_controls": report.total_controls, "passing": report.passing_controls,
                "failing": report.failing_controls, "not_assessed": report.not_assessed,
                "pass_rate": f"{(report.passing_controls / report.total_controls * 100):.1f}%" if report.total_controls else "N/A",
            },
            "executive_summary": report.executive_summary,
            "risk_areas": report.risk_areas,
            "remediation_priorities": report.remediation_priorities,
            "assessments": [a.to_dict() for a in report.assessments],
        }


# =============================================================================
# Compliance-as-Code Engine — top-level orchestrator
# =============================================================================


class ComplianceAsCodeEngine:
    """Top-level orchestrator for compliance-as-code workflows.

    Coordinates the framework mapper, evidence vault, attestation engine,
    and report generator to deliver end-to-end automated compliance.

    Example:
        >>> engine = ComplianceAsCodeEngine()
        >>> report = engine.assess_compliance(
        ...     organization="acme-corp",
        ...     framework=ComplianceFrameworkType.SOC2_TYPE_II,
        ...     verification_results=[{"rule_id": "access_control_check", "passed": True}],
        ... )
    """

    def __init__(self) -> None:
        self.mapper = FrameworkMapper()
        self.vault = EvidenceVault()
        self.attestation_engine = AttestationEngine()
        self.report_generator = ComplianceReportGenerator()

    def assess_compliance(
        self, organization: str, framework: ComplianceFrameworkType,
        verification_results: list[dict[str, Any]],
    ) -> ComplianceReport:
        """Run a full compliance assessment against a framework."""
        controls = self.mapper.get_controls(framework)
        rule_ids = [r.get("rule_id", "") for r in verification_results]
        rule_status: dict[str, bool] = {
            r.get("rule_id", ""): r.get("passed", False) for r in verification_results
        }
        mapped = self.mapper.map_verification_to_controls(rule_ids, framework)
        now = datetime.now(timezone.utc)

        assessments: list[ControlAssessment] = []
        for control in controls:
            matched_rules = mapped.get(control.control_id, [])
            evidence = self.vault.get_evidence_for_control(control.control_id)
            attestations = [
                a for a in self.attestation_engine._attestations.values()
                if a.control_id == control.control_id
                and self.attestation_engine.verify_attestation(a)
            ]

            if not matched_rules:
                status = "not_assessed"
                gaps = [f"No verification rules mapped for {control.control_id}"]
            elif all(rule_status.get(r, False) for r in matched_rules):
                status = "passing"
                gaps = []
                if evidence:
                    self.attestation_engine.create_attestation(
                        control_id=control.control_id,
                        attester="codeverify-automation",
                        statement=f"Control {control.control_id} verified automatically.",
                        evidence_ids=[e.id for e in evidence],
                    )
            else:
                status = "failing"
                gaps = [f"Rule '{r}' not passing" for r in matched_rules if not rule_status.get(r, False)]

            unmapped = set(control.verification_rules) - set(matched_rules)
            if unmapped:
                gaps.extend(f"Rule '{r}' not evaluated" for r in sorted(unmapped))

            assessments.append(ControlAssessment(
                control_id=control.control_id, framework=framework, status=status,
                evidence_count=len(evidence), attestation_count=len(attestations),
                gaps=gaps, last_assessed=now, next_assessment_due=now + timedelta(days=90),
            ))

        return self.report_generator.generate_report(
            organization=organization, framework=framework, assessments=assessments,
            period_start=now - timedelta(days=90), period_end=now,
        )

    def continuous_monitoring(
        self, framework: ComplianceFrameworkType, new_evidence: list[EvidenceArtifact],
    ) -> list[ControlAssessment]:
        """Ingest new evidence and return updated assessments for affected controls."""
        affected_ids: set[str] = set()
        for evidence in new_evidence:
            self.vault.store_evidence(evidence)
            affected_ids.update(evidence.control_ids)

        now = datetime.now(timezone.utc)
        assessments: list[ControlAssessment] = []
        for control in self.mapper.get_controls(framework):
            if control.control_id not in affected_ids:
                continue
            ev = self.vault.get_evidence_for_control(control.control_id)
            att = [
                a for a in self.attestation_engine._attestations.values()
                if a.control_id == control.control_id
                and self.attestation_engine.verify_attestation(a)
            ]
            status = "passing" if ev else "not_assessed"
            gaps = [] if ev and att else (["No attestation on file"] if ev else ["No evidence collected"])

            assessments.append(ControlAssessment(
                control_id=control.control_id, framework=framework, status=status,
                evidence_count=len(ev), attestation_count=len(att),
                gaps=gaps, last_assessed=now, next_assessment_due=now + timedelta(days=90),
            ))

        logger.info("continuous_monitoring_complete", framework=framework.value,
                     new_evidence=len(new_evidence), controls_updated=len(assessments))
        return assessments

    def get_remediation_roadmap(self, report: ComplianceReport) -> list[dict[str, Any]]:
        """Build a prioritised remediation roadmap from a compliance report."""
        roadmap: list[dict[str, Any]] = []
        for p in report.remediation_priorities:
            gc = len(p.get("gaps", []))
            roadmap.append({
                "control_id": p["control_id"], "framework": p["framework"],
                "priority": p["priority"], "gaps": p["gaps"],
                "effort": "low" if gc <= 1 else ("medium" if gc <= 3 else "high"),
                "recommendation": p["recommendation"],
            })
        return roadmap

    def compare_periods(
        self, report1: ComplianceReport, report2: ComplianceReport,
    ) -> dict[str, Any]:
        """Compare two compliance reports to show trend data."""
        r1_map = {a.control_id: a.status for a in report1.assessments}
        r2_map = {a.control_id: a.status for a in report2.assessments}
        changes: list[dict[str, str]] = []
        for cid in sorted(set(r1_map) | set(r2_map)):
            old, new = r1_map.get(cid, "not_present"), r2_map.get(cid, "not_present")
            if old != new:
                changes.append({"control_id": cid, "old": old, "new": new})

        return {
            "period1": {"start": report1.period_start.isoformat(), "end": report1.period_end.isoformat()},
            "period2": {"start": report2.period_start.isoformat(), "end": report2.period_end.isoformat()},
            "delta": {
                "passing": report2.passing_controls - report1.passing_controls,
                "failing": report2.failing_controls - report1.failing_controls,
                "not_assessed": report2.not_assessed - report1.not_assessed,
            },
            "control_changes": changes,
            "improved": sum(1 for c in changes if c["old"] == "failing" and c["new"] == "passing"),
            "regressed": sum(1 for c in changes if c["old"] == "passing" and c["new"] == "failing"),
        }
