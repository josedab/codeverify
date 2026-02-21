"""Enterprise Compliance Framework — SOC2/HIPAA/PCI-DSS mapping and audit.

Maps verification rules to compliance controls, provides exception workflows
with role-based approvals, and generates audit-ready compliance reports.

.. deprecated::
    This module is superseded by ``codeverify_core.compliance_engine``.
    It remains importable for backward compatibility but will be
    removed in a future release.
"""

import warnings as _warnings
_warnings.warn(
    "codeverify_core.compliance_framework is deprecated. Use codeverify_core.compliance_engine instead.",
    DeprecationWarning,
    stacklevel=2,
)


import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ComplianceStandard(str, Enum):
    """Supported compliance standards."""

    SOC2 = "SOC2"
    HIPAA = "HIPAA"
    PCI_DSS = "PCI-DSS"
    ISO_27001 = "ISO-27001"
    GDPR = "GDPR"
    NIST_800_53 = "NIST-800-53"


class ControlStatus(str, Enum):
    """Status of a compliance control."""

    PASSING = "passing"
    FAILING = "failing"
    NOT_ASSESSED = "not_assessed"
    EXEMPTED = "exempted"
    PARTIALLY_PASSING = "partially_passing"


class ExceptionStatus(str, Enum):
    """Status of a compliance exception request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REVOKED = "revoked"


class Role(str, Enum):
    """Roles for RBAC enforcement."""

    DEVELOPER = "developer"
    REVIEWER = "reviewer"
    SECURITY_LEAD = "security_lead"
    COMPLIANCE_OFFICER = "compliance_officer"
    ADMIN = "admin"


@dataclass
class ComplianceControl:
    """A specific control within a compliance standard."""

    id: str
    standard: ComplianceStandard
    name: str
    description: str
    category: str
    verification_rules: list[str] = field(default_factory=list)
    evidence_types: list[str] = field(default_factory=list)
    status: ControlStatus = ControlStatus.NOT_ASSESSED
    last_assessed: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "standard": self.standard.value,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "status": self.status.value,
            "verification_rules": self.verification_rules,
            "last_assessed": self.last_assessed.isoformat() if self.last_assessed else None,
        }


@dataclass
class ComplianceException:
    """An approved exception to a compliance control."""

    id: str
    control_id: str
    repository: str
    justification: str
    requestor: str
    requestor_role: Role
    approver: str | None = None
    approver_role: Role | None = None
    status: ExceptionStatus = ExceptionStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    approval_comment: str = ""
    file_patterns: list[str] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        if self.status != ExceptionStatus.APPROVED:
            return False
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "control_id": self.control_id,
            "repository": self.repository,
            "justification": self.justification,
            "requestor": self.requestor,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
        }


@dataclass
class EvidenceArtifact:
    """An evidence artifact for compliance audit trail."""

    id: str
    control_id: str
    repository: str
    artifact_type: str  # "verification_result", "scan_report", "approval_record"
    content: dict[str, Any] = field(default_factory=dict)
    collected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    collector: str = "codeverify"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "control_id": self.control_id,
            "artifact_type": self.artifact_type,
            "collected_at": self.collected_at.isoformat(),
            "collector": self.collector,
        }


@dataclass
class AuditReport:
    """Generated audit report for compliance review."""

    id: str
    standard: ComplianceStandard
    organization: str
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    controls: list[ComplianceControl] = field(default_factory=list)
    exceptions: list[ComplianceException] = field(default_factory=list)
    evidence: list[EvidenceArtifact] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        assessed = [c for c in self.controls if c.status != ControlStatus.NOT_ASSESSED]
        if not assessed:
            return 0.0
        passing = sum(
            1 for c in assessed
            if c.status in (ControlStatus.PASSING, ControlStatus.EXEMPTED)
        )
        return passing / len(assessed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "standard": self.standard.value,
            "organization": self.organization,
            "generated_at": self.generated_at.isoformat(),
            "summary": {
                "total_controls": len(self.controls),
                "passing": sum(1 for c in self.controls if c.status == ControlStatus.PASSING),
                "failing": sum(1 for c in self.controls if c.status == ControlStatus.FAILING),
                "exempted": sum(1 for c in self.controls if c.status == ControlStatus.EXEMPTED),
                "pass_rate": f"{self.pass_rate:.1%}",
                "active_exceptions": sum(1 for e in self.exceptions if e.is_active),
                "evidence_artifacts": len(self.evidence),
            },
            "controls": [c.to_dict() for c in self.controls],
            "exceptions": [e.to_dict() for e in self.exceptions],
        }


# --- Built-in Compliance Control Mappings ---

def _soc2_controls() -> list[ComplianceControl]:
    return [
        ComplianceControl(
            id="CC6.1", standard=ComplianceStandard.SOC2,
            name="Logical Access Security",
            description="Restrict logical access to information assets",
            category="Logical and Physical Access Controls",
            verification_rules=["access_control_check", "auth_verification"],
            evidence_types=["verification_result", "access_log"],
        ),
        ComplianceControl(
            id="CC7.2", standard=ComplianceStandard.SOC2,
            name="System Monitoring",
            description="Monitor system components for anomalies",
            category="System Operations",
            verification_rules=["security_scan", "vulnerability_detection"],
            evidence_types=["scan_report", "finding_log"],
        ),
        ComplianceControl(
            id="CC8.1", standard=ComplianceStandard.SOC2,
            name="Change Management",
            description="Changes to infrastructure and software are authorized",
            category="Change Management",
            verification_rules=["code_review_required", "approval_workflow"],
            evidence_types=["approval_record", "pr_review"],
        ),
        ComplianceControl(
            id="CC6.6", standard=ComplianceStandard.SOC2,
            name="Boundary Protection",
            description="Protect system boundaries from unauthorized access",
            category="Logical and Physical Access Controls",
            verification_rules=["input_validation", "injection_check"],
            evidence_types=["verification_result", "security_scan"],
        ),
        ComplianceControl(
            id="CC6.8", standard=ComplianceStandard.SOC2,
            name="Software Integrity",
            description="Prevent or detect unauthorized software deployment",
            category="Logical and Physical Access Controls",
            verification_rules=["dependency_scan", "supply_chain_check"],
            evidence_types=["sbom_report", "dependency_audit"],
        ),
    ]


def _hipaa_controls() -> list[ComplianceControl]:
    return [
        ComplianceControl(
            id="164.312(a)(1)", standard=ComplianceStandard.HIPAA,
            name="Access Control",
            description="Implement access controls for ePHI systems",
            category="Technical Safeguards",
            verification_rules=["access_control_check", "auth_verification", "encryption_check"],
            evidence_types=["verification_result", "access_log"],
        ),
        ComplianceControl(
            id="164.312(a)(2)(iv)", standard=ComplianceStandard.HIPAA,
            name="Encryption and Decryption",
            description="Encrypt and decrypt ePHI",
            category="Technical Safeguards",
            verification_rules=["encryption_check", "tls_verification"],
            evidence_types=["verification_result", "config_audit"],
        ),
        ComplianceControl(
            id="164.312(b)", standard=ComplianceStandard.HIPAA,
            name="Audit Controls",
            description="Record and examine system activity",
            category="Technical Safeguards",
            verification_rules=["logging_check", "audit_trail"],
            evidence_types=["log_analysis", "audit_config"],
        ),
        ComplianceControl(
            id="164.312(c)(1)", standard=ComplianceStandard.HIPAA,
            name="Integrity",
            description="Protect ePHI from improper alteration or destruction",
            category="Technical Safeguards",
            verification_rules=["input_validation", "data_integrity_check"],
            evidence_types=["verification_result"],
        ),
    ]


def _pci_dss_controls() -> list[ComplianceControl]:
    return [
        ComplianceControl(
            id="6.5.1", standard=ComplianceStandard.PCI_DSS,
            name="Injection Flaws",
            description="Protect against injection flaws (SQL, OS, LDAP)",
            category="Secure Development",
            verification_rules=["injection_check", "input_validation"],
            evidence_types=["scan_report", "verification_result"],
        ),
        ComplianceControl(
            id="6.5.3", standard=ComplianceStandard.PCI_DSS,
            name="Insecure Cryptographic Storage",
            description="Prevent insecure cryptographic storage",
            category="Secure Development",
            verification_rules=["encryption_check", "secret_detection"],
            evidence_types=["scan_report"],
        ),
        ComplianceControl(
            id="6.5.7", standard=ComplianceStandard.PCI_DSS,
            name="Cross-Site Scripting",
            description="Prevent cross-site scripting (XSS) vulnerabilities",
            category="Secure Development",
            verification_rules=["xss_check", "output_encoding"],
            evidence_types=["scan_report", "verification_result"],
        ),
        ComplianceControl(
            id="6.5.10", standard=ComplianceStandard.PCI_DSS,
            name="Broken Authentication",
            description="Prevent broken authentication and session management",
            category="Secure Development",
            verification_rules=["auth_verification", "session_check"],
            evidence_types=["scan_report"],
        ),
    ]


STANDARD_CONTROLS: dict[ComplianceStandard, list[ComplianceControl]] = {
    ComplianceStandard.SOC2: _soc2_controls(),
    ComplianceStandard.HIPAA: _hipaa_controls(),
    ComplianceStandard.PCI_DSS: _pci_dss_controls(),
}


# --- Role Permission Matrix ---

ROLE_PERMISSIONS: dict[Role, set[str]] = {
    Role.DEVELOPER: {"view_controls", "request_exception", "view_reports"},
    Role.REVIEWER: {"view_controls", "request_exception", "view_reports", "review_findings"},
    Role.SECURITY_LEAD: {
        "view_controls", "request_exception", "view_reports",
        "review_findings", "approve_exception",
    },
    Role.COMPLIANCE_OFFICER: {
        "view_controls", "request_exception", "view_reports",
        "review_findings", "approve_exception", "generate_report",
        "manage_controls",
    },
    Role.ADMIN: {
        "view_controls", "request_exception", "view_reports",
        "review_findings", "approve_exception", "generate_report",
        "manage_controls", "manage_roles",
    },
}


class ComplianceFramework:
    """Enterprise compliance engine with RBAC, exceptions, and audit reporting.

    Example:
        >>> framework = ComplianceFramework(organization="acme-corp")
        >>> framework.load_standard(ComplianceStandard.SOC2)
        >>> results = framework.assess("my-repo", verification_results)
        >>> report = framework.generate_report(ComplianceStandard.SOC2, "admin")
    """

    def __init__(self, organization: str) -> None:
        self.organization = organization
        self._controls: dict[str, ComplianceControl] = {}
        self._exceptions: dict[str, ComplianceException] = {}
        self._evidence: list[EvidenceArtifact] = []
        self._user_roles: dict[str, Role] = {}

    def load_standard(self, standard: ComplianceStandard) -> int:
        """Load all controls for a compliance standard. Returns count loaded."""
        controls = STANDARD_CONTROLS.get(standard, [])
        for control in controls:
            self._controls[control.id] = control
        logger.info(
            "Compliance standard loaded",
            standard=standard.value,
            controls=len(controls),
        )
        return len(controls)

    def set_user_role(self, user: str, role: Role) -> None:
        """Assign a role to a user for RBAC."""
        self._user_roles[user] = role

    def check_permission(self, user: str, permission: str) -> bool:
        """Check if a user has a specific permission."""
        role = self._user_roles.get(user, Role.DEVELOPER)
        return permission in ROLE_PERMISSIONS.get(role, set())

    def assess(
        self,
        repository: str,
        verification_results: list[dict[str, Any]],
    ) -> dict[str, ControlStatus]:
        """Assess compliance controls against verification results.

        Args:
            repository: Repository being assessed
            verification_results: List of verification finding dicts with
                'rule_id', 'severity', 'passed' keys.

        Returns:
            Dict mapping control_id to assessment status.
        """
        rule_results: dict[str, bool] = {}
        for result in verification_results:
            rule_id = result.get("rule_id", "")
            passed = result.get("passed", False)
            rule_results[rule_id] = rule_results.get(rule_id, True) and passed

        statuses: dict[str, ControlStatus] = {}
        now = datetime.now(timezone.utc)

        for control_id, control in self._controls.items():
            # Check if exempted
            active_exceptions = [
                e for e in self._exceptions.values()
                if e.control_id == control_id
                and e.repository == repository
                and e.is_active
            ]
            if active_exceptions:
                control.status = ControlStatus.EXEMPTED
                control.last_assessed = now
                statuses[control_id] = ControlStatus.EXEMPTED
                continue

            # Evaluate against verification rules
            if not control.verification_rules:
                control.status = ControlStatus.NOT_ASSESSED
                statuses[control_id] = ControlStatus.NOT_ASSESSED
                continue

            rule_statuses = [
                rule_results.get(rule, True)
                for rule in control.verification_rules
            ]
            assessed_rules = [r for r in control.verification_rules if r in rule_results]

            if not assessed_rules:
                control.status = ControlStatus.NOT_ASSESSED
            elif all(rule_statuses):
                control.status = ControlStatus.PASSING
            elif any(rule_statuses):
                control.status = ControlStatus.PARTIALLY_PASSING
            else:
                control.status = ControlStatus.FAILING

            control.last_assessed = now
            statuses[control_id] = control.status

            # Collect evidence
            self._evidence.append(
                EvidenceArtifact(
                    id=str(uuid.uuid4()),
                    control_id=control_id,
                    repository=repository,
                    artifact_type="assessment_result",
                    content={
                        "status": control.status.value,
                        "rules_evaluated": assessed_rules,
                        "rules_passed": [r for r, s in zip(control.verification_rules, rule_statuses) if s],
                    },
                )
            )

        logger.info(
            "Compliance assessment complete",
            repository=repository,
            controls_assessed=len(statuses),
            passing=sum(1 for s in statuses.values() if s == ControlStatus.PASSING),
        )
        return statuses

    def request_exception(
        self,
        user: str,
        control_id: str,
        repository: str,
        justification: str,
        expiry_days: int = 30,
        file_patterns: list[str] | None = None,
    ) -> ComplianceException | None:
        """Request an exception to a compliance control."""
        if not self.check_permission(user, "request_exception"):
            logger.warning("Permission denied for exception request", user=user)
            return None

        if control_id not in self._controls:
            logger.warning("Unknown control for exception", control_id=control_id)
            return None

        from datetime import timedelta

        exception = ComplianceException(
            id=str(uuid.uuid4()),
            control_id=control_id,
            repository=repository,
            justification=justification,
            requestor=user,
            requestor_role=self._user_roles.get(user, Role.DEVELOPER),
            expires_at=datetime.now(timezone.utc) + timedelta(days=expiry_days),
            file_patterns=file_patterns or [],
        )
        self._exceptions[exception.id] = exception
        logger.info(
            "Compliance exception requested",
            exception_id=exception.id,
            control_id=control_id,
            requestor=user,
        )
        return exception

    def approve_exception(
        self,
        user: str,
        exception_id: str,
        approved: bool,
        comment: str = "",
    ) -> ComplianceException | None:
        """Approve or reject an exception request."""
        if not self.check_permission(user, "approve_exception"):
            logger.warning("Permission denied for exception approval", user=user)
            return None

        exception = self._exceptions.get(exception_id)
        if not exception or exception.status != ExceptionStatus.PENDING:
            return None

        exception.approver = user
        exception.approver_role = self._user_roles.get(user)
        exception.approval_comment = comment

        if approved:
            exception.status = ExceptionStatus.APPROVED
        else:
            exception.status = ExceptionStatus.REJECTED

        # Record evidence
        self._evidence.append(
            EvidenceArtifact(
                id=str(uuid.uuid4()),
                control_id=exception.control_id,
                repository=exception.repository,
                artifact_type="exception_decision",
                content={
                    "exception_id": exception_id,
                    "decision": "approved" if approved else "rejected",
                    "approver": user,
                    "comment": comment,
                },
            )
        )

        logger.info(
            "Exception decision recorded",
            exception_id=exception_id,
            decision="approved" if approved else "rejected",
            approver=user,
        )
        return exception

    def generate_report(
        self,
        standard: ComplianceStandard,
        user: str,
    ) -> AuditReport | None:
        """Generate a compliance audit report."""
        if not self.check_permission(user, "generate_report"):
            logger.warning("Permission denied for report generation", user=user)
            return None

        controls = [
            c for c in self._controls.values() if c.standard == standard
        ]
        exceptions = [
            e for e in self._exceptions.values()
            if self._controls.get(e.control_id, ComplianceControl(
                id="", standard=standard, name="", description="", category=""
            )).standard == standard
        ]
        evidence = [
            e for e in self._evidence
            if e.control_id in {c.id for c in controls}
        ]

        report = AuditReport(
            id=str(uuid.uuid4()),
            standard=standard,
            organization=self.organization,
            controls=controls,
            exceptions=exceptions,
            evidence=evidence,
        )

        logger.info(
            "Compliance report generated",
            standard=standard.value,
            controls=len(controls),
            pass_rate=f"{report.pass_rate:.1%}",
        )
        return report

    def get_control_status(self) -> dict[str, Any]:
        """Get overview of all control statuses."""
        by_standard: dict[str, dict[str, int]] = {}
        for control in self._controls.values():
            std = control.standard.value
            if std not in by_standard:
                by_standard[std] = {}
            status = control.status.value
            by_standard[std][status] = by_standard[std].get(status, 0) + 1
        return by_standard
