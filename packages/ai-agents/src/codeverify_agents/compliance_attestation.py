"""Compliance Attestation Engine - Auto-generate compliance reports from verification results."""

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from codeverify_agents.base import AgentConfig, AgentResult, BaseAgent

logger = structlog.get_logger()


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""
    SOC2 = "soc2"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    GDPR = "gdpr"
    ISO_27001 = "iso_27001"
    NIST_CSF = "nist_csf"
    CIS = "cis"
    OWASP = "owasp"


class ControlStatus(str, Enum):
    """Status of a compliance control."""
    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    NOT_APPLICABLE = "not_applicable"
    NEEDS_REVIEW = "needs_review"


@dataclass
class EvidenceItem:
    """A piece of evidence supporting compliance."""
    evidence_id: str
    evidence_type: str  # verification_result, code_scan, test_result, documentation
    source: str  # File path, API endpoint, etc.
    timestamp: datetime
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)  # Links to artifacts


@dataclass
class ControlMapping:
    """Mapping between a control and verification results."""
    control_id: str
    control_name: str
    description: str
    status: ControlStatus
    evidence: list[EvidenceItem] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    verification_coverage: float = 0.0  # 0-100%
    manual_review_needed: bool = False


@dataclass
class ComplianceReport:
    """Complete compliance report for a framework."""
    report_id: str
    framework: ComplianceFramework
    scope: str  # What was assessed
    generated_at: datetime
    generated_by: str
    
    # Overall status
    overall_status: ControlStatus = ControlStatus.NEEDS_REVIEW
    compliance_score: float = 0.0  # 0-100
    
    # Control details
    controls: list[ControlMapping] = field(default_factory=list)
    
    # Summary statistics
    total_controls: int = 0
    compliant_controls: int = 0
    partial_controls: int = 0
    non_compliant_controls: int = 0
    not_applicable_controls: int = 0
    
    # Audit trail
    audit_log: list[dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    version: str = "1.0"
    valid_until: datetime | None = None
    attestation_signature: str | None = None


# Control definitions for each framework
FRAMEWORK_CONTROLS: dict[ComplianceFramework, list[dict[str, Any]]] = {
    ComplianceFramework.SOC2: [
        {
            "id": "CC6.1",
            "name": "Logical and Physical Access Controls",
            "description": "The entity implements logical access security software, infrastructure, and architectures over protected information assets.",
            "verification_mappings": ["access_control", "authentication"],
        },
        {
            "id": "CC6.2",
            "name": "Authentication Mechanisms",
            "description": "Prior to issuing system credentials and granting system access, the entity registers and authorizes new internal and external users.",
            "verification_mappings": ["authentication", "authorization"],
        },
        {
            "id": "CC6.6",
            "name": "Logical Access Security Measures",
            "description": "The entity implements logical access security measures to protect against threats from sources outside its system boundaries.",
            "verification_mappings": ["input_validation", "security_boundaries"],
        },
        {
            "id": "CC6.7",
            "name": "Data Transmission Protection",
            "description": "The entity restricts the transmission, movement, and removal of information to authorized internal and external users.",
            "verification_mappings": ["encryption", "data_handling"],
        },
        {
            "id": "CC7.1",
            "name": "Security Monitoring",
            "description": "The entity detects, monitors, and measures the effectiveness of security activities.",
            "verification_mappings": ["logging", "monitoring"],
        },
        {
            "id": "CC7.2",
            "name": "Vulnerability Management",
            "description": "The entity monitors system components and the operation of those components for anomalies.",
            "verification_mappings": ["vulnerability_scan", "code_analysis"],
        },
        {
            "id": "CC8.1",
            "name": "Change Management",
            "description": "The entity authorizes, designs, develops or acquires, configures, documents, tests, approves, and implements changes.",
            "verification_mappings": ["code_review", "testing"],
        },
    ],
    ComplianceFramework.HIPAA: [
        {
            "id": "164.312(a)(1)",
            "name": "Access Control",
            "description": "Implement technical policies and procedures for electronic information systems that maintain ePHI.",
            "verification_mappings": ["access_control", "authentication", "authorization"],
        },
        {
            "id": "164.312(a)(2)(i)",
            "name": "Unique User Identification",
            "description": "Assign a unique name and/or number for identifying and tracking user identity.",
            "verification_mappings": ["user_identification"],
        },
        {
            "id": "164.312(b)",
            "name": "Audit Controls",
            "description": "Implement hardware, software, and/or procedural mechanisms that record and examine activity.",
            "verification_mappings": ["audit_logging", "monitoring"],
        },
        {
            "id": "164.312(c)(1)",
            "name": "Integrity Controls",
            "description": "Implement policies and procedures to protect ePHI from improper alteration or destruction.",
            "verification_mappings": ["data_integrity", "validation"],
        },
        {
            "id": "164.312(d)",
            "name": "Person or Entity Authentication",
            "description": "Implement procedures to verify that a person or entity seeking access to ePHI is the one claimed.",
            "verification_mappings": ["authentication", "identity_verification"],
        },
        {
            "id": "164.312(e)(1)",
            "name": "Transmission Security",
            "description": "Implement technical security measures to guard against unauthorized access to ePHI during transmission.",
            "verification_mappings": ["encryption", "secure_transmission"],
        },
    ],
    ComplianceFramework.PCI_DSS: [
        {
            "id": "6.3",
            "name": "Secure Software Development",
            "description": "Develop software applications in accordance with PCI DSS and industry best practices.",
            "verification_mappings": ["secure_coding", "code_review"],
        },
        {
            "id": "6.4",
            "name": "Change Control Processes",
            "description": "Follow change control processes and procedures for all changes to system components.",
            "verification_mappings": ["change_management", "testing"],
        },
        {
            "id": "6.5",
            "name": "Common Coding Vulnerabilities",
            "description": "Address common coding vulnerabilities in software-development processes.",
            "verification_mappings": ["injection_prevention", "xss_prevention", "buffer_overflow"],
        },
        {
            "id": "6.6",
            "name": "Application Security",
            "description": "For public-facing web applications, address new threats and vulnerabilities on an ongoing basis.",
            "verification_mappings": ["vulnerability_scan", "penetration_testing"],
        },
        {
            "id": "8.1",
            "name": "User Identification",
            "description": "Define and implement policies and procedures to ensure proper user identification management.",
            "verification_mappings": ["user_management", "authentication"],
        },
        {
            "id": "10.1",
            "name": "Audit Trail",
            "description": "Implement audit trails to link all access to system components to each individual user.",
            "verification_mappings": ["audit_logging", "user_tracking"],
        },
    ],
}


class ComplianceAttestationEngine(BaseAgent):
    """
    Engine for generating compliance attestation reports from verification results.
    
    Maps CodeVerify verification findings to compliance framework controls
    and generates audit-ready reports with evidence artifacts.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize compliance attestation engine."""
        super().__init__(config)
        self._verification_category_mapping: dict[str, list[str]] = {
            "security": ["access_control", "authentication", "authorization", "encryption"],
            "injection": ["injection_prevention", "input_validation"],
            "null_safety": ["data_integrity", "validation"],
            "bounds_check": ["buffer_overflow", "data_integrity"],
            "authentication": ["authentication", "user_identification", "identity_verification"],
            "authorization": ["authorization", "access_control"],
            "logging": ["audit_logging", "monitoring", "user_tracking"],
            "cryptography": ["encryption", "secure_transmission"],
            "data_handling": ["data_handling", "data_integrity"],
            "code_quality": ["secure_coding", "code_review"],
        }

    async def analyze(self, code: str, context: dict[str, Any]) -> AgentResult:
        """
        Generate compliance attestation from verification results.

        Args:
            code: Not used directly - compliance is based on verification results
            context: Must include:
                - framework: ComplianceFramework to assess
                - verification_results: List of verification result dicts
                - scope: Description of what's being assessed
                - organization: Organization name

        Returns:
            AgentResult with compliance report
        """
        start_time = time.time()
        
        framework = context.get("framework", ComplianceFramework.SOC2)
        if isinstance(framework, str):
            framework = ComplianceFramework(framework)
        
        verification_results = context.get("verification_results", [])
        scope = context.get("scope", "Code verification assessment")
        organization = context.get("organization", "Unknown")
        
        try:
            report = await self._generate_report(
                framework=framework,
                verification_results=verification_results,
                scope=scope,
                organization=organization,
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            logger.info(
                "Compliance report generated",
                framework=framework.value,
                score=report.compliance_score,
                controls=report.total_controls,
                latency_ms=elapsed_ms,
            )
            
            return AgentResult(
                success=True,
                data=self._report_to_dict(report),
                latency_ms=elapsed_ms,
            )
            
        except Exception as e:
            logger.error("Compliance report generation failed", error=str(e))
            return AgentResult(
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
            )

    async def _generate_report(
        self,
        framework: ComplianceFramework,
        verification_results: list[dict[str, Any]],
        scope: str,
        organization: str,
    ) -> ComplianceReport:
        """Generate a compliance report."""
        report_id = self._generate_report_id(framework, scope)
        
        # Get controls for framework
        controls_def = FRAMEWORK_CONTROLS.get(framework, [])
        
        # Map verification results to controls
        controls = []
        for control_def in controls_def:
            control = await self._assess_control(
                control_def=control_def,
                verification_results=verification_results,
            )
            controls.append(control)
        
        # Calculate overall statistics
        total = len(controls)
        compliant = sum(1 for c in controls if c.status == ControlStatus.COMPLIANT)
        partial = sum(1 for c in controls if c.status == ControlStatus.PARTIAL)
        non_compliant = sum(1 for c in controls if c.status == ControlStatus.NON_COMPLIANT)
        not_applicable = sum(1 for c in controls if c.status == ControlStatus.NOT_APPLICABLE)
        
        # Calculate compliance score
        if total - not_applicable > 0:
            score = (
                (compliant * 100 + partial * 50) / 
                ((total - not_applicable) * 100)
            ) * 100
        else:
            score = 0.0
        
        # Determine overall status
        if score >= 90:
            overall_status = ControlStatus.COMPLIANT
        elif score >= 70:
            overall_status = ControlStatus.PARTIAL
        else:
            overall_status = ControlStatus.NON_COMPLIANT
        
        return ComplianceReport(
            report_id=report_id,
            framework=framework,
            scope=scope,
            generated_at=datetime.utcnow(),
            generated_by=f"CodeVerify Compliance Engine for {organization}",
            overall_status=overall_status,
            compliance_score=round(score, 1),
            controls=controls,
            total_controls=total,
            compliant_controls=compliant,
            partial_controls=partial,
            non_compliant_controls=non_compliant,
            not_applicable_controls=not_applicable,
            audit_log=[
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": "report_generated",
                    "details": f"Compliance report generated for {framework.value}",
                }
            ],
        )

    async def _assess_control(
        self,
        control_def: dict[str, Any],
        verification_results: list[dict[str, Any]],
    ) -> ControlMapping:
        """Assess a single control against verification results."""
        control_id = control_def["id"]
        control_name = control_def["name"]
        description = control_def["description"]
        verification_mappings = control_def.get("verification_mappings", [])
        
        # Find relevant verification results
        evidence = []
        relevant_findings = []
        total_coverage = 0
        
        for result in verification_results:
            result_categories = self._get_result_categories(result)
            
            # Check if this result is relevant to the control
            is_relevant = any(
                mapping in self._expand_categories(result_categories)
                for mapping in verification_mappings
            )
            
            if is_relevant:
                relevant_findings.append(result)
                
                # Create evidence item
                evidence.append(EvidenceItem(
                    evidence_id=f"ev_{control_id}_{len(evidence)}",
                    evidence_type="verification_result",
                    source=result.get("file_path", "unknown"),
                    timestamp=datetime.utcnow(),
                    summary=result.get("summary", "Verification completed"),
                    details={
                        "status": result.get("status", "unknown"),
                        "findings_count": len(result.get("findings", [])),
                        "verified_properties": result.get("verified_properties", []),
                    },
                ))
                
                # Calculate coverage contribution
                if result.get("status") == "verified":
                    total_coverage += 100
                elif result.get("status") == "partial":
                    total_coverage += 50
        
        # Calculate verification coverage
        verification_coverage = (
            total_coverage / len(relevant_findings)
            if relevant_findings else 0
        )
        
        # Determine control status
        if not relevant_findings:
            status = ControlStatus.NEEDS_REVIEW
            gaps = ["No verification evidence found for this control"]
            recommendations = [
                f"Add verification tests covering: {', '.join(verification_mappings)}"
            ]
            manual_review = True
        elif verification_coverage >= 80:
            status = ControlStatus.COMPLIANT
            gaps = []
            recommendations = []
            manual_review = False
        elif verification_coverage >= 50:
            status = ControlStatus.PARTIAL
            gaps = self._identify_gaps(verification_mappings, relevant_findings)
            recommendations = self._generate_recommendations(gaps)
            manual_review = True
        else:
            status = ControlStatus.NON_COMPLIANT
            gaps = self._identify_gaps(verification_mappings, relevant_findings)
            recommendations = self._generate_recommendations(gaps)
            manual_review = True
        
        return ControlMapping(
            control_id=control_id,
            control_name=control_name,
            description=description,
            status=status,
            evidence=evidence,
            gaps=gaps,
            recommendations=recommendations,
            verification_coverage=verification_coverage,
            manual_review_needed=manual_review,
        )

    def _get_result_categories(self, result: dict[str, Any]) -> list[str]:
        """Extract categories from a verification result."""
        categories = []
        
        # From findings
        for finding in result.get("findings", []):
            if "category" in finding:
                categories.append(finding["category"])
        
        # From verified properties
        for prop in result.get("verified_properties", []):
            if isinstance(prop, str):
                categories.append(prop)
            elif isinstance(prop, dict) and "type" in prop:
                categories.append(prop["type"])
        
        # From result type
        if "type" in result:
            categories.append(result["type"])
        
        return list(set(categories))

    def _expand_categories(self, categories: list[str]) -> list[str]:
        """Expand categories to all related compliance mappings."""
        expanded = []
        for cat in categories:
            expanded.append(cat)
            if cat in self._verification_category_mapping:
                expanded.extend(self._verification_category_mapping[cat])
        return list(set(expanded))

    def _identify_gaps(
        self,
        verification_mappings: list[str],
        findings: list[dict[str, Any]],
    ) -> list[str]:
        """Identify gaps in verification coverage."""
        covered = set()
        for finding in findings:
            categories = self._get_result_categories(finding)
            covered.update(self._expand_categories(categories))
        
        gaps = []
        for mapping in verification_mappings:
            if mapping not in covered:
                gaps.append(f"Missing verification for: {mapping}")
        
        return gaps

    def _generate_recommendations(self, gaps: list[str]) -> list[str]:
        """Generate recommendations based on gaps."""
        recommendations = []
        
        for gap in gaps:
            if "authentication" in gap.lower():
                recommendations.append(
                    "Add authentication verification tests covering credential handling"
                )
            elif "encryption" in gap.lower():
                recommendations.append(
                    "Add cryptography verification for data at rest and in transit"
                )
            elif "logging" in gap.lower() or "audit" in gap.lower():
                recommendations.append(
                    "Implement audit logging verification for security events"
                )
            elif "injection" in gap.lower():
                recommendations.append(
                    "Add input validation and injection prevention verification"
                )
            else:
                recommendations.append(f"Address gap: {gap}")
        
        return recommendations

    def _generate_report_id(
        self, framework: ComplianceFramework, scope: str
    ) -> str:
        """Generate a unique report ID."""
        content = f"{framework.value}:{scope}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _report_to_dict(self, report: ComplianceReport) -> dict[str, Any]:
        """Convert ComplianceReport to dictionary."""
        return {
            "report_id": report.report_id,
            "framework": report.framework.value,
            "scope": report.scope,
            "generated_at": report.generated_at.isoformat(),
            "generated_by": report.generated_by,
            "overall_status": report.overall_status.value,
            "compliance_score": report.compliance_score,
            "summary": {
                "total_controls": report.total_controls,
                "compliant": report.compliant_controls,
                "partial": report.partial_controls,
                "non_compliant": report.non_compliant_controls,
                "not_applicable": report.not_applicable_controls,
            },
            "controls": [
                {
                    "id": c.control_id,
                    "name": c.control_name,
                    "description": c.description,
                    "status": c.status.value,
                    "verification_coverage": round(c.verification_coverage, 1),
                    "manual_review_needed": c.manual_review_needed,
                    "evidence_count": len(c.evidence),
                    "gaps": c.gaps,
                    "recommendations": c.recommendations,
                }
                for c in report.controls
            ],
            "audit_log": report.audit_log,
            "version": report.version,
        }

    async def generate_multi_framework_report(
        self,
        frameworks: list[ComplianceFramework],
        verification_results: list[dict[str, Any]],
        scope: str,
        organization: str,
    ) -> dict[str, Any]:
        """Generate compliance reports for multiple frameworks."""
        reports = {}
        
        for framework in frameworks:
            result = await self.analyze(
                code="",
                context={
                    "framework": framework,
                    "verification_results": verification_results,
                    "scope": scope,
                    "organization": organization,
                },
            )
            
            if result.success:
                reports[framework.value] = result.data
        
        # Generate cross-framework summary
        summary = {
            "organization": organization,
            "scope": scope,
            "generated_at": datetime.utcnow().isoformat(),
            "frameworks_assessed": len(reports),
            "reports": reports,
            "overall_summary": self._generate_cross_framework_summary(reports),
        }
        
        return summary

    def _generate_cross_framework_summary(
        self, reports: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate summary across all frameworks."""
        total_controls = 0
        total_compliant = 0
        total_gaps = []
        
        for framework, report in reports.items():
            summary = report.get("summary", {})
            total_controls += summary.get("total_controls", 0)
            total_compliant += summary.get("compliant", 0)
            
            for control in report.get("controls", []):
                total_gaps.extend(control.get("gaps", []))
        
        return {
            "total_controls_assessed": total_controls,
            "total_compliant": total_compliant,
            "overall_compliance_rate": (
                round(total_compliant / total_controls * 100, 1)
                if total_controls > 0 else 0
            ),
            "unique_gaps": list(set(total_gaps)),
            "gap_count": len(set(total_gaps)),
        }

    def get_supported_frameworks(self) -> list[str]:
        """Get list of supported compliance frameworks."""
        return [f.value for f in ComplianceFramework]

    def get_framework_controls(
        self, framework: ComplianceFramework
    ) -> list[dict[str, Any]]:
        """Get controls for a specific framework."""
        return FRAMEWORK_CONTROLS.get(framework, [])

    async def generate_attestation_certificate(
        self,
        report: ComplianceReport,
        signatory: str,
        attestation_statement: str | None = None,
    ) -> dict[str, Any]:
        """Generate a formal attestation certificate."""
        if attestation_statement is None:
            attestation_statement = (
                f"This is to certify that {report.scope} has been assessed against "
                f"{report.framework.value} controls and achieved a compliance score of "
                f"{report.compliance_score}%."
            )
        
        certificate = {
            "certificate_id": f"CERT-{report.report_id}",
            "type": "compliance_attestation",
            "framework": report.framework.value,
            "scope": report.scope,
            "assessment_date": report.generated_at.isoformat(),
            "compliance_score": report.compliance_score,
            "overall_status": report.overall_status.value,
            "attestation_statement": attestation_statement,
            "signatory": signatory,
            "issued_at": datetime.utcnow().isoformat(),
            "valid_until": None,  # Would be set based on policy
            "controls_summary": {
                "total": report.total_controls,
                "compliant": report.compliant_controls,
                "requires_remediation": report.non_compliant_controls + report.partial_controls,
            },
            "signature_placeholder": "DIGITAL_SIGNATURE_REQUIRED",
        }
        
        # Generate signature hash
        content_for_signature = json.dumps(certificate, sort_keys=True)
        certificate["content_hash"] = hashlib.sha256(
            content_for_signature.encode()
        ).hexdigest()
        
        return certificate
