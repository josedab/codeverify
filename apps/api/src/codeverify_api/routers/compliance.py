"""Compliance Attestation API endpoints."""

from typing import Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter()


class VerificationResultInput(BaseModel):
    """A verification result for compliance mapping."""

    file_path: str
    status: str = Field(..., pattern="^(verified|partial|failed)$")
    findings: list[dict[str, Any]] = Field(default_factory=list)
    verified_properties: list[str] = Field(default_factory=list)


class ComplianceReportRequest(BaseModel):
    """Request to generate a compliance report."""

    framework: str = Field(
        ..., description="Compliance framework: soc2, hipaa, pci_dss, gdpr, iso_27001"
    )
    verification_results: list[VerificationResultInput]
    scope: str = Field(..., description="Description of what's being assessed")
    organization: str = Field(..., description="Organization name")


class ControlResponse(BaseModel):
    """A compliance control assessment."""

    id: str
    name: str
    description: str
    status: str
    verification_coverage: float
    manual_review_needed: bool
    evidence_count: int
    gaps: list[str]
    recommendations: list[str]


class ComplianceReportResponse(BaseModel):
    """Response with compliance report."""

    report_id: str
    framework: str
    scope: str
    generated_at: str
    generated_by: str
    overall_status: str
    compliance_score: float
    summary: dict[str, int]
    controls: list[ControlResponse]
    audit_log: list[dict[str, Any]]
    version: str


class MultiFrameworkRequest(BaseModel):
    """Request for multi-framework compliance report."""

    frameworks: list[str]
    verification_results: list[VerificationResultInput]
    scope: str
    organization: str


class AttestationCertificateRequest(BaseModel):
    """Request to generate an attestation certificate."""

    report_id: str
    signatory: str
    attestation_statement: str | None = None


@router.post("", response_model=ComplianceReportResponse)
async def generate_compliance_report(
    request: ComplianceReportRequest,
) -> ComplianceReportResponse:
    """
    Generate a compliance attestation report.

    Maps verification results to compliance framework controls
    and generates an audit-ready report.
    """
    from codeverify_agents import ComplianceAttestationEngine, ComplianceFramework

    try:
        framework = ComplianceFramework(request.framework)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid framework: {request.framework}. Valid options: soc2, hipaa, pci_dss, gdpr, iso_27001",
        )

    engine = ComplianceAttestationEngine()

    verification_results = [r.model_dump() for r in request.verification_results]

    result = await engine.analyze("", {
        "framework": framework,
        "verification_results": verification_results,
        "scope": request.scope,
        "organization": request.organization,
    })

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Compliance report generation failed: {result.error}",
        )

    return ComplianceReportResponse(
        report_id=result.data["report_id"],
        framework=result.data["framework"],
        scope=result.data["scope"],
        generated_at=result.data["generated_at"],
        generated_by=result.data["generated_by"],
        overall_status=result.data["overall_status"],
        compliance_score=result.data["compliance_score"],
        summary=result.data["summary"],
        controls=[ControlResponse(**c) for c in result.data["controls"]],
        audit_log=result.data["audit_log"],
        version=result.data["version"],
    )


@router.post("/multi-framework")
async def generate_multi_framework_report(
    request: MultiFrameworkRequest,
) -> dict[str, Any]:
    """
    Generate compliance reports for multiple frameworks.

    Useful for organizations that need to comply with multiple standards.
    """
    from codeverify_agents import ComplianceAttestationEngine, ComplianceFramework

    frameworks = []
    for f in request.frameworks:
        try:
            frameworks.append(ComplianceFramework(f))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid framework: {f}",
            )

    engine = ComplianceAttestationEngine()

    verification_results = [r.model_dump() for r in request.verification_results]

    result = await engine.generate_multi_framework_report(
        frameworks=frameworks,
        verification_results=verification_results,
        scope=request.scope,
        organization=request.organization,
    )

    return result


@router.post("/certificate")
async def generate_attestation_certificate(
    request: AttestationCertificateRequest,
) -> dict[str, Any]:
    """
    Generate a formal attestation certificate from a compliance report.
    """
    from codeverify_agents import ComplianceAttestationEngine

    engine = ComplianceAttestationEngine()

    # In production, you'd retrieve the report from storage
    # For now, return a mock certificate structure
    from codeverify_agents.compliance_attestation import ComplianceReport, ComplianceFramework, ControlStatus
    from datetime import datetime

    mock_report = ComplianceReport(
        report_id=request.report_id,
        framework=ComplianceFramework.SOC2,
        scope="Mock scope",
        generated_at=datetime.utcnow(),
        generated_by="CodeVerify",
        overall_status=ControlStatus.COMPLIANT,
        compliance_score=85.0,
        total_controls=10,
        compliant_controls=8,
        partial_controls=2,
        non_compliant_controls=0,
        not_applicable_controls=0,
    )

    certificate = await engine.generate_attestation_certificate(
        report=mock_report,
        signatory=request.signatory,
        attestation_statement=request.attestation_statement,
    )

    return certificate


@router.get("/frameworks")
async def get_supported_frameworks() -> dict[str, list[dict[str, str]]]:
    """Get supported compliance frameworks."""
    from codeverify_agents import ComplianceAttestationEngine

    engine = ComplianceAttestationEngine()
    frameworks = engine.get_supported_frameworks()

    descriptions = {
        "soc2": "Service Organization Control 2 - Trust Services Criteria",
        "hipaa": "Health Insurance Portability and Accountability Act",
        "pci_dss": "Payment Card Industry Data Security Standard",
        "gdpr": "General Data Protection Regulation",
        "iso_27001": "ISO/IEC 27001 Information Security Management",
        "nist_csf": "NIST Cybersecurity Framework",
        "cis": "CIS Critical Security Controls",
        "owasp": "OWASP Application Security Verification Standard",
    }

    return {
        "frameworks": [
            {"id": f, "name": f.upper().replace("_", " "), "description": descriptions.get(f, "")}
            for f in frameworks
        ]
    }


@router.get("/frameworks/{framework}/controls")
async def get_framework_controls(framework: str) -> dict[str, list[dict[str, str]]]:
    """Get controls for a specific compliance framework."""
    from codeverify_agents import ComplianceAttestationEngine, ComplianceFramework

    try:
        fw = ComplianceFramework(framework)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid framework: {framework}",
        )

    engine = ComplianceAttestationEngine()
    controls = engine.get_framework_controls(fw)

    return {
        "framework": framework,
        "controls": [
            {
                "id": c["id"],
                "name": c["name"],
                "description": c["description"],
            }
            for c in controls
        ],
    }
