"""Threat Modeling API endpoints."""

from typing import Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter()


class ThreatModelRequest(BaseModel):
    """Request to generate a threat model."""

    code: str = Field(..., description="The code to analyze")
    system_name: str = Field(default="Unknown System", description="Name of the system")
    architecture_description: str | None = Field(
        default=None, description="High-level architecture description"
    )
    language: str = Field(default="python", description="Programming language")
    framework: str | None = Field(default=None, description="Web framework if applicable")
    deployment_context: str | None = Field(
        default=None, description="Deployment context (cloud, on-prem, etc.)"
    )


class AttackSurfaceResponse(BaseModel):
    """Attack surface in the threat model."""

    name: str
    type: str
    entry_points: list[str]
    data_flows: list[str]
    trust_level: str


class ThreatResponse(BaseModel):
    """A threat in the threat model."""

    id: str
    title: str
    description: str
    stride_category: str
    owasp_category: str | None
    attack_surface: str
    likelihood: str
    impact: str
    risk_score: float
    affected_components: list[str]
    mitigations: list[str]


class ThreatModelResponse(BaseModel):
    """Response with generated threat model."""

    system_name: str
    description: str
    attack_surfaces: list[AttackSurfaceResponse]
    threats: list[ThreatResponse]
    trust_boundaries: list[dict[str, Any]]
    data_flows: list[dict[str, Any]]
    overall_risk_score: float
    recommendations: list[str]
    threat_summary: dict[str, Any]


@router.post("", response_model=ThreatModelResponse)
async def generate_threat_model(request: ThreatModelRequest) -> ThreatModelResponse:
    """
    Generate a security threat model for the provided code.

    Analyzes code to identify attack surfaces, threats using STRIDE methodology,
    and maps findings to OWASP Top 10 categories.
    """
    from codeverify_agents import ThreatModelingAgent

    agent = ThreatModelingAgent()

    context = {
        "system_name": request.system_name,
        "architecture_description": request.architecture_description,
        "language": request.language,
        "framework": request.framework,
        "deployment_context": request.deployment_context,
    }

    result = await agent.analyze(request.code, context)

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Threat modeling failed: {result.error}",
        )

    return ThreatModelResponse(**result.data)


@router.post("/architecture")
async def generate_architecture_threat_model(
    architecture_description: str,
    components: list[dict[str, Any]],
    data_flows: list[dict[str, Any]],
) -> ThreatModelResponse:
    """
    Generate threat model from architecture description without code.
    """
    from codeverify_agents import ThreatModelingAgent

    agent = ThreatModelingAgent()
    result = await agent.analyze_architecture(
        architecture_description=architecture_description,
        components=components,
        data_flows=data_flows,
    )

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Threat modeling failed: {result.error}",
        )

    return ThreatModelResponse(**result.data)


@router.get("/categories/stride")
async def get_stride_categories() -> dict[str, list[dict[str, str]]]:
    """Get STRIDE threat categories with descriptions."""
    from codeverify_agents import STRIDECategory

    return {
        "categories": [
            {
                "id": cat.value,
                "name": cat.name.replace("_", " ").title(),
                "description": _get_stride_description(cat),
            }
            for cat in STRIDECategory
        ]
    }


@router.get("/categories/owasp")
async def get_owasp_categories() -> dict[str, list[dict[str, str]]]:
    """Get OWASP Top 10 2021 categories."""
    from codeverify_agents.threat_modeling import OWASPCategory

    return {
        "categories": [
            {
                "id": cat.value,
                "name": _get_owasp_name(cat),
            }
            for cat in OWASPCategory
        ]
    }


def _get_stride_description(category) -> str:
    """Get description for STRIDE category."""
    descriptions = {
        "spoofing": "Impersonating a user or system",
        "tampering": "Modifying data or code without authorization",
        "repudiation": "Denying actions without proof",
        "information_disclosure": "Exposing sensitive information",
        "denial_of_service": "Making system unavailable",
        "elevation_of_privilege": "Gaining unauthorized permissions",
    }
    return descriptions.get(category.value, "")


def _get_owasp_name(category) -> str:
    """Get name for OWASP category."""
    names = {
        "A01:2021": "Broken Access Control",
        "A02:2021": "Cryptographic Failures",
        "A03:2021": "Injection",
        "A04:2021": "Insecure Design",
        "A05:2021": "Security Misconfiguration",
        "A06:2021": "Vulnerable and Outdated Components",
        "A07:2021": "Identification and Authentication Failures",
        "A08:2021": "Software and Data Integrity Failures",
        "A09:2021": "Security Logging and Monitoring Failures",
        "A10:2021": "Server-Side Request Forgery",
    }
    return names.get(category.value, category.value)
