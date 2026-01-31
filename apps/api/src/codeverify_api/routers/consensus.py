"""Consensus Verification API endpoints."""

from typing import Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter()


class ConsensusVerificationRequest(BaseModel):
    """Request for multi-model consensus verification."""

    code: str = Field(..., description="The code to verify")
    file_path: str = Field(default="unknown", description="Path to the file")
    language: str = Field(default="python", description="Programming language")
    consensus_strategy: str = Field(
        default="majority",
        description="Consensus strategy: unanimous, majority, weighted, any",
    )


class ModelFindingResponse(BaseModel):
    """A finding from a single model."""

    id: str
    severity: str
    category: str
    title: str
    confidence: float


class ConsensusFindingResponse(BaseModel):
    """A finding that achieved consensus."""

    id: str
    severity: str
    category: str
    title: str
    description: str
    location: dict[str, Any]
    confidence: float
    agreeing_models: list[str]
    dissenting_models: list[str]
    suggested_fix: str | None


class ConsensusVerificationResponse(BaseModel):
    """Response with consensus verification results."""

    code_hash: str
    consensus_strategy: str
    models_queried: list[str]
    overall_confidence: float
    total_latency_ms: float
    consensus_findings: list[ConsensusFindingResponse]
    model_only_findings: dict[str, list[ModelFindingResponse]]
    summary: dict[str, Any]


class ModelConfigRequest(BaseModel):
    """Configuration for a model in consensus."""

    provider: str = Field(..., description="Model provider: openai_gpt5, anthropic_claude, etc.")
    model_name: str = Field(..., description="Specific model name")
    weight: float = Field(default=1.0, ge=0.0, le=2.0, description="Weight for weighted consensus")


class UpdateModelsRequest(BaseModel):
    """Request to update model configuration."""

    models: list[ModelConfigRequest]


@router.post("", response_model=ConsensusVerificationResponse)
async def verify_with_consensus(
    request: ConsensusVerificationRequest,
) -> ConsensusVerificationResponse:
    """
    Verify code using multiple LLM models and return consensus findings.

    Only findings that achieve consensus across models are included
    in the consensus_findings. Model-specific findings are tracked separately.
    """
    from codeverify_agents import MultiModelConsensus, ConsensusStrategy

    try:
        strategy = ConsensusStrategy(request.consensus_strategy)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid consensus strategy: {request.consensus_strategy}",
        )

    consensus = MultiModelConsensus(consensus_strategy=strategy)

    result = await consensus.analyze(request.code, {
        "file_path": request.file_path,
        "language": request.language,
    })

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Consensus verification failed: {result.error}",
        )

    return ConsensusVerificationResponse(
        code_hash=result.data["code_hash"],
        consensus_strategy=result.data["consensus_strategy"],
        models_queried=result.data["models_queried"],
        overall_confidence=result.data["overall_confidence"],
        total_latency_ms=result.data["total_latency_ms"],
        consensus_findings=[
            ConsensusFindingResponse(**f)
            for f in result.data["consensus_findings"]
        ],
        model_only_findings={
            model: [ModelFindingResponse(**f) for f in findings]
            for model, findings in result.data["model_only_findings"].items()
        },
        summary=result.data["summary"],
    )


@router.post("/escalate", response_model=ConsensusVerificationResponse)
async def verify_with_escalation(
    request: ConsensusVerificationRequest,
) -> ConsensusVerificationResponse:
    """
    Verify with automatic escalation from fast to consensus.

    Starts with a single fast model. If findings are uncertain or
    high-severity, escalates to full consensus verification.
    """
    from codeverify_agents import MultiModelConsensus

    consensus = MultiModelConsensus()

    result = await consensus.verify_with_escalation(
        code=request.code,
        context={
            "file_path": request.file_path,
            "language": request.language,
        },
    )

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {result.error}",
        )

    # Handle both fast mode and full consensus response
    if result.data.get("mode") == "fast":
        # Fast mode - convert to consensus response format
        return ConsensusVerificationResponse(
            code_hash="",
            consensus_strategy="fast",
            models_queried=["primary"],
            overall_confidence=0.9,
            total_latency_ms=result.latency_ms,
            consensus_findings=[],
            model_only_findings={
                "primary": [
                    ModelFindingResponse(**f)
                    for f in result.data.get("findings", [])
                ]
            },
            summary={"mode": "fast", "escalated": False},
        )

    return ConsensusVerificationResponse(**result.data)


@router.get("/strategies")
async def get_consensus_strategies() -> dict[str, list[dict[str, str]]]:
    """Get available consensus strategies with descriptions."""
    from codeverify_agents import ConsensusStrategy

    descriptions = {
        "unanimous": "All models must agree for a finding to be reported",
        "majority": "More than 50% of models must agree",
        "weighted": "Models are weighted by confidence, majority of weight must agree",
        "any": "Any model finding is reported (for comparison)",
    }

    return {
        "strategies": [
            {
                "id": s.value,
                "name": s.name.title(),
                "description": descriptions.get(s.value, ""),
            }
            for s in ConsensusStrategy
        ]
    }


@router.get("/models")
async def get_available_models() -> dict[str, list[dict[str, str]]]:
    """Get available models for consensus verification."""
    from codeverify_agents.multi_model_consensus import ModelProvider

    return {
        "models": [
            {"id": m.value, "name": m.name.replace("_", " ").title()}
            for m in ModelProvider
        ]
    }
