"""Risk Prediction API endpoints (Regression Oracle)."""

from typing import Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter()


class ChangeMetricsRequest(BaseModel):
    """Metrics about a code change."""

    lines_added: int = Field(default=0, ge=0)
    lines_deleted: int = Field(default=0, ge=0)
    files_changed: int = Field(default=1, ge=1)
    functions_modified: int = Field(default=0, ge=0)
    complexity_delta: float = Field(default=0.0)
    touches_critical_path: bool = Field(default=False)
    modifies_interfaces: bool = Field(default=False)
    cross_module_changes: bool = Field(default=False)


class RiskPredictionRequest(BaseModel):
    """Request to predict bug risk for a code change."""

    diff: str = Field(..., description="The diff or changed code")
    change_id: str | None = Field(default=None, description="Unique change identifier")
    file_paths: list[str] = Field(default_factory=list, description="Files changed")
    author: str | None = Field(default=None, description="Author of the change")
    commit_message: str | None = Field(default=None, description="Commit message")
    base_branch: str = Field(default="main", description="Branch being merged into")
    change_metrics: ChangeMetricsRequest | None = Field(
        default=None, description="Pre-computed change metrics"
    )


class RiskFactorResponse(BaseModel):
    """A factor contributing to risk score."""

    factor: str
    details: str
    contribution: int


class SimilarBugResponse(BaseModel):
    """A similar historical bug."""

    bug_id: str
    similarity: float
    root_cause: str
    severity: str


class RiskPredictionResponse(BaseModel):
    """Response with risk prediction."""

    change_id: str
    risk_level: str = Field(..., description="Risk level: low, medium, high, critical")
    risk_score: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=1)
    verification_priority: int = Field(..., ge=1, le=4)
    risk_factors: list[RiskFactorResponse]
    recommended_actions: list[str]
    similar_past_bugs: list[SimilarBugResponse]


class BugRecordRequest(BaseModel):
    """Request to record a bug for training."""

    bug_id: str
    file_path: str
    function_name: str | None = None
    author: str
    severity: str = Field(..., pattern="^(low|medium|high|critical)$")
    root_cause: str
    fix_complexity: str = Field(..., pattern="^(low|medium|high)$")
    change_metrics: ChangeMetricsRequest | None = None


class BatchRiskRequest(BaseModel):
    """Request to predict risk for multiple changes."""

    changes: list[RiskPredictionRequest]


class BatchRiskResponse(BaseModel):
    """Response with risk predictions sorted by priority."""

    predictions: list[RiskPredictionResponse]
    high_risk_count: int
    total_estimated_risk: float


@router.post("", response_model=RiskPredictionResponse)
async def predict_risk(request: RiskPredictionRequest) -> RiskPredictionResponse:
    """
    Predict bug risk for a code change.

    Uses historical data, change metrics, and semantic analysis to
    estimate the likelihood of the change introducing bugs.
    """
    from codeverify_agents import RegressionOracle

    oracle = RegressionOracle()

    context = {
        "change_id": request.change_id,
        "file_paths": request.file_paths,
        "author": request.author,
        "commit_message": request.commit_message,
        "base_branch": request.base_branch,
    }

    if request.change_metrics:
        from codeverify_agents.regression_oracle import ChangeMetrics

        context["change_metrics"] = ChangeMetrics(
            lines_added=request.change_metrics.lines_added,
            lines_deleted=request.change_metrics.lines_deleted,
            files_changed=request.change_metrics.files_changed,
            functions_modified=request.change_metrics.functions_modified,
            complexity_delta=request.change_metrics.complexity_delta,
            touches_critical_path=request.change_metrics.touches_critical_path,
            modifies_interfaces=request.change_metrics.modifies_interfaces,
            cross_module_changes=request.change_metrics.cross_module_changes,
        )

    result = await oracle.analyze(request.diff, context)

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk prediction failed: {result.error}",
        )

    return RiskPredictionResponse(
        change_id=result.data["change_id"],
        risk_level=result.data["risk_level"],
        risk_score=result.data["risk_score"],
        confidence=result.data["confidence"],
        verification_priority=result.data["verification_priority"],
        risk_factors=[RiskFactorResponse(**f) for f in result.data["risk_factors"]],
        recommended_actions=result.data["recommended_actions"],
        similar_past_bugs=[SimilarBugResponse(**b) for b in result.data["similar_past_bugs"]],
    )


@router.post("/batch", response_model=BatchRiskResponse)
async def predict_risk_batch(request: BatchRiskRequest) -> BatchRiskResponse:
    """
    Predict risk for multiple changes and prioritize verification.

    Returns predictions sorted by risk score (highest first).
    """
    from codeverify_agents import RegressionOracle

    oracle = RegressionOracle()
    predictions = []

    for change in request.changes:
        context = {
            "change_id": change.change_id,
            "file_paths": change.file_paths,
            "author": change.author,
            "commit_message": change.commit_message,
        }

        result = await oracle.analyze(change.diff, context)

        if result.success:
            predictions.append(
                RiskPredictionResponse(
                    change_id=result.data["change_id"],
                    risk_level=result.data["risk_level"],
                    risk_score=result.data["risk_score"],
                    confidence=result.data["confidence"],
                    verification_priority=result.data["verification_priority"],
                    risk_factors=[RiskFactorResponse(**f) for f in result.data["risk_factors"]],
                    recommended_actions=result.data["recommended_actions"],
                    similar_past_bugs=[SimilarBugResponse(**b) for b in result.data["similar_past_bugs"]],
                )
            )

    # Sort by risk score
    predictions.sort(key=lambda p: p.risk_score, reverse=True)

    high_risk_count = sum(1 for p in predictions if p.risk_level in ("high", "critical"))
    total_risk = sum(p.risk_score for p in predictions)

    return BatchRiskResponse(
        predictions=predictions,
        high_risk_count=high_risk_count,
        total_estimated_risk=total_risk,
    )


@router.post("/bugs")
async def record_bug(request: BugRecordRequest) -> dict[str, Any]:
    """
    Record a bug for training the regression oracle.

    This helps improve future predictions by learning from actual bugs.
    """
    from codeverify_agents import RegressionOracle, BugRecord
    from codeverify_agents.regression_oracle import ChangeMetrics

    oracle = RegressionOracle()

    metrics = ChangeMetrics()
    if request.change_metrics:
        metrics = ChangeMetrics(
            lines_added=request.change_metrics.lines_added,
            lines_deleted=request.change_metrics.lines_deleted,
            files_changed=request.change_metrics.files_changed,
        )

    bug = BugRecord(
        bug_id=request.bug_id,
        timestamp=datetime.utcnow(),
        file_path=request.file_path,
        function_name=request.function_name,
        change_metrics=metrics,
        author=request.author,
        severity=request.severity,
        root_cause=request.root_cause,
        fix_complexity=request.fix_complexity,
    )

    oracle.record_bug(bug)

    return {
        "status": "recorded",
        "bug_id": request.bug_id,
        "message": "Bug recorded for model training",
    }


@router.post("/feedback")
async def submit_prediction_feedback(
    change_id: str,
    predicted_risk: float,
    was_bug: bool,
    notes: str | None = None,
) -> dict[str, Any]:
    """
    Submit feedback on a risk prediction to improve the model.
    """
    from codeverify_agents import RegressionOracle

    oracle = RegressionOracle()
    oracle.update_weights([
        {
            "change_id": change_id,
            "predicted_risk": predicted_risk,
            "was_bug": was_bug,
        }
    ])

    return {
        "status": "received",
        "change_id": change_id,
        "message": "Feedback recorded for model improvement",
    }
