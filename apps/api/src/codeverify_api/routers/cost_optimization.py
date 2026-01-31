"""Cost Optimization API endpoints."""

from typing import Any
from fastapi import APIRouter, HTTPException, status, Query
from pydantic import BaseModel, Field

router = APIRouter()


class RiskProfileRequest(BaseModel):
    """Risk profile for code."""

    code_hash: str | None = None
    risk_score: float = Field(default=50.0, ge=0, le=100)
    is_critical_path: bool = False
    is_security_sensitive: bool = False
    change_size: int = Field(default=100, ge=0)
    complexity: float = Field(default=5.0, ge=0)
    has_previous_bugs: bool = False
    is_ai_generated: bool = False


class BudgetConstraintsRequest(BaseModel):
    """Budget constraints for verification."""

    max_cost_usd: float | None = Field(default=None, ge=0)
    max_time_seconds: float | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=0)
    min_accuracy: float = Field(default=0.7, ge=0, le=1)


class VerificationPlanRequest(BaseModel):
    """Request for verification planning."""

    code: str = Field(..., description="The code to verify")
    risk_profile: RiskProfileRequest | None = None
    budget: BudgetConstraintsRequest | None = None


class VerificationPlanResponse(BaseModel):
    """Response with verification plan."""

    code_hash: str
    selected_depth: str
    estimated_cost_usd: float
    estimated_time_ms: float
    estimated_accuracy: float
    rationale: list[str]
    fallback_depth: str | None


class BatchPlanRequest(BaseModel):
    """Request for batch verification planning."""

    code_items: list[VerificationPlanRequest]
    total_budget: BudgetConstraintsRequest


class BudgetUsageResponse(BaseModel):
    """Response with budget usage statistics."""

    period: str
    total_cost_usd: float
    total_tokens: int
    total_time_seconds: float
    total_verifications: int
    avg_cost_per_verification: float
    verifications_by_depth: dict[str, int]
    depth_distribution: dict[str, float]


class CostModelResponse(BaseModel):
    """Cost model for each verification depth."""

    depth: str
    avg_time_ms: float
    avg_tokens: int
    avg_cost_usd: float
    accuracy: float
    recall: float


@router.post("/plan", response_model=VerificationPlanResponse)
async def plan_verification(request: VerificationPlanRequest) -> VerificationPlanResponse:
    """
    Plan verification depth based on code risk profile and budget.

    Returns the recommended verification depth with cost estimates.
    """
    from codeverify_core import VerificationCostOptimizer, BudgetConstraints

    optimizer = VerificationCostOptimizer()

    risk_profile = None
    if request.risk_profile:
        from codeverify_core.cost_optimizer import RiskProfile

        risk_profile = RiskProfile(
            code_hash=request.risk_profile.code_hash or "",
            risk_score=request.risk_profile.risk_score,
            is_critical_path=request.risk_profile.is_critical_path,
            is_security_sensitive=request.risk_profile.is_security_sensitive,
            change_size=request.risk_profile.change_size,
            complexity=request.risk_profile.complexity,
            has_previous_bugs=request.risk_profile.has_previous_bugs,
            is_ai_generated=request.risk_profile.is_ai_generated,
        )

    budget = None
    if request.budget:
        budget = BudgetConstraints(
            max_cost_usd=request.budget.max_cost_usd,
            max_time_seconds=request.budget.max_time_seconds,
            max_tokens=request.budget.max_tokens,
            min_accuracy=request.budget.min_accuracy,
        )

    plan = optimizer.plan_verification(
        code=request.code,
        risk_profile=risk_profile,
        budget=budget,
    )

    return VerificationPlanResponse(
        code_hash=plan.code_hash,
        selected_depth=plan.selected_depth.value,
        estimated_cost_usd=plan.estimated_cost_usd,
        estimated_time_ms=plan.estimated_time_ms,
        estimated_accuracy=plan.estimated_accuracy,
        rationale=plan.rationale,
        fallback_depth=plan.fallback_depth.value if plan.fallback_depth else None,
    )


@router.post("/plan/batch")
async def plan_batch_verification(request: BatchPlanRequest) -> dict[str, Any]:
    """
    Plan verification for multiple code items within a budget.

    Optimizes allocation across all items based on risk.
    """
    from codeverify_core import VerificationCostOptimizer, BudgetConstraints

    optimizer = VerificationCostOptimizer()

    code_items = []
    for item in request.code_items:
        risk_profile = None
        if item.risk_profile:
            from codeverify_core.cost_optimizer import RiskProfile

            risk_profile = RiskProfile(
                code_hash=item.risk_profile.code_hash or "",
                risk_score=item.risk_profile.risk_score,
                is_critical_path=item.risk_profile.is_critical_path,
                is_security_sensitive=item.risk_profile.is_security_sensitive,
                change_size=item.risk_profile.change_size,
                complexity=item.risk_profile.complexity,
                has_previous_bugs=item.risk_profile.has_previous_bugs,
                is_ai_generated=item.risk_profile.is_ai_generated,
            )
        code_items.append((item.code, risk_profile))

    budget = BudgetConstraints(
        max_cost_usd=request.total_budget.max_cost_usd,
        max_time_seconds=request.total_budget.max_time_seconds,
        max_tokens=request.total_budget.max_tokens,
        min_accuracy=request.total_budget.min_accuracy,
    )

    plans = optimizer.optimize_batch(code_items, budget)

    return {
        "plans": [
            VerificationPlanResponse(
                code_hash=p.code_hash,
                selected_depth=p.selected_depth.value,
                estimated_cost_usd=p.estimated_cost_usd,
                estimated_time_ms=p.estimated_time_ms,
                estimated_accuracy=p.estimated_accuracy,
                rationale=p.rationale,
                fallback_depth=p.fallback_depth.value if p.fallback_depth else None,
            ).model_dump()
            for p in plans
        ],
        "total_estimated_cost": sum(p.estimated_cost_usd for p in plans),
        "total_estimated_time_ms": sum(p.estimated_time_ms for p in plans),
    }


@router.post("/suggest-budget")
async def suggest_budget(
    code_items: list[str],
    target_accuracy: float = Query(default=0.9, ge=0.5, le=1.0),
) -> dict[str, Any]:
    """
    Suggest a budget for verifying a set of code items.
    """
    from codeverify_core import VerificationCostOptimizer

    optimizer = VerificationCostOptimizer()

    items = [(code, None) for code in code_items]
    budget = optimizer.suggest_budget(items, target_accuracy)

    return {
        "suggested_budget": {
            "max_cost_usd": budget.max_cost_usd,
            "max_time_seconds": budget.max_time_seconds,
            "min_accuracy": budget.min_accuracy,
        },
        "items_count": len(code_items),
        "target_accuracy": target_accuracy,
    }


@router.get("/usage", response_model=BudgetUsageResponse)
async def get_budget_usage(
    period: str = Query(default="all", pattern="^(all|today|week|month)$"),
) -> BudgetUsageResponse:
    """
    Get budget usage statistics for a period.
    """
    from codeverify_core import VerificationCostOptimizer

    optimizer = VerificationCostOptimizer()
    usage = optimizer.get_budget_usage(period)

    return BudgetUsageResponse(**usage)


@router.get("/cost-model")
async def get_cost_model() -> dict[str, list[CostModelResponse]]:
    """
    Get current cost model for each verification depth.
    """
    from codeverify_core import VerificationCostOptimizer

    optimizer = VerificationCostOptimizer()
    model = optimizer.get_cost_model()

    return {
        "cost_model": [
            CostModelResponse(depth=depth, **costs)
            for depth, costs in model.items()
        ]
    }


@router.get("/depths")
async def get_verification_depths() -> dict[str, list[dict[str, str]]]:
    """
    Get available verification depths with descriptions.
    """
    from codeverify_core import VerificationDepth

    descriptions = {
        "pattern": "Fast pattern matching only (lowest cost, lowest accuracy)",
        "static": "Static analysis with AST parsing",
        "ai": "LLM-based semantic analysis",
        "formal": "Full Z3 formal verification (highest accuracy)",
        "consensus": "Multi-model consensus verification (highest cost)",
    }

    return {
        "depths": [
            {
                "id": d.value,
                "name": d.name.title(),
                "description": descriptions.get(d.value, ""),
            }
            for d in VerificationDepth
        ]
    }


@router.post("/record-outcome")
async def record_verification_outcome(
    code_hash: str,
    selected_depth: str,
    actual_cost_usd: float,
    actual_time_ms: float,
    actual_tokens: int,
    found_issues: int,
    false_positives: int,
) -> dict[str, Any]:
    """
    Record verification outcome for cost model learning.
    """
    from codeverify_core import VerificationCostOptimizer, VerificationDepth
    from codeverify_core.cost_optimizer import VerificationPlan

    try:
        depth = VerificationDepth(selected_depth)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid depth: {selected_depth}",
        )

    optimizer = VerificationCostOptimizer()

    plan = VerificationPlan(
        code_hash=code_hash,
        selected_depth=depth,
        estimated_cost_usd=0,
        estimated_time_ms=0,
        estimated_accuracy=0,
    )

    optimizer.record_outcome(
        plan=plan,
        actual_cost_usd=actual_cost_usd,
        actual_time_ms=actual_time_ms,
        actual_tokens=actual_tokens,
        found_issues=found_issues,
        false_positives=false_positives,
    )

    return {
        "status": "recorded",
        "message": "Outcome recorded for cost model learning",
    }
