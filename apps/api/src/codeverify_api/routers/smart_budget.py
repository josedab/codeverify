"""Smart Verification Budget — ML-based cost optimizer.

Routes code through appropriate verification tiers based on risk scores:
  static → pattern → AI → Z3
Instruments cost tracking per verification type and enforces per-org budget caps.
"""

import math
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

router = APIRouter()


# ---------------------------------------------------------------------------
# Verification tiers and cost model
# ---------------------------------------------------------------------------

class VerificationTier:
    STATIC = "static"       # Fast pattern matching, ~$0.001/check
    PATTERN = "pattern"     # AST-based analysis, ~$0.005/check
    AI = "ai"               # LLM analysis, ~$0.05/check
    Z3 = "z3"               # Formal verification, ~$0.10/check

TIER_COSTS: dict[str, float] = {
    VerificationTier.STATIC: 0.001,
    VerificationTier.PATTERN: 0.005,
    VerificationTier.AI: 0.05,
    VerificationTier.Z3: 0.10,
}

TIER_LATENCY_MS: dict[str, float] = {
    VerificationTier.STATIC: 10,
    VerificationTier.PATTERN: 50,
    VerificationTier.AI: 2000,
    VerificationTier.Z3: 5000,
}

# ---------------------------------------------------------------------------
# Risk scoring model
# ---------------------------------------------------------------------------

LANGUAGE_RISK: dict[str, float] = {
    "python": 0.3,
    "javascript": 0.5,
    "typescript": 0.25,
    "go": 0.2,
    "java": 0.25,
    "rust": 0.15,
    "c": 0.7,
    "cpp": 0.65,
}

FILE_PATTERN_RISK: dict[str, float] = {
    "auth": 0.9,
    "login": 0.9,
    "payment": 0.95,
    "crypto": 0.85,
    "security": 0.8,
    "sql": 0.7,
    "database": 0.65,
    "api": 0.5,
    "handler": 0.5,
    "controller": 0.4,
    "model": 0.3,
    "util": 0.2,
    "test": 0.1,
    "spec": 0.1,
    "mock": 0.05,
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RiskAssessmentRequest(BaseModel):
    file_path: str
    language: str = Field(default="python")
    lines_changed: int = Field(default=10, ge=0)
    is_new_file: bool = False
    author_experience_score: float = Field(default=0.5, ge=0.0, le=1.0)
    has_tests: bool = True
    file_history_changes: int = Field(default=5, ge=0)
    code_snippet: str | None = None


class RiskAssessmentResponse(BaseModel):
    file_path: str
    risk_score: float
    risk_level: str
    recommended_tier: str
    estimated_cost: float
    estimated_latency_ms: float
    risk_factors: dict[str, float]
    explanation: str


class BudgetConfig(BaseModel):
    org_id: str
    monthly_budget_cents: int = Field(default=10000, description="Monthly budget in cents ($100)")
    tier_weights: dict[str, float] = Field(
        default_factory=lambda: {
            VerificationTier.STATIC: 1.0,
            VerificationTier.PATTERN: 1.0,
            VerificationTier.AI: 1.0,
            VerificationTier.Z3: 1.0,
        }
    )
    auto_downgrade: bool = Field(default=True, description="Auto-downgrade tier when over budget")


class CostRecord(BaseModel):
    id: str
    org_id: str
    verification_type: str
    cost_cents: float
    file_path: str
    timestamp: str


class BudgetStatus(BaseModel):
    org_id: str
    monthly_budget_cents: int
    spent_cents: float
    remaining_cents: float
    utilization_pct: float
    spending_by_tier: dict[str, float]
    projected_monthly_spend: float
    on_track: bool
    days_remaining: int


class CostDashboard(BaseModel):
    org_id: str
    current_month: BudgetStatus
    daily_spending: list[dict[str, Any]]
    tier_breakdown: dict[str, dict[str, Any]]
    savings_from_routing: float
    optimization_suggestions: list[str]


# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

_budgets: dict[str, BudgetConfig] = {}
_cost_records: list[CostRecord] = []


def _get_month_start() -> datetime:
    now = datetime.utcnow()
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Risk classifier
# ---------------------------------------------------------------------------

def _compute_risk_score(req: RiskAssessmentRequest) -> tuple[float, dict[str, float]]:
    """ML-based risk classifier using file history, change size, language, author seniority."""
    factors: dict[str, float] = {}

    # Language risk
    factors["language"] = LANGUAGE_RISK.get(req.language, 0.4)

    # File path risk (check if path contains high-risk keywords)
    path_lower = req.file_path.lower()
    max_path_risk = 0.2
    for keyword, risk in FILE_PATTERN_RISK.items():
        if keyword in path_lower:
            max_path_risk = max(max_path_risk, risk)
    factors["file_path"] = max_path_risk

    # Change size risk (larger changes = higher risk)
    factors["change_size"] = min(1.0, req.lines_changed / 500)

    # New file risk
    factors["new_file"] = 0.7 if req.is_new_file else 0.1

    # Author experience (inverted: less experience = higher risk)
    factors["author_experience"] = 1.0 - req.author_experience_score

    # Test coverage
    factors["test_coverage"] = 0.1 if req.has_tests else 0.6

    # File churn (frequently changed files = higher risk)
    factors["file_churn"] = min(1.0, req.file_history_changes / 50)

    # Weighted average
    weights = {
        "language": 0.10,
        "file_path": 0.25,
        "change_size": 0.15,
        "new_file": 0.10,
        "author_experience": 0.15,
        "test_coverage": 0.10,
        "file_churn": 0.15,
    }

    score = sum(factors[k] * weights[k] for k in factors)
    return round(min(1.0, score), 3), factors


def _tier_for_risk(score: float, budget: BudgetConfig | None = None) -> str:
    """Map risk score to verification tier, respecting budget constraints."""
    if score >= 0.7:
        tier = VerificationTier.Z3
    elif score >= 0.45:
        tier = VerificationTier.AI
    elif score >= 0.2:
        tier = VerificationTier.PATTERN
    else:
        tier = VerificationTier.STATIC

    # Budget-aware downgrade
    if budget and budget.auto_downgrade:
        month_start = _get_month_start()
        spent = sum(
            r.cost_cents for r in _cost_records
            if r.org_id == budget.org_id
            and datetime.fromisoformat(r.timestamp) >= month_start
        )
        if spent >= budget.monthly_budget_cents:
            tier = VerificationTier.STATIC  # Fallback to cheapest tier
        elif spent >= budget.monthly_budget_cents * 0.9:
            if tier == VerificationTier.Z3:
                tier = VerificationTier.AI
            elif tier == VerificationTier.AI:
                tier = VerificationTier.PATTERN

    return tier


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/assess-risk", response_model=RiskAssessmentResponse)
async def assess_risk(req: RiskAssessmentRequest) -> RiskAssessmentResponse:
    """Assess risk and recommend verification tier for a code change."""
    score, factors = _compute_risk_score(req)

    risk_level = "low"
    if score >= 0.7:
        risk_level = "critical"
    elif score >= 0.45:
        risk_level = "high"
    elif score >= 0.2:
        risk_level = "medium"

    tier = _tier_for_risk(score)

    explanations = []
    sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
    for name, val in sorted_factors[:3]:
        if val > 0.3:
            explanations.append(f"{name} contributes {val:.0%} risk")
    explanation = "; ".join(explanations) if explanations else "Low overall risk"

    return RiskAssessmentResponse(
        file_path=req.file_path,
        risk_score=score,
        risk_level=risk_level,
        recommended_tier=tier,
        estimated_cost=TIER_COSTS[tier],
        estimated_latency_ms=TIER_LATENCY_MS[tier],
        risk_factors=factors,
        explanation=explanation,
    )


@router.post("/record-cost")
async def record_cost(
    org_id: str,
    verification_type: str,
    file_path: str,
    cost_override: float | None = None,
) -> CostRecord:
    """Record a verification cost event."""
    if verification_type not in TIER_COSTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid type. Must be one of: {list(TIER_COSTS.keys())}",
        )

    cost = cost_override if cost_override is not None else TIER_COSTS[verification_type]
    record = CostRecord(
        id=str(uuid.uuid4()),
        org_id=org_id,
        verification_type=verification_type,
        cost_cents=cost * 100,
        file_path=file_path,
        timestamp=datetime.utcnow().isoformat(),
    )
    _cost_records.append(record)
    return record


@router.put("/budget/{org_id}")
async def set_budget(org_id: str, config: BudgetConfig) -> BudgetConfig:
    """Set or update the monthly budget cap for an organization."""
    config.org_id = org_id
    _budgets[org_id] = config
    return config


@router.get("/budget/{org_id}/status", response_model=BudgetStatus)
async def get_budget_status(org_id: str) -> BudgetStatus:
    """Get current budget utilization status."""
    budget = _budgets.get(org_id, BudgetConfig(org_id=org_id))
    month_start = _get_month_start()
    now = datetime.utcnow()

    month_records = [
        r for r in _cost_records
        if r.org_id == org_id and datetime.fromisoformat(r.timestamp) >= month_start
    ]

    spent = sum(r.cost_cents for r in month_records)
    by_tier: dict[str, float] = {}
    for r in month_records:
        by_tier[r.verification_type] = by_tier.get(r.verification_type, 0) + r.cost_cents

    days_elapsed = max((now - month_start).days, 1)
    days_in_month = 30
    daily_rate = spent / days_elapsed
    projected = daily_rate * days_in_month

    return BudgetStatus(
        org_id=org_id,
        monthly_budget_cents=budget.monthly_budget_cents,
        spent_cents=round(spent, 2),
        remaining_cents=round(max(0, budget.monthly_budget_cents - spent), 2),
        utilization_pct=round(spent / max(budget.monthly_budget_cents, 1) * 100, 1),
        spending_by_tier=by_tier,
        projected_monthly_spend=round(projected, 2),
        on_track=projected <= budget.monthly_budget_cents,
        days_remaining=days_in_month - days_elapsed,
    )


@router.get("/dashboard/{org_id}", response_model=CostDashboard)
async def get_cost_dashboard(org_id: str) -> CostDashboard:
    """Get the full cost dashboard with optimization suggestions."""
    budget_status = await get_budget_status(org_id)
    month_start = _get_month_start()

    # Daily spending aggregation
    daily: dict[str, float] = {}
    tier_totals: dict[str, dict[str, Any]] = {
        t: {"count": 0, "total_cost": 0.0} for t in TIER_COSTS
    }

    for r in _cost_records:
        if r.org_id != org_id:
            continue
        if datetime.fromisoformat(r.timestamp) < month_start:
            continue
        day = r.timestamp[:10]
        daily[day] = daily.get(day, 0) + r.cost_cents
        if r.verification_type in tier_totals:
            tier_totals[r.verification_type]["count"] += 1
            tier_totals[r.verification_type]["total_cost"] += r.cost_cents

    # Calculate savings from intelligent routing vs. running everything through Z3
    z3_cost_per = TIER_COSTS[VerificationTier.Z3] * 100
    total_checks = sum(t["count"] for t in tier_totals.values())
    naive_cost = total_checks * z3_cost_per
    actual_cost = budget_status.spent_cents
    savings = max(0, naive_cost - actual_cost)

    # Optimization suggestions
    suggestions: list[str] = []
    if tier_totals[VerificationTier.Z3]["count"] > total_checks * 0.5:
        suggestions.append("Over 50% of checks use Z3. Consider lowering tier for low-risk files.")
    if budget_status.utilization_pct > 80:
        suggestions.append("Budget utilization is high. Enable auto-downgrade to stay within budget.")
    if not budget_status.on_track:
        suggestions.append("Projected spending exceeds budget. Reduce AI/Z3 checks on test files.")
    if tier_totals[VerificationTier.STATIC]["count"] < total_checks * 0.3:
        suggestions.append("Increase use of static checks for low-risk files to save costs.")

    return CostDashboard(
        org_id=org_id,
        current_month=budget_status,
        daily_spending=[{"date": d, "cost_cents": c} for d, c in sorted(daily.items())],
        tier_breakdown=tier_totals,
        savings_from_routing=round(savings, 2),
        optimization_suggestions=suggestions,
    )
