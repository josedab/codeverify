"""Performance & Cost Dashboard.

Real-time dashboard for LLM token usage, Z3 solver time, cost per review,
budget alerts, ROI calculations, and optimization recommendations.

Features:
- LLM token usage tracking per model and operation
- Z3 solver time and resource metrics
- Cost per review/verification calculation
- Budget alerts with configurable thresholds
- ROI calculation (cost savings from automated verification)
- Optimization recommendations (model downgrades, caching)
"""

from __future__ import annotations

import statistics
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

import structlog

logger = structlog.get_logger()


class CostCategory(str, Enum):
    """Categories of cost."""

    LLM_TOKENS = "llm_tokens"
    Z3_SOLVER = "z3_solver"
    INFRASTRUCTURE = "infrastructure"
    API_CALLS = "api_calls"
    STORAGE = "storage"


class ModelProvider(str, Enum):
    """LLM model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class AlertLevel(str, Enum):
    """Budget alert levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class OptimizationType(str, Enum):
    """Types of optimization recommendations."""

    MODEL_DOWNGRADE = "model_downgrade"
    CACHING = "caching"
    BATCH_VERIFICATION = "batch_verification"
    SKIP_LOW_RISK = "skip_low_risk"
    REDUCE_RETRIES = "reduce_retries"


@dataclass
class TokenUsageRecord:
    """Record of LLM token usage."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    provider: ModelProvider = ModelProvider.OPENAI
    model: str = "gpt-4"
    operation: str = ""  # semantic_analysis, security_scan, etc.
    input_tokens: int = 0
    output_tokens: int = 0
    cost_cents: float = 0.0
    latency_ms: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class SolverMetric:
    """Z3 solver performance metric."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    check_type: str = ""  # null_safety, bounds_check, etc.
    solve_time_ms: int = 0
    constraint_count: int = 0
    result: str = ""  # sat, unsat, timeout
    memory_mb: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ReviewCost:
    """Cost breakdown for a single review."""

    review_id: str = ""
    llm_cost_cents: float = 0.0
    solver_cost_cents: float = 0.0
    infra_cost_cents: float = 0.0
    total_cost_cents: float = 0.0
    finding_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def cost_per_finding(self) -> float:
        if self.finding_count == 0:
            return 0.0
        return round(self.total_cost_cents / self.finding_count, 2)


@dataclass
class BudgetConfig:
    """Budget configuration with alert thresholds."""

    monthly_budget_cents: int = 10000  # $100
    warning_threshold: float = 0.75  # Alert at 75%
    critical_threshold: float = 0.90  # Alert at 90%
    auto_pause_at_limit: bool = False


@dataclass
class BudgetAlert:
    """A budget alert."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    level: AlertLevel = AlertLevel.INFO
    title: str = ""
    message: str = ""
    current_spend_cents: float = 0.0
    budget_cents: int = 0
    utilization: float = 0.0
    triggered_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ROIMetrics:
    """Return on investment metrics."""

    total_reviews: int = 0
    total_findings: int = 0
    critical_findings_caught: int = 0
    estimated_bug_cost_saved_cents: int = 0
    verification_cost_cents: int = 0
    net_savings_cents: int = 0
    roi_percentage: float = 0.0
    manual_review_hours_saved: float = 0.0


@dataclass
class OptimizationRecommendation:
    """A cost optimization recommendation."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: OptimizationType = OptimizationType.CACHING
    title: str = ""
    description: str = ""
    estimated_savings_percent: float = 0.0
    estimated_savings_cents: int = 0
    effort: str = "low"  # low, medium, high
    priority: int = 0  # 1-10


@dataclass
class CostDashboardData:
    """Aggregated data for the cost dashboard."""

    period_start: datetime = field(default_factory=lambda: datetime.now(UTC))
    period_end: datetime = field(default_factory=lambda: datetime.now(UTC))
    total_cost_cents: float = 0.0
    cost_by_category: dict[str, float] = field(default_factory=dict)
    cost_by_model: dict[str, float] = field(default_factory=dict)
    cost_by_operation: dict[str, float] = field(default_factory=dict)
    total_tokens: int = 0
    total_solver_time_ms: int = 0
    avg_cost_per_review_cents: float = 0.0
    review_count: int = 0
    budget_utilization: float = 0.0
    roi: ROIMetrics | None = None
    alerts: list[BudgetAlert] = field(default_factory=list)
    recommendations: list[OptimizationRecommendation] = field(default_factory=list)


# Token pricing (cents per 1K tokens)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4": {"input": 3.0, "output": 6.0},
    "gpt-4-turbo": {"input": 1.0, "output": 3.0},
    "gpt-3.5-turbo": {"input": 0.05, "output": 0.15},
    "claude-3-opus": {"input": 1.5, "output": 7.5},
    "claude-3-sonnet": {"input": 0.3, "output": 1.5},
    "claude-3-haiku": {"input": 0.025, "output": 0.125},
    "local": {"input": 0.0, "output": 0.0},
}


class CostCalculator:
    """Calculates costs from usage records."""

    def calculate_token_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost in cents for token usage."""
        pricing = MODEL_PRICING.get(model, MODEL_PRICING.get("gpt-4", {}))
        input_cost = (input_tokens / 1000) * pricing.get("input", 0)
        output_cost = (output_tokens / 1000) * pricing.get("output", 0)
        return round(input_cost + output_cost, 4)

    def calculate_solver_cost(self, solve_time_ms: int) -> float:
        """Estimate solver cost based on compute time (cents)."""
        hours = solve_time_ms / (1000 * 3600)
        cost_per_hour = 10.0  # Rough estimate: $0.10/hour for compute
        return round(hours * cost_per_hour, 4)


class OptimizationEngine:
    """Generates cost optimization recommendations."""

    def analyze(
        self,
        token_records: list[TokenUsageRecord],
        _solver_metrics: list[SolverMetric],
        review_costs: list[ReviewCost],
    ) -> list[OptimizationRecommendation]:
        """Analyze usage and generate optimization recommendations."""
        recommendations: list[OptimizationRecommendation] = []

        # Check for model downgrade opportunities
        gpt4_usage = [r for r in token_records if r.model == "gpt-4"]
        if gpt4_usage:
            total_gpt4_cost = sum(r.cost_cents for r in gpt4_usage)
            if total_gpt4_cost > 1000:  # > $10
                recommendations.append(
                    OptimizationRecommendation(
                        type=OptimizationType.MODEL_DOWNGRADE,
                        title="Consider GPT-4 Turbo for routine analysis",
                        description=(
                            f"You've spent ${total_gpt4_cost / 100:.2f} on GPT-4. "
                            "GPT-4 Turbo provides similar quality at 66% lower cost."
                        ),
                        estimated_savings_percent=40.0,
                        estimated_savings_cents=int(total_gpt4_cost * 0.4),
                        effort="low",
                        priority=9,
                    )
                )

        # Check for caching opportunities
        if len(token_records) > 50:
            operations: defaultdict[str, int] = defaultdict(int)
            for r in token_records:
                operations[r.operation] += 1
            repeated = {op: count for op, count in operations.items() if count > 10}
            if repeated:
                recommendations.append(
                    OptimizationRecommendation(
                        type=OptimizationType.CACHING,
                        title="Enable verification caching for repeated analyses",
                        description=(
                            f"Found {sum(repeated.values())} repeated operations. "
                            "Caching could eliminate ~30% of LLM calls."
                        ),
                        estimated_savings_percent=30.0,
                        effort="medium",
                        priority=8,
                    )
                )

        # Check for batch verification
        if len(review_costs) > 20:
            avg_cost = statistics.mean(r.total_cost_cents for r in review_costs)
            if avg_cost > 50:  # > $0.50 per review
                recommendations.append(
                    OptimizationRecommendation(
                        type=OptimizationType.BATCH_VERIFICATION,
                        title="Use batch verification for scheduled scans",
                        description=(
                            f"Average review cost is ${avg_cost / 100:.2f}. "
                            "Batching can reduce overhead by 20-30%."
                        ),
                        estimated_savings_percent=25.0,
                        effort="medium",
                        priority=6,
                    )
                )

        recommendations.sort(key=lambda r: r.priority, reverse=True)
        return recommendations


class PerformanceCostDashboardService:
    """Main service for the performance & cost dashboard."""

    def __init__(self, budget: BudgetConfig | None = None) -> None:
        self._calculator = CostCalculator()
        self._optimizer = OptimizationEngine()
        self._budget = budget or BudgetConfig()
        self._token_records: list[TokenUsageRecord] = []
        self._solver_metrics: list[SolverMetric] = []
        self._review_costs: list[ReviewCost] = []
        self._alerts: list[BudgetAlert] = []

    @property
    def budget(self) -> BudgetConfig:
        return self._budget

    def record_token_usage(
        self,
        provider: ModelProvider,
        model: str,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int = 0,
    ) -> TokenUsageRecord:
        """Record LLM token usage."""
        cost = self._calculator.calculate_token_cost(model, input_tokens, output_tokens)
        record = TokenUsageRecord(
            provider=provider,
            model=model,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_cents=cost,
            latency_ms=latency_ms,
        )
        self._token_records.append(record)
        self._check_budget()
        return record

    def record_solver_metric(
        self,
        check_type: str,
        solve_time_ms: int,
        constraint_count: int = 0,
        result: str = "sat",
        memory_mb: float = 0.0,
    ) -> SolverMetric:
        """Record Z3 solver metric."""
        metric = SolverMetric(
            check_type=check_type,
            solve_time_ms=solve_time_ms,
            constraint_count=constraint_count,
            result=result,
            memory_mb=memory_mb,
        )
        self._solver_metrics.append(metric)
        return metric

    def record_review_cost(
        self,
        review_id: str,
        finding_count: int = 0,
    ) -> ReviewCost:
        """Calculate and record cost for a review."""
        recent_tokens = self._token_records[-50:]
        llm_cost = sum(r.cost_cents for r in recent_tokens[-5:])

        recent_solver = self._solver_metrics[-5:]
        solver_cost = sum(
            self._calculator.calculate_solver_cost(m.solve_time_ms) for m in recent_solver
        )

        infra_cost = 0.5  # baseline per review

        cost = ReviewCost(
            review_id=review_id,
            llm_cost_cents=llm_cost,
            solver_cost_cents=solver_cost,
            infra_cost_cents=infra_cost,
            total_cost_cents=llm_cost + solver_cost + infra_cost,
            finding_count=finding_count,
        )
        self._review_costs.append(cost)
        return cost

    def get_dashboard_data(self) -> CostDashboardData:
        """Get aggregated dashboard data."""
        total_cost = sum(r.cost_cents for r in self._token_records)
        total_cost += sum(
            self._calculator.calculate_solver_cost(m.solve_time_ms) for m in self._solver_metrics
        )

        cost_by_model: dict[str, float] = defaultdict(float)
        cost_by_operation: dict[str, float] = defaultdict(float)
        for r in self._token_records:
            cost_by_model[r.model] += r.cost_cents
            cost_by_operation[r.operation] += r.cost_cents

        avg_review_cost = 0.0
        if self._review_costs:
            avg_review_cost = statistics.mean(r.total_cost_cents for r in self._review_costs)

        utilization = (
            total_cost / self._budget.monthly_budget_cents
            if self._budget.monthly_budget_cents > 0
            else 0.0
        )

        recommendations = self._optimizer.analyze(
            self._token_records, self._solver_metrics, self._review_costs
        )

        roi = self._calculate_roi()

        return CostDashboardData(
            total_cost_cents=round(total_cost, 2),
            cost_by_category={
                CostCategory.LLM_TOKENS.value: round(
                    sum(r.cost_cents for r in self._token_records), 2
                ),
                CostCategory.Z3_SOLVER.value: round(
                    sum(
                        self._calculator.calculate_solver_cost(m.solve_time_ms)
                        for m in self._solver_metrics
                    ),
                    2,
                ),
            },
            cost_by_model={k: round(v, 2) for k, v in cost_by_model.items()},
            cost_by_operation={k: round(v, 2) for k, v in cost_by_operation.items()},
            total_tokens=sum(r.total_tokens for r in self._token_records),
            total_solver_time_ms=sum(m.solve_time_ms for m in self._solver_metrics),
            avg_cost_per_review_cents=round(avg_review_cost, 2),
            review_count=len(self._review_costs),
            budget_utilization=round(utilization, 3),
            roi=roi,
            alerts=self._alerts[-10:],
            recommendations=recommendations,
        )

    def _calculate_roi(self) -> ROIMetrics:
        total_findings = sum(r.finding_count for r in self._review_costs)
        critical = total_findings // 10  # Rough estimate
        bug_cost_saved = critical * 50000 + (total_findings - critical) * 5000  # cents
        verification_cost = int(sum(r.total_cost_cents for r in self._review_costs))

        net_savings = bug_cost_saved - verification_cost
        roi_pct = (net_savings / verification_cost * 100) if verification_cost > 0 else 0.0
        hours_saved = len(self._review_costs) * 0.5  # 30 min per manual review

        return ROIMetrics(
            total_reviews=len(self._review_costs),
            total_findings=total_findings,
            critical_findings_caught=critical,
            estimated_bug_cost_saved_cents=bug_cost_saved,
            verification_cost_cents=verification_cost,
            net_savings_cents=net_savings,
            roi_percentage=round(roi_pct, 1),
            manual_review_hours_saved=hours_saved,
        )

    def _check_budget(self) -> None:
        total = sum(r.cost_cents for r in self._token_records)
        utilization = (
            total / self._budget.monthly_budget_cents
            if self._budget.monthly_budget_cents > 0
            else 0.0
        )

        if utilization >= self._budget.critical_threshold and not any(
            a.level == AlertLevel.CRITICAL for a in self._alerts[-5:]
        ):
            self._alerts.append(
                BudgetAlert(
                    level=AlertLevel.CRITICAL,
                    title="Budget Critical",
                    message=f"Spending at {utilization:.0%} of monthly budget",
                    current_spend_cents=total,
                    budget_cents=self._budget.monthly_budget_cents,
                    utilization=utilization,
                )
            )
        elif utilization >= self._budget.warning_threshold and not any(
            a.level == AlertLevel.WARNING for a in self._alerts[-5:]
        ):
            self._alerts.append(
                BudgetAlert(
                    level=AlertLevel.WARNING,
                    title="Budget Warning",
                    message=f"Spending at {utilization:.0%} of monthly budget",
                    current_spend_cents=total,
                    budget_cents=self._budget.monthly_budget_cents,
                    utilization=utilization,
                )
            )

    def get_alerts(self) -> list[BudgetAlert]:
        return list(self._alerts)

    def set_budget(self, budget: BudgetConfig) -> None:
        self._budget = budget


# ─── Singleton Access ──────────────────────────────────────────────────


_cost_dashboard_instance: PerformanceCostDashboardService | None = None


def get_cost_dashboard_service() -> PerformanceCostDashboardService:
    """Get or create the singleton PerformanceCostDashboardService."""
    global _cost_dashboard_instance
    if _cost_dashboard_instance is None:
        _cost_dashboard_instance = PerformanceCostDashboardService()
    return _cost_dashboard_instance


def reset_cost_dashboard_service() -> None:
    """Reset the singleton (for testing)."""
    global _cost_dashboard_instance
    _cost_dashboard_instance = None
