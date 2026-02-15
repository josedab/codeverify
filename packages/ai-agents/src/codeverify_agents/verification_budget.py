"""Verification Budget Optimizer — Intelligent verification depth allocation.

Uses the Behavioral Regression Oracle's risk predictions to allocate
verification compute budget, reducing cost while maintaining coverage.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class VerificationDepth(str, Enum):
    """How deeply to verify a code change."""

    SKIP = "skip"
    PATTERN_ONLY = "pattern_only"
    STATIC_ANALYSIS = "static_analysis"
    AI_REVIEW = "ai_review"
    FORMAL_VERIFICATION = "formal_verification"
    FULL_PIPELINE = "full_pipeline"


# Estimated costs per verification depth (relative units)
DEPTH_COSTS: dict[VerificationDepth, float] = {
    VerificationDepth.SKIP: 0.0,
    VerificationDepth.PATTERN_ONLY: 0.5,
    VerificationDepth.STATIC_ANALYSIS: 2.0,
    VerificationDepth.AI_REVIEW: 10.0,
    VerificationDepth.FORMAL_VERIFICATION: 25.0,
    VerificationDepth.FULL_PIPELINE: 40.0,
}


@dataclass
class BudgetConfig:
    """Configuration for verification budget allocation."""

    daily_budget_units: float = 1000.0
    min_critical_depth: VerificationDepth = VerificationDepth.FORMAL_VERIFICATION
    min_high_depth: VerificationDepth = VerificationDepth.AI_REVIEW
    min_medium_depth: VerificationDepth = VerificationDepth.STATIC_ANALYSIS
    default_depth: VerificationDepth = VerificationDepth.PATTERN_ONLY
    risk_threshold_high: float = 0.7
    risk_threshold_medium: float = 0.3
    reserve_fraction: float = 0.2  # Keep 20% budget for unexpected spikes


@dataclass
class VerificationAllocation:
    """Allocated verification depth for a code change."""

    file_path: str
    risk_score: float
    allocated_depth: VerificationDepth
    estimated_cost: float
    reason: str
    override: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "risk_score": round(self.risk_score, 3),
            "allocated_depth": self.allocated_depth.value,
            "estimated_cost": round(self.estimated_cost, 2),
            "reason": self.reason,
            "override": self.override,
        }


@dataclass
class BudgetUsage:
    """Tracks budget consumption over time."""

    date: str
    allocated: float = 0.0
    consumed: float = 0.0
    files_processed: int = 0
    overrides: int = 0

    @property
    def remaining(self) -> float:
        return max(0, self.allocated - self.consumed)

    @property
    def utilization(self) -> float:
        return self.consumed / self.allocated if self.allocated > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "allocated": round(self.allocated, 2),
            "consumed": round(self.consumed, 2),
            "remaining": round(self.remaining, 2),
            "utilization": f"{self.utilization:.1%}",
            "files_processed": self.files_processed,
            "overrides": self.overrides,
        }


@dataclass
class CostReport:
    """Cost tracking report for verification operations."""

    period_start: datetime
    period_end: datetime
    total_cost: float = 0.0
    total_files: int = 0
    by_depth: dict[str, float] = field(default_factory=dict)
    by_risk_level: dict[str, float] = field(default_factory=dict)
    savings_vs_full: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "total_cost": round(self.total_cost, 2),
            "total_files": self.total_files,
            "savings_vs_full": round(self.savings_vs_full, 2),
            "savings_percent": f"{(self.savings_vs_full / max(1, self.total_cost + self.savings_vs_full)):.0%}",
            "by_depth": {k: round(v, 2) for k, v in self.by_depth.items()},
            "by_risk_level": {k: round(v, 2) for k, v in self.by_risk_level.items()},
        }


class VerificationBudgetOptimizer:
    """Allocates verification depth based on risk scores and available budget.

    Uses risk predictions from the Regression Oracle to intelligently route
    code changes to appropriate verification depth, minimizing cost while
    maintaining high coverage for risky changes.

    Example:
        >>> optimizer = VerificationBudgetOptimizer()
        >>> allocations = optimizer.allocate([
        ...     {"file_path": "auth.py", "risk_score": 0.9},
        ...     {"file_path": "readme.md", "risk_score": 0.05},
        ... ])
        >>> allocations[0].allocated_depth
        VerificationDepth.FORMAL_VERIFICATION
    """

    def __init__(self, config: BudgetConfig | None = None) -> None:
        self.config = config or BudgetConfig()
        self._usage_history: list[BudgetUsage] = []
        self._allocation_log: list[VerificationAllocation] = []
        self._today_usage: BudgetUsage = self._new_day_usage()

    def allocate(
        self,
        changes: list[dict[str, Any]],
        force_depth: VerificationDepth | None = None,
    ) -> list[VerificationAllocation]:
        """Allocate verification depth for a batch of code changes.

        Args:
            changes: List of dicts with 'file_path' and 'risk_score' (0-1).
            force_depth: Optional override to force all changes to a specific depth.

        Returns:
            List of VerificationAllocation objects.
        """
        if force_depth:
            return [
                VerificationAllocation(
                    file_path=c["file_path"],
                    risk_score=c.get("risk_score", 0.5),
                    allocated_depth=force_depth,
                    estimated_cost=DEPTH_COSTS[force_depth],
                    reason="Manual override",
                    override=True,
                )
                for c in changes
            ]

        # Sort by risk score (highest first) to prioritize risky changes
        sorted_changes = sorted(changes, key=lambda c: c.get("risk_score", 0), reverse=True)

        available = self._today_usage.remaining
        reserve = self.config.daily_budget_units * self.config.reserve_fraction
        effective_budget = max(0, available - reserve)

        allocations: list[VerificationAllocation] = []
        spent = 0.0

        for change in sorted_changes:
            file_path = change["file_path"]
            risk_score = change.get("risk_score", 0.5)

            depth = self._select_depth(risk_score)
            cost = DEPTH_COSTS[depth]

            # Downgrade if over budget (but never downgrade critical items)
            if spent + cost > effective_budget and risk_score < self.config.risk_threshold_high:
                depth = self._downgrade_depth(depth)
                cost = DEPTH_COSTS[depth]
                reason = f"Downgraded due to budget (risk={risk_score:.2f})"
            else:
                reason = f"Risk-based allocation (risk={risk_score:.2f})"

            allocation = VerificationAllocation(
                file_path=file_path,
                risk_score=risk_score,
                allocated_depth=depth,
                estimated_cost=cost,
                reason=reason,
            )
            allocations.append(allocation)
            spent += cost

        # Update usage tracking
        total_cost = sum(a.estimated_cost for a in allocations)
        self._today_usage.consumed += total_cost
        self._today_usage.files_processed += len(allocations)
        self._allocation_log.extend(allocations)

        logger.info(
            "Budget allocation complete",
            files=len(allocations),
            total_cost=total_cost,
            budget_remaining=self._today_usage.remaining,
        )

        return allocations

    def record_actual_cost(self, file_path: str, actual_cost: float) -> None:
        """Record the actual cost of a verification (for cost tracking accuracy)."""
        self._today_usage.consumed += actual_cost - sum(
            a.estimated_cost for a in self._allocation_log if a.file_path == file_path
        )

    def get_cost_report(self) -> CostReport:
        """Generate a cost report from allocation history."""
        if not self._allocation_log:
            now = datetime.now(timezone.utc)
            return CostReport(period_start=now, period_end=now)

        by_depth: dict[str, float] = {}
        by_risk: dict[str, float] = {}
        total_cost = 0.0
        full_pipeline_cost = 0.0

        for alloc in self._allocation_log:
            depth_key = alloc.allocated_depth.value
            by_depth[depth_key] = by_depth.get(depth_key, 0) + alloc.estimated_cost
            total_cost += alloc.estimated_cost
            full_pipeline_cost += DEPTH_COSTS[VerificationDepth.FULL_PIPELINE]

            if alloc.risk_score >= self.config.risk_threshold_high:
                risk_key = "high"
            elif alloc.risk_score >= self.config.risk_threshold_medium:
                risk_key = "medium"
            else:
                risk_key = "low"
            by_risk[risk_key] = by_risk.get(risk_key, 0) + alloc.estimated_cost

        return CostReport(
            period_start=self._allocation_log[0].to_dict().get(
                "timestamp", datetime.now(timezone.utc)
            )
            if isinstance(self._allocation_log[0].to_dict().get("timestamp"), datetime)
            else datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            total_cost=total_cost,
            total_files=len(self._allocation_log),
            by_depth=by_depth,
            by_risk_level=by_risk,
            savings_vs_full=full_pipeline_cost - total_cost,
        )

    def get_usage(self) -> dict[str, Any]:
        """Get current budget usage status."""
        return {
            "today": self._today_usage.to_dict(),
            "config": {
                "daily_budget": self.config.daily_budget_units,
                "reserve_fraction": self.config.reserve_fraction,
            },
        }

    def _select_depth(self, risk_score: float) -> VerificationDepth:
        """Select verification depth based on risk score."""
        if risk_score >= self.config.risk_threshold_high:
            return self.config.min_critical_depth
        elif risk_score >= self.config.risk_threshold_medium:
            return self.config.min_medium_depth
        return self.config.default_depth

    def _downgrade_depth(self, depth: VerificationDepth) -> VerificationDepth:
        """Downgrade verification depth by one level."""
        order = list(VerificationDepth)
        idx = order.index(depth)
        if idx > 0:
            return order[idx - 1]
        return depth

    def _new_day_usage(self) -> BudgetUsage:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return BudgetUsage(date=today, allocated=self.config.daily_budget_units)
