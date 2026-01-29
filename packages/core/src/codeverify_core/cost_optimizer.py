"""Verification Cost Optimizer - Smart routing based on risk profile and budget."""

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

import structlog

logger = structlog.get_logger()


class VerificationDepth(str, Enum):
    """Depth of verification to perform."""
    PATTERN = "pattern"  # Fast pattern matching only
    STATIC = "static"  # Static analysis
    AI = "ai"  # LLM-based analysis
    FORMAL = "formal"  # Full Z3 formal verification
    CONSENSUS = "consensus"  # Multi-model consensus


@dataclass
class VerificationCost:
    """Cost metrics for a verification type."""
    depth: VerificationDepth
    avg_time_ms: float
    avg_tokens: int
    avg_cost_usd: float
    accuracy: float  # 0-1 precision
    recall: float  # 0-1 recall


@dataclass
class RiskProfile:
    """Risk profile for a piece of code."""
    code_hash: str
    risk_score: float  # 0-100
    is_critical_path: bool
    is_security_sensitive: bool
    change_size: int  # Lines changed
    complexity: float
    has_previous_bugs: bool
    is_ai_generated: bool


@dataclass
class BudgetConstraints:
    """Budget constraints for verification."""
    max_cost_usd: float | None = None
    max_time_seconds: float | None = None
    max_tokens: int | None = None
    min_accuracy: float = 0.7


@dataclass
class VerificationPlan:
    """Planned verification approach."""
    code_hash: str
    selected_depth: VerificationDepth
    estimated_cost_usd: float
    estimated_time_ms: float
    estimated_accuracy: float
    rationale: list[str] = field(default_factory=list)
    fallback_depth: VerificationDepth | None = None


@dataclass
class CostMetrics:
    """Tracked cost metrics over time."""
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    total_time_ms: float = 0.0
    verifications_by_depth: dict[str, int] = field(default_factory=dict)
    period_start: datetime = field(default_factory=datetime.utcnow)


# Default cost model (can be updated based on actual provider pricing)
DEFAULT_COSTS: dict[VerificationDepth, VerificationCost] = {
    VerificationDepth.PATTERN: VerificationCost(
        depth=VerificationDepth.PATTERN,
        avg_time_ms=10,
        avg_tokens=0,
        avg_cost_usd=0.0001,
        accuracy=0.6,
        recall=0.4,
    ),
    VerificationDepth.STATIC: VerificationCost(
        depth=VerificationDepth.STATIC,
        avg_time_ms=100,
        avg_tokens=0,
        avg_cost_usd=0.001,
        accuracy=0.75,
        recall=0.6,
    ),
    VerificationDepth.AI: VerificationCost(
        depth=VerificationDepth.AI,
        avg_time_ms=3000,
        avg_tokens=2000,
        avg_cost_usd=0.02,
        accuracy=0.85,
        recall=0.8,
    ),
    VerificationDepth.FORMAL: VerificationCost(
        depth=VerificationDepth.FORMAL,
        avg_time_ms=10000,
        avg_tokens=3000,
        avg_cost_usd=0.05,
        accuracy=0.98,
        recall=0.95,
    ),
    VerificationDepth.CONSENSUS: VerificationCost(
        depth=VerificationDepth.CONSENSUS,
        avg_time_ms=15000,
        avg_tokens=8000,
        avg_cost_usd=0.15,
        accuracy=0.99,
        recall=0.97,
    ),
}


class VerificationCostOptimizer:
    """
    Smart routing that chooses verification depth based on code risk
    profile and budget constraints.
    
    Implements a tiered approach:
    - Low-risk code: Pattern matching only
    - Medium-risk: Static analysis + selective AI
    - High-risk: Full AI analysis
    - Critical: Formal verification + consensus
    """

    def __init__(
        self,
        cost_model: dict[VerificationDepth, VerificationCost] | None = None,
        default_budget: BudgetConstraints | None = None,
    ) -> None:
        """Initialize the cost optimizer."""
        self.cost_model = cost_model or DEFAULT_COSTS
        self.default_budget = default_budget or BudgetConstraints()
        
        # Metrics tracking
        self._metrics = CostMetrics()
        self._daily_metrics: dict[str, CostMetrics] = {}
        
        # Risk thresholds
        self._risk_thresholds = {
            "low": 30,
            "medium": 60,
            "high": 80,
        }
        
        # Learning from outcomes
        self._outcome_history: list[dict[str, Any]] = []

    def plan_verification(
        self,
        code: str,
        risk_profile: RiskProfile | None = None,
        budget: BudgetConstraints | None = None,
    ) -> VerificationPlan:
        """
        Plan the verification approach for given code.
        
        Args:
            code: The code to verify
            risk_profile: Pre-computed risk profile (computed if not provided)
            budget: Budget constraints (uses default if not provided)
            
        Returns:
            VerificationPlan with selected depth and estimates
        """
        # Compute risk profile if not provided
        if risk_profile is None:
            risk_profile = self._compute_risk_profile(code)
        
        budget = budget or self.default_budget
        
        # Select verification depth based on risk and budget
        selected_depth, rationale = self._select_depth(risk_profile, budget)
        
        # Get cost estimates
        cost_info = self.cost_model[selected_depth]
        
        # Determine fallback
        fallback = self._get_fallback_depth(selected_depth)
        
        plan = VerificationPlan(
            code_hash=risk_profile.code_hash,
            selected_depth=selected_depth,
            estimated_cost_usd=cost_info.avg_cost_usd,
            estimated_time_ms=cost_info.avg_time_ms,
            estimated_accuracy=cost_info.accuracy,
            rationale=rationale,
            fallback_depth=fallback,
        )
        
        logger.info(
            "Verification plan created",
            code_hash=risk_profile.code_hash[:8],
            depth=selected_depth.value,
            risk_score=risk_profile.risk_score,
            estimated_cost=cost_info.avg_cost_usd,
        )
        
        return plan

    def _compute_risk_profile(self, code: str) -> RiskProfile:
        """Compute risk profile from code."""
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        
        lines = code.split("\n")
        change_size = len(lines)
        
        # Estimate complexity (simple heuristic)
        complexity_indicators = [
            "if ", "else", "for ", "while ", "try:", "except",
            "match ", "case ", "lambda", "async ", "await ",
        ]
        complexity = sum(
            code.count(indicator) for indicator in complexity_indicators
        )
        
        # Check for security-sensitive patterns
        security_patterns = [
            "password", "secret", "token", "key", "auth",
            "sql", "exec", "eval", "shell", "command",
            "crypto", "encrypt", "decrypt", "hash",
        ]
        is_security_sensitive = any(
            pattern in code.lower() for pattern in security_patterns
        )
        
        # Check for AI-generated markers
        ai_markers = [
            "# generated by", "// generated by", "copilot",
            "gpt", "claude", "ai-generated",
        ]
        is_ai_generated = any(
            marker in code.lower() for marker in ai_markers
        )
        
        # Compute risk score
        risk_score = 0.0
        risk_score += min(change_size / 100, 30)  # Size factor
        risk_score += min(complexity / 20, 25)  # Complexity factor
        
        if is_security_sensitive:
            risk_score += 25
        if is_ai_generated:
            risk_score += 15
        
        risk_score = min(risk_score, 100)
        
        return RiskProfile(
            code_hash=code_hash,
            risk_score=risk_score,
            is_critical_path=False,  # Would need context
            is_security_sensitive=is_security_sensitive,
            change_size=change_size,
            complexity=complexity,
            has_previous_bugs=False,  # Would need history
            is_ai_generated=is_ai_generated,
        )

    def _select_depth(
        self,
        risk_profile: RiskProfile,
        budget: BudgetConstraints,
    ) -> tuple[VerificationDepth, list[str]]:
        """Select verification depth based on risk and budget."""
        rationale = []
        
        # Start with depth based on risk
        if risk_profile.risk_score < self._risk_thresholds["low"]:
            base_depth = VerificationDepth.PATTERN
            rationale.append(f"Low risk score ({risk_profile.risk_score:.0f})")
        elif risk_profile.risk_score < self._risk_thresholds["medium"]:
            base_depth = VerificationDepth.STATIC
            rationale.append(f"Medium risk score ({risk_profile.risk_score:.0f})")
        elif risk_profile.risk_score < self._risk_thresholds["high"]:
            base_depth = VerificationDepth.AI
            rationale.append(f"High risk score ({risk_profile.risk_score:.0f})")
        else:
            base_depth = VerificationDepth.FORMAL
            rationale.append(f"Critical risk score ({risk_profile.risk_score:.0f})")
        
        # Escalate for special conditions
        if risk_profile.is_security_sensitive and base_depth.value < VerificationDepth.AI.value:
            base_depth = VerificationDepth.AI
            rationale.append("Security-sensitive code detected")
        
        if risk_profile.is_ai_generated and base_depth.value < VerificationDepth.AI.value:
            base_depth = VerificationDepth.AI
            rationale.append("AI-generated code detected")
        
        if risk_profile.is_critical_path:
            base_depth = VerificationDepth.FORMAL
            rationale.append("Critical path code")
        
        # Check budget constraints
        selected_depth = base_depth
        cost_info = self.cost_model[selected_depth]
        
        if budget.max_cost_usd is not None and cost_info.avg_cost_usd > budget.max_cost_usd:
            # Need to downgrade
            for depth in [VerificationDepth.AI, VerificationDepth.STATIC, VerificationDepth.PATTERN]:
                if self.cost_model[depth].avg_cost_usd <= budget.max_cost_usd:
                    selected_depth = depth
                    rationale.append(f"Downgraded due to cost budget (${budget.max_cost_usd})")
                    break
        
        if budget.max_time_seconds is not None:
            max_time_ms = budget.max_time_seconds * 1000
            if cost_info.avg_time_ms > max_time_ms:
                for depth in [VerificationDepth.AI, VerificationDepth.STATIC, VerificationDepth.PATTERN]:
                    if self.cost_model[depth].avg_time_ms <= max_time_ms:
                        selected_depth = depth
                        rationale.append(f"Downgraded due to time budget ({budget.max_time_seconds}s)")
                        break
        
        # Check accuracy requirement
        if self.cost_model[selected_depth].accuracy < budget.min_accuracy:
            # Need to upgrade if possible
            for depth in [VerificationDepth.FORMAL, VerificationDepth.CONSENSUS]:
                if self.cost_model[depth].accuracy >= budget.min_accuracy:
                    # Check if within budget
                    if budget.max_cost_usd is None or self.cost_model[depth].avg_cost_usd <= budget.max_cost_usd:
                        selected_depth = depth
                        rationale.append(f"Upgraded to meet accuracy requirement ({budget.min_accuracy})")
                        break
        
        return selected_depth, rationale

    def _get_fallback_depth(
        self, selected: VerificationDepth
    ) -> VerificationDepth | None:
        """Get fallback verification depth if selected fails."""
        depth_order = [
            VerificationDepth.PATTERN,
            VerificationDepth.STATIC,
            VerificationDepth.AI,
            VerificationDepth.FORMAL,
            VerificationDepth.CONSENSUS,
        ]
        
        current_idx = depth_order.index(selected)
        
        # Fallback is one level down
        if current_idx > 0:
            return depth_order[current_idx - 1]
        return None

    def record_outcome(
        self,
        plan: VerificationPlan,
        actual_cost_usd: float,
        actual_time_ms: float,
        actual_tokens: int,
        found_issues: int,
        false_positives: int,
    ) -> None:
        """Record outcome for learning and metrics."""
        # Update metrics
        self._metrics.total_cost_usd += actual_cost_usd
        self._metrics.total_tokens += actual_tokens
        self._metrics.total_time_ms += actual_time_ms
        
        depth_key = plan.selected_depth.value
        self._metrics.verifications_by_depth[depth_key] = (
            self._metrics.verifications_by_depth.get(depth_key, 0) + 1
        )
        
        # Update daily metrics
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if today not in self._daily_metrics:
            self._daily_metrics[today] = CostMetrics()
        
        daily = self._daily_metrics[today]
        daily.total_cost_usd += actual_cost_usd
        daily.total_tokens += actual_tokens
        daily.total_time_ms += actual_time_ms
        daily.verifications_by_depth[depth_key] = (
            daily.verifications_by_depth.get(depth_key, 0) + 1
        )
        
        # Record for learning
        self._outcome_history.append({
            "code_hash": plan.code_hash,
            "depth": plan.selected_depth.value,
            "estimated_cost": plan.estimated_cost_usd,
            "actual_cost": actual_cost_usd,
            "estimated_time": plan.estimated_time_ms,
            "actual_time": actual_time_ms,
            "found_issues": found_issues,
            "false_positives": false_positives,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Keep history bounded
        if len(self._outcome_history) > 10000:
            self._outcome_history = self._outcome_history[-5000:]
        
        logger.info(
            "Verification outcome recorded",
            depth=plan.selected_depth.value,
            cost=actual_cost_usd,
            issues=found_issues,
        )

    def update_cost_model(self) -> None:
        """Update cost model based on recorded outcomes."""
        if len(self._outcome_history) < 100:
            return  # Need more data
        
        # Group by depth
        by_depth: dict[str, list[dict]] = {}
        for outcome in self._outcome_history[-1000:]:
            depth = outcome["depth"]
            if depth not in by_depth:
                by_depth[depth] = []
            by_depth[depth].append(outcome)
        
        # Update model for each depth
        for depth_str, outcomes in by_depth.items():
            if len(outcomes) < 10:
                continue
            
            depth = VerificationDepth(depth_str)
            
            # Calculate averages
            avg_cost = sum(o["actual_cost"] for o in outcomes) / len(outcomes)
            avg_time = sum(o["actual_time"] for o in outcomes) / len(outcomes)
            
            # Update model with exponential moving average
            alpha = 0.2
            current = self.cost_model[depth]
            
            self.cost_model[depth] = VerificationCost(
                depth=depth,
                avg_time_ms=alpha * avg_time + (1 - alpha) * current.avg_time_ms,
                avg_tokens=current.avg_tokens,  # Keep existing
                avg_cost_usd=alpha * avg_cost + (1 - alpha) * current.avg_cost_usd,
                accuracy=current.accuracy,  # Keep existing
                recall=current.recall,  # Keep existing
            )
        
        logger.info("Cost model updated from outcomes")

    def get_budget_usage(
        self,
        period: str = "all",  # "all", "today", "week", "month"
    ) -> dict[str, Any]:
        """Get budget usage statistics."""
        if period == "today":
            today = datetime.utcnow().strftime("%Y-%m-%d")
            metrics = self._daily_metrics.get(today, CostMetrics())
        elif period == "week":
            # Sum last 7 days
            metrics = CostMetrics()
            for i in range(7):
                day = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
                if day in self._daily_metrics:
                    daily = self._daily_metrics[day]
                    metrics.total_cost_usd += daily.total_cost_usd
                    metrics.total_tokens += daily.total_tokens
                    metrics.total_time_ms += daily.total_time_ms
                    for k, v in daily.verifications_by_depth.items():
                        metrics.verifications_by_depth[k] = (
                            metrics.verifications_by_depth.get(k, 0) + v
                        )
        elif period == "month":
            # Sum last 30 days
            metrics = CostMetrics()
            for i in range(30):
                day = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
                if day in self._daily_metrics:
                    daily = self._daily_metrics[day]
                    metrics.total_cost_usd += daily.total_cost_usd
                    metrics.total_tokens += daily.total_tokens
                    metrics.total_time_ms += daily.total_time_ms
                    for k, v in daily.verifications_by_depth.items():
                        metrics.verifications_by_depth[k] = (
                            metrics.verifications_by_depth.get(k, 0) + v
                        )
        else:
            metrics = self._metrics
        
        total_verifications = sum(metrics.verifications_by_depth.values())
        
        return {
            "period": period,
            "total_cost_usd": round(metrics.total_cost_usd, 4),
            "total_tokens": metrics.total_tokens,
            "total_time_seconds": round(metrics.total_time_ms / 1000, 2),
            "total_verifications": total_verifications,
            "avg_cost_per_verification": (
                round(metrics.total_cost_usd / total_verifications, 4)
                if total_verifications > 0 else 0
            ),
            "verifications_by_depth": metrics.verifications_by_depth,
            "depth_distribution": {
                k: round(v / total_verifications * 100, 1)
                for k, v in metrics.verifications_by_depth.items()
            } if total_verifications > 0 else {},
        }

    def optimize_batch(
        self,
        code_items: list[tuple[str, RiskProfile | None]],
        total_budget: BudgetConstraints,
    ) -> list[VerificationPlan]:
        """
        Optimize verification for a batch of code items within a total budget.
        
        Args:
            code_items: List of (code, risk_profile) tuples
            total_budget: Total budget for all verifications
            
        Returns:
            List of VerificationPlans optimized for the budget
        """
        # First, compute all risk profiles
        items_with_risk = []
        for code, risk_profile in code_items:
            if risk_profile is None:
                risk_profile = self._compute_risk_profile(code)
            items_with_risk.append((code, risk_profile))
        
        # Sort by risk (highest first)
        items_with_risk.sort(key=lambda x: x[1].risk_score, reverse=True)
        
        # Allocate budget
        plans = []
        remaining_cost = total_budget.max_cost_usd or float("inf")
        remaining_time = (total_budget.max_time_seconds or float("inf")) * 1000
        
        for code, risk_profile in items_with_risk:
            # Create individual budget
            item_budget = BudgetConstraints(
                max_cost_usd=remaining_cost / max(1, len(items_with_risk) - len(plans)),
                max_time_seconds=remaining_time / 1000 / max(1, len(items_with_risk) - len(plans)),
                min_accuracy=total_budget.min_accuracy,
            )
            
            plan = self.plan_verification(code, risk_profile, item_budget)
            plans.append(plan)
            
            # Update remaining budget
            remaining_cost -= plan.estimated_cost_usd
            remaining_time -= plan.estimated_time_ms
        
        return plans

    def suggest_budget(
        self,
        code_items: list[tuple[str, RiskProfile | None]],
        target_accuracy: float = 0.9,
    ) -> BudgetConstraints:
        """
        Suggest a budget for verifying a set of code items.
        
        Args:
            code_items: List of (code, risk_profile) tuples
            target_accuracy: Desired minimum accuracy
            
        Returns:
            Suggested BudgetConstraints
        """
        total_cost = 0.0
        total_time = 0.0
        
        for code, risk_profile in code_items:
            if risk_profile is None:
                risk_profile = self._compute_risk_profile(code)
            
            # Determine appropriate depth for accuracy
            for depth in [VerificationDepth.PATTERN, VerificationDepth.STATIC,
                         VerificationDepth.AI, VerificationDepth.FORMAL]:
                if self.cost_model[depth].accuracy >= target_accuracy:
                    cost_info = self.cost_model[depth]
                    total_cost += cost_info.avg_cost_usd
                    total_time += cost_info.avg_time_ms
                    break
        
        return BudgetConstraints(
            max_cost_usd=round(total_cost * 1.2, 2),  # 20% buffer
            max_time_seconds=round(total_time * 1.2 / 1000, 1),
            min_accuracy=target_accuracy,
        )

    def set_risk_thresholds(
        self,
        low: int = 30,
        medium: int = 60,
        high: int = 80,
    ) -> None:
        """Update risk thresholds for depth selection."""
        self._risk_thresholds = {
            "low": low,
            "medium": medium,
            "high": high,
        }
        logger.info("Risk thresholds updated", thresholds=self._risk_thresholds)

    def get_cost_model(self) -> dict[str, dict[str, Any]]:
        """Get current cost model as dictionary."""
        return {
            depth.value: {
                "avg_time_ms": cost.avg_time_ms,
                "avg_tokens": cost.avg_tokens,
                "avg_cost_usd": cost.avg_cost_usd,
                "accuracy": cost.accuracy,
                "recall": cost.recall,
            }
            for depth, cost in self.cost_model.items()
        }
