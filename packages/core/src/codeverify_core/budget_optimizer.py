"""Verification Budget Optimizer - ML-based verification routing.

Routes verification requests to appropriate depth based on risk factors,
optimizing cost while maintaining quality for high-volume customers.
"""

import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class VerificationDepth(str, Enum):
    """Depth of verification."""
    PATTERN = "pattern"  # Fast pattern matching only
    STATIC = "static"  # Static analysis
    AI = "ai"  # LLM-based analysis
    FORMAL = "formal"  # Full Z3 formal verification
    FULL = "full"  # All methods combined


@dataclass
class RiskFactors:
    """Factors contributing to risk score."""
    file_complexity: float = 0.0  # 0-1
    change_size: float = 0.0  # 0-1
    historical_bug_rate: float = 0.0  # 0-1
    author_experience: float = 0.0  # 0-1, higher = more experienced
    file_criticality: float = 0.0  # 0-1, based on path patterns
    ai_generated_probability: float = 0.0  # 0-1
    has_security_patterns: bool = False
    is_public_api: bool = False
    test_coverage: float = 0.0  # 0-1


@dataclass
class CostModel:
    """Cost model for verification operations."""
    pattern_cost: float = 0.01  # $ per verification
    static_cost: float = 0.05
    ai_cost: float = 0.10  # LLM API cost
    formal_cost: float = 0.25  # Z3 compute cost
    
    pattern_time_ms: float = 50
    static_time_ms: float = 200
    ai_time_ms: float = 2000
    formal_time_ms: float = 5000


@dataclass
class Budget:
    """Verification budget constraints."""
    max_cost_per_pr: float = 5.0  # $
    max_time_per_pr_ms: float = 60000  # 1 minute
    max_cost_per_file: float = 1.0
    remaining_monthly_budget: float = 1000.0
    tier: str = "standard"  # free, standard, premium


@dataclass
class VerificationDecision:
    """Decision about how to verify a piece of code."""
    file_path: str
    depth: VerificationDepth
    risk_score: float
    estimated_cost: float
    estimated_time_ms: float
    reasoning: list[str]
    skip_checks: list[str] = field(default_factory=list)


@dataclass
class BatchOptimizationResult:
    """Result of optimizing verification for a batch of files."""
    decisions: list[VerificationDecision]
    total_estimated_cost: float
    total_estimated_time_ms: float
    budget_utilization: float
    files_at_risk: list[str]


class RiskScorer:
    """Scores risk for verification prioritization."""

    # Critical file patterns that need full verification
    CRITICAL_PATTERNS = [
        r"auth",
        r"security",
        r"payment",
        r"crypto",
        r"password",
        r"token",
        r"secret",
        r"admin",
        r"permission",
        r"api/v\d+",
    ]

    # Low-risk patterns that can use lighter verification
    LOW_RISK_PATTERNS = [
        r"test",
        r"mock",
        r"fixture",
        r"__pycache__",
        r"\.d\.ts$",
        r"types?\.",
        r"constants?",
        r"config",
    ]

    def score(self, factors: RiskFactors, file_path: str) -> float:
        """Calculate risk score from 0-1."""
        import re
        
        # Base score from factors
        score = 0.0
        
        # Complexity contributes 20%
        score += factors.file_complexity * 0.20
        
        # Change size contributes 15%
        score += factors.change_size * 0.15
        
        # Historical bugs contribute 25%
        score += factors.historical_bug_rate * 0.25
        
        # Author experience (inverse - less experience = higher risk)
        score += (1 - factors.author_experience) * 0.10
        
        # AI-generated code needs more scrutiny
        score += factors.ai_generated_probability * 0.15
        
        # File criticality
        score += factors.file_criticality * 0.15
        
        # Modifiers based on patterns
        file_lower = file_path.lower()
        
        # Critical patterns increase risk
        for pattern in self.CRITICAL_PATTERNS:
            if re.search(pattern, file_lower):
                score = min(score * 1.5, 1.0)
                break
        
        # Low-risk patterns decrease risk
        for pattern in self.LOW_RISK_PATTERNS:
            if re.search(pattern, file_lower):
                score = score * 0.5
                break
        
        # Boolean factors
        if factors.has_security_patterns:
            score = min(score + 0.2, 1.0)
        
        if factors.is_public_api:
            score = min(score + 0.1, 1.0)
        
        # Good test coverage reduces risk
        if factors.test_coverage > 0.8:
            score = score * 0.7
        
        return min(max(score, 0.0), 1.0)


class CostEstimator:
    """Estimates cost and time for verification operations."""

    def __init__(self, cost_model: CostModel | None = None) -> None:
        """Initialize with cost model."""
        self.model = cost_model or CostModel()

    def estimate(
        self,
        depth: VerificationDepth,
        file_size_lines: int,
    ) -> tuple[float, float]:
        """Estimate cost and time for verification."""
        # Scale factor based on file size
        size_factor = math.log2(max(file_size_lines, 10)) / 10
        
        if depth == VerificationDepth.PATTERN:
            cost = self.model.pattern_cost * size_factor
            time = self.model.pattern_time_ms * size_factor
        elif depth == VerificationDepth.STATIC:
            cost = self.model.static_cost * size_factor
            time = self.model.static_time_ms * size_factor
        elif depth == VerificationDepth.AI:
            cost = self.model.ai_cost * size_factor
            time = self.model.ai_time_ms * size_factor
        elif depth == VerificationDepth.FORMAL:
            cost = self.model.formal_cost * size_factor
            time = self.model.formal_time_ms * size_factor
        else:  # FULL
            cost = (
                self.model.pattern_cost +
                self.model.static_cost +
                self.model.ai_cost +
                self.model.formal_cost
            ) * size_factor
            time = (
                self.model.pattern_time_ms +
                self.model.static_time_ms +
                self.model.ai_time_ms +
                self.model.formal_time_ms
            ) * size_factor
        
        return cost, time


class DepthSelector:
    """Selects verification depth based on risk and budget."""

    def __init__(
        self,
        cost_estimator: CostEstimator | None = None,
    ) -> None:
        """Initialize selector."""
        self.estimator = cost_estimator or CostEstimator()
        
        # Risk thresholds for depth selection
        self.thresholds = {
            "pattern_max": 0.2,  # Below this: pattern only
            "static_max": 0.4,  # Below this: static analysis
            "ai_max": 0.7,  # Below this: AI analysis
            "formal_min": 0.7,  # Above this: formal verification
        }

    def select(
        self,
        risk_score: float,
        budget: Budget,
        file_size_lines: int,
    ) -> tuple[VerificationDepth, list[str]]:
        """Select verification depth based on risk and budget."""
        reasoning = []
        
        # Check tier-based restrictions
        if budget.tier == "free":
            reasoning.append("Free tier: limited to pattern matching")
            return VerificationDepth.PATTERN, reasoning
        
        # Risk-based selection
        if risk_score < self.thresholds["pattern_max"]:
            depth = VerificationDepth.PATTERN
            reasoning.append(f"Low risk ({risk_score:.2f}): pattern matching sufficient")
        elif risk_score < self.thresholds["static_max"]:
            depth = VerificationDepth.STATIC
            reasoning.append(f"Medium-low risk ({risk_score:.2f}): static analysis")
        elif risk_score < self.thresholds["ai_max"]:
            depth = VerificationDepth.AI
            reasoning.append(f"Medium risk ({risk_score:.2f}): AI analysis")
        else:
            depth = VerificationDepth.FORMAL
            reasoning.append(f"High risk ({risk_score:.2f}): formal verification required")
        
        # Check budget constraints
        estimated_cost, estimated_time = self.estimator.estimate(depth, file_size_lines)
        
        if estimated_cost > budget.max_cost_per_file:
            # Downgrade depth to fit budget
            original_depth = depth
            while estimated_cost > budget.max_cost_per_file and depth != VerificationDepth.PATTERN:
                depth = self._downgrade_depth(depth)
                estimated_cost, _ = self.estimator.estimate(depth, file_size_lines)
            
            reasoning.append(
                f"Downgraded from {original_depth.value} to {depth.value} due to cost budget"
            )
        
        # Premium tier gets full verification for high-risk
        if budget.tier == "premium" and risk_score > 0.8:
            depth = VerificationDepth.FULL
            reasoning.append("Premium tier: full verification for high-risk code")
        
        return depth, reasoning

    def _downgrade_depth(self, depth: VerificationDepth) -> VerificationDepth:
        """Downgrade to next cheaper verification depth."""
        order = [
            VerificationDepth.FULL,
            VerificationDepth.FORMAL,
            VerificationDepth.AI,
            VerificationDepth.STATIC,
            VerificationDepth.PATTERN,
        ]
        
        idx = order.index(depth)
        if idx < len(order) - 1:
            return order[idx + 1]
        return depth


class OutcomeLearner:
    """Learns from verification outcomes to improve routing."""

    def __init__(self) -> None:
        """Initialize the learner."""
        self._outcomes: list[dict[str, Any]] = []
        self._feature_weights: dict[str, float] = {
            "complexity": 1.0,
            "change_size": 1.0,
            "historical_bugs": 1.0,
            "author_experience": 1.0,
            "ai_generated": 1.0,
        }

    def record_outcome(
        self,
        factors: RiskFactors,
        depth_used: VerificationDepth,
        found_issues: int,
        false_positives: int,
    ) -> None:
        """Record a verification outcome for learning."""
        self._outcomes.append({
            "factors": factors,
            "depth": depth_used,
            "issues": found_issues,
            "false_positives": false_positives,
            "timestamp": datetime.utcnow(),
        })
        
        # Simple weight adjustment based on outcomes
        self._adjust_weights(factors, depth_used, found_issues, false_positives)
        
        # Keep only recent outcomes
        if len(self._outcomes) > 10000:
            self._outcomes = self._outcomes[-5000:]

    def _adjust_weights(
        self,
        factors: RiskFactors,
        depth: VerificationDepth,
        issues: int,
        false_positives: int,
    ) -> None:
        """Adjust feature weights based on outcome."""
        learning_rate = 0.01
        
        # If we found issues with lighter verification, complexity was underweighted
        if issues > 0 and depth in (VerificationDepth.PATTERN, VerificationDepth.STATIC):
            self._feature_weights["complexity"] *= (1 + learning_rate)
        
        # If we had false positives with heavy verification on AI code
        if false_positives > issues and factors.ai_generated_probability > 0.7:
            self._feature_weights["ai_generated"] *= (1 - learning_rate)

    def get_adjusted_factors(self, factors: RiskFactors) -> RiskFactors:
        """Apply learned weights to factors."""
        return RiskFactors(
            file_complexity=factors.file_complexity * self._feature_weights["complexity"],
            change_size=factors.change_size * self._feature_weights["change_size"],
            historical_bug_rate=factors.historical_bug_rate * self._feature_weights["historical_bugs"],
            author_experience=factors.author_experience * self._feature_weights["author_experience"],
            ai_generated_probability=factors.ai_generated_probability * self._feature_weights["ai_generated"],
            file_criticality=factors.file_criticality,
            has_security_patterns=factors.has_security_patterns,
            is_public_api=factors.is_public_api,
            test_coverage=factors.test_coverage,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get learning statistics."""
        if not self._outcomes:
            return {"total_outcomes": 0}
        
        total = len(self._outcomes)
        total_issues = sum(o["issues"] for o in self._outcomes)
        total_fp = sum(o["false_positives"] for o in self._outcomes)
        
        return {
            "total_outcomes": total,
            "total_issues_found": total_issues,
            "total_false_positives": total_fp,
            "precision": total_issues / max(total_issues + total_fp, 1),
            "feature_weights": self._feature_weights.copy(),
        }


class VerificationBudgetOptimizer:
    """
    Optimizes verification routing based on risk and budget.
    
    Routes verification to appropriate depth (pattern -> static -> AI -> formal)
    based on file risk factors and available budget.
    """

    def __init__(
        self,
        cost_model: CostModel | None = None,
        enable_learning: bool = True,
    ) -> None:
        """Initialize the optimizer."""
        self.scorer = RiskScorer()
        self.estimator = CostEstimator(cost_model)
        self.selector = DepthSelector(self.estimator)
        self.learner = OutcomeLearner() if enable_learning else None
        
        # Usage tracking
        self._usage: dict[str, float] = {
            "total_cost": 0.0,
            "total_files": 0,
            "total_time_ms": 0.0,
        }

    def optimize_file(
        self,
        file_path: str,
        file_size_lines: int,
        factors: RiskFactors,
        budget: Budget,
    ) -> VerificationDecision:
        """Optimize verification for a single file."""
        # Apply learned adjustments
        if self.learner:
            factors = self.learner.get_adjusted_factors(factors)
        
        # Calculate risk score
        risk_score = self.scorer.score(factors, file_path)
        
        # Select depth based on risk and budget
        depth, reasoning = self.selector.select(risk_score, budget, file_size_lines)
        
        # Estimate cost and time
        estimated_cost, estimated_time = self.estimator.estimate(depth, file_size_lines)
        
        # Determine which checks to skip for efficiency
        skip_checks = self._determine_skip_checks(factors, depth)
        
        decision = VerificationDecision(
            file_path=file_path,
            depth=depth,
            risk_score=risk_score,
            estimated_cost=estimated_cost,
            estimated_time_ms=estimated_time,
            reasoning=reasoning,
            skip_checks=skip_checks,
        )
        
        logger.info(
            "Verification decision made",
            file_path=file_path,
            depth=depth.value,
            risk_score=risk_score,
            estimated_cost=estimated_cost,
        )
        
        return decision

    def optimize_batch(
        self,
        files: list[dict[str, Any]],
        budget: Budget,
    ) -> BatchOptimizationResult:
        """
        Optimize verification for a batch of files.
        
        Uses a knapsack-like approach to maximize coverage within budget.
        
        Args:
            files: List of dicts with file_path, size_lines, and factors
            budget: Budget constraints
            
        Returns:
            Optimized decisions for all files
        """
        decisions: list[VerificationDecision] = []
        total_cost = 0.0
        total_time = 0.0
        files_at_risk = []
        
        # First pass: calculate risk scores and sort by risk
        scored_files = []
        for file_info in files:
            factors = file_info.get("factors", RiskFactors())
            risk_score = self.scorer.score(factors, file_info["file_path"])
            scored_files.append({
                **file_info,
                "risk_score": risk_score,
            })
        
        # Sort by risk descending - high risk files get budget priority
        scored_files.sort(key=lambda x: x["risk_score"], reverse=True)
        
        remaining_cost = budget.max_cost_per_pr
        remaining_time = budget.max_time_per_pr_ms
        
        for file_info in scored_files:
            file_path = file_info["file_path"]
            size_lines = file_info.get("size_lines", 100)
            factors = file_info.get("factors", RiskFactors())
            risk_score = file_info["risk_score"]
            
            # Create per-file budget based on remaining
            file_budget = Budget(
                max_cost_per_pr=remaining_cost,
                max_time_per_pr_ms=remaining_time,
                max_cost_per_file=min(budget.max_cost_per_file, remaining_cost),
                remaining_monthly_budget=budget.remaining_monthly_budget,
                tier=budget.tier,
            )
            
            decision = self.optimize_file(file_path, size_lines, factors, file_budget)
            decisions.append(decision)
            
            # Update remaining budget
            remaining_cost -= decision.estimated_cost
            remaining_time -= decision.estimated_time_ms
            total_cost += decision.estimated_cost
            total_time += decision.estimated_time_ms
            
            # Track high-risk files that got downgraded
            if risk_score > 0.7 and decision.depth != VerificationDepth.FORMAL:
                files_at_risk.append(file_path)
        
        budget_utilization = total_cost / budget.max_cost_per_pr if budget.max_cost_per_pr > 0 else 0
        
        logger.info(
            "Batch optimization complete",
            total_files=len(files),
            total_cost=total_cost,
            budget_utilization=budget_utilization,
            files_at_risk=len(files_at_risk),
        )
        
        return BatchOptimizationResult(
            decisions=decisions,
            total_estimated_cost=total_cost,
            total_estimated_time_ms=total_time,
            budget_utilization=budget_utilization,
            files_at_risk=files_at_risk,
        )

    def _determine_skip_checks(
        self,
        factors: RiskFactors,
        depth: VerificationDepth,
    ) -> list[str]:
        """Determine which checks to skip for efficiency."""
        skip = []
        
        # Skip certain checks based on context
        if factors.test_coverage > 0.9:
            skip.append("redundant_coverage_check")
        
        if depth == VerificationDepth.PATTERN:
            skip.extend(["semantic_analysis", "formal_verification"])
        elif depth == VerificationDepth.STATIC:
            skip.append("formal_verification")
        
        if not factors.has_security_patterns:
            skip.append("deep_security_scan")
        
        return skip

    def record_outcome(
        self,
        decision: VerificationDecision,
        found_issues: int,
        false_positives: int,
        actual_cost: float | None = None,
    ) -> None:
        """Record verification outcome for learning."""
        if self.learner:
            # We need to reconstruct factors - in practice, store with decision
            factors = RiskFactors()  # Would be stored with decision
            self.learner.record_outcome(
                factors,
                decision.depth,
                found_issues,
                false_positives,
            )
        
        # Update usage tracking
        cost = actual_cost if actual_cost is not None else decision.estimated_cost
        self._usage["total_cost"] += cost
        self._usage["total_files"] += 1
        self._usage["total_time_ms"] += decision.estimated_time_ms

    def get_usage_report(self) -> dict[str, Any]:
        """Get usage report."""
        report = self._usage.copy()
        
        if self.learner:
            report["learning_stats"] = self.learner.get_statistics()
        
        return report

    def estimate_monthly_usage(
        self,
        avg_prs_per_day: int,
        avg_files_per_pr: int,
    ) -> dict[str, float]:
        """Estimate monthly usage and cost."""
        if self._usage["total_files"] == 0:
            return {"error": "No historical data"}
        
        avg_cost_per_file = self._usage["total_cost"] / self._usage["total_files"]
        
        daily_files = avg_prs_per_day * avg_files_per_pr
        monthly_files = daily_files * 30
        
        return {
            "avg_cost_per_file": avg_cost_per_file,
            "estimated_monthly_files": monthly_files,
            "estimated_monthly_cost": monthly_files * avg_cost_per_file,
            "estimated_daily_cost": daily_files * avg_cost_per_file,
        }
