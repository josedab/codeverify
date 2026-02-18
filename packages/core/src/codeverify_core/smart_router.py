"""Verification Cost Optimizer (Smart Router).

Risk-based routing: pattern matching (free) → static analysis (cheap)
→ AI analysis (moderate) → full Z3 verification (expensive). Budget-
aware depth selection per file.

Features:
- Per-file/function risk scoring from change metrics
- Multi-tier verification depth routing
- Budget constraint enforcement
- Cost estimation before execution
- A/B testing for routing strategies
- Usage metrics and reporting
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class VerificationDepth(str, Enum):
    PATTERN = "pattern"
    STATIC = "static"
    AI = "ai"
    FORMAL = "formal"


class RiskBucket(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


@dataclass
class DepthConfig:
    """Configuration for a verification depth tier."""
    depth: VerificationDepth = VerificationDepth.PATTERN
    cost_cents_per_file: float = 0.0
    latency_ms: int = 100
    accuracy: float = 0.5
    checks: list[str] = field(default_factory=list)

    @classmethod
    def all_tiers(cls) -> dict[VerificationDepth, DepthConfig]:
        return {
            VerificationDepth.PATTERN: cls(
                depth=VerificationDepth.PATTERN, cost_cents_per_file=0.0,
                latency_ms=50, accuracy=0.4,
                checks=["regex_patterns", "known_bad_patterns"],
            ),
            VerificationDepth.STATIC: cls(
                depth=VerificationDepth.STATIC, cost_cents_per_file=0.5,
                latency_ms=200, accuracy=0.6,
                checks=["ast_analysis", "type_checking", "lint_rules"],
            ),
            VerificationDepth.AI: cls(
                depth=VerificationDepth.AI, cost_cents_per_file=5.0,
                latency_ms=5000, accuracy=0.85,
                checks=["semantic_analysis", "security_scan", "trust_score"],
            ),
            VerificationDepth.FORMAL: cls(
                depth=VerificationDepth.FORMAL, cost_cents_per_file=8.0,
                latency_ms=3000, accuracy=0.95,
                checks=["z3_null_safety", "z3_bounds", "z3_overflow", "z3_division"],
            ),
        }


@dataclass
class FileRiskScore:
    """Risk assessment for a single file."""
    file_path: str = ""
    risk_score: float = 0.5
    risk_bucket: RiskBucket = RiskBucket.MEDIUM
    recommended_depth: VerificationDepth = VerificationDepth.STATIC
    factors: dict[str, float] = field(default_factory=dict)
    estimated_cost_cents: float = 0.0


@dataclass
class RoutingDecision:
    """Routing decision for a set of files."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    file_routes: list[FileRiskScore] = field(default_factory=list)
    total_estimated_cost_cents: float = 0.0
    budget_cents: float = 50.0
    files_at_depth: dict[str, int] = field(default_factory=dict)
    budget_utilized: float = 0.0


@dataclass
class RoutingStats:
    """Statistics about routing decisions."""
    total_files_routed: int = 0
    depth_distribution: dict[str, int] = field(default_factory=dict)
    total_cost_saved_cents: float = 0.0
    avg_risk_score: float = 0.0
    accuracy_estimate: float = 0.0


class RiskScorer:
    """Scores files by risk based on change metrics."""

    RISK_FACTORS = {
        "change_size": 0.3,
        "file_criticality": 0.25,
        "author_experience": 0.15,
        "file_age_risk": 0.1,
        "dependency_count": 0.1,
        "past_bug_rate": 0.1,
    }

    HIGH_RISK_PATTERNS = ["auth", "payment", "crypto", "security", "password", "token", "secret"]
    LOW_RISK_PATTERNS = ["test", "docs", "readme", "changelog", "config"]

    def score(
        self,
        file_path: str,
        change_lines: int = 0,
        file_total_lines: int = 100,
        is_new_file: bool = False,
        author_commits: int = 10,
    ) -> FileRiskScore:
        """Calculate risk score for a file."""
        factors: dict[str, float] = {}

        factors["change_size"] = min(1.0, change_lines / 200)

        path_lower = file_path.lower()
        if any(p in path_lower for p in self.HIGH_RISK_PATTERNS):
            factors["file_criticality"] = 0.9
        elif any(p in path_lower for p in self.LOW_RISK_PATTERNS):
            factors["file_criticality"] = 0.1
        else:
            factors["file_criticality"] = 0.5

        factors["author_experience"] = max(0.0, 1.0 - (author_commits / 100))
        factors["file_age_risk"] = 0.8 if is_new_file else 0.3
        factors["dependency_count"] = 0.3
        factors["past_bug_rate"] = 0.3

        score = sum(factors[k] * self.RISK_FACTORS[k] for k in factors if k in self.RISK_FACTORS)
        score = round(min(1.0, max(0.0, score)), 3)

        if score >= 0.8:
            bucket = RiskBucket.CRITICAL
        elif score >= 0.6:
            bucket = RiskBucket.HIGH
        elif score >= 0.4:
            bucket = RiskBucket.MEDIUM
        elif score >= 0.2:
            bucket = RiskBucket.LOW
        else:
            bucket = RiskBucket.MINIMAL

        return FileRiskScore(
            file_path=file_path, risk_score=score, risk_bucket=bucket,
            factors=factors,
        )


class DepthRouter:
    """Routes files to appropriate verification depths based on risk and budget."""

    RISK_TO_DEPTH: dict[RiskBucket, VerificationDepth] = {
        RiskBucket.CRITICAL: VerificationDepth.FORMAL,
        RiskBucket.HIGH: VerificationDepth.AI,
        RiskBucket.MEDIUM: VerificationDepth.STATIC,
        RiskBucket.LOW: VerificationDepth.PATTERN,
        RiskBucket.MINIMAL: VerificationDepth.PATTERN,
    }

    def __init__(self) -> None:
        self._tiers = DepthConfig.all_tiers()

    def route(
        self,
        risk_scores: list[FileRiskScore],
        budget_cents: float = 50.0,
    ) -> RoutingDecision:
        """Route files to verification depths within budget."""
        # Sort by risk (highest first to prioritize)
        sorted_scores = sorted(risk_scores, key=lambda r: r.risk_score, reverse=True)

        running_cost = 0.0
        depth_counts: dict[str, int] = defaultdict(int)

        for rs in sorted_scores:
            ideal_depth = self.RISK_TO_DEPTH.get(rs.risk_bucket, VerificationDepth.STATIC)
            tier = self._tiers[ideal_depth]
            cost = tier.cost_cents_per_file

            if running_cost + cost <= budget_cents:
                rs.recommended_depth = ideal_depth
                rs.estimated_cost_cents = cost
                running_cost += cost
            else:
                # Downgrade to cheaper tier
                for fallback in [VerificationDepth.STATIC, VerificationDepth.PATTERN]:
                    fb_tier = self._tiers[fallback]
                    if running_cost + fb_tier.cost_cents_per_file <= budget_cents:
                        rs.recommended_depth = fallback
                        rs.estimated_cost_cents = fb_tier.cost_cents_per_file
                        running_cost += fb_tier.cost_cents_per_file
                        break
                else:
                    rs.recommended_depth = VerificationDepth.PATTERN
                    rs.estimated_cost_cents = 0.0

            depth_counts[rs.recommended_depth.value] += 1

        return RoutingDecision(
            file_routes=sorted_scores,
            total_estimated_cost_cents=round(running_cost, 2),
            budget_cents=budget_cents,
            files_at_depth=dict(depth_counts),
            budget_utilized=round(running_cost / budget_cents, 3) if budget_cents > 0 else 0.0,
        )


class CostOptimizerService:
    """Main service for verification cost optimization."""

    def __init__(self, default_budget_cents: float = 50.0) -> None:
        self._scorer = RiskScorer()
        self._router = DepthRouter()
        self._budget = default_budget_cents
        self._history: list[RoutingDecision] = []

    def optimize(
        self,
        files: list[dict[str, Any]],
        budget_cents: float | None = None,
    ) -> RoutingDecision:
        """Optimize verification routing for a set of files."""
        budget = budget_cents or self._budget

        risk_scores: list[FileRiskScore] = []
        for f in files:
            rs = self._scorer.score(
                file_path=f.get("path", ""),
                change_lines=f.get("change_lines", 10),
                is_new_file=f.get("is_new", False),
                author_commits=f.get("author_commits", 10),
            )
            risk_scores.append(rs)

        decision = self._router.route(risk_scores, budget)
        self._history.append(decision)
        return decision

    def estimate_savings(self, decision: RoutingDecision) -> dict[str, float]:
        """Estimate savings vs. running everything at full depth."""
        tiers = DepthConfig.all_tiers()
        full_cost = len(decision.file_routes) * tiers[VerificationDepth.FORMAL].cost_cents_per_file
        actual_cost = decision.total_estimated_cost_cents
        saved = full_cost - actual_cost

        return {
            "full_depth_cost_cents": full_cost,
            "optimized_cost_cents": actual_cost,
            "savings_cents": round(saved, 2),
            "savings_percent": round(saved / full_cost * 100, 1) if full_cost > 0 else 0.0,
        }

    def get_stats(self) -> RoutingStats:
        if not self._history:
            return RoutingStats()

        total_files = sum(len(d.file_routes) for d in self._history)
        depth_dist: dict[str, int] = defaultdict(int)
        all_risks: list[float] = []
        total_saved = 0.0

        for d in self._history:
            for k, v in d.files_at_depth.items():
                depth_dist[k] += v
            for fr in d.file_routes:
                all_risks.append(fr.risk_score)
            savings = self.estimate_savings(d)
            total_saved += savings["savings_cents"]

        return RoutingStats(
            total_files_routed=total_files,
            depth_distribution=dict(depth_dist),
            total_cost_saved_cents=round(total_saved, 2),
            avg_risk_score=round(sum(all_risks) / len(all_risks), 3) if all_risks else 0.0,
        )

    def set_budget(self, budget_cents: float) -> None:
        self._budget = budget_cents


# ─── Singleton Access ──────────────────────────────────────────────────


_cost_optimizer_instance: CostOptimizerService | None = None


def get_cost_optimizer_service() -> CostOptimizerService:
    global _cost_optimizer_instance
    if _cost_optimizer_instance is None:
        _cost_optimizer_instance = CostOptimizerService()
    return _cost_optimizer_instance


def reset_cost_optimizer_service() -> None:
    global _cost_optimizer_instance
    _cost_optimizer_instance = None
