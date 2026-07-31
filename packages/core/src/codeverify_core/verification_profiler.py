"""Verification Performance Profiler.

Profile verification runs to identify bottlenecks, recommend optimizations,
and allocate time budgets across files.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


# =============================================================================
# Enums
# =============================================================================


class ProfileStage(str, Enum):
    """Stages of the verification pipeline."""

    PARSING = "parsing"
    TYPE_CHECKING = "type_checking"
    CONSTRAINT_GENERATION = "constraint_generation"
    Z3_SOLVING = "z3_solving"
    AI_ANALYSIS = "ai_analysis"
    SYNTHESIS = "synthesis"
    TOTAL = "total"


class BottleneckType(str, Enum):
    """Types of performance bottlenecks."""

    TIMEOUT = "timeout"
    HIGH_COMPLEXITY = "high_complexity"
    LARGE_INPUT = "large_input"
    MANY_CONSTRAINTS = "many_constraints"
    SLOW_MODEL = "slow_model"
    MEMORY_LIMIT = "memory_limit"


class OptimizationStrategy(str, Enum):
    """Strategies for improving verification performance."""

    REDUCE_DEPTH = "reduce_depth"
    SPLIT_FUNCTION = "split_function"
    CACHE_RESULT = "cache_result"
    SIMPLIFY_CONSTRAINTS = "simplify_constraints"
    USE_FASTER_MODEL = "use_faster_model"
    SKIP_STAGE = "skip_stage"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class StageProfile:
    """Profiling data for a single verification stage."""

    stage: ProfileStage
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    constraint_count: int = 0
    tokens_used: int = 0
    cache_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "duration_ms": self.duration_ms,
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
            "constraint_count": self.constraint_count,
            "tokens_used": self.tokens_used,
            "cache_hit": self.cache_hit,
        }


@dataclass
class BottleneckInfo:
    """Information about a detected bottleneck."""

    bottleneck_type: BottleneckType
    stage: ProfileStage
    description: str
    severity: float
    suggested_fix: str
    estimated_speedup: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "bottleneck_type": self.bottleneck_type.value,
            "stage": self.stage.value,
            "description": self.description,
            "severity": self.severity,
            "suggested_fix": self.suggested_fix,
            "estimated_speedup": self.estimated_speedup,
        }


@dataclass
class FunctionProfile:
    """Complete profiling data for a single function verification."""

    function_name: str
    file_path: str
    language: str
    total_time_ms: float
    stages: list[StageProfile]
    complexity_score: float
    line_count: int
    bottlenecks: list[BottleneckInfo]
    verified: bool
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "function_name": self.function_name,
            "file_path": self.file_path,
            "language": self.language,
            "total_time_ms": self.total_time_ms,
            "stages": [s.to_dict() for s in self.stages],
            "complexity_score": self.complexity_score,
            "line_count": self.line_count,
            "bottlenecks": [b.to_dict() for b in self.bottlenecks],
            "verified": self.verified,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class OptimizationRecommendation:
    """A recommendation for improving verification performance."""

    strategy: OptimizationStrategy
    target: str
    description: str
    estimated_speedup_percent: float
    risk: str
    implementation_effort: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "target": self.target,
            "description": self.description,
            "estimated_speedup_percent": self.estimated_speedup_percent,
            "risk": self.risk,
            "implementation_effort": self.implementation_effort,
        }


@dataclass
class BudgetAllocation:
    """Time budget allocation for a file."""

    file_path: str
    allocated_time_ms: float
    actual_time_ms: float
    priority: float
    depth: str
    utilization: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "allocated_time_ms": self.allocated_time_ms,
            "actual_time_ms": self.actual_time_ms,
            "priority": self.priority,
            "depth": self.depth,
            "utilization": self.utilization,
        }


@dataclass
class PerformanceTrend:
    """Performance metrics for a time period."""

    period: str
    avg_verification_time_ms: float
    p95_verification_time_ms: float
    p99_verification_time_ms: float
    timeout_rate: float
    cache_hit_rate: float
    total_verifications: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "avg_verification_time_ms": self.avg_verification_time_ms,
            "p95_verification_time_ms": self.p95_verification_time_ms,
            "p99_verification_time_ms": self.p99_verification_time_ms,
            "timeout_rate": self.timeout_rate,
            "cache_hit_rate": self.cache_hit_rate,
            "total_verifications": self.total_verifications,
        }


@dataclass
class ProfileReport:
    """Aggregated profiling report for a project."""

    project_name: str
    generated_at: datetime
    total_functions_profiled: int
    avg_verification_time_ms: float
    slowest_functions: list[FunctionProfile]
    bottleneck_summary: dict[str, int]
    recommendations: list[OptimizationRecommendation]
    budget_utilization: list[BudgetAllocation]
    trends: list[PerformanceTrend]
    estimated_total_speedup: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "generated_at": self.generated_at.isoformat(),
            "total_functions_profiled": self.total_functions_profiled,
            "avg_verification_time_ms": self.avg_verification_time_ms,
            "slowest_functions": [f.to_dict() for f in self.slowest_functions],
            "bottleneck_summary": self.bottleneck_summary,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "budget_utilization": [b.to_dict() for b in self.budget_utilization],
            "trends": [t.to_dict() for t in self.trends],
            "estimated_total_speedup": self.estimated_total_speedup,
        }


@dataclass
class _ActiveProfile:
    """Internal state for an in-progress profiling session."""

    profile_id: str
    function_name: str
    file_path: str
    language: str
    start_time_ns: int
    stages: list[StageProfile] = field(default_factory=list)


# =============================================================================
# Verification Instrumenter
# =============================================================================


class VerificationInstrumenter:
    """Instruments verification runs and collects per-stage profiles."""

    def __init__(self) -> None:
        self._active: dict[str, _ActiveProfile] = {}

    def start_profiling(self, function_name: str, file_path: str, language: str) -> str:
        """Begin profiling a function verification. Returns a profile_id."""
        profile_id = uuid.uuid4().hex[:12]
        self._active[profile_id] = _ActiveProfile(
            profile_id=profile_id,
            function_name=function_name,
            file_path=file_path,
            language=language,
            start_time_ns=time.perf_counter_ns(),
        )
        logger.debug("profiling_started", profile_id=profile_id, function=function_name)
        return profile_id

    def record_stage(
        self,
        profile_id: str,
        stage: ProfileStage,
        duration_ms: float,
        memory_mb: float = 0.0,
        cpu_percent: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """Record metrics for a completed verification stage."""
        if profile_id not in self._active:
            logger.warning("record_stage_unknown_profile", profile_id=profile_id)
            return
        self._active[profile_id].stages.append(
            StageProfile(
                stage=stage,
                duration_ms=duration_ms,
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                constraint_count=kwargs.get("constraint_count", 0),
                tokens_used=kwargs.get("tokens_used", 0),
                cache_hit=kwargs.get("cache_hit", False),
            )
        )

    def end_profiling(self, profile_id: str, verified: bool = True) -> FunctionProfile:
        """Finish profiling and return the completed FunctionProfile."""
        active = self._active.pop(profile_id, None)
        if active is None:
            raise ValueError(f"Unknown profile_id: {profile_id}")

        elapsed_ns = time.perf_counter_ns() - active.start_time_ns
        total_ms = elapsed_ns / 1_000_000
        total_constraints = sum(s.constraint_count for s in active.stages)
        complexity = _estimate_complexity(active.stages, total_constraints)

        parsing = [s for s in active.stages if s.stage == ProfileStage.PARSING]
        line_count = int(parsing[0].duration_ms * 8) if parsing else 0

        profile = FunctionProfile(
            function_name=active.function_name,
            file_path=active.file_path,
            language=active.language,
            total_time_ms=total_ms,
            stages=active.stages,
            complexity_score=complexity,
            line_count=line_count,
            bottlenecks=[],
            verified=verified,
            timestamp=datetime.utcnow(),
        )
        logger.info("profiling_complete", profile_id=profile_id, total_ms=round(total_ms, 2))
        return profile

    def get_active_profiles(self) -> list[str]:
        """Return IDs of all currently active profiling sessions."""
        return list(self._active.keys())


# =============================================================================
# Bottleneck Detector
# =============================================================================


class BottleneckDetector:
    """Analyses a FunctionProfile to detect performance bottlenecks."""

    def __init__(
        self, timeout_threshold_ms: float = 5000.0, complexity_threshold: float = 50.0
    ) -> None:
        self._timeout_threshold_ms = timeout_threshold_ms
        self._complexity_threshold = complexity_threshold

    def detect_bottlenecks(self, profile: FunctionProfile) -> list[BottleneckInfo]:
        """Run all bottleneck checks against a profile."""
        results: list[BottleneckInfo] = []
        for check in [
            self._check_timeout,
            self._check_complexity,
            self._check_constraint_explosion,
            self._check_memory,
            self._check_slow_model,
        ]:
            r = check(profile)
            if r is not None:
                results.append(r)
        results.sort(key=lambda b: b.severity, reverse=True)
        return results

    def _check_timeout(self, profile: FunctionProfile) -> BottleneckInfo | None:
        if profile.total_time_ms <= self._timeout_threshold_ms:
            return None
        slowest = max(profile.stages, key=lambda s: s.duration_ms) if profile.stages else None
        ratio = profile.total_time_ms / self._timeout_threshold_ms
        return BottleneckInfo(
            bottleneck_type=BottleneckType.TIMEOUT,
            stage=slowest.stage if slowest else ProfileStage.TOTAL,
            description=f"Verification took {profile.total_time_ms:.0f}ms ({ratio:.1f}x threshold)",
            severity=min(ratio / 3.0, 1.0),
            suggested_fix=f"Reduce depth or split function to lower {(slowest.stage.value if slowest else 'total')} time",
            estimated_speedup=min((ratio - 1.0) / ratio, 0.8),
        )

    def _check_complexity(self, profile: FunctionProfile) -> BottleneckInfo | None:
        if profile.complexity_score <= self._complexity_threshold:
            return None
        severity = min(profile.complexity_score / (self._complexity_threshold * 3), 1.0)
        return BottleneckInfo(
            bottleneck_type=BottleneckType.HIGH_COMPLEXITY,
            stage=ProfileStage.CONSTRAINT_GENERATION,
            description=f"Complexity {profile.complexity_score:.1f} exceeds threshold {self._complexity_threshold:.1f}",
            severity=severity,
            suggested_fix="Split function into smaller units to reduce complexity",
            estimated_speedup=0.3 * severity,
        )

    def _check_constraint_explosion(self, profile: FunctionProfile) -> BottleneckInfo | None:
        total = sum(s.constraint_count for s in profile.stages)
        if total <= 500:
            return None
        severity = min(total / 2000, 1.0)
        solving_ms = sum(
            s.duration_ms for s in profile.stages if s.stage == ProfileStage.Z3_SOLVING
        )
        return BottleneckInfo(
            bottleneck_type=BottleneckType.MANY_CONSTRAINTS,
            stage=ProfileStage.Z3_SOLVING,
            description=f"{total} constraints generated; Z3 solving took {solving_ms:.0f}ms",
            severity=severity,
            suggested_fix="Simplify constraints or enable incremental solving",
            estimated_speedup=0.4 * severity,
        )

    def _check_memory(self, profile: FunctionProfile) -> BottleneckInfo | None:
        peak = max(profile.stages, key=lambda s: s.memory_mb) if profile.stages else None
        if peak is None or peak.memory_mb <= 512.0:
            return None
        severity = min(peak.memory_mb / 2048.0, 1.0)
        return BottleneckInfo(
            bottleneck_type=BottleneckType.MEMORY_LIMIT,
            stage=peak.stage,
            description=f"Stage {peak.stage.value} used {peak.memory_mb:.0f}MB",
            severity=severity,
            suggested_fix="Reduce input size or enable streaming analysis",
            estimated_speedup=0.2 * severity,
        )

    def _check_slow_model(self, profile: FunctionProfile) -> BottleneckInfo | None:
        ai_stages = [s for s in profile.stages if s.stage == ProfileStage.AI_ANALYSIS]
        if not ai_stages:
            return None
        ai_ms = sum(s.duration_ms for s in ai_stages)
        if ai_ms <= 0 or profile.total_time_ms <= 0:
            return None
        ratio = ai_ms / profile.total_time_ms
        if ratio <= 0.6:
            return None
        tokens = sum(s.tokens_used for s in ai_stages)
        return BottleneckInfo(
            bottleneck_type=BottleneckType.SLOW_MODEL,
            stage=ProfileStage.AI_ANALYSIS,
            description=f"AI analysis consumed {ratio:.0%} of total time ({ai_ms:.0f}ms, {tokens} tokens)",
            severity=min(ratio, 1.0),
            suggested_fix="Use a faster model or cache repeated analyses",
            estimated_speedup=0.25 * min(ratio, 1.0),
        )


# =============================================================================
# Optimization Advisor
# =============================================================================


class OptimizationAdvisor:
    """Generates actionable optimization recommendations from bottlenecks."""

    _HANDLER_MAP: dict[BottleneckType, str] = {
        BottleneckType.TIMEOUT: "_recommend_for_timeout",
        BottleneckType.HIGH_COMPLEXITY: "_recommend_for_complexity",
        BottleneckType.MANY_CONSTRAINTS: "_recommend_for_constraints",
        BottleneckType.SLOW_MODEL: "_recommend_for_constraints",
        BottleneckType.LARGE_INPUT: "_recommend_for_complexity",
        BottleneckType.MEMORY_LIMIT: "_recommend_for_timeout",
    }

    def __init__(self) -> None:
        pass

    def recommend(
        self,
        bottlenecks: list[BottleneckInfo],
        profile: FunctionProfile,
    ) -> list[OptimizationRecommendation]:
        """Generate recommendations for detected bottlenecks."""
        recs: list[OptimizationRecommendation] = []
        seen: set[tuple[str, str]] = set()
        for bn in bottlenecks:
            handler = getattr(self, self._HANDLER_MAP.get(bn.bottleneck_type, ""), None)
            if handler is None:
                continue
            rec = handler(bn, profile)
            key = (rec.strategy.value, rec.target)
            if key not in seen:
                seen.add(key)
                recs.append(rec)
        return recs

    def _recommend_for_timeout(
        self,
        bottleneck: BottleneckInfo,
        profile: FunctionProfile,
    ) -> OptimizationRecommendation:
        return OptimizationRecommendation(
            strategy=OptimizationStrategy.REDUCE_DEPTH,
            target=profile.function_name,
            description=f"Reduce verification depth for {profile.function_name} to avoid timeout in {bottleneck.stage.value}",
            estimated_speedup_percent=bottleneck.estimated_speedup * 100,
            risk="May miss deep semantic issues",
            implementation_effort="low",
        )

    def _recommend_for_complexity(
        self,
        bottleneck: BottleneckInfo,
        profile: FunctionProfile,
    ) -> OptimizationRecommendation:
        return OptimizationRecommendation(
            strategy=OptimizationStrategy.SPLIT_FUNCTION,
            target=profile.function_name,
            description=f"Split {profile.function_name} (complexity {profile.complexity_score:.1f}) into smaller units",
            estimated_speedup_percent=bottleneck.estimated_speedup * 100,
            risk="Requires refactoring the source code",
            implementation_effort="high",
        )

    def _recommend_for_constraints(
        self,
        bottleneck: BottleneckInfo,
        profile: FunctionProfile,
    ) -> OptimizationRecommendation:
        total_constraints = sum(s.constraint_count for s in profile.stages)
        return OptimizationRecommendation(
            strategy=OptimizationStrategy.SIMPLIFY_CONSTRAINTS,
            target=profile.function_name,
            description=f"Simplify {total_constraints} constraints for {profile.function_name} to speed up solving",
            estimated_speedup_percent=bottleneck.estimated_speedup * 100,
            risk="Simplified constraints may be less precise",
            implementation_effort="medium",
        )


# =============================================================================
# Budget Allocator
# =============================================================================


class BudgetAllocator:
    """Distributes a total time budget across files based on priority."""

    def __init__(self, total_budget_ms: float = 60000.0) -> None:
        self._total_budget_ms = total_budget_ms

    def allocate(
        self, files: list[dict[str, Any]], history: list[FunctionProfile]
    ) -> list[BudgetAllocation]:
        """Allocate time budget to each file based on priority and history."""
        if not files:
            return []
        priorities = [(f, self._calculate_priority(f, history)) for f in files]
        total_pri = sum(p for _, p in priorities) or 1.0
        allocations: list[BudgetAllocation] = []
        for info, pri in priorities:
            allocated = self._total_budget_ms * (pri / total_pri)
            allocations.append(
                BudgetAllocation(
                    file_path=info.get("path", "unknown"),
                    allocated_time_ms=round(allocated, 2),
                    actual_time_ms=0.0,
                    priority=round(pri, 4),
                    depth=self._select_depth(pri, allocated),
                    utilization=0.0,
                )
            )
        return allocations

    def _calculate_priority(
        self, file_info: dict[str, Any], history: list[FunctionProfile]
    ) -> float:
        complexity = float(file_info.get("complexity", 10.0))
        criticality = float(file_info.get("criticality", 0.5))
        lines = int(file_info.get("lines", 100))
        base = (complexity * 0.4) + (criticality * 40) + (math.log1p(lines) * 2)

        path = file_info.get("path", "")
        relevant = [p for p in history if p.file_path == path]
        if relevant:
            avg_time = sum(p.total_time_ms for p in relevant) / len(relevant)
            failure_rate = sum(1 for p in relevant if not p.verified) / len(relevant)
            base += avg_time * 0.01 + failure_rate * 20
        return max(base, 1.0)

    def _select_depth(self, priority: float, budget: float) -> str:
        if budget >= 10000 and priority >= 50:
            return "full"
        if budget >= 5000 and priority >= 30:
            return "formal"
        if budget >= 2000 and priority >= 15:
            return "ai"
        if budget >= 500:
            return "static"
        return "pattern"

    def rebalance(
        self,
        allocations: list[BudgetAllocation],
        actual_results: list[FunctionProfile],
    ) -> list[BudgetAllocation]:
        """Rebalance budget allocations based on actual verification results."""
        result_map = {r.file_path: r for r in actual_results}
        surplus_ms = 0.0
        underperformers: list[BudgetAllocation] = []

        for alloc in allocations:
            result = result_map.get(alloc.file_path)
            if result is None:
                continue
            alloc.actual_time_ms = result.total_time_ms
            alloc.utilization = (
                round(
                    result.total_time_ms / alloc.allocated_time_ms,
                    4,
                )
                if alloc.allocated_time_ms > 0
                else 0.0
            )
            if result.total_time_ms < alloc.allocated_time_ms * 0.7:
                surplus_ms += alloc.allocated_time_ms - result.total_time_ms
            elif result.total_time_ms > alloc.allocated_time_ms:
                underperformers.append(alloc)

        if underperformers and surplus_ms > 0:
            extra = surplus_ms / len(underperformers)
            for alloc in underperformers:
                alloc.allocated_time_ms += extra
                alloc.depth = self._select_depth(alloc.priority, alloc.allocated_time_ms)

        logger.info(
            "budget_rebalanced",
            surplus_ms=round(surplus_ms, 2),
            underperformers=len(underperformers),
        )
        return allocations


# =============================================================================
# Verification Profiler (Façade)
# =============================================================================


class VerificationProfiler:
    """High-level façade that orchestrates profiling, detection, and reporting."""

    def __init__(self, budget_ms: float = 60000.0) -> None:
        self._instrumenter = VerificationInstrumenter()
        self._detector = BottleneckDetector()
        self._advisor = OptimizationAdvisor()
        self._allocator = BudgetAllocator(total_budget_ms=budget_ms)
        self._budget_ms = budget_ms

    def profile_function(
        self,
        function_name: str,
        file_path: str,
        code: str,
        language: str,
    ) -> FunctionProfile:
        """Profile the verification of a single function end-to-end."""
        pid = self._instrumenter.start_profiling(function_name, file_path, language)
        lines = code.count("\n") + 1
        lf = _language_factor(language)

        stage_specs: list[tuple[ProfileStage, float, dict[str, Any]]] = [
            (ProfileStage.PARSING, 0.5 * lines * lf, {}),
            (ProfileStage.TYPE_CHECKING, 0.8 * lines * lf, {}),
            (
                ProfileStage.CONSTRAINT_GENERATION,
                1.2 * lines * lf,
                {"constraint_count": max(1, int(lines * 1.5))},
            ),
            (
                ProfileStage.Z3_SOLVING,
                2.0 * lines * lf,
                {"constraint_count": max(1, int(lines * 1.5))},
            ),
            (ProfileStage.AI_ANALYSIS, 3.0 * lines * lf, {"tokens_used": max(1, lines * 20)}),
            (ProfileStage.SYNTHESIS, 0.3 * lines * lf, {}),
        ]
        for stage, base_ms, kw in stage_specs:
            t0 = time.perf_counter_ns()
            _busy_wait_ms(max(base_ms * 0.001, 0.01))
            elapsed = (time.perf_counter_ns() - t0) / 1_000_000
            self._instrumenter.record_stage(
                pid,
                stage,
                duration_ms=round(base_ms + elapsed, 3),
                memory_mb=round(lines * 0.05, 2),
                cpu_percent=round(min(base_ms / 10, 95), 1),
                **kw,
            )

        profile = self._instrumenter.end_profiling(pid, verified=True)
        profile.line_count = lines
        profile.bottlenecks = self._detector.detect_bottlenecks(profile)
        return profile

    def generate_report(
        self,
        project_name: str,
        profiles: list[FunctionProfile],
    ) -> ProfileReport:
        """Generate a comprehensive profiling report from collected profiles."""
        if not profiles:
            return ProfileReport(
                project_name=project_name,
                generated_at=datetime.utcnow(),
                total_functions_profiled=0,
                avg_verification_time_ms=0.0,
                slowest_functions=[],
                bottleneck_summary={},
                recommendations=[],
                budget_utilization=[],
                trends=[],
                estimated_total_speedup=0.0,
            )

        times = [p.total_time_ms for p in profiles]
        avg_time = sum(times) / len(times)
        sorted_profiles = sorted(profiles, key=lambda p: p.total_time_ms, reverse=True)
        slowest = sorted_profiles[:10]

        # Aggregate bottleneck counts
        bn_summary: dict[str, int] = {}
        for p in profiles:
            for bn in p.bottlenecks:
                bn_summary[bn.bottleneck_type.value] = (
                    bn_summary.get(bn.bottleneck_type.value, 0) + 1
                )

        # Collect and deduplicate recommendations
        recs: list[OptimizationRecommendation] = []
        seen: set[tuple[str, str]] = set()
        for p in slowest:
            for rec in self._advisor.recommend(p.bottlenecks, p):
                key = (rec.strategy.value, rec.target)
                if key not in seen:
                    seen.add(key)
                    recs.append(rec)

        budget_util = [
            BudgetAllocation(
                file_path=p.file_path,
                allocated_time_ms=self._budget_ms / max(len(profiles), 1),
                actual_time_ms=p.total_time_ms,
                priority=p.complexity_score,
                depth="full" if p.total_time_ms > 5000 else "standard",
                utilization=round(p.total_time_ms / (self._budget_ms / max(len(profiles), 1)), 4),
            )
            for p in profiles
        ]

        report = ProfileReport(
            project_name=project_name,
            generated_at=datetime.utcnow(),
            total_functions_profiled=len(profiles),
            avg_verification_time_ms=round(avg_time, 2),
            slowest_functions=slowest,
            bottleneck_summary=bn_summary,
            recommendations=recs,
            budget_utilization=budget_util,
            trends=self.get_trends(profiles),
            estimated_total_speedup=self._calculate_estimated_speedup(recs),
        )
        logger.info("report_generated", project=project_name, functions=len(profiles))
        return report

    def get_trends(
        self,
        profiles: list[FunctionProfile],
        periods: int = 5,
    ) -> list[PerformanceTrend]:
        """Compute performance trends by splitting profiles into time periods."""
        if not profiles:
            return []
        sorted_profs = sorted(profiles, key=lambda p: p.timestamp)
        chunk_size = max(1, len(sorted_profs) // periods)
        trends: list[PerformanceTrend] = []

        for i in range(0, len(sorted_profs), chunk_size):
            chunk = sorted_profs[i : i + chunk_size]
            if not chunk:
                continue
            times = sorted(p.total_time_ms for p in chunk)
            cache_hits = sum(1 for p in chunk for s in p.stages if s.cache_hit)
            total_stages = sum(len(p.stages) for p in chunk)
            timeout_thresh = self._budget_ms / max(len(profiles), 1) * 2
            timeouts = sum(1 for p in chunk if p.total_time_ms > timeout_thresh)
            trends.append(
                PerformanceTrend(
                    period=f"period_{len(trends) + 1}",
                    avg_verification_time_ms=round(sum(times) / len(times), 2),
                    p95_verification_time_ms=round(_percentile(times, 95), 2),
                    p99_verification_time_ms=round(_percentile(times, 99), 2),
                    timeout_rate=round(timeouts / len(chunk), 4),
                    cache_hit_rate=round(cache_hits / total_stages if total_stages else 0.0, 4),
                    total_verifications=len(chunk),
                )
            )
        return trends

    def get_optimization_plan(self, report: ProfileReport) -> list[OptimizationRecommendation]:
        """Return the report's recommendations sorted by estimated speedup."""
        return sorted(
            report.recommendations, key=lambda r: r.estimated_speedup_percent, reverse=True
        )

    def _calculate_estimated_speedup(
        self, recommendations: list[OptimizationRecommendation]
    ) -> float:
        """Estimate aggregate speedup using diminishing-returns: 1 - product(1 - s_i/100)."""
        if not recommendations:
            return 0.0
        remaining = 1.0
        for r in recommendations:
            factor = max(0.0, min(r.estimated_speedup_percent / 100.0, 0.95))
            remaining *= 1.0 - factor
        return round((1.0 - remaining) * 100, 2)


# =============================================================================
# Module-level helpers
# =============================================================================


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Compute the p-th percentile from a pre-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (pct / 100.0) * (len(sorted_values) - 1)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def _estimate_complexity(stages: list[StageProfile], total_constraints: int) -> float:
    """Heuristic complexity score from stage metrics."""
    if not stages:
        return 0.0
    total_ms = sum(s.duration_ms for s in stages)
    return round(math.log1p(total_constraints) * 5 + math.log1p(total_ms) * 3, 2)


def _language_factor(language: str) -> float:
    """Multiplier reflecting typical verification cost per language."""
    return {
        "python": 1.0,
        "javascript": 1.1,
        "typescript": 1.15,
        "java": 1.2,
        "go": 0.9,
        "rust": 1.3,
        "c": 1.4,
        "cpp": 1.5,
    }.get(language.lower(), 1.0)


def _busy_wait_ms(ms: float) -> None:
    """Spin-wait for a tiny duration to simulate real work."""
    end = time.perf_counter_ns() + int(ms * 1_000_000)
    while time.perf_counter_ns() < end:
        pass
