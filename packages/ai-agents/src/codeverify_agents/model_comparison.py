"""Model Comparison Engine — A/B testing and cost optimization for LLM selection.

Benchmarks multiple LLM providers on the same code samples, tracks accuracy
and cost, and recommends optimal model routing per use case.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ModelProvider(str, Enum):
    """Supported LLM providers."""

    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo"
    GPT5 = "gpt-5"
    CLAUDE_SONNET = "claude-sonnet"
    CLAUDE_OPUS = "claude-opus"
    GEMINI_PRO = "gemini-pro"
    LOCAL_LLAMA = "local-llama"
    CUSTOM = "custom"


@dataclass
class ModelCostProfile:
    """Cost profile for an LLM model."""

    model: ModelProvider
    input_cost_per_1k: float  # $ per 1K input tokens
    output_cost_per_1k: float  # $ per 1K output tokens
    avg_latency_ms: float = 0.0
    max_context_tokens: int = 128000

    @property
    def avg_cost_per_call(self) -> float:
        """Estimate based on ~2K input + ~1K output per typical call."""
        return (self.input_cost_per_1k * 2) + (self.output_cost_per_1k * 1)


DEFAULT_COSTS: dict[ModelProvider, ModelCostProfile] = {
    ModelProvider.GPT4: ModelCostProfile(ModelProvider.GPT4, 0.03, 0.06, 2000),
    ModelProvider.GPT4_TURBO: ModelCostProfile(ModelProvider.GPT4_TURBO, 0.01, 0.03, 1500),
    ModelProvider.GPT5: ModelCostProfile(ModelProvider.GPT5, 0.05, 0.15, 3000),
    ModelProvider.CLAUDE_SONNET: ModelCostProfile(ModelProvider.CLAUDE_SONNET, 0.003, 0.015, 1200),
    ModelProvider.CLAUDE_OPUS: ModelCostProfile(ModelProvider.CLAUDE_OPUS, 0.015, 0.075, 2500),
    ModelProvider.GEMINI_PRO: ModelCostProfile(ModelProvider.GEMINI_PRO, 0.00125, 0.005, 1000),
    ModelProvider.LOCAL_LLAMA: ModelCostProfile(ModelProvider.LOCAL_LLAMA, 0.0, 0.0, 800),
}


@dataclass
class BenchmarkSample:
    """A single benchmark sample with known-good results."""

    id: str
    code: str
    language: str
    expected_findings: list[dict[str, Any]]
    category: str  # "security", "correctness", "style"
    difficulty: str = "medium"  # "easy", "medium", "hard"


@dataclass
class ModelResult:
    """Result of running a model on a benchmark sample."""

    model: ModelProvider
    sample_id: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0
    tokens_used: int = 0
    cost: float = 0.0
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for a model on benchmark samples."""

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def recall(self) -> float:
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1_score": round(self.f1_score, 3),
        }


@dataclass
class ModelBenchmark:
    """Complete benchmark results for a model."""

    model: ModelProvider
    accuracy: AccuracyMetrics = field(default_factory=AccuracyMetrics)
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    samples_tested: int = 0
    errors: int = 0

    @property
    def avg_latency(self) -> float:
        return self.total_latency_ms / max(1, self.samples_tested)

    @property
    def avg_cost_per_sample(self) -> float:
        return self.total_cost / max(1, self.samples_tested)

    @property
    def cost_effectiveness(self) -> float:
        """F1 score per dollar spent (higher is better)."""
        if self.total_cost == 0:
            return self.accuracy.f1_score * 1000  # Free models get bonus
        return self.accuracy.f1_score / self.total_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.value,
            "accuracy": self.accuracy.to_dict(),
            "total_cost": round(self.total_cost, 4),
            "avg_latency_ms": round(self.avg_latency, 1),
            "avg_cost_per_sample": round(self.avg_cost_per_sample, 6),
            "cost_effectiveness": round(self.cost_effectiveness, 2),
            "samples_tested": self.samples_tested,
            "errors": self.errors,
        }


@dataclass
class RoutingRecommendation:
    """Recommendation for which model to use in a given context."""

    context: str  # "security", "correctness", "quick_scan"
    recommended_model: ModelProvider
    fallback_model: ModelProvider
    reason: str
    estimated_cost: float
    estimated_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context,
            "recommended_model": self.recommended_model.value,
            "fallback_model": self.fallback_model.value,
            "reason": self.reason,
            "estimated_cost": round(self.estimated_cost, 4),
            "estimated_accuracy": round(self.estimated_accuracy, 3),
        }


class ModelComparisonEngine:
    """A/B tests LLMs and recommends optimal model routing.

    Example:
        >>> engine = ModelComparisonEngine()
        >>> engine.add_sample(BenchmarkSample(...))
        >>> engine.record_result(ModelResult(model=ModelProvider.GPT4, ...))
        >>> report = engine.get_comparison_report()
        >>> recommendations = engine.get_routing_recommendations()
    """

    def __init__(self) -> None:
        self._samples: dict[str, BenchmarkSample] = {}
        self._results: dict[str, list[ModelResult]] = {}  # model -> results
        self._benchmarks: dict[str, ModelBenchmark] = {}  # model -> benchmark
        self._cost_profiles: dict[ModelProvider, ModelCostProfile] = dict(DEFAULT_COSTS)

    def add_sample(self, sample: BenchmarkSample) -> None:
        """Add a benchmark sample with known expected findings."""
        self._samples[sample.id] = sample

    def set_cost_profile(self, model: ModelProvider, profile: ModelCostProfile) -> None:
        """Override cost profile for a model."""
        self._cost_profiles[model] = profile

    def record_result(self, result: ModelResult) -> None:
        """Record a model's result on a benchmark sample."""
        key = result.model.value
        if key not in self._results:
            self._results[key] = []
        self._results[key].append(result)

        # Update benchmark
        benchmark = self._benchmarks.setdefault(key, ModelBenchmark(model=result.model))
        benchmark.samples_tested += 1
        benchmark.total_latency_ms += result.latency_ms
        benchmark.total_cost += result.cost

        if result.error:
            benchmark.errors += 1
            return

        # Compute accuracy against expected findings
        sample = self._samples.get(result.sample_id)
        if sample:
            self._update_accuracy(benchmark, result, sample)

    def get_comparison_report(self) -> dict[str, Any]:
        """Generate comparison report across all tested models."""
        benchmarks = sorted(
            self._benchmarks.values(),
            key=lambda b: b.accuracy.f1_score,
            reverse=True,
        )

        return {
            "total_samples": len(self._samples),
            "models_tested": len(benchmarks),
            "rankings": {
                "by_accuracy": [
                    {"model": b.model.value, "f1": round(b.accuracy.f1_score, 3)}
                    for b in sorted(benchmarks, key=lambda b: b.accuracy.f1_score, reverse=True)
                ],
                "by_cost": [
                    {"model": b.model.value, "cost": round(b.total_cost, 4)}
                    for b in sorted(benchmarks, key=lambda b: b.total_cost)
                ],
                "by_speed": [
                    {"model": b.model.value, "avg_latency_ms": round(b.avg_latency, 1)}
                    for b in sorted(benchmarks, key=lambda b: b.avg_latency)
                ],
                "by_cost_effectiveness": [
                    {"model": b.model.value, "score": round(b.cost_effectiveness, 2)}
                    for b in sorted(benchmarks, key=lambda b: b.cost_effectiveness, reverse=True)
                ],
            },
            "detailed": [b.to_dict() for b in benchmarks],
        }

    def get_routing_recommendations(self) -> list[RoutingRecommendation]:
        """Generate model routing recommendations by use case."""
        if not self._benchmarks:
            return []

        benchmarks = list(self._benchmarks.values())
        recommendations = []

        # Best accuracy (for security-critical)
        best_accuracy = max(benchmarks, key=lambda b: b.accuracy.f1_score)
        second_best = sorted(benchmarks, key=lambda b: b.accuracy.f1_score, reverse=True)
        fallback_accuracy = second_best[1] if len(second_best) > 1 else best_accuracy

        recommendations.append(
            RoutingRecommendation(
                context="security_critical",
                recommended_model=best_accuracy.model,
                fallback_model=fallback_accuracy.model,
                reason=f"Highest accuracy (F1={best_accuracy.accuracy.f1_score:.3f})",
                estimated_cost=best_accuracy.avg_cost_per_sample,
                estimated_accuracy=best_accuracy.accuracy.f1_score,
            )
        )

        # Best cost-effectiveness (for routine checks)
        best_value = max(benchmarks, key=lambda b: b.cost_effectiveness)
        recommendations.append(
            RoutingRecommendation(
                context="routine_check",
                recommended_model=best_value.model,
                fallback_model=best_accuracy.model,
                reason=f"Best cost-effectiveness ({best_value.cost_effectiveness:.1f})",
                estimated_cost=best_value.avg_cost_per_sample,
                estimated_accuracy=best_value.accuracy.f1_score,
            )
        )

        # Fastest (for real-time IDE checks)
        fastest = min(benchmarks, key=lambda b: b.avg_latency)
        recommendations.append(
            RoutingRecommendation(
                context="real_time_ide",
                recommended_model=fastest.model,
                fallback_model=best_value.model,
                reason=f"Lowest latency ({fastest.avg_latency:.0f}ms avg)",
                estimated_cost=fastest.avg_cost_per_sample,
                estimated_accuracy=fastest.accuracy.f1_score,
            )
        )

        # Cheapest (for budget-constrained)
        cheapest = min(benchmarks, key=lambda b: b.total_cost)
        recommendations.append(
            RoutingRecommendation(
                context="budget_constrained",
                recommended_model=cheapest.model,
                fallback_model=best_value.model,
                reason=f"Lowest cost (${cheapest.avg_cost_per_sample:.4f}/sample)",
                estimated_cost=cheapest.avg_cost_per_sample,
                estimated_accuracy=cheapest.accuracy.f1_score,
            )
        )

        return recommendations

    def estimate_monthly_cost(
        self,
        model: ModelProvider,
        daily_analyses: int = 100,
    ) -> dict[str, float]:
        """Estimate monthly cost for a model at given usage."""
        profile = self._cost_profiles.get(model)
        if not profile:
            return {"error": "Unknown model"}

        daily_cost = profile.avg_cost_per_call * daily_analyses
        return {
            "model": model.value,
            "daily_analyses": daily_analyses,
            "daily_cost": round(daily_cost, 2),
            "monthly_cost": round(daily_cost * 30, 2),
            "yearly_cost": round(daily_cost * 365, 2),
        }

    def _update_accuracy(
        self,
        benchmark: ModelBenchmark,
        result: ModelResult,
        sample: BenchmarkSample,
    ) -> None:
        """Update accuracy metrics by comparing result to expected findings."""
        expected_ids = {
            f.get("category", "") + ":" + f.get("title", "") for f in sample.expected_findings
        }
        actual_ids = {f.get("category", "") + ":" + f.get("title", "") for f in result.findings}

        benchmark.accuracy.true_positives += len(expected_ids & actual_ids)
        benchmark.accuracy.false_positives += len(actual_ids - expected_ids)
        benchmark.accuracy.false_negatives += len(expected_ids - actual_ids)
