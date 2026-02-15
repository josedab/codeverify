"""LLM Cost Optimizer with Local Model Support.

Provides intelligent routing between local models (Ollama, vLLM) and cloud
providers (OpenAI, Anthropic). Routes simple checks to local models and complex
analysis to cloud models, with cost tracking and budget management.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ModelProvider(str, Enum):
    """Available model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    VLLM = "vllm"
    AZURE_OPENAI = "azure_openai"
    LOCAL = "local"


class RoutingStrategy(str, Enum):
    """How to decide which model to use."""
    COST_OPTIMIZED = "cost_optimized"
    QUALITY_FIRST = "quality_first"
    LOCAL_FIRST = "local_first"
    CLOUD_ONLY = "cloud_only"


class CheckComplexity(str, Enum):
    """Complexity classification for a verification check."""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class ModelEndpoint:
    """Configuration for a model endpoint."""
    provider: ModelProvider
    model_name: str
    endpoint_url: str = ""
    api_key: str = ""
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    max_tokens: int = 4096
    avg_latency_ms: float = 1000.0
    quality_score: float = 0.8  # 0.0-1.0

    @property
    def is_local(self) -> bool:
        return self.provider in (ModelProvider.OLLAMA, ModelProvider.VLLM, ModelProvider.LOCAL)


@dataclass
class RoutingDecision:
    """Result of the routing decision."""
    endpoint: ModelEndpoint
    reason: str = ""
    estimated_cost: float = 0.0
    complexity: CheckComplexity = CheckComplexity.SIMPLE
    fallback_endpoint: ModelEndpoint | None = None


@dataclass
class CostRecord:
    """Records the cost of a single LLM call."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    provider: ModelProvider = ModelProvider.OPENAI
    model_name: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    check_type: str = ""
    complexity: CheckComplexity = CheckComplexity.SIMPLE
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class CostBudget:
    """Monthly cost budget configuration."""
    monthly_limit_usd: float = 100.0
    alert_threshold_pct: float = 80.0
    current_spend_usd: float = 0.0
    month: str = ""

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.monthly_limit_usd - self.current_spend_usd)

    @property
    def usage_pct(self) -> float:
        if self.monthly_limit_usd <= 0:
            return 100.0
        return (self.current_spend_usd / self.monthly_limit_usd) * 100

    @property
    def is_over_alert(self) -> bool:
        return self.usage_pct >= self.alert_threshold_pct

    @property
    def is_exhausted(self) -> bool:
        return self.current_spend_usd >= self.monthly_limit_usd


@dataclass
class CostReport:
    """Summary cost report."""
    total_cost_usd: float = 0.0
    total_calls: int = 0
    cost_by_provider: dict[str, float] = field(default_factory=dict)
    cost_by_complexity: dict[str, float] = field(default_factory=dict)
    local_calls: int = 0
    cloud_calls: int = 0
    estimated_savings_usd: float = 0.0
    avg_cost_per_call: float = 0.0
    period: str = ""


class ComplexityClassifier:
    """Classifies the complexity of a verification check."""

    def classify(self, code: str, check_type: str) -> CheckComplexity:
        """Classify check complexity based on code and check type."""
        lines = len(code.strip().split("\n"))
        has_loops = any(kw in code for kw in ["for ", "while ", "loop "])
        has_concurrency = any(kw in code for kw in ["async ", "await ", "go ", "spawn", "thread"])

        if lines < 10 and not has_loops:
            return CheckComplexity.TRIVIAL
        elif lines < 30 and not has_concurrency:
            return CheckComplexity.SIMPLE
        elif lines < 100 or has_concurrency:
            return CheckComplexity.MODERATE
        else:
            return CheckComplexity.COMPLEX


class CostOptimizer:
    """Intelligent routing and cost optimization for LLM calls."""

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.COST_OPTIMIZED,
        budget: CostBudget | None = None,
    ) -> None:
        self._strategy = strategy
        self._budget = budget or CostBudget(month=datetime.now(timezone.utc).strftime("%Y-%m"))
        self._endpoints: list[ModelEndpoint] = []
        self._records: list[CostRecord] = []
        self._classifier = ComplexityClassifier()

    def add_endpoint(self, endpoint: ModelEndpoint) -> None:
        self._endpoints.append(endpoint)

    def route(self, code: str, check_type: str) -> RoutingDecision:
        """Decide which model endpoint to use for this check."""
        if not self._endpoints:
            raise ValueError("No model endpoints configured")

        complexity = self._classifier.classify(code, check_type)
        local_endpoints = [e for e in self._endpoints if e.is_local]
        cloud_endpoints = [e for e in self._endpoints if not e.is_local]

        if self._strategy == RoutingStrategy.CLOUD_ONLY:
            endpoint = self._cheapest(cloud_endpoints or self._endpoints)
            return RoutingDecision(
                endpoint=endpoint,
                reason="Cloud-only strategy",
                complexity=complexity,
                estimated_cost=self._estimate_cost(endpoint, code),
            )

        if self._strategy == RoutingStrategy.LOCAL_FIRST and local_endpoints:
            endpoint = local_endpoints[0]
            fallback = self._cheapest(cloud_endpoints) if cloud_endpoints else None
            return RoutingDecision(
                endpoint=endpoint,
                reason="Local-first strategy",
                complexity=complexity,
                estimated_cost=0.0,
                fallback_endpoint=fallback,
            )

        # COST_OPTIMIZED or QUALITY_FIRST
        if complexity in (CheckComplexity.TRIVIAL, CheckComplexity.SIMPLE) and local_endpoints:
            endpoint = local_endpoints[0]
            fallback = self._cheapest(cloud_endpoints) if cloud_endpoints else None
            return RoutingDecision(
                endpoint=endpoint,
                reason=f"Simple check ({complexity.value}), using local model",
                complexity=complexity,
                estimated_cost=0.0,
                fallback_endpoint=fallback,
            )

        if self._budget.is_exhausted and local_endpoints:
            return RoutingDecision(
                endpoint=local_endpoints[0],
                reason="Budget exhausted, falling back to local",
                complexity=complexity,
                estimated_cost=0.0,
            )

        if self._strategy == RoutingStrategy.QUALITY_FIRST:
            endpoint = max(cloud_endpoints or self._endpoints, key=lambda e: e.quality_score)
        else:
            endpoint = self._cheapest(cloud_endpoints or self._endpoints)

        fallback = local_endpoints[0] if local_endpoints else None
        return RoutingDecision(
            endpoint=endpoint,
            reason=f"Complex check ({complexity.value}), using cloud model",
            complexity=complexity,
            estimated_cost=self._estimate_cost(endpoint, code),
            fallback_endpoint=fallback,
        )

    def record_call(
        self,
        provider: ModelProvider,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        check_type: str = "",
        complexity: CheckComplexity = CheckComplexity.SIMPLE,
    ) -> CostRecord:
        """Record the cost of an LLM call."""
        endpoint = next(
            (e for e in self._endpoints if e.provider == provider and e.model_name == model_name),
            None,
        )
        cost = 0.0
        if endpoint:
            cost = (
                (input_tokens / 1000) * endpoint.cost_per_1k_input_tokens
                + (output_tokens / 1000) * endpoint.cost_per_1k_output_tokens
            )

        record = CostRecord(
            provider=provider,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            check_type=check_type,
            complexity=complexity,
        )
        self._records.append(record)
        self._budget.current_spend_usd += cost
        return record

    def get_report(self, period: str = "") -> CostReport:
        """Generate a cost report."""
        if not period:
            period = datetime.now(timezone.utc).strftime("%Y-%m")

        records = [r for r in self._records if r.timestamp.strftime("%Y-%m") == period]
        total_cost = sum(r.cost_usd for r in records)
        local_calls = sum(1 for r in records if r.provider in (ModelProvider.OLLAMA, ModelProvider.VLLM, ModelProvider.LOCAL))
        cloud_calls = len(records) - local_calls

        cost_by_provider: dict[str, float] = defaultdict(float)
        cost_by_complexity: dict[str, float] = defaultdict(float)
        for r in records:
            cost_by_provider[r.provider.value] += r.cost_usd
            cost_by_complexity[r.complexity.value] += r.cost_usd

        # Estimate savings: what if all local calls had used cloud?
        cloud_default = next((e for e in self._endpoints if not e.is_local), None)
        savings = 0.0
        if cloud_default:
            for r in records:
                if r.provider in (ModelProvider.OLLAMA, ModelProvider.VLLM, ModelProvider.LOCAL):
                    savings += (
                        (r.input_tokens / 1000) * cloud_default.cost_per_1k_input_tokens
                        + (r.output_tokens / 1000) * cloud_default.cost_per_1k_output_tokens
                    )

        return CostReport(
            total_cost_usd=total_cost,
            total_calls=len(records),
            cost_by_provider=dict(cost_by_provider),
            cost_by_complexity=dict(cost_by_complexity),
            local_calls=local_calls,
            cloud_calls=cloud_calls,
            estimated_savings_usd=savings,
            avg_cost_per_call=total_cost / len(records) if records else 0.0,
            period=period,
        )

    def get_budget(self) -> CostBudget:
        return self._budget

    def _cheapest(self, endpoints: list[ModelEndpoint]) -> ModelEndpoint:
        return min(endpoints, key=lambda e: e.cost_per_1k_input_tokens + e.cost_per_1k_output_tokens)

    def _estimate_cost(self, endpoint: ModelEndpoint, code: str) -> float:
        # Rough estimate: ~4 chars per token
        est_input_tokens = len(code) / 4
        est_output_tokens = est_input_tokens * 0.3
        return (
            (est_input_tokens / 1000) * endpoint.cost_per_1k_input_tokens
            + (est_output_tokens / 1000) * endpoint.cost_per_1k_output_tokens
        )


# Singleton
_cost_optimizer_instance: CostOptimizer | None = None


def get_cost_optimizer() -> CostOptimizer:
    global _cost_optimizer_instance
    if _cost_optimizer_instance is None:
        _cost_optimizer_instance = CostOptimizer()
    return _cost_optimizer_instance


def reset_cost_optimizer() -> None:
    global _cost_optimizer_instance
    _cost_optimizer_instance = None
