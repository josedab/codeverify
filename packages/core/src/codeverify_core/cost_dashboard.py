"""CI/CD Cost & Time Dashboard.

Instruments LLM token usage, Z3 solve time, and cache hits per analysis run.
Provides Prometheus metrics export and budget tracking for engineering leaders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# =============================================================================
# Cost Models
# =============================================================================


class CostCategory(str, Enum):
    LLM_TOKENS = "llm_tokens"
    Z3_COMPUTE = "z3_compute"
    CACHE_LOOKUP = "cache_lookup"
    NETWORK = "network"
    STORAGE = "storage"


# Default cost rates (USD)
DEFAULT_COST_RATES: dict[str, float] = {
    "gpt-4-input": 0.03 / 1000,  # $0.03 per 1K input tokens
    "gpt-4-output": 0.06 / 1000,  # $0.06 per 1K output tokens
    "claude-input": 0.015 / 1000,  # $0.015 per 1K input tokens
    "claude-output": 0.075 / 1000,  # $0.075 per 1K output tokens
    "gpt-4o-mini-input": 0.00015 / 1000,
    "gpt-4o-mini-output": 0.0006 / 1000,
    "z3-cpu-second": 0.001,  # $0.001 per CPU-second
    "cache-hit": 0.0,  # Free
}


@dataclass
class CostRecord:
    """A single cost event."""

    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    category: str = ""
    description: str = ""
    amount_usd: float = 0.0
    quantity: float = 0.0
    unit: str = ""
    model: str = ""
    analysis_id: str = ""
    repository: str = ""
    pr_number: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "category": self.category,
            "description": self.description,
            "amount_usd": round(self.amount_usd, 6),
            "quantity": self.quantity,
            "unit": self.unit,
            "model": self.model,
            "analysis_id": self.analysis_id,
            "repository": self.repository,
        }


@dataclass
class BudgetConfig:
    """Budget configuration for a team/org."""

    monthly_budget_usd: float = 100.0
    alert_threshold_pct: float = 80.0
    hard_limit: bool = False  # If True, reject analysis when over budget
    cost_rates: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_COST_RATES))


# =============================================================================
# Metrics Collector
# =============================================================================


class CostMetricsCollector:
    """Collects and aggregates cost/time metrics per analysis run.

    Usage:
        collector = CostMetricsCollector()
        collector.record_llm_usage("gpt-4", input_tokens=500, output_tokens=200, analysis_id="abc")
        collector.record_z3_time(1500.0, analysis_id="abc")
        collector.record_cache_hit(saved_ms=42.0, analysis_id="abc")
        summary = collector.get_summary()
    """

    def __init__(self, budget: BudgetConfig | None = None) -> None:
        self._budget = budget or BudgetConfig()
        self._records: list[CostRecord] = []
        self._llm_tokens_total = 0
        self._z3_time_total_ms = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        self._analyses_total = 0

    def record_llm_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        analysis_id: str = "",
        repository: str = "",
    ) -> CostRecord:
        """Record LLM token usage."""
        rates = self._budget.cost_rates
        input_rate = rates.get(f"{model}-input", rates.get("gpt-4-input", 0.00003))
        output_rate = rates.get(f"{model}-output", rates.get("gpt-4-output", 0.00006))

        cost = (input_tokens * input_rate) + (output_tokens * output_rate)
        total_tokens = input_tokens + output_tokens
        self._llm_tokens_total += total_tokens

        record = CostRecord(
            category=CostCategory.LLM_TOKENS.value,
            description=f"{model}: {input_tokens} in + {output_tokens} out",
            amount_usd=cost,
            quantity=total_tokens,
            unit="tokens",
            model=model,
            analysis_id=analysis_id,
            repository=repository,
        )
        self._records.append(record)
        return record

    def record_z3_time(
        self,
        solve_time_ms: float,
        analysis_id: str = "",
        repository: str = "",
    ) -> CostRecord:
        """Record Z3 solver compute time."""
        cpu_seconds = solve_time_ms / 1000
        cost = cpu_seconds * self._budget.cost_rates.get("z3-cpu-second", 0.001)
        self._z3_time_total_ms += solve_time_ms

        record = CostRecord(
            category=CostCategory.Z3_COMPUTE.value,
            description=f"Z3 solve: {solve_time_ms:.0f}ms",
            amount_usd=cost,
            quantity=solve_time_ms,
            unit="ms",
            analysis_id=analysis_id,
            repository=repository,
        )
        self._records.append(record)
        return record

    def record_cache_hit(
        self,
        saved_ms: float = 0.0,
        analysis_id: str = "",
    ) -> None:
        """Record a cache hit."""
        self._cache_hits += 1

    def record_cache_miss(self, analysis_id: str = "") -> None:
        """Record a cache miss."""
        self._cache_misses += 1

    def record_analysis_start(self, analysis_id: str = "") -> None:
        """Record the start of an analysis."""
        self._analyses_total += 1

    def get_total_cost(self) -> float:
        """Get total cost across all records."""
        return sum(r.amount_usd for r in self._records)

    def get_budget_usage(self) -> dict[str, Any]:
        """Get current budget usage status."""
        total = self.get_total_cost()
        budget = self._budget.monthly_budget_usd
        pct = (total / budget * 100) if budget > 0 else 0

        return {
            "total_cost_usd": round(total, 4),
            "monthly_budget_usd": budget,
            "usage_pct": round(pct, 1),
            "remaining_usd": round(max(0, budget - total), 4),
            "over_budget": total > budget,
            "alert_triggered": pct >= self._budget.alert_threshold_pct,
            "hard_limit_active": self._budget.hard_limit,
        }

    def is_within_budget(self) -> bool:
        """Check if current spending is within budget."""
        if not self._budget.hard_limit:
            return True
        return self.get_total_cost() <= self._budget.monthly_budget_usd

    def get_summary(self) -> dict[str, Any]:
        """Get a complete metrics summary."""
        cache_total = self._cache_hits + self._cache_misses
        return {
            "total_cost_usd": round(self.get_total_cost(), 4),
            "llm_tokens_total": self._llm_tokens_total,
            "z3_time_total_ms": round(self._z3_time_total_ms, 1),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": round(self._cache_hits / cache_total, 3) if cache_total > 0 else 0.0,
            "analyses_total": self._analyses_total,
            "records_count": len(self._records),
            "cost_by_category": self._cost_by_category(),
            "cost_by_model": self._cost_by_model(),
            "budget": self.get_budget_usage(),
        }

    def _cost_by_category(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for r in self._records:
            result[r.category] = result.get(r.category, 0) + r.amount_usd
        return {k: round(v, 6) for k, v in result.items()}

    def _cost_by_model(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for r in self._records:
            if r.model:
                result[r.model] = result.get(r.model, 0) + r.amount_usd
        return {k: round(v, 6) for k, v in result.items()}

    def get_records(
        self,
        category: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get cost records with optional filtering."""
        records = self._records
        if category:
            records = [r for r in records if r.category == category]
        return [r.to_dict() for r in records[-limit:]]


# =============================================================================
# Prometheus Metrics Export
# =============================================================================


class CostPrometheusMetrics:
    """Exports cost/time metrics in Prometheus text format."""

    NAMESPACE = "codeverify_cost"

    def __init__(self, collector: CostMetricsCollector) -> None:
        self._collector = collector

    def export(self) -> str:
        """Export as Prometheus text exposition format."""
        s = self._collector.get_summary()
        ns = self.NAMESPACE
        lines = [
            f"# HELP {ns}_total_usd Total cost in USD.",
            f"# TYPE {ns}_total_usd counter",
            f"{ns}_total_usd {s['total_cost_usd']}",
            "",
            f"# HELP {ns}_llm_tokens_total Total LLM tokens consumed.",
            f"# TYPE {ns}_llm_tokens_total counter",
            f"{ns}_llm_tokens_total {s['llm_tokens_total']}",
            "",
            f"# HELP {ns}_z3_time_ms_total Total Z3 solver time in milliseconds.",
            f"# TYPE {ns}_z3_time_ms_total counter",
            f"{ns}_z3_time_ms_total {s['z3_time_total_ms']}",
            "",
            f"# HELP {ns}_cache_hits_total Cache hits total.",
            f"# TYPE {ns}_cache_hits_total counter",
            f"{ns}_cache_hits_total {s['cache_hits']}",
            "",
            f"# HELP {ns}_cache_hit_rate Cache hit rate.",
            f"# TYPE {ns}_cache_hit_rate gauge",
            f"{ns}_cache_hit_rate {s['cache_hit_rate']}",
            "",
            f"# HELP {ns}_analyses_total Total analyses performed.",
            f"# TYPE {ns}_analyses_total counter",
            f"{ns}_analyses_total {s['analyses_total']}",
            "",
            f"# HELP {ns}_budget_usage_pct Budget usage percentage.",
            f"# TYPE {ns}_budget_usage_pct gauge",
            f"{ns}_budget_usage_pct {s['budget']['usage_pct']}",
            "",
        ]

        # Per-model costs
        for model, cost in s["cost_by_model"].items():
            safe_model = model.replace("-", "_").replace(".", "_")
            lines.append(f'{ns}_model_cost_usd{{model="{model}"}} {cost}')

        lines.append("")
        return "\n".join(lines) + "\n"


# =============================================================================
# Singleton
# =============================================================================


_collector: CostMetricsCollector | None = None


def get_cost_collector(budget: BudgetConfig | None = None) -> CostMetricsCollector:
    """Get the global cost metrics collector."""
    global _collector
    if _collector is None:
        _collector = CostMetricsCollector(budget)
    return _collector


def reset_cost_collector() -> None:
    """Reset the global collector (for testing)."""
    global _collector
    _collector = None
