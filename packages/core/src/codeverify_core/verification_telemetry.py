"""Verification Telemetry & Benchmarking.

Opt-in anonymized telemetry comparing verification effectiveness
across organizations with cross-org benchmarking and reports.

Features:
- Anonymized metric collection (FP rate, detection rate, cost/bug, time saved)
- Cross-org benchmarking with percentile rankings
- Quarterly report data generation
- Privacy-preserving aggregation
"""

from __future__ import annotations

import hashlib
import statistics
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class MetricType(str, Enum):
    FALSE_POSITIVE_RATE = "false_positive_rate"
    DETECTION_RATE = "detection_rate"
    COST_PER_BUG = "cost_per_bug"
    TIME_SAVED_HOURS = "time_saved_hours"
    VERIFICATION_COVERAGE = "verification_coverage"
    FIX_RATE = "fix_rate"
    MTTR_HOURS = "mttr_hours"


class BenchmarkTier(str, Enum):
    TOP_10 = "top_10_percent"
    TOP_25 = "top_25_percent"
    MEDIAN = "median"
    BELOW_MEDIAN = "below_median"


@dataclass
class OrgTelemetry:
    """Telemetry data for an organization."""

    org_id_hash: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    period: str = ""
    opted_in: bool = True
    collected_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class BenchmarkResult:
    """Benchmarking result for an org against industry."""

    org_id_hash: str = ""
    metric: MetricType = MetricType.FALSE_POSITIVE_RATE
    org_value: float = 0.0
    industry_median: float = 0.0
    industry_p25: float = 0.0
    industry_p75: float = 0.0
    percentile: float = 0.0
    tier: BenchmarkTier = BenchmarkTier.MEDIAN


@dataclass
class QuarterlyReport:
    """Quarterly State of Code Verification report data."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    period: str = ""
    participating_orgs: int = 0
    total_verifications: int = 0
    industry_averages: dict[str, float] = field(default_factory=dict)
    trends: dict[str, str] = field(default_factory=dict)
    top_findings: list[dict[str, Any]] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class TelemetryAggregator:
    """Aggregates telemetry with privacy guarantees."""

    def aggregate(self, telemetry: list[OrgTelemetry]) -> dict[str, dict[str, float]]:
        if len(telemetry) < 5:
            return {}
        metric_values: dict[str, list[float]] = defaultdict(list)
        for t in telemetry:
            for metric, value in t.metrics.items():
                metric_values[metric].append(value)
        result: dict[str, dict[str, float]] = {}
        for metric, values in metric_values.items():
            if len(values) < 5:
                continue
            sorted_vals = sorted(values)
            result[metric] = {
                "median": round(statistics.median(values), 4),
                "mean": round(statistics.mean(values), 4),
                "p25": round(sorted_vals[len(sorted_vals) // 4], 4),
                "p75": round(sorted_vals[3 * len(sorted_vals) // 4], 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "count": len(values),
            }
        return result


class VerificationTelemetryService:
    """Main service for verification telemetry and benchmarking."""

    def __init__(self) -> None:
        self._telemetry: list[OrgTelemetry] = []
        self._aggregator = TelemetryAggregator()
        self._reports: list[QuarterlyReport] = []

    def submit_telemetry(
        self, org_id: str, metrics: dict[str, float], period: str = ""
    ) -> OrgTelemetry:
        org_hash = hashlib.sha256(org_id.encode()).hexdigest()[:10]
        t = OrgTelemetry(org_id_hash=org_hash, metrics=metrics, period=period)
        self._telemetry.append(t)
        return t

    def benchmark(self, org_id: str) -> list[BenchmarkResult]:
        org_hash = hashlib.sha256(org_id.encode()).hexdigest()[:10]
        org_data = [t for t in self._telemetry if t.org_id_hash == org_hash]
        if not org_data:
            return []
        agg = self._aggregator.aggregate(self._telemetry)
        results: list[BenchmarkResult] = []
        latest = org_data[-1]
        for metric_name, org_value in latest.metrics.items():
            stats = agg.get(metric_name)
            if not stats:
                continue
            all_vals = sorted(
                [t.metrics.get(metric_name, 0) for t in self._telemetry if metric_name in t.metrics]
            )
            rank = sum(1 for v in all_vals if v <= org_value)
            percentile = round(rank / len(all_vals) * 100, 1) if all_vals else 50.0
            tier = (
                BenchmarkTier.TOP_10
                if percentile >= 90
                else BenchmarkTier.TOP_25
                if percentile >= 75
                else BenchmarkTier.MEDIAN
                if percentile >= 50
                else BenchmarkTier.BELOW_MEDIAN
            )
            try:
                mt = MetricType(metric_name)
            except ValueError:
                continue
            results.append(
                BenchmarkResult(
                    org_id_hash=org_hash,
                    metric=mt,
                    org_value=org_value,
                    industry_median=stats["median"],
                    industry_p25=stats["p25"],
                    industry_p75=stats["p75"],
                    percentile=percentile,
                    tier=tier,
                )
            )
        return results

    def generate_report(self, period: str = "Q1-2026") -> QuarterlyReport:
        agg = self._aggregator.aggregate(self._telemetry)
        averages = {k: v["mean"] for k, v in agg.items()}
        trends: dict[str, str] = {}
        for k in averages:
            trends[k] = "stable"
        report = QuarterlyReport(
            period=period,
            participating_orgs=len({t.org_id_hash for t in self._telemetry}),
            total_verifications=int(
                sum(t.metrics.get("total_verifications", 0) for t in self._telemetry)
            ),
            industry_averages=averages,
            trends=trends,
        )
        self._reports.append(report)
        return report

    def get_reports(self) -> list[QuarterlyReport]:
        return list(self._reports)


_telemetry_bench_instance: VerificationTelemetryService | None = None


def get_telemetry_benchmark_service() -> VerificationTelemetryService:
    global _telemetry_bench_instance
    if _telemetry_bench_instance is None:
        _telemetry_bench_instance = VerificationTelemetryService()
    return _telemetry_bench_instance


def reset_telemetry_benchmark_service() -> None:
    global _telemetry_bench_instance
    _telemetry_bench_instance = None
