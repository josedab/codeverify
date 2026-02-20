"""Predictive Defect Heatmap.

ML model predicting defect-prone files/functions from historical
findings, git history, and complexity metrics.

Features:
- Defect prediction model using weighted feature scoring
- File/function risk heatmap generation
- Trend analysis for risk trajectory
- Integration points for Jira/Linear ticket creation
"""

from __future__ import annotations

import hashlib
import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class RiskTrend(str, Enum):
    INCREASING = "increasing"
    STABLE = "stable"
    DECREASING = "decreasing"


class HeatmapLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SAFE = "safe"


@dataclass
class FileHistory:
    """Historical data for a file."""
    file_path: str = ""
    total_findings: int = 0
    critical_findings: int = 0
    bug_fix_commits: int = 0
    total_commits: int = 0
    authors: int = 1
    age_days: int = 30
    lines_of_code: int = 100
    complexity: float = 5.0
    last_finding_days_ago: int = 30
    churn_rate: float = 0.1


@dataclass
class PredictionResult:
    """Defect prediction for a file/function."""
    file_path: str = ""
    function_name: str = ""
    risk_score: float = 0.0
    heatmap_level: HeatmapLevel = HeatmapLevel.MEDIUM
    predicted_defects_next_sprint: float = 0.0
    confidence: float = 0.5
    top_factors: list[str] = field(default_factory=list)
    trend: RiskTrend = RiskTrend.STABLE


@dataclass
class HeatmapData:
    """Complete heatmap for a repository."""
    repo: str = ""
    predictions: list[PredictionResult] = field(default_factory=list)
    total_files: int = 0
    high_risk_count: int = 0
    avg_risk: float = 0.0
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def risk_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = defaultdict(int)
        for p in self.predictions:
            dist[p.heatmap_level.value] += 1
        return dict(dist)


@dataclass
class TicketSuggestion:
    """Suggested Jira/Linear ticket from prediction."""
    title: str = ""
    description: str = ""
    priority: str = "medium"
    labels: list[str] = field(default_factory=list)
    file_path: str = ""
    risk_score: float = 0.0


class DefectPredictor:
    """Predicts defect probability using weighted features."""

    FEATURE_WEIGHTS = {
        "historical_density": 0.25,
        "churn_rate": 0.20,
        "complexity": 0.15,
        "recency": 0.15,
        "author_count": 0.10,
        "critical_ratio": 0.10,
        "age_factor": 0.05,
    }

    def predict(self, history: FileHistory) -> PredictionResult:
        factors: dict[str, float] = {}
        top_factors: list[str] = []

        density = history.total_findings / max(history.lines_of_code / 1000, 0.1)
        factors["historical_density"] = min(1.0, density / 10)
        factors["churn_rate"] = min(1.0, history.churn_rate)
        factors["complexity"] = min(1.0, history.complexity / 30)
        factors["recency"] = max(0.0, 1.0 - history.last_finding_days_ago / 90)
        factors["author_count"] = min(1.0, history.authors / 10)
        factors["critical_ratio"] = history.critical_findings / max(history.total_findings, 1)
        factors["age_factor"] = min(1.0, 30 / max(history.age_days, 1))

        score = sum(factors[k] * self.FEATURE_WEIGHTS[k] for k in factors)
        score = round(min(1.0, max(0.0, score)), 3)

        sorted_factors = sorted(factors.items(), key=lambda x: x[1] * self.FEATURE_WEIGHTS.get(x[0], 0), reverse=True)
        for name, val in sorted_factors[:3]:
            if val > 0.3:
                top_factors.append(f"{name}: {val:.2f}")

        level = (HeatmapLevel.CRITICAL if score >= 0.8 else HeatmapLevel.HIGH if score >= 0.6
                 else HeatmapLevel.MEDIUM if score >= 0.4 else HeatmapLevel.LOW if score >= 0.2 else HeatmapLevel.SAFE)

        predicted = round(score * 5, 1)

        trend = RiskTrend.INCREASING if factors["recency"] > 0.6 else RiskTrend.DECREASING if factors["recency"] < 0.2 else RiskTrend.STABLE

        return PredictionResult(
            file_path=history.file_path, risk_score=score,
            heatmap_level=level, predicted_defects_next_sprint=predicted,
            confidence=0.6 + 0.3 * min(history.total_commits / 50, 1.0),
            top_factors=top_factors, trend=trend,
        )


class DefectHeatmapService:
    """Main service for predictive defect heatmaps."""

    def __init__(self) -> None:
        self._predictor = DefectPredictor()
        self._heatmaps: list[HeatmapData] = []

    def generate_heatmap(self, repo: str, file_histories: list[FileHistory]) -> HeatmapData:
        predictions = [self._predictor.predict(h) for h in file_histories]
        predictions.sort(key=lambda p: p.risk_score, reverse=True)
        high_risk = sum(1 for p in predictions if p.heatmap_level in (HeatmapLevel.CRITICAL, HeatmapLevel.HIGH))
        avg = sum(p.risk_score for p in predictions) / len(predictions) if predictions else 0.0
        heatmap = HeatmapData(
            repo=repo, predictions=predictions, total_files=len(predictions),
            high_risk_count=high_risk, avg_risk=round(avg, 3),
        )
        self._heatmaps.append(heatmap)
        return heatmap

    def suggest_tickets(self, heatmap: HeatmapData, max_tickets: int = 5) -> list[TicketSuggestion]:
        tickets: list[TicketSuggestion] = []
        for p in heatmap.predictions[:max_tickets]:
            if p.heatmap_level not in (HeatmapLevel.CRITICAL, HeatmapLevel.HIGH):
                continue
            tickets.append(TicketSuggestion(
                title=f"Verification focus: {p.file_path}",
                description=f"Predicted {p.predicted_defects_next_sprint} defects. Risk: {p.risk_score:.0%}. Factors: {', '.join(p.top_factors)}",
                priority="high" if p.heatmap_level == HeatmapLevel.CRITICAL else "medium",
                labels=["verification", "predicted-risk"], file_path=p.file_path, risk_score=p.risk_score,
            ))
        return tickets

    def get_heatmaps(self) -> list[HeatmapData]:
        return list(self._heatmaps)


_heatmap_instance: DefectHeatmapService | None = None
def get_defect_heatmap_service() -> DefectHeatmapService:
    global _heatmap_instance
    if _heatmap_instance is None: _heatmap_instance = DefectHeatmapService()
    return _heatmap_instance
def reset_defect_heatmap_service() -> None:
    global _heatmap_instance
    _heatmap_instance = None
