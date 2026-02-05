"""
Code Evolution Tracker

Tracks how code quality and verification status evolves over time
with git history integration. Provides historical trend analysis,
regression detection, and quality metrics over commits.
"""

from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4


class MetricType(str, Enum):
    """Types of code quality metrics."""
    COMPLEXITY = "complexity"
    COVERAGE = "coverage"
    SECURITY_SCORE = "security_score"
    MAINTAINABILITY = "maintainability"
    VERIFICATION_RATE = "verification_rate"
    BUG_DENSITY = "bug_density"
    TECH_DEBT = "tech_debt"
    AI_CODE_RATIO = "ai_code_ratio"


class TrendDirection(str, Enum):
    """Direction of a trend."""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"


class RegressionSeverity(str, Enum):
    """Severity of a detected regression."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class CommitSnapshot:
    """Snapshot of code quality at a specific commit."""
    id: str
    commit_sha: str
    commit_message: str
    author: str
    timestamp: datetime
    metrics: Dict[str, float]
    files_changed: int
    lines_added: int
    lines_removed: int
    findings_count: int
    verified_functions: int
    total_functions: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "commit_sha": self.commit_sha,
            "commit_message": self.commit_message,
            "author": self.author,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "files_changed": self.files_changed,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "findings_count": self.findings_count,
            "verified_functions": self.verified_functions,
            "total_functions": self.total_functions,
            "verification_rate": self.verified_functions / max(1, self.total_functions),
        }


@dataclass
class MetricTrend:
    """Trend analysis for a specific metric."""
    metric_type: MetricType
    direction: TrendDirection
    current_value: float
    previous_value: float
    change_percent: float
    trend_slope: float
    volatility: float
    data_points: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_type": self.metric_type.value,
            "direction": self.direction.value,
            "current_value": self.current_value,
            "previous_value": self.previous_value,
            "change_percent": self.change_percent,
            "trend_slope": self.trend_slope,
            "volatility": self.volatility,
            "data_points": self.data_points,
        }


@dataclass
class DetectedRegression:
    """A detected quality regression."""
    id: str
    severity: RegressionSeverity
    metric_type: MetricType
    description: str
    introduced_in_commit: str
    previous_commit: str
    value_before: float
    value_after: float
    change_percent: float
    detected_at: datetime
    affected_files: List[str]
    suggested_action: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "severity": self.severity.value,
            "metric_type": self.metric_type.value,
            "description": self.description,
            "introduced_in_commit": self.introduced_in_commit,
            "previous_commit": self.previous_commit,
            "value_before": self.value_before,
            "value_after": self.value_after,
            "change_percent": self.change_percent,
            "detected_at": self.detected_at.isoformat(),
            "affected_files": self.affected_files[:10],
            "suggested_action": self.suggested_action,
        }


@dataclass
class EvolutionReport:
    """Complete evolution report for a time period."""
    id: str
    repository: str
    start_date: datetime
    end_date: datetime
    total_commits: int
    trends: List[MetricTrend]
    regressions: List[DetectedRegression]
    top_contributors: List[Dict[str, Any]]
    quality_summary: Dict[str, Any]
    generated_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "repository": self.repository,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_commits": self.total_commits,
            "trends": [t.to_dict() for t in self.trends],
            "regressions": [r.to_dict() for r in self.regressions],
            "top_contributors": self.top_contributors,
            "quality_summary": self.quality_summary,
            "generated_at": self.generated_at.isoformat(),
        }


class TrendAnalyzer:
    """Analyzes metric trends over time."""

    def __init__(self, volatility_threshold: float = 0.15):
        self.volatility_threshold = volatility_threshold

    def analyze(
        self,
        metric_type: MetricType,
        values: List[Tuple[datetime, float]],
    ) -> MetricTrend:
        """Analyze trend for a metric."""
        if len(values) < 2:
            return MetricTrend(
                metric_type=metric_type,
                direction=TrendDirection.STABLE,
                current_value=values[-1][1] if values else 0.0,
                previous_value=values[0][1] if values else 0.0,
                change_percent=0.0,
                trend_slope=0.0,
                volatility=0.0,
                data_points=len(values),
            )

        # Sort by timestamp
        sorted_values = sorted(values, key=lambda x: x[0])
        just_values = [v for _, v in sorted_values]

        current = just_values[-1]
        previous = just_values[0]

        # Calculate change percent
        if previous != 0:
            change_percent = ((current - previous) / abs(previous)) * 100
        else:
            change_percent = 100.0 if current > 0 else 0.0

        # Calculate trend slope using simple linear regression
        n = len(just_values)
        x_mean = (n - 1) / 2
        y_mean = sum(just_values) / n

        numerator = sum((i - x_mean) * (just_values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0

        # Calculate volatility (coefficient of variation)
        if y_mean != 0:
            std_dev = statistics.stdev(just_values) if len(just_values) > 1 else 0
            volatility = std_dev / abs(y_mean)
        else:
            volatility = 0.0

        # Determine direction
        if volatility > self.volatility_threshold:
            direction = TrendDirection.VOLATILE
        elif abs(change_percent) < 5:
            direction = TrendDirection.STABLE
        elif self._is_higher_better(metric_type):
            direction = TrendDirection.IMPROVING if change_percent > 0 else TrendDirection.DECLINING
        else:
            direction = TrendDirection.DECLINING if change_percent > 0 else TrendDirection.IMPROVING

        return MetricTrend(
            metric_type=metric_type,
            direction=direction,
            current_value=current,
            previous_value=previous,
            change_percent=change_percent,
            trend_slope=slope,
            volatility=volatility,
            data_points=len(values),
        )

    def _is_higher_better(self, metric_type: MetricType) -> bool:
        """Check if higher values are better for a metric."""
        higher_better = {
            MetricType.COVERAGE,
            MetricType.SECURITY_SCORE,
            MetricType.MAINTAINABILITY,
            MetricType.VERIFICATION_RATE,
        }
        return metric_type in higher_better


class RegressionDetector:
    """Detects quality regressions between commits."""

    # Thresholds for regression detection (percent change)
    THRESHOLDS = {
        MetricType.COMPLEXITY: 20.0,  # 20% increase in complexity
        MetricType.COVERAGE: -10.0,  # 10% decrease in coverage
        MetricType.SECURITY_SCORE: -15.0,  # 15% decrease in security
        MetricType.MAINTAINABILITY: -15.0,  # 15% decrease in maintainability
        MetricType.VERIFICATION_RATE: -10.0,  # 10% decrease in verification
        MetricType.BUG_DENSITY: 25.0,  # 25% increase in bug density
        MetricType.TECH_DEBT: 30.0,  # 30% increase in tech debt
        MetricType.AI_CODE_RATIO: 50.0,  # 50% increase in AI code (informational)
    }

    def detect(
        self,
        snapshots: List[CommitSnapshot],
    ) -> List[DetectedRegression]:
        """Detect regressions between consecutive snapshots."""
        if len(snapshots) < 2:
            return []

        regressions: List[DetectedRegression] = []
        sorted_snapshots = sorted(snapshots, key=lambda s: s.timestamp)

        for i in range(1, len(sorted_snapshots)):
            prev = sorted_snapshots[i - 1]
            curr = sorted_snapshots[i]

            for metric_type in MetricType:
                prev_value = prev.metrics.get(metric_type.value, 0.0)
                curr_value = curr.metrics.get(metric_type.value, 0.0)

                if prev_value == 0:
                    continue

                change_percent = ((curr_value - prev_value) / abs(prev_value)) * 100
                threshold = self.THRESHOLDS.get(metric_type, 25.0)

                # Check if regression detected based on metric type
                is_regression = self._check_regression(metric_type, change_percent, threshold)

                if is_regression:
                    severity = self._calculate_severity(metric_type, change_percent)
                    regression = DetectedRegression(
                        id=str(uuid4()),
                        severity=severity,
                        metric_type=metric_type,
                        description=self._generate_description(metric_type, change_percent),
                        introduced_in_commit=curr.commit_sha,
                        previous_commit=prev.commit_sha,
                        value_before=prev_value,
                        value_after=curr_value,
                        change_percent=change_percent,
                        detected_at=datetime.now(),
                        affected_files=[],  # Would come from git diff
                        suggested_action=self._suggest_action(metric_type),
                    )
                    regressions.append(regression)

        return regressions

    def _check_regression(
        self,
        metric_type: MetricType,
        change_percent: float,
        threshold: float,
    ) -> bool:
        """Check if a change represents a regression."""
        # For metrics where lower is better
        if metric_type in {MetricType.COMPLEXITY, MetricType.BUG_DENSITY, MetricType.TECH_DEBT}:
            return change_percent > threshold
        # For metrics where higher is better
        else:
            return change_percent < threshold

    def _calculate_severity(
        self,
        metric_type: MetricType,
        change_percent: float,
    ) -> RegressionSeverity:
        """Calculate severity based on change magnitude."""
        abs_change = abs(change_percent)

        # Critical metrics get higher severity
        critical_metrics = {MetricType.SECURITY_SCORE, MetricType.VERIFICATION_RATE}

        if metric_type in critical_metrics:
            if abs_change > 30:
                return RegressionSeverity.CRITICAL
            elif abs_change > 20:
                return RegressionSeverity.HIGH
            elif abs_change > 10:
                return RegressionSeverity.MEDIUM
            else:
                return RegressionSeverity.LOW
        else:
            if abs_change > 50:
                return RegressionSeverity.CRITICAL
            elif abs_change > 30:
                return RegressionSeverity.HIGH
            elif abs_change > 15:
                return RegressionSeverity.MEDIUM
            else:
                return RegressionSeverity.LOW

    def _generate_description(
        self,
        metric_type: MetricType,
        change_percent: float,
    ) -> str:
        """Generate human-readable description."""
        direction = "increased" if change_percent > 0 else "decreased"
        metric_name = metric_type.value.replace("_", " ").title()
        return f"{metric_name} {direction} by {abs(change_percent):.1f}%"

    def _suggest_action(self, metric_type: MetricType) -> str:
        """Suggest action for regression."""
        suggestions = {
            MetricType.COMPLEXITY: "Review and simplify complex code paths",
            MetricType.COVERAGE: "Add tests for new or modified code",
            MetricType.SECURITY_SCORE: "Run security scan and address vulnerabilities",
            MetricType.MAINTAINABILITY: "Refactor code to improve readability",
            MetricType.VERIFICATION_RATE: "Add formal specifications for new functions",
            MetricType.BUG_DENSITY: "Review code quality and add more testing",
            MetricType.TECH_DEBT: "Address accumulated technical debt",
            MetricType.AI_CODE_RATIO: "Review AI-generated code for quality",
        }
        return suggestions.get(metric_type, "Review the changes and assess impact")


class CodeEvolutionTracker:
    """Main tracker for code evolution over time."""

    def __init__(self):
        self.snapshots: Dict[str, List[CommitSnapshot]] = defaultdict(list)  # repo -> snapshots
        self.trend_analyzer = TrendAnalyzer()
        self.regression_detector = RegressionDetector()

    def record_snapshot(
        self,
        repository: str,
        commit_sha: str,
        commit_message: str,
        author: str,
        timestamp: Optional[datetime] = None,
        metrics: Optional[Dict[str, float]] = None,
        files_changed: int = 0,
        lines_added: int = 0,
        lines_removed: int = 0,
        findings_count: int = 0,
        verified_functions: int = 0,
        total_functions: int = 0,
    ) -> CommitSnapshot:
        """Record a snapshot for a commit."""
        snapshot = CommitSnapshot(
            id=str(uuid4()),
            commit_sha=commit_sha,
            commit_message=commit_message,
            author=author,
            timestamp=timestamp or datetime.now(),
            metrics=metrics or {},
            files_changed=files_changed,
            lines_added=lines_added,
            lines_removed=lines_removed,
            findings_count=findings_count,
            verified_functions=verified_functions,
            total_functions=total_functions,
        )

        self.snapshots[repository].append(snapshot)
        return snapshot

    def get_snapshots(
        self,
        repository: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[CommitSnapshot]:
        """Get snapshots for a repository."""
        snapshots = self.snapshots.get(repository, [])

        if start_date:
            snapshots = [s for s in snapshots if s.timestamp >= start_date]
        if end_date:
            snapshots = [s for s in snapshots if s.timestamp <= end_date]

        # Sort by timestamp descending
        snapshots = sorted(snapshots, key=lambda s: s.timestamp, reverse=True)

        return snapshots[:limit]

    def analyze_trends(
        self,
        repository: str,
        metric_types: Optional[List[MetricType]] = None,
        days: int = 30,
    ) -> List[MetricTrend]:
        """Analyze trends for specified metrics."""
        cutoff = datetime.now() - timedelta(days=days)
        snapshots = [s for s in self.snapshots.get(repository, []) if s.timestamp >= cutoff]

        if not snapshots:
            return []

        if metric_types is None:
            metric_types = list(MetricType)

        trends = []
        for metric_type in metric_types:
            values = [
                (s.timestamp, s.metrics.get(metric_type.value, 0.0))
                for s in snapshots
                if metric_type.value in s.metrics
            ]

            if values:
                trend = self.trend_analyzer.analyze(metric_type, values)
                trends.append(trend)

        return trends

    def detect_regressions(
        self,
        repository: str,
        days: int = 7,
    ) -> List[DetectedRegression]:
        """Detect regressions in recent commits."""
        cutoff = datetime.now() - timedelta(days=days)
        snapshots = [s for s in self.snapshots.get(repository, []) if s.timestamp >= cutoff]

        return self.regression_detector.detect(snapshots)

    def generate_report(
        self,
        repository: str,
        days: int = 30,
    ) -> EvolutionReport:
        """Generate comprehensive evolution report."""
        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()

        snapshots = self.get_snapshots(repository, start_date, end_date, limit=1000)
        trends = self.analyze_trends(repository, days=days)
        regressions = self.detect_regressions(repository, days=min(days, 7))

        # Calculate contributor statistics
        contributor_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"commits": 0, "lines_added": 0, "lines_removed": 0}
        )
        for s in snapshots:
            contributor_stats[s.author]["commits"] += 1
            contributor_stats[s.author]["lines_added"] += s.lines_added
            contributor_stats[s.author]["lines_removed"] += s.lines_removed

        top_contributors = sorted(
            [{"author": k, **v} for k, v in contributor_stats.items()],
            key=lambda x: x["commits"],
            reverse=True,
        )[:10]

        # Quality summary
        if snapshots:
            latest = snapshots[0]
            oldest = snapshots[-1]

            quality_summary = {
                "current_verification_rate": latest.verified_functions / max(1, latest.total_functions),
                "initial_verification_rate": oldest.verified_functions / max(1, oldest.total_functions),
                "current_findings": latest.findings_count,
                "initial_findings": oldest.findings_count,
                "total_lines_added": sum(s.lines_added for s in snapshots),
                "total_lines_removed": sum(s.lines_removed for s in snapshots),
                "avg_findings_per_commit": sum(s.findings_count for s in snapshots) / len(snapshots),
            }
        else:
            quality_summary = {}

        return EvolutionReport(
            id=str(uuid4()),
            repository=repository,
            start_date=start_date,
            end_date=end_date,
            total_commits=len(snapshots),
            trends=trends,
            regressions=regressions,
            top_contributors=top_contributors,
            quality_summary=quality_summary,
            generated_at=datetime.now(),
        )

    def compare_commits(
        self,
        repository: str,
        commit_sha_1: str,
        commit_sha_2: str,
    ) -> Dict[str, Any]:
        """Compare two specific commits."""
        snapshots = self.snapshots.get(repository, [])

        snapshot_1 = next((s for s in snapshots if s.commit_sha == commit_sha_1), None)
        snapshot_2 = next((s for s in snapshots if s.commit_sha == commit_sha_2), None)

        if not snapshot_1 or not snapshot_2:
            return {
                "error": "One or both commits not found",
                "found_1": snapshot_1 is not None,
                "found_2": snapshot_2 is not None,
            }

        # Calculate differences
        metric_diffs: Dict[str, Dict[str, float]] = {}
        all_metrics = set(snapshot_1.metrics.keys()) | set(snapshot_2.metrics.keys())

        for metric in all_metrics:
            val_1 = snapshot_1.metrics.get(metric, 0.0)
            val_2 = snapshot_2.metrics.get(metric, 0.0)

            if val_1 != 0:
                change_percent = ((val_2 - val_1) / abs(val_1)) * 100
            else:
                change_percent = 100.0 if val_2 > 0 else 0.0

            metric_diffs[metric] = {
                "value_1": val_1,
                "value_2": val_2,
                "change": val_2 - val_1,
                "change_percent": change_percent,
            }

        return {
            "commit_1": snapshot_1.to_dict(),
            "commit_2": snapshot_2.to_dict(),
            "metric_diffs": metric_diffs,
            "verification_rate_change": (
                (snapshot_2.verified_functions / max(1, snapshot_2.total_functions)) -
                (snapshot_1.verified_functions / max(1, snapshot_1.total_functions))
            ),
            "findings_change": snapshot_2.findings_count - snapshot_1.findings_count,
        }

    def get_metric_history(
        self,
        repository: str,
        metric_type: MetricType,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get historical values for a specific metric."""
        cutoff = datetime.now() - timedelta(days=days)
        snapshots = [
            s for s in self.snapshots.get(repository, [])
            if s.timestamp >= cutoff and metric_type.value in s.metrics
        ]

        snapshots = sorted(snapshots, key=lambda s: s.timestamp)

        return [
            {
                "commit_sha": s.commit_sha,
                "timestamp": s.timestamp.isoformat(),
                "value": s.metrics.get(metric_type.value, 0.0),
            }
            for s in snapshots
        ]

    def get_repositories(self) -> List[str]:
        """Get list of tracked repositories."""
        return list(self.snapshots.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        total_snapshots = sum(len(s) for s in self.snapshots.values())

        repo_stats = {}
        for repo, snapshots in self.snapshots.items():
            if snapshots:
                repo_stats[repo] = {
                    "total_commits": len(snapshots),
                    "first_commit": min(s.timestamp for s in snapshots).isoformat(),
                    "last_commit": max(s.timestamp for s in snapshots).isoformat(),
                }

        return {
            "total_repositories": len(self.snapshots),
            "total_snapshots": total_snapshots,
            "repositories": repo_stats,
        }

    def clear_repository(self, repository: str) -> bool:
        """Clear snapshots for a repository."""
        if repository in self.snapshots:
            del self.snapshots[repository]
            return True
        return False
