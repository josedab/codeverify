"""
Continuous Learning from Production

Learns from production telemetry to improve verification accuracy:
- Runtime monitoring integration: capture production failures, correlate with commits
- ML learning pipeline: learn from production bugs, auto-tune detection thresholds
- Adaptive verification rules: update rules based on production data
- A/B testing framework for evaluating new detection rules
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

logger = structlog.get_logger()


# =============================================================================
# Enums
# =============================================================================


class IncidentSeverity(str, Enum):
    """Severity levels for production incidents."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class LearningStrategy(str, Enum):
    """Strategies for learning from production data."""

    SUPERVISED = "supervised"
    REINFORCEMENT = "reinforcement"
    THRESHOLD_TUNING = "threshold_tuning"
    PATTERN_EXTRACTION = "pattern_extraction"


class RuleUpdateAction(str, Enum):
    """Actions that can be taken on verification rules."""

    CREATE = "create"
    MODIFY = "modify"
    DISABLE = "disable"
    DELETE = "delete"
    TUNE_THRESHOLD = "tune_threshold"


class ABTestStatus(str, Enum):
    """Status of an A/B test."""

    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ProductionIncident:
    """A production incident captured from runtime monitoring."""

    id: str
    timestamp: datetime
    service: str
    error_type: str
    stack_trace: str
    severity: IncidentSeverity
    commit_sha: str | None
    file_path: str | None
    function_name: str | None
    root_cause: str | None
    resolved: bool = False
    resolution_time_hours: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "service": self.service,
            "error_type": self.error_type,
            "stack_trace": self.stack_trace,
            "severity": self.severity.value,
            "commit_sha": self.commit_sha,
            "file_path": self.file_path,
            "function_name": self.function_name,
            "root_cause": self.root_cause,
            "resolved": self.resolved,
            "resolution_time_hours": self.resolution_time_hours,
            "metadata": self.metadata,
        }


@dataclass
class IncidentCorrelation:
    """Correlation between a production incident and a code commit."""

    incident_id: str
    commit_sha: str
    confidence: float
    verification_result_id: str | None
    was_detected: bool
    detection_rule: str | None
    gap_analysis: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "incident_id": self.incident_id,
            "commit_sha": self.commit_sha,
            "confidence": self.confidence,
            "verification_result_id": self.verification_result_id,
            "was_detected": self.was_detected,
            "detection_rule": self.detection_rule,
            "gap_analysis": self.gap_analysis,
        }


@dataclass
class DetectionThreshold:
    """Threshold tuning data for a detection rule."""

    rule_id: str
    current_value: float
    recommended_value: float
    false_positive_rate: float
    false_negative_rate: float
    sample_size: int
    last_tuned: datetime | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
            "sample_size": self.sample_size,
            "last_tuned": self.last_tuned.isoformat() if self.last_tuned else None,
        }


@dataclass
class LearnedPattern:
    """A pattern extracted from production incidents."""

    id: str
    pattern_type: str
    description: str
    code_pattern: str
    detection_rule: str
    confidence: float
    source_incidents: list[str]
    false_positive_rate: float
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "code_pattern": self.code_pattern,
            "detection_rule": self.detection_rule,
            "confidence": self.confidence,
            "source_incidents": self.source_incidents,
            "false_positive_rate": self.false_positive_rate,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class RuleUpdate:
    """A proposed or applied update to a verification rule."""

    rule_id: str
    action: RuleUpdateAction
    before_state: dict[str, Any]
    after_state: dict[str, Any]
    justification: str
    confidence: float
    ab_test_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "action": self.action.value,
            "before_state": self.before_state,
            "after_state": self.after_state,
            "justification": self.justification,
            "confidence": self.confidence,
            "ab_test_id": self.ab_test_id,
        }


@dataclass
class ABTest:
    """An A/B test comparing two detection rule variants."""

    id: str
    name: str
    description: str
    status: ABTestStatus
    control_rule: dict[str, Any]
    variant_rule: dict[str, Any]
    control_metrics: dict[str, Any] = field(default_factory=dict)
    variant_metrics: dict[str, Any] = field(default_factory=dict)
    traffic_split: float = 0.5
    start_date: datetime | None = None
    end_date: datetime | None = None
    winner: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "control_rule": self.control_rule,
            "variant_rule": self.variant_rule,
            "control_metrics": self.control_metrics,
            "variant_metrics": self.variant_metrics,
            "traffic_split": self.traffic_split,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "winner": self.winner,
        }


@dataclass
class LearningReport:
    """Summary report for a production learning cycle."""

    period_start: datetime
    period_end: datetime
    incidents_analyzed: int
    patterns_learned: int
    rules_updated: int
    false_positive_reduction: float
    detection_improvement: float
    ab_tests_completed: int
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "incidents_analyzed": self.incidents_analyzed,
            "patterns_learned": self.patterns_learned,
            "rules_updated": self.rules_updated,
            "false_positive_reduction": self.false_positive_reduction,
            "detection_improvement": self.detection_improvement,
            "ab_tests_completed": self.ab_tests_completed,
            "recommendations": self.recommendations,
        }


# =============================================================================
# Incident Collection & Correlation
# =============================================================================


class IncidentCollector:
    """Collects production incidents and correlates them with code commits."""

    def __init__(self) -> None:
        self._incidents: dict[str, ProductionIncident] = {}
        self._correlations: dict[str, IncidentCorrelation] = {}
        self._verification_results: dict[str, dict[str, Any]] = {}

    def record_incident(self, incident: ProductionIncident) -> None:
        """Record a new production incident."""
        self._incidents[incident.id] = incident
        logger.info(
            "Production incident recorded",
            incident_id=incident.id,
            severity=incident.severity.value,
            service=incident.service,
        )

    def get_incidents(
        self,
        since: datetime | None = None,
        severity: IncidentSeverity | None = None,
    ) -> list[ProductionIncident]:
        """Retrieve incidents, optionally filtered by time and severity."""
        results: list[ProductionIncident] = []
        for incident in self._incidents.values():
            if since and incident.timestamp < since:
                continue
            if severity and incident.severity != severity:
                continue
            results.append(incident)
        results.sort(key=lambda i: i.timestamp, reverse=True)
        return results

    def correlate_with_commits(
        self, incident: ProductionIncident
    ) -> IncidentCorrelation | None:
        """Attempt to correlate an incident with the commit that introduced it."""
        if not incident.commit_sha:
            return None

        verification_id = self._verification_results.get(
            incident.commit_sha, {}
        ).get("id")
        was_detected = False
        detection_rule = None

        if verification_id:
            findings = self._verification_results[incident.commit_sha].get(
                "findings", []
            )
            for finding in findings:
                if (
                    finding.get("file_path") == incident.file_path
                    or finding.get("error_type") == incident.error_type
                ):
                    was_detected = True
                    detection_rule = finding.get("rule_id")
                    break

        if was_detected:
            gap = (
                f"Detected by rule '{detection_rule}' but incident still "
                f"occurred — finding may have been ignored or fix incomplete."
            )
        else:
            parts = [f"'{incident.error_type}' in '{incident.service}' was NOT detected."]
            if incident.file_path:
                parts.append(f"File: {incident.file_path}.")
            if incident.root_cause:
                parts.append(f"Root cause: {incident.root_cause}.")
            parts.append("New detection rule or threshold adjustment needed.")
            gap = " ".join(parts)

        # Confidence from available signal strength
        confidence = 0.0
        if incident.commit_sha:
            confidence += 0.3
        if incident.file_path:
            confidence += 0.2
        if incident.function_name:
            confidence += 0.15
        if verification_id:
            confidence += 0.2
        if incident.root_cause:
            confidence += 0.15

        correlation = IncidentCorrelation(
            incident_id=incident.id,
            commit_sha=incident.commit_sha,
            confidence=min(confidence, 1.0),
            verification_result_id=verification_id,
            was_detected=was_detected,
            detection_rule=detection_rule,
            gap_analysis=gap,
        )
        self._correlations[incident.id] = correlation
        logger.info(
            "Incident correlated",
            incident_id=incident.id,
            was_detected=was_detected,
            confidence=round(confidence, 2),
        )
        return correlation

    def get_undetected_incidents(self) -> list[ProductionIncident]:
        """Return incidents that were NOT detected by verification."""
        undetected: list[ProductionIncident] = []
        for iid, corr in self._correlations.items():
            if not corr.was_detected and iid in self._incidents:
                undetected.append(self._incidents[iid])
        for incident in self._incidents.values():
            if incident.id not in self._correlations:
                undetected.append(incident)
        return undetected

    def register_verification_result(
        self, commit_sha: str, result: dict[str, Any]
    ) -> None:
        """Register a verification result to enable commit correlation."""
        self._verification_results[commit_sha] = result


# =============================================================================
# Threshold Tuning
# =============================================================================


class ThresholdTuner:
    """Analyzes and tunes detection thresholds based on production outcomes."""

    def __init__(self) -> None:
        self._thresholds: dict[str, DetectionThreshold] = {}
        self._history: list[dict[str, Any]] = []

    def analyze_threshold(
        self,
        rule_id: str,
        incidents: list[ProductionIncident],
        verification_results: list[dict[str, Any]],
    ) -> DetectionThreshold:
        """Analyze a detection rule's threshold against production data."""
        tp, fp, fn = 0, 0, 0
        incident_files = {i.file_path for i in incidents if i.file_path}
        incident_errors = {i.error_type for i in incidents}

        for result in verification_results:
            for finding in result.get("findings", []):
                if finding.get("rule_id") != rule_id:
                    continue
                matched = (
                    finding.get("file_path") in incident_files
                    or finding.get("error_type") in incident_errors
                )
                if matched:
                    tp += 1
                else:
                    fp += 1

        for incident in incidents:
            caught = any(
                any(
                    f.get("rule_id") == rule_id
                    and (f.get("file_path") == incident.file_path
                         or f.get("error_type") == incident.error_type)
                    for f in r.get("findings", [])
                )
                for r in verification_results
            )
            if not caught:
                fn += 1

        tn = max(0, len(verification_results) - tp - fp - fn)
        sample_size = tp + fp + fn + tn
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        prev = self._thresholds.get(rule_id)
        current = prev.current_value if prev else 0.5
        recommended = self._calculate_optimal_threshold(fp_rate, fn_rate, current)

        threshold = DetectionThreshold(
            rule_id=rule_id,
            current_value=current,
            recommended_value=recommended,
            false_positive_rate=round(fp_rate, 4),
            false_negative_rate=round(fn_rate, 4),
            sample_size=sample_size,
            last_tuned=datetime.utcnow(),
        )
        self._thresholds[rule_id] = threshold
        logger.info(
            "Threshold analyzed",
            rule_id=rule_id,
            fp_rate=threshold.false_positive_rate,
            fn_rate=threshold.false_negative_rate,
            recommended=recommended,
        )
        return threshold

    def recommend_adjustments(
        self, thresholds: list[DetectionThreshold]
    ) -> list[RuleUpdate]:
        """Generate rule update recommendations from threshold analysis."""
        updates: list[RuleUpdate] = []
        for t in thresholds:
            delta = abs(t.recommended_value - t.current_value)
            if delta < 0.02 or t.sample_size < 10:
                continue

            parts: list[str] = []
            if t.false_positive_rate > 0.3:
                parts.append(f"High FP rate ({t.false_positive_rate:.1%})")
            if t.false_negative_rate > 0.2:
                parts.append(f"High FN rate ({t.false_negative_rate:.1%})")
            if not parts:
                parts.append(f"Optimization from {t.sample_size} samples")

            confidence = min(0.95, 0.5 + (t.sample_size / 200) + (delta * 0.5))
            updates.append(RuleUpdate(
                rule_id=t.rule_id,
                action=RuleUpdateAction.TUNE_THRESHOLD,
                before_state={"threshold": t.current_value, "fp_rate": t.false_positive_rate},
                after_state={"threshold": t.recommended_value},
                justification="; ".join(parts),
                confidence=round(confidence, 3),
            ))
            self._history.append({
                "rule_id": t.rule_id,
                "from": t.current_value,
                "to": t.recommended_value,
                "timestamp": datetime.utcnow().isoformat(),
            })
        return updates

    def _calculate_optimal_threshold(
        self, fp_rate: float, fn_rate: float, current: float
    ) -> float:
        """Calculate optimal threshold balancing false positives and negatives.

        False negatives (missed bugs) are weighted 2x since they are more
        costly in production than false positives.
        """
        fn_weight = 2.0
        if fn_rate > fp_rate * fn_weight:
            adjustment = -min(0.1, fn_rate * 0.3)
        elif fp_rate > fn_rate:
            adjustment = min(0.1, fp_rate * 0.3)
        else:
            adjustment = 0.0
        return round(max(0.05, min(0.95, current + adjustment)), 4)


# =============================================================================
# Pattern Extraction
# =============================================================================


class PatternExtractor:
    """Extracts reusable detection patterns from production incidents."""

    def __init__(self) -> None:
        self._patterns: dict[str, LearnedPattern] = {}

    def extract_patterns(
        self, incidents: list[ProductionIncident]
    ) -> list[LearnedPattern]:
        """Analyze incidents and extract common patterns for detection rules."""
        if not incidents:
            return []

        clusters = self._cluster_incidents(incidents)
        patterns: list[LearnedPattern] = []

        for key, group in clusters.items():
            if len(group) < 2:
                continue

            code_pattern = self._extract_code_pattern(group)
            detection_rule = self._generate_detection_rule(key, group)
            resolved = sum(1 for i in group if i.resolved)
            fp_estimate = 1.0 - (resolved / len(group)) if resolved else 0.1
            pid = hashlib.sha256(f"{key}:{code_pattern}".encode()).hexdigest()[:12]

            services = {i.service for i in group}
            desc = (
                f"Pattern '{key}' from {len(group)} incidents in "
                f"{', '.join(sorted(services))}."
            )
            root_causes = [i.root_cause for i in group if i.root_cause]
            if root_causes:
                desc += f" Causes: {'; '.join(root_causes[:3])}."

            pattern = LearnedPattern(
                id=pid,
                pattern_type=key,
                description=desc,
                code_pattern=code_pattern,
                detection_rule=detection_rule,
                confidence=min(0.95, len(group) / 10.0),
                source_incidents=[i.id for i in group],
                false_positive_rate=round(max(0.0, min(1.0, fp_estimate)), 4),
                created_at=datetime.utcnow(),
            )
            self._patterns[pattern.id] = pattern
            patterns.append(pattern)
            logger.info(
                "Pattern extracted",
                pattern_id=pattern.id,
                incident_count=len(group),
                confidence=pattern.confidence,
            )

        return patterns

    def _cluster_incidents(
        self, incidents: list[ProductionIncident]
    ) -> dict[str, list[ProductionIncident]]:
        """Group incidents by similar characteristics for pattern mining."""
        clusters: dict[str, list[ProductionIncident]] = defaultdict(list)
        for incident in incidents:
            clusters[incident.error_type].append(incident)
            clusters[f"{incident.service}:{incident.error_type}"].append(incident)

        # Keep clusters with enough data, prefer more specific keys
        filtered: dict[str, list[ProductionIncident]] = {}
        seen: list[set[str]] = []
        for key in sorted(clusters, key=lambda k: len(clusters[k]), reverse=True):
            ids = {i.id for i in clusters[key]}
            if not any(ids <= s for s in seen) and len(clusters[key]) >= 2:
                filtered[key] = clusters[key]
                seen.append(ids)
        return filtered

    def _generate_detection_rule(
        self, pattern: str, examples: list[ProductionIncident]
    ) -> str:
        """Generate a detection rule definition from a pattern and examples."""
        error_types = sorted({e.error_type for e in examples})
        services = sorted({e.service for e in examples})
        parts = [f"match_errors: [{', '.join(error_types)}]"]
        if len(services) <= 3:
            parts.append(f"match_services: [{', '.join(services)}]")
        file_paths = [e.file_path for e in examples if e.file_path]
        if file_paths:
            prefix = _common_prefix(file_paths)
            if prefix:
                parts.append(f"path_prefix: {prefix}")
        return "; ".join(parts)

    def _extract_code_pattern(
        self, incidents: list[ProductionIncident]
    ) -> str:
        """Extract a common code pattern from incident stack traces."""
        func_counts: dict[str, int] = defaultdict(int)
        for incident in incidents:
            if not incident.stack_trace:
                continue
            for func in re.findall(r"(?:in |at )(\w+(?:\.\w+)*)\s*\(", incident.stack_trace):
                func_counts[func] += 1
        if not func_counts:
            return incidents[0].error_type if incidents else "unknown"
        top = sorted(func_counts, key=func_counts.get, reverse=True)[:3]  # type: ignore[arg-type]
        return " -> ".join(top)


# =============================================================================
# A/B Testing
# =============================================================================


class ABTestManager:
    """Manages A/B tests for comparing detection rule variants."""

    def __init__(self) -> None:
        self._tests: dict[str, ABTest] = {}

    def create_test(
        self,
        name: str,
        control_rule: dict[str, Any],
        variant_rule: dict[str, Any],
        traffic_split: float = 0.5,
    ) -> ABTest:
        """Create a new A/B test between two rule configurations."""
        test_id = str(uuid4())[:8]
        test = ABTest(
            id=test_id,
            name=name,
            description=f"A/B test comparing control vs variant for '{name}'",
            status=ABTestStatus.DRAFT,
            control_rule=control_rule,
            variant_rule=variant_rule,
            control_metrics={"detections": 0, "true_positives": 0, "false_positives": 0, "total_checks": 0},
            variant_metrics={"detections": 0, "true_positives": 0, "false_positives": 0, "total_checks": 0},
            traffic_split=max(0.1, min(0.9, traffic_split)),
        )
        self._tests[test_id] = test
        logger.info("A/B test created", test_id=test_id, name=name)
        return test

    def start_test(self, test_id: str) -> None:
        """Start a draft A/B test."""
        test = self._tests.get(test_id)
        if not test:
            raise ValueError(f"A/B test '{test_id}' not found")
        if test.status != ABTestStatus.DRAFT:
            raise ValueError(f"Cannot start test in status '{test.status.value}'")
        test.status = ABTestStatus.RUNNING
        test.start_date = datetime.utcnow()
        logger.info("A/B test started", test_id=test_id)

    def record_result(
        self, test_id: str, is_variant: bool, detected: bool, was_real_bug: bool
    ) -> None:
        """Record a single detection result for an A/B test."""
        test = self._tests.get(test_id)
        if not test:
            raise ValueError(f"A/B test '{test_id}' not found")
        if test.status != ABTestStatus.RUNNING:
            raise ValueError(f"Cannot record for test in status '{test.status.value}'")

        metrics = test.variant_metrics if is_variant else test.control_metrics
        metrics["total_checks"] = metrics.get("total_checks", 0) + 1
        if detected:
            metrics["detections"] = metrics.get("detections", 0) + 1
            key = "true_positives" if was_real_bug else "false_positives"
            metrics[key] = metrics.get(key, 0) + 1

    def evaluate_test(self, test_id: str) -> ABTest:
        """Evaluate an A/B test and determine the winner."""
        test = self._tests.get(test_id)
        if not test:
            raise ValueError(f"A/B test '{test_id}' not found")

        c, v = test.control_metrics, test.variant_metrics
        c_score = self._score(c)
        v_score = self._score(v)
        significance = self._calculate_significance(c, v)

        if significance >= 0.95:
            test.winner = "variant" if v_score > c_score else "control"
            test.status = ABTestStatus.COMPLETED
            test.end_date = datetime.utcnow()
        elif significance < 0.5 and (c.get("total_checks", 0) + v.get("total_checks", 0)) > 100:
            test.status = ABTestStatus.COMPLETED
            test.end_date = datetime.utcnow()

        logger.info(
            "A/B test evaluated",
            test_id=test_id,
            significance=round(significance, 4),
            winner=test.winner,
        )
        return test

    def _calculate_significance(
        self, control: dict[str, Any], variant: dict[str, Any]
    ) -> float:
        """Calculate statistical significance using a z-test approximation."""
        n_c = control.get("total_checks", 0)
        n_v = variant.get("total_checks", 0)
        if n_c < 5 or n_v < 5:
            return 0.0

        p_c = control.get("true_positives", 0) / n_c
        p_v = variant.get("true_positives", 0) / n_v
        pooled = (control.get("true_positives", 0) + variant.get("true_positives", 0)) / (n_c + n_v)
        if pooled == 0 or pooled == 1:
            return 0.0

        se = math.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_v))
        if se == 0:
            return 0.0
        z = abs(p_c - p_v) / se

        # Approximate two-tailed p-value via logistic approximation
        try:
            p_value = 2.0 / (1.0 + math.exp(0.07056 * z**3 + 1.5976 * z))
        except OverflowError:
            p_value = 0.0
        return round(1.0 - p_value, 4)

    def _score(self, metrics: dict[str, Any]) -> float:
        """Combined score: 40% precision + 60% detection rate."""
        total = metrics.get("total_checks", 0)
        detections = metrics.get("detections", 0)
        tp = metrics.get("true_positives", 0)
        precision = tp / detections if detections else 0.0
        det_rate = detections / total if total else 0.0
        return precision * 0.4 + det_rate * 0.6


# =============================================================================
# Production Learning Engine
# =============================================================================


class ProductionLearningEngine:
    """Orchestrates the full production learning cycle."""

    def __init__(self) -> None:
        self.collector = IncidentCollector()
        self.tuner = ThresholdTuner()
        self.extractor = PatternExtractor()
        self.ab_manager = ABTestManager()
        self._applied_updates: list[RuleUpdate] = []
        self._active_rules: dict[str, dict[str, Any]] = {}

    def run_learning_cycle(
        self, since: datetime | None = None
    ) -> LearningReport:
        """Execute a full learning cycle: collect, analyze, recommend, report."""
        cycle_start = datetime.utcnow()
        if since is None:
            since = cycle_start - timedelta(days=30)
        logger.info("Starting learning cycle", since=since.isoformat())

        # 1. Gather and correlate incidents
        incidents = self.collector.get_incidents(since=since)
        correlations: list[IncidentCorrelation] = []
        for incident in incidents:
            corr = self.collector.correlate_with_commits(incident)
            if corr:
                correlations.append(corr)

        # 2. Extract patterns from undetected incidents
        undetected = self.collector.get_undetected_incidents()
        patterns = self.extractor.extract_patterns(undetected)

        # 3. Tune thresholds for correlated rules
        rule_ids = {c.detection_rule for c in correlations if c.detection_rule}
        thresholds: list[DetectionThreshold] = []
        ver_results = list(self.collector._verification_results.values())
        for rid in rule_ids:
            thresholds.append(
                self.tuner.analyze_threshold(rid, incidents, ver_results)
            )

        # 4. Collect all recommended updates
        updates = self.tuner.recommend_adjustments(thresholds)
        updates.extend(self._create_pattern_rules(patterns))

        # 5. Evaluate running A/B tests
        completed_tests = 0
        for test in list(self.ab_manager._tests.values()):
            if test.status == ABTestStatus.RUNNING:
                result = self.ab_manager.evaluate_test(test.id)
                if result.status == ABTestStatus.COMPLETED:
                    completed_tests += 1

        # 6. Build report
        detected_count = sum(1 for c in correlations if c.was_detected)
        detection_rate = detected_count / len(correlations) if correlations else 0.0
        fp_reduction = 0.0
        if thresholds:
            fp_reduction = sum(max(0, 0.3 - t.false_positive_rate) for t in thresholds) / len(thresholds)

        recommendations = self._generate_recommendations(
            incidents, undetected, patterns, updates
        )
        cycle_end = datetime.utcnow()

        report = LearningReport(
            period_start=since,
            period_end=cycle_end,
            incidents_analyzed=len(incidents),
            patterns_learned=len(patterns),
            rules_updated=len(updates),
            false_positive_reduction=round(fp_reduction, 4),
            detection_improvement=round(detection_rate, 4),
            ab_tests_completed=completed_tests,
            recommendations=recommendations,
        )
        logger.info(
            "Learning cycle completed",
            incidents=report.incidents_analyzed,
            patterns=report.patterns_learned,
            duration_sec=(cycle_end - cycle_start).total_seconds(),
        )
        return report

    def get_recommended_updates(self) -> list[RuleUpdate]:
        """Return all pending rule update recommendations."""
        updates: list[RuleUpdate] = []
        thresholds = list(self.tuner._thresholds.values())
        if thresholds:
            updates.extend(self.tuner.recommend_adjustments(thresholds))
        updates.extend(self._create_pattern_rules(list(self.extractor._patterns.values())))
        return updates

    def apply_updates(self, updates: list[RuleUpdate]) -> int:
        """Apply a list of rule updates. Returns the number successfully applied."""
        applied = 0
        for update in updates:
            try:
                if update.action == RuleUpdateAction.CREATE:
                    self._active_rules[update.rule_id] = update.after_state
                elif update.action == RuleUpdateAction.MODIFY:
                    self._active_rules.setdefault(update.rule_id, {}).update(update.after_state)
                elif update.action == RuleUpdateAction.TUNE_THRESHOLD:
                    rule = self._active_rules.setdefault(update.rule_id, {})
                    rule["threshold"] = update.after_state.get("threshold")
                elif update.action == RuleUpdateAction.DISABLE:
                    if update.rule_id in self._active_rules:
                        self._active_rules[update.rule_id]["enabled"] = False
                elif update.action == RuleUpdateAction.DELETE:
                    self._active_rules.pop(update.rule_id, None)
                self._applied_updates.append(update)
                applied += 1
                logger.info("Rule update applied", rule_id=update.rule_id, action=update.action.value)
            except Exception as exc:
                logger.error("Failed to apply update", rule_id=update.rule_id, error=str(exc))
        return applied

    def get_learning_report(self, period_days: int = 30) -> LearningReport:
        """Generate a summary report for the given period."""
        now = datetime.utcnow()
        start = now - timedelta(days=period_days)
        incidents = self.collector.get_incidents(since=start)
        patterns = [p for p in self.extractor._patterns.values() if p.created_at >= start]
        completed_tests = sum(
            1 for t in self.ab_manager._tests.values()
            if t.status == ABTestStatus.COMPLETED and t.end_date and t.end_date >= start
        )

        fp_reduction = 0.0
        tuned = [t for t in self.tuner._thresholds.values() if t.last_tuned and t.last_tuned >= start]
        if tuned:
            fp_reduction = sum(max(0, t.false_positive_rate - 0.1) for t in tuned) / len(tuned)

        recommendations = self._generate_recommendations(
            incidents, self.collector.get_undetected_incidents(), patterns, self._applied_updates
        )
        return LearningReport(
            period_start=start,
            period_end=now,
            incidents_analyzed=len(incidents),
            patterns_learned=len(patterns),
            rules_updated=len(self._applied_updates),
            false_positive_reduction=round(fp_reduction, 4),
            detection_improvement=round(min(0.5, len(patterns) * 0.05), 4),
            ab_tests_completed=completed_tests,
            recommendations=recommendations,
        )

    def _create_pattern_rules(self, patterns: list[LearnedPattern]) -> list[RuleUpdate]:
        """Create rule update proposals from learned patterns."""
        updates: list[RuleUpdate] = []
        for p in patterns:
            if p.confidence < 0.3:
                continue
            rule_id = f"learned_{p.id}"
            if rule_id in self._active_rules:
                continue
            updates.append(RuleUpdate(
                rule_id=rule_id,
                action=RuleUpdateAction.CREATE,
                before_state={},
                after_state={
                    "pattern_type": p.pattern_type,
                    "code_pattern": p.code_pattern,
                    "detection_rule": p.detection_rule,
                    "confidence": p.confidence,
                    "source": "production_learning",
                },
                justification=(
                    f"Learned from {len(p.source_incidents)} incidents "
                    f"with {p.confidence:.0%} confidence."
                ),
                confidence=p.confidence,
            ))
        return updates

    def _generate_recommendations(
        self,
        incidents: list[ProductionIncident],
        undetected: list[ProductionIncident],
        patterns: list[LearnedPattern],
        updates: list[RuleUpdate],
    ) -> list[str]:
        """Generate actionable recommendations from analysis results."""
        recs: list[str] = []
        if not incidents:
            recs.append("No incidents recorded. Ensure production monitoring is configured.")
            return recs

        undetected_rate = len(undetected) / len(incidents) if incidents else 0
        if undetected_rate > 0.5:
            recs.append(
                f"{undetected_rate:.0%} of incidents were not detected pre-deployment. "
                f"Consider adding {len(patterns)} new detection rules."
            )

        critical = sum(1 for i in incidents if i.severity == IncidentSeverity.CRITICAL)
        if critical > 0:
            recs.append(f"{critical} critical incidents. Prioritize these failure modes.")

        tune_count = sum(1 for u in updates if u.action == RuleUpdateAction.TUNE_THRESHOLD)
        if tune_count:
            recs.append(f"{tune_count} thresholds need adjustment based on production data.")

        high_conf = [p for p in patterns if p.confidence >= 0.7]
        if high_conf:
            recs.append(f"{len(high_conf)} high-confidence patterns ready for rule creation.")

        resolved = [i for i in incidents if i.resolved and i.resolution_time_hours]
        if resolved:
            avg_h = sum(i.resolution_time_hours for i in resolved) / len(resolved)  # type: ignore[arg-type]
            if avg_h > 24:
                recs.append(f"Avg resolution time is {avg_h:.1f}h. Faster detection could help.")

        return recs


# =============================================================================
# Utilities
# =============================================================================


def _common_prefix(paths: list[str]) -> str:
    """Find the longest common path prefix among a list of file paths."""
    if not paths:
        return ""
    if len(paths) == 1:
        parts = paths[0].rsplit("/", 1)
        return parts[0] if len(parts) > 1 else ""
    split = [p.split("/") for p in paths]
    prefix: list[str] = []
    for segments in zip(*split):
        if len(set(segments)) == 1:
            prefix.append(segments[0])
        else:
            break
    return "/".join(prefix)
