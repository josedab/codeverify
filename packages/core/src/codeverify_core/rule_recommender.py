"""Rule Recommendation Engine.

Analyzes finding history to recommend additional rules, identify patterns
of missed bugs, and suggest severity adjustments.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuleRecommendation:
    """A recommendation to enable or adjust a rule."""

    rule_id: str
    action: str  # "enable", "increase_severity", "decrease_severity", "disable"
    reason: str
    confidence: float = 0.0  # 0-1
    evidence_count: int = 0


@dataclass
class PatternInsight:
    """Insight about a recurring pattern in findings."""

    pattern: str
    frequency: int
    affected_files: list[str] = field(default_factory=list)
    suggestion: str = ""


# =============================================================================
# Built-in rule catalog (rules available but not necessarily enabled)
# =============================================================================

AVAILABLE_RULES: dict[str, dict[str, Any]] = {
    "null_safety": {
        "category": "safety",
        "default_severity": "high",
        "languages": ["python", "typescript", "go", "java"],
    },
    "array_bounds": {
        "category": "safety",
        "default_severity": "high",
        "languages": ["python", "typescript", "go", "java"],
    },
    "integer_overflow": {
        "category": "safety",
        "default_severity": "medium",
        "languages": ["go", "java", "typescript"],
    },
    "division_by_zero": {
        "category": "safety",
        "default_severity": "high",
        "languages": ["python", "typescript", "go", "java"],
    },
    "type_safety": {
        "category": "quality",
        "default_severity": "medium",
        "languages": ["python", "typescript"],
    },
    "sql_injection": {
        "category": "security",
        "default_severity": "critical",
        "languages": ["python", "typescript", "java"],
    },
    "xss": {
        "category": "security",
        "default_severity": "high",
        "languages": ["typescript", "java"],
    },
    "command_injection": {
        "category": "security",
        "default_severity": "critical",
        "languages": ["python", "typescript"],
    },
    "path_traversal": {
        "category": "security",
        "default_severity": "high",
        "languages": ["python", "typescript", "go", "java"],
    },
    "hardcoded_secrets": {
        "category": "security",
        "default_severity": "critical",
        "languages": ["python", "typescript", "go", "java"],
    },
    "error_handling": {
        "category": "quality",
        "default_severity": "medium",
        "languages": ["python", "typescript", "go", "java"],
    },
    "concurrency": {"category": "safety", "default_severity": "high", "languages": ["go", "java"]},
    "resource_leak": {
        "category": "quality",
        "default_severity": "medium",
        "languages": ["python", "java", "go"],
    },
    "deprecated_api": {
        "category": "quality",
        "default_severity": "low",
        "languages": ["python", "typescript", "java"],
    },
}


class RuleRecommender:
    """Recommends rules based on finding history and project characteristics.

    Usage:
        recommender = RuleRecommender()
        recommendations = recommender.analyze(
            findings_history=[...],
            enabled_rules=["null_safety", "array_bounds"],
            languages=["python", "typescript"],
        )
    """

    def __init__(self, rule_catalog: dict[str, dict[str, Any]] | None = None):
        self._catalog = rule_catalog or AVAILABLE_RULES

    def analyze(
        self,
        findings_history: list[dict[str, Any]],
        enabled_rules: list[str],
        languages: list[str],
    ) -> list[RuleRecommendation]:
        """Analyze finding history and produce recommendations."""
        recommendations: list[RuleRecommendation] = []

        recommendations.extend(self._recommend_missing_rules(enabled_rules, languages))
        recommendations.extend(self._recommend_severity_adjustments(findings_history))
        recommendations.extend(self._recommend_from_patterns(findings_history, enabled_rules))

        # Sort by confidence
        recommendations.sort(key=lambda r: r.confidence, reverse=True)
        return recommendations

    def _recommend_missing_rules(
        self,
        enabled: list[str],
        languages: list[str],
    ) -> list[RuleRecommendation]:
        """Recommend rules that are available but not enabled."""
        recs: list[RuleRecommendation] = []
        enabled_set = set(enabled)

        for rule_id, info in self._catalog.items():
            if rule_id in enabled_set:
                continue

            # Check if rule applies to any of the project's languages
            rule_langs = set(info.get("languages", []))
            if not rule_langs.intersection(languages):
                continue

            # Security rules get higher confidence
            category = info.get("category", "")
            confidence = 0.9 if category == "security" else 0.6

            recs.append(
                RuleRecommendation(
                    rule_id=rule_id,
                    action="enable",
                    reason=f"Available {category} rule for {', '.join(rule_langs & set(languages))}",
                    confidence=confidence,
                )
            )

        return recs

    def _recommend_severity_adjustments(
        self,
        findings: list[dict[str, Any]],
    ) -> list[RuleRecommendation]:
        """Recommend severity changes based on finding frequency."""
        recs: list[RuleRecommendation] = []
        rule_counts: Counter[str] = Counter()
        rule_severities: dict[str, list[str]] = defaultdict(list)

        for f in findings:
            rid = f.get("rule_id", "")
            rule_counts[rid] += 1
            rule_severities[rid].append(f.get("severity", "medium"))

        for rid, count in rule_counts.items():
            if count >= 10:
                common_sev = Counter(rule_severities[rid]).most_common(1)[0][0]
                if common_sev in ("low", "medium"):
                    recs.append(
                        RuleRecommendation(
                            rule_id=rid,
                            action="increase_severity",
                            reason=f"Rule triggered {count} times — consider raising severity from {common_sev}",
                            confidence=min(0.9, count / 20),
                            evidence_count=count,
                        )
                    )

        return recs

    def _recommend_from_patterns(
        self,
        findings: list[dict[str, Any]],
        enabled: list[str],
    ) -> list[RuleRecommendation]:
        """Detect finding patterns that suggest enabling related rules."""
        recs: list[RuleRecommendation] = []
        categories_seen: set[str] = set()

        for f in findings:
            cat = f.get("category", "")
            if cat:
                categories_seen.add(cat)

        # If we see security findings, recommend all security rules
        if "security" in categories_seen:
            for rid, info in self._catalog.items():
                if info.get("category") == "security" and rid not in enabled:
                    recs.append(
                        RuleRecommendation(
                            rule_id=rid,
                            action="enable",
                            reason="Security findings detected — enable all security rules",
                            confidence=0.85,
                        )
                    )

        return recs

    def get_pattern_insights(
        self,
        findings: list[dict[str, Any]],
    ) -> list[PatternInsight]:
        """Extract insights about finding patterns."""
        insights: list[PatternInsight] = []

        # Files with most findings
        file_counts: Counter[str] = Counter()
        file_rules: dict[str, set[str]] = defaultdict(set)
        for f in findings:
            fp = f.get("file_path", "")
            file_counts[fp] += 1
            file_rules[fp].add(f.get("rule_id", ""))

        hot_files = file_counts.most_common(5)
        for fp, count in hot_files:
            if count >= 3:
                insights.append(
                    PatternInsight(
                        pattern="hot_file",
                        frequency=count,
                        affected_files=[fp],
                        suggestion=f"File '{fp}' has {count} findings across {len(file_rules[fp])} rules — consider refactoring",
                    )
                )

        # Most common rule violations
        rule_counts: Counter[str] = Counter()
        for f in findings:
            rule_counts[f.get("rule_id", "")] += 1

        for rid, count in rule_counts.most_common(3):
            if count >= 5:
                insights.append(
                    PatternInsight(
                        pattern="frequent_violation",
                        frequency=count,
                        suggestion=f"Rule '{rid}' triggered {count} times — may indicate a systemic issue",
                    )
                )

        return insights
