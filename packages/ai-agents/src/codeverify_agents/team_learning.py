"""Team Learning Mode - Aggregate findings to identify systemic patterns.

Analyzes findings across an organization to identify:
- Common bug patterns by team/repo
- Training gaps based on recurring issues
- Code health trends over time
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import structlog

from codeverify_core.models import Finding, FindingCategory, FindingSeverity

logger = structlog.get_logger()


class TrendDirection(str, Enum):
    """Direction of a trend."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"


@dataclass
class PatternOccurrence:
    """A single occurrence of a pattern."""
    finding_id: str
    file_path: str
    repository: str
    author: str
    team: str
    timestamp: datetime
    category: FindingCategory
    severity: FindingSeverity
    message: str


@dataclass
class SystemicPattern:
    """A systemic pattern identified across multiple occurrences."""
    pattern_id: str
    name: str
    description: str
    category: FindingCategory
    occurrences: list[PatternOccurrence]
    affected_teams: set[str]
    affected_repos: set[str]
    first_seen: datetime
    last_seen: datetime
    frequency_per_week: float
    severity_distribution: dict[str, int]
    recommended_action: str


@dataclass
class TeamMetrics:
    """Metrics for a team."""
    team_name: str
    total_findings: int
    findings_by_category: dict[str, int]
    findings_by_severity: dict[str, int]
    avg_findings_per_pr: float
    common_patterns: list[str]
    trend: TrendDirection
    improvement_areas: list[str]


@dataclass
class TrainingRecommendation:
    """A training recommendation based on patterns."""
    title: str
    description: str
    target_teams: list[str]
    target_skills: list[str]
    priority: int  # 1-10
    estimated_impact: str
    supporting_data: dict[str, Any]


@dataclass
class OrgHealthReport:
    """Organization-wide code health report."""
    report_date: datetime
    total_findings: int
    total_prs_analyzed: int
    findings_by_category: dict[str, int]
    findings_by_severity: dict[str, int]
    team_metrics: list[TeamMetrics]
    systemic_patterns: list[SystemicPattern]
    training_recommendations: list[TrainingRecommendation]
    trend_vs_last_period: TrendDirection
    top_improving_teams: list[str]
    teams_needing_attention: list[str]


class PatternDetector:
    """Detects patterns in findings."""

    # Common code issue patterns
    KNOWN_PATTERNS = {
        "null_reference": {
            "keywords": ["null", "undefined", "none", "nil", "nullpointer"],
            "name": "Null Reference Issues",
            "recommendation": "Implement null safety practices and optional chaining",
        },
        "error_handling": {
            "keywords": ["exception", "error", "try", "catch", "throw", "unhandled"],
            "name": "Error Handling Gaps",
            "recommendation": "Establish error handling guidelines and review patterns",
        },
        "input_validation": {
            "keywords": ["validation", "sanitize", "input", "injection", "escape"],
            "name": "Input Validation Issues",
            "recommendation": "Implement input validation framework and security training",
        },
        "resource_leak": {
            "keywords": ["leak", "resource", "close", "dispose", "cleanup", "memory"],
            "name": "Resource Leaks",
            "recommendation": "Use RAII patterns and automatic resource management",
        },
        "concurrency": {
            "keywords": ["race", "deadlock", "lock", "thread", "async", "concurrent"],
            "name": "Concurrency Issues",
            "recommendation": "Review concurrency patterns and use thread-safe constructs",
        },
        "boundary": {
            "keywords": ["overflow", "underflow", "bounds", "index", "range", "array"],
            "name": "Boundary Violations",
            "recommendation": "Use bounded types and explicit range checking",
        },
        "security": {
            "keywords": ["sql", "xss", "csrf", "auth", "permission", "secret", "credential"],
            "name": "Security Vulnerabilities",
            "recommendation": "Security training and automated security scanning",
        },
        "performance": {
            "keywords": ["n+1", "cache", "slow", "timeout", "complexity", "optimize"],
            "name": "Performance Issues",
            "recommendation": "Performance profiling and optimization guidelines",
        },
    }

    def detect_pattern(self, finding: Finding) -> str | None:
        """Detect which pattern a finding matches."""
        text = f"{finding.message} {finding.description or ''}".lower()
        
        for pattern_id, pattern_info in self.KNOWN_PATTERNS.items():
            for keyword in pattern_info["keywords"]:
                if keyword in text:
                    return pattern_id
        
        return None

    def get_pattern_info(self, pattern_id: str) -> dict[str, str] | None:
        """Get info about a pattern."""
        return self.KNOWN_PATTERNS.get(pattern_id)


class TrendAnalyzer:
    """Analyzes trends over time."""

    def analyze_trend(
        self,
        current_count: int,
        previous_count: int,
        threshold_pct: float = 10.0,
    ) -> TrendDirection:
        """Determine trend direction."""
        if previous_count == 0:
            return TrendDirection.STABLE
        
        change_pct = ((current_count - previous_count) / previous_count) * 100
        
        if change_pct < -threshold_pct:
            return TrendDirection.IMPROVING
        elif change_pct > threshold_pct:
            return TrendDirection.DEGRADING
        else:
            return TrendDirection.STABLE

    def calculate_weekly_frequency(
        self,
        occurrences: list[PatternOccurrence],
        weeks: int = 4,
    ) -> float:
        """Calculate average weekly frequency."""
        if not occurrences:
            return 0.0
        
        cutoff = datetime.utcnow() - timedelta(weeks=weeks)
        recent = [o for o in occurrences if o.timestamp >= cutoff]
        
        return len(recent) / weeks


class FindingsAggregator:
    """Aggregates findings for analysis."""

    def __init__(self) -> None:
        """Initialize aggregator."""
        self._occurrences: list[PatternOccurrence] = []
        self._pattern_detector = PatternDetector()
        self._trend_analyzer = TrendAnalyzer()
        self._team_mapping: dict[str, str] = {}  # author -> team

    def set_team_mapping(self, mapping: dict[str, str]) -> None:
        """Set author to team mapping."""
        self._team_mapping = mapping

    def add_finding(
        self,
        finding: Finding,
        repository: str,
        author: str,
        file_path: str,
    ) -> None:
        """Add a finding for aggregation."""
        team = self._team_mapping.get(author, "unknown")
        
        occurrence = PatternOccurrence(
            finding_id=finding.id,
            file_path=file_path,
            repository=repository,
            author=author,
            team=team,
            timestamp=datetime.utcnow(),
            category=finding.category,
            severity=finding.severity,
            message=finding.message,
        )
        
        self._occurrences.append(occurrence)
        
        # Prune old occurrences (keep 90 days)
        cutoff = datetime.utcnow() - timedelta(days=90)
        self._occurrences = [o for o in self._occurrences if o.timestamp >= cutoff]

    def add_findings_batch(
        self,
        findings: list[Finding],
        repository: str,
        author: str,
        file_paths: dict[str, str],  # finding_id -> file_path
    ) -> None:
        """Add multiple findings."""
        for finding in findings:
            file_path = file_paths.get(finding.id, "unknown")
            self.add_finding(finding, repository, author, file_path)


class TeamLearningAgent:
    """
    Analyzes findings across an organization to identify systemic patterns.
    
    Identifies:
    - Common bug patterns by team/repository
    - Training gaps based on recurring issues
    - Code health trends over time
    """

    def __init__(self) -> None:
        """Initialize the agent."""
        self.aggregator = FindingsAggregator()
        self.pattern_detector = PatternDetector()
        self.trend_analyzer = TrendAnalyzer()

    def configure_teams(self, team_mapping: dict[str, str]) -> None:
        """Configure author to team mapping."""
        self.aggregator.set_team_mapping(team_mapping)

    def record_findings(
        self,
        findings: list[Finding],
        repository: str,
        author: str,
        file_paths: dict[str, str] | None = None,
    ) -> None:
        """Record findings for analysis."""
        file_paths = file_paths or {}
        self.aggregator.add_findings_batch(findings, repository, author, file_paths)

    def identify_systemic_patterns(
        self,
        min_occurrences: int = 5,
    ) -> list[SystemicPattern]:
        """Identify systemic patterns across the organization."""
        occurrences = self.aggregator._occurrences
        
        # Group by detected pattern
        pattern_groups: dict[str, list[PatternOccurrence]] = defaultdict(list)
        
        for occ in occurrences:
            # Create a pseudo-finding for pattern detection
            finding = Finding(
                id=occ.finding_id,
                message=occ.message,
                category=occ.category,
                severity=occ.severity,
                file_path=occ.file_path,
                line_number=0,
            )
            
            pattern_id = self.pattern_detector.detect_pattern(finding)
            if pattern_id:
                pattern_groups[pattern_id].append(occ)
        
        # Convert to SystemicPattern objects
        patterns = []
        for pattern_id, group in pattern_groups.items():
            if len(group) < min_occurrences:
                continue
            
            pattern_info = self.pattern_detector.get_pattern_info(pattern_id)
            if not pattern_info:
                continue
            
            # Analyze the group
            affected_teams = {o.team for o in group}
            affected_repos = {o.repository for o in group}
            timestamps = [o.timestamp for o in group]
            
            severity_dist = Counter(o.severity.value for o in group)
            
            pattern = SystemicPattern(
                pattern_id=pattern_id,
                name=pattern_info["name"],
                description=f"Detected {len(group)} occurrences across {len(affected_teams)} teams",
                category=group[0].category,
                occurrences=group,
                affected_teams=affected_teams,
                affected_repos=affected_repos,
                first_seen=min(timestamps),
                last_seen=max(timestamps),
                frequency_per_week=self.trend_analyzer.calculate_weekly_frequency(group),
                severity_distribution=dict(severity_dist),
                recommended_action=pattern_info["recommendation"],
            )
            
            patterns.append(pattern)
        
        # Sort by frequency
        patterns.sort(key=lambda p: len(p.occurrences), reverse=True)
        
        logger.info(
            "Systemic patterns identified",
            pattern_count=len(patterns),
            total_occurrences=len(occurrences),
        )
        
        return patterns

    def analyze_team(self, team_name: str) -> TeamMetrics | None:
        """Analyze metrics for a specific team."""
        occurrences = self.aggregator._occurrences
        team_occurrences = [o for o in occurrences if o.team == team_name]
        
        if not team_occurrences:
            return None
        
        # Calculate metrics
        total = len(team_occurrences)
        
        by_category = Counter(o.category.value for o in team_occurrences)
        by_severity = Counter(o.severity.value for o in team_occurrences)
        
        # Find common patterns
        pattern_counts: Counter[str] = Counter()
        for occ in team_occurrences:
            finding = Finding(
                id=occ.finding_id,
                message=occ.message,
                category=occ.category,
                severity=occ.severity,
                file_path=occ.file_path,
                line_number=0,
            )
            pattern = self.pattern_detector.detect_pattern(finding)
            if pattern:
                pattern_counts[pattern] += 1
        
        common_patterns = [p for p, _ in pattern_counts.most_common(3)]
        
        # Determine improvement areas
        improvement_areas = []
        for pattern_id, count in pattern_counts.most_common(3):
            pattern_info = self.pattern_detector.get_pattern_info(pattern_id)
            if pattern_info:
                improvement_areas.append(pattern_info["name"])
        
        # Calculate trend (compare last 2 weeks vs previous 2 weeks)
        now = datetime.utcnow()
        week_ago = now - timedelta(weeks=1)
        two_weeks_ago = now - timedelta(weeks=2)
        
        recent = [o for o in team_occurrences if o.timestamp >= week_ago]
        previous = [o for o in team_occurrences if two_weeks_ago <= o.timestamp < week_ago]
        
        trend = self.trend_analyzer.analyze_trend(len(recent), len(previous))
        
        return TeamMetrics(
            team_name=team_name,
            total_findings=total,
            findings_by_category=dict(by_category),
            findings_by_severity=dict(by_severity),
            avg_findings_per_pr=total / max(len(set(o.finding_id for o in team_occurrences)), 1),
            common_patterns=common_patterns,
            trend=trend,
            improvement_areas=improvement_areas,
        )

    def generate_training_recommendations(
        self,
        patterns: list[SystemicPattern],
    ) -> list[TrainingRecommendation]:
        """Generate training recommendations based on patterns."""
        recommendations = []
        
        for pattern in patterns[:5]:  # Top 5 patterns
            # Determine priority based on severity and frequency
            severity_weight = {
                "critical": 10,
                "error": 7,
                "warning": 4,
                "info": 1,
            }
            
            avg_severity = sum(
                severity_weight.get(s, 1) * c
                for s, c in pattern.severity_distribution.items()
            ) / max(sum(pattern.severity_distribution.values()), 1)
            
            priority = min(int(avg_severity + pattern.frequency_per_week), 10)
            
            # Map patterns to skills
            skill_mapping = {
                "null_reference": ["Defensive Programming", "Null Safety"],
                "error_handling": ["Error Handling", "Exception Management"],
                "input_validation": ["Input Validation", "Security Basics"],
                "resource_leak": ["Resource Management", "Memory Safety"],
                "concurrency": ["Concurrent Programming", "Thread Safety"],
                "boundary": ["Safe Arithmetic", "Bounds Checking"],
                "security": ["Security Fundamentals", "OWASP Top 10"],
                "performance": ["Performance Optimization", "Algorithmic Thinking"],
            }
            
            skills = skill_mapping.get(pattern.pattern_id, ["Code Quality"])
            
            rec = TrainingRecommendation(
                title=f"Training: {pattern.name}",
                description=f"Address {pattern.name} issues affecting {len(pattern.affected_teams)} teams",
                target_teams=list(pattern.affected_teams),
                target_skills=skills,
                priority=priority,
                estimated_impact=f"Could reduce {pattern.name} by 50-70%",
                supporting_data={
                    "occurrences": len(pattern.occurrences),
                    "frequency_per_week": pattern.frequency_per_week,
                    "severity_distribution": pattern.severity_distribution,
                },
            )
            
            recommendations.append(rec)
        
        # Sort by priority
        recommendations.sort(key=lambda r: r.priority, reverse=True)
        
        return recommendations

    def generate_org_health_report(self) -> OrgHealthReport:
        """Generate comprehensive organization health report."""
        occurrences = self.aggregator._occurrences
        
        if not occurrences:
            return OrgHealthReport(
                report_date=datetime.utcnow(),
                total_findings=0,
                total_prs_analyzed=0,
                findings_by_category={},
                findings_by_severity={},
                team_metrics=[],
                systemic_patterns=[],
                training_recommendations=[],
                trend_vs_last_period=TrendDirection.STABLE,
                top_improving_teams=[],
                teams_needing_attention=[],
            )
        
        # Aggregate totals
        total = len(occurrences)
        by_category = Counter(o.category.value for o in occurrences)
        by_severity = Counter(o.severity.value for o in occurrences)
        
        # Unique PRs (approximation)
        unique_prs = len(set((o.repository, o.author, o.timestamp.date()) for o in occurrences))
        
        # Get all teams
        teams = list(set(o.team for o in occurrences))
        team_metrics = []
        improving_teams = []
        attention_teams = []
        
        for team in teams:
            metrics = self.analyze_team(team)
            if metrics:
                team_metrics.append(metrics)
                
                if metrics.trend == TrendDirection.IMPROVING:
                    improving_teams.append(team)
                elif metrics.trend == TrendDirection.DEGRADING:
                    attention_teams.append(team)
        
        # Sort team metrics by total findings
        team_metrics.sort(key=lambda m: m.total_findings, reverse=True)
        
        # Get patterns and recommendations
        patterns = self.identify_systemic_patterns()
        recommendations = self.generate_training_recommendations(patterns)
        
        # Overall trend
        now = datetime.utcnow()
        mid = now - timedelta(weeks=2)
        recent = len([o for o in occurrences if o.timestamp >= mid])
        previous = len([o for o in occurrences if o.timestamp < mid])
        overall_trend = self.trend_analyzer.analyze_trend(recent, previous)
        
        report = OrgHealthReport(
            report_date=datetime.utcnow(),
            total_findings=total,
            total_prs_analyzed=unique_prs,
            findings_by_category=dict(by_category),
            findings_by_severity=dict(by_severity),
            team_metrics=team_metrics,
            systemic_patterns=patterns,
            training_recommendations=recommendations,
            trend_vs_last_period=overall_trend,
            top_improving_teams=improving_teams[:5],
            teams_needing_attention=attention_teams[:5],
        )
        
        logger.info(
            "Organization health report generated",
            total_findings=total,
            pattern_count=len(patterns),
            recommendation_count=len(recommendations),
        )
        
        return report

    def export_report_markdown(self, report: OrgHealthReport) -> str:
        """Export report as markdown."""
        lines = [
            "# Organization Code Health Report",
            f"**Generated:** {report.report_date.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "## Summary",
            f"- **Total Findings:** {report.total_findings}",
            f"- **PRs Analyzed:** {report.total_prs_analyzed}",
            f"- **Overall Trend:** {report.trend_vs_last_period.value}",
            "",
            "## Findings by Severity",
        ]
        
        for severity, count in sorted(report.findings_by_severity.items()):
            lines.append(f"- {severity}: {count}")
        
        lines.extend(["", "## Findings by Category"])
        for category, count in sorted(report.findings_by_category.items()):
            lines.append(f"- {category}: {count}")
        
        lines.extend(["", "## Systemic Patterns", ""])
        for pattern in report.systemic_patterns[:5]:
            lines.append(f"### {pattern.name}")
            lines.append(f"- **Occurrences:** {len(pattern.occurrences)}")
            lines.append(f"- **Affected Teams:** {', '.join(pattern.affected_teams)}")
            lines.append(f"- **Recommendation:** {pattern.recommended_action}")
            lines.append("")
        
        lines.extend(["## Training Recommendations", ""])
        for rec in report.training_recommendations[:5]:
            lines.append(f"### {rec.title} (Priority: {rec.priority}/10)")
            lines.append(f"- **Teams:** {', '.join(rec.target_teams)}")
            lines.append(f"- **Skills:** {', '.join(rec.target_skills)}")
            lines.append(f"- **Impact:** {rec.estimated_impact}")
            lines.append("")
        
        if report.top_improving_teams:
            lines.append("## Top Improving Teams")
            for team in report.top_improving_teams:
                lines.append(f"- {team}")
            lines.append("")
        
        if report.teams_needing_attention:
            lines.append("## Teams Needing Attention")
            for team in report.teams_needing_attention:
                lines.append(f"- {team}")
            lines.append("")
        
        return "\n".join(lines)
