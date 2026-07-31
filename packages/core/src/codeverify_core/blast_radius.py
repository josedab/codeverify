"""Blast Radius Analysis — Visualize and report cross-repository change impact.

Extends the cross-repo impact analysis with blast radius scoring,
team notification routing, and structured impact reports.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ChangeType(str, Enum):
    """Type of code change."""

    BREAKING = "breaking"
    DEPRECATION = "deprecation"
    BEHAVIOR_CHANGE = "behavior_change"
    ADDITIVE = "additive"
    INTERNAL = "internal"


class ImpactSeverity(str, Enum):
    """Severity of the impact on downstream consumers."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class AffectedService:
    """A service affected by a code change."""

    name: str
    repository: str
    dependency_path: list[str]  # chain from source to this service
    impact_severity: ImpactSeverity
    affected_files: list[str] = field(default_factory=list)
    team_owner: str = ""
    notification_contacts: list[str] = field(default_factory=list)

    @property
    def hop_count(self) -> int:
        return len(self.dependency_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "repository": self.repository,
            "dependency_path": self.dependency_path,
            "impact_severity": self.impact_severity.value,
            "hop_count": self.hop_count,
            "team_owner": self.team_owner,
            "affected_files_count": len(self.affected_files),
        }


@dataclass
class BlastRadiusReport:
    """Complete blast radius analysis for a set of changes."""

    source_repository: str
    source_files: list[str]
    change_type: ChangeType
    affected_services: list[AffectedService] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def total_affected(self) -> int:
        return len(self.affected_services)

    @property
    def critical_count(self) -> int:
        return sum(
            1 for s in self.affected_services if s.impact_severity == ImpactSeverity.CRITICAL
        )

    @property
    def radius_score(self) -> float:
        """0-100 score indicating blast radius magnitude."""
        if not self.affected_services:
            return 0.0
        weights = {
            ImpactSeverity.CRITICAL: 25,
            ImpactSeverity.HIGH: 15,
            ImpactSeverity.MEDIUM: 5,
            ImpactSeverity.LOW: 1,
            ImpactSeverity.NONE: 0,
        }
        total = sum(
            weights.get(s.impact_severity, 0) * (1.0 / max(s.hop_count, 1))
            for s in self.affected_services
        )
        return min(100.0, total)

    @property
    def affected_teams(self) -> list[str]:
        """Unique teams that need to be notified."""
        return list({s.team_owner for s in self.affected_services if s.team_owner})

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_repository": self.source_repository,
            "source_files": self.source_files,
            "change_type": self.change_type.value,
            "summary": {
                "total_affected": self.total_affected,
                "critical": self.critical_count,
                "radius_score": round(self.radius_score, 1),
                "affected_teams": self.affected_teams,
                "max_hop_count": max((s.hop_count for s in self.affected_services), default=0),
            },
            "affected_services": [s.to_dict() for s in self.affected_services],
            "generated_at": self.generated_at.isoformat(),
        }

    def to_markdown(self) -> str:
        """Generate a markdown summary of the blast radius."""
        lines = [
            "# 💥 Blast Radius Report",
            "",
            f"**Source:** `{self.source_repository}`",
            f"**Change Type:** {self.change_type.value}",
            f"**Radius Score:** {self.radius_score:.1f}/100",
            f"**Affected Services:** {self.total_affected}",
            f"**Affected Teams:** {', '.join(self.affected_teams) or 'None'}",
            "",
        ]

        if self.critical_count:
            lines.append(f"⚠️ **{self.critical_count} CRITICAL impact(s) detected**")
            lines.append("")

        if self.affected_services:
            lines.append("## Affected Services")
            lines.append("")
            lines.append("| Service | Severity | Hops | Team |")
            lines.append("|---------|----------|------|------|")
            for svc in sorted(
                self.affected_services,
                key=lambda s: list(ImpactSeverity).index(s.impact_severity),
            ):
                lines.append(
                    f"| {svc.name} | {svc.impact_severity.value} | {svc.hop_count} | {svc.team_owner} |"
                )

        return "\n".join(lines)


@dataclass
class TeamNotification:
    """Notification to send to an affected team."""

    team: str
    contacts: list[str]
    severity: ImpactSeverity
    affected_services: list[str]
    source_repository: str
    change_description: str
    report_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "contacts": self.contacts,
            "severity": self.severity.value,
            "affected_services": self.affected_services,
            "source_repository": self.source_repository,
        }


class DependencyGraph:
    """In-memory dependency graph for blast radius computation.

    Stores service-to-service dependencies and team ownership.
    """

    def __init__(self) -> None:
        self._edges: dict[str, set[str]] = {}  # service -> set of dependents
        self._reverse: dict[str, set[str]] = {}  # service -> set of dependencies
        self._teams: dict[str, str] = {}  # service -> team owner
        self._contacts: dict[str, list[str]] = {}  # team -> contacts

    def add_dependency(self, source: str, depends_on: str) -> None:
        """Record that `source` depends on `depends_on`."""
        self._edges.setdefault(depends_on, set()).add(source)
        self._reverse.setdefault(source, set()).add(depends_on)

    def set_team_owner(self, service: str, team: str) -> None:
        self._teams[service] = team

    def set_team_contacts(self, team: str, contacts: list[str]) -> None:
        self._contacts[team] = contacts

    def get_dependents(self, service: str, max_depth: int = 5) -> list[tuple[str, list[str]]]:
        """Get all transitive dependents of a service with dependency paths.

        Returns list of (service_name, dependency_path) tuples.
        """
        result: list[tuple[str, list[str]]] = []
        visited: set[str] = set()
        queue: list[tuple[str, list[str]]] = [(service, [service])]

        while queue:
            current, path = queue.pop(0)
            if len(path) > max_depth + 1:
                continue
            for dependent in self._edges.get(current, set()):
                if dependent not in visited:
                    visited.add(dependent)
                    new_path = path + [dependent]
                    result.append((dependent, new_path))
                    queue.append((dependent, new_path))

        return result

    def get_team(self, service: str) -> str:
        return self._teams.get(service, "")

    def get_contacts(self, team: str) -> list[str]:
        return self._contacts.get(team, [])


class BlastRadiusAnalyzer:
    """Analyzes the blast radius of code changes across services.

    Example:
        >>> graph = DependencyGraph()
        >>> graph.add_dependency("api-gateway", "auth-service")
        >>> graph.add_dependency("web-app", "api-gateway")
        >>> graph.set_team_owner("web-app", "frontend-team")
        >>> analyzer = BlastRadiusAnalyzer(graph)
        >>> report = analyzer.analyze(
        ...     source_repository="auth-service",
        ...     changed_files=["src/auth.py"],
        ...     change_type=ChangeType.BREAKING,
        ... )
    """

    def __init__(self, graph: DependencyGraph) -> None:
        self._graph = graph

    def analyze(
        self,
        source_repository: str,
        changed_files: list[str],
        change_type: ChangeType = ChangeType.BEHAVIOR_CHANGE,
    ) -> BlastRadiusReport:
        """Analyze blast radius for a set of changes."""
        dependents = self._graph.get_dependents(source_repository)

        affected: list[AffectedService] = []
        for service_name, dep_path in dependents:
            severity = self._compute_severity(change_type, len(dep_path) - 1)
            team = self._graph.get_team(service_name)
            contacts = self._graph.get_contacts(team) if team else []

            affected.append(
                AffectedService(
                    name=service_name,
                    repository=service_name,
                    dependency_path=dep_path,
                    impact_severity=severity,
                    team_owner=team,
                    notification_contacts=contacts,
                )
            )

        report = BlastRadiusReport(
            source_repository=source_repository,
            source_files=changed_files,
            change_type=change_type,
            affected_services=affected,
        )

        logger.info(
            "Blast radius analysis complete",
            source=source_repository,
            affected=report.total_affected,
            critical=report.critical_count,
            radius_score=report.radius_score,
        )
        return report

    def generate_notifications(self, report: BlastRadiusReport) -> list[TeamNotification]:
        """Generate team notifications from a blast radius report."""
        team_map: dict[str, list[AffectedService]] = {}
        for svc in report.affected_services:
            if svc.team_owner:
                team_map.setdefault(svc.team_owner, []).append(svc)

        notifications = []
        for team, services in team_map.items():
            # Use highest severity among affected services
            worst_severity = min(
                services,
                key=lambda s: list(ImpactSeverity).index(s.impact_severity),
            ).impact_severity

            all_contacts: list[str] = []
            for svc in services:
                all_contacts.extend(svc.notification_contacts)
            contacts = list(set(all_contacts))

            notifications.append(
                TeamNotification(
                    team=team,
                    contacts=contacts,
                    severity=worst_severity,
                    affected_services=[s.name for s in services],
                    source_repository=report.source_repository,
                    change_description=f"{report.change_type.value} change in {', '.join(report.source_files[:3])}",
                )
            )

        return notifications

    def _compute_severity(self, change_type: ChangeType, hop_distance: int) -> ImpactSeverity:
        """Compute impact severity based on change type and distance."""
        if change_type == ChangeType.BREAKING:
            if hop_distance <= 1:
                return ImpactSeverity.CRITICAL
            elif hop_distance <= 2:
                return ImpactSeverity.HIGH
            return ImpactSeverity.MEDIUM
        elif change_type == ChangeType.DEPRECATION:
            if hop_distance <= 1:
                return ImpactSeverity.HIGH
            return ImpactSeverity.MEDIUM
        elif change_type == ChangeType.BEHAVIOR_CHANGE:
            if hop_distance <= 1:
                return ImpactSeverity.MEDIUM
            return ImpactSeverity.LOW
        elif change_type == ChangeType.ADDITIVE:
            return ImpactSeverity.LOW
        return ImpactSeverity.NONE
