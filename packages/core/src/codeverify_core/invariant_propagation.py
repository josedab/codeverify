"""Multi-Repository Invariant Propagation.

Propagates verified invariants across dependent repositories,
maintaining a central registry with governance controls.

Features:
- Central invariant registry with scope (repo, org, global)
- Propagation engine across dependent repos
- Violation detection in downstream repos
- Governance dashboard data
- Conflict resolution for cross-repo invariants
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class InvariantScope(str, Enum):
    REPO = "repo"
    ORG = "org"
    GLOBAL = "global"


class PropagationStatus(str, Enum):
    PENDING = "pending"
    PROPAGATED = "propagated"
    VIOLATED = "violated"
    COMPLIANT = "compliant"
    SKIPPED = "skipped"


@dataclass
class RegisteredInvariant:
    """An invariant registered in the central registry."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    z3_assertion: str = ""
    scope: InvariantScope = InvariantScope.REPO
    source_repo: str = ""
    target_repos: list[str] = field(default_factory=list)
    check_type: str = ""
    is_active: bool = True
    verified_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    created_by: str = ""


@dataclass
class PropagationResult:
    """Result of propagating an invariant to a repo."""

    invariant_id: str = ""
    target_repo: str = ""
    status: PropagationStatus = PropagationStatus.PENDING
    violations: list[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class PropagationReport:
    """Report from propagating invariants across repos."""

    invariant_id: str = ""
    invariant_name: str = ""
    results: list[PropagationResult] = field(default_factory=list)
    total_repos: int = 0
    compliant_repos: int = 0
    violated_repos: int = 0
    compliance_rate: float = 0.0


@dataclass
class GovernanceSummary:
    """Governance dashboard summary."""

    total_invariants: int = 0
    active_invariants: int = 0
    repos_covered: int = 0
    org_compliance_rate: float = 0.0
    top_violations: list[dict[str, Any]] = field(default_factory=list)


class InvariantRegistry:
    """Central registry for cross-repo invariants."""

    def __init__(self) -> None:
        self._invariants: dict[str, RegisteredInvariant] = {}

    def register(
        self,
        name: str,
        description: str,
        z3_assertion: str,
        source_repo: str,
        scope: InvariantScope = InvariantScope.REPO,
        target_repos: list[str] | None = None,
        check_type: str = "",
        created_by: str = "",
    ) -> RegisteredInvariant:
        inv = RegisteredInvariant(
            name=name,
            description=description,
            z3_assertion=z3_assertion,
            scope=scope,
            source_repo=source_repo,
            target_repos=target_repos or [],
            check_type=check_type,
            created_by=created_by,
        )
        self._invariants[inv.id] = inv
        return inv

    def get(self, invariant_id: str) -> RegisteredInvariant | None:
        return self._invariants.get(invariant_id)

    def list_for_repo(self, repo: str) -> list[RegisteredInvariant]:
        return [
            inv
            for inv in self._invariants.values()
            if inv.is_active
            and (
                repo in inv.target_repos or inv.scope in (InvariantScope.ORG, InvariantScope.GLOBAL)
            )
        ]

    def list_all(self, active_only: bool = True) -> list[RegisteredInvariant]:
        invs = list(self._invariants.values())
        if active_only:
            invs = [i for i in invs if i.is_active]
        return invs

    def deactivate(self, invariant_id: str) -> bool:
        inv = self._invariants.get(invariant_id)
        if inv:
            inv.is_active = False
            return True
        return False


class PropagationEngine:
    """Propagates and checks invariants across repos."""

    def propagate(
        self, invariant: RegisteredInvariant, repo_code: dict[str, dict[str, str]]
    ) -> PropagationReport:
        """Check an invariant across multiple repos."""
        results: list[PropagationResult] = []
        compliant = 0
        violated = 0

        target_repos = invariant.target_repos
        if invariant.scope in (InvariantScope.ORG, InvariantScope.GLOBAL):
            target_repos = list(repo_code.keys())

        for repo in target_repos:
            code_files = repo_code.get(repo, {})
            if not code_files:
                results.append(
                    PropagationResult(
                        invariant_id=invariant.id,
                        target_repo=repo,
                        status=PropagationStatus.SKIPPED,
                    )
                )
                continue

            violations = self._check_invariant(invariant, code_files)
            if violations:
                results.append(
                    PropagationResult(
                        invariant_id=invariant.id,
                        target_repo=repo,
                        status=PropagationStatus.VIOLATED,
                        violations=violations,
                    )
                )
                violated += 1
            else:
                results.append(
                    PropagationResult(
                        invariant_id=invariant.id,
                        target_repo=repo,
                        status=PropagationStatus.COMPLIANT,
                    )
                )
                compliant += 1

        total = compliant + violated
        return PropagationReport(
            invariant_id=invariant.id,
            invariant_name=invariant.name,
            results=results,
            total_repos=total,
            compliant_repos=compliant,
            violated_repos=violated,
            compliance_rate=round(compliant / total, 3) if total > 0 else 0.0,
        )

    def _check_invariant(
        self, invariant: RegisteredInvariant, code_files: dict[str, str]
    ) -> list[str]:
        violations: list[str] = []
        for path, content in code_files.items():
            if invariant.check_type == "null_safety":
                if "None" in content and "is not None" not in content and ".get(" not in content:
                    violations.append(f"{path}: missing null check (invariant: {invariant.name})")
            elif (
                invariant.check_type == "encryption"
                and "password" in content.lower()
                and "encrypt" not in content.lower()
                and "hash" not in content.lower()
            ):
                violations.append(
                    f"{path}: unencrypted sensitive data (invariant: {invariant.name})"
                )
            if (
                invariant.check_type == "error_handling"
                and "except:" in content
                and "except Exception" not in content
            ):
                violations.append(f"{path}: bare except clause (invariant: {invariant.name})")
        return violations


class InvariantPropagationService:
    """Main service for multi-repository invariant propagation."""

    def __init__(self) -> None:
        self._registry = InvariantRegistry()
        self._engine = PropagationEngine()
        self._reports: list[PropagationReport] = []

    @property
    def registry(self) -> InvariantRegistry:
        return self._registry

    def register_invariant(self, **kwargs: Any) -> RegisteredInvariant:
        return self._registry.register(**kwargs)

    def propagate(
        self, invariant_id: str, repo_code: dict[str, dict[str, str]]
    ) -> PropagationReport | None:
        inv = self._registry.get(invariant_id)
        if not inv:
            return None
        report = self._engine.propagate(inv, repo_code)
        self._reports.append(report)
        return report

    def propagate_all(self, repo_code: dict[str, dict[str, str]]) -> list[PropagationReport]:
        reports: list[PropagationReport] = []
        for inv in self._registry.list_all():
            report = self._engine.propagate(inv, repo_code)
            reports.append(report)
            self._reports.append(report)
        return reports

    def get_governance_summary(self) -> GovernanceSummary:
        all_invs = self._registry.list_all(active_only=False)
        active = [i for i in all_invs if i.is_active]
        repos: set[str] = set()
        for i in active:
            repos.update(i.target_repos)

        total_checks = sum(r.total_repos for r in self._reports)
        total_compliant = sum(r.compliant_repos for r in self._reports)
        compliance = round(total_compliant / total_checks, 3) if total_checks > 0 else 0.0

        violation_counts: dict[str, int] = defaultdict(int)
        for r in self._reports:
            for res in r.results:
                if res.status == PropagationStatus.VIOLATED:
                    violation_counts[r.invariant_name] += 1

        top = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return GovernanceSummary(
            total_invariants=len(all_invs),
            active_invariants=len(active),
            repos_covered=len(repos),
            org_compliance_rate=compliance,
            top_violations=[{"invariant": k, "violations": v} for k, v in top],
        )


# ─── Singleton Access ──────────────────────────────────────────────────

_invariant_prop_instance: InvariantPropagationService | None = None


def get_invariant_propagation_service() -> InvariantPropagationService:
    global _invariant_prop_instance
    if _invariant_prop_instance is None:
        _invariant_prop_instance = InvariantPropagationService()
    return _invariant_prop_instance


def reset_invariant_propagation_service() -> None:
    global _invariant_prop_instance
    _invariant_prop_instance = None
