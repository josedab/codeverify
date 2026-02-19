"""Verification-Aware Code Review Assignments.

Auto-assigns PR reviewers based on verification results, matching
risk profiles to reviewer expertise with load balancing.

Features:
- PR risk classification from verification findings
- Reviewer-expertise matching (security → AppSec, types → TS expert)
- Load balancing across team with configurable caps
- Review queue optimization
- Assignment history and metrics tracking
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class PRRiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ExpertiseArea(str, Enum):
    SECURITY = "security"
    FORMAL_VERIFICATION = "formal_verification"
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    GO = "go"
    PERFORMANCE = "performance"
    DATABASE = "database"
    API_DESIGN = "api_design"
    GENERAL = "general"


@dataclass
class Reviewer:
    """A potential code reviewer."""
    id: str = ""
    name: str = ""
    expertise: list[ExpertiseArea] = field(default_factory=list)
    seniority: str = "mid"  # junior, mid, senior, staff
    current_load: int = 0
    max_load: int = 5
    is_available: bool = True

    @property
    def capacity(self) -> int:
        return max(0, self.max_load - self.current_load)


@dataclass
class PRRiskProfile:
    """Risk profile of a pull request."""
    pr_id: str = ""
    risk_level: PRRiskLevel = PRRiskLevel.MEDIUM
    finding_categories: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    critical_findings: int = 0
    high_findings: int = 0
    has_security_issues: bool = False
    has_formal_failures: bool = False
    changed_files: int = 0
    risk_score: float = 0.5


@dataclass
class ReviewAssignment:
    """A review assignment decision."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    pr_id: str = ""
    reviewer_id: str = ""
    reviewer_name: str = ""
    risk_level: PRRiskLevel = PRRiskLevel.MEDIUM
    match_reason: str = ""
    match_score: float = 0.0
    assigned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AssignmentStats:
    """Statistics for review assignments."""
    total_assignments: int = 0
    avg_match_score: float = 0.0
    load_distribution: dict[str, int] = field(default_factory=dict)
    risk_distribution: dict[str, int] = field(default_factory=dict)


class RiskClassifier:
    """Classifies PR risk from verification findings."""

    def classify(
        self, critical: int = 0, high: int = 0, medium: int = 0,
        has_security: bool = False, has_formal_failures: bool = False,
        changed_files: int = 0,
    ) -> PRRiskProfile:
        if critical > 0 or (has_security and has_formal_failures):
            level = PRRiskLevel.CRITICAL
            score = 0.95
        elif high > 2 or has_security:
            level = PRRiskLevel.HIGH
            score = 0.75
        elif high > 0 or medium > 5 or changed_files > 20:
            level = PRRiskLevel.MEDIUM
            score = 0.5
        else:
            level = PRRiskLevel.LOW
            score = 0.2

        categories: list[str] = []
        if has_security:
            categories.append("security")
        if has_formal_failures:
            categories.append("formal_verification")

        return PRRiskProfile(
            risk_level=level, risk_score=score,
            critical_findings=critical, high_findings=high,
            has_security_issues=has_security, has_formal_failures=has_formal_failures,
            changed_files=changed_files, finding_categories=categories,
        )


class ReviewerMatcher:
    """Matches PR risk profiles to reviewer expertise."""

    CATEGORY_TO_EXPERTISE: dict[str, ExpertiseArea] = {
        "security": ExpertiseArea.SECURITY,
        "credential_exposure": ExpertiseArea.SECURITY,
        "injection": ExpertiseArea.SECURITY,
        "formal_verification": ExpertiseArea.FORMAL_VERIFICATION,
        "null_safety": ExpertiseArea.FORMAL_VERIFICATION,
        "division_by_zero": ExpertiseArea.FORMAL_VERIFICATION,
        "type_safety": ExpertiseArea.TYPESCRIPT,
        "performance": ExpertiseArea.PERFORMANCE,
        "database": ExpertiseArea.DATABASE,
    }

    SENIORITY_ORDER = {"staff": 4, "senior": 3, "mid": 2, "junior": 1}

    def match(
        self, profile: PRRiskProfile, reviewers: list[Reviewer]
    ) -> list[tuple[Reviewer, float, str]]:
        """Match reviewers to a PR. Returns [(reviewer, score, reason)]."""
        scored: list[tuple[Reviewer, float, str]] = []

        needed_expertise = set()
        for cat in profile.finding_categories:
            exp = self.CATEGORY_TO_EXPERTISE.get(cat)
            if exp:
                needed_expertise.add(exp)

        for reviewer in reviewers:
            if not reviewer.is_available or reviewer.capacity <= 0:
                continue

            score = 0.0
            reasons: list[str] = []

            # Expertise match
            overlap = set(reviewer.expertise) & needed_expertise
            if overlap:
                score += 0.4 * (len(overlap) / max(len(needed_expertise), 1))
                reasons.append(f"expertise: {', '.join(e.value for e in overlap)}")

            # Seniority match for risk
            seniority_score = self.SENIORITY_ORDER.get(reviewer.seniority, 2)
            if profile.risk_level in (PRRiskLevel.CRITICAL, PRRiskLevel.HIGH):
                if seniority_score >= 3:
                    score += 0.3
                    reasons.append(f"senior reviewer for {profile.risk_level.value} risk")
            else:
                score += 0.2
                reasons.append("available reviewer")

            # Load balancing
            load_factor = reviewer.capacity / reviewer.max_load if reviewer.max_load > 0 else 0
            score += 0.2 * load_factor
            if load_factor > 0.5:
                reasons.append("low current load")

            # General availability
            score += 0.1

            scored.append((reviewer, round(score, 3), "; ".join(reasons)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


class ReviewAssignmentService:
    """Main service for verification-aware review assignments."""

    def __init__(self) -> None:
        self._classifier = RiskClassifier()
        self._matcher = ReviewerMatcher()
        self._reviewers: dict[str, Reviewer] = {}
        self._assignments: list[ReviewAssignment] = []

    def register_reviewer(self, reviewer: Reviewer) -> None:
        self._reviewers[reviewer.id] = reviewer

    def assign_reviewer(
        self, pr_id: str, critical: int = 0, high: int = 0, medium: int = 0,
        has_security: bool = False, has_formal_failures: bool = False,
        changed_files: int = 0, max_reviewers: int = 2,
    ) -> list[ReviewAssignment]:
        """Assign reviewers to a PR based on verification results."""
        profile = self._classifier.classify(
            critical, high, medium, has_security, has_formal_failures, changed_files
        )
        profile.pr_id = pr_id

        matches = self._matcher.match(profile, list(self._reviewers.values()))
        assignments: list[ReviewAssignment] = []

        for reviewer, score, reason in matches[:max_reviewers]:
            assignment = ReviewAssignment(
                pr_id=pr_id, reviewer_id=reviewer.id,
                reviewer_name=reviewer.name,
                risk_level=profile.risk_level,
                match_reason=reason, match_score=score,
            )
            assignments.append(assignment)
            self._assignments.append(assignment)
            reviewer.current_load += 1

        return assignments

    def complete_review(self, reviewer_id: str) -> None:
        reviewer = self._reviewers.get(reviewer_id)
        if reviewer and reviewer.current_load > 0:
            reviewer.current_load -= 1

    def get_stats(self) -> AssignmentStats:
        load_dist: dict[str, int] = {}
        for r in self._reviewers.values():
            load_dist[r.name] = r.current_load

        risk_dist: dict[str, int] = defaultdict(int)
        scores: list[float] = []
        for a in self._assignments:
            risk_dist[a.risk_level.value] += 1
            scores.append(a.match_score)

        return AssignmentStats(
            total_assignments=len(self._assignments),
            avg_match_score=round(sum(scores) / len(scores), 3) if scores else 0.0,
            load_distribution=load_dist,
            risk_distribution=dict(risk_dist),
        )


# ─── Singleton Access ──────────────────────────────────────────────────

_review_assign_instance: ReviewAssignmentService | None = None

def get_review_assignment_service() -> ReviewAssignmentService:
    global _review_assign_instance
    if _review_assign_instance is None:
        _review_assign_instance = ReviewAssignmentService()
    return _review_assign_instance

def reset_review_assignment_service() -> None:
    global _review_assign_instance
    _review_assign_instance = None
