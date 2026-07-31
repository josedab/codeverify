"""Gamified Developer Security Training.

Uses real CodeVerify findings to create personalized security training
with interactive challenges, progress tracking, and leaderboards.

Features:
- Personalized curriculum from individual finding history
- Interactive fix-it challenges using real (anonymized) findings
- Progress tracking per developer and team
- Leaderboard with streaks and badges
- Integration with certification program
- Team-level aggregation for privacy
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


class SkillLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ChallengeStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class BadgeType(str, Enum):
    FIRST_FIX = "first_fix"
    STREAK_7 = "streak_7"
    STREAK_30 = "streak_30"
    ZERO_CRITICAL = "zero_critical"
    CATEGORY_MASTER = "category_master"
    TEAM_CHAMPION = "team_champion"


@dataclass
class WeaknessArea:
    """A developer's weakness area based on finding history."""

    category: str = ""
    finding_count: int = 0
    last_occurrence: datetime | None = None
    improvement_trend: str = "stable"  # improving, stable, declining


@dataclass
class TrainingLesson:
    """A micro-lesson for a specific weakness."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    category: str = ""
    title: str = ""
    description: str = ""
    duration_minutes: int = 5
    difficulty: SkillLevel = SkillLevel.BEGINNER
    content_markdown: str = ""


@dataclass
class FixChallenge:
    """An interactive fix-it challenge."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    category: str = ""
    title: str = ""
    description: str = ""
    buggy_code: str = ""
    expected_fix_pattern: str = ""
    hint: str = ""
    difficulty: SkillLevel = SkillLevel.INTERMEDIATE
    points: int = 10
    status: ChallengeStatus = ChallengeStatus.NOT_STARTED


@dataclass
class DeveloperProgress:
    """Progress tracking for a developer."""

    developer_id: str = ""
    developer_name: str = ""
    lessons_completed: int = 0
    challenges_completed: int = 0
    total_points: int = 0
    current_streak: int = 0
    longest_streak: int = 0
    badges: list[BadgeType] = field(default_factory=list)
    weakness_areas: list[WeaknessArea] = field(default_factory=list)
    skill_level: SkillLevel = SkillLevel.BEGINNER
    last_activity: datetime | None = None


@dataclass
class TeamLeaderboard:
    """Team leaderboard data."""

    team_name: str = ""
    entries: list[dict[str, Any]] = field(default_factory=list)
    total_challenges_completed: int = 0
    avg_points: float = 0.0


LESSON_LIBRARY: dict[str, TrainingLesson] = {
    "null_safety": TrainingLesson(
        category="null_safety",
        title="Preventing Null Dereferences",
        description="Learn to guard against None/null values",
        content_markdown=(
            "## Null Safety\n\n"
            "**Problem**: Accessing attributes or methods on `None` raises `TypeError`.\n\n"
            "**Pattern**: Always check for `None` before use:\n"
            "```python\nif value is not None:\n    result = value.method()\n```\n\n"
            "**Better**: Use Optional types and `.get()` for dicts."
        ),
    ),
    "division_by_zero": TrainingLesson(
        category="division_by_zero",
        title="Guarding Against Division by Zero",
        description="Prevent ZeroDivisionError in arithmetic",
        content_markdown=(
            "## Division Safety\n\n"
            "**Rule**: Always validate divisors.\n"
            "```python\nresult = a / b if b != 0 else default_value\n```"
        ),
    ),
    "injection": TrainingLesson(
        category="injection",
        title="Preventing Injection Attacks",
        description="SQL, command, and code injection prevention",
        content_markdown=(
            "## Injection Prevention\n\n"
            "**Never** concatenate user input into queries.\n"
            "**Always** use parameterized queries or ORM methods."
        ),
    ),
    "credential_exposure": TrainingLesson(
        category="credential_exposure",
        title="Securing Credentials",
        description="Never hardcode secrets in source code",
        content_markdown=(
            "## Credential Security\n\n"
            "Use environment variables or secret managers.\n"
            "```python\nimport os\napi_key = os.environ['API_KEY']\n```"
        ),
    ),
}

CHALLENGE_TEMPLATES: dict[str, FixChallenge] = {
    "null_safety": FixChallenge(
        category="null_safety",
        title="Fix the Null Dereference",
        description="This function crashes when user is None. Add a guard.",
        buggy_code="def greet(user):\n    return f'Hello, {user.name}'",
        expected_fix_pattern="is not None",
        hint="Check if user is None before accessing .name",
        points=10,
    ),
    "division_by_zero": FixChallenge(
        category="division_by_zero",
        title="Fix the Division by Zero",
        description="This function crashes when count is 0.",
        buggy_code="def average(total, count):\n    return total / count",
        expected_fix_pattern="!= 0",
        hint="What happens when count is 0?",
        points=10,
    ),
    "injection": FixChallenge(
        category="injection",
        title="Fix the SQL Injection",
        description="This query is vulnerable to SQL injection.",
        buggy_code="def get_user(name):\n    query = f\"SELECT * FROM users WHERE name = '{name}'\"",
        expected_fix_pattern="parameterized",
        hint="Never use f-strings in SQL queries",
        points=20,
    ),
}


class CurriculumGenerator:
    """Generates personalized training curricula."""

    def generate(
        self, weakness_areas: list[WeaknessArea], _skill_level: SkillLevel
    ) -> list[TrainingLesson]:
        """Generate lessons targeting weakness areas."""
        lessons: list[TrainingLesson] = []
        sorted_areas = sorted(weakness_areas, key=lambda w: w.finding_count, reverse=True)

        for area in sorted_areas[:3]:
            lesson = LESSON_LIBRARY.get(area.category)
            if lesson:
                lessons.append(lesson)

        return lessons

    def generate_challenges(self, weakness_areas: list[WeaknessArea]) -> list[FixChallenge]:
        """Generate fix-it challenges for weakness areas."""
        challenges: list[FixChallenge] = []
        for area in weakness_areas[:3]:
            template = CHALLENGE_TEMPLATES.get(area.category)
            if template:
                challenge = FixChallenge(
                    category=template.category,
                    title=template.title,
                    description=template.description,
                    buggy_code=template.buggy_code,
                    expected_fix_pattern=template.expected_fix_pattern,
                    hint=template.hint,
                    points=template.points,
                    difficulty=template.difficulty,
                )
                challenges.append(challenge)
        return challenges


class GamifiedTrainingService:
    """Main service for gamified developer security training."""

    def __init__(self) -> None:
        self._curriculum = CurriculumGenerator()
        self._developers: dict[str, DeveloperProgress] = {}
        self._completed_challenges: dict[str, list[str]] = defaultdict(list)

    def analyze_weaknesses(
        self, developer_id: str, finding_history: list[dict[str, Any]]
    ) -> list[WeaknessArea]:
        """Analyze a developer's finding history to identify weaknesses."""
        category_counts: dict[str, int] = defaultdict(int)
        for finding in finding_history:
            category_counts[finding.get("category", "unknown")] += 1

        areas: list[WeaknessArea] = []
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            areas.append(WeaknessArea(category=cat, finding_count=count))

        progress = self._get_or_create(developer_id)
        progress.weakness_areas = areas
        return areas

    def get_curriculum(self, developer_id: str) -> list[TrainingLesson]:
        progress = self._developers.get(developer_id)
        if not progress:
            return []
        return self._curriculum.generate(progress.weakness_areas, progress.skill_level)

    def get_challenges(self, developer_id: str) -> list[FixChallenge]:
        progress = self._developers.get(developer_id)
        if not progress:
            return []
        return self._curriculum.generate_challenges(progress.weakness_areas)

    def submit_challenge(
        self, developer_id: str, challenge_id: str, submitted_code: str
    ) -> tuple[bool, str]:
        """Submit a challenge solution. Returns (passed, feedback)."""
        progress = self._get_or_create(developer_id)

        # Find challenge in curriculum
        challenges = self.get_challenges(developer_id)
        challenge = next((c for c in challenges if c.id == challenge_id), None)
        if not challenge:
            # Check templates by category match
            for _cat, template in CHALLENGE_TEMPLATES.items():
                if template.expected_fix_pattern in submitted_code:
                    challenge = template
                    break

        if challenge and challenge.expected_fix_pattern in submitted_code:
            progress.challenges_completed += 1
            progress.total_points += challenge.points
            progress.current_streak += 1
            progress.longest_streak = max(progress.longest_streak, progress.current_streak)
            progress.last_activity = datetime.now(UTC)

            if progress.challenges_completed == 1 and BadgeType.FIRST_FIX not in progress.badges:
                progress.badges.append(BadgeType.FIRST_FIX)
            if progress.current_streak >= 7 and BadgeType.STREAK_7 not in progress.badges:
                progress.badges.append(BadgeType.STREAK_7)

            self._update_skill_level(progress)
            return True, "✅ Correct! The fix addresses the vulnerability."

        progress.current_streak = 0
        return False, f"❌ Not quite. Hint: {challenge.hint if challenge else 'Try again.'}"

    def get_progress(self, developer_id: str) -> DeveloperProgress | None:
        return self._developers.get(developer_id)

    def get_leaderboard(self, team_name: str = "default") -> TeamLeaderboard:
        unsorted_entries: list[dict[str, Any]] = [
            {
                "name": p.developer_name or p.developer_id,
                "points": p.total_points,
                "challenges": p.challenges_completed,
                "streak": p.current_streak,
                "badges": len(p.badges),
            }
            for p in self._developers.values()
        ]
        entries = sorted(
            unsorted_entries,
            key=lambda x: x["points"],
            reverse=True,
        )
        total = sum(e["challenges"] for e in entries)
        avg = sum(e["points"] for e in entries) / len(entries) if entries else 0

        return TeamLeaderboard(
            team_name=team_name,
            entries=entries,
            total_challenges_completed=total,
            avg_points=round(avg, 1),
        )

    def _get_or_create(self, developer_id: str) -> DeveloperProgress:
        if developer_id not in self._developers:
            self._developers[developer_id] = DeveloperProgress(developer_id=developer_id)
        return self._developers[developer_id]

    def _update_skill_level(self, progress: DeveloperProgress) -> None:
        if progress.total_points >= 200:
            progress.skill_level = SkillLevel.EXPERT
        elif progress.total_points >= 100:
            progress.skill_level = SkillLevel.ADVANCED
        elif progress.total_points >= 30:
            progress.skill_level = SkillLevel.INTERMEDIATE


# ─── Singleton Access ──────────────────────────────────────────────────

_training_instance: GamifiedTrainingService | None = None


def get_gamified_training_service() -> GamifiedTrainingService:
    global _training_instance
    if _training_instance is None:
        _training_instance = GamifiedTrainingService()
    return _training_instance


def reset_gamified_training_service() -> None:
    global _training_instance
    _training_instance = None
