"""Marketplace Community — Community review workflows and collaborative verification.

Companion to proof_marketplace_v2.py, focusing on community features:
peer review workflows, reputation engine, voting, challenges, awards,
and the main MarketplaceCommunity orchestrator.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


# =============================================================================
# Enums
# =============================================================================


class ReviewStatus(str, Enum):
    """Status of a proof submission in the review pipeline."""

    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    WITHDRAWN = "withdrawn"


class ContributorRole(str, Enum):
    """Roles within the community."""

    SUBMITTER = "submitter"
    REVIEWER = "reviewer"
    MAINTAINER = "maintainer"
    MODERATOR = "moderator"
    ADMIN = "admin"


class ReputationTier(str, Enum):
    """Tier derived from cumulative reputation score."""

    NEWCOMER = "newcomer"
    CONTRIBUTOR = "contributor"
    TRUSTED = "trusted"
    EXPERT = "expert"
    ELITE = "elite"


class VoteType(str, Enum):
    """Types of votes community members can cast."""

    UPVOTE = "upvote"
    DOWNVOTE = "downvote"
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"


class ChallengeType(str, Enum):
    """Categories of community challenges."""

    WEEKLY_PROOF = "weekly_proof"
    BUG_BOUNTY = "bug_bounty"
    OPTIMIZATION = "optimization"
    DOCUMENTATION = "documentation"
    MENTORING = "mentoring"


class AwardType(str, Enum):
    """Achievement awards members can earn."""

    FIRST_PROOF = "first_proof"
    HELPFUL_REVIEWER = "helpful_reviewer"
    TOP_CONTRIBUTOR = "top_contributor"
    BUG_HUNTER = "bug_hunter"
    MENTOR = "mentor"
    STREAK_7 = "streak_7"
    STREAK_30 = "streak_30"


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class CommunityMember:
    """A registered member of the marketplace community."""

    id: str
    username: str
    role: ContributorRole = ContributorRole.SUBMITTER
    reputation_score: int = 0
    tier: ReputationTier = ReputationTier.NEWCOMER
    proofs_submitted: int = 0
    reviews_given: int = 0
    proofs_approved: int = 0
    awards: list[str] = field(default_factory=list)
    joined_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    streak_days: int = 0
    last_active: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role.value,
            "reputation_score": self.reputation_score,
            "tier": self.tier.value,
            "proofs_submitted": self.proofs_submitted,
            "reviews_given": self.reviews_given,
            "proofs_approved": self.proofs_approved,
            "awards": self.awards,
            "joined_at": self.joined_at.isoformat(),
            "streak_days": self.streak_days,
        }


@dataclass
class ProofSubmission:
    """A proof submitted for community review."""

    id: str
    author_id: str
    title: str
    description: str
    code: str
    language: str
    proof_type: str
    tags: list[str] = field(default_factory=list)
    status: ReviewStatus = ReviewStatus.PENDING
    submitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    upvotes: int = 0
    downvotes: int = 0
    view_count: int = 0
    review_comments: list[dict[str, Any]] = field(default_factory=list)
    assigned_reviewers: list[str] = field(default_factory=list)

    @property
    def net_votes(self) -> int:
        return self.upvotes - self.downvotes

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "author_id": self.author_id,
            "title": self.title,
            "description": self.description,
            "language": self.language,
            "proof_type": self.proof_type,
            "tags": self.tags,
            "status": self.status.value,
            "submitted_at": self.submitted_at.isoformat(),
            "upvotes": self.upvotes,
            "downvotes": self.downvotes,
            "net_votes": self.net_votes,
            "view_count": self.view_count,
            "review_comments_count": len(self.review_comments),
        }


@dataclass
class ReviewComment:
    """A review comment on a proof submission."""

    id: str
    reviewer_id: str
    submission_id: str
    comment: str
    line_number: int | None = None
    category: str = "general"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    helpful_votes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "reviewer_id": self.reviewer_id,
            "submission_id": self.submission_id,
            "comment": self.comment,
            "line_number": self.line_number,
            "category": self.category,
            "created_at": self.created_at.isoformat(),
            "helpful_votes": self.helpful_votes,
        }


@dataclass
class Vote:
    """A vote cast by a community member."""

    voter_id: str
    target_id: str
    target_type: str
    vote_type: VoteType
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "voter_id": self.voter_id,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "vote_type": self.vote_type.value,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Challenge:
    """A community challenge or competition."""

    id: str
    challenge_type: ChallengeType
    title: str
    description: str
    reward_points: int
    start_date: datetime
    end_date: datetime
    participants: list[str] = field(default_factory=list)
    submissions: list[str] = field(default_factory=list)
    winner_id: str | None = None
    active: bool = True

    @property
    def is_expired(self) -> bool:
        return datetime.now(UTC) > self.end_date

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "challenge_type": self.challenge_type.value,
            "title": self.title,
            "description": self.description,
            "reward_points": self.reward_points,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "participants_count": len(self.participants),
            "submissions_count": len(self.submissions),
            "winner_id": self.winner_id,
            "active": self.active,
        }


@dataclass
class Award:
    """An achievement award earned by a member."""

    id: str
    award_type: AwardType
    member_id: str
    title: str
    description: str
    earned_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    points: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "award_type": self.award_type.value,
            "member_id": self.member_id,
            "title": self.title,
            "description": self.description,
            "earned_at": self.earned_at.isoformat(),
            "points": self.points,
        }


@dataclass
class LeaderboardEntry:
    """A single row in the community leaderboard."""

    rank: int
    member_id: str
    username: str
    tier: ReputationTier
    score: int
    proofs_count: int
    reviews_count: int
    awards_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "member_id": self.member_id,
            "username": self.username,
            "tier": self.tier.value,
            "score": self.score,
            "proofs_count": self.proofs_count,
            "reviews_count": self.reviews_count,
            "awards_count": self.awards_count,
        }


@dataclass
class CommunityStats:
    """Aggregate statistics for the community."""

    total_members: int
    active_members_30d: int
    total_proofs: int
    approved_proofs: int
    pending_reviews: int
    active_challenges: int
    top_contributors: list[LeaderboardEntry]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_members": self.total_members,
            "active_members_30d": self.active_members_30d,
            "total_proofs": self.total_proofs,
            "approved_proofs": self.approved_proofs,
            "pending_reviews": self.pending_reviews,
            "active_challenges": self.active_challenges,
            "top_contributors": [e.to_dict() for e in self.top_contributors],
        }


# =============================================================================
# Reputation Engine
# =============================================================================


class ReputationEngine:
    """Calculate and manage community reputation scores."""

    # Point values for actions
    POINTS_SUBMIT = 10
    POINTS_APPROVE = 25
    POINTS_REVIEW = 5
    POINTS_UPVOTE_RECEIVED = 2
    POINTS_DOWNVOTE_RECEIVED = -1
    POINTS_HELPFUL_VOTE = 1
    POINTS_CHALLENGE_WIN = 50

    def __init__(self) -> None:
        self._members: dict[str, CommunityMember] = {}
        self._action_log: list[dict[str, Any]] = []

    def register(self, member: CommunityMember) -> None:
        """Register a member with the engine."""
        self._members[member.id] = member

    def award_points(self, member_id: str, action: str, amount: int) -> int:
        """Award reputation points and recalculate tier.

        Returns the member's new total reputation score.
        """
        member = self._members.get(member_id)
        if member is None:
            logger.warning("reputation_member_not_found", member_id=member_id)
            return 0

        member.reputation_score = max(0, member.reputation_score + amount)
        member.tier = self.calculate_tier(member.reputation_score)
        member.last_active = datetime.now(UTC)

        self._action_log.append(
            {
                "member_id": member_id,
                "action": action,
                "amount": amount,
                "new_total": member.reputation_score,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        logger.info(
            "reputation_points_awarded",
            member_id=member_id,
            action=action,
            amount=amount,
            new_total=member.reputation_score,
            tier=member.tier.value,
        )
        return member.reputation_score

    def calculate_tier(self, reputation_score: int) -> ReputationTier:
        """Determine tier from cumulative reputation score."""
        thresholds = self._tier_thresholds()
        # Walk thresholds from highest to lowest
        for tier in reversed(list(thresholds.keys())):
            if reputation_score >= thresholds[tier]:
                return tier
        return ReputationTier.NEWCOMER

    def get_leaderboard(self, limit: int = 10) -> list[LeaderboardEntry]:
        """Return the top members ordered by reputation score."""
        sorted_members = sorted(
            self._members.values(),
            key=lambda m: m.reputation_score,
            reverse=True,
        )[:limit]

        entries: list[LeaderboardEntry] = []
        for rank, member in enumerate(sorted_members, start=1):
            entries.append(
                LeaderboardEntry(
                    rank=rank,
                    member_id=member.id,
                    username=member.username,
                    tier=member.tier,
                    score=member.reputation_score,
                    proofs_count=member.proofs_submitted,
                    reviews_count=member.reviews_given,
                    awards_count=len(member.awards),
                )
            )
        return entries

    def decay_inactive(self, days_inactive: int = 90) -> int:
        """Apply reputation decay to members inactive for *days_inactive* days.

        Returns the number of members affected.
        """
        cutoff = datetime.now(UTC) - timedelta(days=days_inactive)
        affected = 0
        for member in self._members.values():
            if member.last_active < cutoff and member.reputation_score > 0:
                decay = max(1, member.reputation_score // 20)  # 5% decay
                member.reputation_score = max(0, member.reputation_score - decay)
                member.tier = self.calculate_tier(member.reputation_score)
                affected += 1
                logger.info(
                    "reputation_decay_applied",
                    member_id=member.id,
                    decay=decay,
                    new_total=member.reputation_score,
                )
        return affected

    # -- private helpers -----------------------------------------------------

    @staticmethod
    def _tier_thresholds() -> dict[ReputationTier, int]:
        """Minimum reputation score required for each tier."""
        return {
            ReputationTier.NEWCOMER: 0,
            ReputationTier.CONTRIBUTOR: 100,
            ReputationTier.TRUSTED: 500,
            ReputationTier.EXPERT: 2000,
            ReputationTier.ELITE: 5000,
        }


# =============================================================================
# Review Workflow
# =============================================================================


class ReviewWorkflow:
    """Manage the proof review workflow."""

    # Minimum tier required to review proofs
    MIN_REVIEWER_TIER = ReputationTier.CONTRIBUTOR

    def __init__(self) -> None:
        self._submissions: dict[str, ProofSubmission] = {}
        self._members: dict[str, CommunityMember] = {}
        self._comments: list[ReviewComment] = []
        self._reputation: ReputationEngine | None = None

    def set_reputation_engine(self, engine: ReputationEngine) -> None:
        self._reputation = engine

    def register_member(self, member: CommunityMember) -> None:
        self._members[member.id] = member

    def submit_for_review(self, submission: ProofSubmission) -> ProofSubmission:
        """Accept a new submission into the review pipeline."""
        submission.status = ReviewStatus.PENDING
        self._submissions[submission.id] = submission

        # Award points for submitting
        author = self._members.get(submission.author_id)
        if author:
            author.proofs_submitted += 1
        if self._reputation:
            self._reputation.award_points(
                submission.author_id,
                "proof_submitted",
                ReputationEngine.POINTS_SUBMIT,
            )

        logger.info(
            "submission_created",
            submission_id=submission.id,
            author_id=submission.author_id,
        )
        return submission

    def assign_reviewer(self, submission_id: str, reviewer_id: str) -> bool:
        """Assign a reviewer to a submission.  Returns False on failure."""
        submission = self._submissions.get(submission_id)
        reviewer = self._members.get(reviewer_id)
        if submission is None or reviewer is None:
            return False
        if submission.author_id == reviewer_id:
            logger.warning("self_review_blocked", submission_id=submission_id)
            return False
        if not self._check_reviewer_eligibility(reviewer):
            logger.warning("reviewer_ineligible", reviewer_id=reviewer_id)
            return False

        submission.assigned_reviewers.append(reviewer_id)
        submission.status = ReviewStatus.IN_REVIEW
        logger.info(
            "reviewer_assigned",
            submission_id=submission_id,
            reviewer_id=reviewer_id,
        )
        return True

    def add_review(
        self,
        submission_id: str,
        reviewer_id: str,
        comment: str,
        approve: bool = False,
    ) -> ReviewComment | None:
        """Add a review comment.  Optionally approve the submission."""
        submission = self._submissions.get(submission_id)
        reviewer = self._members.get(reviewer_id)
        if submission is None or reviewer is None:
            return None

        review_comment = ReviewComment(
            id=uuid.uuid4().hex[:12],
            reviewer_id=reviewer_id,
            submission_id=submission_id,
            comment=comment,
        )
        self._comments.append(review_comment)
        submission.review_comments.append(review_comment.to_dict())
        reviewer.reviews_given += 1

        if self._reputation:
            self._reputation.award_points(
                reviewer_id,
                "review_given",
                ReputationEngine.POINTS_REVIEW,
            )

        if approve:
            self.approve(submission_id, reviewer_id)

        logger.info(
            "review_added",
            submission_id=submission_id,
            reviewer_id=reviewer_id,
            approve=approve,
        )
        return review_comment

    def approve(self, submission_id: str, reviewer_id: str) -> ProofSubmission | None:
        """Approve a submission."""
        submission = self._submissions.get(submission_id)
        if submission is None:
            return None

        submission.status = ReviewStatus.APPROVED

        # Reward the author for an approved proof
        author = self._members.get(submission.author_id)
        if author:
            author.proofs_approved += 1
        if self._reputation:
            self._reputation.award_points(
                submission.author_id,
                "proof_approved",
                ReputationEngine.POINTS_APPROVE,
            )

        logger.info(
            "submission_approved",
            submission_id=submission_id,
            reviewer_id=reviewer_id,
        )
        return submission

    def reject(
        self,
        submission_id: str,
        reviewer_id: str,
        reason: str,
    ) -> ProofSubmission | None:
        """Reject a submission with a reason."""
        submission = self._submissions.get(submission_id)
        if submission is None:
            return None

        submission.status = ReviewStatus.REJECTED
        submission.review_comments.append(
            {
                "reviewer_id": reviewer_id,
                "comment": f"Rejected: {reason}",
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        logger.info(
            "submission_rejected",
            submission_id=submission_id,
            reviewer_id=reviewer_id,
            reason=reason,
        )
        return submission

    def request_revision(
        self,
        submission_id: str,
        reviewer_id: str,
        comments: str,
    ) -> ProofSubmission | None:
        """Request revisions on a submission."""
        submission = self._submissions.get(submission_id)
        if submission is None:
            return None

        submission.status = ReviewStatus.NEEDS_REVISION
        submission.review_comments.append(
            {
                "reviewer_id": reviewer_id,
                "comment": f"Revision requested: {comments}",
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        logger.info(
            "revision_requested",
            submission_id=submission_id,
            reviewer_id=reviewer_id,
        )
        return submission

    # -- private helpers -----------------------------------------------------

    def _check_reviewer_eligibility(self, reviewer: CommunityMember) -> bool:
        """A reviewer must be at least CONTRIBUTOR tier or have a reviewer role."""
        if reviewer.role in (
            ContributorRole.REVIEWER,
            ContributorRole.MAINTAINER,
            ContributorRole.MODERATOR,
            ContributorRole.ADMIN,
        ):
            return True

        tier_order = list(ReputationTier)
        return tier_order.index(reviewer.tier) >= tier_order.index(self.MIN_REVIEWER_TIER)


# =============================================================================
# Voting System
# =============================================================================


class VotingSystem:
    """Handle community voting on proofs and reviews."""

    def __init__(self) -> None:
        self._votes: list[Vote] = []
        # Map target_id -> list of voter_ids to prevent duplicate votes
        self._voter_index: dict[str, set[str]] = defaultdict(set)
        self._reputation: ReputationEngine | None = None
        # Map target_id -> author_id for reputation updates
        self._target_authors: dict[str, str] = {}

    def set_reputation_engine(self, engine: ReputationEngine) -> None:
        self._reputation = engine

    def register_target(self, target_id: str, author_id: str) -> None:
        """Register who authored a given target so votes update reputation."""
        self._target_authors[target_id] = author_id

    def vote(
        self,
        voter_id: str,
        target_id: str,
        target_type: str,
        vote_type: VoteType,
    ) -> Vote | None:
        """Cast a vote.  Returns None if the vote is invalid."""
        if self._prevent_self_vote(voter_id, target_id):
            logger.warning("self_vote_blocked", voter_id=voter_id, target_id=target_id)
            return None

        # Prevent duplicate votes from the same voter
        if voter_id in self._voter_index[target_id]:
            logger.warning("duplicate_vote_blocked", voter_id=voter_id, target_id=target_id)
            return None

        new_vote = Vote(
            voter_id=voter_id,
            target_id=target_id,
            target_type=target_type,
            vote_type=vote_type,
        )
        self._votes.append(new_vote)
        self._voter_index[target_id].add(voter_id)

        self._update_reputation_from_votes(target_id, vote_type)

        logger.info(
            "vote_cast",
            voter_id=voter_id,
            target_id=target_id,
            vote_type=vote_type.value,
        )
        return new_vote

    def get_score(self, target_id: str) -> int:
        """Net score (upvotes − downvotes) for a target."""
        score = 0
        for v in self._votes:
            if v.target_id != target_id:
                continue
            if v.vote_type in (VoteType.UPVOTE, VoteType.HELPFUL):
                score += 1
            elif v.vote_type in (VoteType.DOWNVOTE, VoteType.NOT_HELPFUL):
                score -= 1
        return score

    def get_voters(self, target_id: str) -> list[Vote]:
        """Return all votes for a given target."""
        return [v for v in self._votes if v.target_id == target_id]

    # -- private helpers -----------------------------------------------------

    def _prevent_self_vote(self, voter_id: str, target_id: str) -> bool:
        """Return True if the voter is the author of the target."""
        author_id = self._target_authors.get(target_id)
        return author_id is not None and author_id == voter_id

    def _update_reputation_from_votes(self, target_id: str, vote_type: VoteType) -> None:
        """Credit or debit the author's reputation when their content is voted on."""
        if self._reputation is None:
            return
        author_id = self._target_authors.get(target_id)
        if author_id is None:
            return

        if vote_type in (VoteType.UPVOTE, VoteType.HELPFUL):
            self._reputation.award_points(
                author_id,
                "upvote_received",
                ReputationEngine.POINTS_UPVOTE_RECEIVED,
            )
        elif vote_type in (VoteType.DOWNVOTE, VoteType.NOT_HELPFUL):
            self._reputation.award_points(
                author_id,
                "downvote_received",
                ReputationEngine.POINTS_DOWNVOTE_RECEIVED,
            )


# =============================================================================
# Challenge Manager
# =============================================================================


class ChallengeManager:
    """Manage community challenges and competitions."""

    def __init__(self) -> None:
        self._challenges: dict[str, Challenge] = {}
        self._reputation: ReputationEngine | None = None

    def set_reputation_engine(self, engine: ReputationEngine) -> None:
        self._reputation = engine

    def create_challenge(
        self,
        challenge_type: ChallengeType,
        title: str,
        description: str,
        reward_points: int,
        duration_days: int = 7,
    ) -> Challenge:
        """Create a new community challenge."""
        now = datetime.now(UTC)
        challenge = Challenge(
            id=uuid.uuid4().hex[:12],
            challenge_type=challenge_type,
            title=title,
            description=description,
            reward_points=reward_points,
            start_date=now,
            end_date=now + timedelta(days=duration_days),
        )
        self._challenges[challenge.id] = challenge
        logger.info("challenge_created", challenge_id=challenge.id, title=title)
        return challenge

    def join_challenge(self, challenge_id: str, member_id: str) -> bool:
        """Add a member to an active challenge."""
        challenge = self._challenges.get(challenge_id)
        if challenge is None or not challenge.active or challenge.is_expired:
            return False
        if member_id in challenge.participants:
            return False

        challenge.participants.append(member_id)
        logger.info(
            "challenge_joined",
            challenge_id=challenge_id,
            member_id=member_id,
        )
        return True

    def submit_entry(
        self,
        challenge_id: str,
        member_id: str,
        submission_id: str,
    ) -> bool:
        """Record a submission for a challenge."""
        challenge = self._challenges.get(challenge_id)
        if challenge is None or not challenge.active:
            return False
        if member_id not in challenge.participants:
            return False

        challenge.submissions.append(submission_id)
        logger.info(
            "challenge_entry_submitted",
            challenge_id=challenge_id,
            member_id=member_id,
            submission_id=submission_id,
        )
        return True

    def complete_challenge(self, challenge_id: str) -> Challenge | None:
        """Mark a challenge as complete and award the winner."""
        challenge = self._challenges.get(challenge_id)
        if challenge is None:
            return None

        challenge.active = False

        # Pick the first participant with a submission as the winner (simplified)
        if challenge.participants and challenge.submissions:
            challenge.winner_id = challenge.participants[0]
            if self._reputation and challenge.winner_id:
                self._reputation.award_points(
                    challenge.winner_id,
                    "challenge_won",
                    challenge.reward_points,
                )

        logger.info(
            "challenge_completed",
            challenge_id=challenge_id,
            winner_id=challenge.winner_id,
        )
        return challenge

    def get_active_challenges(self) -> list[Challenge]:
        """Return all currently active, non-expired challenges."""
        active: list[Challenge] = []
        for ch in self._challenges.values():
            if ch.active and not ch.is_expired:
                active.append(ch)
        return active


# =============================================================================
# Award System
# =============================================================================


_AWARD_DESCRIPTIONS: dict[AwardType, tuple[str, str, int]] = {
    # (title, description, bonus_points)
    AwardType.FIRST_PROOF: (
        "First Proof",
        "Submitted your very first proof",
        10,
    ),
    AwardType.HELPFUL_REVIEWER: (
        "Helpful Reviewer",
        "Received 10+ helpful votes on reviews",
        20,
    ),
    AwardType.TOP_CONTRIBUTOR: (
        "Top Contributor",
        "Reached 50 approved proofs",
        50,
    ),
    AwardType.BUG_HUNTER: (
        "Bug Hunter",
        "Found and reported 10+ bugs in proofs",
        30,
    ),
    AwardType.MENTOR: (
        "Mentor",
        "Helped 5+ newcomers get their first proof approved",
        40,
    ),
    AwardType.STREAK_7: (
        "Week Warrior",
        "Active for 7 consecutive days",
        15,
    ),
    AwardType.STREAK_30: (
        "Monthly Maven",
        "Active for 30 consecutive days",
        50,
    ),
}


class AwardSystem:
    """Award achievements and badges."""

    def __init__(self) -> None:
        self._awards: list[Award] = []
        self._member_awards: dict[str, list[Award]] = defaultdict(list)
        self._reputation: ReputationEngine | None = None

    def set_reputation_engine(self, engine: ReputationEngine) -> None:
        self._reputation = engine

    def check_awards(self, member: CommunityMember) -> list[Award]:
        """Check which new awards a member qualifies for and grant them."""
        existing = {a.award_type.value for a in self._member_awards.get(member.id, [])}
        new_awards: list[Award] = []

        for award_type in AwardType:
            if award_type.value in existing:
                continue
            if self._check_milestone(member, award_type):
                award = self.grant_award(member.id, award_type)
                new_awards.append(award)

        return new_awards

    def grant_award(self, member_id: str, award_type: AwardType) -> Award:
        """Grant an award to a member."""
        title, description, points = _AWARD_DESCRIPTIONS[award_type]
        award = Award(
            id=uuid.uuid4().hex[:12],
            award_type=award_type,
            member_id=member_id,
            title=title,
            description=description,
            points=points,
        )
        self._awards.append(award)
        self._member_awards[member_id].append(award)

        if self._reputation:
            self._reputation.award_points(member_id, f"award_{award_type.value}", points)

        logger.info(
            "award_granted",
            member_id=member_id,
            award_type=award_type.value,
            points=points,
        )
        return award

    def get_member_awards(self, member_id: str) -> list[Award]:
        """Return all awards earned by a member."""
        return list(self._member_awards.get(member_id, []))

    # -- private helpers -----------------------------------------------------

    @staticmethod
    def _check_milestone(member: CommunityMember, award_type: AwardType) -> bool:
        """Return True if the member meets the criteria for *award_type*."""
        if award_type == AwardType.FIRST_PROOF:
            return member.proofs_submitted >= 1
        if award_type == AwardType.HELPFUL_REVIEWER:
            return member.reviews_given >= 10
        if award_type == AwardType.TOP_CONTRIBUTOR:
            return member.proofs_approved >= 50
        if award_type == AwardType.BUG_HUNTER:
            return member.reviews_given >= 25
        if award_type == AwardType.MENTOR:
            return member.reviews_given >= 15 and member.proofs_approved >= 5
        if award_type == AwardType.STREAK_7:
            return member.streak_days >= 7
        if award_type == AwardType.STREAK_30:
            return member.streak_days >= 30
        return False


# =============================================================================
# Marketplace Community — Main Orchestrator
# =============================================================================


class MarketplaceCommunity:
    """Main orchestrator for the marketplace community.

    Wires together the reputation engine, review workflow, voting system,
    challenge manager, and award system into a single coherent API.
    """

    def __init__(self) -> None:
        self._reputation = ReputationEngine()
        self._workflow = ReviewWorkflow()
        self._voting = VotingSystem()
        self._challenges = ChallengeManager()
        self._awards = AwardSystem()

        # Wire sub-systems to the shared reputation engine
        self._workflow.set_reputation_engine(self._reputation)
        self._voting.set_reputation_engine(self._reputation)
        self._challenges.set_reputation_engine(self._reputation)
        self._awards.set_reputation_engine(self._reputation)

        self._members: dict[str, CommunityMember] = {}
        self._submissions: dict[str, ProofSubmission] = {}

        logger.info("marketplace_community_initialized")

    # -- member management ---------------------------------------------------

    def register_member(
        self,
        username: str,
        role: ContributorRole = ContributorRole.SUBMITTER,
    ) -> CommunityMember:
        """Register a new community member."""
        member = CommunityMember(
            id=uuid.uuid4().hex[:12],
            username=username,
            role=role,
        )
        self._members[member.id] = member
        self._reputation.register(member)
        self._workflow.register_member(member)

        logger.info("member_registered", member_id=member.id, username=username)
        return member

    # -- proof lifecycle -----------------------------------------------------

    def submit_proof(
        self,
        author_id: str,
        title: str,
        description: str,
        code: str,
        language: str,
        proof_type: str,
        tags: list[str] | None = None,
    ) -> ProofSubmission | None:
        """Submit a proof for community review."""
        if author_id not in self._members:
            logger.warning("submit_unknown_author", author_id=author_id)
            return None

        submission = ProofSubmission(
            id=uuid.uuid4().hex[:12],
            author_id=author_id,
            title=title,
            description=description,
            code=code,
            language=language,
            proof_type=proof_type,
            tags=tags or [],
        )
        self._submissions[submission.id] = submission
        self._voting.register_target(submission.id, author_id)
        self._workflow.submit_for_review(submission)

        # Update streak
        member = self._members[author_id]
        self._update_streak(member)

        # Check for new awards
        self._awards.check_awards(member)

        return submission

    def review_proof(
        self,
        submission_id: str,
        reviewer_id: str,
        comment: str,
        approve: bool = False,
    ) -> ReviewComment | None:
        """Add a review to a proof submission."""
        if reviewer_id not in self._members:
            return None

        # Ensure reviewer is assigned
        submission = self._submissions.get(submission_id)
        if submission is not None and reviewer_id not in submission.assigned_reviewers:
            self._workflow.assign_reviewer(submission_id, reviewer_id)

        review_comment = self._workflow.add_review(
            submission_id,
            reviewer_id,
            comment,
            approve=approve,
        )

        if review_comment is not None:
            member = self._members[reviewer_id]
            self._update_streak(member)
            self._awards.check_awards(member)

        return review_comment

    # -- voting --------------------------------------------------------------

    def vote(
        self,
        voter_id: str,
        target_id: str,
        target_type: str,
        vote_type: VoteType,
    ) -> Vote | None:
        """Cast a vote on a proof or review."""
        if voter_id not in self._members:
            return None
        return self._voting.vote(voter_id, target_id, target_type, vote_type)

    # -- leaderboard & stats -------------------------------------------------

    def get_leaderboard(self, limit: int = 10) -> list[LeaderboardEntry]:
        """Return the community leaderboard."""
        return self._reputation.get_leaderboard(limit=limit)

    def get_community_stats(self) -> CommunityStats:
        """Aggregate community statistics."""
        cutoff_30d = datetime.now(UTC) - timedelta(days=30)
        active_30d = sum(1 for m in self._members.values() if m.last_active >= cutoff_30d)
        approved = sum(1 for s in self._submissions.values() if s.status == ReviewStatus.APPROVED)
        pending = sum(
            1
            for s in self._submissions.values()
            if s.status in (ReviewStatus.PENDING, ReviewStatus.IN_REVIEW)
        )
        active_challenges = len(self._challenges.get_active_challenges())

        return CommunityStats(
            total_members=len(self._members),
            active_members_30d=active_30d,
            total_proofs=len(self._submissions),
            approved_proofs=approved,
            pending_reviews=pending,
            active_challenges=active_challenges,
            top_contributors=self._reputation.get_leaderboard(limit=5),
        )

    # -- challenges ----------------------------------------------------------

    def create_challenge(
        self,
        title: str,
        description: str,
        reward_points: int,
        challenge_type: ChallengeType = ChallengeType.WEEKLY_PROOF,
    ) -> Challenge:
        """Create a new community challenge."""
        return self._challenges.create_challenge(
            challenge_type=challenge_type,
            title=title,
            description=description,
            reward_points=reward_points,
        )

    # -- member profiles -----------------------------------------------------

    def get_member_profile(self, member_id: str) -> dict[str, Any]:
        """Return a member's full profile including awards and stats."""
        member = self._members.get(member_id)
        if member is None:
            return {}

        awards = self._awards.get_member_awards(member_id)
        submissions = [s.to_dict() for s in self._submissions.values() if s.author_id == member_id]

        profile = member.to_dict()
        profile["awards_detail"] = [a.to_dict() for a in awards]
        profile["submissions"] = submissions
        profile["vote_score"] = sum(self._voting.get_score(s["id"]) for s in submissions)
        return profile

    # -- private helpers -----------------------------------------------------

    @staticmethod
    def _update_streak(member: CommunityMember) -> None:
        """Update the member's consecutive-activity streak."""
        now = datetime.now(UTC)
        delta = now - member.last_active
        if delta < timedelta(days=1):
            # Already active today — no change
            pass
        elif delta < timedelta(days=2):
            # Active yesterday — extend streak
            member.streak_days += 1
        else:
            # Gap — reset streak
            member.streak_days = 1
        member.last_active = now
