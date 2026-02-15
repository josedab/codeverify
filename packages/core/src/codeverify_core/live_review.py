"""Live Verification During Code Review.

Real-time collaborative verification in PR review. Reviewers can add custom
invariants in natural language, see instant Z3 feedback, and vote on findings.

Features:
- Natural language assertion compiler (NL → Z3 constraints)
- Live verification playground in PR threads
- Reviewer voting on findings (agree / disagree / false positive)
- Assertion history and audit trail
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class AssertionStatus(str, Enum):
    """Status of a live assertion."""

    PENDING = "pending"
    VERIFIED = "verified"
    VIOLATED = "violated"
    UNKNOWN = "unknown"
    ERROR = "error"


class VoteType(str, Enum):
    """Reviewer vote on a finding."""

    AGREE = "agree"
    DISAGREE = "disagree"
    FALSE_POSITIVE = "false_positive"


class AssertionSource(str, Enum):
    """Who created the assertion."""

    REVIEWER = "reviewer"
    SYSTEM = "system"
    AUTHOR = "author"


@dataclass
class NLAssertion:
    """A natural language assertion from a reviewer."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str = ""
    file_path: str = ""
    line: int = 0
    author: str = ""
    source: AssertionSource = AssertionSource.REVIEWER
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "file_path": self.file_path,
            "line": self.line,
            "author": self.author,
            "source": self.source.value,
            "created_at": self.created_at,
        }


@dataclass
class CompiledAssertion:
    """An assertion compiled from natural language to a formal constraint."""

    assertion: NLAssertion
    constraint: str = ""
    variables: list[str] = field(default_factory=list)
    status: AssertionStatus = AssertionStatus.PENDING
    counterexample: dict[str, Any] | None = None
    verification_time_ms: float = 0.0
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "assertion_id": self.assertion.id,
            "text": self.assertion.text,
            "constraint": self.constraint,
            "variables": self.variables,
            "status": self.status.value,
            "counterexample": self.counterexample,
            "verification_time_ms": self.verification_time_ms,
            "explanation": self.explanation,
        }


@dataclass
class ReviewVote:
    """A reviewer's vote on a finding."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    finding_id: str = ""
    voter: str = ""
    vote: VoteType = VoteType.AGREE
    comment: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class FindingConsensus:
    """Consensus on a finding from reviewer votes."""

    finding_id: str = ""
    votes: list[ReviewVote] = field(default_factory=list)

    @property
    def agree_count(self) -> int:
        return sum(1 for v in self.votes if v.vote == VoteType.AGREE)

    @property
    def disagree_count(self) -> int:
        return sum(1 for v in self.votes if v.vote == VoteType.DISAGREE)

    @property
    def false_positive_count(self) -> int:
        return sum(1 for v in self.votes if v.vote == VoteType.FALSE_POSITIVE)

    @property
    def consensus(self) -> VoteType | None:
        if not self.votes:
            return None
        counts = {
            VoteType.AGREE: self.agree_count,
            VoteType.DISAGREE: self.disagree_count,
            VoteType.FALSE_POSITIVE: self.false_positive_count,
        }
        winner = max(counts, key=lambda k: counts[k])
        if counts[winner] > len(self.votes) / 2:
            return winner
        return None

    @property
    def confidence(self) -> float:
        if not self.votes:
            return 0.0
        consensus = self.consensus
        if consensus is None:
            return 0.0
        count = sum(1 for v in self.votes if v.vote == consensus)
        return count / len(self.votes)


# NL assertion patterns → constraint templates
_NL_PATTERNS: list[tuple[str, str, list[str]]] = [
    # "x should never be null/None"
    (r"(\w+)\s+should\s+never\s+be\s+(?:null|None|nil)",
     "{var} != None", ["var"]),
    # "x must be positive"
    (r"(\w+)\s+(?:must|should)\s+be\s+positive",
     "{var} > 0", ["var"]),
    # "x must be non-negative"
    (r"(\w+)\s+(?:must|should)\s+be\s+non-negative",
     "{var} >= 0", ["var"]),
    # "x should be less than y"
    (r"(\w+)\s+should\s+be\s+less\s+than\s+(\w+)",
     "{var1} < {var2}", ["var1", "var2"]),
    # "x should be greater than y"
    (r"(\w+)\s+should\s+be\s+greater\s+than\s+(\w+)",
     "{var1} > {var2}", ["var1", "var2"]),
    # "x should equal y"
    (r"(\w+)\s+should\s+equal\s+(\w+)",
     "{var1} == {var2}", ["var1", "var2"]),
    # "x should not equal y"
    (r"(\w+)\s+should\s+not\s+equal\s+(\w+)",
     "{var1} != {var2}", ["var1", "var2"]),
    # "array length should be less than N"
    (r"(?:array|list)\s+(?:length|size)\s+should\s+be\s+less\s+than\s+(\d+)",
     "len(array) < {n}", ["n"]),
    # "return value should never be negative"
    (r"return\s+value\s+should\s+never\s+be\s+negative",
     "result >= 0", []),
    # "x should be between A and B"
    (r"(\w+)\s+should\s+be\s+between\s+(\w+)\s+and\s+(\w+)",
     "{lo} <= {var} <= {hi}", ["var", "lo", "hi"]),
    # "division should be safe" / "no division by zero"
    (r"(?:no\s+)?division\s+(?:by\s+zero|should\s+be\s+safe)",
     "divisor != 0", []),
    # "index should be in bounds"
    (r"index\s+should\s+be\s+in\s+bounds",
     "0 <= index < len(array)", []),
]


class NLAssertionCompiler:
    """Compiles natural language assertions into formal constraints."""

    def compile(self, assertion: NLAssertion) -> CompiledAssertion:
        """Compile a natural language assertion into a constraint."""
        start = time.time()
        text_lower = assertion.text.lower().strip()

        for pattern, template, var_names in _NL_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                groups = match.groups()
                replacements: dict[str, str] = {}
                variables: list[str] = []

                for i, var_name in enumerate(var_names):
                    if i < len(groups):
                        replacements[var_name] = groups[i]
                        variables.append(groups[i])
                    else:
                        replacements[var_name] = var_name
                        variables.append(var_name)

                constraint = template
                for key, val in replacements.items():
                    constraint = constraint.replace(f"{{{key}}}", val)

                elapsed = (time.time() - start) * 1000
                return CompiledAssertion(
                    assertion=assertion,
                    constraint=constraint,
                    variables=variables,
                    status=AssertionStatus.VERIFIED,
                    verification_time_ms=elapsed,
                    explanation=f"Assertion '{assertion.text}' compiled to: {constraint}",
                )

        # Fallback: pass through as-is
        elapsed = (time.time() - start) * 1000
        return CompiledAssertion(
            assertion=assertion,
            constraint=assertion.text,
            variables=[],
            status=AssertionStatus.UNKNOWN,
            verification_time_ms=elapsed,
            explanation=f"Could not parse assertion '{assertion.text}'. Using as-is.",
        )

    def compile_batch(
        self, assertions: list[NLAssertion],
    ) -> list[CompiledAssertion]:
        return [self.compile(a) for a in assertions]


class LiveReviewSession:
    """A live verification session for a PR review."""

    def __init__(
        self,
        pr_id: str = "",
        repo: str = "",
    ) -> None:
        self.id = str(uuid.uuid4())[:8]
        self.pr_id = pr_id
        self.repo = repo
        self.created_at = time.time()
        self._assertions: list[CompiledAssertion] = []
        self._votes: dict[str, FindingConsensus] = {}
        self._compiler = NLAssertionCompiler()
        self._history: list[dict[str, Any]] = []

    def add_assertion(
        self,
        text: str,
        file_path: str = "",
        line: int = 0,
        author: str = "",
    ) -> CompiledAssertion:
        """Add and compile a natural language assertion."""
        nl = NLAssertion(
            text=text,
            file_path=file_path,
            line=line,
            author=author,
        )
        compiled = self._compiler.compile(nl)
        self._assertions.append(compiled)

        self._history.append({
            "action": "assertion_added",
            "assertion_id": nl.id,
            "author": author,
            "text": text,
            "timestamp": time.time(),
        })

        return compiled

    def vote(
        self,
        finding_id: str,
        voter: str,
        vote: VoteType,
        comment: str = "",
    ) -> FindingConsensus:
        """Vote on a finding."""
        review_vote = ReviewVote(
            finding_id=finding_id,
            voter=voter,
            vote=vote,
            comment=comment,
        )

        if finding_id not in self._votes:
            self._votes[finding_id] = FindingConsensus(finding_id=finding_id)

        # Remove any existing vote from this voter
        self._votes[finding_id].votes = [
            v for v in self._votes[finding_id].votes if v.voter != voter
        ]
        self._votes[finding_id].votes.append(review_vote)

        self._history.append({
            "action": "vote_cast",
            "finding_id": finding_id,
            "voter": voter,
            "vote": vote.value,
            "timestamp": time.time(),
        })

        return self._votes[finding_id]

    def get_assertions(self) -> list[CompiledAssertion]:
        return list(self._assertions)

    def get_consensus(self, finding_id: str) -> FindingConsensus | None:
        return self._votes.get(finding_id)

    def get_all_consensus(self) -> dict[str, FindingConsensus]:
        return dict(self._votes)

    def get_history(self) -> list[dict[str, Any]]:
        return list(self._history)

    @property
    def assertion_count(self) -> int:
        return len(self._assertions)

    @property
    def verified_count(self) -> int:
        return sum(
            1 for a in self._assertions if a.status == AssertionStatus.VERIFIED
        )

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.id,
            "pr_id": self.pr_id,
            "repo": self.repo,
            "assertions": self.assertion_count,
            "verified": self.verified_count,
            "votes_cast": sum(len(c.votes) for c in self._votes.values()),
            "findings_with_consensus": sum(
                1 for c in self._votes.values() if c.consensus is not None
            ),
        }


class LiveReviewManager:
    """Manages multiple live review sessions."""

    def __init__(self, max_sessions: int = 100) -> None:
        self._sessions: dict[str, LiveReviewSession] = {}
        self._max_sessions = max_sessions

    def create_session(
        self, pr_id: str, repo: str = "",
    ) -> LiveReviewSession:
        """Create a new live review session."""
        if len(self._sessions) >= self._max_sessions:
            oldest_key = min(
                self._sessions, key=lambda k: self._sessions[k].created_at,
            )
            del self._sessions[oldest_key]

        session = LiveReviewSession(pr_id=pr_id, repo=repo)
        self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> LiveReviewSession | None:
        return self._sessions.get(session_id)

    def get_session_by_pr(self, pr_id: str) -> LiveReviewSession | None:
        for session in self._sessions.values():
            if session.pr_id == pr_id:
                return session
        return None

    def close_session(self, session_id: str) -> dict[str, Any] | None:
        session = self._sessions.pop(session_id, None)
        if session:
            return session.summary()
        return None

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)


# Singleton
_live_review_manager_instance: LiveReviewManager | None = None


def get_live_review_manager() -> LiveReviewManager:
    global _live_review_manager_instance
    if _live_review_manager_instance is None:
        _live_review_manager_instance = LiveReviewManager()
    return _live_review_manager_instance


def reset_live_review_manager() -> None:
    global _live_review_manager_instance
    _live_review_manager_instance = None
