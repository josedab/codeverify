"""Base classes and interfaces for VCS abstraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator


class CheckStatus(str, Enum):
    """Status of a check run."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class CheckConclusion(str, Enum):
    """Conclusion of a completed check run."""

    SUCCESS = "success"
    FAILURE = "failure"
    NEUTRAL = "neutral"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"
    ACTION_REQUIRED = "action_required"


@dataclass
class VCSConfig:
    """Configuration for VCS client."""

    provider: str  # "github", "gitlab", "bitbucket"
    base_url: str | None = None  # For self-hosted instances
    token: str = ""
    app_id: str | None = None  # For GitHub App
    private_key: str | None = None  # For GitHub App
    webhook_secret: str | None = None
    timeout_seconds: int = 30


@dataclass
class User:
    """Represents a VCS user."""

    id: str | int
    username: str
    display_name: str | None = None
    email: str | None = None
    avatar_url: str | None = None


@dataclass
class Repository:
    """Represents a VCS repository."""

    id: str | int
    name: str
    full_name: str  # owner/repo format
    owner: str
    description: str | None = None
    default_branch: str = "main"
    private: bool = False
    clone_url: str | None = None
    html_url: str | None = None


@dataclass
class PullRequestFile:
    """A file changed in a pull request."""

    filename: str
    status: str  # "added", "removed", "modified", "renamed"
    additions: int = 0
    deletions: int = 0
    changes: int = 0
    patch: str | None = None
    previous_filename: str | None = None  # For renames
    blob_url: str | None = None


@dataclass
class PullRequest:
    """Represents a pull/merge request."""

    id: int
    number: int
    title: str
    body: str | None
    state: str  # "open", "closed", "merged"
    head_sha: str
    head_ref: str  # Branch name
    base_sha: str
    base_ref: str  # Target branch
    author: User
    created_at: datetime
    updated_at: datetime
    merged_at: datetime | None = None
    html_url: str | None = None
    diff_url: str | None = None
    files: list[PullRequestFile] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    reviewers: list[User] = field(default_factory=list)


@dataclass
class PullRequestComment:
    """A comment on a pull request."""

    id: int
    body: str
    author: User
    created_at: datetime
    updated_at: datetime | None = None
    path: str | None = None  # For inline comments
    line: int | None = None  # For inline comments
    side: str | None = None  # "LEFT" or "RIGHT" for diff comments
    in_reply_to_id: int | None = None


@dataclass
class CheckRunAnnotation:
    """An annotation on a check run (inline feedback)."""

    path: str
    start_line: int
    end_line: int
    annotation_level: str  # "notice", "warning", "failure"
    message: str
    title: str | None = None
    raw_details: str | None = None
    start_column: int | None = None
    end_column: int | None = None


@dataclass
class CheckRun:
    """A check run (CI status)."""

    id: int | None = None
    name: str = "CodeVerify"
    status: CheckStatus = CheckStatus.QUEUED
    conclusion: CheckConclusion | None = None
    title: str | None = None
    summary: str | None = None
    text: str | None = None
    annotations: list[CheckRunAnnotation] = field(default_factory=list)
    details_url: str | None = None
    external_id: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class VCSClient(ABC):
    """Abstract base class for VCS clients."""

    def __init__(self, config: VCSConfig) -> None:
        """Initialize the VCS client."""
        self.config = config

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass

    # Repository operations
    @abstractmethod
    async def get_repository(self, repo_full_name: str) -> Repository:
        """Get repository information."""
        pass

    @abstractmethod
    async def get_file_content(
        self,
        repo_full_name: str,
        path: str,
        ref: str | None = None,
    ) -> str:
        """Get file content from repository."""
        pass

    @abstractmethod
    async def list_files(
        self,
        repo_full_name: str,
        path: str = "",
        ref: str | None = None,
    ) -> list[str]:
        """List files in a directory."""
        pass

    # Pull Request operations
    @abstractmethod
    async def get_pull_request(
        self,
        repo_full_name: str,
        pr_number: int,
    ) -> PullRequest:
        """Get pull request details."""
        pass

    @abstractmethod
    async def get_pull_request_files(
        self,
        repo_full_name: str,
        pr_number: int,
    ) -> list[PullRequestFile]:
        """Get files changed in a pull request."""
        pass

    @abstractmethod
    async def get_pull_request_diff(
        self,
        repo_full_name: str,
        pr_number: int,
    ) -> str:
        """Get the diff for a pull request."""
        pass

    @abstractmethod
    async def create_pull_request_comment(
        self,
        repo_full_name: str,
        pr_number: int,
        body: str,
    ) -> PullRequestComment:
        """Create a comment on a pull request."""
        pass

    @abstractmethod
    async def create_review_comment(
        self,
        repo_full_name: str,
        pr_number: int,
        body: str,
        commit_sha: str,
        path: str,
        line: int,
        side: str = "RIGHT",
    ) -> PullRequestComment:
        """Create an inline review comment."""
        pass

    @abstractmethod
    async def update_comment(
        self,
        repo_full_name: str,
        comment_id: int,
        body: str,
    ) -> PullRequestComment:
        """Update an existing comment."""
        pass

    @abstractmethod
    async def delete_comment(
        self,
        repo_full_name: str,
        comment_id: int,
    ) -> bool:
        """Delete a comment."""
        pass

    # Check/Status operations
    @abstractmethod
    async def create_check_run(
        self,
        repo_full_name: str,
        head_sha: str,
        check_run: CheckRun,
    ) -> CheckRun:
        """Create a check run."""
        pass

    @abstractmethod
    async def update_check_run(
        self,
        repo_full_name: str,
        check_run_id: int,
        check_run: CheckRun,
    ) -> CheckRun:
        """Update an existing check run."""
        pass

    @abstractmethod
    async def create_commit_status(
        self,
        repo_full_name: str,
        sha: str,
        state: str,
        context: str,
        description: str | None = None,
        target_url: str | None = None,
    ) -> dict[str, Any]:
        """Create a commit status (simpler alternative to check runs)."""
        pass

    # Webhook operations
    @abstractmethod
    def verify_webhook_signature(
        self,
        payload: bytes,
        signature: str,
    ) -> bool:
        """Verify webhook signature."""
        pass

    @abstractmethod
    def parse_webhook_event(
        self,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Parse webhook event into normalized format."""
        pass

    # Suggested changes (GitHub-specific, optional for others)
    async def create_suggested_change(
        self,
        repo_full_name: str,
        pr_number: int,
        path: str,
        line: int,
        suggestion: str,
        body: str | None = None,
    ) -> PullRequestComment:
        """
        Create a suggested change comment.

        Not all providers support this - default implementation creates
        a regular comment with the suggestion in a code block.
        """
        suggestion_body = body or "Suggested fix:"
        suggestion_body += f"\n\n```suggestion\n{suggestion}\n```"
        return await self.create_review_comment(
            repo_full_name=repo_full_name,
            pr_number=pr_number,
            body=suggestion_body,
            commit_sha="HEAD",
            path=path,
            line=line,
        )
