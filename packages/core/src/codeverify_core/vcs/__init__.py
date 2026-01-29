"""VCS (Version Control System) abstraction layer for multi-provider support."""

from codeverify_core.vcs.base import (
    CheckConclusion,
    CheckRun,
    CheckRunAnnotation,
    CheckStatus,
    PullRequest,
    PullRequestComment,
    PullRequestFile,
    Repository,
    User,
    VCSClient,
    VCSConfig,
)
from codeverify_core.vcs.bitbucket import BitbucketClient
from codeverify_core.vcs.factory import create_vcs_client, get_provider_from_url
from codeverify_core.vcs.github import GitHubAppAuthenticator, GitHubClient
from codeverify_core.vcs.gitlab import GitLabClient

__all__ = [
    "BitbucketClient",
    "CheckConclusion",
    "CheckRun",
    "CheckRunAnnotation",
    "CheckStatus",
    "GitHubAppAuthenticator",
    "GitHubClient",
    "GitLabClient",
    "PullRequest",
    "PullRequestComment",
    "PullRequestFile",
    "Repository",
    "User",
    "VCSClient",
    "VCSConfig",
    "create_vcs_client",
    "get_provider_from_url",
]
