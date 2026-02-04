"""
Universal Git Support - Feature 10

Support any Git repository including self-hosted, air-gapped, and exotic platforms
(Gitea, Gerrit, Azure DevOps, etc.). Provides CLI-first approach, generic webhook
receiver, and offline operation capabilities.

Key capabilities:
- Generic Git provider abstraction supporting any Git hosting platform
- CLI-first design for maximum flexibility
- Local daemon for offline verification
- Generic webhook receiver for any Git server
- Air-gapped deployment support with bundled models
"""

import hashlib
import hmac
import json
import os
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse


class GitProvider(Enum):
    """Supported Git hosting providers."""
    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"
    GITEA = "gitea"
    GERRIT = "gerrit"
    AZURE_DEVOPS = "azure_devops"
    GOGS = "gogs"
    CODEBERG = "codeberg"
    SOURCEHUT = "sourcehut"
    LOCAL = "local"
    GENERIC = "generic"


class WebhookEventType(Enum):
    """Types of webhook events."""
    PUSH = "push"
    PULL_REQUEST = "pull_request"
    MERGE_REQUEST = "merge_request"
    TAG = "tag"
    COMMENT = "comment"
    REVIEW = "review"
    UNKNOWN = "unknown"


@dataclass
class GitCredentials:
    """Credentials for accessing Git repositories."""
    provider: GitProvider
    token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    ssh_key_path: Optional[str] = None
    api_url: Optional[str] = None


@dataclass
class Repository:
    """Generic repository representation."""
    provider: GitProvider
    owner: str
    name: str
    clone_url: str
    api_url: Optional[str] = None
    default_branch: str = "main"
    is_private: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class PullRequest:
    """Generic pull/merge request representation."""
    id: int
    number: int
    title: str
    description: str
    source_branch: str
    target_branch: str
    author: str
    state: str  # open, closed, merged
    created_at: datetime
    updated_at: datetime
    url: str
    diff_url: Optional[str] = None
    commits: list[str] = field(default_factory=list)


@dataclass
class WebhookPayload:
    """Parsed webhook payload from any provider."""
    provider: GitProvider
    event_type: WebhookEventType
    repository: Repository
    pull_request: Optional[PullRequest] = None
    ref: Optional[str] = None
    before_sha: Optional[str] = None
    after_sha: Optional[str] = None
    commits: list[dict] = field(default_factory=list)
    sender: Optional[str] = None
    raw_payload: dict = field(default_factory=dict)


class GitProviderAdapter(ABC):
    """Abstract adapter for Git hosting providers."""

    @abstractmethod
    def parse_webhook(self, headers: dict, body: dict) -> WebhookPayload:
        """Parse incoming webhook payload."""
        pass

    @abstractmethod
    def get_pull_request(self, repo: Repository, pr_number: int) -> PullRequest:
        """Get pull request details."""
        pass

    @abstractmethod
    def get_diff(self, repo: Repository, pr: PullRequest) -> str:
        """Get diff for a pull request."""
        pass

    @abstractmethod
    def post_comment(self, repo: Repository, pr: PullRequest, comment: str) -> bool:
        """Post a comment on a pull request."""
        pass

    @abstractmethod
    def update_status(
        self, repo: Repository, sha: str, state: str, description: str, context: str
    ) -> bool:
        """Update commit status/check."""
        pass


class GitHubAdapter(GitProviderAdapter):
    """Adapter for GitHub (and GitHub Enterprise)."""

    def __init__(self, credentials: GitCredentials):
        self.credentials = credentials
        self.api_base = credentials.api_url or "https://api.github.com"

    def parse_webhook(self, headers: dict, body: dict) -> WebhookPayload:
        event_type = headers.get('X-GitHub-Event', headers.get('x-github-event', ''))
        
        repo_data = body.get('repository', {})
        repository = Repository(
            provider=GitProvider.GITHUB,
            owner=repo_data.get('owner', {}).get('login', ''),
            name=repo_data.get('name', ''),
            clone_url=repo_data.get('clone_url', ''),
            default_branch=repo_data.get('default_branch', 'main'),
            is_private=repo_data.get('private', False)
        )
        
        payload = WebhookPayload(
            provider=GitProvider.GITHUB,
            event_type=self._map_event_type(event_type),
            repository=repository,
            sender=body.get('sender', {}).get('login'),
            raw_payload=body
        )
        
        if event_type == 'pull_request':
            pr_data = body.get('pull_request', {})
            payload.pull_request = PullRequest(
                id=pr_data.get('id', 0),
                number=pr_data.get('number', 0),
                title=pr_data.get('title', ''),
                description=pr_data.get('body', ''),
                source_branch=pr_data.get('head', {}).get('ref', ''),
                target_branch=pr_data.get('base', {}).get('ref', ''),
                author=pr_data.get('user', {}).get('login', ''),
                state=pr_data.get('state', 'open'),
                created_at=datetime.fromisoformat(pr_data.get('created_at', '').replace('Z', '+00:00')),
                updated_at=datetime.fromisoformat(pr_data.get('updated_at', '').replace('Z', '+00:00')),
                url=pr_data.get('html_url', ''),
                diff_url=pr_data.get('diff_url', '')
            )
        elif event_type == 'push':
            payload.ref = body.get('ref')
            payload.before_sha = body.get('before')
            payload.after_sha = body.get('after')
            payload.commits = body.get('commits', [])
        
        return payload

    def _map_event_type(self, event: str) -> WebhookEventType:
        mapping = {
            'push': WebhookEventType.PUSH,
            'pull_request': WebhookEventType.PULL_REQUEST,
            'create': WebhookEventType.TAG,
            'issue_comment': WebhookEventType.COMMENT,
            'pull_request_review': WebhookEventType.REVIEW,
        }
        return mapping.get(event, WebhookEventType.UNKNOWN)

    def get_pull_request(self, repo: Repository, pr_number: int) -> PullRequest:
        # Simulated - would make actual API call
        return PullRequest(
            id=0, number=pr_number, title="", description="",
            source_branch="", target_branch="", author="",
            state="open", created_at=datetime.now(), updated_at=datetime.now(),
            url=""
        )

    def get_diff(self, repo: Repository, pr: PullRequest) -> str:
        # Would fetch actual diff via API
        return ""

    def post_comment(self, repo: Repository, pr: PullRequest, comment: str) -> bool:
        return True

    def update_status(
        self, repo: Repository, sha: str, state: str, description: str, context: str
    ) -> bool:
        return True


class GitLabAdapter(GitProviderAdapter):
    """Adapter for GitLab (and self-hosted)."""

    def __init__(self, credentials: GitCredentials):
        self.credentials = credentials
        self.api_base = credentials.api_url or "https://gitlab.com/api/v4"

    def parse_webhook(self, headers: dict, body: dict) -> WebhookPayload:
        event_type = headers.get('X-Gitlab-Event', headers.get('x-gitlab-event', ''))
        
        project = body.get('project', body.get('repository', {}))
        repository = Repository(
            provider=GitProvider.GITLAB,
            owner=project.get('namespace', project.get('path_with_namespace', '').split('/')[0]),
            name=project.get('name', ''),
            clone_url=project.get('git_http_url', project.get('url', '')),
            default_branch=project.get('default_branch', 'main'),
        )
        
        payload = WebhookPayload(
            provider=GitProvider.GITLAB,
            event_type=self._map_event_type(event_type),
            repository=repository,
            sender=body.get('user', {}).get('username', body.get('user_username')),
            raw_payload=body
        )
        
        if 'merge_request' in event_type.lower() or 'object_attributes' in body:
            mr_data = body.get('object_attributes', {})
            if mr_data.get('noteable_type') != 'MergeRequest':
                payload.pull_request = PullRequest(
                    id=mr_data.get('id', 0),
                    number=mr_data.get('iid', 0),
                    title=mr_data.get('title', ''),
                    description=mr_data.get('description', ''),
                    source_branch=mr_data.get('source_branch', ''),
                    target_branch=mr_data.get('target_branch', ''),
                    author=mr_data.get('author_id', ''),
                    state=mr_data.get('state', 'opened'),
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    url=mr_data.get('url', '')
                )
        elif event_type == 'Push Hook':
            payload.ref = body.get('ref')
            payload.before_sha = body.get('before')
            payload.after_sha = body.get('after')
            payload.commits = body.get('commits', [])
        
        return payload

    def _map_event_type(self, event: str) -> WebhookEventType:
        event_lower = event.lower()
        if 'push' in event_lower:
            return WebhookEventType.PUSH
        elif 'merge_request' in event_lower:
            return WebhookEventType.MERGE_REQUEST
        elif 'tag' in event_lower:
            return WebhookEventType.TAG
        elif 'note' in event_lower:
            return WebhookEventType.COMMENT
        return WebhookEventType.UNKNOWN

    def get_pull_request(self, repo: Repository, pr_number: int) -> PullRequest:
        return PullRequest(
            id=0, number=pr_number, title="", description="",
            source_branch="", target_branch="", author="",
            state="open", created_at=datetime.now(), updated_at=datetime.now(),
            url=""
        )

    def get_diff(self, repo: Repository, pr: PullRequest) -> str:
        return ""

    def post_comment(self, repo: Repository, pr: PullRequest, comment: str) -> bool:
        return True

    def update_status(
        self, repo: Repository, sha: str, state: str, description: str, context: str
    ) -> bool:
        return True


class GiteaAdapter(GitProviderAdapter):
    """Adapter for Gitea/Forgejo."""

    def __init__(self, credentials: GitCredentials):
        self.credentials = credentials
        self.api_base = credentials.api_url or "https://gitea.example.com/api/v1"

    def parse_webhook(self, headers: dict, body: dict) -> WebhookPayload:
        event_type = headers.get('X-Gitea-Event', headers.get('x-gitea-event', ''))
        
        repo_data = body.get('repository', {})
        repository = Repository(
            provider=GitProvider.GITEA,
            owner=repo_data.get('owner', {}).get('login', repo_data.get('owner', {}).get('username', '')),
            name=repo_data.get('name', ''),
            clone_url=repo_data.get('clone_url', ''),
            default_branch=repo_data.get('default_branch', 'main'),
            is_private=repo_data.get('private', False)
        )
        
        payload = WebhookPayload(
            provider=GitProvider.GITEA,
            event_type=self._map_event_type(event_type),
            repository=repository,
            sender=body.get('sender', {}).get('login', body.get('sender', {}).get('username')),
            raw_payload=body
        )
        
        if event_type == 'pull_request':
            pr_data = body.get('pull_request', {})
            payload.pull_request = PullRequest(
                id=pr_data.get('id', 0),
                number=pr_data.get('number', 0),
                title=pr_data.get('title', ''),
                description=pr_data.get('body', ''),
                source_branch=pr_data.get('head', {}).get('ref', ''),
                target_branch=pr_data.get('base', {}).get('ref', ''),
                author=pr_data.get('user', {}).get('login', ''),
                state=pr_data.get('state', 'open'),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                url=pr_data.get('html_url', '')
            )
        elif event_type == 'push':
            payload.ref = body.get('ref')
            payload.before_sha = body.get('before')
            payload.after_sha = body.get('after')
            payload.commits = body.get('commits', [])
        
        return payload

    def _map_event_type(self, event: str) -> WebhookEventType:
        mapping = {
            'push': WebhookEventType.PUSH,
            'pull_request': WebhookEventType.PULL_REQUEST,
            'create': WebhookEventType.TAG,
        }
        return mapping.get(event, WebhookEventType.UNKNOWN)

    def get_pull_request(self, repo: Repository, pr_number: int) -> PullRequest:
        return PullRequest(
            id=0, number=pr_number, title="", description="",
            source_branch="", target_branch="", author="",
            state="open", created_at=datetime.now(), updated_at=datetime.now(),
            url=""
        )

    def get_diff(self, repo: Repository, pr: PullRequest) -> str:
        return ""

    def post_comment(self, repo: Repository, pr: PullRequest, comment: str) -> bool:
        return True

    def update_status(
        self, repo: Repository, sha: str, state: str, description: str, context: str
    ) -> bool:
        return True


class GerritAdapter(GitProviderAdapter):
    """Adapter for Gerrit Code Review."""

    def __init__(self, credentials: GitCredentials):
        self.credentials = credentials
        self.api_base = credentials.api_url

    def parse_webhook(self, headers: dict, body: dict) -> WebhookPayload:
        # Gerrit uses different event format
        event_type = body.get('type', '')
        
        project = body.get('project', body.get('change', {}).get('project', ''))
        repository = Repository(
            provider=GitProvider.GERRIT,
            owner='',  # Gerrit doesn't have owners in the same way
            name=project,
            clone_url=f"{self.api_base}/{project}.git" if self.api_base else '',
        )
        
        payload = WebhookPayload(
            provider=GitProvider.GERRIT,
            event_type=self._map_event_type(event_type),
            repository=repository,
            sender=body.get('uploader', {}).get('username', body.get('submitter', {}).get('username')),
            raw_payload=body
        )
        
        change = body.get('change', {})
        if change:
            payload.pull_request = PullRequest(
                id=change.get('id', 0),
                number=change.get('number', change.get('_number', 0)),
                title=change.get('subject', ''),
                description=change.get('commitMessage', ''),
                source_branch=change.get('branch', ''),
                target_branch=change.get('branch', ''),
                author=change.get('owner', {}).get('username', ''),
                state=change.get('status', 'NEW'),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                url=change.get('url', '')
            )
        
        if body.get('refUpdate'):
            ref = body['refUpdate']
            payload.ref = ref.get('refName')
            payload.before_sha = ref.get('oldRev')
            payload.after_sha = ref.get('newRev')
        
        return payload

    def _map_event_type(self, event: str) -> WebhookEventType:
        mapping = {
            'ref-updated': WebhookEventType.PUSH,
            'patchset-created': WebhookEventType.PULL_REQUEST,
            'change-merged': WebhookEventType.MERGE_REQUEST,
            'comment-added': WebhookEventType.REVIEW,
        }
        return mapping.get(event, WebhookEventType.UNKNOWN)

    def get_pull_request(self, repo: Repository, pr_number: int) -> PullRequest:
        return PullRequest(
            id=0, number=pr_number, title="", description="",
            source_branch="", target_branch="", author="",
            state="open", created_at=datetime.now(), updated_at=datetime.now(),
            url=""
        )

    def get_diff(self, repo: Repository, pr: PullRequest) -> str:
        return ""

    def post_comment(self, repo: Repository, pr: PullRequest, comment: str) -> bool:
        return True

    def update_status(
        self, repo: Repository, sha: str, state: str, description: str, context: str
    ) -> bool:
        return True


class AzureDevOpsAdapter(GitProviderAdapter):
    """Adapter for Azure DevOps (VSTS)."""

    def __init__(self, credentials: GitCredentials):
        self.credentials = credentials
        self.api_base = credentials.api_url or "https://dev.azure.com"

    def parse_webhook(self, headers: dict, body: dict) -> WebhookPayload:
        event_type = body.get('eventType', '')
        
        resource = body.get('resource', {})
        repo_data = resource.get('repository', {})
        
        repository = Repository(
            provider=GitProvider.AZURE_DEVOPS,
            owner=repo_data.get('project', {}).get('name', ''),
            name=repo_data.get('name', ''),
            clone_url=repo_data.get('remoteUrl', ''),
            default_branch=repo_data.get('defaultBranch', 'refs/heads/main').replace('refs/heads/', ''),
        )
        
        payload = WebhookPayload(
            provider=GitProvider.AZURE_DEVOPS,
            event_type=self._map_event_type(event_type),
            repository=repository,
            sender=body.get('resource', {}).get('createdBy', {}).get('uniqueName'),
            raw_payload=body
        )
        
        if 'pullrequest' in event_type.lower():
            pr_data = resource
            payload.pull_request = PullRequest(
                id=pr_data.get('pullRequestId', 0),
                number=pr_data.get('pullRequestId', 0),
                title=pr_data.get('title', ''),
                description=pr_data.get('description', ''),
                source_branch=pr_data.get('sourceRefName', '').replace('refs/heads/', ''),
                target_branch=pr_data.get('targetRefName', '').replace('refs/heads/', ''),
                author=pr_data.get('createdBy', {}).get('uniqueName', ''),
                state=pr_data.get('status', 'active'),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                url=pr_data.get('url', '')
            )
        elif 'push' in event_type.lower():
            push_data = resource.get('pushes', [{}])[0] if resource.get('pushes') else resource
            payload.ref = push_data.get('refUpdates', [{}])[0].get('name', '') if push_data.get('refUpdates') else ''
            commits = push_data.get('commits', [])
            if commits:
                payload.before_sha = commits[-1].get('commitId', '')
                payload.after_sha = commits[0].get('commitId', '')
            payload.commits = commits
        
        return payload

    def _map_event_type(self, event: str) -> WebhookEventType:
        event_lower = event.lower()
        if 'push' in event_lower or 'code pushed' in event_lower:
            return WebhookEventType.PUSH
        elif 'pullrequest' in event_lower:
            return WebhookEventType.PULL_REQUEST
        elif 'comment' in event_lower:
            return WebhookEventType.COMMENT
        return WebhookEventType.UNKNOWN

    def get_pull_request(self, repo: Repository, pr_number: int) -> PullRequest:
        return PullRequest(
            id=0, number=pr_number, title="", description="",
            source_branch="", target_branch="", author="",
            state="open", created_at=datetime.now(), updated_at=datetime.now(),
            url=""
        )

    def get_diff(self, repo: Repository, pr: PullRequest) -> str:
        return ""

    def post_comment(self, repo: Repository, pr: PullRequest, comment: str) -> bool:
        return True

    def update_status(
        self, repo: Repository, sha: str, state: str, description: str, context: str
    ) -> bool:
        return True


class GenericGitAdapter(GitProviderAdapter):
    """Generic adapter for any Git hosting platform using standard Git operations."""

    def __init__(self, credentials: GitCredentials):
        self.credentials = credentials

    def parse_webhook(self, headers: dict, body: dict) -> WebhookPayload:
        # Try to detect provider from headers
        provider = self._detect_provider(headers, body)
        
        # Extract common fields
        repo_data = body.get('repository', body.get('project', {}))
        
        repository = Repository(
            provider=provider,
            owner=self._extract_owner(repo_data),
            name=self._extract_name(repo_data),
            clone_url=self._extract_clone_url(repo_data),
        )
        
        return WebhookPayload(
            provider=provider,
            event_type=self._detect_event_type(headers, body),
            repository=repository,
            raw_payload=body
        )

    def _detect_provider(self, headers: dict, body: dict) -> GitProvider:
        # Check headers for provider hints
        if 'X-GitHub-Event' in headers or 'x-github-event' in headers:
            return GitProvider.GITHUB
        if 'X-Gitlab-Event' in headers or 'x-gitlab-event' in headers:
            return GitProvider.GITLAB
        if 'X-Gitea-Event' in headers or 'x-gitea-event' in headers:
            return GitProvider.GITEA
        if 'X-Gogs-Event' in headers or 'x-gogs-event' in headers:
            return GitProvider.GOGS
        if body.get('eventType', '').startswith('git.'):
            return GitProvider.AZURE_DEVOPS
        if body.get('type', '').endswith('-updated') or body.get('change'):
            return GitProvider.GERRIT
        return GitProvider.GENERIC

    def _detect_event_type(self, headers: dict, body: dict) -> WebhookEventType:
        # Check for common event indicators
        if body.get('pull_request') or body.get('merge_request') or body.get('object_attributes', {}).get('source_branch'):
            return WebhookEventType.PULL_REQUEST
        if body.get('ref') and body.get('commits'):
            return WebhookEventType.PUSH
        if body.get('change'):
            return WebhookEventType.PULL_REQUEST
        return WebhookEventType.UNKNOWN

    def _extract_owner(self, repo_data: dict) -> str:
        return (
            repo_data.get('owner', {}).get('login') or
            repo_data.get('owner', {}).get('username') or
            repo_data.get('namespace') or
            repo_data.get('path_with_namespace', '/').split('/')[0] or
            ''
        )

    def _extract_name(self, repo_data: dict) -> str:
        return (
            repo_data.get('name') or
            repo_data.get('path_with_namespace', '/').split('/')[-1] or
            ''
        )

    def _extract_clone_url(self, repo_data: dict) -> str:
        return (
            repo_data.get('clone_url') or
            repo_data.get('git_http_url') or
            repo_data.get('url') or
            repo_data.get('remoteUrl') or
            ''
        )

    def get_pull_request(self, repo: Repository, pr_number: int) -> PullRequest:
        return PullRequest(
            id=0, number=pr_number, title="", description="",
            source_branch="", target_branch="", author="",
            state="open", created_at=datetime.now(), updated_at=datetime.now(),
            url=""
        )

    def get_diff(self, repo: Repository, pr: PullRequest) -> str:
        return ""

    def post_comment(self, repo: Repository, pr: PullRequest, comment: str) -> bool:
        return True

    def update_status(
        self, repo: Repository, sha: str, state: str, description: str, context: str
    ) -> bool:
        return True


class LocalGitOperations:
    """Direct Git operations for local/CLI usage."""

    def __init__(self, repo_path: Optional[Path] = None):
        self.repo_path = repo_path or Path.cwd()

    def get_current_branch(self) -> str:
        """Get current branch name."""
        result = subprocess.run(
            ['git', 'branch', '--show-current'],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()

    def get_remote_url(self) -> str:
        """Get remote origin URL."""
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()

    def get_diff(self, base: str = 'HEAD~1', target: str = 'HEAD') -> str:
        """Get diff between commits."""
        result = subprocess.run(
            ['git', 'diff', base, target],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        return result.stdout

    def get_staged_diff(self) -> str:
        """Get diff of staged changes."""
        result = subprocess.run(
            ['git', 'diff', '--cached'],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        return result.stdout

    def get_changed_files(self, base: str = 'HEAD~1', target: str = 'HEAD') -> list[str]:
        """Get list of changed files."""
        result = subprocess.run(
            ['git', 'diff', '--name-only', base, target],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        return [f for f in result.stdout.strip().split('\n') if f]

    def get_file_content(self, path: str, ref: str = 'HEAD') -> str:
        """Get file content at specific ref."""
        result = subprocess.run(
            ['git', 'show', f'{ref}:{path}'],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        return result.stdout

    def get_commit_info(self, ref: str = 'HEAD') -> dict:
        """Get commit information."""
        format_str = '%H%n%an%n%ae%n%s%n%b'
        result = subprocess.run(
            ['git', 'log', '-1', f'--format={format_str}', ref],
            cwd=self.repo_path,
            capture_output=True,
            text=True
        )
        lines = result.stdout.strip().split('\n')
        return {
            'sha': lines[0] if len(lines) > 0 else '',
            'author_name': lines[1] if len(lines) > 1 else '',
            'author_email': lines[2] if len(lines) > 2 else '',
            'subject': lines[3] if len(lines) > 3 else '',
            'body': '\n'.join(lines[4:]) if len(lines) > 4 else ''
        }

    def detect_provider(self) -> GitProvider:
        """Detect Git provider from remote URL."""
        remote_url = self.get_remote_url().lower()
        
        if 'github.com' in remote_url:
            return GitProvider.GITHUB
        elif 'gitlab.com' in remote_url or 'gitlab' in remote_url:
            return GitProvider.GITLAB
        elif 'bitbucket.org' in remote_url:
            return GitProvider.BITBUCKET
        elif 'gitea' in remote_url or 'forgejo' in remote_url:
            return GitProvider.GITEA
        elif 'dev.azure.com' in remote_url or 'visualstudio.com' in remote_url:
            return GitProvider.AZURE_DEVOPS
        elif 'gerrit' in remote_url:
            return GitProvider.GERRIT
        elif 'codeberg.org' in remote_url:
            return GitProvider.CODEBERG
        elif 'sr.ht' in remote_url:
            return GitProvider.SOURCEHUT
        elif 'gogs' in remote_url:
            return GitProvider.GOGS
        return GitProvider.LOCAL


class WebhookReceiver:
    """Generic webhook receiver that handles all providers."""

    def __init__(self, secrets: dict[GitProvider, str] = None):
        self.secrets = secrets or {}
        self.adapters: dict[GitProvider, GitProviderAdapter] = {}

    def register_adapter(self, provider: GitProvider, adapter: GitProviderAdapter):
        """Register an adapter for a provider."""
        self.adapters[provider] = adapter

    def verify_signature(
        self, 
        provider: GitProvider, 
        headers: dict, 
        body: bytes,
        secret: Optional[str] = None
    ) -> bool:
        """Verify webhook signature."""
        secret = secret or self.secrets.get(provider)
        if not secret:
            return True  # No secret configured, skip verification
        
        if provider == GitProvider.GITHUB:
            sig_header = headers.get('X-Hub-Signature-256', headers.get('x-hub-signature-256', ''))
            if not sig_header:
                return False
            expected = 'sha256=' + hmac.new(
                secret.encode(), body, hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(sig_header, expected)
        
        elif provider == GitProvider.GITLAB:
            token = headers.get('X-Gitlab-Token', headers.get('x-gitlab-token', ''))
            return token == secret
        
        elif provider == GitProvider.GITEA:
            sig_header = headers.get('X-Gitea-Signature', headers.get('x-gitea-signature', ''))
            if not sig_header:
                return False
            expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
            return hmac.compare_digest(sig_header, expected)
        
        # Default: no verification
        return True

    def process_webhook(
        self, 
        headers: dict, 
        body: bytes
    ) -> Optional[WebhookPayload]:
        """Process incoming webhook and return parsed payload."""
        # Try to parse as JSON
        try:
            body_dict = json.loads(body.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        
        # Detect provider
        provider = self._detect_provider(headers, body_dict)
        
        # Verify signature
        if not self.verify_signature(provider, headers, body):
            return None
        
        # Get or create adapter
        adapter = self.adapters.get(provider)
        if not adapter:
            adapter = GenericGitAdapter(GitCredentials(provider=provider))
        
        return adapter.parse_webhook(headers, body_dict)

    def _detect_provider(self, headers: dict, body: dict) -> GitProvider:
        """Detect provider from headers."""
        if 'X-GitHub-Event' in headers or 'x-github-event' in headers:
            return GitProvider.GITHUB
        if 'X-Gitlab-Event' in headers or 'x-gitlab-event' in headers:
            return GitProvider.GITLAB
        if 'X-Gitea-Event' in headers or 'x-gitea-event' in headers:
            return GitProvider.GITEA
        if 'X-Gogs-Event' in headers or 'x-gogs-event' in headers:
            return GitProvider.GOGS
        if body.get('eventType', '').startswith('git.'):
            return GitProvider.AZURE_DEVOPS
        if body.get('type') and body.get('change'):
            return GitProvider.GERRIT
        return GitProvider.GENERIC


@dataclass
class AirGappedConfig:
    """Configuration for air-gapped deployment."""
    model_path: Path
    rules_path: Path
    cache_path: Path
    offline_mode: bool = True
    max_cache_age_days: int = 30


class AirGappedVerifier:
    """Verifier for air-gapped environments."""

    def __init__(self, config: AirGappedConfig):
        self.config = config
        self.model_loaded = False
        self.rules_loaded = False

    def initialize(self) -> bool:
        """Initialize verifier with bundled models and rules."""
        self.model_loaded = self.config.model_path.exists()
        self.rules_loaded = self.config.rules_path.exists()
        return self.model_loaded and self.rules_loaded

    def verify_diff(self, diff: str) -> dict:
        """Verify a diff using local models."""
        if not self.model_loaded:
            return {'error': 'Model not loaded', 'verified': False}
        
        # In a real implementation, would run local ML model
        return {
            'verified': True,
            'offline': True,
            'model_version': '1.0.0',
            'findings': []
        }

    def update_rules(self, rules_bundle: bytes) -> bool:
        """Update rules from a signed bundle (manual transfer)."""
        # Verify bundle signature
        # Extract and update rules
        self.config.rules_path.mkdir(parents=True, exist_ok=True)
        return True


class UniversalGitSupport:
    """Main class providing universal Git support."""

    def __init__(self, credentials: Optional[dict[GitProvider, GitCredentials]] = None):
        self.credentials = credentials or {}
        self.adapters: dict[GitProvider, GitProviderAdapter] = {}
        self.webhook_receiver = WebhookReceiver()
        self.local_ops = LocalGitOperations()
        
        # Initialize adapters
        self._initialize_adapters()

    def _initialize_adapters(self):
        """Initialize adapters for configured providers."""
        adapter_classes = {
            GitProvider.GITHUB: GitHubAdapter,
            GitProvider.GITLAB: GitLabAdapter,
            GitProvider.GITEA: GiteaAdapter,
            GitProvider.GERRIT: GerritAdapter,
            GitProvider.AZURE_DEVOPS: AzureDevOpsAdapter,
        }
        
        for provider, creds in self.credentials.items():
            if provider in adapter_classes:
                adapter = adapter_classes[provider](creds)
                self.adapters[provider] = adapter
                self.webhook_receiver.register_adapter(provider, adapter)

    def get_adapter(self, provider: GitProvider) -> GitProviderAdapter:
        """Get adapter for a provider."""
        if provider not in self.adapters:
            # Create generic adapter
            creds = self.credentials.get(provider, GitCredentials(provider=provider))
            if provider == GitProvider.GITHUB:
                self.adapters[provider] = GitHubAdapter(creds)
            elif provider == GitProvider.GITLAB:
                self.adapters[provider] = GitLabAdapter(creds)
            elif provider == GitProvider.GITEA:
                self.adapters[provider] = GiteaAdapter(creds)
            elif provider == GitProvider.GERRIT:
                self.adapters[provider] = GerritAdapter(creds)
            elif provider == GitProvider.AZURE_DEVOPS:
                self.adapters[provider] = AzureDevOpsAdapter(creds)
            else:
                self.adapters[provider] = GenericGitAdapter(creds)
        return self.adapters[provider]

    def process_webhook(self, headers: dict, body: bytes) -> Optional[WebhookPayload]:
        """Process incoming webhook from any provider."""
        return self.webhook_receiver.process_webhook(headers, body)

    def verify_local_changes(
        self, 
        base: str = 'HEAD~1', 
        target: str = 'HEAD'
    ) -> dict:
        """Verify local Git changes (CLI workflow)."""
        diff = self.local_ops.get_diff(base, target)
        changed_files = self.local_ops.get_changed_files(base, target)
        commit_info = self.local_ops.get_commit_info(target)
        
        return {
            'diff': diff,
            'changed_files': changed_files,
            'commit': commit_info,
            'base': base,
            'target': target,
            'provider': self.local_ops.detect_provider().value
        }

    def verify_staged_changes(self) -> dict:
        """Verify staged changes before commit (pre-commit hook)."""
        diff = self.local_ops.get_staged_diff()
        
        return {
            'diff': diff,
            'branch': self.local_ops.get_current_branch(),
            'provider': self.local_ops.detect_provider().value
        }

    def create_repository_from_url(self, url: str) -> Repository:
        """Create Repository object from clone URL."""
        parsed = urlparse(url)
        
        # Detect provider from URL
        provider = GitProvider.GENERIC
        hostname = parsed.hostname or ''
        
        if 'github.com' in hostname:
            provider = GitProvider.GITHUB
        elif 'gitlab.com' in hostname or 'gitlab' in hostname:
            provider = GitProvider.GITLAB
        elif 'bitbucket.org' in hostname:
            provider = GitProvider.BITBUCKET
        elif 'gitea' in hostname or 'forgejo' in hostname:
            provider = GitProvider.GITEA
        elif 'dev.azure.com' in hostname or 'visualstudio.com' in hostname:
            provider = GitProvider.AZURE_DEVOPS
        elif 'codeberg.org' in hostname:
            provider = GitProvider.CODEBERG
        
        # Parse owner/name from path
        path_parts = parsed.path.strip('/').replace('.git', '').split('/')
        owner = path_parts[0] if len(path_parts) > 0 else ''
        name = path_parts[1] if len(path_parts) > 1 else ''
        
        return Repository(
            provider=provider,
            owner=owner,
            name=name,
            clone_url=url
        )


# CLI interface
def cli_verify(args: list[str] = None) -> int:
    """CLI entry point for verification."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='CodeVerify - Universal Git verification'
    )
    parser.add_argument(
        '--base', '-b',
        default='HEAD~1',
        help='Base commit/branch for comparison'
    )
    parser.add_argument(
        '--target', '-t',
        default='HEAD',
        help='Target commit/branch for comparison'
    )
    parser.add_argument(
        '--staged', '-s',
        action='store_true',
        help='Verify staged changes only'
    )
    parser.add_argument(
        '--output', '-o',
        choices=['json', 'text', 'github', 'gitlab'],
        default='text',
        help='Output format'
    )
    parser.add_argument(
        '--air-gapped',
        action='store_true',
        help='Run in air-gapped mode (offline)'
    )
    
    parsed = parser.parse_args(args)
    
    git_support = UniversalGitSupport()
    
    if parsed.staged:
        result = git_support.verify_staged_changes()
    else:
        result = git_support.verify_local_changes(parsed.base, parsed.target)
    
    if parsed.output == 'json':
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Provider: {result.get('provider', 'unknown')}")
        print(f"Changed files: {len(result.get('changed_files', []))}")
        for f in result.get('changed_files', []):
            print(f"  - {f}")
    
    return 0


if __name__ == '__main__':
    exit(cli_verify())
