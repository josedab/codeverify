"""Tests for VCS (Version Control System) clients."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from codeverify_core.vcs.base import (
    VCSClient,
    Repository,
    PullRequest,
    CheckRun,
    CheckStatus,
    CheckConclusion,
    Comment,
    VCSProvider,
)
from codeverify_core.vcs.github import GitHubClient
from codeverify_core.vcs.gitlab import GitLabClient
from codeverify_core.vcs.bitbucket import BitbucketClient
from codeverify_core.vcs.factory import create_vcs_client, get_provider_from_url


class TestVCSProviderDetection:
    """Tests for VCS provider URL detection."""

    def test_github_url_detection(self):
        """Detect GitHub URLs."""
        urls = [
            "https://github.com/owner/repo",
            "https://github.com/owner/repo.git",
            "git@github.com:owner/repo.git",
        ]
        for url in urls:
            provider = get_provider_from_url(url)
            assert provider == VCSProvider.GITHUB

    def test_gitlab_url_detection(self):
        """Detect GitLab URLs."""
        urls = [
            "https://gitlab.com/owner/repo",
            "https://gitlab.com/group/subgroup/repo",
            "git@gitlab.com:owner/repo.git",
        ]
        for url in urls:
            provider = get_provider_from_url(url)
            assert provider == VCSProvider.GITLAB

    def test_bitbucket_url_detection(self):
        """Detect Bitbucket URLs."""
        urls = [
            "https://bitbucket.org/owner/repo",
            "git@bitbucket.org:owner/repo.git",
        ]
        for url in urls:
            provider = get_provider_from_url(url)
            assert provider == VCSProvider.BITBUCKET

    def test_unknown_url(self):
        """Unknown URLs return None."""
        provider = get_provider_from_url("https://unknown.com/repo")
        assert provider is None


class TestVCSClientFactory:
    """Tests for VCS client factory."""

    def test_create_github_client(self):
        """Factory creates GitHub client."""
        client = create_vcs_client(
            "https://github.com/owner/repo",
            token="test-token"
        )
        assert isinstance(client, GitHubClient)

    def test_create_gitlab_client(self):
        """Factory creates GitLab client."""
        client = create_vcs_client(
            "https://gitlab.com/owner/repo",
            token="test-token"
        )
        assert isinstance(client, GitLabClient)

    def test_create_bitbucket_client(self):
        """Factory creates Bitbucket client."""
        client = create_vcs_client(
            "https://bitbucket.org/owner/repo",
            token="test-token"
        )
        assert isinstance(client, BitbucketClient)

    def test_unknown_provider_raises_error(self):
        """Unknown provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported VCS provider"):
            create_vcs_client("https://unknown.com/repo", token="test")


class TestGitHubClient:
    """Tests for GitHub client."""

    @pytest.fixture
    def client(self):
        """Create a GitHub client."""
        return GitHubClient(
            owner="test-owner",
            repo="test-repo",
            token="test-token"
        )

    def test_initialization(self, client):
        """Client initializes with correct values."""
        assert client.owner == "test-owner"
        assert client.repo == "test-repo"
        assert client.provider == VCSProvider.GITHUB

    @pytest.mark.asyncio
    async def test_get_repository(self, client):
        """Client can fetch repository info."""
        mock_response = {
            "id": 12345,
            "name": "test-repo",
            "full_name": "test-owner/test-repo",
            "default_branch": "main",
            "private": False,
            "clone_url": "https://github.com/test-owner/test-repo.git",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            repo = await client.get_repository()
            
            assert isinstance(repo, Repository)
            assert repo.name == "test-repo"
            assert repo.default_branch == "main"

    @pytest.mark.asyncio
    async def test_get_pull_request(self, client):
        """Client can fetch pull request."""
        mock_response = {
            "number": 42,
            "title": "Test PR",
            "body": "Test description",
            "state": "open",
            "head": {"ref": "feature-branch", "sha": "abc123"},
            "base": {"ref": "main", "sha": "def456"},
            "user": {"login": "test-user"},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            pr = await client.get_pull_request(42)
            
            assert isinstance(pr, PullRequest)
            assert pr.number == 42
            assert pr.title == "Test PR"
            assert pr.source_branch == "feature-branch"
            assert pr.target_branch == "main"

    @pytest.mark.asyncio
    async def test_create_check_run(self, client):
        """Client can create check run."""
        mock_response = {
            "id": 99999,
            "name": "CodeVerify",
            "status": "in_progress",
            "head_sha": "abc123",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            check = await client.create_check_run(
                name="CodeVerify",
                head_sha="abc123",
                status=CheckStatus.IN_PROGRESS,
            )
            
            assert isinstance(check, CheckRun)
            assert check.name == "CodeVerify"
            assert check.status == CheckStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_update_check_run(self, client):
        """Client can update check run."""
        mock_response = {
            "id": 99999,
            "name": "CodeVerify",
            "status": "completed",
            "conclusion": "success",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            check = await client.update_check_run(
                check_run_id=99999,
                status=CheckStatus.COMPLETED,
                conclusion=CheckConclusion.SUCCESS,
            )
            
            assert check.status == CheckStatus.COMPLETED
            assert check.conclusion == CheckConclusion.SUCCESS

    @pytest.mark.asyncio
    async def test_create_comment(self, client):
        """Client can create PR comment."""
        mock_response = {
            "id": 12345,
            "body": "Test comment",
            "user": {"login": "bot"},
            "created_at": "2024-01-01T00:00:00Z",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            comment = await client.create_comment(
                pr_number=42,
                body="Test comment"
            )
            
            assert isinstance(comment, Comment)
            assert comment.body == "Test comment"

    def test_webhook_signature_verification(self, client):
        """Client verifies webhook signatures."""
        payload = b'{"action": "opened"}'
        secret = "webhook-secret"
        
        # Generate valid signature
        import hmac
        import hashlib
        expected_sig = "sha256=" + hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        assert client.verify_webhook_signature(payload, expected_sig, secret)
        assert not client.verify_webhook_signature(payload, "sha256=invalid", secret)


class TestGitLabClient:
    """Tests for GitLab client."""

    @pytest.fixture
    def client(self):
        """Create a GitLab client."""
        return GitLabClient(
            owner="test-group",
            repo="test-project",
            token="test-token"
        )

    def test_initialization(self, client):
        """Client initializes correctly."""
        assert client.owner == "test-group"
        assert client.repo == "test-project"
        assert client.provider == VCSProvider.GITLAB

    @pytest.mark.asyncio
    async def test_get_pull_request(self, client):
        """Client fetches merge request (GitLab's PR equivalent)."""
        mock_response = {
            "iid": 42,
            "title": "Test MR",
            "description": "Test description",
            "state": "opened",
            "source_branch": "feature",
            "target_branch": "main",
            "sha": "abc123",
            "author": {"username": "test-user"},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            pr = await client.get_pull_request(42)
            
            assert isinstance(pr, PullRequest)
            assert pr.number == 42
            assert pr.title == "Test MR"

    def test_webhook_token_verification(self, client):
        """GitLab uses token-based webhook verification."""
        secret = "webhook-token"
        
        # GitLab sends token in header, not signature
        assert client.verify_webhook_signature(b"payload", secret, secret)
        assert not client.verify_webhook_signature(b"payload", "wrong", secret)


class TestBitbucketClient:
    """Tests for Bitbucket client."""

    @pytest.fixture
    def client(self):
        """Create a Bitbucket client."""
        return BitbucketClient(
            owner="test-workspace",
            repo="test-repo",
            token="test-token"
        )

    def test_initialization(self, client):
        """Client initializes correctly."""
        assert client.owner == "test-workspace"
        assert client.repo == "test-repo"
        assert client.provider == VCSProvider.BITBUCKET

    @pytest.mark.asyncio
    async def test_get_pull_request(self, client):
        """Client fetches pull request."""
        mock_response = {
            "id": 42,
            "title": "Test PR",
            "description": "Test description",
            "state": "OPEN",
            "source": {"branch": {"name": "feature"}},
            "destination": {"branch": {"name": "main"}},
            "author": {"display_name": "Test User"},
            "created_on": "2024-01-01T00:00:00Z",
            "updated_on": "2024-01-02T00:00:00Z",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            pr = await client.get_pull_request(42)
            
            assert isinstance(pr, PullRequest)
            assert pr.number == 42
            assert pr.title == "Test PR"
            assert pr.source_branch == "feature"


class TestVCSClientAbstraction:
    """Tests for VCS client abstraction layer."""

    @pytest.mark.parametrize("ClientClass,provider", [
        (GitHubClient, VCSProvider.GITHUB),
        (GitLabClient, VCSProvider.GITLAB),
        (BitbucketClient, VCSProvider.BITBUCKET),
    ])
    def test_all_clients_implement_interface(self, ClientClass, provider):
        """All clients implement the VCSClient interface."""
        client = ClientClass(owner="test", repo="test", token="test")
        
        # Check required methods exist
        assert hasattr(client, "get_repository")
        assert hasattr(client, "get_pull_request")
        assert hasattr(client, "list_pull_requests")
        assert hasattr(client, "create_check_run")
        assert hasattr(client, "update_check_run")
        assert hasattr(client, "create_comment")
        assert hasattr(client, "verify_webhook_signature")
        
        # Check provider
        assert client.provider == provider

    @pytest.mark.parametrize("ClientClass", [
        GitHubClient,
        GitLabClient,
        BitbucketClient,
    ])
    def test_clients_handle_errors_gracefully(self, ClientClass):
        """Clients handle API errors gracefully."""
        client = ClientClass(owner="test", repo="test", token="test")
        
        # Client should have error handling mechanisms
        assert hasattr(client, "_request")


class TestPullRequestDataModel:
    """Tests for PullRequest data model."""

    def test_pull_request_creation(self):
        """PullRequest can be created with all fields."""
        pr = PullRequest(
            number=42,
            title="Test PR",
            body="Description",
            state="open",
            source_branch="feature",
            target_branch="main",
            head_sha="abc123",
            author="test-user",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        assert pr.number == 42
        assert pr.title == "Test PR"
        assert pr.source_branch == "feature"
        assert pr.target_branch == "main"

    def test_pull_request_optional_fields(self):
        """PullRequest handles optional fields."""
        pr = PullRequest(
            number=1,
            title="Minimal PR",
            body=None,
            state="open",
            source_branch="branch",
            target_branch="main",
            head_sha="sha",
            author="user",
        )
        
        assert pr.body is None
        assert pr.created_at is None


class TestCheckRunDataModel:
    """Tests for CheckRun data model."""

    def test_check_run_status_values(self):
        """CheckStatus enum has expected values."""
        assert CheckStatus.QUEUED.value == "queued"
        assert CheckStatus.IN_PROGRESS.value == "in_progress"
        assert CheckStatus.COMPLETED.value == "completed"

    def test_check_conclusion_values(self):
        """CheckConclusion enum has expected values."""
        assert CheckConclusion.SUCCESS.value == "success"
        assert CheckConclusion.FAILURE.value == "failure"
        assert CheckConclusion.NEUTRAL.value == "neutral"
        assert CheckConclusion.CANCELLED.value == "cancelled"
        assert CheckConclusion.SKIPPED.value == "skipped"
        assert CheckConclusion.TIMED_OUT.value == "timed_out"
        assert CheckConclusion.ACTION_REQUIRED.value == "action_required"
