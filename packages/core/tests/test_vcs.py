"""Tests for VCS (Version Control System) clients."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codeverify_core.vcs.base import (
    CheckConclusion,
    CheckRun,
    CheckStatus,
    PullRequest,
    PullRequestComment,
    Repository,
    User,
    VCSClient,
    VCSConfig,
)
from codeverify_core.vcs.bitbucket import BitbucketClient
from codeverify_core.vcs.factory import create_vcs_client, get_provider_from_url
from codeverify_core.vcs.github import GitHubClient
from codeverify_core.vcs.gitlab import GitLabClient


def _make_config(provider: str, token: str = "test-token", **kwargs) -> VCSConfig:
    """Helper to create a VCSConfig."""
    return VCSConfig(provider=provider, token=token, **kwargs)


class TestVCSProviderDetection:
    """Tests for VCS provider URL detection."""

    def test_github_url_detection(self):
        """Detect GitHub URLs."""
        urls = [
            "https://github.com/owner/repo",
            "https://github.com/owner/repo.git",
        ]
        for url in urls:
            provider = get_provider_from_url(url)
            assert provider == "github"

    def test_gitlab_url_detection(self):
        """Detect GitLab URLs."""
        urls = [
            "https://gitlab.com/owner/repo",
            "https://gitlab.com/group/subgroup/repo",
        ]
        for url in urls:
            provider = get_provider_from_url(url)
            assert provider == "gitlab"

    def test_bitbucket_url_detection(self):
        """Detect Bitbucket URLs."""
        urls = [
            "https://bitbucket.org/owner/repo",
        ]
        for url in urls:
            provider = get_provider_from_url(url)
            assert provider == "bitbucket"

    def test_unknown_url(self):
        """Unknown URLs raise ValueError."""
        with pytest.raises(ValueError, match="Cannot determine VCS provider"):
            get_provider_from_url("https://unknown.com/repo")


class TestVCSClientFactory:
    """Tests for VCS client factory."""

    def test_create_github_client(self):
        """Factory creates GitHub client."""
        client = create_vcs_client(url="https://github.com/owner/repo", token="test-token")
        assert isinstance(client, GitHubClient)

    def test_create_gitlab_client(self):
        """Factory creates GitLab client."""
        client = create_vcs_client(url="https://gitlab.com/owner/repo", token="test-token")
        assert isinstance(client, GitLabClient)

    def test_create_bitbucket_client(self):
        """Factory creates Bitbucket client."""
        client = create_vcs_client(url="https://bitbucket.org/owner/repo", token="test-token")
        assert isinstance(client, BitbucketClient)

    def test_unknown_provider_raises_error(self):
        """Unknown provider raises ValueError."""
        with pytest.raises(ValueError):
            create_vcs_client(provider="unknown", token="test")


class TestGitHubClient:
    """Tests for GitHub client."""

    @pytest.fixture
    def client(self):
        """Create a GitHub client."""
        config = _make_config("github", webhook_secret="webhook-secret")
        return GitHubClient(config)

    def test_initialization(self, client):
        """Client initializes with correct values."""
        assert client.provider_name == "github"
        assert client.config.provider == "github"

    @pytest.mark.asyncio
    async def test_get_repository(self, client):
        """Client can fetch repository info."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "id": 12345,
            "name": "test-repo",
            "full_name": "test-owner/test-repo",
            "owner": {"login": "test-owner"},
            "default_branch": "main",
            "private": False,
            "clone_url": "https://github.com/test-owner/test-repo.git",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_resp

            repo = await client.get_repository("test-owner/test-repo")

            assert isinstance(repo, Repository)
            assert repo.name == "test-repo"
            assert repo.default_branch == "main"

    @pytest.mark.asyncio
    async def test_get_pull_request(self, client):
        """Client can fetch pull request."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "id": 100,
            "number": 42,
            "title": "Test PR",
            "body": "Test description",
            "state": "open",
            "head": {"ref": "feature-branch", "sha": "abc123"},
            "base": {"ref": "main", "sha": "def456"},
            "user": {"id": 1, "login": "test-user"},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_resp

            pr = await client.get_pull_request("test-owner/test-repo", 42)

            assert isinstance(pr, PullRequest)
            assert pr.number == 42
            assert pr.title == "Test PR"
            assert pr.head_ref == "feature-branch"
            assert pr.base_ref == "main"

    @pytest.mark.asyncio
    async def test_create_check_run(self, client):
        """Client can create check run."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "id": 99999,
            "name": "CodeVerify",
            "status": "in_progress",
            "head_sha": "abc123",
        }

        check_run = CheckRun(
            name="CodeVerify",
            status=CheckStatus.IN_PROGRESS,
        )

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_resp

            result = await client.create_check_run(
                repo_full_name="test-owner/test-repo",
                head_sha="abc123",
                check_run=check_run,
            )

            assert isinstance(result, CheckRun)
            assert result.name == "CodeVerify"
            assert result.status == CheckStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_update_check_run(self, client):
        """Client can update check run."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "id": 99999,
            "name": "CodeVerify",
            "status": "completed",
            "conclusion": "success",
        }

        check_run = CheckRun(
            name="CodeVerify",
            status=CheckStatus.COMPLETED,
            conclusion=CheckConclusion.SUCCESS,
        )

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_resp

            result = await client.update_check_run(
                repo_full_name="test-owner/test-repo",
                check_run_id=99999,
                check_run=check_run,
            )

            assert result.status == CheckStatus.COMPLETED
            assert result.conclusion == CheckConclusion.SUCCESS

    @pytest.mark.asyncio
    async def test_create_comment(self, client):
        """Client can create PR comment."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "id": 12345,
            "body": "Test comment",
            "user": {"id": 1, "login": "bot"},
            "created_at": "2024-01-01T00:00:00Z",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_resp

            comment = await client.create_pull_request_comment(
                repo_full_name="test-owner/test-repo",
                pr_number=42,
                body="Test comment",
            )

            assert isinstance(comment, PullRequestComment)
            assert comment.body == "Test comment"

    def test_webhook_signature_verification(self, client):
        """Client verifies webhook signatures."""
        payload = b'{"action": "opened"}'

        # Generate valid signature
        import hashlib
        import hmac

        expected_sig = (
            "sha256="
            + hmac.new(b"webhook-secret", payload, hashlib.sha256).hexdigest()
        )

        assert client.verify_webhook_signature(payload, expected_sig)
        assert not client.verify_webhook_signature(payload, "sha256=invalid")


class TestGitLabClient:
    """Tests for GitLab client."""

    @pytest.fixture
    def client(self):
        """Create a GitLab client."""
        config = _make_config("gitlab", webhook_secret="webhook-token")
        return GitLabClient(config)

    def test_initialization(self, client):
        """Client initializes correctly."""
        assert client.provider_name == "gitlab"
        assert client.config.provider == "gitlab"

    @pytest.mark.asyncio
    async def test_get_pull_request(self, client):
        """Client fetches merge request (GitLab's PR equivalent)."""
        mock_response = {
            "id": 100,
            "iid": 42,
            "title": "Test MR",
            "description": "Test description",
            "state": "opened",
            "source_branch": "feature",
            "target_branch": "main",
            "sha": "abc123",
            "diff_refs": {"base_sha": "def456"},
            "author": {"id": 1, "username": "test-user"},
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-02T00:00:00Z",
        }

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = MagicMock()
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status = MagicMock()
            mock_http.get = AsyncMock(return_value=mock_resp)
            mock_get_client.return_value = mock_http

            pr = await client.get_pull_request("test-group/test-project", 42)

            assert isinstance(pr, PullRequest)
            assert pr.number == 42
            assert pr.title == "Test MR"

    def test_webhook_token_verification(self, client):
        """GitLab uses token-based webhook verification."""
        # GitLab compares the secret token directly
        assert client.verify_webhook_signature(b"payload", "webhook-token")
        assert not client.verify_webhook_signature(b"payload", "wrong")


class TestBitbucketClient:
    """Tests for Bitbucket client."""

    @pytest.fixture
    def client(self):
        """Create a Bitbucket client."""
        config = _make_config("bitbucket")
        return BitbucketClient(config)

    def test_initialization(self, client):
        """Client initializes correctly."""
        assert client.provider_name == "bitbucket"
        assert client.config.provider == "bitbucket"


class TestVCSClientAbstraction:
    """Tests for VCS client abstraction layer."""

    @pytest.mark.parametrize(
        "ClientClass,provider",
        [
            (GitHubClient, "github"),
            (GitLabClient, "gitlab"),
            (BitbucketClient, "bitbucket"),
        ],
    )
    def test_all_clients_implement_interface(self, ClientClass, provider):
        """All clients implement the VCSClient interface."""
        config = _make_config(provider)
        client = ClientClass(config)

        # Check required methods exist
        assert hasattr(client, "get_repository")
        assert hasattr(client, "get_pull_request")
        assert hasattr(client, "create_check_run")
        assert hasattr(client, "update_check_run")
        assert hasattr(client, "create_pull_request_comment")
        assert hasattr(client, "verify_webhook_signature")

        # Check provider
        assert client.provider_name == provider

    @pytest.mark.parametrize(
        "ClientClass,provider",
        [
            (GitHubClient, "github"),
            (GitLabClient, "gitlab"),
            (BitbucketClient, "bitbucket"),
        ],
    )
    def test_clients_are_vcs_clients(self, ClientClass, provider):
        """Clients are instances of VCSClient."""
        config = _make_config(provider)
        client = ClientClass(config)
        assert isinstance(client, VCSClient)


class TestPullRequestDataModel:
    """Tests for PullRequest data model."""

    def test_pull_request_creation(self):
        """PullRequest can be created with all fields."""
        now = datetime.now()
        author = User(id=1, username="test-user")
        pr = PullRequest(
            id=100,
            number=42,
            title="Test PR",
            body="Description",
            state="open",
            head_ref="feature",
            base_ref="main",
            head_sha="abc123",
            base_sha="def456",
            author=author,
            created_at=now,
            updated_at=now,
        )

        assert pr.number == 42
        assert pr.title == "Test PR"
        assert pr.head_ref == "feature"
        assert pr.base_ref == "main"

    def test_pull_request_optional_fields(self):
        """PullRequest handles optional fields."""
        now = datetime.now()
        author = User(id=1, username="user")
        pr = PullRequest(
            id=1,
            number=1,
            title="Minimal PR",
            body=None,
            state="open",
            head_ref="branch",
            base_ref="main",
            head_sha="sha",
            base_sha="sha2",
            author=author,
            created_at=now,
            updated_at=now,
        )

        assert pr.body is None
        assert pr.merged_at is None


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
