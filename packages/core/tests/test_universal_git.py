"""Tests for Universal Git Support (Feature 10)."""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from codeverify_core.universal_git import (
    GitProvider,
    GitCredentials,
    Repository,
    PullRequest,
    WebhookPayload,
    WebhookEventType,
    GitHubAdapter,
    GitLabAdapter,
    GiteaAdapter,
    GerritAdapter,
    AzureDevOpsAdapter,
    GenericGitAdapter,
    LocalGitOperations,
    WebhookReceiver,
    AirGappedConfig,
    AirGappedVerifier,
    UniversalGitSupport,
    cli_verify,
)


# Test fixtures
@pytest.fixture
def github_webhook_payload():
    """Sample GitHub webhook payload for pull_request event."""
    return {
        'action': 'opened',
        'number': 42,
        'pull_request': {
            'id': 123456,
            'number': 42,
            'title': 'Add new feature',
            'body': 'This PR adds a cool feature',
            'state': 'open',
            'head': {'ref': 'feature-branch'},
            'base': {'ref': 'main'},
            'user': {'login': 'developer'},
            'html_url': 'https://github.com/owner/repo/pull/42',
            'diff_url': 'https://github.com/owner/repo/pull/42.diff',
            'created_at': '2024-01-15T10:00:00Z',
            'updated_at': '2024-01-15T10:00:00Z',
        },
        'repository': {
            'name': 'test-repo',
            'owner': {'login': 'test-owner'},
            'clone_url': 'https://github.com/test-owner/test-repo.git',
            'default_branch': 'main',
            'private': False
        },
        'sender': {'login': 'developer'}
    }


@pytest.fixture
def github_push_payload():
    """Sample GitHub webhook payload for push event."""
    return {
        'ref': 'refs/heads/main',
        'before': 'abc123',
        'after': 'def456',
        'commits': [
            {'id': 'def456', 'message': 'Fix bug'},
            {'id': 'xyz789', 'message': 'Add test'}
        ],
        'repository': {
            'name': 'test-repo',
            'owner': {'login': 'test-owner'},
            'clone_url': 'https://github.com/test-owner/test-repo.git',
            'default_branch': 'main',
            'private': False
        },
        'sender': {'login': 'developer'}
    }


@pytest.fixture
def gitlab_webhook_payload():
    """Sample GitLab webhook payload for merge_request event."""
    return {
        'object_kind': 'merge_request',
        'project': {
            'name': 'test-repo',
            'namespace': 'test-owner',
            'git_http_url': 'https://gitlab.com/test-owner/test-repo.git',
            'default_branch': 'main',
            'path_with_namespace': 'test-owner/test-repo'
        },
        'object_attributes': {
            'id': 789,
            'iid': 15,
            'title': 'Implement feature',
            'description': 'Feature description',
            'source_branch': 'feature',
            'target_branch': 'main',
            'state': 'opened',
            'author_id': 123,
            'url': 'https://gitlab.com/test-owner/test-repo/-/merge_requests/15'
        },
        'user': {'username': 'developer'}
    }


@pytest.fixture
def gitea_webhook_payload():
    """Sample Gitea webhook payload."""
    return {
        'action': 'opened',
        'number': 5,
        'pull_request': {
            'id': 555,
            'number': 5,
            'title': 'Gitea PR',
            'body': 'Description',
            'head': {'ref': 'feature'},
            'base': {'ref': 'main'},
            'user': {'login': 'dev'},
            'state': 'open',
            'html_url': 'https://gitea.example.com/owner/repo/pulls/5'
        },
        'repository': {
            'name': 'repo',
            'owner': {'login': 'owner'},
            'clone_url': 'https://gitea.example.com/owner/repo.git',
            'default_branch': 'main',
            'private': False
        },
        'sender': {'login': 'dev'}
    }


@pytest.fixture
def azure_devops_webhook_payload():
    """Sample Azure DevOps webhook payload."""
    return {
        'eventType': 'git.pullrequest.created',
        'resource': {
            'pullRequestId': 100,
            'title': 'Azure DevOps PR',
            'description': 'PR description',
            'sourceRefName': 'refs/heads/feature',
            'targetRefName': 'refs/heads/main',
            'status': 'active',
            'createdBy': {'uniqueName': 'user@example.com'},
            'repository': {
                'name': 'repo',
                'project': {'name': 'project'},
                'remoteUrl': 'https://dev.azure.com/org/project/_git/repo',
                'defaultBranch': 'refs/heads/main'
            },
            'url': 'https://dev.azure.com/org/project/_git/repo/pullrequest/100'
        }
    }


@pytest.fixture
def gerrit_webhook_payload():
    """Sample Gerrit webhook payload."""
    return {
        'type': 'patchset-created',
        'change': {
            'id': 'I123456',
            'number': 1001,
            'subject': 'Fix bug in parser',
            'commitMessage': 'Fix bug in parser\n\nDetailed description',
            'branch': 'main',
            'owner': {'username': 'developer'},
            'status': 'NEW',
            'url': 'https://gerrit.example.com/c/project/+/1001'
        },
        'project': 'myproject',
        'uploader': {'username': 'developer'}
    }


class TestGitHubAdapter:
    """Tests for GitHub adapter."""

    def test_parse_pull_request_webhook(self, github_webhook_payload):
        creds = GitCredentials(provider=GitProvider.GITHUB)
        adapter = GitHubAdapter(creds)
        
        headers = {'X-GitHub-Event': 'pull_request'}
        payload = adapter.parse_webhook(headers, github_webhook_payload)
        
        assert payload.provider == GitProvider.GITHUB
        assert payload.event_type == WebhookEventType.PULL_REQUEST
        assert payload.repository.name == 'test-repo'
        assert payload.repository.owner == 'test-owner'
        assert payload.pull_request is not None
        assert payload.pull_request.number == 42
        assert payload.pull_request.title == 'Add new feature'

    def test_parse_push_webhook(self, github_push_payload):
        creds = GitCredentials(provider=GitProvider.GITHUB)
        adapter = GitHubAdapter(creds)
        
        headers = {'X-GitHub-Event': 'push'}
        payload = adapter.parse_webhook(headers, github_push_payload)
        
        assert payload.event_type == WebhookEventType.PUSH
        assert payload.ref == 'refs/heads/main'
        assert payload.before_sha == 'abc123'
        assert payload.after_sha == 'def456'
        assert len(payload.commits) == 2

    def test_map_event_types(self):
        creds = GitCredentials(provider=GitProvider.GITHUB)
        adapter = GitHubAdapter(creds)
        
        assert adapter._map_event_type('push') == WebhookEventType.PUSH
        assert adapter._map_event_type('pull_request') == WebhookEventType.PULL_REQUEST
        assert adapter._map_event_type('create') == WebhookEventType.TAG
        assert adapter._map_event_type('unknown') == WebhookEventType.UNKNOWN


class TestGitLabAdapter:
    """Tests for GitLab adapter."""

    def test_parse_merge_request_webhook(self, gitlab_webhook_payload):
        creds = GitCredentials(provider=GitProvider.GITLAB)
        adapter = GitLabAdapter(creds)
        
        headers = {'X-Gitlab-Event': 'Merge Request Hook'}
        payload = adapter.parse_webhook(headers, gitlab_webhook_payload)
        
        assert payload.provider == GitProvider.GITLAB
        assert payload.event_type == WebhookEventType.MERGE_REQUEST
        assert payload.repository.name == 'test-repo'
        assert payload.pull_request is not None
        assert payload.pull_request.number == 15

    def test_parse_push_webhook(self):
        creds = GitCredentials(provider=GitProvider.GITLAB)
        adapter = GitLabAdapter(creds)
        
        headers = {'X-Gitlab-Event': 'Push Hook'}
        body = {
            'ref': 'refs/heads/main',
            'before': 'abc',
            'after': 'def',
            'commits': [{'id': 'def'}],
            'project': {'name': 'repo', 'namespace': 'owner', 'git_http_url': 'url'}
        }
        payload = adapter.parse_webhook(headers, body)
        
        assert payload.event_type == WebhookEventType.PUSH
        assert payload.ref == 'refs/heads/main'


class TestGiteaAdapter:
    """Tests for Gitea adapter."""

    def test_parse_pull_request_webhook(self, gitea_webhook_payload):
        creds = GitCredentials(provider=GitProvider.GITEA)
        adapter = GiteaAdapter(creds)
        
        headers = {'X-Gitea-Event': 'pull_request'}
        payload = adapter.parse_webhook(headers, gitea_webhook_payload)
        
        assert payload.provider == GitProvider.GITEA
        assert payload.event_type == WebhookEventType.PULL_REQUEST
        assert payload.pull_request.number == 5


class TestAzureDevOpsAdapter:
    """Tests for Azure DevOps adapter."""

    def test_parse_pull_request_webhook(self, azure_devops_webhook_payload):
        creds = GitCredentials(provider=GitProvider.AZURE_DEVOPS)
        adapter = AzureDevOpsAdapter(creds)
        
        headers = {}
        payload = adapter.parse_webhook(headers, azure_devops_webhook_payload)
        
        assert payload.provider == GitProvider.AZURE_DEVOPS
        assert payload.event_type == WebhookEventType.PULL_REQUEST
        assert payload.pull_request.number == 100
        assert payload.pull_request.source_branch == 'feature'


class TestGerritAdapter:
    """Tests for Gerrit adapter."""

    def test_parse_patchset_webhook(self, gerrit_webhook_payload):
        creds = GitCredentials(provider=GitProvider.GERRIT, api_url='https://gerrit.example.com')
        adapter = GerritAdapter(creds)
        
        headers = {}
        payload = adapter.parse_webhook(headers, gerrit_webhook_payload)
        
        assert payload.provider == GitProvider.GERRIT
        assert payload.event_type == WebhookEventType.PULL_REQUEST
        assert payload.pull_request.number == 1001
        assert payload.pull_request.title == 'Fix bug in parser'


class TestGenericGitAdapter:
    """Tests for generic Git adapter."""

    def test_detect_github_provider(self, github_webhook_payload):
        creds = GitCredentials(provider=GitProvider.GENERIC)
        adapter = GenericGitAdapter(creds)
        
        headers = {'X-GitHub-Event': 'pull_request'}
        provider = adapter._detect_provider(headers, github_webhook_payload)
        
        assert provider == GitProvider.GITHUB

    def test_detect_gitlab_provider(self, gitlab_webhook_payload):
        creds = GitCredentials(provider=GitProvider.GENERIC)
        adapter = GenericGitAdapter(creds)
        
        headers = {'X-Gitlab-Event': 'Merge Request Hook'}
        provider = adapter._detect_provider(headers, gitlab_webhook_payload)
        
        assert provider == GitProvider.GITLAB

    def test_detect_unknown_provider(self):
        creds = GitCredentials(provider=GitProvider.GENERIC)
        adapter = GenericGitAdapter(creds)
        
        headers = {}
        provider = adapter._detect_provider(headers, {})
        
        assert provider == GitProvider.GENERIC

    def test_parse_generic_webhook(self, github_webhook_payload):
        creds = GitCredentials(provider=GitProvider.GENERIC)
        adapter = GenericGitAdapter(creds)
        
        headers = {}
        payload = adapter.parse_webhook(headers, github_webhook_payload)
        
        assert payload is not None
        assert payload.repository.name == 'test-repo'


class TestLocalGitOperations:
    """Tests for local Git operations."""

    def test_init_with_path(self, tmp_path):
        ops = LocalGitOperations(repo_path=tmp_path)
        assert ops.repo_path == tmp_path

    def test_init_default_path(self):
        ops = LocalGitOperations()
        assert ops.repo_path == Path.cwd()

    @patch('subprocess.run')
    def test_get_current_branch(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(stdout='main\n', returncode=0)
        
        ops = LocalGitOperations(repo_path=tmp_path)
        branch = ops.get_current_branch()
        
        assert branch == 'main'
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_get_remote_url(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(stdout='https://github.com/owner/repo.git\n')
        
        ops = LocalGitOperations(repo_path=tmp_path)
        url = ops.get_remote_url()
        
        assert url == 'https://github.com/owner/repo.git'

    @patch('subprocess.run')
    def test_get_diff(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(stdout='diff --git a/file.py b/file.py\n')
        
        ops = LocalGitOperations(repo_path=tmp_path)
        diff = ops.get_diff('HEAD~1', 'HEAD')
        
        assert 'diff --git' in diff

    @patch('subprocess.run')
    def test_get_changed_files(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(stdout='file1.py\nfile2.py\n')
        
        ops = LocalGitOperations(repo_path=tmp_path)
        files = ops.get_changed_files()
        
        assert len(files) == 2
        assert 'file1.py' in files

    @patch('subprocess.run')
    def test_detect_github_provider(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(stdout='https://github.com/owner/repo.git\n')
        
        ops = LocalGitOperations(repo_path=tmp_path)
        provider = ops.detect_provider()
        
        assert provider == GitProvider.GITHUB

    @patch('subprocess.run')
    def test_detect_gitlab_provider(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(stdout='https://gitlab.com/owner/repo.git\n')
        
        ops = LocalGitOperations(repo_path=tmp_path)
        provider = ops.detect_provider()
        
        assert provider == GitProvider.GITLAB

    @patch('subprocess.run')
    def test_get_commit_info(self, mock_run, tmp_path):
        mock_run.return_value = MagicMock(
            stdout='abc123\nJohn Doe\njohn@example.com\nFix bug\nDetailed description'
        )
        
        ops = LocalGitOperations(repo_path=tmp_path)
        info = ops.get_commit_info()
        
        assert info['sha'] == 'abc123'
        assert info['author_name'] == 'John Doe'
        assert info['subject'] == 'Fix bug'


class TestWebhookReceiver:
    """Tests for webhook receiver."""

    def test_register_adapter(self):
        receiver = WebhookReceiver()
        creds = GitCredentials(provider=GitProvider.GITHUB)
        adapter = GitHubAdapter(creds)
        
        receiver.register_adapter(GitProvider.GITHUB, adapter)
        
        assert GitProvider.GITHUB in receiver.adapters

    def test_process_github_webhook(self, github_webhook_payload):
        receiver = WebhookReceiver()
        creds = GitCredentials(provider=GitProvider.GITHUB)
        receiver.register_adapter(GitProvider.GITHUB, GitHubAdapter(creds))
        
        headers = {'X-GitHub-Event': 'pull_request'}
        body = json.dumps(github_webhook_payload).encode()
        
        payload = receiver.process_webhook(headers, body)
        
        assert payload is not None
        assert payload.provider == GitProvider.GITHUB
        assert payload.event_type == WebhookEventType.PULL_REQUEST

    def test_verify_github_signature(self, github_webhook_payload):
        secret = 'test-secret'
        receiver = WebhookReceiver(secrets={GitProvider.GITHUB: secret})
        
        body = json.dumps(github_webhook_payload).encode()
        import hashlib
        import hmac
        signature = 'sha256=' + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        headers = {'X-Hub-Signature-256': signature}
        
        assert receiver.verify_signature(GitProvider.GITHUB, headers, body) == True

    def test_invalid_github_signature(self, github_webhook_payload):
        secret = 'test-secret'
        receiver = WebhookReceiver(secrets={GitProvider.GITHUB: secret})
        
        body = json.dumps(github_webhook_payload).encode()
        headers = {'X-Hub-Signature-256': 'sha256=invalid'}
        
        assert receiver.verify_signature(GitProvider.GITHUB, headers, body) == False

    def test_verify_gitlab_token(self):
        secret = 'gitlab-token'
        receiver = WebhookReceiver(secrets={GitProvider.GITLAB: secret})
        
        headers = {'X-Gitlab-Token': secret}
        
        assert receiver.verify_signature(GitProvider.GITLAB, headers, b'{}') == True

    def test_process_invalid_json(self):
        receiver = WebhookReceiver()
        
        headers = {'X-GitHub-Event': 'push'}
        body = b'not valid json'
        
        payload = receiver.process_webhook(headers, body)
        
        assert payload is None


class TestAirGappedVerifier:
    """Tests for air-gapped verifier."""

    def test_initialize_missing_files(self, tmp_path):
        config = AirGappedConfig(
            model_path=tmp_path / 'model',
            rules_path=tmp_path / 'rules',
            cache_path=tmp_path / 'cache'
        )
        verifier = AirGappedVerifier(config)
        
        result = verifier.initialize()
        
        assert result == False
        assert verifier.model_loaded == False

    def test_initialize_with_files(self, tmp_path):
        model_path = tmp_path / 'model'
        rules_path = tmp_path / 'rules'
        model_path.mkdir()
        rules_path.mkdir()
        
        config = AirGappedConfig(
            model_path=model_path,
            rules_path=rules_path,
            cache_path=tmp_path / 'cache'
        )
        verifier = AirGappedVerifier(config)
        
        result = verifier.initialize()
        
        assert result == True
        assert verifier.model_loaded == True

    def test_verify_diff_not_initialized(self, tmp_path):
        config = AirGappedConfig(
            model_path=tmp_path / 'model',
            rules_path=tmp_path / 'rules',
            cache_path=tmp_path / 'cache'
        )
        verifier = AirGappedVerifier(config)
        
        result = verifier.verify_diff('diff content')
        
        assert result['verified'] == False
        assert 'error' in result

    def test_verify_diff_initialized(self, tmp_path):
        model_path = tmp_path / 'model'
        rules_path = tmp_path / 'rules'
        model_path.mkdir()
        rules_path.mkdir()
        
        config = AirGappedConfig(
            model_path=model_path,
            rules_path=rules_path,
            cache_path=tmp_path / 'cache'
        )
        verifier = AirGappedVerifier(config)
        verifier.initialize()
        
        result = verifier.verify_diff('diff content')
        
        assert result['verified'] == True
        assert result['offline'] == True


class TestUniversalGitSupport:
    """Tests for main universal Git support class."""

    def test_init_empty_credentials(self):
        support = UniversalGitSupport()
        
        assert support.credentials == {}
        assert support.adapters == {}

    def test_init_with_credentials(self):
        creds = {
            GitProvider.GITHUB: GitCredentials(
                provider=GitProvider.GITHUB,
                token='test-token'
            )
        }
        support = UniversalGitSupport(credentials=creds)
        
        assert GitProvider.GITHUB in support.adapters
        assert isinstance(support.adapters[GitProvider.GITHUB], GitHubAdapter)

    def test_get_adapter_creates_if_missing(self):
        support = UniversalGitSupport()
        
        adapter = support.get_adapter(GitProvider.GITHUB)
        
        assert adapter is not None
        assert isinstance(adapter, GitHubAdapter)

    def test_process_webhook(self, github_webhook_payload):
        support = UniversalGitSupport()
        
        headers = {'X-GitHub-Event': 'pull_request'}
        body = json.dumps(github_webhook_payload).encode()
        
        payload = support.process_webhook(headers, body)
        
        assert payload is not None
        assert payload.provider == GitProvider.GITHUB

    @patch('subprocess.run')
    def test_verify_local_changes(self, mock_run):
        mock_run.side_effect = [
            MagicMock(stdout='diff content\n'),  # get_diff
            MagicMock(stdout='file1.py\nfile2.py\n'),  # get_changed_files
            MagicMock(stdout='abc123\nAuthor\nemail\nSubject\n'),  # get_commit_info
            MagicMock(stdout='https://github.com/owner/repo.git\n')  # detect_provider
        ]
        
        support = UniversalGitSupport()
        result = support.verify_local_changes()
        
        assert 'diff' in result
        assert 'changed_files' in result
        assert 'commit' in result
        assert 'provider' in result

    @patch('subprocess.run')
    def test_verify_staged_changes(self, mock_run):
        mock_run.side_effect = [
            MagicMock(stdout='staged diff\n'),  # get_staged_diff
            MagicMock(stdout='feature-branch\n'),  # get_current_branch
            MagicMock(stdout='https://github.com/owner/repo.git\n')  # detect_provider
        ]
        
        support = UniversalGitSupport()
        result = support.verify_staged_changes()
        
        assert 'diff' in result
        assert 'branch' in result

    def test_create_repository_from_github_url(self):
        support = UniversalGitSupport()
        
        repo = support.create_repository_from_url('https://github.com/owner/repo.git')
        
        assert repo.provider == GitProvider.GITHUB
        assert repo.owner == 'owner'
        assert repo.name == 'repo'

    def test_create_repository_from_gitlab_url(self):
        support = UniversalGitSupport()
        
        repo = support.create_repository_from_url('https://gitlab.com/group/project.git')
        
        assert repo.provider == GitProvider.GITLAB
        assert repo.owner == 'group'
        assert repo.name == 'project'

    def test_create_repository_from_azure_url(self):
        support = UniversalGitSupport()
        
        repo = support.create_repository_from_url('https://dev.azure.com/org/project/_git/repo')
        
        assert repo.provider == GitProvider.AZURE_DEVOPS


class TestDataClasses:
    """Tests for data classes."""

    def test_git_credentials(self):
        creds = GitCredentials(
            provider=GitProvider.GITHUB,
            token='token123',
            api_url='https://api.github.com'
        )
        
        assert creds.provider == GitProvider.GITHUB
        assert creds.token == 'token123'

    def test_repository(self):
        repo = Repository(
            provider=GitProvider.GITHUB,
            owner='owner',
            name='repo',
            clone_url='https://github.com/owner/repo.git',
            default_branch='main',
            is_private=True
        )
        
        assert repo.provider == GitProvider.GITHUB
        assert repo.owner == 'owner'
        assert repo.is_private == True

    def test_pull_request(self):
        pr = PullRequest(
            id=1,
            number=42,
            title='Test PR',
            description='Description',
            source_branch='feature',
            target_branch='main',
            author='developer',
            state='open',
            created_at=datetime.now(),
            updated_at=datetime.now(),
            url='https://github.com/owner/repo/pull/42'
        )
        
        assert pr.number == 42
        assert pr.state == 'open'

    def test_webhook_payload(self):
        repo = Repository(
            provider=GitProvider.GITHUB,
            owner='owner',
            name='repo',
            clone_url='url'
        )
        payload = WebhookPayload(
            provider=GitProvider.GITHUB,
            event_type=WebhookEventType.PUSH,
            repository=repo,
            ref='refs/heads/main'
        )
        
        assert payload.event_type == WebhookEventType.PUSH
        assert payload.ref == 'refs/heads/main'


class TestEnums:
    """Tests for enum types."""

    def test_git_provider_values(self):
        assert GitProvider.GITHUB.value == 'github'
        assert GitProvider.GITLAB.value == 'gitlab'
        assert GitProvider.BITBUCKET.value == 'bitbucket'
        assert GitProvider.GITEA.value == 'gitea'
        assert GitProvider.GERRIT.value == 'gerrit'
        assert GitProvider.AZURE_DEVOPS.value == 'azure_devops'

    def test_webhook_event_type_values(self):
        assert WebhookEventType.PUSH.value == 'push'
        assert WebhookEventType.PULL_REQUEST.value == 'pull_request'
        assert WebhookEventType.MERGE_REQUEST.value == 'merge_request'


class TestCLI:
    """Tests for CLI interface."""

    @patch('codeverify_core.universal_git.UniversalGitSupport')
    def test_cli_verify_staged(self, mock_support_class):
        mock_instance = MagicMock()
        mock_instance.verify_staged_changes.return_value = {
            'diff': 'test diff',
            'branch': 'main',
            'provider': 'github'
        }
        mock_support_class.return_value = mock_instance
        
        result = cli_verify(['--staged'])
        
        assert result == 0
        mock_instance.verify_staged_changes.assert_called_once()

    @patch('codeverify_core.universal_git.UniversalGitSupport')
    def test_cli_verify_commits(self, mock_support_class):
        mock_instance = MagicMock()
        mock_instance.verify_local_changes.return_value = {
            'diff': 'test diff',
            'changed_files': ['file1.py'],
            'commit': {'sha': 'abc'},
            'base': 'HEAD~1',
            'target': 'HEAD',
            'provider': 'github'
        }
        mock_support_class.return_value = mock_instance
        
        result = cli_verify(['--base', 'HEAD~5', '--target', 'HEAD'])
        
        assert result == 0
        mock_instance.verify_local_changes.assert_called_once_with('HEAD~5', 'HEAD')


class TestIntegration:
    """Integration tests."""

    def test_full_webhook_processing_workflow(self, github_webhook_payload):
        # Setup
        support = UniversalGitSupport()
        
        # Process webhook
        headers = {'X-GitHub-Event': 'pull_request'}
        body = json.dumps(github_webhook_payload).encode()
        
        payload = support.process_webhook(headers, body)
        
        # Verify result
        assert payload is not None
        assert payload.provider == GitProvider.GITHUB
        assert payload.event_type == WebhookEventType.PULL_REQUEST
        assert payload.repository.name == 'test-repo'
        assert payload.pull_request.number == 42
        assert payload.pull_request.title == 'Add new feature'
        assert payload.sender == 'developer'

    def test_multi_provider_support(self, github_webhook_payload, gitlab_webhook_payload):
        support = UniversalGitSupport()
        
        # Process GitHub webhook
        gh_headers = {'X-GitHub-Event': 'pull_request'}
        gh_payload = support.process_webhook(gh_headers, json.dumps(github_webhook_payload).encode())
        
        # Process GitLab webhook
        gl_headers = {'X-Gitlab-Event': 'Merge Request Hook'}
        gl_payload = support.process_webhook(gl_headers, json.dumps(gitlab_webhook_payload).encode())
        
        # Both should be processed correctly
        assert gh_payload.provider == GitProvider.GITHUB
        assert gl_payload.provider == GitProvider.GITLAB
