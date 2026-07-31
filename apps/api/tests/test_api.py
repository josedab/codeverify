"""Tests for API endpoints."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from codeverify_api.db.database import get_db
from codeverify_api.main import app
from codeverify_api.routers import health
from codeverify_api.services.analysis_service import AnalysisService


@pytest.fixture
def mock_db() -> AsyncMock:
    """Return a database session mock."""
    return AsyncMock()


@pytest.fixture
def client(mock_db: AsyncMock):
    """Create a test client without opening a database connection."""

    async def override_get_db():
        yield mock_db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_db, None)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, client: TestClient) -> None:
        """Test basic health check."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_readiness_check(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test readiness check."""
        monkeypatch.setattr(
            health,
            "check_database",
            AsyncMock(return_value={"status": "healthy", "latency_ms": 0}),
        )
        monkeypatch.setattr(
            health,
            "check_redis",
            AsyncMock(return_value={"status": "healthy", "latency_ms": 0}),
        )

        response = client.get("/health/ready")

        assert response.status_code == 200
        assert response.json() == {
            "status": "ready",
            "checks": {
                "database": {"status": "healthy", "latency_ms": 0},
                "redis": {"status": "healthy", "latency_ms": 0},
            },
        }

    def test_liveness_check(self, client: TestClient) -> None:
        """Test liveness check."""
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root(self, client: TestClient) -> None:
        """Test root endpoint returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "CodeVerify API"
        assert data["status"] == "running"


class TestWebhookEndpoints:
    """Test GitHub webhook endpoints."""

    def test_webhook_ping(self, client: TestClient) -> None:
        """Test webhook ping event."""
        response = client.post(
            "/webhooks/github",
            json={"zen": "test"},
            headers={
                "X-GitHub-Event": "ping",
                "X-GitHub-Delivery": "test-delivery-id",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "pong"

    def test_webhook_pr_opened(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test webhook for PR opened event."""
        queue_analysis = AsyncMock(return_value="job-123")
        monkeypatch.setattr(AnalysisService, "queue_analysis", queue_analysis)

        response = client.post(
            "/webhooks/github",
            json={
                "action": "opened",
                "pull_request": {
                    "number": 123,
                    "title": "Test PR",
                    "head": {"sha": "abc123def456"},
                    "base": {"sha": "def456abc123"},
                },
                "repository": {
                    "id": 12345,
                    "full_name": "owner/repo",
                },
                "installation": {"id": 67890},
            },
            headers={
                "X-GitHub-Event": "pull_request",
                "X-GitHub-Delivery": "test-delivery-id",
            },
        )
        assert response.status_code == 200
        assert response.json() == {
            "status": "queued",
            "provider": "github",
            "job_id": "job-123",
            "pr_number": 123,
            "delivery_id": "test-delivery-id",
        }
        queue_analysis.assert_awaited_once_with(
            repo_full_name="owner/repo",
            repo_id=12345,
            pr_number=123,
            pr_title="Test PR",
            head_sha="abc123def456",
            base_sha="def456abc123",
            installation_id=67890,
            vcs_provider="github",
        )

    def test_webhook_pr_ignored_action(self, client: TestClient) -> None:
        """Test webhook ignores non-tracked PR actions."""
        response = client.post(
            "/webhooks/github",
            json={
                "action": "closed",
                "pull_request": {"number": 123},
                "repository": {"full_name": "owner/repo"},
            },
            headers={
                "X-GitHub-Event": "pull_request",
                "X-GitHub-Delivery": "test-delivery-id",
            },
        )
        assert response.status_code == 200
        assert response.json() == {
            "status": "ignored",
            "reason": "action 'closed' not tracked",
        }


class TestAnalysesEndpoints:
    """Test analyses API endpoints."""

    def test_list_analyses(self, client: TestClient, mock_db: AsyncMock) -> None:
        """Test listing analyses."""
        analyses_result = MagicMock()
        analyses_result.scalars.return_value.all.return_value = []
        count_result = MagicMock()
        count_result.scalar.return_value = 0
        mock_db.execute.side_effect = [analyses_result, count_result]

        response = client.get("/api/v1/analyses")

        assert response.status_code == 200
        assert response.json() == {
            "analyses": [],
            "total": 0,
            "limit": 50,
            "offset": 0,
        }

    def test_get_analysis_not_found(self, client: TestClient, mock_db: AsyncMock) -> None:
        """Test getting non-existent analysis."""
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = result

        response = client.get("/api/v1/analyses/00000000-0000-0000-0000-000000000000")

        assert response.status_code == 404
        assert response.json() == {"detail": "Analysis not found"}
