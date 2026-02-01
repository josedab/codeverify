"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from codeverify_api.main import app


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, client: TestClient) -> None:
        """Test basic health check."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_readiness_check(self, client: TestClient) -> None:
        """Test readiness check."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

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

    def test_webhook_pr_opened(self, client: TestClient) -> None:
        """Test webhook for PR opened event."""
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
        data = response.json()
        assert data["status"] == "queued"
        assert data["pr_number"] == 123

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
        assert response.json()["status"] == "ignored"


class TestAnalysesEndpoints:
    """Test analyses API endpoints."""

    def test_list_analyses(self, client: TestClient) -> None:
        """Test listing analyses."""
        response = client.get("/api/v1/analyses")
        assert response.status_code == 200
        data = response.json()
        assert "analyses" in data
        assert "total" in data

    def test_get_analysis_not_found(self, client: TestClient) -> None:
        """Test getting non-existent analysis."""
        response = client.get("/api/v1/analyses/00000000-0000-0000-0000-000000000000")
        assert response.status_code == 404
