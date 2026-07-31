"""Integration tests for the current FastAPI application routes."""

import hashlib
import hmac
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch):
    """Create a full-app client with an in-memory database dependency."""
    monkeypatch.setenv("CORS_ORIGINS", '["http://test"]')
    monkeypatch.setenv("ENVIRONMENT", "development")

    from codeverify_api.db.database import get_db
    from codeverify_api.main import app

    empty_result = MagicMock()
    empty_result.scalars.return_value.all.return_value = []
    empty_result.scalar.return_value = 0

    db = AsyncMock()
    db.execute.return_value = empty_result
    db.get.return_value = None

    async def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app, base_url="http://test") as client:
            yield client, db
    finally:
        app.dependency_overrides.pop(get_db, None)


class TestHealthEndpoints:
    """Test health and service discovery endpoints."""

    def test_health_check(self, api_client):
        """Health endpoint returns the current service status."""
        client, _ = api_client

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_root_endpoint(self, api_client):
        """Root endpoint identifies the running API service."""
        client, _ = api_client

        response = client.get("/")

        assert response.status_code == 200
        assert response.json() == {
            "service": "CodeVerify API",
            "version": "0.1.0",
            "status": "running",
        }


class TestAuthEndpoints:
    """Test authentication endpoints without external OAuth or Redis."""

    def test_login_redirect(self, api_client, monkeypatch: pytest.MonkeyPatch):
        """Login stores OAuth state and redirects to the provider."""
        client, _ = api_client
        from codeverify_api.routers import auth

        authorize_url = "https://github.com/login/oauth/authorize?client_id=test"
        monkeypatch.setattr(auth.settings, "CORS_ORIGINS", ["http://test"])

        with (
            patch(
                "codeverify_api.routers.auth._store_oauth_state",
                new_callable=AsyncMock,
            ) as store_state,
            patch(
                "codeverify_api.routers.auth.GitHubOAuth.get_authorize_url",
                return_value=authorize_url,
            ) as get_authorize_url,
        ):
            response = client.get(
                "/api/v1/auth/login",
                params={"redirect_uri": "http://test/callback"},
                follow_redirects=False,
            )

        assert response.status_code == 307
        assert response.headers["location"] == authorize_url
        stored_state, stored_redirect = store_state.await_args.args
        assert stored_state
        assert stored_redirect == "http://test/callback"
        assert get_authorize_url.call_args.kwargs["state"] == stored_state

    def test_me_unauthorized(self, api_client):
        """Current-user endpoint rejects requests without a bearer token."""
        client, _ = api_client

        response = client.get("/api/v1/auth/me")

        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"


class TestAnalysesEndpoints:
    """Test analyses routes with the database dependency overridden."""

    def test_list_analyses_empty(self, api_client):
        """List endpoint serializes an empty database result."""
        client, db = api_client

        response = client.get("/api/v1/analyses", params={"limit": 10, "offset": 5})

        assert response.status_code == 200
        assert response.json() == {
            "analyses": [],
            "total": 0,
            "limit": 10,
            "offset": 5,
        }
        assert db.execute.await_count == 2

    def test_get_analysis_not_found(self, api_client):
        """A valid but unknown analysis ID returns a precise 404."""
        client, _ = api_client

        with patch(
            "codeverify_api.routers.analyses.AnalysisRepository.get_with_findings",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = client.get(f"/api/v1/analyses/{uuid4()}")

        assert response.status_code == 404
        assert response.json()["detail"] == "Analysis not found"


class TestWebhooksEndpoints:
    """Test signed GitHub webhook handling without external services."""

    def test_webhook_missing_signature(self, api_client, monkeypatch: pytest.MonkeyPatch):
        """Production-mode GitHub webhooks require a valid signature."""
        client, _ = api_client
        from codeverify_api.routers import webhooks

        monkeypatch.setattr(webhooks.settings, "ENVIRONMENT", "production")
        monkeypatch.setattr(webhooks.settings, "GITHUB_WEBHOOK_SECRET", "test-secret")

        response = client.post(
            "/webhooks/github",
            content=b'{"action":"opened"}',
            headers={
                "Content-Type": "application/json",
                "X-GitHub-Event": "pull_request",
                "X-GitHub-Delivery": "delivery-1",
            },
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid signature"

    def test_webhook_ping_event(self, api_client, monkeypatch: pytest.MonkeyPatch):
        """A correctly signed ping is acknowledged with its delivery ID."""
        client, _ = api_client
        from codeverify_api.routers import webhooks

        secret = "test-secret"
        payload = b'{"zen":"test"}'
        signature = "sha256=" + hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        monkeypatch.setattr(webhooks.settings, "ENVIRONMENT", "production")
        monkeypatch.setattr(webhooks.settings, "GITHUB_WEBHOOK_SECRET", secret)

        response = client.post(
            "/webhooks/github",
            content=payload,
            headers={
                "Content-Type": "application/json",
                "X-GitHub-Event": "ping",
                "X-GitHub-Delivery": "delivery-ping",
                "X-Hub-Signature-256": signature,
            },
        )

        assert response.status_code == 200
        assert response.json() == {"status": "pong", "delivery_id": "delivery-ping"}


class TestStatsEndpoints:
    """Test authenticated statistics endpoints."""

    def test_dashboard_stats_unauthorized(self, api_client):
        """The registered dashboard statistics route requires authentication."""
        client, _ = api_client

        response = client.get("/api/v1/stats/stats/dashboard")

        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"


class TestRepositoriesEndpoints:
    """Test repository routes with deterministic database results."""

    def test_list_repositories_empty(self, api_client):
        """Repository listing returns its current pagination envelope."""
        client, _ = api_client

        response = client.get("/api/v1/repositories")

        assert response.status_code == 200
        assert response.json() == {
            "repositories": [],
            "total": 0,
            "limit": 50,
            "offset": 0,
        }


class TestOrganizationsEndpoints:
    """Test organization routes with deterministic database results."""

    def test_list_organizations_empty(self, api_client):
        """Organization listing returns its current pagination envelope."""
        client, _ = api_client

        response = client.get("/api/v1/organizations")

        assert response.status_code == 200
        assert response.json() == {
            "organizations": [],
            "total": 0,
            "limit": 50,
            "offset": 0,
        }
